#!/bin/sh
# bash-compile.sh — compile the pinned GNU bash for one target.
#
# POSIX sh so it runs unchanged on a macOS runner and inside the Alpine
# container the linux builds use; patch/build-bash.sh is the entrypoint that
# sets either up and packages the result.
#
# Usage: patch/bash-compile.sh <target-triple> <work-dir>
#
# Downloads the GNU source tarball and every official patch listed in
# patch/bash-sources.sha256 into <work-dir>/download, refusing any file whose
# SHA-256 differs from that list, applies the patches in order, builds, and
# leaves the stripped executable at <work-dir>/bash. The version the
# executable reports must be 5.3.<number of pinned patches>.
set -eu

target="${1:?usage: bash-compile.sh <target-triple> <work-dir>}"
work="${2:?usage: bash-compile.sh <target-triple> <work-dir>}"
here="$(cd "$(dirname "$0")" && pwd)"
pins="$here/bash-sources.sha256"
mirror=https://ftp.gnu.org/gnu/bash

mkdir -p "$work/download"
work="$(cd "$work" && pwd)"

checksum() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}

case "$target" in
    *-unknown-linux-musl)
        # Alpine is musl-native, so the build is static without a cross toolchain.
        apk add --no-cache build-base curl patch binutils >/dev/null
        link_flags=--enable-static-link
        ;;
    aarch64-apple-darwin)
        # Binaries run on every macOS the aither release supports.
        export MACOSX_DEPLOYMENT_TARGET=11.0
        link_flags=
        ;;
    *) echo "unsupported bash target: $target" >&2; exit 2 ;;
esac

while read -r _ file; do
    case "$file" in
        bash53-*) url="$mirror/bash-5.3-patches/$file" ;;
        *) url="$mirror/$file" ;;
    esac
    curl -fsSL --retry 3 -o "$work/download/$file" "$url"
done <"$pins"
(cd "$work/download" && checksum -c "$pins")

rm -rf "$work/source"
mkdir -p "$work/source"
tar -xzf "$work/download/bash-5.3.tar.gz" -C "$work/source"
cd "$work/source/bash-5.3"
patches=0
for patch_file in $(awk '/ bash53-/ { print $2 }' "$pins" | sort); do
    patch -p0 -s <"$work/download/$patch_file"
    patches=$((patches + 1))
done

# --without-curses: where the system has no curses library to link, as in
# the Alpine container (no ncurses development package is installed, so the
# static link has none), readline uses the termcap code bundled with bash;
# macOS links its own libncurses. Scripts run with -c never reach either.
# --without-bash-malloc: the system allocator, as musl and macOS require.
# --disable-nls: no gettext, so messages are the same on every host.
# shellcheck disable=SC2086
./configure --without-bash-malloc --without-curses --disable-nls $link_flags >/dev/null
make -j"$(getconf _NPROCESSORS_ONLN)" >/dev/null
strip bash
case "$target" in
    # Stripping invalidates the linker's ad-hoc signature, which arm64 macOS
    # refuses to run without.
    *-apple-darwin) codesign --force --sign - bash ;;
esac

expected="GNU bash, version 5.3.$patches(1)-release"
banner="$(./bash --version | head -n 1)"
case "$banner" in
    "$expected "*) ;;
    *) echo "built bash reports '$banner', expected '$expected'" >&2; exit 1 ;;
esac
case "$target" in
    *-unknown-linux-musl)
        # A static executable has no program interpreter.
        if readelf -l bash | grep -q INTERP; then
            echo "bash is dynamically linked" >&2; exit 1
        fi ;;
    *-apple-darwin)
        # Only the system's own libraries exist on every Mac.
        if otool -L bash | tail -n +2 | grep -v -e '^[[:space:]]*/usr/lib/' -e '^[[:space:]]*/System/'; then
            echo "bash links a library outside the system" >&2; exit 1
        fi ;;
esac
cp bash "$work/bash"
echo "built bash 5.3.$patches for $target"
