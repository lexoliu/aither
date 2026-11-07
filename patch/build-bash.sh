#!/usr/bin/env bash
# build-bash.sh — build the pinned GNU bash for one target and emit a
# hash-verified distribution tarball, like build-artifact.sh does for
# `apply_patch`.
#
# Usage:
#   patch/build-bash.sh <target-triple> [out-dir]
#
# Targets: x86_64-unknown-linux-musl and aarch64-unknown-linux-musl (a static
# executable built in an Alpine container, which must match the host's
# architecture) and aarch64-apple-darwin (built natively, deployment target
# 11.0). The source and patch pins are patch/bash-sources.sha256; the
# compilation itself is patch/bash-compile.sh.
#
# Writes <out-dir>/bash-<version>-<target>.tar.gz containing the executable,
# plus <name>.sha256 and <name>.json metadata, and prints the metadata on
# stdout. patch/build-manifest.sh merges the metadata into the tool
# distribution manifest.
set -euo pipefail

target="${1:?usage: build-bash.sh <target-triple> [out-dir]}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="${2:-$root/target/bash-artifacts}"
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

# Pinned by digest so the toolchain is as fixed as the sources.
alpine_image="alpine:3.22@sha256:2e0b81d60597f449baa56dc25d355367869d2dca4cd57e866f1e2b52af3eb55f"

patches="$(grep -c ' bash53-' "$root/patch/bash-sources.sha256")"
version="5.3.${patches}"
work="$out_dir/work/$target"
rm -rf "$work"
mkdir -p "$work"

checksum=(shasum -a 256)
case "$target" in
    *-unknown-linux-musl)
        [ "${target%%-*}" = "$(uname -m)" ] \
            || { echo "$target must be built on a host of its own architecture, not $(uname -m)" >&2; exit 2; }
        docker run --rm -v "$root/patch:/patch:ro" -v "$work:/work" "$alpine_image" \
            sh /patch/bash-compile.sh "$target" /work
        checksum=(sha256sum)
        ;;
    aarch64-apple-darwin)
        sh "$root/patch/bash-compile.sh" "$target" "$work"
        ;;
    *) echo "unsupported bash target: $target" >&2; exit 2 ;;
esac

bin="$work/bash"
[ -x "$bin" ] || { echo "bash build produced no executable at $bin" >&2; exit 1; }

name="bash-${version}-${target}"
(cd "$out_dir" && tar -czf "$name.tar.gz" -C "$work" bash)
(cd "$out_dir" && "${checksum[@]}" "$name.tar.gz" > "$name.tar.gz.sha256")
bytes="$(wc -c < "$bin" | tr -d '[:space:]')"
sha="$(cut -d' ' -f1 "$out_dir/$name.tar.gz.sha256")"
size="$(wc -c < "$out_dir/$name.tar.gz" | tr -d '[:space:]')"

jq -n --arg name bash --arg version "$version" --arg target "$target" \
    --arg file "$name.tar.gz" --arg sha256 "$sha" --argjson size "$size" --argjson executable_bytes "$bytes" \
    '{name: $name, version: $version, target: $target, file: $file, sha256: $sha256, size: $size, executable_bytes: $executable_bytes}' \
    > "$out_dir/$name.json"
cat "$out_dir/$name.json"; echo
