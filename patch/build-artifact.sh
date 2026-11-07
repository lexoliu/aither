#!/usr/bin/env bash
# build-artifact.sh — build the size-optimized executables of every aither-built
# tool listed for one target and emit a hash-verified distribution tarball for
# each.
#
# This is the artifact-build entrypoint consumed by the shared tool-runtime
# initializer (aither #106). The tools are listed in patch/built-tools.json
# (tool name, cargo package, bin, targets); a tool is built for a target only
# if the list names it. All of a target's tools build in one cargo invocation
# under the dedicated `artifact` Cargo profile and a pinned nightly's rebuilt
# std; no other workspace crate's build settings change.
#
# Usage:
#   patch/build-artifact.sh <target-triple> [out-dir]
#
# For each tool writes <out-dir>/<tool>-<version>-<target>.tar.gz containing
# the executable, plus <name>.sha256 and <name>.json metadata. Prints one JSON
# object per produced artifact on stdout.
#
# Environment:
#   ARTIFACT_NIGHTLY   pinned toolchain (default nightly-2026-10-01).
#                      Requires the rust-src component and, for musl
#                      targets, the target's rust-std component for the
#                      self-contained crt objects.
#   ARTIFACT_ABORT     "immediate" (default): panics abort without a
#                      message; ordinary errors keep full text.
#                      "panic": keep panic=abort diagnostics at ~3x size.
set -euo pipefail

target="${1:?usage: build-artifact.sh <target-triple> [out-dir]}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="${2:-$root/target/artifacts}"
nightly="${ARTIFACT_NIGHTLY:-nightly-2026-10-01}"
abort="${ARTIFACT_ABORT:-immediate}"

# jq.exe on Windows ends its output lines with CRLF; every value the script
# reads back drops the CR.
query() { jq "$@" | tr -d '\r'; }

# The listed tools of this target, one compact JSON object per line.
tools="$(query -c --arg target "$target" '.tools[] | select(.targets | index($target))' "$root/patch/built-tools.json")"
[ -n "$tools" ] || { echo "no tool in patch/built-tools.json is built for $target" >&2; exit 2; }
packages=()
while IFS= read -r package; do packages+=(-p "$package"); done < <(query -r '.package' <<<"$tools")
metadata="$(cargo metadata --locked --no-deps --format-version 1 --manifest-path "$root/Cargo.toml")"

case "$abort" in
    immediate) flags=(-Zunstable-options -Cpanic=immediate-abort) ;;
    panic)     flags=() ;;
    *)         echo "ARTIFACT_ABORT must be 'immediate' or 'panic'" >&2; exit 2 ;;
esac

export CARGO_PROFILE_ARTIFACT_PANIC=abort
suffix=""
checksum=(shasum -a 256)
case "$target" in
    *-apple-darwin) linker_flags=() ;;
    *-unknown-linux-*) linker_flags=(-Clinker=rust-lld) ;;
    *-pc-windows-msvc) linker_flags=(); suffix=".exe"; checksum=(sha256sum) ;;
    *) echo "unsupported artifact target: $target" >&2; exit 2 ;;
esac
export RUSTFLAGS="${flags[*]-} ${linker_flags[*]-}"
export CARGO_TARGET_DIR="$out_dir/build"

cargo +"$nightly" build "${packages[@]}" --profile artifact \
    --target "$target" -Zbuild-std=std,panic_abort \
    --manifest-path "$root/Cargo.toml"

mkdir -p "$out_dir"
while IFS= read -r entry; do
    tool="$(query -r '.tool' <<<"$entry")"
    package="$(query -r '.package' <<<"$entry")"
    exe="$(query -r '.bin' <<<"$entry")${suffix}"
    version="$(query -r --arg package "$package" '.packages[] | select(.name == $package) | .version' <<<"$metadata")"

    bin="$CARGO_TARGET_DIR/$target/artifact/$exe"
    [ -x "$bin" ] || { echo "artifact build produced no executable at $bin" >&2; exit 1; }
    bin="$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"

    name="${tool}-${version}-${target}"
    (cd "$out_dir" && tar -czf "$name.tar.gz" -C "$(dirname "$bin")" "$(basename "$bin")")
    (cd "$out_dir" && "${checksum[@]}" "$name.tar.gz" > "$name.tar.gz.sha256")
    bytes="$(wc -c < "$bin" | tr -d '[:space:]')"
    sha="$(cut -d' ' -f1 "$out_dir/$name.tar.gz.sha256")"
    size="$(wc -c < "$out_dir/$name.tar.gz" | tr -d '[:space:]')"

    jq -n --arg name "$tool" --arg version "$version" --arg target "$target" \
        --arg abort "$abort" --arg toolchain "$nightly" --arg file "$name.tar.gz" \
        --arg sha256 "$sha" --argjson size "$size" --argjson executable_bytes "$bytes" \
        '{name: $name, version: $version, target: $target, abort: $abort, toolchain: $toolchain, file: $file, sha256: $sha256, size: $size, executable_bytes: $executable_bytes}' \
        > "$out_dir/$name.json"
    cat "$out_dir/$name.json"; echo
done <<<"$tools"
