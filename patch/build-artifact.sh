#!/usr/bin/env bash
# build-artifact.sh — build the size-optimized `apply_patch` executable for
# one target and emit a hash-verified distribution tarball.
#
# This is the artifact-build entrypoint consumed by the shared tool-runtime
# initializer (aither #106). It builds ONLY aither-patch under the dedicated
# `apply-patch` Cargo profile and a pinned nightly's rebuilt std; no other
# workspace crate's build settings change.
#
# Usage:
#   patch/build-artifact.sh <target-triple> [out-dir]
#
# Writes <out-dir>/apply_patch-<version>-<target>.tar.gz containing the
# executable, plus <name>.sha256 and <name>.json metadata. Prints a single
# JSON object describing the produced artifact on stdout.
#
# Environment:
#   APPLY_PATCH_NIGHTLY   pinned toolchain (default nightly-2026-10-01).
#                         Requires the rust-src component and, for musl
#                         targets, the target's rust-std component for the
#                         self-contained crt objects.
#   APPLY_PATCH_ABORT     "immediate" (default): panics abort without a
#                         message; ordinary patch errors keep full text.
#                         "panic": keep panic=abort diagnostics at ~3x size.
set -euo pipefail

target="${1:?usage: build-artifact.sh <target-triple> [out-dir]}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="${2:-$root/target/apply-patch-artifacts}"
nightly="${APPLY_PATCH_NIGHTLY:-nightly-2026-10-01}"
abort="${APPLY_PATCH_ABORT:-immediate}"
version="$(cargo metadata --locked --no-deps --format-version 1 --manifest-path "$root/Cargo.toml" | jq -r '.packages[] | select(.name == "aither-patch") | .version')"

case "$abort" in
    immediate) flags=(-Zunstable-options -Cpanic=immediate-abort) ;;
    panic)     flags=() ;;
    *)         echo "APPLY_PATCH_ABORT must be 'immediate' or 'panic'" >&2; exit 2 ;;
esac

export CARGO_PROFILE_APPLY_PATCH_PANIC=abort
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

cargo +"$nightly" build -p aither-patch --profile apply-patch \
    --target "$target" -Zbuild-std=std,panic_abort \
    --manifest-path "$root/Cargo.toml"

bin="$CARGO_TARGET_DIR/$target/apply-patch/apply_patch${suffix}"
[ -x "$bin" ] || { echo "artifact build produced no executable at $bin" >&2; exit 1; }
bin="$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"

name="apply_patch-${version}-${target}"
mkdir -p "$out_dir"
(cd "$out_dir" && tar -czf "$name.tar.gz" -C "$(dirname "$bin")" "$(basename "$bin")")
(cd "$out_dir" && "${checksum[@]}" "$name.tar.gz" > "$name.tar.gz.sha256")
bytes="$(wc -c < "$bin" | tr -d '[:space:]')"
sha="$(cut -d' ' -f1 "$out_dir/$name.tar.gz.sha256")"

jq -n --arg name apply_patch --arg version "$version" --arg target "$target" \
    --arg abort "$abort" --arg toolchain "$nightly" --arg file "$name.tar.gz" \
    --arg sha256 "$sha" --argjson executable_bytes "$bytes" \
    '{name: $name, version: $version, target: $target, abort: $abort, toolchain: $toolchain, file: $file, sha256: $sha256, executable_bytes: $executable_bytes}' \
    > "$out_dir/$name.json"
cat "$out_dir/$name.json"; echo
