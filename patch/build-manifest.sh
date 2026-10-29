#!/usr/bin/env bash
# build-manifest.sh — assemble the tool-distribution JSON manifest consumed
# by the shared tool-runtime initializer (aither #106).
#
# Usage:
#   patch/build-manifest.sh <metadata-dir> <release-tag> <repo>
#
# <metadata-dir> holds the apply_patch-<version>-<target>.json records
# emitted by patch/build-artifact.sh; the release tag and repo
# ("owner/name") fix the asset download base URL. The manifest —
# patch/third-party-tools.json's pinned entries plus one apply_patch entry
# per metadata record, and its pinned host programs — is written to stdout.
#
# os/arch are derived from the target triple itself (arch is its first
# component), not from a lookup table.
set -euo pipefail

dir="${1:?usage: build-manifest.sh <metadata-dir> <release-tag> <repo>}"
tag="${2:?usage: build-manifest.sh <metadata-dir> <release-tag> <repo>}"
repo="${3:?usage: build-manifest.sh <metadata-dir> <release-tag> <repo>}"
root="$(cd "$(dirname "$0")/.." && pwd)"

shopt -s nullglob
metas=("$dir"/apply_patch-*.json)
[ "${#metas[@]}" -gt 0 ] || { echo "no apply_patch-*.json metadata in $dir" >&2; exit 1; }

apply_patch_entries="$(jq -cs --arg base "https://github.com/${repo}/releases/download/${tag}" '
    def os_of: if endswith("-apple-darwin") then "macos"
               elif test("linux") then "linux"
               elif test("windows") then "windows"
               else error("cannot map target to an os") end;
    sort_by(.target) | map(. as $m | ($m.target | os_of) as $os | {
        tool: "apply_patch",
        version: $m.version,
        target: {os: $os, arch: ($m.target | split("-")[0])},
        url: "\($base)/\($m.file)",
        sha256: $m.sha256,
        encoding: {
            format: "tar_gzip",
            executable: (if $os == "windows" then "apply_patch.exe" else "apply_patch" end)
        }
    })
' "${metas[@]}")"

jq -n --slurpfile third_party "$root/patch/third-party-tools.json" \
      --argjson apply_patch "$apply_patch_entries" \
      '{artifacts: ($third_party[0].artifacts + $apply_patch),
        host_programs: $third_party[0].host_programs}'
