#!/usr/bin/env bash
# build-manifest.sh — assemble the tool-distribution JSON manifest consumed
# by the shared tool-runtime initializer (aither #106).
#
# Usage:
#   patch/build-manifest.sh <metadata-dir> <release-tag> <repo>
#
# <metadata-dir> holds the <tool>-<version>-<target>.json records emitted by
# patch/build-artifact.sh for every tool in patch/built-tools.json and the
# bash-<version>-<target>.json records emitted by patch/build-bash.sh; the
# release tag and repo ("owner/name") fix the asset download base URL. The
# manifest — patch/third-party-tools.json's pinned entries plus one entry per
# metadata record, and its pinned host programs and interpreter pins — is
# written to stdout.
#
# Each listed tool must have a record for exactly the targets the list gives
# it, no more and no fewer.
#
# os/arch are derived from the target triple itself (arch is its first
# component), not from a lookup table (patch/targets.jq).
set -euo pipefail

dir="${1:?usage: build-manifest.sh <metadata-dir> <release-tag> <repo>}"
tag="${2:?usage: build-manifest.sh <metadata-dir> <release-tag> <repo>}"
repo="${3:?usage: build-manifest.sh <metadata-dir> <release-tag> <repo>}"
root="$(cd "$(dirname "$0")/.." && pwd)"

shopt -s nullglob

# One manifest entry per metadata record of tool "$1", whose executable in
# the tarball is named "$1" (with ".exe" on Windows).
entries() {
    local tool="$1"; shift
    jq -cs -L "$root/patch" --arg tool "$tool" --arg base "https://github.com/${repo}/releases/download/${tag}" '
        include "targets";
        sort_by(.target) | map(. as $m | ($m.target | platform_of) as $platform | {
            tool: $tool,
            version: $m.version,
            target: $platform,
            url: "\($base)/\($m.file)",
            sha256: $m.sha256,
            size: $m.size,
            encoding: {
                format: "tar_gzip",
                executable: (if $platform.os == "windows" then "\($tool).exe" else $tool end)
            }
        })
    ' "$@"
}

# The entries of every tool, as one JSON array.
all_entries='[]'

# Collect the records of tool "$1" (files "<tool>-<digit>*.json"), requiring
# that their targets are exactly "$2", a JSON array of target triples, or any
# set when "$2" is empty.
collect() {
    local tool="$1" expected="$2" metas
    metas=("$dir/$tool"-[0-9]*.json)
    [ "${#metas[@]}" -gt 0 ] || { echo "no $tool-*.json metadata in $dir" >&2; exit 1; }
    if [ -n "$expected" ]; then
        jq -s --argjson expected "$expected" --arg tool "$tool" '
            if (map(.target) | sort) == ($expected | sort) then empty
            else error("\($tool) targets \(map(.target) | sort) differ from the listed \($expected | sort)") end
        ' "${metas[@]}" > /dev/null
    fi
    all_entries="$(jq -cn --argjson all "$all_entries" --argjson new "$(entries "$tool" "${metas[@]}")" '$all + $new')"
}

while IFS= read -r listed; do
    collect "$(jq -r '.tool' <<<"$listed")" "$(jq -c '.targets' <<<"$listed")"
done < <(jq -c '.tools[]' "$root/patch/built-tools.json")
collect bash ''

jq -n --slurpfile third_party "$root/patch/third-party-tools.json" \
      --argjson built "$all_entries" \
      '{artifacts: ($third_party[0].artifacts + $built),
        host_programs: $third_party[0].host_programs,
        python: $third_party[0].python}'
