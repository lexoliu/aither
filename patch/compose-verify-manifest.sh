#!/usr/bin/env bash
# compose-verify-manifest.sh — compose the tool-distribution manifest that
# sandbox/examples/verify_tools.rs verifies (aither #411).
#
# Usage:
#   patch/compose-verify-manifest.sh <repo> <output.json>
#
# patch/third-party-tools.json of this checkout holds only third-party tools,
# the host programs and the interpreter pins. The tools aither builds itself
# (patch/built-tools.json, plus the bash that build-bash.sh builds) exist only
# as assets of a release, so the manifest is this checkout's file plus those
# tools' entries from the tool-distribution asset of the latest `patch-v*`
# release of <repo> ("owner/name"). `gh` authenticates through GH_TOKEN.
set -euo pipefail

repo="${1:?usage: compose-verify-manifest.sh <repo> <output.json>}"
out="${2:?usage: compose-verify-manifest.sh <repo> <output.json>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

tag="$(gh release list -R "$repo" --limit 200 --json tagName \
    --jq '.[].tagName | select(startswith("patch-v"))' | sort -V | tail -n 1)"
[ -n "$tag" ] || { echo "no patch-v* release in $repo" >&2; exit 1; }
gh release download "$tag" -R "$repo" -p 'tool-distribution-*.json' -O "$work/release.json"
# Every asset's byte size as GitHub stores it, keyed by download URL.
gh release view "$tag" -R "$repo" --json assets --jq '.assets | map({(.url): .size}) | add' \
    > "$work/sizes.json"

# This checkout's artifacts, plus the release's entries of every tool this
# checkout does not pin itself. Each taken entry carries the size of the
# release asset it names; one whose pinned size disagrees with the asset
# stops the composition.
jq --slurpfile release "$work/release.json" --slurpfile sizes "$work/sizes.json" '
    (.artifacts | map(.tool)) as $pinned
    | .artifacts += ($release[0].artifacts
        | map(select(.tool as $tool | $pinned | index($tool) | not))
        | map(. as $a
            | ($sizes[0][$a.url] // error("\($a.url) is not an asset of the release"))
            | if ($a.size // .) == . then $a + {size: .}
              else error("\($a.url): pinned size \($a.size) but the asset is \(.) bytes") end))
' "$root/patch/third-party-tools.json" > "$out"
echo "composed $out from $tag" >&2
