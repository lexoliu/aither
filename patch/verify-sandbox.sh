#!/usr/bin/env bash
set -euo pipefail
binary="${1:?usage: verify-sandbox.sh <apply_patch-executable>}"
binary="$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")"
heel="${HEEL_BIN:?HEEL_BIN must name the verified heel executable}"
root="$(mktemp -d)"
trap 'rm -rf -- "$root"' EXIT
mkdir "$root/allowed" "$root/denied"
printf 'original\n' > "$root/denied/sentinel.txt"
"$heel" run --network deny --keep-working-dir --working-dir "$root/allowed" --executable "${binary%/*}" --readable "${binary%/*}" --readable "$root/denied" -- "$binary" <<'PATCH'
*** Begin Patch
*** Add File: permitted.txt
+allowed
*** End Patch
PATCH
printf 'allowed\n' | cmp - "$root/allowed/permitted.txt"
if "$heel" run --network deny --keep-working-dir --working-dir "$root/allowed" --executable "${binary%/*}" --readable "${binary%/*}" --readable "$root/denied" -- "$binary" <<'PATCH'
*** Begin Patch
*** Update File: ../denied/sentinel.txt
@@
-original
+modified
*** End Patch
PATCH
then
  printf 'sandbox allowed a forbidden write\n' >&2
  exit 1
fi
printf 'original\n' | cmp - "$root/denied/sentinel.txt"
printf 'allowed edit succeeded; forbidden edit denied without changing the file\n'
