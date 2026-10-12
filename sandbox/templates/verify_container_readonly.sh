#!/usr/bin/env bash
set -euo pipefail
bin_dir="${1:?usage: verify_container_readonly.sh <prepared-bin-directory>}"
test "$(id -u)" -eq 0
before="$(sha256sum "$bin_dir/rg")"
if printf 'tampered\n' > "$bin_dir/rg"; then exit 1; fi
if chmod u+w "$bin_dir/rg"; then exit 1; fi
if printf 'tampered\n' > "$bin_dir/rejected-write"; then exit 1; fi
after="$(sha256sum "$bin_dir/rg")"
test "$before" = "$after"
test ! -e "$bin_dir/rejected-write"
printf 'root writes and chmod denied; prepared executable unchanged\n'
