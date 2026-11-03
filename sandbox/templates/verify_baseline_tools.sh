set -eu
apply_patch <<'PATCH'
*** Begin Patch
*** Add File: src/main.rs
+fn old_call() {}
*** End Patch
PATCH
rg -n 'old_call' src/main.rs >text-search.txt &
text_pid=$!
ast-grep -p 'fn $NAME() { $$$BODY }' --lang rust src/main.rs >structural-search.txt &
structural_pid=$!
wait "$text_pid"
wait "$structural_pid"
[ -s text-search.txt ]
[ -s structural-search.txt ]
printf '{"value":1}\n' | jq '.value + 1' >json-result.txt
[ "$(cat json-result.txt)" = 2 ]
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: src/main.rs
@@
-fn old_call() {}
+fn new_call() {}
*** End Patch
PATCH
rg -q 'new_call' src/main.rs
# Managed uutils tools win over the host's BSD or GNU ones, so GNU options
# behave identically on macOS and Linux.
date --version | rg -q 'uutils coreutils'
xargs --version | rg -q '^xargs [0-9]'
fd --version | rg -q '^fd [0-9]'
[ "$(TZ=UTC date -d @0 +%F)" = 1970-01-01 ]
printf 'abcd' >size.txt
[ "$(stat -c %s size.txt)" = 4 ]
status=0
timeout 1 tail -f size.txt || status=$?
[ "$status" = 124 ]
[ "$(printf 'a\0b' | xargs -0 echo)" = 'a b' ]
[ "$(fd -t f '^main\.rs$' src)" = src/main.rs ]
printf 'baseline acceptance passed\n'
