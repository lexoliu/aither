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
printf 'baseline acceptance passed\n'
