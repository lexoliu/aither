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
# The script itself runs in the managed bash: `bash` on PATH is the shell
# running this script, and it is the pinned 5.3 series, not the host's.
banner=$(bash --version)
banner=${banner%%$'\n'*}
case $banner in
    "GNU bash, version $BASH_VERSION "*) ;;
    *) echo "bash on PATH is not the running managed bash: $banner ($BASH_VERSION)" >&2; exit 1 ;;
esac
case $BASH_VERSION in 5.3.*) ;; *) echo "not bash 5.3: $BASH_VERSION" >&2; exit 1 ;; esac
# The managed aither-sed prints line ranges and fails on anything else.
[ "$(printf 'a\nb\nc\n' | aither-sed -n '2,3p')" = "$(printf 'b\nc')" ]
if aither-sed 's/a/b/' /dev/null 2>/dev/null; then exit 1; fi
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
# The managed uv and python3 lead PATH; the versions they report are the pins
# the caller compares.
uv --version
python3 --version
[ "$(command -v python3)" = "$(dirname "$(command -v uv)")/python3" ]
# uv keeps its cache and the Pythons it installs in one directory shared by
# every session, outside the session's HOME.
[ -n "$UV_CACHE_DIR" ] && [ -n "$UV_PYTHON_INSTALL_DIR" ]
case "$UV_CACHE_DIR:$UV_PYTHON_INSTALL_DIR" in *"$HOME"/*) false;; esac
# A project's own Python wins over the managed one. This needs network access
# the first time: uv downloads the pinned CPython into UV_PYTHON_INSTALL_DIR.
mkdir pinned
printf '3.12\n' >pinned/.python-version
(cd pinned && uv run python --version) | rg -q '^Python 3\.12\.[0-9]+$'
mkdir venv-project
(
    cd venv-project
    uv venv --python 3.12 -q
    printf '[project]\nname = "venv-project"\nversion = "0"\nrequires-python = ">=3.12"\n' >pyproject.toml
    uv run python --version | rg -q '^Python 3\.12\.[0-9]+$'
)
# Without one, uv run uses the managed python3.
mkdir unpinned
(cd unpinned && uv run python --version) | rg -q "^$(python3 --version)\$"
printf 'baseline acceptance passed\n'
