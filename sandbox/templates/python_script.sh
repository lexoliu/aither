aither_py="${TMPDIR:-/tmp}/aither-{{ task_id }}.py"
trap 'rm -f "$aither_py"' EXIT
cat > "$aither_py" <<'AITHER_PYTHON_{{ task_id }}'
{{ code }}
AITHER_PYTHON_{{ task_id }}
{{ interpreter }} "$aither_py"
