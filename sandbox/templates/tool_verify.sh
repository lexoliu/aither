set -eu
directory={{ directory }}
if [ ! -d "$directory/bin" ]; then echo "aither-image:missing-directory" >&2; exit 1; fi
{% for tool in tools %}if [ ! -f "$directory/bin/{{ tool.command }}" ] || [ -L "$directory/bin/{{ tool.command }}" ]; then echo "aither-image:missing-binary:{{ tool.command }}" >&2; exit 1; fi
{% if !tool.aliases.is_empty() %}for alias in {{ tool.aliases }}; do
    if [ ! -L "$directory/bin/$alias" ] || [ "$(readlink "$directory/bin/$alias")" != '{{ tool.command }}' ]; then echo "aither-image:invalid-alias:$alias" >&2; exit 1; fi
done
{% endif %}{% endfor %}{% if let Some(python) = python %}if [ ! -L "$directory/bin/python3" ] || [ "$(readlink "$directory/bin/python3")" != '{{ python.link_target }}' ] || [ ! -f "$directory/bin/{{ python.link_target }}" ]; then echo "aither-image:invalid-python" >&2; exit 1; fi
{% endif %}scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT HUP INT TERM
set --
{% for tool in tools %}heel run --network deny --working-dir "$scratch" --executable "$directory/bin" -- "$directory/bin/{{ tool.command }}" --version >"$scratch/{{ loop.index0 }}" &
set -- "$@" "$!"
{% endfor %}{% if python.is_some() %}heel run --network deny --working-dir "$scratch" --executable "$directory/bin" -- "$directory/bin/python3" --version >"$scratch/python" &
set -- "$@" "$!"
{% endif %}for child in "$@"; do wait "$child"; done
{% for tool in tools %}printf '%s\0' "$(cat "$scratch/{{ loop.index0 }}")"
{% endfor %}{% if python.is_some() %}printf '%s\0' "$(cat "$scratch/python")"
{% endif %}
