set -eu
directory={{ directory }}
if [ ! -d "$directory/bin" ]; then echo "aither-image:missing-directory" >&2; exit 1; fi
{% for tool in tools %}if [ ! -f "$directory/bin/{{ tool }}" ] || [ -L "$directory/bin/{{ tool }}" ]; then echo "aither-image:missing-binary:{{ tool }}" >&2; exit 1; fi
{% endfor %}scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT HUP INT TERM
set --
{% for tool in tools %}heel run --network deny --working-dir "$scratch" --executable "$directory/bin" -- "$directory/bin/{{ tool }}" --version >"$scratch/{{ loop.index0 }}" &
set -- "$@" "$!"
{% endfor %}for child in "$@"; do wait "$child"; done
{% for tool in tools %}printf '%s\0' "$(cat "$scratch/{{ loop.index0 }}")"
{% endfor %}
