set -eu
directory={{ directory }}
{% for tool in tools %}[ -f "$directory/bin/{{ tool.command }}" ] && [ ! -L "$directory/bin/{{ tool.command }}" ]
actual=$(sha256sum "$directory/bin/{{ tool.command }}")
[ "${actual%% *}" = '{{ tool.sha256 }}' ]
{% endfor %}scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT HUP INT TERM
set --
{% for tool in tools %}heel run --network deny --working-dir "$scratch" --executable "$directory/bin" -- "$directory/bin/{{ tool.command }}" --version >"$scratch/{{ loop.index0 }}" &
set -- "$@" "$!"
{% endfor %}for child in "$@"; do wait "$child"; done
{% for tool in tools %}printf '%s\0' "$(cat "$scratch/{{ loop.index0 }}")"
{% endfor %}