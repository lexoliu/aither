set -eu
umask 077
root={{ root }}
destination="$root/{{ cache_key }}"
stage=''
cleanup() {
    if [ -n "$stage" ]; then
        chmod 700 "$stage/bin" 2>/dev/null || true
        rm -rf "$stage"
    fi
}
trap cleanup EXIT HUP INT TERM
verify_tools() {
    bin="$1/bin"
    [ -d "$bin" ] && [ ! -L "$bin" ] || return 1
{% for tool in tools %}    [ -f "$bin/{{ tool.command }}" ] && [ ! -L "$bin/{{ tool.command }}" ] || return 1
{% if macos %}    actual=$(shasum -a 256 "$bin/{{ tool.command }}")
    mode=$(stat -f '%Lp' "$bin/{{ tool.command }}")
{% else %}    actual=$(sha256sum "$bin/{{ tool.command }}")
    mode=$(stat -c '%a' "$bin/{{ tool.command }}")
{% endif %}    [ "${actual%% *}" = '{{ tool.sha256 }}' ] && [ "$mode" = '555' ] || return 1
{% endfor %}}
if [ -e "$destination" ] || [ -L "$destination" ]; then
    [ -L "$destination" ] && verify_tools "$destination" || { echo 'tool cache integrity failure' >&2; exit 1; }
{% if !probe_only %}    cat >/dev/null
{% endif %}else
{% if probe_only %}    exit 3
{% else %}    mkdir -p "$root"
    chmod 700 "$root"
    stage=$(mktemp -d "$root/.initializing.XXXXXXXX")
    tar -xf - -C "$stage"
    chmod 555 "$stage/bin/"*
    chmod 555 "$stage/bin"
    verify_tools "$stage" || { echo 'tool installation integrity failure' >&2; exit 1; }
    if ln -s -n "$stage" "$destination" 2>/dev/null; then
        stage=''
    else
        [ -L "$destination" ] && verify_tools "$destination" || { echo 'tool activation failed' >&2; exit 1; }
    fi
{% endif %}fi
printf '%s\n' "$destination/bin"
