set -eu
umask 077
root={{ root }}
destination="$root/{{ cache_key }}"
stage=''
mkdir -p "$root/uv"
cleanup() {
    if [ -n "$stage" ]; then
        chmod -R u+w "$stage" 2>/dev/null || true
        rm -rf "$stage"
    fi
}
trap cleanup EXIT HUP INT TERM
file_digest() {
{% if macos %}    set -- $(shasum -a 256 "$1")
{% else %}    set -- $(sha256sum "$1")
{% endif %}    printf '%s' "$1"
}
file_mode() {
{% if macos %}    stat -f '%Lp' "$1"
{% else %}    stat -c '%a' "$1"
{% endif %}}
verify_tools() {
    bin="$1/bin"
    [ -d "$bin" ] && [ ! -L "$bin" ] || return 1
{% for tool in tools %}    [ -f "$bin/{{ tool.command }}" ] && [ ! -L "$bin/{{ tool.command }}" ] || return 1
    [ "$(file_digest "$bin/{{ tool.command }}")" = '{{ tool.sha256 }}' ] && [ "$(file_mode "$bin/{{ tool.command }}")" = '555' ] || return 1
{% if !tool.aliases.is_empty() %}    for alias in {{ tool.aliases }}; do
        [ -L "$bin/$alias" ] && [ "$(readlink "$bin/$alias")" = '{{ tool.command }}' ] || return 1
    done
{% endif %}{% endfor %}}
{% if let Some(python) = python %}verify_python() {
    bin="$1/bin"
    [ -L "$bin/python3" ] && [ "$(readlink "$bin/python3")" = '{{ python.link_target }}' ] || return 1
    interpreter="$bin/{{ python.link_target }}"
    [ -f "$interpreter" ] && [ ! -L "$interpreter" ] || return 1
    [ "$(file_digest "$interpreter")" = '{{ python.interpreter_sha256 }}' ] && [ "$(file_mode "$interpreter")" = '555' ] || return 1
}
# Only directories, regular files and links that stay inside the tree.
verify_python_tree() {
    tree="$1"
    [ -d "$tree" ] && [ ! -L "$tree" ] || return 1
    [ -z "$(find "$tree" ! -type f ! -type d ! -type l -print -quit)" ] || return 1
    inside=$(cd "$tree" && pwd -P)
    find "$tree" -type l | while IFS= read -r link; do
        target=$(readlink "$link")
        case "$target" in /*) exit 1;; esac
        resolved=$(cd "$(dirname "$link")" && cd "$(dirname "$target")" 2>/dev/null && pwd -P) || continue
        case "$resolved/" in "$inside"/*) ;; *) exit 1;; esac
    done
}
install_python() {
    installed_stage="$1"
    scratch="$installed_stage/.python-install"
    # The caller tests this function's status, which turns `set -e` off in it:
    # every step reports its own failure.
    mkdir "$scratch" "$scratch/python" || return 1
    set -- "UV_PYTHON_INSTALL_DIR=$scratch/python"
    for name in {{ python.network_environment }}; do
        eval "if [ -n \"\${$name+x}\" ]; then set -- \"\$@\" \"$name=\$$name\"; fi"
    done
    env -i PATH=/usr/bin:/bin "$@" "$installed_stage/bin/uv" python install --no-bin --no-config --no-cache '{{ python.request }}' >&2 || return 1
    [ -d "$scratch/python/{{ python.request }}" ] && [ ! -L "$scratch/python/{{ python.request }}" ] || return 1
    mkdir "$installed_stage/bin/.python" || return 1
    mv "$scratch/python/{{ python.request }}" "$installed_stage/bin/.python/{{ python.request }}" || return 1
    rm -rf "$scratch" || return 1
    verify_python_tree "$installed_stage/bin/.python/{{ python.request }}" || return 1
    ln -s '{{ python.link_target }}' "$installed_stage/bin/python3" || return 1
    chmod -R a-w "$installed_stage/bin/.python" || return 1
}
{% endif %}if [ -e "$destination" ] || [ -L "$destination" ]; then
    [ -L "$destination" ] && verify_tools "$destination"{% if python.is_some() %} && verify_python "$destination"{% endif %} || { echo 'tool cache integrity failure' >&2; exit 1; }
{% if !probe_only %}    cat >/dev/null
{% endif %}else
{% if probe_only %}    exit 3
{% else %}    mkdir -p "$root"
    chmod 700 "$root"
    stage=$(mktemp -d "$root/.initializing.XXXXXXXX")
    tar -xf - -C "$stage"
    chmod 555 "$stage/bin/"*
    verify_tools "$stage" || { echo 'tool installation integrity failure' >&2; exit 1; }
{% if python.is_some() %}    install_python "$stage" || { echo 'python installation failed' >&2; exit 1; }
    verify_python "$stage" || { echo 'python installation integrity failure' >&2; exit 1; }
{% endif %}    chmod 555 "$stage/bin"
    if ln -s -n "$stage" "$destination" 2>/dev/null; then
        stage=''
    else
        [ -L "$destination" ] && verify_tools "$destination"{% if python.is_some() %} && verify_python "$destination"{% endif %} || { echo 'tool activation failed' >&2; exit 1; }
    fi
{% endif %}fi
printf '%s\n' "$destination/bin"
