set -eu
directory=$(xcode-select -p 2>/dev/null) || exit 4
[ -x "$directory/usr/bin/git" ] || exit 4
