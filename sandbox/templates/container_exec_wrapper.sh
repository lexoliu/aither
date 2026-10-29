# Records this shell's own in-container pid in the file $1 for the stdin
# probe and the kill path (Docker reports exec pids in the host's namespace),
# then runs the script $2 in a child shell, which sees none of the wrapper's
# arguments or variables. The file is removed when this shell exits. The
# wrapper never redirects its stdin, so its fd 0 stays the job's stdin pipe.
echo $$ > "$1" || { echo "cannot record the exec's pid in $1" >&2; exit 125; }
trap 'rm -f "$1"' EXIT
sh -c "$2"
# Not the last command, so the shell cannot replace itself with the child
# and skip its trap; the status is the script's.
exit
