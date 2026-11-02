# Records this shell's own in-container pid in the file $1 for the stdin
# probe and the kill path (Docker reports exec pids in the host's namespace),
# then runs the script $2 in a child shell, which sees none of the wrapper's
# arguments or variables but the job marker AITHER_JOB, the file's path. The
# file is removed when this shell exits. The wrapper never redirects its
# stdin, so its fd 0 stays the job's stdin pipe.
#
# A job's processes live exactly as long as the job. When the script's shell
# has exited, $3 says what becomes of the processes it left running, found
# through $4, the job script, by their marker:
#   foreground  they are killed, and one line "<file> killed <pid> <name>" per
#               process goes to stderr, the file's path being the nonce that
#               tells it from the script's own output; a kill that fails is
#               reported as "<file> error <message>";
#   background  the wrapper stays until none is left, so the exec ends with its
#               last process; a helper that fails ends the wait with the same
#               error line.
# The text $1 on one line, built without any tool the job script may be
# missing.
oneline() {
  printf '%s' "$1" | while IFS= read -r oneline_part || [ -n "$oneline_part" ]; do
    printf '%s ' "$oneline_part"
  done
}
echo $$ > "$1" || { echo "cannot record the exec's pid in $1" >&2; exit 125; }
trap 'rm -f "$1"' EXIT
AITHER_JOB=$1 sh -c "$2"
status=$?
case $3 in
  foreground)
    if killed=$(sh -c "$4" sh reap "$1" 2>&1); then
      printf '%s\n' "$killed" | while read -r line; do
        [ -n "$line" ] && printf '%s killed %s\n' "$1" "$line" >&2
      done
    else
      printf '%s error %s\n' "$1" "$(oneline "$killed")" >&2
    fi
    ;;
  background)
    while :; do
      left=$(sh -c "$4" sh alive "$1") || {
        printf '%s error %s\n' "$1" "$(oneline "$left")" >&2
        break
      }
      [ -n "$left" ] || break
      sleep 1 || { printf '%s error cannot sleep while the job runs\n' "$1" >&2; break; }
    done
    ;;
esac
# Not the last command, so the shell cannot replace itself with the child
# and skip its trap; the status is the script's.
exit "$status"
