# Inspects or kills the job of one container exec. $1 is the action, $2 the
# file where the exec's wrapper recorded its own in-container pid (Docker
# reports exec pids in the host's namespace); the wrapper removes the file
# when it exits. The file's path is also the job's marker: the wrapper starts
# the job's script with it in AITHER_JOB, so the job's processes are found
# wherever they went — a process that outlives its shell is reparented to the
# container's init, and one that calls setsid leaves the shell's process group.
# A process that cleared the variable from its environment, or whose
# environment this script cannot read, is not found.
#
#   probe  prints "blocked <ticks>" when a process of the job tree is blocked
#          reading the job's stdin pipe, <ticks> being the tree's total CPU
#          time in USER_HZ ticks for the caller's CPU-idle confirmation;
#          prints nothing when it is not, or when the job has ended; prints
#          an error and exits 1 when it cannot probe.
#   kill   stops and then kills every process of the job tree, the wrapper
#          included, and removes the file; exits 3 when no pid is recorded,
#          1 when it cannot kill.
#   alive  prints the pid of every live process the wrapper's shell left
#          behind, one per line; prints nothing when there is none.
#   reap   stops and then kills every process the wrapper's shell left
#          behind and prints "<pid> <name>" for each; exits 1 when it cannot
#          kill one. The wrapper itself is not touched.
#
# The probe reads wchan, which needs only ptrace read access, unlike
# task/*/syscall, which needs attach — a sibling exec gets the former even
# under ptrace_scope=1. The wait-function names are data: kernels charge the
# sleep to the file operation (anon_pipe_read, pipe_read) or to the wait
# function (pipe_wait_readable), so it finds a `cat` sleeping in splice as
# readily as a shell in read. Kernels before 5.6 sleep in pipe_wait, which
# blocked writers share, and a reader parked in poll/select reports names
# like do_select; both stay undetected.
action=$1
pid_file=$2
case $action in
  probe|kill|alive|reap) ;;
  *) echo "unknown action $action"; exit 1 ;;
esac

# The job's processes are found by their environment, which only tr can make
# readable to the shell; without it nothing would be found, silently.
command -v tr >/dev/null 2>&1 || { echo "the job script needs tr"; exit 1; }

# The exec ended and its wrapper removed the file.
ended() {
  [ "$action" = kill ] && exit 3
  exit 0
}

# Succeeds while process $1 exists and is not a zombie.
alive() {
  read -r alive_line < "/proc/$1/stat" 2>/dev/null || return 1
  set -- ${alive_line##*) }
  [ "$1" != Z ]
}

# Succeeds when process $1 carries the job's marker in its environment. tr
# turns the NUL separators into spaces, which a shell can match and a
# command substitution can hold.
marked() {
  marked_env=$(tr '\0' ' ' < "/proc/$1/environ" 2>/dev/null) || return 1
  case " $marked_env " in
    *" AITHER_JOB=$pid_file "*) return 0 ;;
  esac
  return 1
}

# Sets $tree to " <root> <descendant> ... ": the root $1 (none when empty) and
# every live process carrying the job's marker, then every process whose
# parent is in the tree, until a pass adds none.
job_tree() {
  tree=" $1 "
  for seed_stat in /proc/[0-9]*/stat; do
    seed_pid=${seed_stat#/proc/}
    seed_pid=${seed_pid%/stat}
    case $tree in
      *" $seed_pid "*) continue ;;
    esac
    alive "$seed_pid" && marked "$seed_pid" && tree="$tree$seed_pid "
  done
  grown=1
  while [ "$grown" = 1 ]; do
    grown=0
    for tree_stat in /proc/[0-9]*/stat; do
      read -r tree_line < "$tree_stat" 2>/dev/null || continue
      tree_pid=${tree_stat#/proc/}
      tree_pid=${tree_pid%/stat}
      # The command name may contain spaces and parentheses; the fields that
      # matter follow the last ") ".
      set -- ${tree_line##*) }
      case $tree in
        *" $2 "*)
          case $tree in
            *" $tree_pid "*) ;;
            *) tree="$tree$tree_pid "; grown=1 ;;
          esac
          ;;
      esac
    done
  done
}

# Stops and then kills the tree grown from the root $1 (none when empty),
# setting $report to "<pid> <name>" lines for what it killed. A stopped process
# cannot fork, so the tree is stopped until a pass finds no process it has not
# stopped yet; then all of it is killed. Fails when a process survives.
kill_tree() {
  stopped=" "
  report=
  new=1
  while [ "$new" = 1 ]; do
    new=0
    job_tree "$1"
    for pid in $tree; do
      case $stopped in
        *" $pid "*) continue ;;
      esac
      stopped="$stopped$pid "
      # A zombie is already dead: nothing to stop, kill or report.
      alive "$pid" || continue
      kill -STOP "$pid" 2>/dev/null
      read -r comm < "/proc/$pid/comm" 2>/dev/null || comm=unknown
      report="$report$pid $comm
"
      new=1
    done
  done
  failed=0
  for pid in $stopped; do
    if ! kill -KILL "$pid" 2>/dev/null && alive "$pid"; then
      echo "cannot kill process $pid of the job"
      failed=1
    fi
  done
  [ "$failed" = 0 ]
}

# What the wrapper's shell left behind: no root, only the marked processes.
if [ "$action" = alive ]; then
  job_tree ""
  for pid in $tree; do
    echo "$pid"
  done
  exit 0
fi
if [ "$action" = reap ]; then
  kill_tree "" || exit 1
  printf '%s' "$report"
  exit 0
fi

[ -e "$pid_file" ] || ended
root=$(cat "$pid_file" 2>/dev/null)
case $root in
  ''|*[!0-9]*)
    [ -e "$pid_file" ] || ended
    echo "cannot read the job's pid file $pid_file"
    exit 1
    ;;
esac

if [ "$action" = kill ]; then
  alive "$root" || { rm -f "$pid_file"; exit 0; }
  kill_tree "$root" || exit 1
  rm -f "$pid_file"
  exit 0
fi

# The wrapper never redirects its own stdin, so the root's fd 0 is the job's
# stdin pipe.
if ! target=$(readlink "/proc/$root/fd/0" 2>/dev/null); then
  alive "$root" || exit 0
  echo "cannot inspect stdin of job root $root"
  exit 1
fi
case $target in
  pipe:\[*\]) ;;
  *) echo "stdin of job root $root is not a pipe but $target"; exit 1 ;;
esac

wchans=" anon_pipe_read pipe_read pipe_wait_readable "
denied=0

# Succeeds when a thread of process $1 sleeps in a pipe read. The kernel
# prints wchan 0 for a thread the caller may not ptrace-read, the same as for
# one it cannot unwind, so a sleeping thread reporting 0 is checked for
# access: a denied process is an error, never one that is not waiting.
pipe_sleeper() {
  sleeper=$1
  for task in /proc/$sleeper/task/*; do
    read -r task_line < "$task/stat" 2>/dev/null || continue
    set -- ${task_line##*) }
    [ "$1" = S ] || continue
    # The kernel writes wchan without a trailing newline, so `read` would
    # fill the variable and still fail at end of file.
    if ! wchan=$(cat "$task/wchan" 2>/dev/null); then
      [ -e "$task/wchan" ] && denied=1
      continue
    fi
    if [ "$wchan" = 0 ]; then
      readlink "/proc/$sleeper/exe" >/dev/null 2>&1 || ! alive "$sleeper" || denied=1
      continue
    fi
    case $wchans in
      *" $wchan "*) return 0 ;;
    esac
  done
  return 1
}

# Sets $reads and $writes to the link targets — "pipe:[<inode>]" for an
# anonymous pipe, the path for a FIFO — of the pipes and FIFOs process $1
# holds open for reading and for writing, each followed by "|"; $blockreads
# lists those read through a descriptor without O_NONBLOCK. Fails when
# the process has exited or denies inspection.
pipe_fds() {
  reads="|"
  writes="|"
  blockreads="|"
  # The exe link takes the same ptrace check as the descriptor links, which
  # a mode-bit test on the fd directory does not.
  if ! readlink "/proc/$1/exe" >/dev/null 2>&1; then
    alive "$1" && denied=1
    return 1
  fi
  for fd in /proc/$1/fd/*; do
    [ -p "$fd" ] || continue
    flags=
    while read -r key value; do
      [ "$key" = "flags:" ] && flags=$value
    done 2>/dev/null < "/proc/$1/fdinfo/${fd##*/}"
    if [ -z "$flags" ]; then
      [ -e "$fd" ] && denied=1
      continue
    fi
    if ! link=$(readlink "$fd" 2>/dev/null); then
      [ -e "$fd" ] && denied=1
      continue
    fi
    # fdinfo prints the open flags in octal; the access mode is the low two
    # bits: 0 read-only, 1 write-only, 2 both.
    mode=$((flags & 3))
    [ "$mode" != 1 ] && reads="$reads$link|"
    # O_NONBLOCK (04000) is in the same octal word; a read-capable descriptor
    # without it can sleep in a read of its pipe.
    [ "$mode" != 1 ] && [ $((flags & 04000)) = 0 ] && blockreads="$blockreads$link|"
    [ "$mode" != 0 ] && writes="$writes$link|"
  done
  return 0
}

# Succeeds when a process of the tree other than $1 holds open for writing
# one of the pipes listed in $2 ("link|link|"), or cannot be inspected.
fed_elsewhere() {
  for other in $tree; do
    [ "$other" = "$1" ] && continue
    if ! pipe_fds "$other"; then
      alive "$other" && return 0
      continue
    fi
    pending=$2
    while [ -n "$pending" ]; do
      link=${pending%%|*}
      pending=${pending#*|}
      case $writes in
        *"|$link|"*) return 0 ;;
      esac
    done
  done
  return 1
}

# wchan names the wait function, not the file, so a sleep in a pipe read
# proves a wait on the job's pipe only when nothing could feed any other pipe
# or FIFO the process holds open for reading. Each such pipe makes the wait
# ambiguous unless it is an anonymous pipe that only this single-threaded
# process holds open for writing — the private splice pipe of a `cat`. That
# excludes the shell of an `out=$(cmd)`, which sleeps on the substitution's
# pipe while holding the job's stdin; a FIFO, which any process may open and
# feed; a pipe whose write end a child also holds; and a pipe another thread
# of the reader may feed.
job_tree "$root"
blocked=0
total=0
for pid in $tree; do
  stat=/proc/$pid/stat
  if ! read -r line < "$stat" 2>/dev/null; then
    [ -e "$stat" ] && denied=1
    continue
  fi
  set -- ${line##*) }
  [ "$1" = Z ] && continue
  total=$((total + ${12} + ${13}))
  [ "$blocked" = 1 ] && continue
  pipe_sleeper "$pid" || continue
  pipe_fds "$pid" || continue
  case $reads in
    *"|$target|"*) ;;
    *) continue ;;
  esac
  set -- /proc/$pid/task/*
  threads=$#
  self_fed=
  ambiguous=0
  pending=${reads#|}
  while [ -n "$pending" ]; do
    link=${pending%%|*}
    pending=${pending#*|}
    [ "$link" = "$target" ] && continue
    # A pipe read only through O_NONBLOCK descriptors never sleeps a reader.
    case $link in
      pipe:\[*\])
        case $blockreads in
          *"|$link|"*) ;;
          *) continue ;;
        esac
        ;;
    esac
    case $link in
      pipe:\[*\]) ;;
      *) ambiguous=1 ;;
    esac
    case $writes in
      *"|$link|"*) ;;
      *) ambiguous=1 ;;
    esac
    [ "$threads" = 1 ] || ambiguous=1
    self_fed="$self_fed$link|"
  done
  [ "$ambiguous" = 0 ] || continue
  if [ -n "$self_fed" ] && fed_elsewhere "$pid" "$self_fed"; then
    continue
  fi
  # The process must still sleep in the pipe read it slept in before its
  # descriptors were read, or it may have moved on in between.
  pipe_sleeper "$pid" && blocked=1
done

if [ "$blocked" = 1 ]; then
  echo "blocked $total"
elif [ "$denied" = 1 ]; then
  alive "$root" || exit 0
  echo "cannot inspect every process of the job tree"
  exit 1
fi
