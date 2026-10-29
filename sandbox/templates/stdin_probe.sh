# Prints "blocked" when a thread of the process tree rooted at $1 is blocked
# reading the pipe that is $1's standard input; prints nothing otherwise.
root=$1
target=$(readlink "/proc/$root/fd/0" 2>/dev/null) || exit 0
case $target in
  pipe:*) ;;
  *) exit 0 ;;
esac
# The read and readv syscall numbers of this architecture.
case $(uname -m) in
  x86_64) reads=" 0 19 " ;;
  aarch64|arm64|riscv64) reads=" 63 65 " ;;
  *) echo "unsupported architecture $(uname -m)"; exit 0 ;;
esac

# Grow the tree to a fixed point: every process whose parent is in it joins.
tree=" $root "
grown=1
while [ "$grown" = 1 ]; do
  grown=0
  for stat in /proc/[0-9]*/stat; do
    read -r line < "$stat" 2>/dev/null || continue
    pid=${stat#/proc/}
    pid=${pid%/stat}
    # The command name may contain spaces and parentheses; the fields that
    # matter follow the last ") ".
    set -- ${line##*) }
    case $tree in
      *" $2 "*)
        case $tree in
          *" $pid "*) ;;
          *) tree="$tree$pid "; grown=1 ;;
        esac
        ;;
    esac
  done
done

for pid in $tree; do
  for syscall in /proc/$pid/task/*/syscall; do
    read -r number fd _ < "$syscall" 2>/dev/null || continue
    case $reads in
      *" $number "*) ;;
      *) continue ;;
    esac
    # The descriptor is printed in hexadecimal.
    [ "$(readlink "/proc/$pid/fd/$((fd))" 2>/dev/null)" = "$target" ] && { echo blocked; exit 0; }
  done
done
