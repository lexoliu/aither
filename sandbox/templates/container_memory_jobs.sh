for dir in /proc/[0-9]*; do
  id=$(tr '\0' '\n' < "$dir/environ" 2>/dev/null | sed -n 's/^AITHER_CONTAINER_JOB=//p' | head -n 1)
  [ -n "$id" ] || continue
  kib=$(sed -n 's/^Pss: *\([0-9][0-9]*\) kB$/\1/p' "$dir/smaps_rollup" 2>/dev/null)
  [ -n "$kib" ] && echo "$id $kib"
done
exit 0
