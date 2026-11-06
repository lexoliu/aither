cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current || exit 1
cat /sys/fs/cgroup/memory.pressure 2>/dev/null || true
