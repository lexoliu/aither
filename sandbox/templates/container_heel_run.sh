{% if resolve_heel %}MAY_HEEL_BIN="$(command -v heel || true)";
if [ -z "$MAY_HEEL_BIN" ]; then echo "heel command not found in container" >&2; exit 127; fi;
{% endif %}{{ heel_bin }} run --isolation permissive --working-dir {{ escaped_working_dir }} --env-passthrough PATH --env-passthrough HOME{% if let Some(endpoint) = escaped_ipc_endpoint %} --ipc-endpoint {{ endpoint }} --grant "$MAY_IPC_DIR=rx"{% endif %}{% if let Some(level) = nice %} --nice {{ level }}{% endif %}{% if let Some(io) = io_priority %} --io-priority {{ io }}{% endif %} -- sh -c {{ escaped_script }}
