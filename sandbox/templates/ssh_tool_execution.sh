{{ binary }} run --network allow --executable {{ tools_dir }}{% if let Some(grant) = uv_grant %} --grant {{ grant }}{% endif %} -- {{ program }} {{ arguments }}
