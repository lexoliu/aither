cwd_report={{ escaped_report }}; eval {{ escaped_script }}; cwd_status=$?; pwd -P > "$cwd_report"; exit "$cwd_status"
