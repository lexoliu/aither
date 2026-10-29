cwd_report={{ escaped_report }}; trap 'pwd -P > "$cwd_report"' EXIT; {{ script }}
