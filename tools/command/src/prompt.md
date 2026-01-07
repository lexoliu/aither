Execute shell commands. The `args` field MUST be a JSON array like ["arg1"], not a string like "[\"arg1\"]".

Examples:
- `{"program": "ls", "args": ["-la"]}`
- `{"program": "grep", "args": ["-r", "TODO", "src/"]}`
- `{"program": "git", "args": ["status"]}`
- `{"program": "pwd", "args": []}`
