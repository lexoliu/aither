Manage files using the "operation" field.

<operations>
- **read**: `{"operation": "read", "path": "file.txt"}`
- **write**: `{"operation": "write", "path": "file.txt", "content": "hello world"}`
- **append**: `{"operation": "append", "path": "file.txt", "content": "more text"}`
- **delete**: `{"operation": "delete", "path": "file.txt"}`
- **list**: `{"operation": "list", "path": "src/"}` or `{"operation": "list"}` for current directory
</operations>
