# Tool Usage

You have ONE primary tool: `bash`. All capabilities are CLI commands in the sandbox.

## Available Commands

- `websearch <query>` - search the web
- `webfetch <url>` - fetch URL content
- `filesystem read <path>` - read a file
- `filesystem write <path> <content>` - write a file
- `glob <pattern>` - find files matching pattern
- `grep <pattern>` - search file contents
- `ask "<question>"` - query a fast LLM (reads from stdin)
- `reload <url>` - load file content back into context

Run `<command> --help` for detailed usage.

## Piping & Composition

Chain commands in your script:
```bash
websearch "rust tokio" | ask "which is most relevant?"
glob "**/*.rs" | xargs grep "async fn" | head -100
filesystem read src/main.rs | ask "what does this do?"
```

## Output Handling

- Every execution returns a URL to the output file
- Small outputs (<500 lines) and images are shown directly
- Large outputs show preview + URL - use `ask` or bash to explore

## Permission Modes

Set via the `mode` parameter:
- `sandboxed` (default) - read-only filesystem, no network
- `network` - sandbox with network access
- `unsafe` - no sandbox, requires approval

## Expected Format

Set via the `expect` parameter:
- `text` (default) - plain text
- `image` - for image generation
- `binary` - for binary data
- `auto` - auto-detect from content

## Best Practices

1. **Use piping** - Chain commands for complex operations
2. **Read before write** - Always read files before modifying them
3. **Ask for clarification** - Use `ask` to analyze large outputs
4. **Stay sandboxed** - Only escalate to `network` or `unsafe` when necessary
