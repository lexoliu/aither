Execute bash scripts in a sandboxed environment.

## Information Gathering

**ALWAYS prefer `websearch` and `webfetch` for getting information:**
- Use `websearch` to find documentation, tutorials, API references, or any online information
- Use `webfetch` to read specific URLs, documentation pages, or web content
- These commands work in sandboxed mode (no network flag needed) via IPC

Do NOT try to access the internet via curl/wget in sandboxed mode - use websearch/webfetch instead.

## Available Commands

- `websearch <query>` - search the web for information
- `webfetch <url>` - fetch and read URL content
- `todo write '<json>'` - manage task list
- `ask "<question>"` - query a fast LLM (reads from stdin)
- `reload <url>` - load file content back into context

Run `<command> --help` for detailed usage.

## Unix Pipelines

Chain commands freely using pipes for efficient data processing:
```bash
websearch "rust async" | ask "summarize the top results"
webfetch "https://docs.rs/tokio" | ask "what are the main features?"
find . -name "*.rs" | xargs grep "async fn" | head -50
cat src/main.rs | ask "explain this code"
```

## Parallel Execution

Run independent commands in parallel using `&` and `wait`:
```bash
websearch "rust error handling" > /tmp/a.txt &
websearch "rust async patterns" > /tmp/b.txt &
wait
cat /tmp/a.txt /tmp/b.txt | ask "compare these approaches"
```

Or use subshells for parallel pipelines:
```bash
(websearch "topic1" | process1) &
(websearch "topic2" | process2) &
wait
```

## Background Execution

Set `background: true` to run long-running commands asynchronously:
- Command returns immediately with status "running"
- Results are injected into context when the command completes
- Use for operations that may take a while (complex searches, large fetches)

## Output Storage

Outputs are automatically managed to prevent context overflow:
- **Small** (<500 lines): shown inline in context
- **Medium**: stored at path, preview shown - use `cat <path> | ask "..."` to process
- **Large**: stored at path only - reference path in subsequent commands

This enables infinite pipelines: bash outputs paths, you pipe paths to more commands.
Example workflow with large data:
```bash
# First call returns: stdout stored at /tmp/sandbox/emails.txt
osascript -e 'tell app "Mail" to get content of messages 1 thru 100 of inbox'
# Second call processes the stored output
cat /tmp/sandbox/emails.txt | ask "summarize unread emails about CI failures"
```

## Permission Modes (by side-effect scope)

Choose based on where side effects occur:

- `sandboxed` (default) - Side effects contained within sandbox:
  - Full host read access: any file, any program, GPU/NPU/accelerators
  - All writes go to sandbox directory only
  - Built-in commands: websearch, webfetch, todo, task, ask

- `network` - sandboxed + network access:
  - Side effects still contained in sandbox
  - Use for: curl, wget, ssh, git clone
  - NOT needed for websearch/webfetch (they work in sandboxed)

- `unsafe` - Side effects OUTSIDE sandbox:
  - Any action that modifies state beyond sandbox: host writes, app automation, system changes
  - **REQUIRED**: provide `reason` field explaining the side effect
  - Example: `{"script": "...", "mode": "unsafe", "reason": "Automating Mail.app to read inbox"}`

**Default to sandboxed.** Escalate only when side effects must escape sandbox.

## Expected Output Format

- `text` (default) - plain text
- `image` - image data (auto-loaded to context)
- `video` - video data
- `binary` - binary data
- `auto` - auto-detect from content
