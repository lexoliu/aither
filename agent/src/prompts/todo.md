Use this tool to create and manage a structured task list for tracking progress on complex work.

## When to Use This Tool

Use this tool proactively in these scenarios:

1. **Complex multi-step tasks** - When a task requires 3 or more distinct steps
2. **User provides multiple tasks** - When users provide a list of things to be done
3. **After receiving new instructions** - Immediately capture requirements as todos
4. **When starting a task** - Mark it as in_progress BEFORE beginning work
5. **After completing a task** - Mark it as completed immediately

## When NOT to Use This Tool

Skip using this tool when:
- There is only a single, straightforward task
- The task can be completed in less than 3 steps
- The task is purely conversational or informational

## Task Format

Each task requires:
- **content**: Imperative form describing what to do (e.g., "Run tests", "Fix bug")
- **status**: One of `pending`, `in_progress`, or `completed`
- **activeForm**: Present continuous form (e.g., "Running tests", "Fixing bug")

## Task Management Rules

1. **Only ONE task in_progress at a time** - Complete current task before starting next
2. **Mark complete IMMEDIATELY** - Don't batch completions
3. **Update in real-time** - Keep the list current as you work
4. **Remove irrelevant tasks** - Clean up tasks no longer needed

## Example Usage

```json
{
  "todos": [
    {"content": "Analyze the codebase structure", "status": "completed", "activeForm": "Analyzing codebase structure"},
    {"content": "Implement the new feature", "status": "in_progress", "activeForm": "Implementing new feature"},
    {"content": "Write tests", "status": "pending", "activeForm": "Writing tests"},
    {"content": "Update documentation", "status": "pending", "activeForm": "Updating documentation"}
  ]
}
```

## Important

- This tool REPLACES the entire todo list each time - include all tasks (completed, in_progress, and pending)
- Never mark a task completed if it encountered errors or is only partially done
- Break complex tasks into smaller, actionable items
