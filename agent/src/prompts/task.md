Launch a specialized subagent to handle complex tasks autonomously.

Use this tool for:
- Exploring codebases (use subagent_type: "explore")
- Planning implementations (use subagent_type: "plan")
- Any multi-step research or analysis task

Available subagent types:
{{types}}

Example: {"description": "Explore codebase", "prompt": "Find all Rust source files and summarize the project structure", "subagent_type": "explore"}
