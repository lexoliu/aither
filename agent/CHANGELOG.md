# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/agent-v0.2.0...agent-v0.3.0) - 2026-09-11

### Added

- *(llm)* [**breaking**] adaptive thinking, wider effort ladder, reasoning-state round-trip
- provider-native tools, sandbox network audit, and handoff file index
- structured terminal errors + BackgroundReason with timer grace window
- add Event::ToolCallDelta for streaming tool call progress
- *(agent)* expose rolling KV-cache statistics on Agent
- *(agent)* add TerminalAgentBuilder::system_text
- *(agent)* add AgentBuilder::system_text for raw prose blocks
- *(agent)* add raw-text system blocks and prefix fingerprint
- apply local workspace changes
- add support for file attachments in OpenAI API
- *(openai)* add Responses API streaming and subagent feedback
- *(mcp)* add MCP client and server support
- enhance CI/CD workflows and add skip-check logic for release management
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- enhance agent framework with memory management, planning, and execution capabilities
- implement conversation memory management with compression strategies
- Implement agent framework with planning and execution capabilities, add sub-agent and todo list management
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(agent)* collapse map/unwrap_or_else in the platform-gated OS probes
- *(agent)* improve subagent display hook with detailed feedback
- *(openai)* improve Gemini compatibility and tool calling

### Other

- clear five advisories, and stop building two copies of Arrow
- drop the dependencies nothing uses, and fix a real typo
- collapse nested conditionals into let-chains
- make the workspace clippy-clean under -D warnings
- *(sandbox)* [**breaking**] follow heel's IpcCommand onto per-instance naming
- Merge production-readiness work into dev
- *(skills)* drop prompt-trigger auto-activation
- checkpoint current workspace changes
- Finish tool result and terminal refactors
- Improve terminal stdin wait detection
- Store request approver state as snapshots
- Refine MCP tool service loop
- Drop synchronous subagent file loaders
- Store todo list as snapshots
- Simplify aither string assembly
- Advertise skill resource catalogs
- Test terminal first tool exposure
- Remove mutex from request approver
- Advertise skill resource catalogs
- Cover skill allowlist enforcement
- Cover skill checkpoint restoration
- Emit permission pause lifecycle from bash
- Normalize aither runtime formatting
- Structure subagent tool results
- Emit pause lifecycle for interactive tools
- Avoid duplicate skill prompt entries
- Bind ask-user requests to session context
- Persist active skills in checkpoints
- Wire skill registries into runtime activation
- Activate skills during agent runs
- Preserve turn accounting after background continuation
- Emit background terminal lifecycle events
- Include tool surface details in checkpoints
- Bind workspace requests to session context
- Add checkpoint export API
- Version subagent schemas and emit run lifecycle events
- Emit turn-boundary checkpoints from agent runtime
- Structure compaction handoff documents
- Remove bash agent format string helpers
- Type bash agent prompt sections
- Type container shell runtime for IPC
- Reassemble context after idle gaps
- Track tasks diffs in aither agent context
- Reassemble context before compaction
- Infer default SSH targets for bash runtimes
- Preserve seeded system blocks on context restore
- Add turn-boundary hooks to aither-agent
- Serialize tool error reminders structurally
- Serialize agent XML context structurally
- Template agent reminder prompts
- Prefer handoff over automatic compaction
- Track assembled context windows in aither-agent
- Expose serializable agent context snapshots
- Rename agent working docs and sandbox mounts
- Tighten native terminal tools for bash agents
- Introduce Context and container/terminal support
- Rename Task tool to SubagentTool
- Enhance RAG and sandbox with semantic processing
- Refactor and improve code quality across multiple modules
- improve code formatting and readability across multiple files
- Add shell session management, multi-question ask_user, and transcript
- Add UI tool request infrastructure and tools
- Merge fix-code-smells
- Integrate new features
- Derive tool descriptions from args rustdoc comments
- Add ask command and YAML frontmatter for subagents
- Add sandboxed bash tool and bash-first agent
- Add WebFetch tool and improve tool handling
- Add model registry and tiered model groups
- format code with cargo fmt
- *(agent)* redesign with minimal core and hook system
- Add repository guidelines, integrate Claude and OpenAI modules, and update workspace configuration
- Rename `ai-types` to aither
- Release v0.0.1
- Fix badge x2
- Fix badge
- Ready for 0.0.1 again
- Better document
- Add audio support
- Ready for 0.0.1
- initial comment
