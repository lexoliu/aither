# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.20.0](https://github.com/May-Agent/aither/compare/agent-v0.19.0...agent-v0.20.0) - 2026-11-06

### Added

- *(sandbox)* [**breaking**] let the embedder prepare each subagent run's space
- *(agent)* [**breaking**] drop the sustained memory pressure alerts
- *(sandbox)* [**breaking**] judge the home environment's memory, a container's included, and never a remote host's
- *(agent)* record subagent starts and finishes on the tasks inbox
- *(agent)* [**breaking**] event-driven tasks.md with a fixed format and a stop gate
- *(agent)* warn about job memory and kill a runaway job left running
- *(sandbox)* [**breaking**] report why a job was killed and read job memory footprints

### Fixed

- gate a unix-only test and take ok_or for an eager error
- *(agent)* [**breaking**] judge tasks.md events at their cause and never read an unknown as unblocked
- *(agent)* [**breaking**] retry a model response whose stream breaks mid-response
- *(core)* [**breaking**] report each model's retry policy
- *(agent)* park a turn whose every call re-read unchanged output

### Other

- Merge remote-tracking branch 'origin/dev' into feat/subagent-run-space
- Merge pull request #553 from May-Agent/feat/171-runaway-guard
- Merge pull request #561 from May-Agent/fix/560-test-git-isolation
- Merge remote-tracking branch 'origin/dev' into feat/171-runaway-guard
- run every test's git through one isolated helper
- Merge pull request #557 from May-Agent/feat/129-event-driven-tasks
- *(agent)* keep the subagent tasks.md test's documents well formed
- Merge pull request #541 from May-Agent/fix/540-park-refused-poll

## [0.19.0](https://github.com/May-Agent/aither/compare/agent-v0.18.0...agent-v0.19.0) - 2026-11-05

### Added

- *(agent)* report each request's prompt-prefix hash chain

### Fixed

- *(agent)* commit each rendered block only to the source that rendered it

### Other

- Merge pull request #536 from May-Agent/chore/backmerge-0.18
- *(agent)* keep the mid-session MCP tool call in the foreground
- Merge main into dev after the 0.18.0 release
- Merge pull request #525 from May-Agent/fix/476-commit-own-blocks

## [0.18.0](https://github.com/May-Agent/aither/compare/agent-v0.17.0...agent-v0.18.0) - 2026-11-05

### Added

- *(agent)* [**breaking**] keep prefix content that changes mid-session live

### Fixed

- *(agent)* return MCP tool results through the full content conversion
- *(agent)* release a subagent call's run and continuation id when it is dropped
- *(agent)* retire the compaction and per-window state on checkpoint restore
- *(sandbox)* [**breaking**] keep a finished job's completion pending until its agent takes it
- *(agent)* let an alert carry bounded external content
- *(agent)* [**breaking**] stop a driven run only between handler calls
- *(agent)* match saved outputs by the platform's path separator
- *(agent)* [**breaking**] keep a run progressing while its caller handles an event
- *(agent)* compaction's file index finds saved outputs named by absolute path
- *(sandbox)* saved-output and killed-descendant alerts name what holds today
- *(mcp)* run one receive loop per client connection

### Other

- Merge pull request #527 from May-Agent/fix/478-mcp-content
- Merge pull request #524 from May-Agent/fix/475-drop-release
- Merge pull request #523 from May-Agent/fix/474-restore-compaction
- Merge pull request #522 from May-Agent/feat/521-live-prefix-content
- Merge pull request #520 from May-Agent/fix/518-pending-completion
- Merge pull request #519 from May-Agent/ci/cache-downloads
- restore pinned downloads from the Actions cache
- Merge pull request #516 from May-Agent/fix/504
- Merge pull request #505 from May-Agent/fix/502-stale-alerts
- *(agent)* build saved-output paths with the platform separator
- Merge pull request #507 from May-Agent/fix/501-run-progress

## [0.16.0](https://github.com/May-Agent/aither/compare/agent-v0.15.0...agent-v0.16.0) - 2026-11-05

### Added

- *(agent)* [**breaking**] give both builders the agent options through one trait
- *(agent)* [**breaking**] report a failed compaction's recovery as an AgentEvent
- *(agent)* [**breaking**] take each model's request parameters on the builders

### Other

- Merge pull request #457 from May-Agent/main
- *(agent)* accept a restoration read that finishes within the first poll

## [0.15.0](https://github.com/May-Agent/aither/compare/agent-v0.14.1...agent-v0.15.0) - 2026-11-04

### Added

- *(agent)* alert the agent when host memory pressure or machine load becomes high and when it ends
- *(agent)* [**breaking**] stamp user messages with UTC time and report timeout kill times
- *(agent)* [**breaking**] communication and working languages enforced on questions, replies and records
- *(agent)* [**breaking**] named subagents defined in TOML
- *(agent)* dispatch every tool call with its id and the agent making it
- *(sandbox)* hand IPC commands and observers the identity of the terminal call that ran them
- *(agent)* native compact tool and a one-shot pre-compaction alert
- *(agent)* [**breaking**] compaction points in tokens, a safe buffer and recovery from failed compaction
- *(agent)* [**breaking**] the subagent type is the subcommand, with a fixed tier and workspace
- *(agent)* bundle audit, review and general subagents with fixed tiers
- *(agent)* [**breaking**] host-ordered system layout with bounded external content

### Fixed

- *(agent)* a fork's inherited compact tool compacts the fork
- *(sandbox)* resolve IPC command paths against the caller's working directory
- *(agent)* keep terminal-run ask_user exchanges through compaction
- *(agent)* keep ask_user exchanges and user messages verbatim through compaction
- *(agent)* commit a compaction apply in one step
- *(agent)* keep post-fork alerts when a compaction handoff applies
- *(agent)* providers ignore other providers' cache sections
- *(agent)* route every compaction through one state
- *(agent)* add a compaction floor to the context policy
- *(agent)* keep a final answer's reasoning in the conversation
- *(context)* expose a checkpoint's sequence boundary
- *(agent)* refuse a context window with no room below the safe buffer

### Other

- Merge remote-tracking branch 'origin/dev' into fix/430-verbatim-user-messages
- Merge remote-tracking branch 'origin/dev' into fix/422-apply-cancel-safety
- Merge pull request #450 from May-Agent/fix/420-post-fork-alerts
- Merge remote-tracking branch 'origin/dev' into fix/420-post-fork-alerts
- Merge remote-tracking branch 'origin/dev' into feat/188-compact-tool
- Merge remote-tracking branch 'origin/dev' into fix/424-one-compaction-state
- Merge pull request #440 from May-Agent/fix/428-caller-cwd
- Merge pull request #437 from May-Agent/fix/426-compaction-floor
- Merge pull request #432 from May-Agent/feat/210-load-alerts
- Merge pull request #436 from May-Agent/fix/423-provider-cache-sections
- Merge remote-tracking branch 'origin/dev' into feat/210-load-alerts
- Merge pull request #427 from May-Agent/feat/415-subagent-subcommands
- merge dev into fix/430-verbatim-user-messages
- Merge pull request #435 from May-Agent/fix/421-final-answer-reasoning
- Merge pull request #431 from May-Agent/feat/417-utc-times
- Merge branch 'fix/424-one-compaction-state' into feat/188-compact-tool
- Merge remote-tracking branch 'origin/dev' into feat/415-subagent-subcommands
- Merge remote-tracking branch 'origin/dev' into feat/210-load-alerts
- Merge remote-tracking branch 'origin/dev' into feat/417-utc-times
- Merge pull request #434 from May-Agent/fix/433-checkpoint-boundary
- Merge remote-tracking branch 'origin/dev' into feat/188-compact-tool
- Merge pull request #425 from May-Agent/feat/418-call-identity
- Merge pull request #429 from May-Agent/feat/211-languages
- *(agent)* build the expected spill path with native separators
- Merge origin/dev into feat/415-subagent-subcommands

## [0.14.0](https://github.com/May-Agent/aither/compare/agent-v0.13.0...agent-v0.14.0) - 2026-11-03

### Added

- *(sed)* aither-sed prints line ranges for sed in agent scripts
- *(sandbox)* managed uv and a pinned python3
- *(sandbox)* build bash 5.3 in CI and run terminal scripts through it
- *(sandbox)* managed uutils coreutils, findutils xargs and fd

### Fixed

- *(sandbox)* read hints name head, tail and rg, not the blocked commands

### Other

- Merge origin/dev into feat/387-managed-bash
- Merge pull request #392 from May-Agent/feat/386-managed-coreutils

## [0.13.0](https://github.com/May-Agent/aither/compare/agent-v0.12.0...agent-v0.13.0) - 2026-11-03

### Added

- *(sandbox)* command results state the outages they ran through

### Other

- Merge remote-tracking branch 'origin/dev' into feat/344-connectivity
- Merge pull request #348 from May-Agent/feat/347-cache-warmth
- Merge pull request #350 from May-Agent/chore/merge-main-into-dev-0.12
- Merge remote-tracking branch 'origin/main' into chore/merge-main-into-dev-0.12

## [0.12.0](https://github.com/May-Agent/aither/compare/agent-v0.11.0...agent-v0.12.0) - 2026-11-03

### Added

- *(sandbox)* monitor tool delivering a watched job's new output as it arrives ([#316](https://github.com/May-Agent/aither/pull/316))
- *(agent)* validate BLOCKING marks through a host resolver
- *(sandbox)* give a committing subagent its own repository ([#298](https://github.com/May-Agent/aither/pull/298))
- *(agent)* let a run end with the model's turn while its background jobs run
- *(sandbox)* [**breaking**] merge terminal_background into terminal with required timeout and promote-at-wait
- apply external hook changes live
- *(agent)* command hooks on every shell command
- *(agent)* external hook runner and lifecycle hooks
- *(agent)* set the cache lifetime on TerminalAgentBuilder and inject the idle clock
- *(agent)* fork subagent that inherits the main agent's context and model
- *(agent)* compact before sending on a cold cache with a compaction model
- *(agent)* compact an idle session before its cache expires
- *(agent,sandbox)* keep subagents inside the main agent's authority

### Fixed

- *(terminal)* report promoted and killed jobs correctly
- *(agent)* honor BLOCKING marks in tasks.md supervision
- *(sandbox)* a command's descendants live exactly as long as its job ([#296](https://github.com/May-Agent/aither/pull/296))
- *(agent)* reviewers run through the shared hook runner
- *(agent)* pair every tool call with a result after a cancelled run
- *(agent,sandbox)* resolve canonical paths without the Windows verbatim prefix

### Other

- refresh README with agent, sandbox, MCP/ACP, and provider crates
- Drop the tool_hints block; give tools model-facing descriptions ([#303](https://github.com/May-Agent/aither/pull/303))
- Merge pull request #290 from lexoliu/feat/279-live-hooks
- build the fork fixture's reviewer hooks without the unix-only helpers
- Merge remote-tracking branch 'origin/dev' into feat/277-hooks
- *(agent)* run the sh-script hook tests on unix only
- Merge pull request #260 from lexoliu/feat/238-fork-subagent
- *(agent)* run the fork terminal-controls test on unix only
- *(agent)* satisfy clippy in the fork task-id diagnostic
- *(agent)* show the fork's messages when no task id is reported
- *(agent)* run the fork's background job with sleep so it starts on every platform
- Merge pull request #251 from lexoliu/feat/227-subagent-authority
- Merge origin/dev into feat/227-subagent-authority

## [0.11.0](https://github.com/lexoliu/aither/compare/agent-v0.10.1...agent-v0.11.0) - 2026-10-29

### Added

- *(sandbox)* [**breaking**] bound foreground terminal runs at 10 s and add terminal_background

### Fixed

- [**breaking**] text returned to the model is natural language, never serialized
- [**breaking**] fail instead of guessing an unknown model context window

### Other

- Merge remote-tracking branch 'origin/dev' into fix/208-no-json
- Merge remote-tracking branch 'origin/dev' into fix/208-no-json
- Emit events for provider-hosted tool calls ([#203](https://github.com/lexoliu/aither/pull/203))
- Merge pull request #200 from lexoliu/feat/189-bounded-terminal
- Merge pull request #185 from lexoliu/fix/181-no-context-fallback
- *(agent)* initialize before building live request messages in skills test

## [0.10.1](https://github.com/lexoliu/aither/compare/agent-v0.10.0...agent-v0.10.1) - 2026-10-20

### Fixed

- *(agent)* register terminal control tools natively again

### Other

- Merge pull request #176 from lexoliu/main

## [0.10.0](https://github.com/lexoliu/aither/compare/agent-v0.9.2...agent-v0.10.0) - 2026-10-18

### Other

- Upgrade dependencies to latest and trim feature sets ([#166](https://github.com/lexoliu/aither/pull/166))
- Merge pull request #164 from lexoliu/fix/tool-delta-incremental
- Emit incremental ToolCallDelta argument fragments
- Add preview_request_messages_with for non-mutating previews
- Box the provider respond stream once at the leaf

## [0.8.0](https://github.com/lexoliu/aither/compare/agent-v0.7.0...agent-v0.8.0) - 2026-10-15

### Added

- *(agent)* adopt an already prepared tool environment

## [0.7.0](https://github.com/lexoliu/aither/compare/agent-v0.6.0...agent-v0.7.0) - 2026-10-13

### Added

- terminal agents initialize a verified baseline tool distribution before running ([#107](https://github.com/lexoliu/aither/pull/107))
- alert on stale `tasks.md` during sustained tool activity ([#95](https://github.com/lexoliu/aither/pull/95))
- `push_system_alert` accepts `impl Into<String>`

### Changed

- render terminal output before compressing it; keep output that ends without a newline unterminated ([#102](https://github.com/lexoliu/aither/pull/102))
- split context metrics from reminder population and deliver ad-hoc alerts

### Removed

- the in-memory todo subsystem ([#92](https://github.com/lexoliu/aither/pull/92))
- the transient system message channel

## [0.6.0](https://github.com/lexoliu/aither/compare/agent-v0.5.0...agent-v0.6.0) - 2026-10-10

### Added

- add aither-codex provider and gate local web tools on hosted capabilities
- [**breaking**] tools report progress during a call; aither-mcp forwards it as notifications/progress ([#79](https://github.com/lexoliu/aither/pull/79))

### Other

- Merge remote-tracking branch 'origin/dev' into codex-provider

## [0.4.0](https://github.com/lexoliu/aither/compare/agent-v0.3.0...agent-v0.4.0) - 2026-09-13

### Other

- Add session/resume and a generic stream transport ([#44](https://github.com/lexoliu/aither/pull/44))

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
