# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.0](https://github.com/May-Agent/aither/compare/sandbox-v0.20.0...sandbox-v0.21.0) - 2026-11-06

### Fixed

- *(sandbox)* give a subagent terminal read-only access to a worktree's git directory

## [0.20.0](https://github.com/May-Agent/aither/compare/sandbox-v0.19.0...sandbox-v0.20.0) - 2026-11-06

### Added

- *(sandbox)* [**breaking**] let the embedder prepare each subagent run's space
- *(sandbox)* [**breaking**] judge the home environment's memory, a container's included, and never a remote host's
- *(sandbox)* [**breaking**] report why a job was killed and read job memory footprints

### Fixed

- *(sandbox)* exempt the managed uv cache from the persistence denylist
- *(sandbox)* depend on heel 0.9.0 and verify its binary once per terminal
- gate a unix-only test and take ok_or for an eager error
- *(sandbox)* grant the session PATH directories read and execute
- *(sandbox)* apply the persistence write denylist to session policies
- *(sandbox)* grant a linked worktree's common git directory
- *(sandbox)* end a finished job's stdin writer thread
- *(sandbox)* record a file read for the unchanged-file guard only after the command exits 0
- *(agent)* park a turn whose every call re-read unchanged output

### Other

- Merge pull request #563 from May-Agent/feat/subagent-run-space
- Merge remote-tracking branch 'origin/dev' into fix/558-heel-0.9
- Merge remote-tracking branch 'origin/dev' into feat/subagent-run-space
- Merge pull request #565 from May-Agent/fix/uv-cache-denylist
- *(sandbox)* judge the uv cache denylist test by outcome on every platform
- Merge pull request #553 from May-Agent/feat/171-runaway-guard
- Merge remote-tracking branch 'origin/dev' into fix/558-heel-0.9
- Merge pull request #556 from May-Agent/fix/551-path-grants
- Merge pull request #555 from May-Agent/fix/550-git-persistence-denylist
- Merge pull request #561 from May-Agent/fix/560-test-git-isolation
- Merge remote-tracking branch 'origin/dev' into feat/171-runaway-guard
- run every test's git through one isolated helper
- Merge pull request #554 from May-Agent/fix/549-worktree-commondir
- Merge branch 'fix/550-git-persistence-denylist' into fix/551-path-grants
- *(sandbox)* check a refused git init by its outcome on every platform

## [0.18.0](https://github.com/May-Agent/aither/compare/sandbox-v0.17.0...sandbox-v0.18.0) - 2026-11-05

### Added

- *(agent)* [**breaking**] keep prefix content that changes mid-session live
- *(sandbox)* [**breaking**] expose the session sandbox policy to hosts

### Fixed

- *(agent)* release a subagent call's run and continuation id when it is dropped
- *(sandbox)* [**breaking**] keep a finished job's completion pending until its agent takes it
- *(sandbox)* [**breaking**] let an argument schema mark an optional field positional
- *(agent)* compaction's file index finds saved outputs named by absolute path
- *(sandbox)* saved-output and killed-descendant alerts name what holds today

### Other

- Merge pull request #524 from May-Agent/fix/475-drop-release
- Merge pull request #522 from May-Agent/feat/521-live-prefix-content
- Merge pull request #520 from May-Agent/fix/518-pending-completion
- Merge pull request #514 from May-Agent/feat/512-sandbox-policy
- Merge remote-tracking branch 'origin/dev' into fix/503

## [0.17.0](https://github.com/May-Agent/aither/compare/sandbox-v0.16.0...sandbox-v0.17.0) - 2026-11-05

### Added

- *(sandbox)* run scripts on the terminal serving a bound command
- *(sandbox)* register IPC commands for sandboxed modes only

### Other

- Merge remote-tracking branch 'origin/main' into chore/merge-main-0.16.0
- Merge pull request #467 from May-Agent/feat/463-bound-terminal-runner

## [0.16.0](https://github.com/May-Agent/aither/compare/sandbox-v0.15.0...sandbox-v0.16.0) - 2026-11-05

### Added

- *(sandbox)* [**breaking**] ask with the parameters of the answering model

## [0.15.0](https://github.com/May-Agent/aither/compare/sandbox-v0.14.1...sandbox-v0.15.0) - 2026-11-04

### Added

- *(agent)* alert the agent when host memory pressure or machine load becomes high and when it ends
- *(agent)* [**breaking**] stamp user messages with UTC time and report timeout kill times
- *(sandbox)* main-exclusive commands and usage on a parse failure
- *(sandbox)* hand IPC commands and observers the identity of the terminal call that ran them
- *(agent)* [**breaking**] the subagent type is the subcommand, with a fixed tier and workspace

### Fixed

- *(sandbox)* retry transient tool artifact download failures
- *(sandbox)* never splice one stream's bytes into another's line
- *(sandbox)* report a timeout-killed job as killed
- *(sandbox)* resolve IPC command paths against the caller's working directory
- *(sandbox)* retry running a freshly installed tool while its file is busy
- *(agent)* keep terminal-run ask_user exchanges through compaction
- *(sandbox)* a process in an exec no longer escapes its container job

### Other

- *(sandbox)* note when query_tool_handler_from panics
- Merge remote-tracking branch 'origin/dev' into fix/430-verbatim-user-messages
- Merge pull request #442 from May-Agent/fix/441-timeout-kill-report
- Merge pull request #443 from May-Agent/fix/371-monitor-stream-splice
- Merge pull request #439 from May-Agent/fix/438-etxtbsy
- Merge pull request #440 from May-Agent/fix/428-caller-cwd
- *(sandbox)* format the busy-retry call
- Merge remote-tracking branch 'origin/dev' into feat/210-load-alerts
- Merge pull request #427 from May-Agent/feat/415-subagent-subcommands
- Merge pull request #431 from May-Agent/feat/417-utc-times
- Merge remote-tracking branch 'origin/dev' into feat/415-subagent-subcommands
- Merge remote-tracking branch 'origin/dev' into feat/210-load-alerts
- Merge pull request #410 from May-Agent/main
- Merge pull request #407 from May-Agent/fix/342-wrapper-leftover
- *(sandbox)* spell the non-ASCII environment value without a flagged word

## [0.14.1](https://github.com/May-Agent/aither/compare/sandbox-v0.14.0...sandbox-v0.14.1) - 2026-11-03

### Fixed

- *(sandbox)* ManagedTool parses aither-sed, the id the release manifest uses

### Other

- Merge pull request #404 from May-Agent/main

## [0.14.0](https://github.com/May-Agent/aither/compare/sandbox-v0.13.0...sandbox-v0.14.0) - 2026-11-03

### Added

- *(sed)* aither-sed prints line ranges for sed in agent scripts
- *(sandbox)* grep, sed, awk and find fail in agent scripts with their replacement
- *(sandbox)* managed uv and a pinned python3
- *(sandbox)* build bash 5.3 in CI and run terminal scripts through it
- *(sandbox)* managed uutils coreutils, findutils xargs and fd
- *(sandbox)* fail initialization on macOS without the Command Line Tools
- *(sandbox)* existing files change only through apply_patch

### Fixed

- *(sandbox)* read hints name head, tail and rg, not the blocked commands
- *(sandbox)* unlock the Python tree only when the bundle has one; type the outage test's cache
- *(sandbox)* leave stdbuf out of the coreutils aliases, which the musl builds lack
- *(sandbox)* keep the apply_patch rule free of `which`

### Other

- Merge origin/dev into feat/387-managed-bash
- Merge pull request #392 from May-Agent/feat/386-managed-coreutils
- Merge pull request #391 from May-Agent/feat/390-clt-check
- Merge pull request #385 from May-Agent/feat/384-apply-patch-only

## [0.13.0](https://github.com/May-Agent/aither/compare/sandbox-v0.12.0...sandbox-v0.13.0) - 2026-11-03

### Added

- *(sandbox)* command results state the outages they ran through

### Other

- *(sandbox)* accept every name the env-cleaned setsid command can be killed under
- Merge remote-tracking branch 'origin/main' into chore/merge-main-into-dev-0.12

## [0.12.0](https://github.com/May-Agent/aither/compare/sandbox-v0.11.0...sandbox-v0.12.0) - 2026-11-03

### Added

- *(sandbox)* [**breaking**] prepare a cached host program without re-extracting its archive
- *(sandbox)* give a committing subagent its own repository ([#298](https://github.com/May-Agent/aither/pull/298))
- *(agent)* let a run end with the model's turn while its background jobs run
- *(sandbox)* [**breaking**] merge terminal_background into terminal with required timeout and promote-at-wait
- *(agent)* command hooks on every shell command
- *(agent)* external hook runner and lifecycle hooks
- *(sandbox)* carry the main agent's working directory across commands
- *(sandbox)* refuse name- and pattern-based kills
- *(agent,sandbox)* keep subagents inside the main agent's authority

### Fixed

- *(sandbox)* a command's descendants live exactly as long as its job ([#296](https://github.com/May-Agent/aither/pull/296))
- *(sandbox)* record the carried working directory without an EXIT trap
- *(sandbox)* spell unparsable consistently and let typos skip option strings
- *(agent,sandbox)* resolve canonical paths without the Windows verbatim prefix

### Other

- Run the host Rust toolchain inside the local terminal
- host toolchain in the local terminal (needs heel#149)
- Add runtime grants to TerminalTool
- Merge pull request #204 from lexoliu/feat/terminal-env-inject
- Add TerminalEnvSource env injection to terminal tool
- refresh README with agent, sandbox, MCP/ACP, and provider crates
- Run approved-unsafe terminal commands under a permissive heel profile ([#304](https://github.com/May-Agent/aither/pull/304))
- Drop the tool_hints block; give tools model-facing descriptions ([#303](https://github.com/May-Agent/aither/pull/303))
- Merge remote-tracking branch 'origin/dev' into feat/277-hooks
- Merge pull request #267 from lexoliu/fix/265-kill-pid-race
- *(sandbox)* compare the unterminated here-document output without its trailing newline
- Merge pull request #260 from lexoliu/feat/238-fork-subagent
- Merge remote-tracking branch 'origin/main' into chore/merge-main-into-dev-0.11
- Merge pull request #251 from lexoliu/feat/227-subagent-authority
- *(sandbox)* gate the recording container exec to unix like its only user
- Merge origin/dev into feat/227-subagent-authority

## [0.11.0](https://github.com/lexoliu/aither/compare/sandbox-v0.10.1...sandbox-v0.11.0) - 2026-10-29

### Added

- *(sandbox)* [**breaking**] bound foreground terminal runs at 10 s and add terminal_background

### Fixed

- [**breaking**] text returned to the model is natural language, never serialized
- *(sandbox)* keep cut output inline when the cut is not smaller

### Other

- Merge remote-tracking branch 'origin/dev' into fix/208-no-json
- Merge remote-tracking branch 'origin/dev' into fix/208-no-json
- Merge pull request #200 from lexoliu/feat/189-bounded-terminal

## [0.10.1](https://github.com/lexoliu/aither/compare/sandbox-v0.10.0...sandbox-v0.10.1) - 2026-10-20

### Fixed

- *(acp)* title and classify the tools that actually arrive natively

### Other

- Merge pull request #176 from lexoliu/main

## [0.10.0](https://github.com/lexoliu/aither/compare/sandbox-v0.9.2...sandbox-v0.10.0) - 2026-10-18

### Other

- Upgrade dependencies to latest and trim feature sets ([#166](https://github.com/lexoliu/aither/pull/166))
- Render container bridge responses like heel ipc does
- deliver tool argv verbatim and render results for the terminal ([#160](https://github.com/lexoliu/aither/pull/160))
- Bump zip to 9.0.5 for RUSTSEC-2026-0367/0368/0369

## [0.9.2](https://github.com/lexoliu/aither/compare/sandbox-v0.9.1...sandbox-v0.9.2) - 2026-10-16

### Other

- Encode container IPC responses with field names

## [0.9.1](https://github.com/lexoliu/aither/compare/sandbox-v0.9.0...sandbox-v0.9.1) - 2026-10-16

### Other

- Run a prepared command in the container acceptance test
- Grant the container heel run execute on the prepared tools directory

## [0.9.0](https://github.com/lexoliu/aither/compare/sandbox-v0.8.0...sandbox-v0.9.0) - 2026-10-15

### Fixed

- *(sandbox)* serve the container IPC bridge on every host

### Other

- Merge pull request #136 from lexoliu/fix/135-container-posix-paths
- Merge pull request #134 from lexoliu/fix/133-windows-ipc-bridge

## [0.8.0](https://github.com/lexoliu/aither/compare/sandbox-v0.7.0...sandbox-v0.8.0) - 2026-10-15

### Added

- *(agent)* adopt an already prepared tool environment

### Other

- *(sandbox)* run the shell-fixture adoption test on unix only
- Keep the general TerminalTool methods on a plain Clone state
- Bind a child terminal's control commands to its own jobs

## [0.7.0](https://github.com/lexoliu/aither/compare/sandbox-v0.6.0...sandbox-v0.7.0) - 2026-10-13

### Added

- baseline tool runtime: `ToolDistribution` and `initialize_tools` prepare verified tools for local, container and SSH execution ([#107](https://github.com/lexoliu/aither/pull/107))
- `TerminalTool::with_process_priority` — typed `ProcessPriority` (`CpuPriority`/`IoPriority`/`NiceLevel`) threaded through one-shot, persistent, background, and container executions ([#112](https://github.com/lexoliu/aither/pull/112))
- `ContainerIpcRoute`/`BridgeAddress` — container IPC binds an explicit host-side address instead of a wildcard ([#112](https://github.com/lexoliu/aither/pull/112))
- `aither-patch`: a sandbox-executed `apply_patch` utility ([#108](https://github.com/lexoliu/aither/pull/108))

### Changed

- [**breaking**] terminal execution requires a prepared tool environment
- [**breaking**] `ContainerShellRuntime::new` takes a `ContainerIpcRoute` instead of an IPC hostname
- upgrade the sandbox backend to heel 0.4.1
- nested-container sandboxes relay IPC through `heel run --ipc-endpoint` instead of environment passthrough

### Fixed

- container IPC uses the runtime's host-side route instead of assuming that containers can reach host loopback

## [0.6.0](https://github.com/lexoliu/aither/compare/sandbox-v0.5.0...sandbox-v0.6.0) - 2026-10-10

### Added

- add aither-codex provider and gate local web tools on hosted capabilities
- [**breaking**] tools report progress during a call; aither-mcp forwards it as notifications/progress ([#79](https://github.com/lexoliu/aither/pull/79))

### Other

- Merge branch 'main' into dev
- Merge remote-tracking branch 'origin/dev' into codex-provider

## [0.5.0](https://github.com/lexoliu/aither/compare/sandbox-v0.4.1...sandbox-v0.5.0) - 2026-09-17

### Other

- add ToolResult::Parts for multi-content tool results ([#71](https://github.com/lexoliu/aither/pull/71))
- poll child.pid until it parses in descendant-kill test ([#65](https://github.com/lexoliu/aither/pull/65))

## [0.4.0](https://github.com/lexoliu/aither/compare/sandbox-v0.3.0...sandbox-v0.4.0) - 2026-09-13

### Other

- Add session/resume and a generic stream transport ([#44](https://github.com/lexoliu/aither/pull/44))

## [0.3.0](https://github.com/lexoliu/aither/compare/sandbox-v0.2.0...sandbox-v0.3.0) - 2026-09-11

### Added

- provider-native tools, sandbox network audit, and handoff file index
- structured terminal errors + BackgroundReason with timer grace window
- apply local workspace changes
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- fix sandbox background terminal stdin and task tracking

### Other

- *(sandbox)* gate the Unix-only sandbox tests behind cfg(unix)
- *(sandbox)* let the tree-sitter grammars follow their feature
- clear five advisories, and stop building two copies of Arrow
- drop the dependencies nothing uses, and fix a real typo
- collapse nested conditionals into let-chains
- *(sandbox)* depend on heel from crates.io instead of a sibling checkout
- *(sandbox)* [**breaking**] follow heel's IpcCommand onto per-instance naming
- Merge production-readiness work into dev
- *(sandbox)* enable tokio macros feature
- checkpoint current workspace changes
- Finish tool result and terminal refactors
- Improve terminal stdin wait detection
- Drop unused container IPC socket field
- Refine MCP tool service loop
- Simplify aither string assembly
- Cover skill allowlist enforcement
- Emit permission pause lifecycle from bash
- Normalize aither runtime formatting
- Unify container IPC argument decoding
- Serialize ask context as XML
- Remove direct command format strings
- Template command CLI error surfaces
- Type container shell runtime for IPC
- Make IPC CLI argument flattening deterministic
- Template container IPC wrapper generation
- Support sandboxed bash execution mode
- Remove sandbox permission compatibility alias
- Make IPC tool responses typed and fail-fast
- Infer default SSH targets for bash runtimes
- Tighten native terminal tools for bash agents
- Introduce Context and container/terminal support
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
