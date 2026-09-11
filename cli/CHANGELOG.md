# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/cli-v0.2.0...cli-v0.3.0) - 2026-09-11

### Added

- provider-native tools, sandbox network audit, and handoff file index
- *(openai)* enable parallel tool calls
- *(openai)* add Responses API streaming and subagent feedback
- *(cli)* add interactive CLI for testing agents
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(mcp)* drive stdio over blocking threads so Windows builds
- *(openai)* improve Gemini compatibility and tool calling

### Other

- collapse nested conditionals into let-chains
- make the workspace clippy-clean under -D warnings
- *(sandbox)* [**breaking**] follow heel's IpcCommand onto per-instance naming
- Merge production-readiness work into dev
- Finish tool result and terminal refactors
- Store CLI domain approvals as snapshots
- Simplify CLI domain approval state
- Make IPC tool responses typed and fail-fast
- Rename Task tool to SubagentTool
- Enhance RAG and sandbox with semantic processing
- Refactor and improve code quality across multiple modules
- Add shell session management, multi-question ask_user, and transcript
- Integrate new features
- Derive tool descriptions from args rustdoc comments
- Add ask command and YAML frontmatter for subagents
- Add sandboxed bash tool and bash-first agent
- Add WebFetch tool and improve tool handling
- Add model registry and tiered model groups
- Add permission prompts and filesystem delete support
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
