# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
