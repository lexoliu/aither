# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/mcp-v0.2.0...mcp-v0.3.0) - 2026-09-11

### Added

- *(acp)* add the client side of the Agent Client Protocol
- provider-native tools, sandbox network audit, and handoff file index
- *(openai)* add Responses API streaming and subagent feedback
- *(mcp)* add MCP client and server support
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(mcp)* build the client feature without http
- *(mcp)* drive stdio over blocking threads so Windows builds

### Other

- Merge pull request #18 from lexoliu/fix/mcp-client-feature
- drop the dependencies nothing uses, and fix a real typo
- collapse nested conditionals into let-chains
- make the workspace clippy-clean under -D warnings
- *(sandbox)* [**breaking**] follow heel's IpcCommand onto per-instance naming
- Merge production-readiness work into dev
- Finish tool result and terminal refactors
- Refine MCP tool service loop
- Enhance RAG and sandbox with semantic processing
- Refactor and improve code quality across multiple modules
- Merge fix-code-smells
- Integrate new features
- Add sandboxed bash tool and bash-first agent
- *(mcp)* add Context7 example and fix HTTP transport
- format code with cargo fmt
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
