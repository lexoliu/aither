# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/acp-v0.2.0...acp-v0.3.0) - 2026-09-11

### Added

- *(acp)* add the client side of the Agent Client Protocol
- provider-native tools, sandbox network audit, and handoff file index
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(acp)* keep the protocol module public
- *(mcp)* drive stdio over blocking threads so Windows builds

### Other

- collapse nested conditionals into let-chains
- *(sandbox)* [**breaking**] follow heel's IpcCommand onto per-instance naming
- Merge production-readiness work into dev
- Finish tool result and terminal refactors
- Simplify CLI domain approval state
- Enhance RAG and sandbox with semantic processing
- Refactor and improve code quality across multiple modules
- Integrate new features
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
