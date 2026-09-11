# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/mem0-v0.2.0...mem0-v0.3.0) - 2026-09-11

### Added

- provider-native tools, sandbox network audit, and handoff file index
- Implement memory extraction and management tools
- enhance provider integrations and add convenience features in Cargo.toml files
- add aither-llama integration with initial implementation and dependencies
- add Mem0 extraction/update demo and core library implementation for long-term memory management
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(openai)* improve Gemini compatibility and tool calling
- *(mem0)* wrap embedder in Mutex for interior mutability

### Other

- drop the dependencies nothing uses, and fix a real typo
- collapse nested conditionals into let-chains
- Merge production-readiness work into dev
- Finish tool result and terminal refactors
- Actorize mem0 runtime
- Collapse mem0 runtime locks
- Stop blocking inside mem0 async paths
- Refactor and improve code quality across multiple modules
- Derive tool descriptions from args rustdoc comments
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
