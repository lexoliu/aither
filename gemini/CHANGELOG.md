# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/gemini-v0.2.0...gemini-v0.3.0) - 2026-09-11

### Added

- *(llm)* [**breaking**] adaptive thinking, wider effort ladder, reasoning-state round-trip
- provider-native tools, sandbox network audit, and handoff file index
- apply local workspace changes
- add support for file attachments in OpenAI API
- *(core)* add ToolChoice enum for tool calling policy
- enhance CI/CD workflows and add skip-check logic for release management
- Implement memory extraction and management tools
- update error handling to use BoxHttpError across multiple modules
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- enhance agent framework with memory management, planning, and execution capabilities
- implement conversation memory management with compression strategies
- Add Gemini backend integration with audio, image, and moderation capabilities
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(openai)* improve Gemini compatibility and tool calling

### Other

- collapse nested conditionals into let-chains
- Merge production-readiness work into dev
- Read Gemini attachments asynchronously
- Introduce Context and container/terminal support
- Enhance RAG and sandbox with semantic processing
- Add wasm32 compatibility to aither-gemini
- Refactor and improve code quality across multiple modules
- Add shell session management, multi-question ask_user, and transcript
- Integrate new features
- Derive tool descriptions from args rustdoc comments
- Add sandboxed bash tool and bash-first agent
- Add WebFetch tool and improve tool handling
- Add model registry and tiered model groups
- format code with cargo fmt
- *(providers)* update to Event-based streaming API
- *(core)* change EmbeddingModel::embed to take &mut self
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
