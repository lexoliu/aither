# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/core-v0.2.0...core-v0.3.0) - 2026-09-11

### Added

- *(llm)* [**breaking**] adaptive thinking, wider effort ladder, reasoning-state round-trip
- provider-native tools, sandbox network audit, and handoff file index
- add Event::ToolCallDelta for streaming tool call progress
- apply local workspace changes
- *(openai)* add Responses API streaming and subagent feedback
- *(cli)* add interactive CLI for testing agents
- *(core)* add ToolChoice enum for tool calling policy
- *(mcp)* add MCP client and server support
- *(core)* add Event-based streaming API for LanguageModel
- enhance CI/CD workflows and add skip-check logic for release management
- Implement memory extraction and management tools
- enhance provider integrations and add convenience features in Cargo.toml files
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

- *(openai)* improve Gemini compatibility and tool calling

### Other

- *(core)* [**breaking**] drop top_a, and say which sampling knobs are local-only
- collapse nested conditionals into let-chains
- Merge production-readiness work into dev
- Finish tool result and terminal refactors
- Introduce Context and container/terminal support
- Enhance RAG and sandbox with semantic processing
- Refactor and improve code quality across multiple modules
- Integrate new features
- Derive tool descriptions from args rustdoc comments
- Add sandboxed bash tool and bash-first agent
- Add model registry and tiered model groups
- format code with cargo fmt
- *(core)* change EmbeddingModel::embed to take &mut self
- Update OpenAI Responses integration
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
