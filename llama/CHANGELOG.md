# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.22.0](https://github.com/May-Agent/aither/compare/llama-v0.21.0...llama-v0.22.0) - 2026-11-07

### Added

- change a running agent's model, effort and grants in place

## [0.18.0](https://github.com/May-Agent/aither/compare/llama-v0.17.0...llama-v0.18.0) - 2026-11-05

### Fixed

- *(llama)* create contexts at the size the profile reports

## [0.13.0](https://github.com/May-Agent/aither/compare/llama-v0.12.0...llama-v0.13.0) - 2026-11-03

### Added

- *(core)* [**breaking**] classify provider failures and carry Retry-After

### Other

- Merge pull request #350 from May-Agent/chore/merge-main-into-dev-0.12

## [0.12.0](https://github.com/May-Agent/aither/compare/llama-v0.11.0...llama-v0.12.0) - 2026-11-03

### Other

- refresh README with agent, sandbox, MCP/ACP, and provider crates

## [0.11.0](https://github.com/lexoliu/aither/compare/llama-v0.10.1...llama-v0.11.0) - 2026-10-29

### Fixed

- [**breaking**] fail instead of guessing an unknown model context window

## [0.6.0](https://github.com/lexoliu/aither/compare/llama-v0.5.0...llama-v0.6.0) - 2026-10-10

### Added

- add Apple Intelligence provider
- [**breaking**] tools report progress during a call; aither-mcp forwards it as notifications/progress ([#79](https://github.com/lexoliu/aither/pull/79))

## [0.3.0](https://github.com/lexoliu/aither/compare/llama-v0.2.0...llama-v0.3.0) - 2026-09-11

### Added

- provider-native tools, sandbox network audit, and handoff file index
- add local mistral and llama providers
- enhance provider integrations and add convenience features in Cargo.toml files
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(llama)* restore the thinking toggle and model-side tool-call parsing

### Other

- drop the dependencies nothing uses, and fix a real typo
- Merge production-readiness work into dev
- Load llama model listings asynchronously
- Introduce Context and container/terminal support
- Enhance RAG and sandbox with semantic processing
- Refactor and improve code quality across multiple modules
- remove llama crate and update release config
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
