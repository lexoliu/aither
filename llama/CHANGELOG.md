# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
