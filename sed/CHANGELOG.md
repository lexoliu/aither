# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.23.0](https://github.com/May-Agent/aither/compare/sed-v0.22.0...sed-v0.23.0) - 2026-11-07

### Added

- *(sandbox)* [**breaking**] command refusals become embedder policies behind CommandPolicy

### Other

- Merge pull request #615 from May-Agent/feat/598-command-policy

## [0.22.0](https://github.com/May-Agent/aither/compare/sed-v0.21.0...sed-v0.22.0) - 2026-11-07

### Other

- release v0.21.0

## [0.21.0](https://github.com/May-Agent/aither/compare/sed-v0.20.0...sed-v0.21.0) - 2026-11-06

### Other

- update Cargo.lock dependencies

## [0.14.0](https://github.com/May-Agent/aither/compare/sed-v0.0.0...sed-v0.14.0) - 2026-11-03

### Added

- *(sed)* aither-sed prints line ranges for sed in agent scripts
- add Apple Intelligence provider
- [**breaking**] tools report progress during a call; aither-mcp forwards it as notifications/progress ([#79](https://github.com/May-Agent/aither/pull/79))
- implement the ACP server and remove the last placeholders
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(sed)* match GNU sed on unterminated lines and build artifacts on Windows

### Other

- *(sed)* a refused script may exit before its input is written
- *(sed)* run the unit tests only on Unix, where aither-sed ships
- refresh README with agent, sandbox, MCP/ACP, and provider crates
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
