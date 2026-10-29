# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0](https://github.com/lexoliu/aither/compare/patch-v0.0.0...patch-v0.12.0) - 2026-10-29

### Added

- *(sandbox)* distribute host programs, starting with the repo-search daemon
- add Apple Intelligence provider
- [**breaking**] tools report progress during a call; aither-mcp forwards it as notifications/progress ([#79](https://github.com/lexoliu/aither/pull/79))
- implement the ACP server and remove the last placeholders
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(patch)* publish aither-patch to crates.io
- *(patch)* own hunk matcher; multi-change hunks apply

### Other

- Merge pull request #262 from lexoliu/chore/merge-main-into-dev-0.11
- publish apply_patch executables with each aither-patch release
- Validate apply_patch against the package version
- Distinguish native Windows launch failures from artifact permission errors
- Package artifacts without treating Windows drives as remote archives
- Validate native Windows patch artifacts alongside Linux and macOS
- Gate portable patch artifacts on executable and sandbox tests
- Verify portable apply_patch artifacts with the executable test suite
- Add sandbox-executed apply_patch binary
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

## [0.11.0](https://github.com/lexoliu/aither/compare/patch-v0.10.1...patch-v0.11.0) - 2026-10-29

### Added

- add Apple Intelligence provider
- [**breaking**] tools report progress during a call; aither-mcp forwards it as notifications/progress ([#79](https://github.com/lexoliu/aither/pull/79))
- implement the ACP server and remove the last placeholders
- add aither-llama integration with initial implementation and dependencies
- implement Coder and DeepResearchAgent for enhanced coding workflows and research capabilities
- add thinking configuration and reasoning capabilities to moderation and content generation
- update CI workflows, add tests for multiple OS, and remove deprecated release-plz workflow
- Add OpenAI integration with support for various models and functionalities
- add ai-types-derive crate for procedural macros

### Fixed

- *(patch)* own hunk matcher; multi-change hunks apply

### Other

- publish apply_patch executables with each aither-patch release
- Validate apply_patch against the package version
- Distinguish native Windows launch failures from artifact permission errors
- Package artifacts without treating Windows drives as remote archives
- Validate native Windows patch artifacts alongside Linux and macOS
- Gate portable patch artifacts on executable and sandbox tests
- Verify portable apply_patch artifacts with the executable test suite
- Add sandbox-executed apply_patch binary
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
