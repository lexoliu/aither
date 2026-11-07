# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.23.0](https://github.com/May-Agent/aither/compare/patch-v0.22.0...patch-v0.23.0) - 2026-11-07

### Added

- *(patch)* export the parsed patch and a read-only hunk locate

### Fixed

- *(patch)* size the release's tool entries from its assets when composing the verify manifest
- *(sandbox)* [**breaking**] pin each tool artifact's byte size instead of capping at 64 MiB

### Other

- merge main into dev after v0.22.0
- *(patch)* drop the test-only apply_hunks

## [0.22.0](https://github.com/May-Agent/aither/compare/patch-v0.21.0...patch-v0.22.0) - 2026-11-07

### Other

- release v0.21.0

## [0.21.0](https://github.com/May-Agent/aither/compare/patch-v0.20.0...patch-v0.21.0) - 2026-11-06

### Other

- update Cargo.lock dependencies

## [0.17.0](https://github.com/May-Agent/aither/compare/patch-v0.16.0...patch-v0.17.0) - 2026-11-05

### Fixed

- *(patch)* carry a file's mode through chained moves

## [0.15.0](https://github.com/May-Agent/aither/compare/patch-v0.14.1...patch-v0.15.0) - 2026-11-04

### Other

- verify the baseline tools on macOS and Linux
- Merge pull request #407 from May-Agent/fix/342-wrapper-leftover

## [0.14.0](https://github.com/May-Agent/aither/compare/patch-v0.13.0...patch-v0.14.0) - 2026-11-03

### Added

- *(sed)* aither-sed prints line ranges for sed in agent scripts
- *(sandbox)* managed uv and a pinned python3
- *(sandbox)* build bash 5.3 in CI and run terminal scripts through it
- *(sandbox)* managed uutils coreutils, findutils xargs and fd

### Fixed

- *(sed)* match GNU sed on unterminated lines and build artifacts on Windows

### Other

- Merge origin/dev into feat/387-managed-bash

## [0.12.0](https://github.com/May-Agent/aither/compare/patch-v0.11.0...patch-v0.12.0) - 2026-11-03

### Added

- *(sandbox)* [**breaking**] prepare a cached host program without re-extracting its archive

### Other

- point repository links at May-Agent/aither
- drop macos-x86_64 artifacts and float the arm64 lane on latest
- refresh README with agent, sandbox, MCP/ACP, and provider crates

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
