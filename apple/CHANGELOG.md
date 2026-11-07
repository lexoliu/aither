# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.22.0](https://github.com/May-Agent/aither/compare/apple-v0.21.0...apple-v0.22.0) - 2026-11-07

### Added

- change a running agent's model, effort and grants in place

## [0.13.0](https://github.com/May-Agent/aither/compare/apple-v0.12.0...apple-v0.13.0) - 2026-11-03

### Added

- *(core)* [**breaking**] classify provider failures and carry Retry-After

### Other

- Merge pull request #350 from May-Agent/chore/merge-main-into-dev-0.12

## [0.12.0](https://github.com/May-Agent/aither/compare/apple-v0.11.0...apple-v0.12.0) - 2026-11-03

### Other

- drop x86_64-apple-ios support in aither-apple
- drop macos-x86_64 artifacts and float the arm64 lane on latest

## [0.11.0](https://github.com/lexoliu/aither/compare/apple-v0.10.1...apple-v0.11.0) - 2026-10-29

### Fixed

- [**breaking**] text returned to the model is natural language, never serialized
- [**breaking**] fail instead of guessing an unknown model context window

### Other

- Emit events for provider-hosted tool calls ([#203](https://github.com/lexoliu/aither/pull/203))

## [0.10.0](https://github.com/lexoliu/aither/compare/apple-v0.9.2...apple-v0.10.0) - 2026-10-18

### Other

- Upgrade dependencies to latest and trim feature sets ([#166](https://github.com/lexoliu/aither/pull/166))

## [0.6.0](https://github.com/lexoliu/aither/compare/apple-v0.0.0...apple-v0.6.0) - 2026-10-10

### Added

- add Apple Intelligence provider
