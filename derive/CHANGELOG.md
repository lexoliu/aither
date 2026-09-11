# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/derive-v0.1.0...derive-v0.3.0) - 2026-09-11

### Added

- Implement agent framework with planning and execution capabilities, add sub-agent and todo list management

### Fixed

- repair silently-broken tool APIs and give structured output a typed error
- *(openai)* improve Gemini compatibility and tool calling

### Other

- Merge production-readiness work into dev
- make the workspace buildable from a clean clone
- Derive tool descriptions from args rustdoc comments
- Add sandboxed bash tool and bash-first agent
