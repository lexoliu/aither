# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/aither/compare/rag-v0.2.0...rag-v0.3.0) - 2026-09-11

### Added

- provider-native tools, sandbox network audit, and handoff file index
- *(rag)* redesign crate with modular production-ready architecture
- enhance provider integrations and add convenience features in Cargo.toml files
- enhance aither-rag with new indexing capabilities, async processing, and improved README examples
- add initial implementation of aither-rag with README, Cargo.toml, and example

### Fixed

- *(openai)* improve Gemini compatibility and tool calling

### Other

- Merge production-readiness work into dev
- Finish tool result and terminal refactors
- Run ORT inference on a worker thread
- Store HNSW index state as snapshots
- Enhance RAG and sandbox with semantic processing
- Refactor and improve code quality across multiple modules
- Derive tool descriptions from args rustdoc comments
- Add sandboxed bash tool and bash-first agent
- format code with cargo fmt
- *(rag)* update to use &mut self for embedding operations
- Update OpenAI Responses integration
