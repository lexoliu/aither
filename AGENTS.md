# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the core traits (`llm`, `embedding`, `audio`, `image`, `moderation`) plus `lib.rs` for re-exports; keep shared types here.
- `derive/` is the `aither-derive` proc-macro crate—limit it to macro glue and avoid depending on provider code.
- `openai/` contains the integration crate that maps the traits onto OpenAI endpoints; keep it lean so other providers can re-use the core API.
- `examples/` (notably `tool_macro.rs`) provides runnable flows; extend it and validate via `cargo run --example tool_macro`.

## Build, Test & Development Commands
- `cargo fmt --all` — formats every crate exactly as CI expects.
- `cargo clippy --all-targets --all-features --workspace -- -D warnings` — pedantic lint gate mirrored in `.github/workflows/ci.yml`.
- `cargo test --all-features --workspace` — runs unit and doc tests, including async cases backed by `tokio-test`.
- `cargo doc --all-features --no-deps --workspace` — ensures docs compile under the `docsrs` cfg.
- `cargo run --example tool_macro` — smoke-tests the showcased example when changing macros or request builders.

## Coding Style & Naming Conventions
Follow `rustfmt` defaults (4 spaces, trailing commas) and keep modules `snake_case`, traits/types `UpperCamelCase`, and feature flags `kebab-case` (`serde`, `derive`). Public APIs need `///` docs because `cargo doc --cfg docsrs` runs in CI. Favor builders such as `Request::new([...])`, keep provider-only structs behind feature flags, and align with `clippy::pedantic` expectations before raising a PR.

## Testing Guidelines
Co-locate unit tests inside each module using `#[cfg(test)] mod tests` so traits stay covered near their definitions. Async logic should use `#[tokio::test]` (multi-thread flavor when polling streams) and exercise `futures_lite::StreamExt` when verifying streaming traits. Every new capability needs at least one success path, one failure path, and—when user facing—a doc test or example that mirrors realistic prompts.

## Commit & Pull Request Guidelines
History favors short, imperative subjects (see `Add serde support`, `Rename ai-types to aither`); follow that pattern and explain breaking changes in the body. Pull requests should provide a concise description, link to the motivating issue, paste the results of `cargo fmt`, `clippy`, and `test`, and call out any docs/examples touched. Attach logs or transcripts when altering provider behavior so reviewers can reproduce the call.

## Security & Configuration Tips
Never commit provider secrets; inject them via environment variables such as `OPENAI_API_KEY` when hacking on `aither-openai`. Keep `.env*` files out of Git and strip identifiers from captured responses before committing examples.
