# aither-apple

On-device Apple Intelligence through aither's `LanguageModel` and
`LanguageModelProvider` traits. Every request owns an independent native
FoundationModels session; no API key, upload, helper process, or Private Cloud
Compute is used.

## Build

Xcode 27 is required: its public `SystemLanguageModel.contextSize` query
back-deploys to OS 26. Runtime deployment supports macOS and iOS/iPadOS 26+.
Set the deployment environment for **every final executable**, including
downstream applications:

```sh
export MACOSX_DEPLOYMENT_TARGET=26.0
export IPHONEOS_DEPLOYMENT_TARGET=26.0
cargo run -p aither-apple --example availability
```

The build script honors `DEVELOPER_DIR` and `SDKROOT`, validates the exact SDK
architecture/platform interface, and fails on missing SDKs or deployment settings.
Swift is compiled into a static archive. The system Swift runtime is linked
through SDK stubs; no machine-specific rpath or `DYLD_LIBRARY_PATH` is needed.

Use `aither-apple` directly or enable the facade's `apple` feature and import
`aither::apple`. On other operating systems availability reports
`UnsupportedOs`; there is no native `LanguageModel` implementation.

## Requests

- `AppleIntelligence::new()` checks real native availability. Discovery returns
  an error when the model cannot run; the logical model ID is `apple-on-device`.
- Text streams deliver append-only Unicode deltas. A native revision of already
  emitted text is an explicit error. Structured output delivers one final valid
  JSON document, never concatenated partial snapshots.
- `generate<T>` and `Parameters.response_format` use native guided generation.
  Required nullable fields remain required and need OS 26.4+. Unsupported JSON
  Schema constraints and conjunctions fail with a schema path.
- `respond` with tool definitions returns a complete native batch and its IDs
  without running Rust tools. Supply assistant calls and matching tool outputs
  in the next request to continue.
- `respond_with_tools` uses native Tool dispatch and borrowed Rust tools,
  resolving each invocation exactly once. Its observed results use
  `BuiltInToolResult`, so consumers do not execute tools twice.
- Temperature, maximum response tokens, top-k/top-p, and seeds map to native
  options. Explicit unsupported controls fail. Required/exact tool selection
  needs OS 27; parallelism overrides are unsupported.
- Image prompting needs OS 27 and the model's vision capability. Attachments
  resolve concurrently (bound 4) from file/data/HTTP URLs before native image
  decoding. Capability validation precedes attachment I/O.
- Reasoning and token usage are surfaced only when the native API supplies
  them. Unsupported reasoning replay is rejected.
- Stream drop schedules cancellation; it never waits on a lock or task.
  Actors own lifecycle and tool continuations. The native owner acknowledges
  final callback-context release after emissions finish.

Run `stream`, `guided`, and `tools` examples on an eligible device:

```sh
cargo run -p aither-apple --example stream
cargo run -p aither-apple --example guided
cargo run -p aither-apple --example tools
```

## Correctness tests

```sh
cargo test -p aither-apple
cargo clippy -p aither-apple --all-targets --features test -- -D warnings
# OS 27 only: explicitly run ignored deterministic framework-session tests.
cargo test -p aither-apple --features test -- --include-ignored
```

The opt-in `test` feature builds a deterministic FoundationModels custom
executor. Its OS 27 tests exercise real sessions, schema construction, tool
dispatch, replay, backpressure and cancellation without invoking the system
model. These tests do **not** establish inference success or device eligibility.
The availability example always reports the real system model state.
