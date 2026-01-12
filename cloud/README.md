# aither-cloud

Unified cloud provider wrapping OpenAI, Claude, and Gemini.

## Usage

```rust
use aither_cloud::{CloudProvider, OpenAI, Claude, Gemini};

// Create from individual providers
let openai = OpenAI::new(api_key).with_model("gpt-4o");
let cloud: CloudProvider = openai.into();

// Or directly
let cloud = CloudProvider::OpenAI(OpenAI::new(api_key).with_model("gpt-4o"));
let cloud = CloudProvider::Claude(Claude::new(api_key).with_model("claude-sonnet-4-20250514"));
let cloud = CloudProvider::Gemini(Gemini::new(api_key).with_text_model("gemini-2.5-flash"));
```

## Re-exports

This crate re-exports the provider types for convenience:
- `OpenAI` from `aither-openai`
- `Claude` from `aither-claude`
- `Gemini` from `aither-gemini`
