# aither-derive

> **📖 Documentation:** For comprehensive documentation with examples, see the inline code documentation in [`src/lib.rs`](./src/lib.rs) and the example file [`examples/tool_macro.rs`](../examples/tool_macro.rs).

Procedural macros for converting Rust functions into AI tools that can be called by language models.

This crate provides the `#[tool]` attribute macro that automatically generates the necessary boilerplate code to make your async functions callable by AI models through the `aither` framework.

## Quick Start

Transform any async function into an AI tool by adding the `#[tool]` attribute:

```rust
use aither::Result;
use aither_derive::tool;

#[tool(description = "Get the current UTC time")]
pub async fn get_time() -> Result<&'static str> {
    Ok("2023-10-01T12:00:00Z")
}
```

That's it! Your function can now be called by AI models as a tool.


For detailed examples and comprehensive documentation, please refer to:
- **API Documentation**: [`src/lib.rs`](./src/lib.rs) contains extensive inline documentation
- **Usage Examples**: [`examples/tool_macro.rs`](../examples/tool_macro.rs) shows various patterns
