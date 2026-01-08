//! # aither-derive
//!
//! Procedural macros for converting Rust functions into AI tools that can be called by language models.
//!
//! This crate provides the `#[tool]` attribute macro that automatically generates the necessary
//! boilerplate code to make your async functions callable by AI models through the `aither` framework.
//!
//! ## Quick Start
//!
//! Transform any async function into an AI tool by adding the `#[tool]` attribute:
//!
//! ```rust
//! use aither::Result;
//! use aither_derive::tool;
//!
//! #[tool(description = "Get the current UTC time")]
//! pub async fn get_time() -> Result<&'static str> {
//!     Ok("2023-10-01T12:00:00Z")
//! }
//! ```
//!
//! ## Function Patterns
//!
//! ### No Parameters
//!
//! ```rust
//! #[tool(description = "Check service health status")]
//! pub async fn health_check() -> Result<String> {
//!     Ok("Service is healthy".to_string())
//! }
//! ```
//!
//! ### Simple Parameters
//!
//! ```rust
//! use serde::Serialize;
//!
//! #[derive(Debug, Serialize)]
//! pub struct SearchResult {
//!     title: String,
//!     url: String,
//! }
//!
//! #[tool(description = "Search the web for content")]
//! pub async fn search(keywords: Vec<String>, limit: u32) -> Result<Vec<SearchResult>> {
//!     // Your search implementation here
//!     Ok(vec![])
//! }
//! ```
//!
//! ### Complex Parameters with Documentation
//!
//! ```rust
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Debug, JsonSchema, Deserialize)]
//! pub struct ImageArgs {
//!     /// The text prompt for image generation
//!     pub prompt: String,
//!     /// Image width in pixels
//!     #[serde(default = "default_width")]
//!     pub width: u32,
//!     /// Image height in pixels  
//!     #[serde(default = "default_height")]
//!     pub height: u32,
//! }
//!
//! fn default_width() -> u32 { 512 }
//! fn default_height() -> u32 { 512 }
//!
//! #[tool(description = "Generate an image from a text prompt")]
//! pub async fn generate_image(args: ImageArgs) -> Result<String> {
//!     // Your image generation logic here
//!     Ok(format!("Generated image: {}", args.prompt))
//! }
//! ```
//!
//! ## Requirements
//!
//! - Functions must be `async`
//! - Return type must be `Result<T>` where `T: serde::Serialize`
//! - Parameters must implement `serde::Deserialize` and `schemars::JsonSchema`
//! - No `self` parameters (static functions only)
//! - No lifetime or generic parameters

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, Ident, ItemFn, LitStr, Token, Type, Visibility,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
};

/// Arguments for the `#[tool]` attribute macro
struct ToolArgs {
    description: String,
    rename: Option<String>,
}

impl Parse for ToolArgs {
    /// Parse the arguments from the `#[tool(...)]` attribute.
    ///
    /// Supports:
    /// - `description = "..."` (required): Tool description for the AI model
    /// - `rename = "..."` (optional): Custom name for the tool (defaults to function name)
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut description = None;
        let mut rename = None;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let value: LitStr = input.parse()?;

            match ident.to_string().as_str() {
                "description" => description = Some(value.value()),
                "rename" => rename = Some(value.value()),
                _ => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        "unknown attribute. Supported: description, rename",
                    ));
                }
            }

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        let description = description
            .ok_or_else(|| syn::Error::new(input.span(), "description attribute is required"))?;

        Ok(Self {
            description,
            rename,
        })
    }
}

/// Converts an async function into an AI tool that can be called by language models.
///
/// This procedural macro generates the necessary boilerplate code to make your function
/// callable through the `aither::llm::Tool` trait.
///
/// # Arguments
///
/// - `description` (required): A clear description of what the tool does. This helps the AI model
///   decide when to use this tool.
/// - `rename` (optional): A custom name for the tool. If not provided, uses the function name.
///
/// # Examples
///
/// ## Basic Usage (No Parameters)
///
/// ```rust
/// use aither::Result;
/// use aither_derive::tool;
///
/// #[tool(description = "Get the current system time")]
/// pub async fn current_time() -> Result<String> {
///     Ok(chrono::Utc::now().to_rfc3339())
/// }
/// ```
///
/// ## With Simple Parameters
///
/// ```rust
/// #[tool(description = "Calculate the sum of two numbers")]
/// pub async fn add(a: f64, b: f64) -> Result<f64> {
///     Ok(a + b)
/// }
/// ```
///
/// ## With Complex Parameters
///
/// ```rust
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(JsonSchema, Deserialize)]
/// pub struct EmailRequest {
///     /// Recipient email address
///     pub to: String,
///     /// Email subject line
///     pub subject: String,
///     /// Email body content
///     pub body: String,
/// }
///
/// #[tool(description = "Send an email to a recipient")]
/// pub async fn send_email(request: EmailRequest) -> Result<String> {
///     // Your email sending logic here
///     Ok(format!("Email sent to {}", request.to))
/// }
/// ```
///
/// ## With Custom Name
///
/// ```rust
/// #[tool(
///     description = "Perform complex mathematical calculations",
///     rename = "calculator"
/// )]
/// pub async fn complex_math_function(expression: String) -> Result<f64> {
///     // Your calculation logic here
///     Ok(42.0)
/// }
/// ```
///
/// # Generated Code
///
/// For a function named `search`, the macro generates:
///
/// 1. A `SearchArgs` struct (if the function has multiple parameters)
/// 2. A `Search` struct that implements `aither::llm::Tool`
/// 3. All necessary trait implementations for JSON schema generation and deserialization
///
/// # Requirements
///
/// - Function must be `async`
/// - Return type must be `Result<T>` where `T` implements `serde::Serialize`
/// - Parameters must implement `serde::Deserialize` and `schemars::JsonSchema`
/// - No `self` parameters (only free functions are supported)
/// - No lifetime parameters or generics
///
/// # Errors
///
/// The macro will produce compile-time errors if:
/// - The function is not async
/// - The function has `self` parameters
/// - The function has more than the supported number of parameters
/// - Required attributes are missing
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as ToolArgs);
    let input_fn = parse_macro_input!(input as ItemFn);

    match tool_impl(args, input_fn) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Implementation details for the `#[tool]` macro.
///
/// This function performs the actual code generation, transforming the annotated async function
/// into a struct that implements the `Tool` trait.
fn tool_impl(args: ToolArgs, input_fn: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let fn_name = &input_fn.sig.ident;
    let tool_name = args.rename.unwrap_or_else(|| fn_name.to_string());
    let description = args.description;
    let fn_vis = &input_fn.vis;

    let tool_struct_name = format_ident!("{}", fn_name.to_string().to_case(Case::Pascal));

    // Analyze function signature
    let AnalyzedArgs {
        args_type,
        params,
        stream,
    } = analyze_function_args(fn_vis, &tool_struct_name, &input_fn.sig.inputs)?;

    if input_fn.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            input_fn.sig,
            "Tool functions must be async",
        ));
    }

    let call_expr = if params.is_empty() {
        // No parameters, call the function directly
        quote! { #fn_name().await }
    } else {
        // Call the function with extracted parameters
        let args_tuple = quote! { #(#params),* };
        quote! { #fn_name(#args_tuple).await }
    };

    let extractor = if params.len() <= 1 {
        quote! {}
    } else {
        quote! { let Self::Arguments { #(#params),* } = args; }
    };

    let expanded = quote! {
        #input_fn

        #stream


        #[derive(::core::default::Default,::core::fmt::Debug)]
        #fn_vis struct #tool_struct_name;

        impl ::aither::llm::Tool for #tool_struct_name {
            fn name(&self) -> ::aither::__hidden::CowStr {
                #tool_name.into()
            }
            fn description(&self) -> ::aither::__hidden::CowStr {
                #description.into()
            }
            type Arguments = #args_type;

            async fn call(&self, args: Self::Arguments) -> ::aither::Result<::aither::llm::ToolOutput> {
                #extractor
                let result: ::aither::Result<_> = #call_expr;
                result.map(|value| {
                    // Convert the result to a JSON ToolOutput
                    ::aither::llm::ToolOutput::text(::aither::llm::tool::json(&value))
                })
            }
        }
    };

    Ok(expanded)
}

/// Container for analyzed function arguments and generated types.
struct AnalyzedArgs {
    /// The type used for the Tool's Arguments associated type
    args_type: Type,
    /// Parameter names extracted from the function signature  
    params: Vec<Ident>,
    /// Generated argument struct definition (if needed)
    stream: proc_macro2::TokenStream,
}

/// Analyzes function parameters and generates appropriate argument types.
///
/// This function handles three cases:
/// - No parameters: Uses unit type `()`
/// - Single parameter: Uses the parameter type directly
/// - Multiple parameters: Generates a new struct with all parameters as fields
fn analyze_function_args(
    fn_vis: &Visibility,
    struct_name: &Ident,
    inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
) -> syn::Result<AnalyzedArgs> {
    match inputs.len() {
        0 => {
            // No arguments - use unit type

            Ok(AnalyzedArgs {
                args_type: parse_quote! { () },
                params: vec![],
                stream: quote! {},
            })
        }
        1 => {
            // Single argument
            if let FnArg::Typed(pat_type) = &inputs[0] {
                Ok(AnalyzedArgs {
                    args_type: (*pat_type.ty).clone(),
                    params: vec![format_ident!("args")],
                    stream: quote! {},
                })
            } else {
                Err(syn::Error::new_spanned(
                    &inputs[0],
                    "self parameters are not supported in tool functions",
                ))
            }
        }
        _ => {
            let mut attributes = Vec::new();

            for arg in inputs {
                if let FnArg::Typed(pat_type) = arg {
                    let pat = &pat_type.pat;
                    let ty = &pat_type.ty;
                    attributes.push(quote! {
                        #pat: #ty,
                    });
                } else {
                    return Err(syn::Error::new_spanned(
                        arg,
                        "self parameters are not supported in tool functions",
                    ));
                }
            }

            let arg_struct_name = format_ident!("{}Args", struct_name);

            let new_type_gen = quote! {
                #[derive(::schemars::JsonSchema, ::serde::Deserialize,::core::fmt::Debug)]
                #fn_vis struct #arg_struct_name {
                    #(
                        #attributes
                    )*
                }
            };

            Ok(AnalyzedArgs {
                args_type: parse_quote! { #arg_struct_name },
                params: inputs
                    .iter()
                    .map(|arg| {
                        if let FnArg::Typed(pat_type) = arg {
                            let pat = &pat_type.pat;
                            format_ident!("{}", quote! {#pat}.to_string())
                        } else {
                            panic!("Expected typed argument")
                        }
                    })
                    .collect(),
                stream: new_type_gen,
            })
        }
    }
}
