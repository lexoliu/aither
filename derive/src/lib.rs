//! # aither-derive
//!
//! Procedural macros for converting Rust functions into AI tools that can be called by language models.
//!
//! This crate provides the `#[tool]` attribute macro that automatically generates the necessary
//! boilerplate code to make your async functions callable by AI models through the `aither` framework.
//!
//! ## Quick Start
//!
//! Transform any async function into an AI tool by adding the `#[tool]` attribute.
//! The tool description is the function's rustdoc, or, when the function has
//! none, the rustdoc on its single Args struct:
//!
//! ```rust
//! use aither::Result;
//! use aither_derive::tool;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// Get the current UTC time.
//! #[derive(JsonSchema, Deserialize)]
//! pub struct GetTimeArgs;
//!
//! #[tool]
//! pub async fn get_time(_args: GetTimeArgs) -> Result<&'static str> {
//!     Ok("2023-10-01T12:00:00Z")
//! }
//! ```
//!
//! ## Function Patterns
//!
//! ### Simple Parameters
//!
//! ```rust
//! use serde::Serialize;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Debug, Serialize)]
//! pub struct SearchResult {
//!     title: String,
//!     url: String,
//! }
//!
//! /// Search the web for content.
//! #[derive(JsonSchema, Deserialize)]
//! pub struct SearchArgs {
//!     pub keywords: Vec<String>,
//!     pub limit: u32,
//! }
//!
//! #[tool]
//! pub async fn search(args: SearchArgs) -> Result<Vec<SearchResult>> {
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
//! /// Generate an image from a text prompt.
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
//! #[tool]
//! pub async fn generate_image(args: ImageArgs) -> Result<String> {
//!     Ok(format!("Generated image: {}", args.prompt))
//! }
//! ```
//!
//! ### Reporting Progress
//!
//! A parameter of type `ToolContext` receives the call's context instead of a
//! model-supplied argument; it is left out of the argument schema. Report
//! progress through it — a no-op when the caller does not listen.
//!
//! ```rust
//! use aither::Result;
//! use aither::llm::ToolContext;
//! use aither::llm::tool::Progress;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// Index the given documents.
//! #[derive(JsonSchema, Deserialize)]
//! pub struct IndexArgs {
//!     pub documents: Vec<String>,
//! }
//!
//! #[tool]
//! pub async fn index(args: IndexArgs, mut cx: ToolContext) -> Result<usize> {
//!     for (done, _document) in args.documents.iter().enumerate() {
//!         cx.report_progress(Progress::new(done as f64 + 1.0)).await?;
//!     }
//!     Ok(args.documents.len())
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
    rename: Option<String>,
}

impl Parse for ToolArgs {
    /// Parse the arguments from the `#[tool(...)]` attribute.
    ///
    /// Supports:
    /// - `rename = "..."` (optional): Custom name for the tool (defaults to function name)
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut rename = None;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            let value: LitStr = input.parse()?;

            match ident.to_string().as_str() {
                "rename" => rename = Some(value.value()),
                _ => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        "unknown attribute. Supported: rename",
                    ));
                }
            }

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(Self { rename })
    }
}

/// Converts an async function into an AI tool that can be called by language models.
///
/// This procedural macro generates the necessary boilerplate code to make your function
/// callable through the `aither::llm::Tool` trait.
///
/// The tool description is the function's rustdoc. A function without one
/// falls back to the rustdoc on its single Args struct, which
/// `schemars::JsonSchema` records.
///
/// # Arguments
///
/// - `rename` (optional): A custom name for the tool. If not provided, uses the function name.
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use aither::Result;
/// use aither_derive::tool;
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// /// Get the current system time.
/// #[derive(JsonSchema, Deserialize)]
/// pub struct CurrentTimeArgs;
///
/// #[tool]
/// pub async fn current_time(_args: CurrentTimeArgs) -> Result<String> {
///     Ok(chrono::Utc::now().to_rfc3339())
/// }
/// ```
///
/// ## With Parameters
///
/// ```rust
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// /// Send an email to a recipient.
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
/// #[tool]
/// pub async fn send_email(request: EmailRequest) -> Result<String> {
///     Ok(format!("Email sent to {}", request.to))
/// }
/// ```
///
/// ## With Custom Name
///
/// ```rust
/// /// Perform complex mathematical calculations.
/// #[derive(JsonSchema, Deserialize)]
/// pub struct CalcArgs {
///     pub expression: String,
/// }
///
/// #[tool(rename = "calculator")]
/// pub async fn complex_math_function(args: CalcArgs) -> Result<f64> {
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
    let fn_vis = &input_fn.vis;

    let tool_struct_name = format_ident!("{}", fn_name.to_string().to_case(Case::Pascal));

    // The function's rustdoc describes the tool. Without it the description
    // falls back to the rustdoc on the arguments type, which only a
    // single-parameter tool has.
    let description = doc_text(&input_fn.attrs).map(|text| {
        quote! {
            fn description(&self) -> ::aither::__hidden::CowStr {
                #text.into()
            }
        }
    });

    // The context parameter, if any, is not an argument the model fills in.
    let (context_position, data_inputs) = split_context_parameter(&input_fn.sig.inputs)?;

    // Analyze function signature
    let AnalyzedArgs {
        args_type,
        params,
        stream,
    } = analyze_function_args(fn_vis, &tool_struct_name, &data_inputs)?;

    if input_fn.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            input_fn.sig,
            "Tool functions must be async",
        ));
    }

    // Call the function with its parameters in declaration order, the
    // context in the position the function declared it.
    let mut call_args: Vec<proc_macro2::TokenStream> =
        params.iter().map(|param| quote! { #param }).collect();
    if let Some(position) = context_position {
        call_args.insert(position, quote! { cx });
    }
    let call_expr = quote! { #fn_name(#(#call_args),*).await };

    let extractor = if params.len() <= 1 {
        quote! {}
    } else {
        quote! { let Self::Arguments { #(#params),* } = args; }
    };

    let args_binding = if params.is_empty() {
        quote! { _args }
    } else {
        quote! { args }
    };
    let context_binding = if context_position.is_some() {
        quote! { cx }
    } else {
        quote! { _cx }
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
            #description
            type Arguments = #args_type;
            type Res = ::aither::llm::ToolResult;

            async fn call(
                &self,
                #args_binding: Self::Arguments,
                #context_binding: ::aither::llm::ToolContext,
            ) -> ::aither::Result<Self::Res> {
                #extractor
                ::aither::llm::IntoToolResult::into_tool_result(#call_expr)
            }
        }
    };

    Ok(expanded)
}

/// The text of the `///` comments in `attrs`, one line per attribute with the
/// single leading space rustdoc strips removed, or `None` when there are none.
fn doc_text(attrs: &[syn::Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| match &attr.meta {
            syn::Meta::NameValue(syn::MetaNameValue {
                value:
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(text),
                        ..
                    }),
                ..
            }) => Some(text.value()),
            _ => None,
        })
        .map(|line| {
            line.strip_prefix(' ')
                .map_or_else(|| line.clone(), str::to_owned)
        })
        .collect();
    let text = lines.join("\n");
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// Separates the call's `ToolContext` parameter from the model-facing ones.
///
/// A parameter is the context when its type is a path ending in
/// `ToolContext`; a proc macro sees tokens, not resolved types, so the name is
/// what identifies it. Returns the context's position among all parameters
/// and the remaining parameters. More than one context parameter is an error.
fn split_context_parameter(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
) -> syn::Result<(
    Option<usize>,
    syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
)> {
    let mut position = None;
    let mut data = syn::punctuated::Punctuated::new();
    for (index, input) in inputs.iter().enumerate() {
        if is_context_parameter(input) {
            if position.is_some() {
                return Err(syn::Error::new_spanned(
                    input,
                    "a tool function takes at most one `ToolContext` parameter",
                ));
            }
            position = Some(index);
        } else {
            data.push(input.clone());
        }
    }
    Ok((position, data))
}

fn is_context_parameter(input: &FnArg) -> bool {
    let FnArg::Typed(pat_type) = input else {
        return false;
    };
    let Type::Path(path) = pat_type.ty.as_ref() else {
        return false;
    };
    path.qself.is_none()
        && path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "ToolContext")
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

            let params = inputs
                .iter()
                .map(|arg| match arg {
                    FnArg::Typed(pat_type) => {
                        let pat = &pat_type.pat;
                        Ok(format_ident!("{}", quote! {#pat}.to_string()))
                    }
                    // A `self` receiver: point at it rather than panicking, so
                    // the user gets a diagnostic on the offending argument
                    // instead of "proc macro panicked".
                    FnArg::Receiver(receiver) => Err(syn::Error::new_spanned(
                        receiver,
                        "#[tool] cannot be applied to a method taking `self`; \
                         use a free function",
                    )),
                })
                .collect::<Result<Vec<_>, syn::Error>>()?;

            Ok(AnalyzedArgs {
                args_type: parse_quote! { #arg_struct_name },
                params,
                stream: new_type_gen,
            })
        }
    }
}
