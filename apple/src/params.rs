//! `Parameters` → native `GenerationOptions` mapping and validation.
//!
//! Anything the framework cannot express is rejected with a typed
//! [`AppleError::UnsupportedParameter`] rather than silently dropped.

use aither_core::llm::model::{Parameters, ToolChoice};
use aither_core::llm::tool::ToolDefinition;

use crate::error::AppleError;
use crate::wire::{WireOptions, WireSampling};

/// Runtime capabilities reported by the bridge.
#[derive(Debug, Clone, Copy, Default)]
pub struct Capabilities {
    bits: u32,
}

impl Capabilities {
    pub const VISION: u32 = 1;
    pub const GUIDED: u32 = 1 << 1;
    pub const TOOLS: u32 = 1 << 3;
    pub const REASONING: u32 = 1 << 2;
    /// Whether the OS27-only `toolCallingMode`/`LanguageModel` surfaces exist.
    /// The bridge reports it through bit 4 of the capabilities word.
    pub const OS27_SURFACE: u32 = 1 << 4;

    pub const fn from_bits(bits: u32) -> Self {
        Self { bits }
    }

    pub const fn supports(self, capability: u32) -> bool {
        self.bits & capability != 0
    }
}

fn unsupported(name: &'static str, reason: impl Into<String>) -> AppleError {
    AppleError::UnsupportedParameter {
        name,
        reason: reason.into(),
    }
}

/// Validates `parameters` against the native surface and returns the wire
/// options. `has_tools` is whether the request carries tool definitions,
/// needed to evaluate the tool-choice policy.
pub fn to_options(parameters: &Parameters, caps: Capabilities) -> Result<WireOptions, AppleError> {
    let p = parameters;
    reject_unsupported_controls(p)?;

    if p.top_k.is_some() && p.top_p.is_some() {
        return Err(unsupported(
            "top_k",
            "top_k and top_p are mutually exclusive in native sampling",
        ));
    }

    if p.reasoning_effort.is_some() {
        return Err(unsupported(
            "reasoning_effort",
            "the on-device model exposes no effort control",
        ));
    }

    if p.parallel_tool_calls.is_some() {
        return Err(unsupported(
            "parallel_tool_calls",
            "the framework does not expose a parallelism toggle",
        ));
    }

    if p.websearch {
        return Err(unsupported("websearch", "no native web search tool"));
    }
    if p.code_execution {
        return Err(unsupported(
            "code_execution",
            "no native code execution tool",
        ));
    }
    if !p.native_tools.is_empty() {
        return Err(unsupported(
            "native_tools",
            "provider-native tool configuration of another provider was supplied",
        ));
    }
    if !p.cache.is_empty() {
        return Err(unsupported(
            "cache",
            "prompt caching controls belong to other providers",
        ));
    }

    if p.include_reasoning && !caps.supports(Capabilities::REASONING) {
        return Err(unsupported(
            "include_reasoning",
            "this model/OS does not expose reasoning text",
        ));
    }

    if p.top_k == Some(0) {
        return Err(unsupported("top_k", "must be positive"));
    }
    if p.top_p
        .is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
    {
        return Err(unsupported("top_p", "must be finite and in (0, 1]"));
    }
    if p.temperature.is_some_and(|t| !t.is_finite() || t < 0.0) {
        return Err(unsupported("temperature", "must be finite and nonnegative"));
    }
    let maximum_response_tokens = p.max_tokens;
    let sampling = match (p.top_k, p.top_p, p.seed) {
        (Some(k), _, seed) => Some(WireSampling::TopK {
            k,
            seed: seed.map(u64::from),
        }),
        (_, Some(p), seed) => Some(WireSampling::TopP {
            p: f64::from(p),
            seed: seed.map(u64::from),
        }),
        (_, _, Some(seed)) => Some(WireSampling::TopP {
            p: 1.0,
            seed: Some(u64::from(seed)),
        }),
        _ => None,
    };

    Ok(WireOptions {
        temperature: p.temperature.map(f64::from),
        maximum_response_tokens,
        sampling,
        tool_calling_mode: None,
    })
}

fn reject_unsupported_controls(p: &Parameters) -> Result<(), AppleError> {
    for (value, name, why) in [
        (
            p.frequency_penalty.is_some(),
            "frequency_penalty",
            "no native penalty control",
        ),
        (
            p.presence_penalty.is_some(),
            "presence_penalty",
            "no native penalty control",
        ),
        (
            p.repetition_penalty.is_some(),
            "repetition_penalty",
            "no native penalty control",
        ),
        (p.min_p.is_some(), "min_p", "no native min-p sampler"),
        (p.logit_bias.is_some(), "logit_bias", "no native logit bias"),
        (
            p.logprobs.is_some(),
            "logprobs",
            "framework does not expose log probabilities",
        ),
        (
            p.top_logprobs.is_some(),
            "top_logprobs",
            "framework does not expose log probabilities",
        ),
        (
            p.stop.is_some(),
            "stop",
            "framework has no stop-sequence control",
        ),
    ] {
        if value {
            return Err(unsupported(name, why));
        }
    }

    Ok(())
}

/// What `tool_choice` resolves to for the wire request.
#[derive(Debug)]
pub enum ToolPlan {
    /// No tools are offered to the model.
    NoTools,
    /// Tools are offered; the model may or may not call them.
    Allowed(Vec<ToolDefinition>),
    /// OS27 `toolCallingMode = .required`.
    Required(Vec<ToolDefinition>),
}

/// Resolves `tool_choice` + definitions into a [`ToolPlan`], rejecting what
/// the running OS cannot express.
pub fn tool_plan(
    parameters: &Parameters,
    definitions: Vec<ToolDefinition>,
    caps: Capabilities,
) -> Result<ToolPlan, AppleError> {
    match &parameters.tool_choice {
        ToolChoice::None => Ok(ToolPlan::NoTools),
        ToolChoice::Auto => {
            if definitions.is_empty() {
                Ok(ToolPlan::NoTools)
            } else {
                Ok(ToolPlan::Allowed(definitions))
            }
        }
        ToolChoice::Required => {
            if definitions.is_empty() {
                return Err(unsupported(
                    "tool_choice",
                    "Required was requested without any tool definitions",
                ));
            }
            if !caps.supports(Capabilities::OS27_SURFACE) {
                return Err(unsupported(
                    "tool_choice",
                    "Required needs GenerationOptions.toolCallingMode, which requires \
                     macOS/iOS 27; it cannot be emulated on 26",
                ));
            }
            Ok(ToolPlan::Required(definitions))
        }
        ToolChoice::Exact(name) => {
            let filtered: Vec<_> = definitions
                .into_iter()
                .filter(|d| d.name() == name)
                .collect();
            if filtered.is_empty() {
                return Err(unsupported(
                    "tool_choice",
                    format!("Exact tool '{name}' is not in the request's tool definitions"),
                ));
            }
            if !caps.supports(Capabilities::OS27_SURFACE) {
                return Err(unsupported(
                    "tool_choice",
                    "Exact needs GenerationOptions.toolCallingMode, which requires \
                     macOS/iOS 27; it cannot be emulated on 26",
                ));
            }
            Ok(ToolPlan::Required(filtered))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::WireSampling;
    use aither_core::llm::model::ToolChoice;
    use aither_core::llm::tool::ToolDefinition;
    use core::future::Future;

    const OS27: Capabilities = Capabilities::from_bits(0x1F);
    const OS26: Capabilities = Capabilities::from_bits(0x0B);

    #[test]
    fn top_k_and_top_p_conflict() {
        let params = Parameters {
            top_k: Some(10),
            top_p: Some(0.9),
            ..Parameters::default()
        };
        match to_options(&params, OS27) {
            Err(AppleError::UnsupportedParameter { name, .. }) => {
                assert_eq!(name, "top_k");
            }
            other => panic!("expected UnsupportedParameter, got {other:?}"),
        }
    }

    #[test]
    fn scalar_parameter_bounds_preserve_native_range() {
        let options = to_options(
            &Parameters {
                max_tokens: Some(u32::MAX),
                ..Parameters::default()
            },
            OS27,
        )
        .unwrap();
        assert_eq!(options.maximum_response_tokens, Some(u32::MAX));
        for params in [
            Parameters {
                top_k: Some(0),
                ..Parameters::default()
            },
            Parameters {
                top_p: Some(0.0),
                ..Parameters::default()
            },
            Parameters {
                top_p: Some(1.1),
                ..Parameters::default()
            },
            Parameters {
                top_p: Some(f32::NAN),
                ..Parameters::default()
            },
        ] {
            assert!(to_options(&params, OS27).is_err());
        }
    }

    #[test]
    fn unsupported_parameters_rejected() {
        type ParameterCase = (fn(&mut Parameters), &'static str);
        let cases: [ParameterCase; 4] = [
            (|p| p.stop = Some(vec!["\n".into()]), "stop"),
            (
                |p| p.parallel_tool_calls = Some(true),
                "parallel_tool_calls",
            ),
            (
                |p| p.reasoning_effort = Some(aither_core::llm::model::ReasoningEffort::Low),
                "reasoning_effort",
            ),
            (|p| p.logprobs = Some(true), "logprobs"),
        ];
        for (set, name) in cases {
            let mut params = Parameters::default();
            set(&mut params);
            match to_options(&params, OS27) {
                Err(AppleError::UnsupportedParameter { name: n, .. }) => assert_eq!(n, name),
                other => panic!("{name}: expected UnsupportedParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn sampling_maps_and_seed_defaults() {
        let params = Parameters {
            top_k: Some(4),
            seed: Some(7),
            temperature: Some(0.5),
            max_tokens: Some(128),
            ..Parameters::default()
        };
        let options = to_options(&params, OS27).expect("valid");
        match options.sampling {
            Some(WireSampling::TopK {
                k: 4,
                seed: Some(7),
            }) => {}
            other => panic!("expected top_k sampling, got {other:?}"),
        }
        assert_eq!(options.temperature, Some(0.5));
        assert_eq!(options.maximum_response_tokens, Some(128));

        // Bare seed still produces a seeded sampling mode.
        let options = to_options(
            &Parameters {
                seed: Some(3),
                ..Parameters::default()
            },
            OS27,
        )
        .expect("valid");
        assert!(matches!(
            options.sampling,
            Some(WireSampling::TopP { p, seed: Some(3) }) if p.to_bits() == 1.0_f64.to_bits()
        ));
    }

    #[test]
    fn tool_choice_required_needs_os27() {
        let params = Parameters {
            tool_choice: ToolChoice::Required,
            ..Parameters::default()
        };
        let defs = vec![ToolDefinition::new(&Dummy)];
        match tool_plan(&params, defs.clone(), OS27) {
            Ok(ToolPlan::Required(d)) => assert_eq!(d.len(), 1),
            other => panic!("expected Required, got {other:?}"),
        }
        match tool_plan(&params, defs, OS26) {
            Err(AppleError::UnsupportedParameter { name, .. }) => {
                assert_eq!(name, "tool_choice");
            }
            other => panic!("expected UnsupportedParameter, got {other:?}"),
        }
    }

    #[test]
    fn tool_choice_exact_filters_definitions() {
        let params = Parameters {
            tool_choice: ToolChoice::Exact("wanted".into()),
            ..Parameters::default()
        };
        let defs = vec![ToolDefinition::new(&Dummy), ToolDefinition::new(&Other)];
        match tool_plan(&params, defs, OS27) {
            Ok(ToolPlan::Required(d)) => assert_eq!(d[0].name(), "wanted"),
            other => panic!("expected Required, got {other:?}"),
        }
    }

    /// No-argument tool input.
    #[derive(schemars::JsonSchema, serde::Deserialize)]
    struct Args {}

    struct Dummy;
    struct Other;

    impl aither_core::llm::tool::Tool for Dummy {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "wanted".into()
        }
        type Arguments = Args;
        type Res = aither_core::llm::tool::ToolResult;
        fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::tool::ToolContext,
        ) -> impl Future<Output = aither_core::Result<Self::Res>> + Send {
            core::future::ready(Ok(aither_core::llm::tool::ToolResult::text("ok")))
        }
    }

    impl aither_core::llm::tool::Tool for Other {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "other".into()
        }
        type Arguments = Args;
        type Res = aither_core::llm::tool::ToolResult;
        fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::tool::ToolContext,
        ) -> impl Future<Output = aither_core::Result<Self::Res>> + Send {
            core::future::ready(Ok(aither_core::llm::tool::ToolResult::text("ok")))
        }
    }
}
