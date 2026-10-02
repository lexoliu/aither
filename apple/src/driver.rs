//! The async driver between the native bridge and the `LanguageModel` stream
//! surface.
//!
//! One [`Driver`] owns one bridge request: it resolves attachments, ships the
//! JSON request, then polls a bounded channel fed by the Swift callback. When
//! the channel is full the bridge suspends rather than dropping events; each
//! successful receive frees a slot and is acknowledged with
//! `request_resume`. The terminal `end` event is always deliverable — the
//! callback force-sends it even when the channel is saturated — so a live
//! consumer always observes termination. Dropping the driver cancels the
//! native task; the bridge fires a release callback once teardown has
//! provably ended all emission, which is where the callback context dies.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use aither_core::llm::tool::{ToolContext, ToolResult, Tools};
use aither_core::llm::{Event, Usage};
use async_channel::Receiver;
use futures_lite::Stream;
use futures_lite::future::Future;

use crate::error::{AppleError, AttachmentError, UnavailableReason};
use crate::ffi::{self, CallbackEvent, CallbackSink};
use crate::wire::{WireEnd, WireStatus};

/// Channel bound between the Swift emitters and the Rust stream. When full,
/// emitters suspend on the resume hub — explicit backpressure, no blocking.
const CHANNEL_CAPACITY: usize = 64;

/// Everything needed to start the native request once planning finished.
pub struct PlannedRequest {
    /// The serialized `WireRequest`.
    pub json: Vec<u8>,
    /// Images registered through `request_add_image`, in index order.
    pub images: Vec<crate::attachments::ResolvedImage>,
    /// Test-only scripted-model JSON (OS27 runtime only).
    #[cfg(aither_scripted)]
    pub script: Option<Vec<u8>>,
}

/// An owned native request; `Drop` cancels and frees it.
struct NativeRequest {
    handle: *mut c_void,
}

// SAFETY: `NativeRequest` is only touched from the task that owns the stream
// and teardown ordering is serialized inside the bridge.
unsafe impl Send for NativeRequest {}

impl NativeRequest {
    fn start(planned: &PlannedRequest, sink: &Arc<CallbackSink>) -> Result<Self, AppleError> {
        // One strong ref of `sink` moves into the callback context. The
        // bridge fires `release_callback` exactly once — after teardown has
        // provably ended all emission — which is where this ref is dropped.
        // `Active` holds no other ref; the context ref is the only one.
        let ctx = Arc::into_raw(Arc::clone(sink)).cast_mut().cast::<c_void>();
        // SAFETY: `ctx` is an owned `Arc<CallbackSink>` ref; `release_cb`
        // reclaims it after the last event callback, so it outlives every
        // use the bridge can make of it. The returned handle is retained by
        // the bridge; `request_free` releases it.
        let handle = unsafe {
            ffi::aither_apple_request_new(ctx, ffi::event_callback, ffi::release_callback)
        };
        if handle.is_null() {
            // The context ref was consumed by the failed call path: reclaim
            // it so the Arc does not leak.
            unsafe { drop(Arc::from_raw(ctx.cast::<CallbackSink>().cast_const())) };
            return Err(AppleError::Native {
                code: "bridge".into(),
                message: "bridge rejected the callback context".into(),
            });
        }
        unsafe {
            ffi::aither_apple_request_set_json(handle, planned.json.as_ptr(), planned.json.len());
        };
        for image in &planned.images {
            // SAFETY: bytes/mime are borrowed for the duration of the call.
            let status = unsafe {
                ffi::aither_apple_request_add_image(
                    handle,
                    image.bytes.as_ptr(),
                    image.bytes.len(),
                    image.media_type.as_ptr(),
                    image.media_type.len(),
                )
            };
            let failure = match status {
                0 => continue,
                3 => AppleError::Attachment(AttachmentError::UnsupportedMediaType(
                    image.media_type.clone(),
                )),
                4 => AppleError::Attachment(AttachmentError::Undecodable),
                // Transcript attachments need OS 27 — a capability problem,
                // not a media-type problem.
                5 => AppleError::UnsupportedCapability(
                    "image attachments require macOS/iOS 27".into(),
                ),
                other => AppleError::Native {
                    code: "bridge".into(),
                    message: format!("unexpected add_image status {other}"),
                },
            };
            unsafe { ffi::aither_apple_request_free(handle) };
            return Err(failure);
        }
        #[cfg(aither_scripted)]
        if let Some(script) = &planned.script {
            // SAFETY: `script` is borrowed for the duration of the call.
            let status = unsafe {
                ffi::aither_apple_request_set_script(handle, script.as_ptr(), script.len())
            };
            if status != 0 {
                unsafe { ffi::aither_apple_request_free(handle) };
                return Err(AppleError::Native {
                    code: "unsupported".into(),
                    message: "scripted models require macOS/iOS 27".into(),
                });
            }
        }
        // SAFETY: the request is configured; events now flow through `ctx`.
        unsafe { ffi::aither_apple_request_start(handle) };
        Ok(Self { handle })
    }

    /// Acknowledges that the consumer freed one channel slot.
    fn resume(&self) {
        unsafe { ffi::aither_apple_request_resume(self.handle) }
    }

    fn cancel(&self) {
        unsafe { ffi::aither_apple_request_cancel(self.handle) }
    }

    /// Delivers a tool result to a suspended `Tool.call`.
    fn tool_result(&self, sequence: u64, is_error: bool, payload: &str) {
        unsafe {
            ffi::aither_apple_request_tool_result(
                self.handle,
                sequence,
                i32::from(is_error),
                payload.as_ptr(),
                payload.len(),
            );
        }
    }
}

impl Drop for NativeRequest {
    fn drop(&mut self) {
        // Cancel unwinds the task and suspended tool continuations; `free`
        // schedules async teardown which fires the release callback once no
        // callback can be in flight — that is where the context Arc dies.
        unsafe {
            ffi::aither_apple_request_cancel(self.handle);
            ffi::aither_apple_request_free(self.handle);
        }
    }
}

/// The live mid-stream state.
struct Active {
    request: NativeRequest,
    /// Pinned: `async_channel::Receiver` is `!Unpin` under `Stream`.
    rx: Pin<Box<Receiver<CallbackEvent>>>,
    terminal: Box<Receiver<CallbackEvent>>,
    /// Cumulative text already surfaced as `Event::Text`.
    emitted_text: String,
    /// Cumulative reasoning already surfaced.
    emitted_reasoning: String,
    /// Latest structured snapshot; emitted once at completion.
    last_structured: Option<String>,
    /// Ready-to-yield events.
    queued: VecDeque<Result<Event, AppleError>>,
    /// The terminal `end` arrived; once `queued` drains the stream is over.
    /// Tracked independently of the end payload so termination is never
    /// confused with a still-open channel.
    terminated: bool,
    /// Whether the external tool boundary was already captured.
    tool_boundary: bool,
}

type ToolTask<'a> =
    Pin<Box<dyn Future<Output = (u64, String, aither_core::Result<ToolResult>)> + Send + 'a>>;

type Starting = Pin<Box<dyn Future<Output = Result<Active, AppleError>> + Send>>;

/// The stream returned by `respond`/`respond_with_tools`.
pub struct Driver<'tools> {
    /// `Some` while an initialization future (attachment resolution +
    /// request start) is still running.
    starting: Option<Starting>,
    active: Option<Active>,
    /// Events queued before any native state exists (initial errors).
    prelude: VecDeque<Result<Event, AppleError>>,
    /// Borrowed tool set for internal execution (`respond_with_tools`).
    /// `Tools::call` takes `&self`, so shared borrows fan out concurrently.
    tools: Option<&'tools Tools>,
    /// In-flight native tool calls.
    pending_tools: Vec<ToolTask<'tools>>,
    done: bool,
}

impl<'tools> Driver<'tools> {
    /// `respond`: tools are captured and surfaced; never executed natively.
    pub fn external(
        plan: impl Future<Output = Result<PlannedRequest, AppleError>> + Send + 'static,
    ) -> Self {
        Self::new(plan, None)
    }

    /// `respond_with_tools`: native tool calls run against `tools`.
    pub fn internal(
        plan: impl Future<Output = Result<PlannedRequest, AppleError>> + Send + 'static,
        tools: &'tools Tools,
    ) -> Self {
        Self::new(plan, Some(tools))
    }

    fn new(
        plan: impl Future<Output = Result<PlannedRequest, AppleError>> + Send + 'static,
        tools: Option<&'tools Tools>,
    ) -> Self {
        Self {
            starting: Some(Box::pin(async move { start(&plan.await?) })),
            active: None,
            prelude: VecDeque::new(),
            tools,
            pending_tools: Vec::new(),
            done: false,
        }
    }

    /// A stream that yields exactly one error — for request-validation
    /// failures detected before any native work begins. No fake native
    /// handle is involved.
    pub fn error(err: AppleError) -> Self {
        Self {
            starting: None,
            active: None,
            prelude: VecDeque::from([Err(err)]),
            tools: None,
            pending_tools: Vec::new(),
            done: false,
        }
    }
}

/// Handles one callback message.
fn on_event<'tools>(
    active: &mut Active,
    tools: Option<&'tools Tools>,
    pending: &mut Vec<ToolTask<'tools>>,
    event: CallbackEvent,
) {
    match event {
        CallbackEvent::Text(text) => append_snapshot(active, text, false),
        CallbackEvent::Reasoning(text) => append_snapshot(active, text, true),
        CallbackEvent::Structured(json) => {
            // Structured snapshots are revised JSON — keep the latest only.
            active.last_structured = Some(json);
        }
        CallbackEvent::ToolCall(call) => {
            if let Some(tools) = tools {
                let name = call.name.clone();
                let args = call.arguments;
                let seq = call.seq;
                pending.push(Box::pin(async move {
                    let result = tools.call(&name, &args, ToolContext::new()).await;
                    (seq, name, result)
                }));
            } else {
                active.queued.push_back(Err(AppleError::Native {
                    code: "unexpected_tool_dispatch".into(),
                    message: "native bridge requested internal execution without a Rust executor"
                        .into(),
                }));
                active.terminated = true;
                active.request.cancel();
            }
        }
        CallbackEvent::ToolBatch(batch) => {
            active.tool_boundary = true;
            for call in batch.calls {
                match serde_json::from_str::<serde_json::Value>(&call.arguments) {
                    Ok(arguments) => {
                        active
                            .queued
                            .push_back(Ok(Event::tool_call(call.id, call.name, arguments)));
                    }
                    Err(error) => {
                        active.queued.push_back(Err(AppleError::Native {
                            code: "malformed_tool_arguments".into(),
                            message: format!("tool '{}': {error}", call.name),
                        }));
                    }
                }
            }
            // The native `Tool.call`s are suspended forever at the boundary;
            // cancelling unwinds them and the turn.
            active.request.cancel();
        }
        CallbackEvent::End(end) => finish(active, end),
        CallbackEvent::Malformed(kind, detail) => {
            active.queued.push_back(Err(AppleError::Native {
                code: "malformed_event".into(),
                message: format!("bridge event {kind}: {detail}"),
            }));
            if kind == ffi::EVENT_END {
                // A malformed terminal still terminates: without it the
                // stream would hang waiting for an end that already ran.
                active.terminated = true;
            }
        }
    }
}

fn append_snapshot(active: &mut Active, text: String, reasoning: bool) {
    let emitted = if reasoning {
        &mut active.emitted_reasoning
    } else {
        &mut active.emitted_text
    };
    if let Ok(delta) = delta(emitted, &text) {
        if !delta.is_empty() {
            let event = if reasoning {
                Event::Reasoning(delta.to_owned())
            } else {
                Event::Text(delta.to_owned())
            };
            active.queued.push_back(Ok(event));
        }
        *emitted = text;
    } else {
        active.queued.push_back(Err(AppleError::Native {
            code: "non_appendable_snapshot".into(),
            message: "native snapshot revised already emitted content".into(),
        }));
        active.terminated = true;
        active.request.cancel();
    }
}

/// Terminal handling for the `end` event. Marks the stream terminated;
/// queued events drain first, then `poll_next` returns `None`.
fn finish(active: &mut Active, end: WireEnd) {
    match end.status {
        WireStatus::Completed => {
            if let Some(json) = active.last_structured.take() {
                active.queued.push_back(Ok(Event::Text(json)));
            }
            if let Some(usage) = end.usage {
                active.queued.push_back(Ok(Event::Usage(Usage {
                    prompt_tokens: u32::try_from(usage.input).ok(),
                    completion_tokens: u32::try_from(usage.output).ok(),
                    total_tokens: u32::try_from(usage.input + usage.output).ok(),
                    reasoning_tokens: u32::try_from(usage.reasoning).ok(),
                    cache_read_tokens: u32::try_from(usage.cached).ok(),
                    cache_write_tokens: None,
                    cost_usd: None,
                    stop_reason: None,
                })));
            }
        }
        // The only clean cancellation is the one the driver initiated at the
        // external tool boundary. Anything else is a real cancellation.
        WireStatus::Cancelled if active.tool_boundary => {}
        WireStatus::Cancelled => {
            active.queued.push_back(Err(AppleError::Cancelled));
        }
        WireStatus::Error => {
            // Errors are never swallowed by the boundary — the cancellation
            // we send ourselves reports as `cancelled`, not `error`.
            active.queued.push_back(Err(map_native_error(&end)));
        }
    }
    active.terminated = true;
}

fn start(planned: &PlannedRequest) -> Result<Active, AppleError> {
    start_observed(
        planned,
        #[cfg(test)]
        None,
    )
}

fn start_observed(
    planned: &PlannedRequest,
    #[cfg(test)] released: Option<async_channel::Sender<()>>,
) -> Result<Active, AppleError> {
    let (tx, rx) = async_channel::bounded(CHANNEL_CAPACITY);
    // `into_raw` moves one strong ref into the callback context; teardown's
    // release callback is the only place it dies.
    let (terminal, terminal_rx) = async_channel::bounded(1);
    let sink = Arc::new(CallbackSink {
        tx,
        terminal,
        #[cfg(test)]
        released,
    });
    let request = NativeRequest::start(planned, &sink)?;
    Ok(Active {
        request,
        rx: Box::pin(rx),
        terminal: Box::new(terminal_rx),
        emitted_text: String::new(),
        emitted_reasoning: String::new(),
        last_structured: None,
        queued: VecDeque::new(),
        terminated: false,
        tool_boundary: false,
    })
}

/// The new tail of a cumulative snapshot, on a UTF-8 boundary. Returns `Err`
/// when the new snapshot is not a pure extension of what was emitted — a
/// revision cannot be represented by append-only `Event::Text`.
fn delta<'a>(emitted: &str, new: &'a str) -> Result<&'a str, ()> {
    new.strip_prefix(emitted).ok_or(())
}

fn map_native_error(end: &WireEnd) -> AppleError {
    let code = end.code.clone().unwrap_or_else(|| "unknown".into());
    let message = end.message.clone().unwrap_or_default();
    match code.as_str() {
        "context_size_exceeded" => AppleError::ContextExceeded(message),
        "rate_limited" => AppleError::RateLimited(message),
        "guardrail_violation" => AppleError::GuardrailViolation(message),
        "refusal" => AppleError::Refusal(message),
        "unsupported_language_or_locale" => AppleError::UnsupportedLanguageOrLocale(message),
        "unsupported_capability" => AppleError::UnsupportedCapability(message),
        "unsupported_transcript_content" => AppleError::UnsupportedRequest(message),
        "unsupported_guide" => AppleError::UnsupportedSchema {
            path: "$".into(),
            reason: message,
        },
        "tool_error" if end.tool.is_some() => AppleError::Tool {
            tool: end.tool.clone().expect("checked tool name"),
            message,
        },
        "decoding_failure" => AppleError::DecodingFailure(message),
        "model_not_ready" => AppleError::Unavailable(UnavailableReason::ModelNotReady),
        _ => AppleError::Native { code, message },
    }
}

impl Stream for Driver<'_> {
    type Item = Result<Event, AppleError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            // Errors queued before native state exists drain first.
            if let Some(item) = this.prelude.pop_front() {
                return Poll::Ready(Some(item));
            }

            // Initialization.
            if let Some(fut) = &mut this.starting {
                match fut.as_mut().poll(cx) {
                    Poll::Ready(Ok(active)) => {
                        this.starting = None;
                        this.active = Some(active);
                        continue;
                    }
                    Poll::Ready(Err(err)) => {
                        this.starting = None;
                        this.done = true;
                        return Poll::Ready(Some(Err(err)));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }

            if this.done {
                return Poll::Ready(None);
            }
            if this.active.is_none() {
                this.done = true;
                return Poll::Ready(None);
            }

            // Completed internal tool calls → feed results back to Swift.
            let mut index = 0;
            while index < this.pending_tools.len() {
                match this.pending_tools[index].as_mut().poll(cx) {
                    Poll::Ready((seq, name, result)) => {
                        drop(this.pending_tools.swap_remove(index));
                        let active = this.active.as_mut().expect("active checked above");
                        let rendered = match result {
                            Ok(tool_result) => render_tool_result(&tool_result),
                            Err(err) => Ok((err.to_string(), true)),
                        };
                        let (output, is_error) = match rendered {
                            Ok(rendered) => rendered,
                            Err(error) => {
                                active.queued.push_back(Err(error));
                                active.terminated = true;
                                active.request.cancel();
                                break;
                            }
                        };
                        active.request.tool_result(seq, is_error, &output);
                        active
                            .queued
                            .push_back(Ok(Event::builtin_result(name, output)));
                    }
                    Poll::Pending => index += 1,
                }
            }

            let active = this.active.as_mut().expect("active checked above");
            if let Some(item) = active.queued.pop_front() {
                return Poll::Ready(Some(item));
            }
            if active.terminated {
                this.done = true;
                return Poll::Ready(None);
            }

            match active.rx.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    if let Ok(event) = active.terminal.try_recv() {
                        on_event(active, this.tools, &mut this.pending_tools, event);
                    } else {
                        active.terminated = true;
                        active.queued.push_back(Err(AppleError::Native {
                            code: "missing_terminal".into(),
                            message: "native owner released without END".into(),
                        }));
                    }
                }
                Poll::Ready(Some(event)) => {
                    // A drained slot wakes one suspended emitter.
                    active.request.resume();
                    on_event(active, this.tools, &mut this.pending_tools, event);
                }
            }
        }
    }
}

/// Renders a `ToolResult` to the text handed to the suspended `Tool.call`.
/// Binary payloads cannot cross the text continuation and become tool-level
/// errors.
fn render_tool_result(result: &ToolResult) -> Result<(String, bool), AppleError> {
    use aither_core::llm::tool::ToolResultPart;
    if matches!(result, ToolResult::Binary { .. })
        || matches!(result, ToolResult::Parts { parts } if parts.iter().any(|p| matches!(p, ToolResultPart::Binary { .. })))
    {
        return Err(AppleError::UnsupportedCapability(
            "binary tool results cannot be represented by the native text tool output".into(),
        ));
    }
    match result.render_for_model() {
        Ok(text) => Ok((text, result.is_error())),
        Err(error) => Err(AppleError::Native {
            code: "tool_result_encoding".into(),
            message: error.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{delta, map_native_error, render_tool_result};
    use crate::{AppleError, wire::WireEnd};
    use aither_core::llm::tool::{ToolResult, ToolResultPart};

    #[test]
    fn timeout_and_tool_errors_retain_native_details() {
        let timeout: WireEnd = serde_json::from_value(
            serde_json::json!({"status":"error","code":"timeout","message":"deadline exceeded"}),
        )
        .unwrap();
        assert!(
            matches!(map_native_error(&timeout), AppleError::Native { code, message } if code == "timeout" && message == "deadline exceeded")
        );
        let tool: WireEnd = serde_json::from_value(serde_json::json!({"status":"error","code":"tool_error","tool":"echo","message":"failed"})).unwrap();
        assert!(
            matches!(map_native_error(&tool), AppleError::Tool { tool, message } if tool == "echo" && message == "failed")
        );
    }

    #[test]
    fn tool_output_preserves_textual_variants_and_rejects_binary() {
        for value in [
            ToolResult::Done,
            ToolResult::text("plain"),
            ToolResult::Tsv {
                text: "a\tb".into(),
            },
            ToolResult::Json {
                value: serde_json::json!({"a":1}),
            },
            ToolResult::Parts {
                parts: vec![
                    ToolResultPart::Text {
                        text: "first".into(),
                    },
                    ToolResultPart::Json {
                        value: serde_json::json!({"b":2}),
                    },
                ],
            },
        ] {
            assert_eq!(
                render_tool_result(&value).unwrap(),
                (value.render_for_model().unwrap(), false)
            );
        }
        for value in [
            ToolResult::Binary {
                mime: "image/png".into(),
                content: vec![1],
            },
            ToolResult::Parts {
                parts: vec![ToolResultPart::Binary {
                    mime: "image/png".into(),
                    content: vec![1],
                }],
            },
        ] {
            assert!(matches!(
                render_tool_result(&value),
                Err(AppleError::UnsupportedCapability(_))
            ));
        }
    }

    /// `delta` is a pure function over UTF-8 strings; the scripted-model
    /// integration tests exercise the same path through the real bridge.
    #[test]
    fn delta_emits_only_the_suffix() {
        assert_eq!(delta("he", "hello").unwrap(), "llo");
        assert_eq!(delta("hello", "hello").unwrap(), "");
        assert_eq!(delta("🦀", "🦀🦀").unwrap(), "🦀");
        assert_eq!(delta("日本語", "日本語テスト").unwrap(), "テスト");
    }

    #[test]
    fn delta_rejects_revisions() {
        // Non-appendable snapshots must not be concatenated into garbage.
        assert!(delta("cat", "car").is_err());
        assert!(delta("hello", "hel").is_err());
        assert!(delta("abc", "abd").is_err());
        assert!(delta("héllo", "héllo x").is_ok());
        assert!(delta("日本語", "日本テスト").is_err());
    }
}

#[cfg(all(test, aither_sdk27, aither_scripted))]
#[path = "native_tests.rs"]
mod native_tests;
