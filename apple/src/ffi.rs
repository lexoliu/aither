//! Raw FFI declarations for the Swift bridge plus the callback sink.
//!
//! # Safety contract
//!
//! * `aither_apple_request_new` returns a retained opaque pointer; ownership
//!   is released by `aither_apple_request_free`.
//! * `ctx` passed to `request_new` stays valid until the exactly-once release
//!   callback. `request_free` schedules teardown without waiting; release
//!   follows completion of all native tasks and event callbacks.
//! * Callback `data`/`len` are valid only for the call's duration; they are
//!   copied before returning.
//! * Callback return codes: 0 accepted, 1 channel full (the emitter suspends
//!   until `request_resume`), 2 closed (the emitter must stop).

use core::ffi::c_void;
use core::slice;
use std::sync::Arc;

use async_channel::{Sender, TrySendError};

use crate::wire::{WireEnd, WireToolBatch, WireToolEvent};

pub type EventCallback = unsafe extern "C" fn(*mut c_void, i32, *const u8, usize) -> i32;

const CB_ACCEPTED: i32 = 0;
const CB_FULL: i32 = 1;
const CB_CLOSED: i32 = 2;
pub const EVENT_END: i32 = 6;

unsafe extern "C" {
    /// Calls `detail` synchronously for an unknown reason; retains neither argument.
    pub fn aither_apple_availability(
        ctx: *mut c_void,
        detail: unsafe extern "C" fn(*mut c_void, *const u8, usize),
    ) -> i32;
    pub fn aither_apple_context_size() -> i64;
    pub fn aither_apple_capabilities() -> u32;
    pub fn aither_apple_request_new(
        ctx: *mut c_void,
        cb: EventCallback,
        release: unsafe extern "C" fn(*mut c_void),
    ) -> *mut c_void;
    pub fn aither_apple_request_set_json(handle: *mut c_void, data: *const u8, len: usize);
    pub fn aither_apple_request_add_image(
        handle: *mut c_void,
        data: *const u8,
        len: usize,
        mime: *const u8,
        mime_len: usize,
    ) -> i32;
    #[cfg(aither_scripted)]
    pub fn aither_apple_request_set_script(handle: *mut c_void, data: *const u8, len: usize)
    -> i32;
    pub fn aither_apple_request_start(handle: *mut c_void);
    pub fn aither_apple_request_cancel(handle: *mut c_void);
    pub fn aither_apple_request_resume(handle: *mut c_void);
    pub fn aither_apple_request_tool_result(
        handle: *mut c_void,
        seq: u64,
        is_error: i32,
        data: *const u8,
        len: usize,
    );
    pub fn aither_apple_request_free(handle: *mut c_void);
}

/// Events decoded from bridge callbacks.
#[derive(Debug)]
pub enum CallbackEvent {
    /// Cumulative text snapshot.
    Text(String),
    /// Cumulative reasoning text.
    Reasoning(String),
    /// Revised structured JSON snapshot.
    Structured(String),
    /// One tool request needing a Rust-side answer.
    ToolCall(WireToolEvent),
    /// The whole captured tool-call batch (external mode).
    ToolBatch(WireToolBatch),
    /// Terminal event; after it no callbacks arrive.
    End(WireEnd),
    /// A payload that failed to decode; surfaced as a native error.
    Malformed(i32, String),
}

/// Lives behind the callback context pointer. Holds the channel sender only;
/// the receiver lives in the stream.
pub struct CallbackSink {
    pub tx: Sender<CallbackEvent>,
    pub terminal: Sender<CallbackEvent>,
    #[cfg(test)]
    pub released: Option<Sender<()>>,
}

/// # Safety
/// Consumes the context reference exactly once after all native emissions end.
pub unsafe extern "C" fn release_callback(ctx: *mut c_void) {
    // SAFETY: the native owner transfers its sole context reference back here.
    let sink = unsafe { Arc::from_raw(ctx.cast::<CallbackSink>()) };
    sink.tx.close();
    #[cfg(test)]
    if let Some(released) = &sink.released {
        let _ = released.try_send(());
    }
}

/// # Safety
///
/// `ctx` must be a `*mut CallbackSink` registered with `request_new` whose
/// owning request has not been freed. `data` must be valid for `len` bytes
/// for the duration of the call.
pub unsafe extern "C" fn event_callback(
    ctx: *mut c_void,
    kind: i32,
    data: *const u8,
    len: usize,
) -> i32 {
    if ctx.is_null() {
        return CB_CLOSED;
    }
    // SAFETY: per the module contract, `ctx` is a live `CallbackSink` while
    // the gate keeps the request un-freed.
    let sink = unsafe { &*ctx.cast::<CallbackSink>() };
    // SAFETY: `data` points to `len` bytes owned by the caller for the
    // duration of this call; we copy into an owned String immediately.
    let payload: &[u8] = unsafe {
        if data.is_null() {
            &[]
        } else {
            slice::from_raw_parts(data, len)
        }
    };
    let event = match decode_event(kind, payload) {
        Ok(event) => event,
        Err(err) => CallbackEvent::Malformed(kind, err),
    };
    if kind == EVENT_END {
        // A dedicated one-element terminal channel cannot be crowded out by
        // data. Close data after publishing END; the driver drains data first.
        let status = match sink.terminal.try_send(event) {
            Ok(()) => CB_ACCEPTED,
            Err(_) => CB_CLOSED,
        };
        sink.tx.close();
        return status;
    }
    match sink.tx.try_send(event) {
        Ok(()) => CB_ACCEPTED,
        Err(TrySendError::Full(_)) => CB_FULL,
        Err(TrySendError::Closed(_)) => CB_CLOSED,
    }
}

fn decode_event(kind: i32, payload: &[u8]) -> Result<CallbackEvent, String> {
    let text = String::from_utf8(payload.to_vec()).map_err(|e| e.to_string())?;
    Ok(match kind {
        1 => CallbackEvent::Text(text),
        2 => CallbackEvent::Reasoning(text),
        3 => CallbackEvent::Structured(text),
        4 => CallbackEvent::ToolCall(
            serde_json::from_str(&text).map_err(|e| format!("tool_call payload: {e}"))?,
        ),
        5 => CallbackEvent::ToolBatch(
            serde_json::from_str(&text).map_err(|e| format!("tool_batch payload: {e}"))?,
        ),
        6 => CallbackEvent::End(
            serde_json::from_str(&text).map_err(|e| format!("end payload: {e}"))?,
        ),
        other => CallbackEvent::Malformed(other, text),
    })
}

/// Queries the native availability of `SystemLanguageModel.default`.
pub fn native_availability() -> crate::Availability {
    let mut detail = None;
    // SAFETY: Swift calls back synchronously without retaining either argument.
    // The stack slot remains exclusively borrowed until this query returns.
    let code = unsafe {
        aither_apple_availability(core::ptr::from_mut(&mut detail).cast(), availability_detail)
    };
    decode_availability(code, detail)
}

/// # Safety
/// `ctx` points to an exclusively borrowed `Option<String>` for this synchronous
/// call. `data` contains `len` UTF-8 bytes borrowed from a Swift String.
unsafe extern "C" fn availability_detail(ctx: *mut c_void, data: *const u8, len: usize) {
    // SAFETY: the query owns the stack slot and callbacks cannot escape or overlap.
    let detail = unsafe { &mut *ctx.cast::<Option<String>>() };
    let bytes = if len == 0 {
        &[]
    } else {
        // SAFETY: Swift borrows its UTF-8 buffer for the duration of this call.
        unsafe { slice::from_raw_parts(data, len) }
    };
    *detail = Some(
        core::str::from_utf8(bytes)
            .expect("native availability detail must be UTF-8")
            .to_owned(),
    );
}

fn decode_availability(code: i32, detail: Option<String>) -> crate::Availability {
    use crate::error::UnavailableReason as R;
    match code {
        0 => crate::Availability::Available,
        1 => crate::Availability::Unavailable(R::UnsupportedOs),
        2 => crate::Availability::Unavailable(R::DeviceNotEligible),
        3 => crate::Availability::Unavailable(R::AppleIntelligenceNotEnabled),
        4 => crate::Availability::Unavailable(R::ModelNotReady),
        5 => crate::Availability::Unavailable(R::Unknown(
            detail.expect("unknown native availability reason must include its description"),
        )),
        other => panic!("unexpected native availability bridge code: {other}"),
    }
}

/// Native context size in tokens; `None` when it cannot be queried.
pub fn native_context_size() -> Option<u32> {
    // SAFETY: pure query.
    let size = unsafe { aither_apple_context_size() };
    u32::try_from(size).ok()
}

/// Runtime capability bitmask.
pub fn native_capabilities() -> u32 {
    // SAFETY: pure query.
    unsafe { aither_apple_capabilities() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_callback_preserves_owned_unknown_detail() {
        let mut detail: Option<String> = None;
        let text = String::from("futureReason(detail: \"未就绪\")");
        // SAFETY: both pointers remain valid for this synchronous callback.
        unsafe {
            availability_detail(
                core::ptr::from_mut(&mut detail).cast(),
                text.as_ptr(),
                text.len(),
            );
        };
        drop(text);
        let expected =
            crate::error::UnavailableReason::Unknown("futureReason(detail: \"未就绪\")".into());
        assert_eq!(
            decode_availability(5, detail),
            crate::Availability::Unavailable(expected)
        );
        assert_eq!(
            decode_availability(2, None),
            crate::Availability::Unavailable(crate::error::UnavailableReason::DeviceNotEligible)
        );
    }

    #[test]
    #[should_panic(expected = "unexpected native availability bridge code: -1")]
    fn unexpected_availability_code_fails_clearly() {
        decode_availability(-1, None);
    }

    #[test]
    #[should_panic(expected = "unknown native availability reason must include its description")]
    fn unknown_availability_requires_native_detail() {
        decode_availability(5, None);
    }

    #[test]
    fn full_channel_preserves_terminal_and_malformed_terminal() {
        for payload in [
            b"{\"status\":\"completed\"}".as_slice(),
            b"malformed".as_slice(),
        ] {
            let (tx, rx) = async_channel::bounded(1);
            let (terminal, end) = async_channel::bounded(1);
            tx.try_send(CallbackEvent::Text("queued".into())).unwrap();
            let sink = Arc::new(CallbackSink {
                tx,
                terminal,
                released: None,
            });
            let context = Arc::into_raw(sink).cast_mut().cast();
            // SAFETY: exactly one retained context reference is released below;
            // payload remains valid throughout the callback.
            assert_eq!(
                unsafe { event_callback(context, EVENT_END, payload.as_ptr(), payload.len()) },
                CB_ACCEPTED
            );
            assert!(matches!(rx.try_recv(), Ok(CallbackEvent::Text(t)) if t == "queued"));
            assert!(rx.is_closed());
            assert!(matches!(
                end.try_recv(),
                Ok(CallbackEvent::End(_) | CallbackEvent::Malformed(EVENT_END, _))
            ));
            // SAFETY: no callbacks remain in flight.
            unsafe {
                release_callback(context);
            }
        }
    }
}
