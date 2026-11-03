//! A runtime-agnostic timer.

use core::time::Duration;

/// Completes once `duration` has passed.
///
/// Native targets park on `async-io`'s reactor. wasm32 has no reactor, and the
/// browser timer is not `Send` while the futures built on this one must be, so
/// the timer runs on the JS event loop and is awaited through a channel.
pub async fn sleep(duration: Duration) {
    #[cfg(not(target_arch = "wasm32"))]
    async_io::Timer::after(duration).await;

    #[cfg(target_arch = "wasm32")]
    {
        let millis = u32::try_from(duration.as_millis()).unwrap_or(u32::MAX);
        let (elapsed_tx, elapsed_rx) = async_channel::bounded::<()>(1);
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::TimeoutFuture::new(millis).await;
            let _ = elapsed_tx.send(()).await;
        });
        // A closed channel means the timer task was dropped, which only
        // happens while the whole future is being torn down.
        let _ = elapsed_rx.recv().await;
    }
}
