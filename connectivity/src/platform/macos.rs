//! macOS: Network.framework path monitor for reachability and `IOKit` system
//! power notifications for sleep and wake.
//!
//! - <https://developer.apple.com/documentation/network/nw_path_monitor_t>
//! - <https://developer.apple.com/documentation/network/nw_path_status_t>
//! - <https://developer.apple.com/documentation/iokit/1557114-ioregisterforsystempower>
//! - <https://developer.apple.com/documentation/iokit/1557115-ioallowpowerchange>

use std::{
    ffi::c_void,
    ptr::{self, NonNull},
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use block2::{DynBlock, RcBlock};
use dispatch2::{DispatchQueue, DispatchRetained};
use objc2_io_kit::{
    IOAllowPowerChange, IODeregisterForSystemPower, IONotificationPort, IONotificationPortRef,
    IORegisterForSystemPower, IOServiceClose, io_object_t, io_service_t, kIOMessageCanSystemSleep,
    kIOMessageSystemHasPoweredOn, kIOMessageSystemWillSleep,
};

use crate::{ConnectivityError, hub::Sink};

/// `nw_path_status_satisfied`: the path can reach a network now.
const NW_PATH_STATUS_SATISFIED: u32 = 1;

#[link(name = "Network", kind = "framework")]
unsafe extern "C" {
    fn nw_path_monitor_create() -> *mut c_void;
    fn nw_path_monitor_set_queue(monitor: *mut c_void, queue: &DispatchQueue);
    fn nw_path_monitor_set_update_handler(
        monitor: *mut c_void,
        handler: &DynBlock<dyn Fn(*mut c_void)>,
    );
    fn nw_path_monitor_start(monitor: *mut c_void);
    fn nw_path_monitor_cancel(monitor: *mut c_void);
    fn nw_path_get_status(path: *mut c_void) -> u32;
}

unsafe extern "C" {
    /// What the `nw_release` macro expands to.
    fn os_release(object: *mut c_void);
}

/// Watches reachability and power state until dropped.
pub struct Monitor {
    path: PathMonitor,
    power: PowerRegistration,
}

impl Monitor {
    pub fn start(sink: Sink) -> Result<Self, ConnectivityError> {
        let queue = DispatchQueue::new("aither.connectivity", None);
        let power = PowerRegistration::register(sink.clone(), &queue)?;
        let path = PathMonitor::start(sink, &queue)?;
        Ok(Self { path, power })
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.path.cancel();
        self.power.deregister();
    }
}

/// An owned `nw_path_monitor_t`.
struct PathMonitor(NonNull<c_void>);

// SAFETY: the pointer is an OS object reference whose only uses after
// construction are `cancel` and `Drop`, both safe to call from any thread.
unsafe impl Send for PathMonitor {}
// SAFETY: as above, there is no shared mutable access.
unsafe impl Sync for PathMonitor {}

impl PathMonitor {
    fn start(sink: Sink, queue: &DispatchQueue) -> Result<Self, ConnectivityError> {
        // SAFETY: creates a new monitor; null is the only failure.
        let raw = unsafe { nw_path_monitor_create() };
        let monitor = NonNull::new(raw).ok_or_else(|| ConnectivityError::Register {
            monitor: "nw_path_monitor",
            reason: "nw_path_monitor_create returned null".into(),
        })?;
        let handler = RcBlock::new(move |path: *mut c_void| {
            // SAFETY: Network.framework passes a live `nw_path_t`.
            let status = unsafe { nw_path_get_status(path) };
            sink.set_online(status == NW_PATH_STATUS_SATISFIED);
        });
        // SAFETY: `monitor` is live; the framework copies the block, so
        // dropping ours afterwards is sound, and the copy owns the sink.
        unsafe {
            nw_path_monitor_set_queue(monitor.as_ptr(), queue);
            nw_path_monitor_set_update_handler(monitor.as_ptr(), &handler);
            nw_path_monitor_start(monitor.as_ptr());
        }
        Ok(Self(monitor))
    }

    fn cancel(&self) {
        // SAFETY: the monitor is live until `Drop`.
        unsafe { nw_path_monitor_cancel(self.0.as_ptr()) };
    }
}

impl Drop for PathMonitor {
    fn drop(&mut self) {
        // SAFETY: releases the reference `nw_path_monitor_create` returned.
        unsafe { os_release(self.0.as_ptr()) };
    }
}

/// What the power callback reads, reachable through the registration's
/// `refcon`.
struct PowerContext {
    sink: Sink,
    /// The `io_connect_t` `IOAllowPowerChange` acknowledges on.
    root_port: AtomicU32,
}

/// An `IORegisterForSystemPower` registration.
struct PowerRegistration {
    context: *const PowerContext,
    port: IONotificationPortRef,
    notifier: io_object_t,
    root_port: u32,
    queue: DispatchRetained<DispatchQueue>,
}

// SAFETY: the raw handles are used only to deregister in `Drop`, and the
// context is `Sync` (an atomic and a `Sink`).
unsafe impl Send for PowerRegistration {}
// SAFETY: as above, there is no shared mutable access.
unsafe impl Sync for PowerRegistration {}

impl PowerRegistration {
    fn register(
        sink: Sink,
        queue: &DispatchRetained<DispatchQueue>,
    ) -> Result<Self, ConnectivityError> {
        let context = Arc::into_raw(Arc::new(PowerContext {
            sink,
            root_port: AtomicU32::new(0),
        }));
        let mut port: IONotificationPortRef = ptr::null_mut();
        let mut notifier: io_object_t = 0;
        // SAFETY: `context` stays alive until `deregister`, which runs after
        // the notification port, and with it every callback, is gone.
        let root_port = unsafe {
            IORegisterForSystemPower(
                context.cast_mut().cast(),
                &raw mut port,
                Some(power_callback),
                &raw mut notifier,
            )
        };
        if root_port == 0 {
            // SAFETY: the registration failed, so nothing else holds it.
            drop(unsafe { Arc::from_raw(context) });
            return Err(ConnectivityError::Register {
                monitor: "IORegisterForSystemPower",
                reason: "the call returned IO_OBJECT_NULL".into(),
            });
        }
        // SAFETY: `context` is live. Stored before the queue is set, which is
        // what starts callbacks.
        unsafe { &*context }
            .root_port
            .store(root_port, Ordering::SeqCst);
        // SAFETY: `port` was returned by the registration just made.
        unsafe { IONotificationPort::set_dispatch_queue(port, Some(queue)) };
        Ok(Self {
            context,
            port,
            notifier,
            root_port,
            queue: queue.clone(),
        })
    }

    fn deregister(&mut self) {
        // SAFETY: each handle is the one the registration returned and is
        // released exactly once; destroying the port stops delivery.
        unsafe {
            IODeregisterForSystemPower(&raw mut self.notifier);
            IOServiceClose(self.root_port);
            IONotificationPort::destroy(self.port);
        }
        // A callback may still be running on the serial queue; once this
        // empty block has run, none is.
        self.queue.exec_sync(|| {});
        // SAFETY: no callback can read the context any more.
        drop(unsafe { Arc::from_raw(self.context) });
    }
}

/// <https://developer.apple.com/documentation/iokit/ioservicecallback>
///
/// Runs on the monitor's serial queue.
#[allow(non_upper_case_globals)] // IOKit's message constants are named `kIOMessage*`.
unsafe extern "C-unwind" fn power_callback(
    refcon: *mut c_void,
    _service: io_service_t,
    message_type: u32,
    message_argument: *mut c_void,
) {
    // SAFETY: `refcon` is the context pointer registered above, alive until
    // the registration is torn down.
    let context = unsafe { &*refcon.cast::<PowerContext>() };
    match message_type {
        kIOMessageSystemWillSleep => {
            context.sink.sleeping();
            allow_power_change(context, message_argument);
        }
        // Idle sleep may be vetoed; acknowledging does not veto it.
        kIOMessageCanSystemSleep => allow_power_change(context, message_argument),
        kIOMessageSystemHasPoweredOn => context.sink.woke(),
        _ => {}
    }
}

/// Acknowledges a will-sleep or can-sleep notification; sleep is otherwise
/// delayed by 30 seconds.
fn allow_power_change(context: &PowerContext, notification_id: *mut c_void) {
    IOAllowPowerChange(
        context.root_port.load(Ordering::SeqCst),
        notification_id as isize,
    );
}
