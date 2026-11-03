//! Windows: the network connectivity hint for reachability and suspend/resume
//! power notifications for sleep and wake.
//!
//! - <https://learn.microsoft.com/en-us/windows/win32/api/netioapi/nf-netioapi-notifynetworkconnectivityhintchange>
//! - <https://learn.microsoft.com/en-us/windows/win32/api/netioapi/ne-netioapi-nl_network_connectivity_level_hint>
//! - <https://learn.microsoft.com/en-us/windows/win32/api/netioapi/nf-netioapi-cancelmibchangenotify2>
//! - <https://learn.microsoft.com/en-us/windows/win32/api/powrprof/nf-powrprof-powerregistersuspendresumenotification>
//! - <https://learn.microsoft.com/en-us/windows/win32/power/wm-powerbroadcast>

use std::{ffi::c_void, ptr};

use windows_sys::Win32::{
    Foundation::{ERROR_SUCCESS, HANDLE},
    NetworkManagement::IpHelper::{CancelMibChangeNotify2, NotifyNetworkConnectivityHintChange},
    Networking::WinSock::{
        NL_NETWORK_CONNECTIVITY_HINT, NetworkConnectivityLevelHintConstrainedInternetAccess,
        NetworkConnectivityLevelHintInternetAccess, NetworkConnectivityLevelHintUnknown,
    },
    System::Power::{
        DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS, PowerRegisterSuspendResumeNotification,
        PowerUnregisterSuspendResumeNotification,
    },
    UI::WindowsAndMessaging::{DEVICE_NOTIFY_CALLBACK, PBT_APMRESUMEAUTOMATIC, PBT_APMSUSPEND},
};

use crate::{ConnectivityError, hub::Sink};

/// Watches reachability and power state until dropped.
pub struct Monitor {
    // Both callbacks borrow this boxed sink through a raw pointer; it is
    // freed after the registrations are cancelled.
    sink: *mut Sink,
    connectivity: HANDLE,
    power: *mut c_void,
    // Referenced by the power registration until it is cancelled.
    _power_parameters: Box<DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS>,
}

// SAFETY: the handles are only used to cancel in `Drop`, and the sink behind
// `sink` is shared and immutable.
unsafe impl Send for Monitor {}
// SAFETY: as above, there is no shared mutable access.
unsafe impl Sync for Monitor {}

impl Monitor {
    pub fn start(sink: Sink) -> Result<Self, ConnectivityError> {
        let sink = Box::into_raw(Box::new(sink));

        let mut connectivity: HANDLE = ptr::null_mut();
        // SAFETY: `sink` outlives the registration (see `Drop`). With an
        // initial notification the callback reports the current level at once.
        let status = unsafe {
            NotifyNetworkConnectivityHintChange(
                Some(connectivity_callback),
                sink.cast(),
                true,
                &raw mut connectivity,
            )
        };
        if status != ERROR_SUCCESS {
            // SAFETY: nothing else holds the sink after a failed registration.
            drop(unsafe { Box::from_raw(sink) });
            return Err(register_error(
                "NotifyNetworkConnectivityHintChange",
                status,
            ));
        }

        let parameters = Box::new(DEVICE_NOTIFY_SUBSCRIBE_PARAMETERS {
            Callback: Some(power_callback),
            Context: sink.cast(),
        });
        let mut power: *mut c_void = ptr::null_mut();
        // SAFETY: `parameters` and `sink` outlive the registration. For
        // `DEVICE_NOTIFY_CALLBACK` the recipient is a pointer to the
        // subscription parameters.
        let status = unsafe {
            PowerRegisterSuspendResumeNotification(
                DEVICE_NOTIFY_CALLBACK,
                ptr::from_ref(&*parameters).cast_mut().cast(),
                &raw mut power,
            )
        };
        if status != ERROR_SUCCESS {
            // SAFETY: the registration just made is cancelled, which waits
            // for a running callback, and then nothing holds the sink.
            unsafe {
                CancelMibChangeNotify2(connectivity);
                drop(Box::from_raw(sink));
            }
            return Err(register_error(
                "PowerRegisterSuspendResumeNotification",
                status,
            ));
        }

        Ok(Self {
            sink,
            connectivity,
            power,
            _power_parameters: parameters,
        })
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        // SAFETY: both handles came from successful registrations and are
        // cancelled once. Cancelling waits for a callback that is running, so
        // none touches the sink afterwards.
        unsafe {
            PowerUnregisterSuspendResumeNotification(self.power as isize);
            CancelMibChangeNotify2(self.connectivity);
            drop(Box::from_raw(self.sink));
        }
    }
}

fn register_error(monitor: &'static str, status: u32) -> ConnectivityError {
    ConnectivityError::Register {
        monitor,
        reason: std::io::Error::from_raw_os_error(status.cast_signed()).to_string(),
    }
}

/// Whether a connectivity level reaches the internet. An unknown level is not
/// evidence of loss, so it counts as reachable.
const fn reaches_internet(level: i32) -> bool {
    level == NetworkConnectivityLevelHintInternetAccess
        || level == NetworkConnectivityLevelHintConstrainedInternetAccess
        || level == NetworkConnectivityLevelHintUnknown
}

/// <https://learn.microsoft.com/en-us/windows/win32/api/netioapi/nc-netioapi-pnetwork_connectivity_hint_change_callback>
unsafe extern "system" fn connectivity_callback(
    context: *const c_void,
    hint: NL_NETWORK_CONNECTIVITY_HINT,
) {
    // SAFETY: `context` is the sink registered in `Monitor::start`, alive
    // until the registration is cancelled.
    let sink = unsafe { &*context.cast::<Sink>() };
    sink.set_online(reaches_internet(hint.ConnectivityLevel));
}

/// <https://learn.microsoft.com/en-us/windows/win32/api/powersetting/nc-powersetting-device_notify_callback_routine>
unsafe extern "system" fn power_callback(
    context: *const c_void,
    event: u32,
    _setting: *const c_void,
) -> u32 {
    // SAFETY: as in `connectivity_callback`.
    let sink = unsafe { &*context.cast::<Sink>() };
    match event {
        PBT_APMSUSPEND => sink.sleeping(),
        // `PBT_APMRESUMEAUTOMATIC` arrives on every resume; the user-initiated
        // `PBT_APMRESUMESUSPEND` follows it only when the user is present, so
        // only the first is a wake.
        PBT_APMRESUMEAUTOMATIC => sink.woke(),
        _ => {}
    }
    0
}
