//! Reachability from rtnetlink: a socket subscribed to link and route
//! changes says when to look, a route dump says what is there.

use std::io;

use async_io::Async;
use netlink_sys::{Socket, SocketAddr, protocols::NETLINK_ROUTE};

use super::decode;
use crate::{ConnectivityError, hub::Sink};

/// `RTMGRP_LINK`, `RTMGRP_IPV4_ROUTE` and `RTMGRP_IPV6_ROUTE`, from
/// `linux/rtnetlink.h`.
const GROUPS: u32 = 0x1 | 0x40 | 0x400;

pub struct Watcher {
    events: Async<Socket>,
    dump: Socket,
}

impl Watcher {
    /// Opens the sockets and reports the initial reachability to `sink`.
    pub fn open(sink: &Sink) -> Result<Self, ConnectivityError> {
        let register = |reason: io::Error| ConnectivityError::Register {
            monitor: "rtnetlink",
            reason: reason.to_string(),
        };
        let mut events = Socket::new(NETLINK_ROUTE).map_err(register)?;
        events.bind(&SocketAddr::new(0, GROUPS)).map_err(register)?;
        events.set_non_blocking(true).map_err(register)?;
        let mut dump = Socket::new(NETLINK_ROUTE).map_err(register)?;
        dump.bind_auto().map_err(register)?;
        sink.set_online(has_default_route(&dump).map_err(register)?);
        Ok(Self {
            events: Async::new(events).map_err(register)?,
            dump,
        })
    }

    /// Re-reads the routes whenever the kernel announces a link or route
    /// change, until the socket fails.
    pub async fn watch(self, sink: &Sink) -> io::Result<()> {
        loop {
            match self
                .events
                .read_with(|socket| socket.recv_from_full())
                .await
            {
                Ok(_) => {}
                // The socket's receive buffer overran, so announcements were
                // lost; the dump below is what resynchronises.
                Err(error) if error.raw_os_error() == Some(libc::ENOBUFS) => {}
                Err(error) => {
                    tracing::error!(%error, "rtnetlink socket failed, reachability is no longer tracked");
                    return Err(error);
                }
            }
            // One change usually arrives as a burst; read the dump once.
            while self.events.get_ref().recv_from_full().is_ok() {}
            match has_default_route(&self.dump) {
                Ok(online) => sink.set_online(online),
                Err(error) => {
                    tracing::error!(%error, "route dump failed, reachability is no longer tracked");
                    return Err(error);
                }
            }
        }
    }
}

/// Dumps every route and reports whether one is a default route.
fn has_default_route(dump: &Socket) -> io::Result<bool> {
    dump.send_to(&decode::dump_request(), &SocketAddr::new(0, 0), 0)?;
    let mut default_route = false;
    loop {
        let (datagram, _) = dump.recv_from_full()?;
        let scan = decode::scan(&datagram).map_err(io::Error::other)?;
        default_route |= scan.default_route;
        if scan.done {
            return Ok(default_route);
        }
    }
}
