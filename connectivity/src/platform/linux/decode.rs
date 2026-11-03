//! Reads `RTM_GETROUTE` dump replies: the machine is online when the main
//! routing table has a default route.
//!
//! See [`rtnetlink(7)`](https://man7.org/linux/man-pages/man7/rtnetlink.7.html)
//! for the message types and [`ip-route(8)`](https://man7.org/linux/man-pages/man8/ip-route.8.html)
//! for the route fields.

use netlink_packet_core::{
    NLM_F_DUMP, NLM_F_REQUEST, NetlinkHeader, NetlinkMessage, NetlinkPayload,
};
use netlink_packet_route::{
    RouteNetlinkMessage,
    route::{RouteAttribute, RouteFlags, RouteHeader, RouteMessage, RouteType},
};

/// Netlink messages are padded to four bytes (`NLMSG_ALIGN`).
const NLMSG_ALIGNTO: usize = 4;

/// What one datagram of a route dump held.
#[derive(Debug, PartialEq, Eq)]
pub struct Scan {
    /// A default route is among the datagram's routes.
    pub default_route: bool,
    /// The datagram carried the dump's terminating `NLMSG_DONE`.
    pub done: bool,
}

/// Why a dump reply was rejected.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// The kernel sent bytes that are not a netlink message.
    #[error("malformed netlink message: {0}")]
    Malformed(String),
    /// The kernel answered the dump request with an error.
    #[error("the kernel refused the route dump: errno {0}")]
    Refused(i32),
}

/// The `RTM_GETROUTE` request for every route of every address family.
pub fn dump_request() -> Vec<u8> {
    let mut header = NetlinkHeader::default();
    header.flags = NLM_F_REQUEST | NLM_F_DUMP;
    header.sequence_number = 1;
    let mut message = NetlinkMessage::new(
        header,
        NetlinkPayload::InnerMessage(RouteNetlinkMessage::GetRoute(RouteMessage::default())),
    );
    message.finalize();
    let mut buffer = vec![0; message.buffer_len()];
    message.serialize(&mut buffer);
    buffer
}

/// Reads one datagram of a dump reply.
pub fn scan(datagram: &[u8]) -> Result<Scan, DecodeError> {
    let mut scan = Scan {
        default_route: false,
        done: false,
    };
    let mut offset = 0;
    while offset < datagram.len() {
        let message = NetlinkMessage::<RouteNetlinkMessage>::deserialize(&datagram[offset..])
            .map_err(|error| DecodeError::Malformed(error.to_string()))?;
        let length = message.header.length as usize;
        match message.payload {
            NetlinkPayload::Done(_) => scan.done = true,
            NetlinkPayload::Error(error) => {
                if let Some(code) = error.code {
                    return Err(DecodeError::Refused(code.get()));
                }
            }
            NetlinkPayload::InnerMessage(RouteNetlinkMessage::NewRoute(route)) => {
                scan.default_route |= is_default_route(&route);
            }
            _ => {}
        }
        offset += length.next_multiple_of(NLMSG_ALIGNTO);
    }
    Ok(scan)
}

/// Whether `route` sends traffic with no more specific route out an interface.
fn is_default_route(route: &RouteMessage) -> bool {
    let table = route
        .attributes
        .iter()
        .find_map(|attribute| match attribute {
            RouteAttribute::Table(table) => Some(*table),
            _ => None,
        })
        .unwrap_or_else(|| u32::from(route.header.table));
    let leaves_the_machine = route.attributes.iter().any(|attribute| {
        matches!(
            attribute,
            RouteAttribute::Oif(_) | RouteAttribute::MultiPath(_)
        )
    });
    route.header.destination_prefix_length == 0
        && route.header.kind == RouteType::Unicast
        && !route.header.flags.contains(RouteFlags::Dead)
        && table == u32::from(RouteHeader::RT_TABLE_MAIN)
        && leaves_the_machine
}

#[cfg(test)]
mod tests {
    use netlink_packet_core::DoneMessage;
    use netlink_packet_route::{AddressFamily, route::RouteHeader};

    use super::*;

    fn route(
        prefix_length: u8,
        table: u8,
        kind: RouteType,
        attributes: Vec<RouteAttribute>,
    ) -> RouteMessage {
        let mut message = RouteMessage::default();
        message.header = RouteHeader {
            address_family: AddressFamily::Inet,
            destination_prefix_length: prefix_length,
            table,
            kind,
            ..RouteHeader::default()
        };
        message.attributes = attributes;
        message
    }

    fn datagram(payloads: Vec<NetlinkPayload<RouteNetlinkMessage>>) -> Vec<u8> {
        let mut bytes = Vec::new();
        for payload in payloads {
            let mut message = NetlinkMessage::new(NetlinkHeader::default(), payload);
            message.finalize();
            let start = bytes.len();
            bytes.resize(
                start + message.buffer_len().next_multiple_of(NLMSG_ALIGNTO),
                0,
            );
            message.serialize(&mut bytes[start..]);
        }
        bytes
    }

    fn new_route(route: RouteMessage) -> NetlinkPayload<RouteNetlinkMessage> {
        NetlinkPayload::InnerMessage(RouteNetlinkMessage::NewRoute(route))
    }

    #[test]
    fn a_default_route_in_the_main_table_is_online() {
        let bytes = datagram(vec![new_route(route(
            0,
            RouteHeader::RT_TABLE_MAIN,
            RouteType::Unicast,
            vec![RouteAttribute::Oif(2)],
        ))]);
        assert_eq!(
            scan(&bytes).unwrap(),
            Scan {
                default_route: true,
                done: false
            }
        );
    }

    #[test]
    fn specific_routes_and_other_tables_are_not_default() {
        let bytes = datagram(vec![
            new_route(route(
                24,
                RouteHeader::RT_TABLE_MAIN,
                RouteType::Unicast,
                vec![RouteAttribute::Oif(2)],
            )),
            new_route(route(
                0,
                255,
                RouteType::Unicast,
                vec![RouteAttribute::Oif(2)],
            )),
            new_route(route(
                0,
                RouteHeader::RT_TABLE_MAIN,
                RouteType::BlackHole,
                vec![],
            )),
            new_route(route(
                0,
                RouteHeader::RT_TABLE_MAIN,
                RouteType::Unicast,
                vec![],
            )),
        ]);
        assert!(!scan(&bytes).unwrap().default_route);
    }

    #[test]
    fn a_large_table_number_travels_as_an_attribute() {
        let bytes = datagram(vec![new_route(route(
            0,
            RouteHeader::RT_TABLE_UNSPEC,
            RouteType::Unicast,
            vec![RouteAttribute::Oif(2), RouteAttribute::Table(1000)],
        ))]);
        assert!(!scan(&bytes).unwrap().default_route);
    }

    #[test]
    fn done_ends_the_dump() {
        let bytes = datagram(vec![NetlinkPayload::Done(DoneMessage::default())]);
        assert!(scan(&bytes).unwrap().done);
    }

    #[test]
    fn the_request_asks_for_a_dump() {
        let request = dump_request();
        let message = NetlinkMessage::<RouteNetlinkMessage>::deserialize(&request).unwrap();
        assert_eq!(message.header.flags, NLM_F_REQUEST | NLM_F_DUMP);
        assert!(matches!(
            message.payload,
            NetlinkPayload::InnerMessage(RouteNetlinkMessage::GetRoute(_))
        ));
    }

    #[test]
    fn garbage_is_malformed() {
        assert!(matches!(scan(&[1, 2, 3]), Err(DecodeError::Malformed(_))));
    }
}
