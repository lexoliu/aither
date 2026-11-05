//! Destination checks that keep a fetch pointed at the public internet.
//!
//! The URL this crate fetches is typically chosen by a language model, which
//! may be acting on text from a page it fetched a moment ago. That makes a
//! plain `GET` a server-side request forgery primitive: a model that is merely
//! confused — never mind one that has been prompt-injected — can be talked into
//! fetching `http://169.254.169.254/latest/meta-data/iam/security-credentials/`
//! and handing the result back as tool output.
//!
//! Every entry point therefore resolves the host and refuses any address that
//! is not publicly routable.
//!
//! Redirects are not a hole in the HTTP client, which is built without
//! redirect-following middleware, so a `302` is returned rather than followed;
//! if that ever changes, each hop must be re-checked through [`guard_url`].
//! The headless browser does follow redirects and load subresources, so it
//! checks every request it makes against this same policy through
//! [`crate::intercept`] instead of trusting the first URL. That interception
//! cannot see which address the browser itself resolves a host to, so DNS
//! rebinding against the browser is not covered; see that module.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use url::{Host, Url};

/// Why a URL was refused before any request was made.
#[derive(Debug, thiserror::Error)]
pub enum UrlPolicyError {
    /// The string could not be parsed as a URL.
    #[error("invalid URL: {0}")]
    Invalid(#[from] url::ParseError),

    /// Only `http` and `https` are fetchable.
    #[error("unsupported URL scheme '{0}': only http and https are allowed")]
    UnsupportedScheme(String),

    /// The URL had no host component (for example `http:///foo`).
    #[error("URL has no host")]
    MissingHost,

    /// The host could not be resolved to any address.
    #[error("could not resolve host '{host}': {source}")]
    Resolve {
        /// The host that failed to resolve.
        host: String,
        /// The underlying resolver error.
        source: std::io::Error,
    },

    /// The host resolved to an address that is not publicly routable.
    #[error(
        "refusing to fetch '{host}': it resolves to {addr}, which is a private, loopback, or \
         link-local address"
    )]
    BlockedAddress {
        /// The host as written in the URL.
        host: String,
        /// The disallowed address it resolved to.
        addr: IpAddr,
    },
}

/// Whether an address is one this crate refuses to connect to.
///
/// Covers loopback, unspecified, multicast and broadcast addresses, the RFC1918
/// private ranges, link-local (including the cloud metadata range), carrier-grade
/// NAT, IPv6 unique-local and link-local, and IPv4-mapped forms of all of these.
#[must_use]
pub fn is_blocked(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => {
            // An IPv4-mapped address reaches the same host as the bare IPv4
            // address, so it must be judged by the same rules.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_blocked_v4(mapped);
            }
            is_blocked_v6(v6)
        }
    }
}

fn is_blocked_v4(addr: Ipv4Addr) -> bool {
    addr.is_loopback()
        || addr.is_private()
        || addr.is_link_local()
        || addr.is_broadcast()
        || addr.is_multicast()
        || addr.is_unspecified()
        || addr.is_documentation()
        // 100.64.0.0/10, carrier-grade NAT (Ipv4Addr::is_shared is unstable).
        || (addr.octets()[0] == 100 && (addr.octets()[1] & 0b1100_0000) == 64)
        // 192.0.0.0/24, IETF protocol assignments.
        || addr.octets()[..3] == [192, 0, 0]
        // 240.0.0.0/4, reserved.
        || addr.octets()[0] >= 240
}

fn is_blocked_v6(addr: Ipv6Addr) -> bool {
    let segments = addr.segments();
    addr.is_loopback()
        || addr.is_unspecified()
        || addr.is_multicast()
        // fc00::/7, unique local (Ipv6Addr::is_unique_local is unstable).
        || (segments[0] & 0xfe00) == 0xfc00
        // fe80::/10, link-local unicast.
        || (segments[0] & 0xffc0) == 0xfe80
}

/// Parses `raw` and confirms it points somewhere publicly routable.
///
/// # Errors
///
/// Returns [`UrlPolicyError`] if the URL is malformed, uses a scheme other than
/// http/https, has no host, cannot be resolved, or resolves to any address that
/// [`is_blocked`] rejects. A host resolving to several addresses is refused if
/// *any* of them is blocked, so a split-horizon DNS answer cannot smuggle one
/// past.
pub async fn guard_url(raw: &str) -> Result<Url, UrlPolicyError> {
    let url = Url::parse(raw)?;

    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(UrlPolicyError::UnsupportedScheme(scheme.to_string()));
    }

    let host = url.host().ok_or(UrlPolicyError::MissingHost)?;
    let host_str = host.to_string();

    match host {
        Host::Ipv4(addr) => check_addr(&host_str, IpAddr::V4(addr))?,
        Host::Ipv6(addr) => check_addr(&host_str, IpAddr::V6(addr))?,
        Host::Domain(domain) => {
            // Port is irrelevant to the address check but `resolve` needs one.
            let port = url.port_or_known_default().unwrap_or(80);
            let addrs = async_net::resolve((domain, port)).await.map_err(|source| {
                UrlPolicyError::Resolve {
                    host: host_str.clone(),
                    source,
                }
            })?;
            if addrs.is_empty() {
                return Err(UrlPolicyError::Resolve {
                    host: host_str.clone(),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "host resolved to no addresses",
                    ),
                });
            }
            for addr in addrs {
                check_addr(&host_str, addr.ip())?;
            }
        }
    }

    Ok(url)
}

fn check_addr(host: &str, addr: IpAddr) -> Result<(), UrlPolicyError> {
    if is_blocked(addr) {
        return Err(UrlPolicyError::BlockedAddress {
            host: host.to_string(),
            addr,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{UrlPolicyError, guard_url, is_blocked};
    use std::net::IpAddr;

    fn ip(s: &str) -> IpAddr {
        s.parse().expect("test address should parse")
    }

    #[test]
    fn blocks_loopback_and_private_ranges() {
        for addr in [
            "127.0.0.1",
            "127.13.2.9",
            "10.0.0.1",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "0.0.0.0",
            "255.255.255.255",
            "100.64.0.1",
            "::1",
            "::",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
        ] {
            assert!(is_blocked(ip(addr)), "{addr} should be blocked");
        }
    }

    #[test]
    fn blocks_the_cloud_metadata_address() {
        assert!(is_blocked(ip("169.254.169.254")));
    }

    #[test]
    fn allows_public_addresses() {
        for addr in ["1.1.1.1", "8.8.8.8", "93.184.216.34", "2606:4700::1111"] {
            assert!(!is_blocked(ip(addr)), "{addr} should be allowed");
        }
    }

    #[tokio::test]
    async fn rejects_non_http_schemes() {
        let err = guard_url("file:///etc/passwd").await.unwrap_err();
        assert!(matches!(err, UrlPolicyError::UnsupportedScheme(_)));
    }

    #[tokio::test]
    async fn rejects_literal_metadata_url_without_dns() {
        let err = guard_url("http://169.254.169.254/latest/meta-data/")
            .await
            .unwrap_err();
        assert!(matches!(err, UrlPolicyError::BlockedAddress { .. }));
    }

    #[tokio::test]
    async fn rejects_localhost_literal() {
        let err = guard_url("http://127.0.0.1:8080/admin").await.unwrap_err();
        assert!(matches!(err, UrlPolicyError::BlockedAddress { .. }));
    }

    #[tokio::test]
    async fn rejects_ipv6_loopback_literal() {
        let err = guard_url("http://[::1]/").await.unwrap_err();
        assert!(matches!(err, UrlPolicyError::BlockedAddress { .. }));
    }
}
