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
//! reaches the network only through a forward proxy of its own (the `proxy`
//! module, behind the `headless` feature) that resolves each host once,
//! judges every address it resolved by this same policy, and connects to a
//! judged address only; the browser never resolves a destination itself.

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

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
    // Every scheme left has a default port.
    let port = url.port_or_known_default().unwrap_or(80);
    resolve_checked(&SystemResolver, &PublicOnly, &host, port).await?;

    Ok(url)
}

/// Resolves host names to socket addresses.
pub trait Resolver: Clone + Send + Sync + 'static {
    /// Every address `host` answers to on `port`.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when `host` cannot be resolved.
    fn resolve(
        &self,
        host: &str,
        port: u16,
    ) -> impl Future<Output = std::io::Result<Vec<SocketAddr>>> + Send;
}

/// The operating system's resolver.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemResolver;

impl Resolver for SystemResolver {
    async fn resolve(&self, host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
        async_net::resolve((host, port)).await
    }
}

/// Decides which addresses a fetch may connect to.
pub trait AddressPolicy: Clone + Send + Sync + 'static {
    /// Judges `addr`, which `host` resolved to.
    ///
    /// # Errors
    ///
    /// Returns why nothing may connect to `addr`.
    fn check(&self, host: &str, addr: SocketAddr) -> Result<(), UrlPolicyError>;
}

/// The policy of every fetch: only publicly routable addresses, as
/// [`is_blocked`] decides.
#[derive(Debug, Clone, Copy, Default)]
pub struct PublicOnly;

impl AddressPolicy for PublicOnly {
    fn check(&self, host: &str, addr: SocketAddr) -> Result<(), UrlPolicyError> {
        if is_blocked(addr.ip()) {
            return Err(UrlPolicyError::BlockedAddress {
                host: host.to_string(),
                addr: addr.ip(),
            });
        }
        Ok(())
    }
}

/// Resolves `host` once and returns the addresses to connect to, every one of
/// them approved by `policy`.
///
/// An IP literal is judged as it is, without consulting `resolver`. A host is
/// refused if *any* address it resolved to is refused, so a split-horizon
/// answer cannot smuggle one past.
///
/// # Errors
///
/// Returns [`UrlPolicyError::Resolve`] when `host` resolves to nothing, and
/// the policy's refusal of the first address it refuses.
pub async fn resolve_checked<R: Resolver, P: AddressPolicy>(
    resolver: &R,
    policy: &P,
    host: &Host<&str>,
    port: u16,
) -> Result<Vec<SocketAddr>, UrlPolicyError> {
    let host_str = host.to_string();
    let addrs =
        match *host {
            Host::Ipv4(addr) => vec![SocketAddr::new(IpAddr::V4(addr), port)],
            Host::Ipv6(addr) => vec![SocketAddr::new(IpAddr::V6(addr), port)],
            Host::Domain(domain) => {
                let addrs = resolver.resolve(domain, port).await.map_err(|source| {
                    UrlPolicyError::Resolve {
                        host: host_str.clone(),
                        source,
                    }
                })?;
                if addrs.is_empty() {
                    return Err(UrlPolicyError::Resolve {
                        host: host_str,
                        source: std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "host resolved to no addresses",
                        ),
                    });
                }
                addrs
            }
        };
    for addr in &addrs {
        policy.check(&host_str, *addr)?;
    }
    Ok(addrs)
}

/// A resolver and a policy tests control.
#[cfg(test)]
pub(crate) mod testing {
    use super::{AddressPolicy, Resolver, UrlPolicyError};
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    /// Answers lookups from a fixed table and records them; a host listed
    /// several times answers with every address listed for it.
    #[derive(Clone, Default)]
    pub(crate) struct Table {
        hosts: Arc<HashMap<String, Vec<SocketAddr>>>,
        asked: Arc<Mutex<Vec<String>>>,
    }

    impl Table {
        pub(crate) fn new(entries: &[(&str, SocketAddr)]) -> Self {
            let mut hosts: HashMap<String, Vec<SocketAddr>> = HashMap::new();
            for (host, addr) in entries {
                hosts.entry((*host).to_string()).or_default().push(*addr);
            }
            Self {
                hosts: Arc::new(hosts),
                asked: Arc::default(),
            }
        }

        /// Every host the table was asked for, in order.
        pub(crate) fn asked(&self) -> Vec<String> {
            self.asked.lock().expect("lookup log poisoned").clone()
        }

        /// How many times the table was asked for `host`.
        pub(crate) fn lookups(&self, host: &str) -> usize {
            self.asked().iter().filter(|asked| *asked == host).count()
        }
    }

    impl Resolver for Table {
        async fn resolve(&self, host: &str, _port: u16) -> std::io::Result<Vec<SocketAddr>> {
            self.asked
                .lock()
                .expect("lookup log poisoned")
                .push(host.to_string());
            self.hosts.get(host).cloned().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "not in the table")
            })
        }
    }

    /// Allows exactly one address, which stands in for a public one.
    #[derive(Clone)]
    pub(crate) struct Only(pub(crate) SocketAddr);

    impl AddressPolicy for Only {
        fn check(&self, host: &str, addr: SocketAddr) -> Result<(), UrlPolicyError> {
            if addr == self.0 {
                Ok(())
            } else {
                Err(UrlPolicyError::BlockedAddress {
                    host: host.to_string(),
                    addr: addr.ip(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Table;
    use super::{
        AddressPolicy, PublicOnly, UrlPolicyError, guard_url, is_blocked, resolve_checked,
    };
    use std::net::{IpAddr, SocketAddr};
    use url::Host;

    fn sock(s: &str) -> SocketAddr {
        s.parse().expect("test socket address should parse")
    }

    #[tokio::test]
    async fn a_host_is_refused_when_any_of_its_addresses_is() {
        let resolver = Table::new(&[
            ("split.test", sock("93.184.216.34:80")),
            ("split.test", sock("10.0.0.5:80")),
        ]);
        let err = resolve_checked(&resolver, &PublicOnly, &Host::Domain("split.test"), 80)
            .await
            .unwrap_err();
        assert!(
            matches!(err, UrlPolicyError::BlockedAddress { addr, .. } if addr == ip("10.0.0.5"))
        );
    }

    #[tokio::test]
    async fn the_checked_addresses_are_the_ones_returned() {
        let resolver = Table::new(&[("ok.test", sock("93.184.216.34:443"))]);
        let addrs = resolve_checked(&resolver, &PublicOnly, &Host::Domain("ok.test"), 443)
            .await
            .unwrap();
        assert_eq!(addrs, vec![sock("93.184.216.34:443")]);
        assert_eq!(resolver.lookups("ok.test"), 1);
    }

    #[tokio::test]
    async fn an_unknown_host_is_a_resolve_error() {
        let err = resolve_checked(&Table::default(), &PublicOnly, &Host::Domain("x.test"), 80)
            .await
            .unwrap_err();
        assert!(matches!(err, UrlPolicyError::Resolve { .. }));
    }

    #[test]
    fn the_public_policy_judges_by_ip() {
        assert!(PublicOnly.check("a", sock("127.0.0.1:443")).is_err());
        assert!(PublicOnly.check("a", sock("1.1.1.1:443")).is_ok());
    }

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
