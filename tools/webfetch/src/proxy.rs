//! The headless browser's only path to the network.
//!
//! A [`Proxy`] is a forward proxy on a loopback port, owned by one fetch and
//! living as long as that fetch's browser. The browser is launched with it as
//! its proxy for every destination, loopback included, so every connection it
//! makes is a request here: `CONNECT host:port` for `https` and WebSockets,
//! an absolute-form request (`GET http://host/path`) for plain `http`.
//!
//! For each request the proxy resolves the host once, judges every address it
//! resolved with an [`AddressPolicy`], and connects to one of those judged
//! addresses; it never resolves the host again. A server that answers a lookup
//! with a public address and a later one with a private address (DNS
//! rebinding) therefore gains nothing: there is no later lookup. Each request
//! is resolved on its own, and a connection the browser reuses for several
//! plain-`http` requests never carries one of them to another request's host.
//!
//! A refused request is answered with `403 Forbidden` (`502 Bad Gateway` when
//! the host does not resolve) and reported through [`Proxy::refusals`], so the
//! fetch can say why its navigation failed.

use std::convert::Infallible;
use std::fmt;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;

use http_body_util::{Either, Full};
use hyper::body::{Bytes, Incoming};
use hyper::header::{self, HeaderMap, HeaderName, HeaderValue};
use hyper::http::uri::{self, Scheme};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode, Uri};
use hyper_util::rt::TokioIo;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use tracing::{debug, warn};
use url::{Host, Url};

use crate::guard::{AddressPolicy, Resolver, UrlPolicyError, resolve_checked};

/// The body of every response the proxy sends to the browser: a forwarded
/// upstream body, or one the proxy wrote itself.
type Body = Either<Incoming, Full<Bytes>>;

/// Work a connection hands back to the proxy so it ends with the proxy.
type Work = Pin<Box<dyn Future<Output = ()> + Send>>;

/// The host and port a request is addressed to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Authority {
    host: Host<String>,
    port: u16,
}

impl Authority {
    /// The authority `url` is fetched from, its port defaulted by its scheme.
    #[must_use]
    pub fn of_url(url: &Url) -> Option<Self> {
        Some(Self {
            host: url.host()?.to_owned(),
            port: url.port_or_known_default()?,
        })
    }

    /// Parses a request target's authority; `default_port` stands in when it
    /// names none.
    fn of_request(authority: &uri::Authority, default_port: Option<u16>) -> Option<Self> {
        Some(Self {
            host: Host::parse(authority.host()).ok()?,
            port: authority.port_u16().or(default_port)?,
        })
    }

    fn host(&self) -> Host<&str> {
        match &self.host {
            Host::Domain(domain) => Host::Domain(domain),
            Host::Ipv4(addr) => Host::Ipv4(*addr),
            Host::Ipv6(addr) => Host::Ipv6(*addr),
        }
    }
}

impl fmt::Display for Authority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// A request the proxy refused to connect.
#[derive(Debug)]
pub struct Refusal {
    /// Where the request was addressed.
    pub authority: Authority,
    /// Why it was refused.
    pub reason: UrlPolicyError,
}

/// A loopback forward proxy that connects only to addresses its policy
/// approved. It stops, closing every connection it holds, when dropped.
#[derive(Debug)]
pub struct Proxy {
    addr: SocketAddr,
    refusals: mpsc::UnboundedReceiver<Refusal>,
    task: JoinHandle<()>,
}

impl Proxy {
    /// Listens on an ephemeral loopback port, resolving hosts with `resolver`
    /// and connecting only to addresses `policy` approves.
    ///
    /// # Errors
    ///
    /// Returns the listener's error if no loopback port can be bound.
    pub async fn start<R: Resolver, P: AddressPolicy>(
        resolver: R,
        policy: P,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
        let addr = listener.local_addr()?;
        let (refused, refusals) = mpsc::unbounded_channel();
        let (work, pending) = mpsc::unbounded_channel();
        let gate = Gate {
            resolver,
            policy,
            refused,
            work,
        };
        let task = tokio::spawn(run(listener, gate, pending));
        Ok(Self {
            addr,
            refusals,
            task,
        })
    }

    /// The address the browser reaches the proxy at.
    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The requests refused since the last call.
    pub fn refusals(&mut self) -> Vec<Refusal> {
        std::iter::from_fn(|| self.refusals.try_recv().ok()).collect()
    }
}

impl Drop for Proxy {
    fn drop(&mut self) {
        // Aborting the accept loop drops its task set, which aborts every
        // connection and tunnel the proxy started.
        self.task.abort();
    }
}

/// Accepts connections and runs everything they start, until aborted.
async fn run<R: Resolver, P: AddressPolicy>(
    listener: TcpListener,
    gate: Gate<R, P>,
    mut pending: mpsc::UnboundedReceiver<Work>,
) {
    let mut tasks = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    tasks.spawn(serve(stream, gate.clone()));
                }
                Err(error) => {
                    // The browser then cannot connect at all, so the fetch
                    // fails instead of proceeding unchecked.
                    warn!(%error, "webfetch proxy stopped accepting connections");
                    return;
                }
            },
            // `gate` holds a sender, so the channel never closes here.
            Some(work) = pending.recv() => {
                tasks.spawn(work);
            }
            Some(_) = tasks.join_next(), if !tasks.is_empty() => {}
        }
    }
}

/// Serves one browser connection.
async fn serve<R: Resolver, P: AddressPolicy>(stream: TcpStream, gate: Gate<R, P>) {
    let service = service_fn(move |request| {
        let gate = gate.clone();
        async move { Ok::<_, Infallible>(gate.handle(request).await) }
    });
    if let Err(error) = hyper::server::conn::http1::Builder::new()
        .serve_connection(TokioIo::new(stream), service)
        .with_upgrades()
        .await
    {
        debug!(%error, "webfetch proxy connection ended");
    }
}

/// What every request is judged and connected with.
#[derive(Clone)]
struct Gate<R, P> {
    resolver: R,
    policy: P,
    refused: mpsc::UnboundedSender<Refusal>,
    work: mpsc::UnboundedSender<Work>,
}

impl<R: Resolver, P: AddressPolicy> Gate<R, P> {
    async fn handle(&self, request: Request<Incoming>) -> Response<Body> {
        if request.method() == Method::CONNECT {
            self.tunnel(request).await
        } else {
            self.forward(request).await
        }
    }

    /// Opens a tunnel for `CONNECT host:port`.
    async fn tunnel(&self, request: Request<Incoming>) -> Response<Body> {
        let Some(target) = request
            .uri()
            .authority()
            .and_then(|authority| Authority::of_request(authority, None))
        else {
            return text(StatusCode::BAD_REQUEST, "CONNECT needs a host and a port");
        };
        let upstream = match self.connect(&target).await {
            Ok(upstream) => upstream,
            Err((status, message)) => return text(status, &message),
        };
        let tunnel = async move {
            match hyper::upgrade::on(request).await {
                Ok(upgraded) => {
                    let (mut browser, mut upstream) = (TokioIo::new(upgraded), upstream);
                    if let Err(error) =
                        tokio::io::copy_bidirectional(&mut browser, &mut upstream).await
                    {
                        debug!(%target, %error, "webfetch proxy tunnel ended");
                    }
                }
                Err(error) => debug!(%target, %error, "CONNECT was never upgraded"),
            }
        };
        self.spawn(Box::pin(tunnel));
        Response::new(Either::Right(Full::default()))
    }

    /// Sends an absolute-form `http://` request on to its host.
    async fn forward(&self, request: Request<Incoming>) -> Response<Body> {
        let uri = request.uri();
        if uri.scheme() != Some(&Scheme::HTTP) {
            return text(
                StatusCode::BAD_REQUEST,
                "only CONNECT and absolute http:// requests are proxied",
            );
        }
        let Some(target) = uri
            .authority()
            .and_then(|authority| Authority::of_request(authority, Some(80)))
        else {
            return text(StatusCode::BAD_REQUEST, "the request names no valid host");
        };
        let upstream = match self.connect(&target).await {
            Ok(upstream) => upstream,
            Err((status, message)) => return text(status, &message),
        };

        let (mut parts, body) = request.into_parts();
        parts.uri = parts
            .uri
            .path_and_query()
            .map_or_else(|| Uri::from_static("/"), |path| Uri::from(path.clone()));
        strip_hop_by_hop(&mut parts.headers);

        let (mut sender, connection) =
            match hyper::client::conn::http1::handshake(TokioIo::new(upstream)).await {
                Ok(handshake) => handshake,
                Err(error) => return text(StatusCode::BAD_GATEWAY, &error.to_string()),
            };
        self.spawn(Box::pin(async move {
            if let Err(error) = connection.await {
                debug!(%error, "webfetch proxy upstream connection ended");
            }
        }));
        match sender.send_request(Request::from_parts(parts, body)).await {
            Ok(response) => {
                let (mut parts, body) = response.into_parts();
                strip_hop_by_hop(&mut parts.headers);
                Response::from_parts(parts, Either::Left(body))
            }
            Err(error) => text(StatusCode::BAD_GATEWAY, &error.to_string()),
        }
    }

    /// Resolves `target` once and connects to an address the policy
    /// approved, or answers why it will not.
    async fn connect(&self, target: &Authority) -> Result<TcpStream, (StatusCode, String)> {
        let addrs = match resolve_checked(&self.resolver, &self.policy, &target.host(), target.port)
            .await
        {
            Ok(addrs) => addrs,
            Err(reason) => {
                warn!(%target, %reason, "headless browser request refused");
                let status = match reason {
                    UrlPolicyError::Resolve { .. } => StatusCode::BAD_GATEWAY,
                    _ => StatusCode::FORBIDDEN,
                };
                let message = reason.to_string();
                // The fetch holding the receiver may already be done.
                let _ = self.refused.send(Refusal {
                    authority: target.clone(),
                    reason,
                });
                return Err((status, message));
            }
        };
        let mut failure = None;
        for addr in addrs {
            match TcpStream::connect(addr).await {
                Ok(stream) => return Ok(stream),
                Err(error) => failure = Some(error),
            }
        }
        let failure = failure.map_or_else(String::new, |error| error.to_string());
        Err((StatusCode::BAD_GATEWAY, failure))
    }

    fn spawn(&self, work: Work) {
        // The accept loop holds the receiver for as long as any connection
        // runs, and its tasks die with it.
        let _ = self.work.send(work);
    }
}

/// A response the proxy writes itself.
fn text(status: StatusCode, message: &str) -> Response<Body> {
    let mut response = Response::new(Either::Right(Full::new(Bytes::from(message.to_owned()))));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    response
}

/// Removes the headers that describe one hop rather than the message; hyper
/// frames each hop's body itself.
fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let listed: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in listed {
        headers.remove(name);
    }
    for name in [
        header::CONNECTION,
        HeaderName::from_static("proxy-connection"),
        HeaderName::from_static("keep-alive"),
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
    ] {
        headers.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guard::PublicOnly;
    use crate::guard::testing::{Only, Table};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A loopback server that answers every connection with `body` and counts
    /// the connections it accepted.
    async fn server(body: &'static str) -> (SocketAddr, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let accepted = Arc::new(AtomicUsize::new(0));
        let counted = accepted.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buffer = vec![0; 4096];
                    let _ = stream.read(&mut buffer).await;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        (addr, accepted)
    }

    /// Sends `head` to the proxy and returns everything it answers before
    /// closing or going quiet.
    async fn ask(proxy: &Proxy, head: &str) -> String {
        let mut stream = TcpStream::connect(proxy.addr()).await.unwrap();
        stream.write_all(head.as_bytes()).await.unwrap();
        let mut answer = Vec::new();
        let mut buffer = vec![0; 4096];
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(2), stream.read(&mut buffer))
                .await
            {
                Ok(Ok(0) | Err(_)) | Err(_) => break,
                Ok(Ok(read)) => answer.extend_from_slice(&buffer[..read]),
            }
        }
        String::from_utf8_lossy(&answer).into_owned()
    }

    #[tokio::test]
    async fn connect_to_a_host_resolving_privately_is_refused() {
        let (private, accepted) = server("secret").await;
        let resolver = Table::new(&[("rebind.test", private)]);
        let mut proxy = Proxy::start(resolver.clone(), PublicOnly).await.unwrap();

        let answer = ask(
            &proxy,
            &format!(
                "CONNECT rebind.test:{0} HTTP/1.1\r\nHost: rebind.test:{0}\r\n\r\n",
                private.port()
            ),
        )
        .await;

        assert!(answer.starts_with("HTTP/1.1 403"), "{answer}");
        assert_eq!(accepted.load(Ordering::SeqCst), 0);
        assert_eq!(resolver.asked(), ["rebind.test"]);
        let refusals = proxy.refusals();
        assert_eq!(refusals.len(), 1);
        assert_eq!(
            refusals[0].authority.to_string(),
            format!("rebind.test:{}", private.port())
        );
        assert!(matches!(
            refusals[0].reason,
            UrlPolicyError::BlockedAddress { addr, .. } if addr == private.ip()
        ));
    }

    #[tokio::test]
    async fn plain_http_to_a_host_resolving_privately_is_refused() {
        let (private, accepted) = server("secret").await;
        let resolver = Table::new(&[("rebind.test", private)]);
        let mut proxy = Proxy::start(resolver, PublicOnly).await.unwrap();

        let answer = ask(
            &proxy,
            &format!(
                "GET http://rebind.test:{0}/ HTTP/1.1\r\nHost: rebind.test:{0}\r\n\r\n",
                private.port()
            ),
        )
        .await;

        assert!(answer.starts_with("HTTP/1.1 403"), "{answer}");
        assert!(!answer.contains("secret"), "{answer}");
        assert_eq!(accepted.load(Ordering::SeqCst), 0);
        assert_eq!(proxy.refusals().len(), 1);
    }

    #[tokio::test]
    async fn a_private_literal_is_refused_without_a_lookup() {
        let (private, accepted) = server("secret").await;
        let resolver = Table::default();
        let proxy = Proxy::start(resolver.clone(), PublicOnly).await.unwrap();

        let answer = ask(
            &proxy,
            &format!("CONNECT {private} HTTP/1.1\r\nHost: {private}\r\n\r\n"),
        )
        .await;

        assert!(answer.starts_with("HTTP/1.1 403"), "{answer}");
        assert_eq!(accepted.load(Ordering::SeqCst), 0);
        assert!(resolver.asked().is_empty());
    }

    #[tokio::test]
    async fn an_unresolvable_host_is_reported() {
        let mut proxy = Proxy::start(Table::default(), PublicOnly).await.unwrap();

        let answer = ask(
            &proxy,
            "GET http://missing.test/ HTTP/1.1\r\nHost: missing.test\r\n\r\n",
        )
        .await;

        assert!(answer.starts_with("HTTP/1.1 502"), "{answer}");
        assert!(matches!(
            proxy.refusals()[0].reason,
            UrlPolicyError::Resolve { .. }
        ));
    }

    #[tokio::test]
    async fn an_approved_host_is_reached_once_resolved() {
        let (allowed, accepted) = server("hello").await;
        let resolver = Table::new(&[("allowed.test", allowed)]);
        let mut proxy = Proxy::start(resolver.clone(), Only(allowed)).await.unwrap();

        let answer = ask(
            &proxy,
            &format!(
                "GET http://allowed.test:{0}/page HTTP/1.1\r\nHost: allowed.test:{0}\r\nConnection: close\r\n\r\n",
                allowed.port()
            ),
        )
        .await;

        assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        assert!(answer.ends_with("hello"), "{answer}");
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
        assert_eq!(resolver.asked(), ["allowed.test"]);
        assert!(proxy.refusals().is_empty());
    }

    #[tokio::test]
    async fn an_approved_tunnel_carries_bytes_both_ways() {
        let (allowed, accepted) = server("tunnelled").await;
        let proxy = Proxy::start(Table::default(), Only(allowed)).await.unwrap();

        let answer = ask(
            &proxy,
            &format!(
                "CONNECT {allowed} HTTP/1.1\r\nHost: {allowed}\r\n\r\nGET / HTTP/1.1\r\nHost: x\r\n\r\n"
            ),
        )
        .await;

        assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        assert!(answer.ends_with("tunnelled"), "{answer}");
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn each_request_on_a_kept_alive_connection_is_judged() {
        let (allowed, allowed_accepted) = server("hello").await;
        let (private, private_accepted) = server("secret").await;
        let resolver = Table::new(&[("allowed.test", allowed), ("rebind.test", private)]);
        let proxy = Proxy::start(resolver, Only(allowed)).await.unwrap();

        let answer = ask(
            &proxy,
            &format!(
                "GET http://allowed.test:{0}/ HTTP/1.1\r\nHost: allowed.test:{0}\r\n\r\n\
                 GET http://rebind.test:{1}/ HTTP/1.1\r\nHost: rebind.test:{1}\r\n\r\n",
                allowed.port(),
                private.port()
            ),
        )
        .await;

        assert!(answer.contains("hello"), "{answer}");
        assert!(answer.contains("HTTP/1.1 403"), "{answer}");
        assert!(!answer.contains("secret"), "{answer}");
        assert_eq!(allowed_accepted.load(Ordering::SeqCst), 1);
        assert_eq!(private_accepted.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn hop_by_hop_headers_are_not_forwarded() {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONNECTION, HeaderValue::from_static("x-hop"));
        headers.insert("x-hop", HeaderValue::from_static("1"));
        headers.insert("proxy-connection", HeaderValue::from_static("keep-alive"));
        headers.insert(header::HOST, HeaderValue::from_static("example.com"));
        strip_hop_by_hop(&mut headers);
        assert_eq!(headers.len(), 1);
        assert!(headers.contains_key(header::HOST));
    }
}
