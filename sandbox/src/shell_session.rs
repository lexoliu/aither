use std::{collections::HashSet, future::Future, pin::Pin, sync::Arc};

use async_channel::{Receiver, Sender};

use crate::stdin_watch::StdinWatchSignal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Outcome of a container execution request.
#[derive(Debug)]
pub enum ContainerExecOutcome {
    /// The command ran to completion; carries its exit status and output.
    Completed(std::process::Output),
    /// The command was terminated before it finished.
    Killed,
}

/// Trait for executing commands inside a container.
///
/// Implementations provide the bridge between aither-sandbox and a specific
/// container runtime (e.g., Docker via bollard). The framework defines this trait;
/// the application (may) provides the implementation.
pub trait ContainerExec: Send + Sync {
    /// Execute a terminal command payload inside the container, returning stdout/stderr and exit code.
    ///
    /// `stdin_rx` is the command's stdin; a closed channel is EOF. When
    /// `stdin_blocked` is given, implementations send one
    /// [`StdinWatchSignal::Blocked`] each time a process of the command's
    /// tree starts waiting to read the stdin pipe with nothing buffered —
    /// and send again only after the command was fed and blocks anew — or a
    /// single [`StdinWatchSignal::Failed`] when the watch cannot run at all.
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdin_blocked: Option<Sender<StdinWatchSignal>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send;

    /// Executes a command like [`Self::exec`], handing its stdout on chunk by
    /// chunk through `stdout` as it arrives instead of collecting it.
    ///
    /// The returned outcome carries the exit status and stderr; its stdout is
    /// empty. `stdout` is bounded, so a consumer that stops reading holds the
    /// command back. The command is killed, as when `kill_rx` fires, once
    /// `kill_rx` fires or closes, or once `stdout` has no receiver left.
    fn exec_streamed(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send;
}

/// Object-safe twin of [`ContainerExec`], used only through
/// [`AnyContainerExec`].
pub trait ContainerExecImpl: Send + Sync {
    /// Execute a command inside a container through a boxed future.
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdin_blocked: Option<Sender<StdinWatchSignal>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>>;

    /// Execute a command inside a container, streaming its stdout, through a
    /// boxed future.
    fn exec_streamed_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>>;
}

/// Backend-agnostic container executor handle.
///
/// This wraps any [`ContainerExec`] implementation into a cloneable concrete
/// type that can be passed around without exposing runtime-specific concrete
/// executor types.
#[derive(Clone)]
pub struct AnyContainerExec {
    inner: Arc<dyn ContainerExecImpl>,
}

impl std::fmt::Debug for AnyContainerExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyContainerExec").finish_non_exhaustive()
    }
}

impl AnyContainerExec {
    /// Creates a new handle from a concrete container executor implementation.
    #[must_use]
    pub fn new<T>(inner: Arc<T>) -> Self
    where
        T: ContainerExec + 'static,
    {
        Self { inner }
    }
}

impl ContainerExec for AnyContainerExec {
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdin_blocked: Option<Sender<StdinWatchSignal>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let inner = Arc::clone(&self.inner);
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();

        async move {
            inner
                .exec_boxed(
                    &container_id,
                    &script,
                    &working_dir,
                    kill_rx,
                    stdin_rx,
                    stdin_blocked,
                )
                .await
        }
    }

    fn exec_streamed(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let inner = Arc::clone(&self.inner);
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();

        async move {
            inner
                .exec_streamed_boxed(
                    &container_id,
                    &script,
                    &working_dir,
                    kill_rx,
                    stdin_rx,
                    stdout,
                )
                .await
        }
    }
}

/// An address the container IPC bridge may listen on.
///
/// The bridge executes host IPC commands for whoever connects, so it is bound
/// only to an address the container network alone reaches — never the
/// unspecified (`0.0.0.0`, `::`) address, which would expose it on every
/// interface including the LAN, nor a multicast address, which cannot serve a
/// listener. Docker Desktop and `OrbStack` forward `host.docker.internal` to the
/// host's loopback, so the bridge binds `127.0.0.1`; native Linux Docker
/// routes the name to the container network's gateway the host owns (`docker0`
/// or the docker `bridge` network gateway, e.g. `172.17.0.1`), so the bridge
/// binds that address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BridgeAddress(std::net::IpAddr);

/// Why an address cannot bind the container IPC bridge.
#[derive(Debug, thiserror::Error)]
pub enum BridgeAddressError {
    /// The unspecified wildcard address would listen on every interface.
    #[error("the unspecified address {0} would expose the container IPC bridge on every interface")]
    Unspecified(std::net::IpAddr),
    /// A multicast address cannot serve a listener.
    #[error("the multicast address {0} cannot bind the container IPC bridge")]
    Multicast(std::net::IpAddr),
}

impl TryFrom<std::net::IpAddr> for BridgeAddress {
    type Error = BridgeAddressError;

    fn try_from(address: std::net::IpAddr) -> Result<Self, Self::Error> {
        if address.is_unspecified() {
            return Err(BridgeAddressError::Unspecified(address));
        }
        if address.is_multicast() {
            return Err(BridgeAddressError::Multicast(address));
        }
        Ok(Self(address))
    }
}

impl BridgeAddress {
    /// The address the bridge listens on.
    #[must_use]
    pub const fn ip(&self) -> std::net::IpAddr {
        self.0
    }
}

/// How a container reaches the host's IPC bridge.
///
/// `host` is the name the container dials (`host.docker.internal` on Docker —
/// made resolvable through `--add-host=host.docker.internal:host-gateway` —
/// `host.containers.internal` on Podman), and `bind` is the host address the
/// bridge listens on: `127.0.0.1` on Docker Desktop and `OrbStack`, which
/// forward the name to host loopback, or the container network's gateway
/// address on native Linux Docker. The two belong to the same route, and only
/// the caller — which knows the container runtime — can pair them correctly.
#[derive(Clone, Debug)]
pub struct ContainerIpcRoute {
    host: String,
    bind: BridgeAddress,
}

impl ContainerIpcRoute {
    /// Creates the route a container uses to reach the host's IPC bridge.
    ///
    /// # Panics
    /// Panics if `host` is empty after trimming.
    #[must_use]
    pub fn new(host: impl Into<String>, bind: BridgeAddress) -> Self {
        let host = host.into();
        assert!(
            !host.trim().is_empty(),
            "container IPC route requires a non-empty host"
        );
        Self { host, bind }
    }

    /// The name the container dials to reach the bridge.
    #[must_use]
    pub fn host(&self) -> &str {
        &self.host
    }

    /// The host address the bridge listens on.
    #[must_use]
    pub const fn bind(&self) -> BridgeAddress {
        self.bind
    }
}

/// Container runtime session metadata.
#[derive(Clone)]
pub struct ContainerShellRuntime {
    exec: Arc<dyn ContainerExecImpl>,
    container_id: String,
    ipc_route: ContainerIpcRoute,
    /// The host directory the container mounts as its workspace, declared by
    /// whoever mounts it.
    workspace_mount: Option<crate::MountRoot>,
}

impl std::fmt::Debug for ContainerShellRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerShellRuntime")
            .field("container_id", &self.container_id)
            .field("ipc_route", &self.ipc_route)
            .field("workspace_mount", &self.workspace_mount)
            .finish_non_exhaustive()
    }
}

impl ContainerShellRuntime {
    /// Creates a container shell runtime.
    ///
    /// # Panics
    /// Panics if `container_id` is empty after trimming.
    #[must_use]
    pub fn new(
        container_id: impl Into<String>,
        ipc_route: ContainerIpcRoute,
        exec: AnyContainerExec,
    ) -> Self {
        let container_id = container_id.into();
        assert!(
            !container_id.trim().is_empty(),
            "container runtime requires a non-empty container_id"
        );
        Self {
            exec: exec.inner,
            container_id,
            ipc_route,
            workspace_mount: None,
        }
    }

    /// Declares the host directory the container mounts as its workspace and
    /// where it appears inside. A subagent's commands are confined in the
    /// container through it, so a container session without it cannot launch
    /// subagents.
    #[must_use]
    pub fn with_workspace_mount(mut self, mount: crate::MountRoot) -> Self {
        self.workspace_mount = Some(mount);
        self
    }

    /// The workspace mount, when declared.
    #[must_use]
    pub const fn workspace_mount(&self) -> Option<&crate::MountRoot> {
        self.workspace_mount.as_ref()
    }

    #[must_use]
    /// Returns the container id.
    pub fn container_id(&self) -> &str {
        &self.container_id
    }

    /// Returns the route the container uses to reach the host's IPC bridge.
    #[must_use]
    pub const fn ipc_route(&self) -> &ContainerIpcRoute {
        &self.ipc_route
    }

    pub(crate) fn exec(&self) -> Arc<dyn ContainerExecImpl> {
        Arc::clone(&self.exec)
    }
}

impl<T: ContainerExec> ContainerExecImpl for T {
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdin_blocked: Option<Sender<StdinWatchSignal>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>> {
        Box::pin(self.exec(
            container_id,
            script,
            working_dir,
            kill_rx,
            stdin_rx,
            stdin_blocked,
        ))
    }

    fn exec_streamed_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>> {
        Box::pin(self.exec_streamed(container_id, script, working_dir, kill_rx, stdin_rx, stdout))
    }
}

/// Where a shell command is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ShellBackend {
    /// On this machine, inside the local sandbox.
    Local,
    /// Inside a container.
    Container,
    /// On a remote host over SSH.
    Ssh,
}

/// A named SSH destination the agent may run commands on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SshServer {
    /// Short name the agent uses to select this server.
    pub name: String,
    /// SSH target, as passed to `ssh` (for example `user@host`).
    pub target: String,
}

impl SshServer {
    /// The name used to address this server.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.name
    }
}

/// How the sandbox runtime is available on a remote host.
#[derive(Debug, Clone)]
pub enum SshRuntimeProfile {
    /// The `heel` sandbox binary is installed at the given path.
    Heel {
        /// Absolute path to the `heel` binary on the remote host.
        binary: String,
    },
}

/// Which shell backends are usable in the current deployment.
///
/// Reported to the model so it does not offer a backend that is not configured.
#[derive(Debug, Clone, Serialize)]
pub struct ShellRuntimeAvailability {
    /// Whether local execution is available. Always true in practice.
    pub local: bool,
    /// Whether a container runtime has been wired up.
    pub container: bool,
    /// Whether any SSH servers are registered.
    pub ssh: bool,
}

impl Default for ShellRuntimeAvailability {
    fn default() -> Self {
        Self {
            local: true,
            container: false,
            ssh: false,
        }
    }
}

/// Approves remote actions before the sandbox performs them.
///
/// Connecting to a host and installing software on it are both decisions a user
/// should make, so they are routed through this trait rather than assumed.
pub trait SshSessionAuthorizer: Send + Sync {
    /// Asks whether the agent may open an SSH connection to `target`.
    fn authorize_connect(
        &self,
        target: &str,
    ) -> impl Future<Output = Result<bool, SshAuthorizationError>> + Send;

    /// Asks whether the harness may prepare runtime components on `target`.
    ///
    /// `details` describes the remote host and components so the user can judge the request.
    fn authorize_runtime_install(
        &self,
        target: &str,
        details: &str,
    ) -> impl Future<Output = Result<bool, SshAuthorizationError>> + Send;
}

trait SshSessionAuthorizerImpl: Send + Sync {
    fn connect<'a>(
        &'a self,
        target: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, SshAuthorizationError>> + Send + 'a>>;
    fn install<'a>(
        &'a self,
        target: &'a str,
        details: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, SshAuthorizationError>> + Send + 'a>>;
}

impl<T: SshSessionAuthorizer> SshSessionAuthorizerImpl for T {
    fn connect<'a>(
        &'a self,
        target: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, SshAuthorizationError>> + Send + 'a>> {
        Box::pin(self.authorize_connect(target))
    }
    fn install<'a>(
        &'a self,
        target: &'a str,
        details: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, SshAuthorizationError>> + Send + 'a>> {
        Box::pin(self.authorize_runtime_install(target, details))
    }
}

/// Shared authorizer with boxing confined to the private implementation boundary.
#[derive(Clone)]
pub struct AnySshSessionAuthorizer {
    inner: Arc<dyn SshSessionAuthorizerImpl>,
}

impl std::fmt::Debug for AnySshSessionAuthorizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnySshSessionAuthorizer")
            .finish_non_exhaustive()
    }
}

impl AnySshSessionAuthorizer {
    /// Wraps a shared application authorizer.
    #[must_use]
    pub fn new<T: SshSessionAuthorizer + 'static>(authorizer: Arc<T>) -> Self {
        Self { inner: authorizer }
    }
}

impl SshSessionAuthorizer for AnySshSessionAuthorizer {
    async fn authorize_connect(&self, target: &str) -> Result<bool, SshAuthorizationError> {
        self.inner.connect(target).await
    }
    async fn authorize_runtime_install(
        &self,
        target: &str,
        details: &str,
    ) -> Result<bool, SshAuthorizationError> {
        self.inner.install(target, details).await
    }
}

/// Failure to obtain an application-level SSH authorization decision.
#[derive(Debug, thiserror::Error)]
pub enum SshAuthorizationError {
    /// The application's permission prompt failed.
    #[error("SSH permission prompt failed: {0}")]
    Prompt(String),
}

/// Tracks which shell backends are configured and how to reach them.
///
/// Cloning shares the same underlying state, so registering a container or SSH
/// server is visible to every holder.
#[derive(Clone)]
pub struct ShellSessionRegistry {
    availability: ShellRuntimeAvailability,
    ssh_servers: Vec<SshServer>,
    ssh_authorizer: Option<AnySshSessionAuthorizer>,
    container_runtime: Option<ContainerShellRuntime>,
    pub(crate) prepared_tools: Option<crate::tool_environment::PreparedToolEnvironments>,
}

impl std::fmt::Debug for ShellSessionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellSessionRegistry")
            .finish_non_exhaustive()
    }
}

impl ShellSessionRegistry {
    /// Creates a registry advertising the given backends.
    #[must_use]
    pub const fn new(availability: ShellRuntimeAvailability) -> Self {
        Self {
            availability,
            ssh_servers: Vec::new(),
            ssh_authorizer: None,
            container_runtime: None,
            prepared_tools: None,
        }
    }

    /// Replaces runtime availability.
    #[must_use]
    pub const fn with_availability(mut self, availability: ShellRuntimeAvailability) -> Self {
        self.availability = availability;
        self
    }

    /// The currently advertised backends.
    #[must_use]
    pub fn with_ssh_authorizer<T: SshSessionAuthorizer + 'static>(
        mut self,
        authorizer: Arc<T>,
    ) -> Self {
        self.ssh_authorizer = Some(AnySshSessionAuthorizer::new(authorizer));
        self
    }

    /// Sets the container runtime.
    #[must_use]
    pub fn with_container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.container_runtime = Some(runtime);
        self
    }

    #[must_use]
    /// Returns runtime availability.
    pub fn availability(&self) -> ShellRuntimeAvailability {
        self.availability.clone()
    }

    /// Sets SSH server definitions.
    ///
    /// # Errors
    /// Returns an error when a server entry has an empty id/target or duplicates another id.
    pub fn with_ssh_servers(mut self, servers: Vec<SshServer>) -> Result<Self, String> {
        let mut seen = HashSet::new();
        let mut deduped = Vec::new();
        for server in servers {
            let id = server.id().trim().to_string();
            let target = server.target.trim().to_string();
            if id.is_empty() || target.is_empty() {
                return Err("ssh server entries require non-empty name and target".to_string());
            }
            if !seen.insert(id.clone()) {
                return Err(format!("duplicate ssh server id: {id}"));
            }
            deduped.push(SshServer { name: id, target });
        }

        self.ssh_servers = deduped;
        Ok(self)
    }

    /// Every registered SSH server.
    #[must_use]
    /// Lists configured SSH servers.
    pub fn list_ssh_servers(&self) -> Vec<SshServer> {
        self.ssh_servers.clone()
    }

    pub(crate) fn ssh_servers(&self) -> &[SshServer] {
        &self.ssh_servers
    }

    pub(crate) const fn container_runtime(&self) -> Option<&ContainerShellRuntime> {
        self.container_runtime.as_ref()
    }

    /// Returns the default SSH server.
    ///
    /// # Errors
    /// Returns an error when no SSH server is configured or more than one server requires disambiguation.
    pub fn default_ssh_server(&self) -> Result<SshServer, String> {
        match self.ssh_servers.as_slice() {
            [] => Err("no ssh servers are configured".to_string()),
            [server] => Ok(server.clone()),
            _ => Err(
                "ssh_server_id is required because multiple ssh servers are configured".to_string(),
            ),
        }
    }

    /// Looks up a registered server by name.
    ///
    /// # Errors
    ///
    /// Returns an error if `server_id` is blank, names no registered server, or
    /// the server lock was poisoned.
    pub fn resolve_ssh_server(&self, server_id: &str) -> Result<SshServer, String> {
        let wanted = server_id.trim();
        if wanted.is_empty() {
            return Err("ssh_server_id is required for ssh mode".to_string());
        }
        self.ssh_servers
            .iter()
            .find(|s| s.id() == wanted)
            .cloned()
            .ok_or_else(|| format!("unknown ssh_server_id: {wanted}"))
    }

    /// Picks the backend to use for a command that did not name one.
    ///
    /// Prefers a container when one is configured, since it isolates better
    /// than running on the host.
    ///
    /// # Errors
    ///
    /// Returns an error if neither a container nor local execution is available.
    pub fn resolve_local_backend(&self) -> Result<ShellBackend, String> {
        let availability = self.availability();
        if availability.container {
            return Ok(ShellBackend::Container);
        }
        if availability.local {
            return Ok(ShellBackend::Local);
        }
        Err("no local backend available".to_string())
    }

    /// Confirms the SSH backend is configured.
    ///
    /// # Errors
    ///
    /// Returns an error if no SSH servers are registered.
    pub fn ensure_ssh_available(&self) -> Result<(), String> {
        if self.availability().ssh {
            Ok(())
        } else {
            Err("ssh backend is not available".to_string())
        }
    }

    #[must_use]
    /// Returns the SSH authorizer when configured.
    pub fn ssh_authorizer(&self) -> Option<AnySshSessionAuthorizer> {
        self.ssh_authorizer.clone()
    }
}

/// Prepares a remote host to run sandboxed commands.
///
/// Confirms the connection is authorized, probes for the `heel` runtime, and —
/// with the user's approval — installs it if it is missing.
///
/// # Errors
///
/// Returns an error if the authorizer denies the connection, the SSH probe
/// fails, or `heel` is absent and either installation was declined or did not
/// succeed.
pub async fn bootstrap_ssh_runtime(
    target: &str,
    authorizer: &Option<AnySshSessionAuthorizer>,
) -> Result<SshRuntimeProfile, anyhow::Error> {
    if let Some(auth) = authorizer {
        let allowed = auth
            .authorize_connect(target)
            .await
            .map_err(anyhow::Error::msg)?;
        if !allowed {
            return Err(anyhow::anyhow!("user denied ssh connection authorization"));
        }
    }

    let remote = detect_remote(target).await?;
    if remote.heel_found {
        return Ok(SshRuntimeProfile::Heel {
            binary: remote.heel_path,
        });
    }

    if let Some(auth) = authorizer {
        let details = format!(
            "Remote {} ({}) does not have heel installed.",
            remote.os, remote.arch
        );
        let approve_install = auth
            .authorize_runtime_install(target, &details)
            .await
            .map_err(anyhow::Error::msg)?;
        if approve_install && install_heel(target, &remote).await? {
            let verified = detect_remote(target).await?;
            if verified.heel_found {
                return Ok(SshRuntimeProfile::Heel {
                    binary: verified.heel_path,
                });
            }
        }
    }

    Err(anyhow::anyhow!(
        "remote heel runtime unavailable; ssh mode requires heel on the remote host"
    ))
}

struct RemoteInfo {
    os: String,
    arch: String,
    heel_found: bool,
    heel_path: String,
}

async fn detect_remote(target: &str) -> Result<RemoteInfo, anyhow::Error> {
    let probe = "uname -s; uname -m; if command -v heel >/dev/null 2>&1; then command -v heel; elif [ -x \"$HOME/.local/bin/heel\" ]; then printf '%s\\n' \"$HOME/.local/bin/heel\"; else echo __NO_HEEL__; fi";
    let output = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg(probe)
        .stdin(async_process::Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("ssh probe failed: {e}"))?;

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "ssh probe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    parse_remote_probe_output(&output.stdout)
}

fn parse_remote_probe_output(stdout: &[u8]) -> Result<RemoteInfo, anyhow::Error> {
    let lines = String::from_utf8_lossy(stdout)
        .lines()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    if lines.len() < 3 {
        return Err(anyhow::anyhow!("ssh probe returned unexpected output"));
    }

    let heel_line = lines[2].clone();
    let heel_found = heel_line != "__NO_HEEL__";
    if !heel_found {
        return Ok(RemoteInfo {
            os: lines[0].clone(),
            arch: lines[1].clone(),
            heel_found,
            heel_path: String::new(),
        });
    }

    Ok(RemoteInfo {
        os: lines[0].clone(),
        arch: lines[1].clone(),
        heel_found,
        heel_path: heel_line,
    })
}

async fn install_heel(target: &str, remote: &RemoteInfo) -> Result<bool, anyhow::Error> {
    let local_os = std::env::consts::OS;
    let remote_os = remote.os.to_lowercase();
    let os_match = (local_os == "macos" && remote_os.contains("darwin"))
        || (local_os == "linux" && remote_os.contains("linux"));
    if !os_match {
        return Ok(false);
    }

    let local_arch = std::env::consts::ARCH;
    if normalize_arch(local_arch) != normalize_arch(&remote.arch) {
        return Ok(false);
    }

    let local_heel = find_local_heel().await?;
    if local_heel.is_empty() {
        return Ok(false);
    }

    let mkdir_status = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg("mkdir -p ~/.local/bin")
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("ssh mkdir failed: {e}"))?;
    if !mkdir_status.success() {
        return Ok(false);
    }

    let dest = format!("{target}:~/.local/bin/heel");
    let scp_status = async_process::Command::new("scp")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(&local_heel)
        .arg(&dest)
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("scp heel failed: {e}"))?;
    if !scp_status.success() {
        return Ok(false);
    }

    let verify_status = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg("chmod +x ~/.local/bin/heel && ~/.local/bin/heel --version")
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("verify heel failed: {e}"))?;

    Ok(verify_status.success())
}

async fn find_local_heel() -> Result<String, anyhow::Error> {
    let out = async_process::Command::new("sh")
        .arg("-c")
        .arg("command -v heel || true")
        .stdin(async_process::Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("failed to locate local heel: {e}"))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn normalize_arch(raw: &str) -> &str {
    match raw {
        "x86_64" | "amd64" => "x86_64",
        "aarch64" | "arm64" => "arm64",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ShellRuntimeAvailability, ShellSessionRegistry, SshServer, parse_remote_probe_output,
    };

    #[test]
    fn parse_remote_probe_detects_local_bin_heel_path() {
        let stdout = b"Linux\nx86_64\n/home/test/.local/bin/heel\n";
        let remote = parse_remote_probe_output(stdout).expect("probe output should parse");
        assert_eq!(remote.os, "Linux");
        assert_eq!(remote.arch, "x86_64");
        assert!(remote.heel_found);
        assert_eq!(remote.heel_path, "/home/test/.local/bin/heel");
    }

    #[test]
    fn parse_remote_probe_handles_missing_heel() {
        let stdout = b"Linux\naarch64\n__NO_HEEL__\n";
        let remote = parse_remote_probe_output(stdout).expect("probe output should parse");
        assert!(!remote.heel_found);
        assert_eq!(remote.heel_path, "");
    }

    #[test]
    fn default_ssh_server_uses_single_configured_target() {
        let registry = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: false,
            ssh: true,
        })
        .with_ssh_servers(vec![SshServer {
            name: "prod".to_string(),
            target: "root@example.com".to_string(),
        }])
        .expect("ssh server should configure");

        let server = registry
            .default_ssh_server()
            .expect("single ssh server should become default");
        assert_eq!(server.id(), "prod");
        assert_eq!(server.target, "root@example.com");
    }

    #[test]
    fn default_ssh_server_requires_disambiguation_when_multiple_exist() {
        let registry = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: false,
            ssh: true,
        })
        .with_ssh_servers(vec![
            SshServer {
                name: "prod".to_string(),
                target: "root@example.com".to_string(),
            },
            SshServer {
                name: "staging".to_string(),
                target: "root@staging.example.com".to_string(),
            },
        ])
        .expect("ssh servers should configure");

        let error = registry
            .default_ssh_server()
            .expect_err("multiple ssh servers must require explicit selection");
        assert!(
            error.contains("ssh_server_id is required"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn bridge_address_rejects_the_unspecified_and_multicast_addresses() {
        use super::{BridgeAddress, BridgeAddressError};
        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

        for address in [
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            IpAddr::V6(Ipv6Addr::UNSPECIFIED),
        ] {
            assert!(
                matches!(
                    BridgeAddress::try_from(address),
                    Err(BridgeAddressError::Unspecified(_))
                ),
                "{address} must be rejected as unspecified"
            );
        }
        for address in [
            IpAddr::V4(Ipv4Addr::new(224, 0, 0, 1)),
            IpAddr::V6(Ipv6Addr::new(0xff02, 0, 0, 0, 0, 0, 0, 1)),
        ] {
            assert!(
                matches!(
                    BridgeAddress::try_from(address),
                    Err(BridgeAddressError::Multicast(_))
                ),
                "{address} must be rejected as multicast"
            );
        }
        for address in [
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V4(Ipv4Addr::new(172, 17, 0, 1)),
        ] {
            assert_eq!(
                BridgeAddress::try_from(address)
                    .expect("a concrete host address should be accepted")
                    .ip(),
                address
            );
        }
    }
}
