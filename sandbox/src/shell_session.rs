use std::{
    borrow::Cow,
    collections::HashSet,
    future::Future,
    pin::Pin,
    sync::{Arc, OnceLock, RwLock},
};

use aither_core::llm::{Tool, ToolOutput};
use async_channel::{Receiver, Sender};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Outcome of a container execution request.
#[derive(Debug)]
pub enum ContainerExecOutcome {
    Completed(std::process::Output),
    Killed,
}

/// Trait for executing commands inside a container.
///
/// Implementations provide the bridge between aither-sandbox and a specific
/// container runtime (e.g., Docker via bollard). The framework defines this trait;
/// the application (may) provides the implementation.
pub trait ContainerExec: Send + Sync {
    /// Execute a bash script inside the container, returning stdout/stderr and exit code.
    ///
    /// Implementations should send a human-readable message to
    /// `stdin_blocked_notice` when they detect the process is blocked on stdin
    /// (for example via `/proc/<pid>/syscall`), allowing callers to auto-promote
    /// foreground execution into background mode.
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send;
}

pub(crate) trait ContainerExecObject: Send + Sync {
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>>;
}

impl<T: ContainerExec> ContainerExecObject for T {
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_blocked_notice: Option<Sender<String>>,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>> {
        Box::pin(self.exec(
            container_id,
            script,
            working_dir,
            kill_rx,
            stdin_blocked_notice,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ShellBackend {
    Local,
    Container,
    Ssh,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SshServer {
    pub name: String,
    pub target: String,
}

impl SshServer {
    #[must_use]
    pub fn id(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub enum SshRuntimeProfile {
    Leash { binary: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct ShellRuntimeAvailability {
    pub local: bool,
    pub container: bool,
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

pub trait SshSessionAuthorizer: Send + Sync {
    fn authorize_connect(
        &self,
        target: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>>;

    fn authorize_leash_install(
        &self,
        target: &str,
        details: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>>;
}

#[derive(Clone)]
pub struct ShellSessionRegistry {
    availability: Arc<RwLock<ShellRuntimeAvailability>>,
    ssh_servers: Arc<RwLock<Vec<SshServer>>>,
    ssh_authorizer: Arc<RwLock<Option<Arc<dyn SshSessionAuthorizer>>>>,
    /// Container executor — set once, shared across all clones via `OnceLock`.
    container_exec: Arc<OnceLock<Arc<dyn ContainerExecObject>>>,
    /// Default container ID — set once, used by container backend.
    container_id: Arc<OnceLock<String>>,
}

impl std::fmt::Debug for ShellSessionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellSessionRegistry")
            .finish_non_exhaustive()
    }
}

impl ShellSessionRegistry {
    #[must_use]
    pub fn new(availability: ShellRuntimeAvailability) -> Self {
        Self {
            availability: Arc::new(RwLock::new(availability)),
            ssh_servers: Arc::new(RwLock::new(Vec::new())),
            ssh_authorizer: Arc::new(RwLock::new(None)),
            container_exec: Arc::new(OnceLock::new()),
            container_id: Arc::new(OnceLock::new()),
        }
    }

    /// Set the container executor (can only be called once; subsequent calls are no-ops).
    pub fn set_container_exec<T>(&self, exec: Arc<T>)
    where
        T: ContainerExec + 'static,
    {
        let exec: Arc<dyn ContainerExecObject> = exec;
        let _ = self.container_exec.set(exec);
    }

    /// Set the default container ID for container backend commands.
    pub fn set_container_id(&self, id: String) {
        let _ = self.container_id.set(id);
    }

    pub fn set_availability(&self, availability: ShellRuntimeAvailability) -> Result<(), String> {
        *self
            .availability
            .write()
            .map_err(|_| "shell availability lock poisoned".to_string())? = availability;
        Ok(())
    }

    pub fn set_ssh_authorizer(
        &self,
        authorizer: Arc<dyn SshSessionAuthorizer>,
    ) -> Result<(), String> {
        *self
            .ssh_authorizer
            .write()
            .map_err(|_| "ssh authorizer lock poisoned".to_string())? = Some(authorizer);
        Ok(())
    }

    #[must_use]
    pub fn availability(&self) -> ShellRuntimeAvailability {
        self.availability
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Get the container executor (if set).
    pub(crate) fn container_exec(&self) -> Option<Arc<dyn ContainerExecObject>> {
        self.container_exec.get().cloned()
    }

    /// Get the configured container ID (if set).
    #[must_use]
    pub fn container_id(&self) -> Option<String> {
        self.container_id.get().cloned()
    }

    pub fn set_ssh_servers(&self, servers: Vec<SshServer>) -> Result<(), String> {
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

        *self
            .ssh_servers
            .write()
            .map_err(|_| "ssh server lock poisoned".to_string())? = deduped;
        Ok(())
    }

    #[must_use]
    pub fn list_ssh_servers(&self) -> Vec<SshServer> {
        self.ssh_servers
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    pub fn resolve_ssh_server(&self, server_id: &str) -> Result<SshServer, String> {
        let wanted = server_id.trim();
        if wanted.is_empty() {
            return Err("ssh_server_id is required for ssh mode".to_string());
        }
        self.ssh_servers
            .read()
            .map_err(|_| "ssh server lock poisoned".to_string())?
            .iter()
            .find(|s| s.id() == wanted)
            .cloned()
            .ok_or_else(|| format!("unknown ssh_server_id: {wanted}"))
    }

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

    pub fn ensure_ssh_available(&self) -> Result<(), String> {
        if self.availability().ssh {
            Ok(())
        } else {
            Err("ssh backend is not available".to_string())
        }
    }

    pub fn ssh_authorizer(&self) -> Option<Arc<dyn SshSessionAuthorizer>> {
        self.ssh_authorizer.read().ok()?.clone()
    }
}

#[derive(Debug, Clone)]
pub struct ListSshTool {
    registry: ShellSessionRegistry,
}

impl ListSshTool {
    #[must_use]
    pub const fn new(registry: ShellSessionRegistry) -> Self {
        Self { registry }
    }
}

impl Tool for ListSshTool {
    fn name(&self) -> Cow<'static, str> {
        "list_ssh".into()
    }

    type Arguments = ();

    async fn call(&self, _args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let payload = serde_json::json!({
            "servers": self.registry.list_ssh_servers(),
        });
        Ok(ToolOutput::text(payload.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OpenSshArgs {
    pub ssh_server_id: String,
}

#[derive(Debug, Clone)]
pub struct OpenSshTool {
    registry: ShellSessionRegistry,
}

impl OpenSshTool {
    #[must_use]
    pub const fn new(registry: ShellSessionRegistry) -> Self {
        Self { registry }
    }
}

impl Tool for OpenSshTool {
    fn name(&self) -> Cow<'static, str> {
        "open_ssh".into()
    }

    type Arguments = OpenSshArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        self.registry
            .ensure_ssh_available()
            .map_err(anyhow::Error::msg)?;
        let server = self
            .registry
            .resolve_ssh_server(&args.ssh_server_id)
            .map_err(anyhow::Error::msg)?;
        let runtime =
            bootstrap_ssh_runtime(&server.target, &self.registry.ssh_authorizer()).await?;
        let payload = serde_json::json!({
            "ssh_server_id": server.id(),
            "target": server.target,
            "runtime": match runtime {
                SshRuntimeProfile::Leash { .. } => "leash",
            }
        });
        Ok(ToolOutput::text(payload.to_string()))
    }
}

pub async fn bootstrap_ssh_runtime(
    target: &str,
    authorizer: &Option<Arc<dyn SshSessionAuthorizer>>,
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
    if remote.leash_found {
        return Ok(SshRuntimeProfile::Leash {
            binary: remote.leash_path,
        });
    }

    if let Some(auth) = authorizer {
        let details = format!(
            "Remote {} ({}) does not have leash installed.",
            remote.os, remote.arch
        );
        let approve_install = auth
            .authorize_leash_install(target, &details)
            .await
            .map_err(anyhow::Error::msg)?;
        if approve_install && install_leash(target, &remote).await? {
            let verified = detect_remote(target).await?;
            if verified.leash_found {
                return Ok(SshRuntimeProfile::Leash {
                    binary: verified.leash_path,
                });
            }
        }
    }

    Err(anyhow::anyhow!(
        "remote leash runtime unavailable; ssh mode requires leash on the remote host"
    ))
}

struct RemoteInfo {
    os: String,
    arch: String,
    leash_found: bool,
    leash_path: String,
}

async fn detect_remote(target: &str) -> Result<RemoteInfo, anyhow::Error> {
    let probe = "uname -s; uname -m; if command -v leash >/dev/null 2>&1; then command -v leash; elif [ -x \"$HOME/.local/bin/leash\" ]; then printf '%s\\n' \"$HOME/.local/bin/leash\"; else echo __NO_LEASH__; fi";
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

    let leash_line = lines[2].clone();
    let leash_found = leash_line != "__NO_LEASH__";
    if !leash_found {
        return Ok(RemoteInfo {
            os: lines[0].clone(),
            arch: lines[1].clone(),
            leash_found,
            leash_path: String::new(),
        });
    }

    Ok(RemoteInfo {
        os: lines[0].clone(),
        arch: lines[1].clone(),
        leash_found,
        leash_path: leash_line,
    })
}

async fn install_leash(target: &str, remote: &RemoteInfo) -> Result<bool, anyhow::Error> {
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

    let local_leash = find_local_leash().await?;
    if local_leash.is_empty() {
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

    let dest = format!("{target}:~/.local/bin/leash");
    let scp_status = async_process::Command::new("scp")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(&local_leash)
        .arg(&dest)
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("scp leash failed: {e}"))?;
    if !scp_status.success() {
        return Ok(false);
    }

    let verify_status = async_process::Command::new("ssh")
        .arg("-o")
        .arg("BatchMode=yes")
        .arg("-o")
        .arg("ConnectTimeout=10")
        .arg(target)
        .arg("chmod +x ~/.local/bin/leash && ~/.local/bin/leash --version")
        .stdin(async_process::Stdio::null())
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("verify leash failed: {e}"))?;

    Ok(verify_status.success())
}

async fn find_local_leash() -> Result<String, anyhow::Error> {
    let out = async_process::Command::new("sh")
        .arg("-c")
        .arg("command -v leash || true")
        .stdin(async_process::Stdio::null())
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("failed to locate local leash: {e}"))?;
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
    use super::parse_remote_probe_output;

    #[test]
    fn parse_remote_probe_detects_local_bin_leash_path() {
        let stdout = b"Linux\nx86_64\n/home/test/.local/bin/leash\n";
        let remote = parse_remote_probe_output(stdout).expect("probe output should parse");
        assert_eq!(remote.os, "Linux");
        assert_eq!(remote.arch, "x86_64");
        assert!(remote.leash_found);
        assert_eq!(remote.leash_path, "/home/test/.local/bin/leash");
    }

    #[test]
    fn parse_remote_probe_handles_missing_leash() {
        let stdout = b"Linux\naarch64\n__NO_LEASH__\n";
        let remote = parse_remote_probe_output(stdout).expect("probe output should parse");
        assert!(!remote.leash_found);
        assert!(remote.leash_path.is_empty());
    }
}
