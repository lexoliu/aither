use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    future::Future,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, RwLock},
};

use aither_core::llm::{Tool, ToolOutput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::naming::random_word_slug;
use crate::permission::BashMode;

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

#[derive(Debug, Clone)]
pub enum SshRuntimeProfile {
    Leash { binary: String },
    Unsafe,
}

#[derive(Debug, Clone)]
pub struct ShellSession {
    pub id: String,
    pub backend: ShellBackend,
    pub host_runtime: &'static str,
    pub mode: BashMode,
    pub cwd: PathBuf,
    pub ssh_target: Option<String>,
    pub ssh_runtime: Option<SshRuntimeProfile>,
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

    fn authorize_unsafe_fallback(
        &self,
        target: &str,
        reason: &str,
    ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>>;
}

#[derive(Clone)]
pub struct ShellSessionRegistry {
    sessions: Arc<RwLock<HashMap<String, ShellSession>>>,
    availability: Arc<RwLock<ShellRuntimeAvailability>>,
    ssh_servers: Arc<RwLock<Vec<SshServer>>>,
    ssh_authorizer: Arc<RwLock<Option<Arc<dyn SshSessionAuthorizer>>>>,
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
            sessions: Arc::new(RwLock::new(HashMap::new())),
            availability: Arc::new(RwLock::new(availability)),
            ssh_servers: Arc::new(RwLock::new(Vec::new())),
            ssh_authorizer: Arc::new(RwLock::new(None)),
        }
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

    pub fn set_ssh_servers(&self, servers: Vec<SshServer>) -> Result<(), String> {
        let mut seen = HashSet::new();
        let mut deduped = Vec::new();
        for server in servers {
            if seen.insert(server.target.clone()) {
                deduped.push(server);
            }
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

    pub fn open(
        &self,
        backend: ShellBackend,
        mode: BashMode,
        cwd: PathBuf,
        ssh_target: Option<String>,
        ssh_runtime: Option<SshRuntimeProfile>,
    ) -> Result<ShellSession, String> {
        let availability = self
            .availability
            .read()
            .map_err(|_| "shell availability lock poisoned".to_string())?
            .clone();
        let enabled = match backend {
            ShellBackend::Local => availability.local || availability.container,
            ShellBackend::Container => availability.container,
            ShellBackend::Ssh => availability.ssh,
        };
        if !enabled {
            return Err(format!("backend {backend:?} is not available"));
        }

        let ssh_target = if matches!(backend, ShellBackend::Ssh) {
            let target = ssh_target
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    "ssh_target is required for ssh backend; use list_ssh to select one".to_string()
                })?;
            let allowed = self
                .ssh_servers
                .read()
                .map_err(|_| "ssh server lock poisoned".to_string())?
                .iter()
                .any(|server| server.target == target);
            if !allowed {
                return Err(
                    "ssh_target is not preconfigured; model may only connect to list_ssh targets"
                        .to_string(),
                );
            }
            Some(target)
        } else {
            None
        };

        if matches!(backend, ShellBackend::Ssh) && ssh_runtime.is_none() {
            return Err("missing ssh runtime profile".to_string());
        }

        let id = random_word_slug(4);
        let host_runtime = match backend {
            ShellBackend::Local => {
                if availability.container {
                    "container"
                } else {
                    "leash"
                }
            }
            ShellBackend::Container => "container",
            ShellBackend::Ssh => "ssh",
        };
        let session = ShellSession {
            id: id.clone(),
            backend,
            host_runtime,
            mode,
            cwd,
            ssh_target,
            ssh_runtime,
        };
        self.sessions
            .write()
            .map_err(|_| "shell session lock poisoned".to_string())?
            .insert(id, session.clone());
        Ok(session)
    }

    pub fn close(&self, shell_id: &str) -> Result<bool, String> {
        let removed = self
            .sessions
            .write()
            .map_err(|_| "shell session lock poisoned".to_string())?
            .remove(shell_id)
            .is_some();
        Ok(removed)
    }

    #[must_use]
    pub fn get(&self, shell_id: &str) -> Option<ShellSession> {
        self.sessions.read().ok()?.get(shell_id).cloned()
    }

    pub fn close_all(&self) -> Result<Vec<String>, String> {
        let mut sessions = self
            .sessions
            .write()
            .map_err(|_| "shell session lock poisoned".to_string())?;
        let ids = sessions.keys().cloned().collect::<Vec<_>>();
        sessions.clear();
        Ok(ids)
    }

    fn ssh_authorizer(&self) -> Option<Arc<dyn SshSessionAuthorizer>> {
        self.ssh_authorizer.read().ok()?.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OpenShellBackend {
    Local,
    Ssh,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OpenShellArgs {
    #[serde(default)]
    pub backend: Option<OpenShellBackend>,
    #[serde(default)]
    pub mode: BashMode,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub ssh_target: Option<String>,
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

#[derive(Debug, Clone)]
pub struct OpenShellTool {
    registry: ShellSessionRegistry,
    default_cwd: PathBuf,
}

impl OpenShellTool {
    #[must_use]
    pub const fn new(registry: ShellSessionRegistry, default_cwd: PathBuf) -> Self {
        Self {
            registry,
            default_cwd,
        }
    }
}

impl Tool for OpenShellTool {
    fn name(&self) -> Cow<'static, str> {
        "open_shell".into()
    }

    type Arguments = OpenShellArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let requested = args.backend.unwrap_or(OpenShellBackend::Local);
        let availability = self.registry.availability();
        let backend = match requested {
            OpenShellBackend::Local => ShellBackend::Local,
            OpenShellBackend::Ssh => ShellBackend::Ssh,
        };
        if matches!(requested, OpenShellBackend::Ssh) && !availability.ssh {
            return Err(anyhow::anyhow!("backend Ssh is not available"));
        }
        if matches!(requested, OpenShellBackend::Local)
            && !(availability.local || availability.container)
        {
            return Err(anyhow::anyhow!("backend Local is not available"));
        }

        let cwd = args
            .cwd.map_or_else(|| self.default_cwd.clone(), PathBuf::from);

        let (mode, ssh_runtime) = if matches!(requested, OpenShellBackend::Ssh) {
            let target = args
                .ssh_target
                .clone()
                .ok_or_else(|| anyhow::anyhow!("ssh_target is required for ssh backend"))?;
            bootstrap_ssh_runtime(&target, &self.registry.ssh_authorizer(), args.mode).await?
        } else {
            (args.mode, None)
        };

        let session = self
            .registry
            .open(backend, mode, cwd, args.ssh_target, ssh_runtime)
            .map_err(anyhow::Error::msg)?;

        let payload = serde_json::json!({
            "shell_id": session.id,
            "backend": session.backend,
            "mode": session.mode,
            "cwd": session.cwd,
            "ssh_target": session.ssh_target,
            "runtime": match session.ssh_runtime {
                Some(SshRuntimeProfile::Leash { .. }) => "leash",
                Some(SshRuntimeProfile::Unsafe) => "unsafe",
                None => "local",
            },
            "availability": self.registry.availability(),
        });

        Ok(ToolOutput::text(payload.to_string()))
    }
}

async fn bootstrap_ssh_runtime(
    target: &str,
    authorizer: &Option<Arc<dyn SshSessionAuthorizer>>,
    requested_mode: BashMode,
) -> Result<(BashMode, Option<SshRuntimeProfile>), anyhow::Error> {
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
        return Ok((
            requested_mode,
            Some(SshRuntimeProfile::Leash {
                binary: remote.leash_path,
            }),
        ));
    }

    let mut installed = false;
    if let Some(auth) = authorizer {
        let details = format!(
            "Remote {} ({}) does not have leash installed.",
            remote.os, remote.arch
        );
        let approve_install = auth
            .authorize_leash_install(target, &details)
            .await
            .map_err(anyhow::Error::msg)?;
        if approve_install {
            installed = install_leash(target, &remote).await?;
        }
    }

    if installed {
        let verified = detect_remote(target).await?;
        if verified.leash_found {
            return Ok((
                requested_mode,
                Some(SshRuntimeProfile::Leash {
                    binary: verified.leash_path,
                }),
            ));
        }
    }

    let reason = if requested_mode == BashMode::Unsafe {
        "Remote leash is unavailable; session will run unsafe without sandbox isolation."
    } else {
        "Remote leash is unavailable; requested sandboxed/network cannot be enforced remotely."
    };
    if let Some(auth) = authorizer {
        let allow_unsafe = auth
            .authorize_unsafe_fallback(target, reason)
            .await
            .map_err(anyhow::Error::msg)?;
        if !allow_unsafe {
            return Err(anyhow::anyhow!("user denied unsafe fallback"));
        }
    } else {
        return Err(anyhow::anyhow!(
            "ssh session requires interactive approval for unsafe fallback"
        ));
    }

    Ok((BashMode::Unsafe, Some(SshRuntimeProfile::Unsafe)))
}

struct RemoteInfo {
    os: String,
    arch: String,
    leash_found: bool,
    leash_path: String,
}

async fn detect_remote(target: &str) -> Result<RemoteInfo, anyhow::Error> {
    let probe = "uname -s; uname -m; command -v leash >/dev/null 2>&1 && command -v leash || echo __NO_LEASH__";
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

    let lines = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    if lines.len() < 3 {
        return Err(anyhow::anyhow!("ssh probe returned unexpected output"));
    }

    let leash_line = lines[2].clone();
    let leash_found = leash_line != "__NO_LEASH__";
    let leash_path = if leash_found {
        leash_line
    } else {
        "~/.local/bin/leash".to_string()
    };

    Ok(RemoteInfo {
        os: lines[0].clone(),
        arch: lines[1].clone(),
        leash_found,
        leash_path,
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CloseShellArgs {
    pub shell_id: String,
}

#[derive(Debug, Clone)]
pub struct CloseShellTool {
    registry: ShellSessionRegistry,
    jobs: crate::job_registry::JobRegistry,
}

impl CloseShellTool {
    #[must_use]
    pub const fn new(registry: ShellSessionRegistry, jobs: crate::job_registry::JobRegistry) -> Self {
        Self { registry, jobs }
    }
}

impl Tool for CloseShellTool {
    fn name(&self) -> Cow<'static, str> {
        "close_shell".into()
    }

    type Arguments = CloseShellArgs;

    async fn call(&self, args: Self::Arguments) -> aither_core::Result<ToolOutput> {
        let shell_id = args.shell_id;
        let closed = self.registry.close(&shell_id).map_err(anyhow::Error::msg)?;
        let killed_jobs = if closed {
            self.jobs.kill_by_session(&shell_id).await
        } else {
            0
        };
        let payload = serde_json::json!({
            "shell_id": shell_id,
            "closed": closed,
            "killed_jobs": killed_jobs,
        });
        Ok(ToolOutput::text(payload.to_string()))
    }
}
