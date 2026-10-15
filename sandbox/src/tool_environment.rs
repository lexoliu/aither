use std::path::Path;

use futures::{StreamExt, TryStreamExt, stream};

use crate::shell_session::{
    ShellBackend, ShellSessionRegistry, SshRuntimeProfile, SshServer, SshSessionAuthorizer,
    bootstrap_ssh_runtime,
};
use crate::tool_distribution::ToolDistribution;
use crate::tool_runtime::{
    PreparedEnvironment, PreparedTools, ToolCatalog, ToolRuntimeError, ToolTarget,
};
use crate::tool_transport::{ContainerToolTransport, SshToolTransport, probe_tool_environment};

#[derive(Debug, Clone)]
struct PreparedHome {
    backend: ShellBackend,
    container_id: Option<String>,
    tools: PreparedTools,
}

#[derive(Debug, Clone)]
struct PreparedSsh {
    server: SshServer,
    runtime: SshRuntimeProfile,
    tools: PreparedTools,
}

#[derive(Debug, Clone)]
pub struct PreparedToolEnvironments {
    home: Option<PreparedHome>,
    ssh: Vec<PreparedSsh>,
    catalog: ToolCatalog,
}

impl ShellSessionRegistry {
    /// Prepares every configured environment before an agent can use this registry.
    ///
    /// # Errors
    /// Fails on unsupported platforms, missing artifacts, denied SSH provisioning,
    /// integrity failures or inconsistent command versions across environments.
    pub async fn initialize_tools(
        mut self,
        distribution: &ToolDistribution,
        cache_root: &Path,
    ) -> Result<Self, ToolRuntimeError> {
        let availability = self.availability();
        let home = async {
            if availability.container {
                let runtime = self
                    .container_runtime()
                    .cloned()
                    .ok_or(ToolRuntimeError::MissingContainerRuntime)?;
                let transport = ContainerToolTransport::new(runtime.clone());
                let info = probe_tool_environment(&transport).await?;
                let bundle = distribution
                    .load(&info.target, &cache_root.join("artifacts"))
                    .await?;
                let tools = bundle.verify_container_mount(&transport).await?;
                Ok::<_, ToolRuntimeError>(Some(PreparedHome {
                    backend: ShellBackend::Container,
                    container_id: Some(runtime.container_id().into()),
                    tools,
                }))
            } else if availability.local {
                let bundle = distribution
                    .load(&ToolTarget::native()?, &cache_root.join("artifacts"))
                    .await?;
                let tools = bundle.prepare_local(&cache_root.join("runtimes")).await?;
                Ok(Some(PreparedHome {
                    backend: ShellBackend::Local,
                    container_id: None,
                    tools,
                }))
            } else {
                Ok(None)
            }
        };
        let remote = async {
            if !availability.ssh {
                return Ok::<_, ToolRuntimeError>(Vec::new());
            }
            let authorizer = self.ssh_authorizer();
            stream::iter(self.list_ssh_servers())
                .map(|server| {
                    let authorizer = authorizer.clone();
                    async move {
                        let runtime = bootstrap_ssh_runtime(&server.target, &authorizer).await.map_err(|error| ToolRuntimeError::SshBootstrap(error.to_string()))?;
                        let transport = SshToolTransport::new(server.clone());
                        let info = probe_tool_environment(&transport).await?;
                        let bundle = distribution.load(&info.target, &cache_root.join("artifacts")).await?;
                        let tools = bundle.prepare_remote(&transport, &info.cache_root, || async {
                            let authorizer = authorizer.as_ref().ok_or_else(|| ToolRuntimeError::InstallDenied(server.target.clone()))?;
                            let approved = authorizer.authorize_runtime_install(&server.target, "Prepare verified apply_patch, rg, ast-grep and jq in the managed tools directory.").await.map_err(ToolRuntimeError::Authorization)?;
                            if !approved { return Err(ToolRuntimeError::InstallDenied(server.target.clone())); }
                            Ok(())
                        }).await?;
                        Ok(PreparedSsh { server, runtime, tools })
                    }
                })
                .buffer_unordered(4)
                .try_collect::<Vec<_>>()
                .await
        };
        let (home, mut ssh) = futures::try_join!(home, remote)?;
        ssh.sort_by(|left, right| left.server.id().cmp(right.server.id()));
        let catalog = home
            .as_ref()
            .map(|home| home.tools.catalog())
            .or_else(|| ssh.first().map(|ssh| ssh.tools.catalog()))
            .ok_or_else(|| {
                ToolRuntimeError::NotInitialized("no configured execution environment".into())
            })?
            .clone();
        if ssh.iter().any(|ssh| ssh.tools.catalog() != &catalog) {
            return Err(ToolRuntimeError::CatalogMismatch);
        }
        self.prepared_tools = Some(PreparedToolEnvironments { home, ssh, catalog });
        Ok(self)
    }

    /// Prepared environments, provided they are exactly the configured backends.
    ///
    /// Preparation is bound to the home backend, the container it mounted
    /// into and the SSH server set; a registry whose configuration changed
    /// after preparation describes environments nobody prepared.
    fn prepared_environments(&self) -> Result<&PreparedToolEnvironments, ToolRuntimeError> {
        let prepared = self
            .prepared_tools
            .as_ref()
            .ok_or_else(|| ToolRuntimeError::NotInitialized("execution environments".into()))?;
        if prepared.home.as_ref().map(|home| home.backend) != self.resolve_local_backend().ok() {
            return Err(ToolRuntimeError::EnvironmentChanged(
                PreparedEnvironment::HomeBackend,
            ));
        }
        if let Some(container_id) = prepared
            .home
            .as_ref()
            .and_then(|home| home.container_id.as_deref())
            && self
                .container_runtime()
                .map(crate::ContainerShellRuntime::container_id)
                != Some(container_id)
        {
            return Err(ToolRuntimeError::EnvironmentChanged(
                PreparedEnvironment::Container,
            ));
        }
        let mut configured = if self.availability().ssh {
            self.ssh_servers().iter().collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        configured.sort_by(|left, right| left.id().cmp(right.id()));
        if !prepared
            .ssh
            .iter()
            .map(|ready| &ready.server)
            .eq(configured)
        {
            return Err(ToolRuntimeError::EnvironmentChanged(
                PreparedEnvironment::SshServers,
            ));
        }
        Ok(prepared)
    }

    pub(crate) fn home_tools(
        &self,
        backend: ShellBackend,
    ) -> Result<&PreparedTools, ToolRuntimeError> {
        self.prepared_environments()?
            .home
            .as_ref()
            .filter(|home| home.backend == backend)
            .map(|home| &home.tools)
            .ok_or_else(|| ToolRuntimeError::NotInitialized("home environment".into()))
    }

    pub(crate) fn ssh_tools(
        &self,
        server: &SshServer,
    ) -> Result<(&SshRuntimeProfile, &PreparedTools), ToolRuntimeError> {
        self.prepared_environments()?
            .ssh
            .iter()
            .find(|ready| &ready.server == server)
            .map(|ready| (&ready.runtime, &ready.tools))
            .ok_or_else(|| ToolRuntimeError::NotInitialized(server.id().into()))
    }

    /// Catalog derived only from environments that completed initialization.
    ///
    /// # Errors
    /// Rejects an uninitialized registry, or one whose home backend, container
    /// or SSH server set changed after preparation.
    pub fn prepared_catalog(&self) -> Result<&ToolCatalog, ToolRuntimeError> {
        self.prepared_environments()
            .map(|prepared| &prepared.catalog)
    }

    #[cfg(test)]
    pub(crate) fn with_test_tools(self, tools: PreparedTools) -> Self {
        self.with_test_home(ShellBackend::Local, None, tools)
    }

    #[cfg(test)]
    pub(crate) fn with_test_container_tools(
        self,
        tools: PreparedTools,
        container_id: String,
    ) -> Self {
        self.with_test_home(ShellBackend::Container, Some(container_id), tools)
    }

    #[cfg(test)]
    fn with_test_home(
        mut self,
        backend: ShellBackend,
        container_id: Option<String>,
        tools: PreparedTools,
    ) -> Self {
        let catalog = tools.catalog().clone();
        self.prepared_tools = Some(PreparedToolEnvironments {
            home: Some(PreparedHome {
                backend,
                container_id,
                tools,
            }),
            ssh: Vec::new(),
            catalog,
        });
        self
    }

    /// Records `server` as prepared with `tools`, as SSH bootstrap would.
    #[cfg(test)]
    pub(crate) fn with_test_ssh(mut self, server: SshServer, tools: PreparedTools) -> Self {
        let prepared = self
            .prepared_tools
            .as_mut()
            .expect("prepare the home environment before an SSH server");
        prepared.ssh.push(PreparedSsh {
            server,
            runtime: SshRuntimeProfile::Heel {
                binary: "/usr/local/bin/heel".into(),
            },
            tools,
        });
        prepared
            .ssh
            .sort_by(|left, right| left.server.id().cmp(right.server.id()));
        self
    }
}
