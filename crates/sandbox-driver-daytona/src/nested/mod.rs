//! Nested Docker resources controlled by Docker CLI commands through Daytona.
//!
//! Docker remains private to the sandbox. No Docker API listener or preview
//! connection is created; the sandbox lifecycle owns all nested resources.

mod exec;
mod fs;
mod one_shot;
mod operation;

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use sandbox_driver::{
    Capabilities, DirEntry, Error, Exec, ExecControls, ExecFailure, ExecResult, ExecSpec,
    ExecStreamingResult, FileMetadata, Filesystem, OneShotCaps, Pty, PtyOptions, PtySession,
    Result, SandboxSpec, SpawnSpec, StdioProcess,
};
use sandbox_driver_daytona_config::{DockerExecutionTarget, NestedDockerConfig};
use sandbox_driver_docker_config::{RegistryAuth, Sidecar};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::Mutex;
use tokio::time;

use self::exec::NestedExec;
use self::fs::NestedFs;
use self::operation::{run_command, run_owned};
use crate::sdk::DaytonaClient;
use crate::{DaytonaExec, DaytonaFs, DaytonaPty, RUNTIME_DIRECTORY, resolve_path};

pub(super) const CONTAINER_NAME: &str = "sandbox-driver-workspace";
const NETWORK: &str = "sandbox-driver-services";
const NETWORK_LABEL: &str = "sh.sandbox-driver.network";
const SIDECAR_NETWORK_LABEL: &str = "sh.sandbox-driver.sidecar-network";
const ONE_SHOT_LABEL: &str = "sh.sandbox-driver.one-shot";
const MANAGED_LABEL: &str = "sh.sandbox-driver.managed=true";
const START_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
const DRAIN_GRACE: Duration = Duration::from_secs(10);
const INITIALIZE_RUNTIME_DIRECTORY: &str = r#"
umask 077
parent=${1%/*}
test ! -L "$parent" && test ! -L "$1" &&
    mkdir -p "$1" && chmod 0700 "$parent" "$1"
"#;
const CLEAN_STALE_PREPARATION: &str = r"
import os, re, shutil, sys
try:
    entries = os.scandir(sys.argv[1])
except FileNotFoundError:
    sys.exit(0)
with entries:
    for entry in entries:
        if re.fullmatch('(pull|build)-[0-9a-f]{16}', entry.name) and entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
";

/// Shared native Daytona facets and Docker command construction. Environment
/// hygiene applies to the CLI itself; user environment belongs inside Docker.
pub(super) struct DockerCli {
    pub(super) exec:        Arc<dyn Exec>,
    pub(super) fs:          Arc<dyn Filesystem>,
    pub(super) pty:         Arc<dyn Pty>,
    pub(super) working_dir: String,
}

impl DockerCli {
    fn new(client: &DaytonaClient, sandbox_id: &str, working_dir: &str) -> Self {
        Self {
            exec:        Arc::new(DaytonaExec::new(
                Arc::clone(client),
                sandbox_id.to_owned(),
                working_dir.to_owned(),
            )),
            fs:          Arc::new(DaytonaFs::new(
                Arc::clone(client),
                sandbox_id.to_owned(),
                working_dir.to_owned(),
            )),
            pty:         Arc::new(DaytonaPty::new(
                Arc::clone(client),
                sandbox_id.to_owned(),
                working_dir.to_owned(),
            )),
            working_dir: working_dir.to_owned(),
        }
    }

    pub(super) fn command(args: Vec<String>) -> ExecSpec {
        ExecSpec::new("docker")
            .args(args)
            .working_dir("/")
            .env_var("DOCKER_HOST", "unix:///var/run/docker.sock")
            .env_var("DOCKER_CONTEXT", "")
            .env_var("DOCKER_TLS_VERIFY", "")
            .env_var("DOCKER_CERT_PATH", "")
            .env_var(
                "DOCKER_CONFIG",
                format!("{RUNTIME_DIRECTORY}/docker-config"),
            )
    }

    pub(super) async fn run(&self, args: Vec<String>) -> Result<ExecResult> {
        self.run_spec(&Self::command(args)).await
    }

    pub(super) async fn run_spec(&self, spec: &ExecSpec) -> Result<ExecResult> {
        checked(self.exec.run(spec).await?, "nested Docker command")
    }

    pub(super) fn resolve(&self, path: &str) -> String {
        resolve_path(&self.working_dir, path)
    }

    pub(super) async fn inspect(&self, container: &str) -> Result<Value> {
        let output = self
            .run(words(["inspect", "--type", "container", container]))
            .await?;
        let mut values: Vec<Value> = serde_json::from_slice(&output.stdout)
            .map_err(|error| Error::invalid_spec("docker inspect", error.to_string()))?;
        values
            .pop()
            .ok_or_else(|| Error::invalid_spec("docker inspect", "container was not returned"))
    }

    async fn image_present(&self, image: &str) -> Result<bool> {
        let output = self
            .exec
            .run(&Self::command(words(["image", "inspect", image])))
            .await?;
        if output.success() {
            return Ok(true);
        }
        if output.stdout_lossy().contains("No such image")
            || output.stderr_lossy().contains("No such image")
        {
            Ok(false)
        } else {
            checked(output, "inspecting nested Docker image").map(|_| false)
        }
    }

    async fn pull(
        &self,
        image: &str,
        auth: Option<&RegistryAuth>,
        platform: Option<&str>,
    ) -> Result<()> {
        if self.image_present(image).await? {
            return Ok(());
        }
        let config = format!("{RUNTIME_DIRECTORY}/pull-{:016x}", rand::random::<u64>());
        let cleanup_config = config.clone();
        let has_auth = auth.is_some();
        let auth = auth.cloned();
        let image = image.to_owned();
        let platform = platform.map(str::to_owned);
        let exec = Arc::clone(&self.exec);
        let fs = Arc::clone(&self.fs);
        run_owned(
            move |cancel| async move {
                if let Some(auth) = &auth {
                    let mut args =
                        words(["login", "--username", &auth.username, "--password-stdin"]);
                    if let Some(server) = &auth.server {
                        args.push(server.clone());
                    }
                    checked(
                        run_command(
                            &*exec,
                            &Self::command(args)
                                .env_var("DOCKER_CONFIG", &config)
                                .stdin(auth.password.as_bytes().to_vec()),
                            &cancel,
                        )
                        .await?,
                        "logging into nested Docker registry",
                    )?;
                }
                let mut args = words(["pull"]);
                if let Some(platform) = &platform {
                    args.extend(words(["--platform", platform]));
                }
                args.push(image);
                checked(
                    run_command(
                        &*exec,
                        &Self::command(args).env_var("DOCKER_CONFIG", &config),
                        &cancel,
                    )
                    .await?,
                    "pulling nested Docker image",
                )?;
                Ok(())
            },
            move || async move {
                if has_auth {
                    fs.delete(&cleanup_config, true).await?;
                }
                Ok(())
            },
        )
        .await
    }

    async fn ids(&self, label: &str) -> Result<Vec<String>> {
        let output = self
            .run(words([
                "ps",
                "--all",
                "--quiet",
                "--filter",
                &format!("label={label}"),
            ]))
            .await?;
        Ok(output
            .stdout_lossy()
            .split_whitespace()
            .map(str::to_owned)
            .collect())
    }

    async fn remove(&self, container: &str) -> Result<()> {
        let output = self
            .exec
            .run(&Self::command(words([
                "rm",
                "--force",
                "--volumes",
                container,
            ])))
            .await?;
        if output.success()
            || output.stdout_lossy().contains("No such container")
            || output.stderr_lossy().contains("No such container")
        {
            Ok(())
        } else {
            checked(output, "removing nested Docker container").map(|_| ())
        }
    }

    async fn await_health(&self, container: &str) -> Result<()> {
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        loop {
            let inspect = time::timeout(
                deadline.saturating_duration_since(Instant::now()),
                self.inspect(container),
            )
            .await
            .map_err(|_| {
                Error::invalid_spec("sidecar", format!("{container} health check timed out"))
            })??;
            let state = &inspect["State"];
            let Some(health) = state["Health"]["Status"].as_str() else {
                return Ok(());
            };
            match health {
                "healthy" => return Ok(()),
                "unhealthy" => {
                    return Err(Error::invalid_spec(
                        "sidecar",
                        format!("{container} reported unhealthy"),
                    ));
                }
                _ if state["Running"] == false || Instant::now() >= deadline => {
                    return Err(Error::invalid_spec(
                        "sidecar",
                        format!("{container} failed to become healthy"),
                    ));
                }
                _ => time::sleep(Duration::from_millis(250)).await,
            }
        }
    }
}

fn words<const N: usize>(values: [&str; N]) -> Vec<String> {
    values.into_iter().map(str::to_owned).collect()
}

fn checked(result: ExecResult, operation: &str) -> Result<ExecResult> {
    if result.success() {
        Ok(result)
    } else {
        Err(Error::Exec(
            ExecFailure::new(
                operation,
                result.termination,
                result.exit_code,
                result.stdout,
                result.stderr,
            )
            .with_duration(result.duration),
        ))
    }
}

/// Where this handle stands with the nested daemon. One value under one
/// lock: the lock serializes readiness work, and the value says whether
/// the next `ensure_ready` must sweep what an earlier generation left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Generation {
    /// Never made ready through this handle. Other handles may own active
    /// image preparation, so the first readiness pass sweeps nothing.
    Fresh,
    /// Docker and its services are up for the current sandbox generation.
    Ready,
    /// The sandbox stopped since this handle was last ready. Every
    /// preparation and action from before the stop is stale.
    Restarted,
}

pub(super) struct NestedDocker {
    target: DockerExecutionTarget,
    cli:    Arc<DockerCli>,
    exec:   Arc<NestedExec>,
    fs:     NestedFs,
    state:  Mutex<Generation>,
}

impl NestedDocker {
    pub(super) fn new(
        client: &DaytonaClient,
        sandbox_id: &str,
        working_dir: &str,
        target: DockerExecutionTarget,
    ) -> Self {
        let cli = Arc::new(DockerCli::new(client, sandbox_id, working_dir));
        Self::from_cli(cli, target)
    }

    fn from_cli(cli: Arc<DockerCli>, target: DockerExecutionTarget) -> Self {
        let exec = Arc::new(NestedExec::new(Arc::clone(&cli)));
        let fs = NestedFs::new(Arc::clone(&cli), Arc::clone(&exec) as Arc<dyn Exec>);
        Self {
            target,
            cli,
            exec,
            fs,
            state: Mutex::new(Generation::Fresh),
        }
    }

    pub(super) fn targets_container(&self) -> bool {
        self.target == DockerExecutionTarget::Container
    }
    pub(super) fn target(&self) -> DockerExecutionTarget {
        self.target
    }

    pub(super) fn capabilities(&self, caps: &mut Capabilities) {
        caps.one_shot = Some(one_shot_capabilities());
        if self.targets_container() {
            caps.fs.native = false;
            caps.git.native = false;
            caps.logs = None;
            caps.access.preview_urls = false;
            caps.access.signed_preview_urls = false;
            caps.access.ssh = false;
            caps.access.ssh_ttl = false;
            caps.access.ssh_revoke = false;
            caps.access.web_terminal = false;
            caps.access.vnc = false;
        }
    }

    async fn bootstrap(&self) -> Result<()> {
        let mut spec = DockerCli::command(Vec::new());
        "start-docker".clone_into(&mut spec.program);
        spec.timeout = Some(START_TIMEOUT);
        self.cli.run_spec(&spec).await.map(|_| ())
    }

    async fn clean_stale_preparation(&self) -> Result<()> {
        let spec = ExecSpec::new("python3")
            .args(["-c", CLEAN_STALE_PREPARATION, RUNTIME_DIRECTORY])
            .timeout(DRAIN_GRACE);
        // Only a new generation sweeps. A ready handle can have active image
        // preparation whose credentials and context still belong to its task.
        time::timeout(DRAIN_GRACE * 2, self.cli.run_spec(&spec))
            .await
            .map_err(|_| {
                Error::io(
                    "recovering nested preparation",
                    io::Error::other("cleanup timed out"),
                )
            })??;
        Ok(())
    }

    pub(super) async fn create(
        &self,
        config: &NestedDockerConfig,
        env: &BTreeMap<String, String>,
    ) -> Result<()> {
        if !self.targets_container() {
            return Ok(());
        }
        let mut state = self.state.lock().await;
        self.bootstrap().await?;
        self.clean_stale_preparation().await?;
        if config.options.auto_pull {
            self.cli
                .pull(
                    &config.image,
                    config.options.registry_auth.as_ref(),
                    config.options.platform.as_deref(),
                )
                .await?;
        }
        let services = !config.options.sidecars.is_empty();
        let network = if services { "none" } else { "bridge" };
        let args = primary_args(config, env, &self.cli.working_dir, network);
        self.cli.run(args).await?;
        // The labeled primary exists before any network or sidecar, so a
        // crashed allocation always remains owned by the outer sandbox.
        if services {
            self.cli
                .run(words([
                    "network",
                    "create",
                    "--label",
                    MANAGED_LABEL,
                    NETWORK,
                ]))
                .await?;
            for sidecar in &config.options.sidecars {
                self.cli
                    .pull(&sidecar.image, sidecar.registry_auth.as_ref(), None)
                    .await?;
                let name = format!("{NETWORK}-{}", sidecar.name);
                self.cli.run(sidecar_args(sidecar, &name)).await?;
                self.cli.run(words(["start", &name])).await?;
            }
            for sidecar in &config.options.sidecars {
                if sidecar.health.is_some() {
                    self.cli
                        .await_health(&format!("{NETWORK}-{}", sidecar.name))
                        .await?;
                }
            }
            self.cli
                .run(words(["network", "disconnect", "none", CONTAINER_NAME]))
                .await?;
            self.cli
                .run(words(["network", "connect", NETWORK, CONTAINER_NAME]))
                .await?;
        }
        self.cli.run(words(["start", CONTAINER_NAME])).await?;
        self.initialize_runtime_directory().await?;
        *state = Generation::Ready;
        Ok(())
    }

    pub(super) async fn stopped(&self) {
        *self.state.lock().await = Generation::Restarted;
    }

    /// Brings the nested daemon, and for a container target its primary
    /// container and services, up for the current sandbox generation.
    /// A handle that is already ready does nothing. A handle whose
    /// sandbox restarted first sweeps the stale preparation and action
    /// containers the previous generation left behind.
    pub(super) async fn ensure_ready(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        let sweep = match *state {
            Generation::Ready => return Ok(()),
            Generation::Fresh => false,
            Generation::Restarted => true,
        };
        self.bootstrap().await?;
        if sweep {
            self.clean_stale_preparation().await?;
        }
        if self.targets_container() {
            self.recover_container_target(sweep).await?;
        } else {
            self.recover_vm_target(sweep).await?;
        }
        *state = Generation::Ready;
        Ok(())
    }

    /// The VM target runs actions straight on the daemon; recovery only
    /// removes the actions a previous generation left.
    async fn recover_vm_target(&self, sweep: bool) -> Result<()> {
        if sweep {
            for action in self.cli.ids(ONE_SHOT_LABEL).await? {
                self.cli.remove(&action).await?;
            }
        }
        Ok(())
    }

    /// The container target restarts its primary container, after its
    /// services when it has any, and reclaims the container-local
    /// runtime directory.
    async fn recover_container_target(&self, sweep: bool) -> Result<()> {
        let primary = self.cli.inspect(CONTAINER_NAME).await?;
        let id = primary["Id"]
            .as_str()
            .ok_or_else(|| Error::invalid_spec("docker inspect", "container id missing"))?;
        if sweep {
            for action in self.cli.ids(&format!("{ONE_SHOT_LABEL}={id}")).await? {
                self.cli.remove(&action).await?;
            }
        }
        if let Some(network) = primary["Config"]["Labels"][SIDECAR_NETWORK_LABEL].as_str() {
            self.restart_services(&primary, network).await?;
        }
        self.cli.run(words(["start", CONTAINER_NAME])).await?;
        self.initialize_runtime_directory().await?;
        Ok(())
    }

    /// Starts every service on the primary's sidecar network and waits
    /// for each to be healthy before the primary itself starts.
    async fn restart_services(&self, primary: &Value, network: &str) -> Result<()> {
        if primary["NetworkSettings"]["Networks"][network].is_null() {
            return Err(Error::invalid_spec(
                "nested Docker",
                "service allocation did not complete",
            ));
        }
        let services = self.cli.ids(&format!("{NETWORK_LABEL}={network}")).await?;
        for service in &services {
            self.cli.run(words(["start", service])).await?;
        }
        for service in services {
            self.cli.await_health(&service).await?;
        }
        Ok(())
    }

    async fn initialize_runtime_directory(&self) -> Result<()> {
        // This is container-local scratch space. Binding the outer runtime
        // would expose Docker credentials and preparation files to the job.
        self.cli
            .run(words([
                "exec",
                CONTAINER_NAME,
                "/bin/sh",
                "-c",
                INITIALIZE_RUNTIME_DIRECTORY,
                "sandbox-driver-runtime",
                RUNTIME_DIRECTORY,
            ]))
            .await?;
        Ok(())
    }
}

pub(super) fn one_shot_capabilities() -> OneShotCaps {
    let mut caps = OneShotCaps::default();
    caps.build = true;
    caps
}

fn flags(args: &mut Vec<String>, name: &str, values: impl IntoIterator<Item = String>) {
    for value in values {
        args.extend([name.to_owned(), value]);
    }
}

fn workspace_mount(path: &str) -> String {
    [
        "type=bind".to_owned(),
        format!("source={path}"),
        format!("target={path}"),
    ]
    .map(|field| {
        if field.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field
        }
    })
    .join(",")
}

fn primary_args(
    config: &NestedDockerConfig,
    env: &BTreeMap<String, String>,
    working_dir: &str,
    network: &str,
) -> Vec<String> {
    let mut args = words([
        "create",
        "--name",
        CONTAINER_NAME,
        "--init",
        "--label",
        MANAGED_LABEL,
        "--network",
        network,
        "--workdir",
        working_dir,
    ]);
    args.extend(words(["--mount", &workspace_mount(working_dir)]));
    if !config.options.sidecars.is_empty() {
        args.extend(words([
            "--label",
            &format!("{SIDECAR_NETWORK_LABEL}={NETWORK}"),
        ]));
    }
    if let Some(user) = &config.user {
        args.extend(words(["--user", user]));
    }
    if let Some(platform) = &config.options.platform {
        args.extend(words(["--platform", platform]));
    }
    if config.options.privileged {
        args.push("--privileged".to_owned());
    }
    flags(
        &mut args,
        "--env",
        env.iter().map(|(key, value)| format!("{key}={value}")),
    );
    flags(&mut args, "--add-host", config.options.extra_hosts.clone());
    flags(&mut args, "--dns", config.options.dns.clone());
    flags(&mut args, "--cap-add", config.options.cap_add.clone());
    args.extend(words([
        "--entrypoint",
        "/bin/sh",
        &config.image,
        "-c",
        "trap 'exit 0' TERM INT; while :; do sleep 3600 & wait $!; done",
    ]));
    args
}

fn sidecar_args(sidecar: &Sidecar, name: &str) -> Vec<String> {
    let mut args = words([
        "create",
        "--name",
        name,
        "--network",
        NETWORK,
        "--network-alias",
        &sidecar.name,
        "--label",
        MANAGED_LABEL,
        "--label",
        &format!("{NETWORK_LABEL}={NETWORK}"),
    ]);
    flags(
        &mut args,
        "--env",
        sidecar
            .env
            .iter()
            .map(|(key, value)| format!("{key}={value}")),
    );
    flags(&mut args, "--dns", sidecar.dns.clone());
    flags(&mut args, "--cap-add", sidecar.cap_add.clone());
    if sidecar.privileged {
        args.push("--privileged".to_owned());
    }
    if let Some(user) = &sidecar.user {
        args.extend(words(["--user", user]));
    }
    if let Some(health) = &sidecar.health {
        args.extend(words(["--health-cmd", &health.cmd]));
        for (flag, value) in [
            ("--health-interval", health.interval_ms),
            ("--health-timeout", health.timeout_ms),
            ("--health-start-period", health.start_period_ms),
        ] {
            if let Some(ms) = value {
                args.extend([flag.to_owned(), format!("{ms}ms")]);
            }
        }
        if let Some(retries) = health.retries {
            args.extend(["--health-retries".to_owned(), retries.to_string()]);
        }
    }
    if let Some(entrypoint) = &sidecar.entrypoint {
        args.extend(words([
            "--entrypoint",
            entrypoint.first().map_or("", String::as_str),
        ]));
    }
    args.push(sidecar.image.clone());
    if let Some(entrypoint) = &sidecar.entrypoint {
        args.extend(entrypoint.iter().skip(1).cloned());
    }
    args
}

pub(super) fn validate(config: &NestedDockerConfig, _spec: &SandboxSpec) -> Result<()> {
    if config.target == DockerExecutionTarget::Container && config.image.trim().is_empty() {
        return Err(Error::invalid_spec(
            "provider_config.docker.image",
            "image must not be empty",
        ));
    }
    if !config.options.binds.is_empty() || config.options.host_network {
        return Err(Error::invalid_spec(
            "provider_config.docker.options",
            "nested Docker owns workspace binds and network placement",
        ));
    }
    if config.target == DockerExecutionTarget::VirtualMachine && !config.options.sidecars.is_empty()
    {
        return Err(Error::invalid_spec(
            "provider_config.docker.options.sidecars",
            "services require a container execution target",
        ));
    }
    Ok(())
}

#[async_trait]
impl Exec for NestedDocker {
    async fn run(&self, spec: &ExecSpec) -> Result<ExecResult> {
        self.ensure_ready().await?;
        self.exec.run(spec).await
    }

    async fn run_streaming(
        &self,
        spec: &ExecSpec,
        controls: ExecControls,
    ) -> Result<ExecStreamingResult> {
        self.ensure_ready().await?;
        self.exec.run_streaming(spec, controls).await
    }

    async fn spawn_stdio(&self, spec: &SpawnSpec) -> Result<StdioProcess> {
        self.ensure_ready().await?;
        self.exec.spawn_stdio(spec).await
    }
}

#[async_trait]
impl Pty for NestedDocker {
    async fn open(&self, options: &PtyOptions) -> Result<Box<dyn PtySession>> {
        self.ensure_ready().await?;
        self.exec.open(options).await
    }
}

#[async_trait]
impl Filesystem for NestedDocker {
    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        self.ensure_ready().await?;
        self.fs.read(path).await
    }
    async fn read_to(
        &self,
        path: &str,
        output: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.read_to(path, output).await
    }
    async fn read_range(&self, path: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>> {
        self.ensure_ready().await?;
        self.fs.read_range(path, offset, length).await
    }
    async fn write(&self, path: &str, content: &[u8]) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.write(path, content).await
    }
    async fn write_from(
        &self,
        path: &str,
        input: &mut (dyn AsyncRead + Unpin + Send),
        length: u64,
    ) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.write_from(path, input, length).await
    }
    async fn write_append(&self, path: &str, content: &[u8]) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.write_append(path, content).await
    }
    async fn delete(&self, path: &str, recursive: bool) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.delete(path, recursive).await
    }
    async fn exists(&self, path: &str) -> Result<bool> {
        self.ensure_ready().await?;
        self.fs.exists(path).await
    }
    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.ensure_ready().await?;
        self.fs.metadata(path).await
    }
    async fn list_dir(&self, path: &str, depth: usize) -> Result<Vec<DirEntry>> {
        self.ensure_ready().await?;
        self.fs.list_dir(path, depth).await
    }
    async fn create_dir(&self, path: &str) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.create_dir(path).await
    }
    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.rename(from, to).await
    }
    async fn set_permissions(&self, path: &str, mode: u32) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.set_permissions(path, mode).await
    }
    async fn upload(&self, local: &Path, remote: &str) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.upload(local, remote).await
    }
    async fn download(&self, remote: &str, local: &Path) -> Result<()> {
        self.ensure_ready().await?;
        self.fs.download(remote, local).await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use sandbox_driver::{Capability, OneShot, OneShotSpec, Termination};
    use sandbox_driver_docker_config::DockerProviderConfig;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::provider::daytona_capabilities;
    use crate::{DaytonaConfig, DaytonaProvider};

    #[tokio::test]
    async fn previews_are_only_available_for_the_vm_execution_target() {
        let provider = DaytonaProvider::connect_with_config(DaytonaConfig {
            api_key: Some("test-key".to_owned()),
            api_url: Some("https://daytona.example/api".to_owned()),
            ..DaytonaConfig::default()
        })
        .await
        .unwrap();
        for target in [
            DockerExecutionTarget::Container,
            DockerExecutionTarget::VirtualMachine,
        ] {
            let nested = NestedDocker::new(&provider.client, "test-vm", "/workspace", target);
            let mut caps = daytona_capabilities();
            nested.capabilities(&mut caps);
            let vm = target == DockerExecutionTarget::VirtualMachine;
            assert_eq!(caps.supports(Capability::PreviewUrls), vm);
            assert_eq!(caps.supports(Capability::SignedPreviewUrls), vm);
        }
    }
    #[tokio::test]
    async fn container_target_keeps_streamed_stdin() {
        let nested = fake_nested(|_| response(0, "")).await;
        let mut caps = daytona_capabilities();
        caps.exec.stdin_stream = true;
        nested.capabilities(&mut caps);
        assert!(caps.exec.stdin_stream);
    }

    struct FakeExec {
        reply: Box<dyn Fn(&ExecSpec) -> ExecResult + Send + Sync>,
    }

    #[async_trait]
    impl Exec for FakeExec {
        async fn run(&self, spec: &ExecSpec) -> Result<ExecResult> {
            Ok((self.reply)(spec))
        }
        async fn run_streaming(
            &self,
            spec: &ExecSpec,
            _: ExecControls,
        ) -> Result<ExecStreamingResult> {
            self.run(spec).await.map(ExecStreamingResult::new)
        }
    }

    fn response(code: i32, stdout: impl Into<Vec<u8>>) -> ExecResult {
        let mut result =
            ExecResult::from_shell_status(Termination::Exited, Some(code), Duration::ZERO);
        result.stdout = stdout.into();
        result
    }

    async fn fake_nested(
        reply: impl Fn(&ExecSpec) -> ExecResult + Send + Sync + 'static,
    ) -> NestedDocker {
        let provider = DaytonaProvider::connect_with_config(DaytonaConfig {
            api_key: Some("test-key".to_owned()),
            api_url: Some("https://daytona.example/api".to_owned()),
            ..Default::default()
        })
        .await
        .unwrap();
        let mut cli = DockerCli::new(&provider.client, "test-sandbox", "/workspace");
        cli.exec = Arc::new(FakeExec {
            reply: Box::new(reply),
        });
        NestedDocker::from_cli(Arc::new(cli), DockerExecutionTarget::Container)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn container_runtime_is_private_on_create_and_recovery_and_rejects_symlinks() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};
        use std::process::Command;
        use std::{env, fs};

        let dir = env::temp_dir().join(format!("nested runtime {:032x}", rand::random::<u128>()));
        let parent = dir.join("sandbox-driver");
        let runtime = parent.join("runtime");
        let outside = dir.join("outside");
        fs::create_dir_all(&outside).expect("test directory");
        let test_runtime = runtime.clone();
        let nested = fake_nested(move |spec| {
            if spec.args.starts_with(&words(["inspect"])) {
                return response(0, br#"[{"Id":"main","Config":{"Labels":{}}}]"#.to_vec());
            }
            if spec.args.starts_with(&words(["exec", CONTAINER_NAME])) {
                let output = Command::new(&spec.args[2])
                    .args(&spec.args[3..spec.args.len() - 1])
                    .arg(&test_runtime)
                    .output()
                    .expect("container runtime command");
                return response(output.status.code().expect("shell exited"), output.stdout);
            }
            response(0, Vec::new())
        })
        .await;
        let config = NestedDockerConfig {
            image:   "alpine:3.20".to_owned(),
            target:  DockerExecutionTarget::Container,
            user:    None,
            options: DockerProviderConfig {
                auto_pull: false,
                ..Default::default()
            },
        };
        nested
            .create(&config, &BTreeMap::new())
            .await
            .expect("created");
        for path in [&parent, &runtime] {
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o700
            );
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        nested.stopped().await;
        nested.ensure_ready().await.expect("recovered");
        for path in [&parent, &runtime] {
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        fs::remove_dir(&runtime).unwrap();
        symlink(&outside, &runtime).unwrap();
        nested.stopped().await;
        assert!(nested.ensure_ready().await.is_err());
        assert_eq!(*nested.state.lock().await, Generation::Restarted);
        fs::remove_file(&runtime).unwrap();
        fs::remove_dir(&parent).unwrap();
        symlink(&outside, &parent).unwrap();
        assert!(nested.ensure_ready().await.is_err());
        assert!(!outside.join("runtime").exists());
        fs::remove_dir_all(dir).expect("remove test runtime");
    }

    #[tokio::test]
    async fn generation_recovery_sweeps_stale_preparation_but_preserves_active_work() {
        use std::process::Command;
        use std::{env, fs};

        let dir = env::temp_dir().join(format!("nested-recovery-{:032x}", rand::random::<u128>()));
        fs::create_dir(&dir).expect("private test runtime");
        let stale = dir.join("pull-0123456789abcdef");
        let active = dir.join("build-fedcba9876543210");
        let unrelated = dir.join("pull-user-data");
        fs::create_dir(&stale).expect("stale credential directory");
        fs::create_dir(&unrelated).expect("unrelated directory");
        let runtime = dir.clone();
        let mut nested = fake_nested(move |spec| {
            if spec.program == "python3" {
                assert_eq!(spec.timeout, Some(DRAIN_GRACE));
                let output = Command::new("python3")
                    .args(["-c", CLEAN_STALE_PREPARATION])
                    .arg(&runtime)
                    .output()
                    .expect("recovery cleanup command");
                assert!(output.status.success(), "{:?}", output.stderr);
            }
            response(0, Vec::new())
        })
        .await;
        nested.target = DockerExecutionTarget::VirtualMachine;
        time::timeout(Duration::from_secs(10), async {
            nested.ensure_ready().await.expect("first generation ready");
            assert!(
                stale.exists(),
                "fresh attachment preserves preparation owned by other handles"
            );
            assert!(unrelated.exists(), "only private generated names are swept");
            fs::create_dir(&active).expect("active preparation");
            nested.ensure_ready().await.expect("already ready");
            assert!(active.exists(), "ready handles preserve active preparation");
            nested.stopped().await;
            nested
                .ensure_ready()
                .await
                .expect("restarted generation ready");
            assert!(!active.exists(), "restart treats old preparation as stale");
            assert!(unrelated.exists());
        })
        .await
        .expect("bounded recovery");
        fs::remove_dir_all(dir).expect("remove test runtime");
    }

    #[tokio::test]
    async fn failed_sidecar_creation_leaves_a_labeled_primary_for_outer_cleanup() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let primary = Arc::new(AtomicBool::new(false));
        let marker = Arc::clone(&primary);
        let nested = fake_nested(move |spec| {
            if spec.program == "start-docker" {
                return response(0, Vec::new());
            }
            if spec
                .args
                .starts_with(&words(["create", "--name", CONTAINER_NAME]))
            {
                assert!(
                    spec.args
                        .contains(&format!("{SIDECAR_NETWORK_LABEL}={NETWORK}"))
                );
                marker.store(true, Ordering::SeqCst);
            } else if spec.args.starts_with(&words(["network", "create"])) {
                assert!(
                    marker.load(Ordering::SeqCst),
                    "a primary must own every dependency"
                );
            } else if spec.args.starts_with(&words(["create"])) {
                return response(1, b"injected service creation failure".to_vec());
            }
            response(0, Vec::new())
        })
        .await;
        let config = NestedDockerConfig {
            image:   "alpine:3.20".to_owned(),
            target:  DockerExecutionTarget::Container,
            user:    None,
            options: DockerProviderConfig {
                auto_pull: false,
                sidecars: vec![Sidecar::new("db", "alpine:3.20")],
                ..Default::default()
            },
        };
        let result = nested.create(&config, &BTreeMap::new()).await;
        assert!(result.is_err());
        assert!(primary.load(Ordering::SeqCst));
        assert_eq!(*nested.state.lock().await, Generation::Fresh);
    }

    #[tokio::test]
    async fn restart_sweeps_actions_and_waits_for_services_before_starting_primary() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let removed = Arc::new(AtomicBool::new(false));
        let starts = Arc::new(AtomicUsize::new(0));
        let health = Arc::new(AtomicUsize::new(0));
        let (observed_removed, observed_starts, observed_health) = (
            Arc::clone(&removed),
            Arc::clone(&starts),
            Arc::clone(&health),
        );
        let nested = fake_nested(move |spec| {
            let args = &spec.args;
            if spec.program == "start-docker" { return response(0, Vec::new()); }
            if args.first().is_some_and(|arg| arg == "inspect") {
                if args.last().is_some_and(|arg| arg == CONTAINER_NAME) {
                    return response(0, serde_json::json!([{"Id":"main", "Config":{"Labels":{SIDECAR_NETWORK_LABEL:NETWORK}}, "NetworkSettings":{"Networks":{NETWORK:{}}}}]).to_string().into_bytes());
                }
                assert_eq!(observed_starts.load(Ordering::SeqCst), 2, "start every service before its health wait");
                observed_health.fetch_add(1, Ordering::SeqCst);
                return response(0, b"[{\"State\":{\"Running\":true,\"Health\":{\"Status\":\"healthy\"}}}]".to_vec());
            }
            if args.first().is_some_and(|arg| arg == "ps") {
                return if args.last().is_some_and(|arg| arg.contains(ONE_SHOT_LABEL)) {
                    response(0, b"old-action\n".to_vec())
                } else { response(0, b"service-a\nservice-b\n".to_vec()) };
            }
            if args.first().is_some_and(|arg| arg == "rm") { observed_removed.store(true, Ordering::SeqCst); }
            if args.first().is_some_and(|arg| arg == "start") {
                assert!(observed_removed.load(Ordering::SeqCst));
                if args.last().is_some_and(|arg| arg == CONTAINER_NAME) { assert_eq!(observed_health.load(Ordering::SeqCst), 2); }
                else { observed_starts.fetch_add(1, Ordering::SeqCst); }
            }
            response(0, Vec::new())
        }).await;
        nested.stopped().await;
        nested.ensure_ready().await.unwrap();
        assert!(removed.load(Ordering::SeqCst));
        assert_eq!(health.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cancelled_one_shot_does_not_bootstrap_pull_or_create() {
        let nested = fake_nested(|_| panic!("a cancelled action must do no work")).await;
        let kill = CancellationToken::new();
        kill.cancel();
        let result = OneShot::run(&nested, &OneShotSpec::registry("alpine"), ExecControls {
            kill: Some(kill),
            ..Default::default()
        })
        .await
        .unwrap();
        assert_eq!(result.result.termination, Termination::Killed);
    }
    #[tokio::test]
    async fn process_scope_creation_does_not_start_docker_or_create_a_helper() {
        let mut nested =
            fake_nested(|_| panic!("process scope creation uses only native Daytona")).await;
        nested.target = DockerExecutionTarget::VirtualMachine;
        let config = NestedDockerConfig {
            image:   String::new(),
            target:  DockerExecutionTarget::VirtualMachine,
            user:    None,
            options: DockerProviderConfig::default(),
        };
        nested.create(&config, &BTreeMap::new()).await.unwrap();
        assert_eq!(*nested.state.lock().await, Generation::Fresh);
    }

    #[tokio::test]
    async fn dropping_an_action_joins_a_late_create_before_removing_it() {
        use tokio::sync::Notify;

        struct DelayedCreate {
            creating: Notify,
            finish:   Notify,
            removed:  Notify,
        }

        #[async_trait]
        impl Exec for DelayedCreate {
            async fn run(&self, spec: &ExecSpec) -> Result<ExecResult> {
                match spec.args.first().map(String::as_str) {
                    Some("image") => {}
                    Some("create") => {
                        self.creating.notify_one();
                        self.finish.notified().await;
                    }
                    Some("rm") => self.removed.notify_one(),
                    _ => panic!("a dropped action must never start: {}", spec.program),
                }
                Ok(response(0, Vec::new()))
            }

            async fn run_streaming(
                &self,
                spec: &ExecSpec,
                _: ExecControls,
            ) -> Result<ExecStreamingResult> {
                self.run(spec).await.map(ExecStreamingResult::new)
            }
        }

        let native = fake_nested(|_| unreachable!()).await;
        let exec = Arc::new(DelayedCreate {
            creating: Notify::new(),
            finish:   Notify::new(),
            removed:  Notify::new(),
        });
        let nested = NestedDocker::from_cli(
            Arc::new(DockerCli {
                exec:        exec.clone(),
                fs:          Arc::clone(&native.cli.fs),
                pty:         Arc::clone(&native.cli.pty),
                working_dir: "/workspace".to_owned(),
            }),
            DockerExecutionTarget::VirtualMachine,
        );
        *nested.state.lock().await = Generation::Ready;
        let action = tokio::spawn(async move {
            OneShot::run(
                &nested,
                &OneShotSpec::registry("alpine"),
                ExecControls::buffered(),
            )
            .await
        });
        exec.creating.notified().await;
        action.abort();
        assert!(action.await.unwrap_err().is_cancelled());
        assert!(
            time::timeout(Duration::from_millis(20), exec.removed.notified())
                .await
                .is_err(),
            "cleanup must wait until the accepted create has finished"
        );
        exec.finish.notify_one();
        time::timeout(Duration::from_secs(1), exec.removed.notified())
            .await
            .expect("the abandoned action is removed after late creation");
    }

    /// The shared drain race asks the action to stop through `docker
    /// kill`: a term is one TERM and the attach keeps draining; the kill
    /// that follows is a KILL, and the run reports it.
    #[tokio::test]
    async fn a_termed_then_killed_action_is_signalled_in_order_and_reports_the_kill() {
        use std::sync::Mutex as StdMutex;

        use tokio::sync::Notify;

        struct SignalledAttach {
            signals:  StdMutex<Vec<String>>,
            attached: Notify,
            killed:   Notify,
        }

        #[async_trait]
        impl Exec for SignalledAttach {
            async fn run(&self, spec: &ExecSpec) -> Result<ExecResult> {
                if spec.args.first().is_some_and(|arg| arg == "kill") {
                    let signal = spec.args[2].clone();
                    let is_kill = signal == "KILL";
                    self.signals.lock().unwrap().push(signal);
                    if is_kill {
                        self.killed.notify_one();
                    }
                }
                Ok(response(0, Vec::new()))
            }

            async fn run_streaming(
                &self,
                spec: &ExecSpec,
                _: ExecControls,
            ) -> Result<ExecStreamingResult> {
                if spec.args.starts_with(&words(["start", "--attach"])) {
                    // The attach ends when the container dies, which only
                    // the KILL does here.
                    self.attached.notify_one();
                    self.killed.notified().await;
                    return Ok(ExecStreamingResult::new(response(137, Vec::new())));
                }
                self.run(spec).await.map(ExecStreamingResult::new)
            }
        }

        let native = fake_nested(|_| unreachable!()).await;
        let exec = Arc::new(SignalledAttach {
            signals:  StdMutex::new(Vec::new()),
            attached: Notify::new(),
            killed:   Notify::new(),
        });
        let nested = NestedDocker::from_cli(
            Arc::new(DockerCli {
                exec:        exec.clone(),
                fs:          Arc::clone(&native.cli.fs),
                pty:         Arc::clone(&native.cli.pty),
                working_dir: "/workspace".to_owned(),
            }),
            DockerExecutionTarget::VirtualMachine,
        );
        *nested.state.lock().await = Generation::Ready;
        let term = CancellationToken::new();
        let kill = CancellationToken::new();
        let controls = ExecControls {
            term: Some(term.clone()),
            kill: Some(kill.clone()),
            ..ExecControls::buffered()
        };
        let action = tokio::spawn(async move {
            OneShot::run(&nested, &OneShotSpec::registry("alpine"), controls).await
        });
        // A stop before the attach ends the run without a signal; the
        // race under test starts with the attach.
        time::timeout(Duration::from_secs(5), exec.attached.notified())
            .await
            .expect("the action attaches");
        term.cancel();
        time::timeout(Duration::from_secs(5), async {
            while exec.signals.lock().unwrap().is_empty() {
                time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the term reaches the action");
        assert_eq!(*exec.signals.lock().unwrap(), ["TERM"]);
        kill.cancel();
        let result = time::timeout(Duration::from_secs(5), action)
            .await
            .expect("the kill ends the run")
            .expect("the run task")
            .expect("the run");
        assert_eq!(result.result.termination, Termination::Killed);
        assert_eq!(*exec.signals.lock().unwrap(), ["TERM", "KILL"]);
    }
}
