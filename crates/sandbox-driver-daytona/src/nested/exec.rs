//! Execute Docker CLI commands through Daytona's byte-preserving exec facet.

use std::fmt::Write as _;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::{io, mem};

use async_trait::async_trait;
use sandbox_driver::{
    BASH_ENV_VAR, Error, Exec, ExecControls, ExecResult, ExecSpec, ExecStreamingResult,
    OutputSink, Pty, PtyOptions, PtySession, PtySize, Result, SpawnSpec, StdioProcess,
    StdioProcessHandle, StopLevel, Termination, drain_with_stops, run_with_stop_grace,
};
use sandbox_driver_docker::DockerExec;
use tokio::runtime::Handle;
use tokio::sync::Mutex;
use tokio::time;
use tokio_util::sync::CancellationToken;

use super::{CONTAINER_NAME, DockerCli};
use crate::shell::{exec_line, shell_quote};

const STOP_TIMEOUT: Duration = Duration::from_secs(10);
const TERM_GRACE: Duration = Duration::from_secs(2);
const DRAIN_GRACE: Duration = Duration::from_secs(10);
/// One inner process group. Native session deletion stops the Docker client;
/// this second command tells the existing in-container watcher to stop the job.
struct Job {
    cli:       Arc<DockerCli>,
    stop_file: String,
    pid_file:  String,
    finished:  AtomicBool,
}

impl Job {
    fn new(cli: Arc<DockerCli>) -> Self {
        let nonce: u128 = rand::random();
        let prefix = format!("/tmp/.sandbox-driver/daytona-{nonce:032x}");
        Self {
            cli,
            stop_file: format!("{prefix}.stop"),
            pid_file: format!("{prefix}.pid"),
            finished: AtomicBool::new(false),
        }
    }

    /// `forward_stdin` keeps the wrapper's descriptor 0 open for the command:
    /// fixed bytes ride the spec, streamed stdin rides the transport call.
    fn command(&self, spec: &ExecSpec, forward_stdin: bool) -> ExecSpec {
        let mut args = vec!["exec".to_owned(), "-i".to_owned(), "--workdir".to_owned()];
        args.push(self.cli.resolve(spec.working_dir.as_deref().unwrap_or(".")));
        args.extend(["--env".to_owned(), format!("{BASH_ENV_VAR}=")]);
        args.push(CONTAINER_NAME.to_owned());
        args.extend([
            "/bin/sh".to_owned(),
            "-c".to_owned(),
            DockerExec::command_wrapper(&self.stop_file, &self.pid_file, forward_stdin),
            "sandbox-driver".to_owned(),
        ]);
        args.extend(
            spec.launch_env()
                .iter()
                .map(|(key, value)| format!("{key}={value}")),
        );
        args.push(spec.program.clone());
        args.extend(spec.args.iter().cloned());
        let mut command = DockerCli::command(args).no_timeout();
        command.stdin.clone_from(&spec.stdin);
        command.output_sanitization = spec.output_sanitization;
        command
    }

    async fn stop(&self, mode: StopLevel) -> Result<()> {
        if self.finished.load(Ordering::Acquire) {
            return Ok(());
        }
        stop_job(&self.cli, &self.stop_file, mode).await
    }
}

async fn stop_job(cli: &DockerCli, stop_file: &str, mode: StopLevel) -> Result<()> {
    let script = stop_script(stop_file, mode);
    let spec = DockerCli::command(vec![
        "exec".to_owned(),
        "--workdir".to_owned(),
        "/".to_owned(),
        "--env".to_owned(),
        format!("{BASH_ENV_VAR}="),
        CONTAINER_NAME.to_owned(),
        "/bin/sh".to_owned(),
        "-c".to_owned(),
        script,
    ]);
    time::timeout(STOP_TIMEOUT, cli.run_spec(&spec))
        .await
        .map_err(|_| {
            Error::io(
                "stopping nested job",
                io::Error::other("stop request timed out"),
            )
        })??;
    Ok(())
}

fn stop_script(stop_file: &str, mode: StopLevel) -> String {
    let mode = match mode {
        StopLevel::Term => "term",
        StopLevel::Kill => "kill",
    };
    let request = format!("{stop_file}.request-{:032x}", rand::random::<u128>());
    // A cancelled TERM RPC can still finish after Drop sent KILL. Publish a
    // complete mode atomically; TERM only creates and can never replace KILL.
    format!(
        "request={}; stop_file={}; mkdir -p \"${{stop_file%/*}}\" || exit; \
         trap 'rm -f \"$request\"' EXIT; printf '%s' {} > \"$request\" || exit; \
         if [ {} = kill ]; then mv -f \"$request\" \"$stop_file\"; \
         else ln \"$request\" \"$stop_file\" 2>/dev/null || [ -e \"$stop_file\" ]; fi",
        shell_quote(&request),
        shell_quote(stop_file),
        shell_quote(mode),
        shell_quote(mode)
    )
}

impl Drop for Job {
    fn drop(&mut self) {
        if self.finished.load(Ordering::Acquire) {
            return;
        }
        if let Ok(runtime) = Handle::try_current() {
            let cli = Arc::clone(&self.cli);
            let stop_file = self.stop_file.clone();
            runtime.spawn(async move {
                if let Err(error) = stop_job(&cli, &stop_file, StopLevel::Kill).await {
                    tracing::warn!(error = %error, "abandoned nested job stop failed");
                }
            });
        }
    }
}

pub(super) struct NestedExec {
    cli: Arc<DockerCli>,
}

impl NestedExec {
    pub(super) fn new(cli: Arc<DockerCli>) -> Self {
        Self { cli }
    }
}

#[async_trait]
impl Exec for NestedExec {
    async fn run(&self, spec: &ExecSpec) -> Result<ExecResult> {
        self.run_streaming(spec, ExecControls::buffered())
            .await?
            .into_complete()
    }

    async fn run_streaming(
        &self,
        spec: &ExecSpec,
        controls: ExecControls,
    ) -> Result<ExecStreamingResult> {
        run_with_stop_grace(spec, controls, |spec, controls| async move {
            self.run_signals(&spec, controls).await
        })
        .await
    }

    async fn spawn_stdio(&self, spec: &SpawnSpec) -> Result<StdioProcess> {
        self.spawn_stdio_raw(spec).await
    }
}

impl NestedExec {
    /// Runs the command under raw stop signals; the trait method wraps
    /// this in the spec's stop grace. Streamed stdin is handed to the
    /// native transport, which feeds `docker exec -i`.
    async fn run_signals(
        &self,
        spec: &ExecSpec,
        controls: ExecControls,
    ) -> Result<ExecStreamingResult> {
        let stopped = if controls
            .kill
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            Some(Termination::Killed)
        } else if controls
            .term
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            Some(Termination::Cancelled)
        } else if spec.timeout == Some(Duration::ZERO) {
            Some(Termination::TimedOut)
        } else {
            None
        };
        if let Some(termination) = stopped {
            return Ok(ExecStreamingResult::new(ExecResult::new(
                termination,
                None,
                Duration::ZERO,
            )));
        }
        let job = Job::new(Arc::clone(&self.cli));
        let command = job.command(spec, spec.stdin.is_some() || controls.stdin.is_some());
        let sink_failed = CancellationToken::new();
        let sink = controls.sink.as_ref().map(|sink| {
            let sink = Arc::clone(sink);
            let failed = sink_failed.clone();
            let sink: OutputSink = Arc::new(move |stream, bytes| {
                let sink = Arc::clone(&sink);
                let failed = failed.clone();
                Box::pin(async move {
                    let result = sink(stream, bytes).await;
                    if result.is_err() {
                        failed.cancel();
                    }
                    result
                })
            });
            sink
        });
        let transport_kill = CancellationToken::new();
        let mut run = pin!(self.cli.exec.run_streaming(&command, ExecControls {
            sink,
            kill: Some(transport_kill.clone()),
            // The native session transport streams stdin to `docker exec -i`.
            stdin: controls.stdin.clone(),
            retained_output_limit: controls.retained_output_limit,
            ..ExecControls::default()
        }));
        let outcome = drain_with_stops(
            run.as_mut(),
            &sink_failed,
            &controls,
            spec.timeout,
            DRAIN_GRACE,
            |level| job.stop(level),
        )
        .await?;
        let mut result = if let Some(result) = outcome.drained {
            result?
        } else {
            // The job was killed and its output still has not closed:
            // end the native transport and give it one more bounded wait.
            transport_kill.cancel();
            time::timeout(STOP_TIMEOUT * 3, &mut run)
                .await
                .map_err(|_| {
                    Error::io(
                        "draining nested job output",
                        io::Error::other("output did not close after the job was killed"),
                    )
                })??
        };
        // A transport that ended without the job exiting leaves the job
        // running unless a kill already went out.
        if result.result.termination != Termination::Exited
            && outcome.stop_sent != Some(StopLevel::Kill)
        {
            job.stop(StopLevel::Kill).await?;
        }
        job.finished.store(true, Ordering::Release);
        if outcome.termination != Termination::Exited {
            result.result.termination = outcome.termination;
        }
        Ok(result)
    }

    async fn spawn_stdio_raw(&self, spec: &SpawnSpec) -> Result<StdioProcess> {
        let job = Job::new(Arc::clone(&self.cli));
        let mut command = ExecSpec::new(&spec.program).args(spec.args.clone());
        command.env = spec.launch_env().into_owned();
        command.working_dir.clone_from(&spec.working_dir);
        // The wrapper must preserve the interactive stdin descriptor.
        command.stdin = Some(Vec::new());
        let command = job.command(&command, true);
        let mut spawn = SpawnSpec::new(command.program).args(command.args);
        spawn.env = command.env;
        spawn.working_dir = command.working_dir;
        let mut process = self.cli.exec.spawn_stdio(&spawn).await?;
        process.handle = Box::new(NestedStdioHandle {
            inner: process.handle,
            job,
        });
        Ok(process)
    }
}

struct NestedStdioHandle {
    inner: Box<dyn StdioProcessHandle>,
    job:   Job,
}

#[async_trait]
impl StdioProcessHandle for NestedStdioHandle {
    async fn terminate(&self) {
        if let Err(error) = self.job.stop(StopLevel::Term).await {
            tracing::warn!(error = %error, "nested stdio TERM failed");
        }
        let stopped = if matches!(
            time::timeout(TERM_GRACE, self.inner.wait()).await,
            Ok((Termination::Exited, _))
        ) {
            true
        } else {
            match self.job.stop(StopLevel::Kill).await {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(error = %error, "nested stdio KILL failed");
                    false
                }
            }
        };
        self.inner.terminate().await;
        self.job.finished.store(stopped, Ordering::Release);
    }

    async fn wait(&self) -> (Termination, Option<i32>) {
        let result = self.inner.wait().await;
        if result.0 != Termination::Exited {
            if let Err(error) = self.job.stop(StopLevel::Kill).await {
                tracing::warn!(error = %error, "nested stdio stop after transport failure failed");
                return (Termination::Unknown, None);
            }
        }
        self.job.finished.store(true, Ordering::Release);
        result
    }
}

#[async_trait]
impl Pty for NestedExec {
    async fn open(&self, options: &PtyOptions) -> Result<Box<dyn PtySession>> {
        let nonce: u128 = rand::random();
        let nonce = format!("{nonce:032x}");
        let pid_file = format!("/tmp/.sandbox-driver/pty-{nonce}.pid");
        let mut environment = options.env.clone();
        environment
            .entry("TERM".to_owned())
            .or_insert_with(|| "xterm-256color".to_owned());
        environment
            .entry("LANG".to_owned())
            .or_insert_with(|| "C.UTF-8".to_owned());
        environment.insert(BASH_ENV_VAR.to_owned(), String::new());
        let mut native_options = PtyOptions::default();
        native_options.size = options.size;
        native_options.working_dir = Some("/".to_owned());
        let command = DockerCli::command(vec![
            "exec".to_owned(),
            "-it".to_owned(),
            "--workdir".to_owned(),
            self.cli
                .resolve(options.working_dir.as_deref().unwrap_or(".")),
        ]);
        let mut bootstrap = exec_line(&command.env, &command.program, &command.args);
        // The PTY echoes its bootstrap before stty can disable echo. Forward
        // values through private environment names, so echoed source contains
        // only keys and references, including for Docker's own environment.
        for (index, (key, value)) in environment.into_iter().enumerate() {
            let alias = format!("SANDBOX_DRIVER_PTY_ENV_{index}");
            native_options.env.insert(alias.clone(), value);
            write!(
                bootstrap,
                " --env {}\"${alias}\"",
                shell_quote(&format!("{key}="))
            )
            .expect("writing a String cannot fail");
        }
        let terminal = format!(
            "mkdir -p /tmp/.sandbox-driver; printf '%s' $$ > {}; \
             printf '\\n%s%s\\n' sandbox-driver-ready- {}; exec sh -l",
            shell_quote(&pid_file),
            shell_quote(&nonce)
        );
        for arg in [CONTAINER_NAME, "/bin/sh", "-c", &terminal] {
            bootstrap.push(' ');
            bootstrap.push_str(&shell_quote(arg));
        }
        let native: Arc<dyn PtySession> = Arc::from(self.cli.pty.open(&native_options).await?);
        let session = NestedPtySession {
            cli: Arc::clone(&self.cli),
            native,
            pid_file,
            pending: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
        };
        session
            .native
            .write_input(format!("stty -echo; {bootstrap}\n").as_bytes())
            .await?;
        let marker = format!("sandbox-driver-ready-{nonce}").into_bytes();
        let ready = async {
            let mut output = Vec::new();
            while let Some(bytes) = session.native.read_output().await? {
                output.extend(bytes);
                if let Some(index) = output
                    .windows(marker.len())
                    .position(|window| window == marker)
                {
                    session
                        .pending
                        .lock()
                        .await
                        .extend_from_slice(&output[index + marker.len()..]);
                    return Ok(());
                }
                // Keep only the suffix needed to match a split marker.
                if output.len() > marker.len() {
                    output.drain(..output.len() - marker.len());
                }
            }
            Err(Error::io(
                "starting nested terminal",
                io::Error::other("terminal ended before startup"),
            ))
        };
        match time::timeout(Duration::from_secs(30), ready).await {
            Ok(Ok(())) => Ok(Box::new(session)),
            result => {
                let _ = session.close().await;
                match result {
                    Ok(Err(error)) => Err(error),
                    Err(_) => Err(Error::io(
                        "starting nested terminal",
                        io::Error::other("terminal startup timed out"),
                    )),
                    Ok(Ok(())) => unreachable!("successful startup returned above"),
                }
            }
        }
    }
}

struct NestedPtySession {
    cli:      Arc<DockerCli>,
    native:   Arc<dyn PtySession>,
    pid_file: String,
    pending:  Mutex<Vec<u8>>,
    closed:   AtomicBool,
}

async fn close_terminal(cli: &DockerCli, pid_file: &str) -> Result<()> {
    let pid_file = shell_quote(pid_file);
    let command = format!(
        "if [ -f {pid_file} ]; then pid=$(cat {pid_file}); rm -f {pid_file}; \
         case \"$pid\" in ''|*[!0-9]*) : ;; *) \
         kill -TERM \"-$pid\" 2>/dev/null || kill -TERM \"$pid\" 2>/dev/null || true; \
         for _ in 1 2 3 4 5; do kill -0 \"$pid\" 2>/dev/null || break; sleep 0.2; done; \
         if kill -0 \"$pid\" 2>/dev/null; then kill -KILL \"-$pid\" 2>/dev/null || true; \
         kill -KILL \"$pid\" 2>/dev/null || true; fi ;; esac; fi"
    );
    let spec = DockerCli::command(vec![
        "exec".to_owned(),
        "--user".to_owned(),
        "0".to_owned(),
        "--workdir".to_owned(),
        "/".to_owned(),
        CONTAINER_NAME.to_owned(),
        "/bin/sh".to_owned(),
        "-c".to_owned(),
        command,
    ])
    .timeout(STOP_TIMEOUT);
    time::timeout(STOP_TIMEOUT, cli.run_spec(&spec))
        .await
        .map_err(|_| {
            Error::io(
                "closing nested terminal",
                io::Error::other("terminal stop timed out"),
            )
        })??;
    Ok(())
}

#[async_trait]
impl PtySession for NestedPtySession {
    async fn write_input(&self, bytes: &[u8]) -> Result<()> {
        self.native.write_input(bytes).await
    }

    async fn read_output(&self) -> Result<Option<Vec<u8>>> {
        let mut pending = self.pending.lock().await;
        if !pending.is_empty() {
            return Ok(Some(mem::take(&mut *pending)));
        }
        drop(pending);
        self.native.read_output().await
    }

    async fn resize(&self, size: PtySize) -> Result<()> {
        self.native.resize(size).await
    }

    async fn close(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Ok(());
        }
        let stopped = close_terminal(&self.cli, &self.pid_file).await;
        let closed = self.native.close().await;
        self.closed.store(true, Ordering::Release);
        stopped.and(closed)
    }
}

impl Drop for NestedPtySession {
    fn drop(&mut self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        if let Ok(runtime) = Handle::try_current() {
            let cli = Arc::clone(&self.cli);
            let native = Arc::clone(&self.native);
            let pid_file = self.pid_file.clone();
            runtime.spawn(async move {
                let _ = close_terminal(&cli, &pid_file).await;
                let _ = native.close().await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::sync::Mutex as StdMutex;
    use std::{env, fs};

    use sandbox_driver::StdinSource;

    use super::*;
    use crate::{DaytonaConfig, DaytonaProvider};

    #[test]
    fn a_late_term_request_cannot_downgrade_a_kill() {
        let dir = env::temp_dir().join(format!("sandbox-stop-{:032x}", rand::random::<u128>()));
        fs::create_dir(&dir).expect("control directory");
        let stop_file = dir.join("job.stop");
        for mode in [StopLevel::Term, StopLevel::Kill, StopLevel::Term] {
            assert!(
                Command::new("/bin/sh")
                    .args([
                        "-c",
                        &stop_script(stop_file.to_str().expect("control path"), mode)
                    ])
                    .status()
                    .expect("stop request command")
                    .success()
            );
        }
        assert_eq!(
            fs::read_to_string(&stop_file).expect("published mode"),
            "kill"
        );
        assert_eq!(fs::read_dir(&dir).expect("control files").count(), 1);
        fs::remove_dir_all(dir).expect("remove test control directory");
    }

    /// Records the Docker commands the nested layer sends over the native
    /// transport, and whether each carried streamed stdin.
    #[derive(Default)]
    struct Recorder {
        calls: StdMutex<Vec<(ExecSpec, bool)>>,
    }

    #[async_trait]
    impl Exec for Recorder {
        async fn run(&self, spec: &ExecSpec) -> Result<ExecResult> {
            self.run_streaming(spec, ExecControls::buffered())
                .await?
                .into_complete()
        }

        async fn run_streaming(
            &self,
            spec: &ExecSpec,
            controls: ExecControls,
        ) -> Result<ExecStreamingResult> {
            self.calls
                .lock()
                .unwrap()
                .push((spec.clone(), controls.stdin.is_some()));
            // Exited at once, so the nested run neither waits nor stops the job.
            Ok(ExecStreamingResult::new(ExecResult::new(
                Termination::Exited,
                Some(0),
                Duration::ZERO,
            )))
        }
    }

    async fn nested_over_recorder() -> (NestedExec, Arc<Recorder>) {
        let provider = DaytonaProvider::connect_with_config(DaytonaConfig {
            api_key: Some("test-key".to_owned()),
            api_url: Some("https://daytona.example/api".to_owned()),
            ..DaytonaConfig::default()
        })
        .await
        .unwrap();
        let recorder = Arc::new(Recorder::default());
        let mut cli = DockerCli::new(&provider.client, "test-sandbox", "/workspace");
        cli.exec = Arc::clone(&recorder) as Arc<dyn Exec>;
        (NestedExec::new(Arc::new(cli)), recorder)
    }

    /// The `/bin/sh -c` script of the one recorded `docker exec`.
    fn wrapper_of(recorder: &Recorder) -> (String, bool) {
        let calls = recorder.calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "one docker exec, no stop request");
        let (spec, streamed) = &calls[0];
        let script = spec
            .args
            .iter()
            .position(|arg| arg == "-c")
            .and_then(|index| spec.args.get(index + 1))
            .expect("wrapper script");
        (script.clone(), *streamed)
    }

    #[tokio::test]
    async fn streamed_stdin_reaches_the_docker_exec_transport() {
        let (nested, recorder) = nested_over_recorder().await;
        let controls = ExecControls {
            stdin: Some(StdinSource::new(io::Cursor::new(b"x".to_vec()))),
            ..ExecControls::buffered()
        };
        nested
            .run_streaming(&ExecSpec::new("cat"), controls)
            .await
            .expect("streamed stdin is supported");
        let (script, streamed) = wrapper_of(&recorder);
        assert!(streamed, "the inner call carries the stdin source");
        assert!(script.contains("exec 3<&0"));
        assert!(!script.contains("< /dev/null"));
    }

    #[tokio::test]
    async fn without_stdin_the_wrapper_reads_dev_null() {
        let (nested, recorder) = nested_over_recorder().await;
        nested
            .run_streaming(&ExecSpec::new("true"), ExecControls::buffered())
            .await
            .expect("plain run");
        let (script, streamed) = wrapper_of(&recorder);
        assert!(!streamed);
        assert!(script.contains("< /dev/null"));
        assert!(!script.contains("exec 3<&0"));
    }
}
