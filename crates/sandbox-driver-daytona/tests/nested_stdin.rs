//! Live gate: streamed stdin into a nested Docker container target.
//! Requires DAYTONA_API_KEY and SANDBOX_DRIVER_DAYTONA_DIND_SNAPSHOT; the
//! sandbox it creates is deleted on success and failure paths.

use std::env;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sandbox_driver::{
    ExecControls, ExecSpec, OutputSink, OutputStream, Sandbox, SandboxKind, SandboxProvider,
    SandboxSource, SandboxSpec, SnapshotId, StdinSource,
};
use sandbox_driver_daytona::{
    DaytonaProvider, DaytonaProviderConfig, DockerExecutionTarget, NestedDockerConfig,
};
use support::init_diagnostics;
use tokio::io::{AsyncWriteExt as _, duplex};
use tokio::sync::Notify;
use tokio::time;

mod support;

fn patterned_bytes(len: usize) -> Vec<u8> {
    let mut state: u32 = 0x9e37_79b9;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state.to_be_bytes()[0]
        })
        .collect()
}

/// ACP-shaped: each request waits for its reply, then EOF ends the loop.
async fn turn_by_turn(sandbox: &dyn Sandbox) -> Result<(), String> {
    const TURNS: usize = 12;
    let stdout = Arc::new(Mutex::new(Vec::<u8>::new()));
    let replied = Arc::new(Notify::new());
    let sink: OutputSink = Arc::new({
        let stdout = Arc::clone(&stdout);
        let replied = Arc::clone(&replied);
        move |stream, bytes| {
            if stream == OutputStream::Stdout {
                stdout.lock().expect("stdout").extend_from_slice(&bytes);
                replied.notify_one();
            }
            Box::pin(async { Ok(()) })
        }
    });
    let (mut requests, input) = duplex(4096);
    let controls = ExecControls {
        stdin: Some(StdinSource::new(input)),
        sink: Some(sink),
        ..ExecControls::buffered()
    };
    let spec = ExecSpec::new("/bin/sh")
        .args([
            "-c",
            r#"while IFS= read -r line; do printf 'ack:%s\n' "$line"; done; echo done"#,
        ])
        .timeout(Duration::from_secs(180));
    let run = sandbox.exec().run_streaming(&spec, controls);
    let turns = async {
        let started = Instant::now();
        for turn in 0..TURNS {
            let request = format!("{{\"id\":{turn},\"text\":\"héllo 世界\"}}\n");
            requests
                .write_all(request.as_bytes())
                .await
                .map_err(|error| format!("write turn {turn}: {error}"))?;
            let expected = format!("ack:{}", request.trim_end());
            time::timeout(Duration::from_secs(30), async {
                while !String::from_utf8_lossy(&stdout.lock().expect("stdout")).contains(&expected)
                {
                    replied.notified().await;
                }
            })
            .await
            .map_err(|_| format!("no reply to turn {turn}"))?;
        }
        eprintln!(
            "nested streamed stdin: {TURNS} turns, {:?} per turn",
            started.elapsed() / u32::try_from(TURNS).expect("turns fit")
        );
        drop(requests);
        Ok::<_, String>(())
    };
    let (result, turns) = tokio::join!(run, turns);
    // The run's own error explains a missing reply, so it goes first.
    let result = result.map_err(|error| format!("turn-by-turn exec: {error:?}"))?;
    turns?;
    if !result.result.success() || !result.result.stdout_lossy().ends_with("done\n") {
        return Err(format!(
            "EOF did not end the loop cleanly: {:?} {:?} {:?}",
            result.result.termination,
            result.result.stdout_lossy(),
            result.result.stderr_lossy()
        ));
    }
    Ok(())
}

/// A mebibyte of every byte value arrives in order and ends at EOF.
async fn exact_bytes(sandbox: &dyn Sandbox) -> Result<(), String> {
    let bytes = patterned_bytes(1 << 20);
    sandbox
        .fs()
        .write("expected.bin", &bytes)
        .await
        .map_err(|error| format!("write expected bytes: {error}"))?;
    let controls = ExecControls {
        stdin: Some(StdinSource::new(Cursor::new(bytes))),
        ..ExecControls::buffered()
    };
    let spec = ExecSpec::new("cmp")
        .args(["-", "expected.bin"])
        .timeout(Duration::from_secs(300));
    let started = Instant::now();
    let result = sandbox
        .exec()
        .run_streaming(&spec, controls)
        .await
        .map_err(|error| format!("exact-bytes exec: {error}"))?;
    eprintln!("nested streamed stdin: 1 MiB in {:?}", started.elapsed());
    if !result.result.success() {
        return Err(format!(
            "streamed bytes differ: {:?} {}{}",
            result.result.exit_code,
            result.result.stdout_lossy(),
            result.result.stderr_lossy()
        ));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires DAYTONA_API_KEY and SANDBOX_DRIVER_DAYTONA_DIND_SNAPSHOT"]
async fn nested_container_streams_stdin_and_ends_at_eof() {
    init_diagnostics();
    env::var("DAYTONA_API_KEY").expect("live gate requires DAYTONA_API_KEY");
    let snapshot = env::var("SANDBOX_DRIVER_DAYTONA_DIND_SNAPSHOT")
        .expect("set a snapshot with start-docker, Python 3, at least 2 CPU and 4 GiB");
    let image = env::var("SANDBOX_DRIVER_NESTED_IMAGE").unwrap_or_else(|_| "alpine:3.20".into());
    let provider = DaytonaProvider::connect().await.unwrap();
    let mut spec = SandboxSpec::new(SandboxSource::Snapshot {
        id: SnapshotId::try_new(snapshot).unwrap(),
    })
    .sandbox_kind(SandboxKind::Container)
    .working_directory("/workspace");
    spec.user = Some("root".to_owned());
    spec.timers.auto_stop_after_idle = Some(Duration::ZERO);
    spec.provider_config = DaytonaProviderConfig {
        docker: Some(NestedDockerConfig {
            image,
            target: DockerExecutionTarget::Container,
            user: None,
            options: sandbox_driver_docker::DockerProviderConfig::default(),
        }),
        ..Default::default()
    }
    .into_value();
    let sandbox = provider.create(&spec, None).await.unwrap();
    let outcome = async {
        if !sandbox.capabilities().exec.stdin_stream {
            return Err("the container target does not declare exec.stdin_stream".to_owned());
        }
        turn_by_turn(sandbox.as_ref()).await?;
        exact_bytes(sandbox.as_ref()).await
    }
    .await;
    let cleanup = sandbox.delete().await;
    cleanup.expect("delete the live sandbox even when an assertion fails");
    outcome.expect("nested streamed stdin");
}
