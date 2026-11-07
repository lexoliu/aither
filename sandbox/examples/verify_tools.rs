#![doc = "Verify prepared baseline commands through the real terminal sandbox."]

use std::path::PathBuf;

use aither_core::llm::{Tool, ToolContext};
use aither_sandbox::{
    TerminalArgs, TerminalTool, ToolDistribution, ToolRegistryBuilder, ToolTarget,
    permission::NoopPermissionHandler,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(manifest), Some(cache), Some(workspace), None) =
        (args.next(), args.next(), args.next(), args.next())
    else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: verify_tools <manifest.json> <cache-directory> <workspace-directory>",
        )
        .into());
    };
    let pinned: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest)?)?;
    let distribution = ToolDistribution::from_path(&PathBuf::from(manifest)).await?;
    let terminal = TerminalTool::new_exact(
        PathBuf::from(workspace),
        NoopPermissionHandler,
        executor_core::tokio::TokioGlobal,
    )
    .await?
    .initialize_tools(&distribution, &PathBuf::from(cache))
    .await?;
    let registry = ToolRegistryBuilder::new().build(terminal.output_store().clone());
    let terminal = terminal.with_registry(std::sync::Arc::new(registry));
    let arguments: TerminalArgs = serde_json::from_value(serde_json::json!({
        "description": "Verify prepared search, JSON and edit commands",
        "script": include_str!("../templates/verify_baseline_tools.sh"),
        "mode": "sandboxed",
        "raw": true,
        "timeout": 120
    }))?;
    let result = terminal.call(arguments, ToolContext::new()).await?;
    let output = result.render_for_model();
    assert!(output.contains("baseline acceptance passed"), "{output}");
    // `uv --version` and `python3 --version` report the manifest's pins.
    let target = ToolTarget::native()?;
    if target.provides_python() {
        let pin_of = |entries: &str, key: &str, matches: &dyn Fn(&serde_json::Value) -> bool| {
            pinned[entries]
                .as_array()
                .and_then(|entries| entries.iter().find(|entry| matches(entry)))
                .and_then(|entry| entry[key].as_str())
                .map(str::to_owned)
        };
        let on_target = |entry: &serde_json::Value| {
            entry["target"]["os"] == target.os() && entry["target"]["arch"] == target.arch()
        };
        let uv = pin_of("artifacts", "version", &|entry| {
            entry["tool"] == "uv" && on_target(entry)
        })
        .expect("the manifest pins uv for the host");
        let python =
            pin_of("python", "version", &on_target).expect("the manifest pins python for the host");
        assert!(
            output.contains(&format!("uv {uv} ")),
            "uv {uv} missing: {output}"
        );
        assert!(
            output.contains(&format!("Python {python}")),
            "Python {python} missing: {output}"
        );
    }
    assert!(!output.contains("Exit code:"), "{output}");
    assert!(!output.contains("Started in background"), "{output}");
    Ok(())
}
