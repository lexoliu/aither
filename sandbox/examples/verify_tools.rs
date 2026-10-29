#![doc = "Verify prepared baseline commands through the real terminal sandbox."]

use std::path::PathBuf;

use aither_core::llm::{Tool, ToolContext};
use aither_sandbox::{
    TerminalArgs, TerminalTool, ToolDistribution, ToolRegistryBuilder,
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
    let distribution = ToolDistribution::from_path(&PathBuf::from(manifest)).await?;
    let terminal = TerminalTool::new_exact(
        PathBuf::from(workspace),
        NoopPermissionHandler,
        executor_core::tokio::TokioGlobal,
    )
    .await?
    .initialize_tools(&distribution, &PathBuf::from(cache))
    .await?;
    let registry = ToolRegistryBuilder::new().build(terminal.outputs_dir());
    let terminal = terminal.with_registry(std::sync::Arc::new(registry));
    let arguments: TerminalArgs = serde_json::from_value(serde_json::json!({
        "description": "Verify prepared search, JSON and edit commands",
        "script": include_str!("../templates/verify_baseline_tools.sh"),
        "mode": "sandboxed",
        "raw": true
    }))?;
    let result = terminal.call(arguments, ToolContext::new()).await?;
    let output = result.render_for_model();
    assert!(output.contains("baseline acceptance passed"), "{output}");
    assert!(!output.contains("Exit code:"), "{output}");
    assert!(!output.contains("Started in background"), "{output}");
    Ok(())
}
