#![doc = "Stage a hash-verified tool directory for a container image build."]

use std::path::{Path, PathBuf};

use aither_sandbox::{ToolDistribution, ToolTarget};

/// Where the staged directory lives on the host and where the image build
/// must copy it inside the image.
#[derive(serde::Serialize)]
struct ImageToolsOutput {
    staged: PathBuf,
    container_path: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [manifest, cache, os, arch, output] = args.as_slice() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: prepare_container_tools <manifest.json> <cache> <os> <arch> <image-tools.json>",
        )
        .into());
    };
    let distribution = ToolDistribution::from_path(Path::new(manifest)).await?;
    let target = ToolTarget::from_probe(os, arch)?;
    let staged = distribution
        .stage_image_tools(
            &target,
            &Path::new(cache).join("artifacts"),
            Path::new(cache),
        )
        .await?;
    let container_path = distribution
        .image_tools(&target)?
        .container_directory()?
        .as_str()
        .to_owned();
    let tools = ImageToolsOutput {
        staged,
        container_path,
    };
    async_fs::write(output, serde_json::to_vec_pretty(&tools)?).await?;
    Ok(())
}
