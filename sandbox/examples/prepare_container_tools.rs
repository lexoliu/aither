#![doc = "Prepare a hash-verified read-only tool mount before creating a container."]

use std::path::Path;

use aither_sandbox::{ToolDistribution, ToolTarget};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [manifest, cache, os, arch, output] = args.as_slice() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: prepare_container_tools <manifest.json> <cache> <os> <arch> <mount.json>",
        )
        .into());
    };
    let distribution = ToolDistribution::from_path(Path::new(manifest)).await?;
    let bundle = distribution
        .load(&ToolTarget::from_probe(os, arch)?, Path::new(cache))
        .await?;
    let (mount, _) = bundle.prepare_container_mount(Path::new(cache)).await?;
    async_fs::write(output, serde_json::to_vec_pretty(&mount)?).await?;
    Ok(())
}
