//! Package a pinned distribution beside the executable's installation.

use std::path::PathBuf;

use aither_sandbox::ToolDistribution;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(manifest), Some(destination), None) = (args.next(), args.next(), args.next()) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: package_tools <pinned-manifest.json> <installation-tools-directory>",
        )
        .into());
    };
    let distribution = ToolDistribution::from_path(&PathBuf::from(manifest)).await?;
    distribution
        .package_into(&PathBuf::from(destination))
        .await?;
    Ok(())
}
