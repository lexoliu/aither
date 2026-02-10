use async_fs as fs;
use std::path::Path;

pub(crate) async fn path_exists(path: &Path) -> bool {
    match fs::metadata(path).await {
        Ok(_) => true,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(err) => {
            panic!("failed to access path {}: {}", path.display(), err);
        }
    }
}
