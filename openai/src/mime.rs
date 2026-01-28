use std::path::Path;

/// Determine MIME type from file extension.
pub(crate) fn mime_from_path(path: &Path) -> Option<&'static str> {
    mime_guess::from_path(path).first_raw()
}
