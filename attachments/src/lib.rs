//! Attachment cache for provider file uploads.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use async_fs::File;
use futures_lite::io::AsyncReadExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CACHE_FILE_NAME: &str = "file_cache.json";

/// Returns the default cache directory for attachments.
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("AITHER_CACHE_DIR").map_or_else(
        || std::env::temp_dir().join("aither").join("cache"),
        PathBuf::from,
    )
}

/// A cache entry for an uploaded file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// SHA-256 hash of file content (for detecting changes).
    pub content_hash: String,
    /// Provider name (e.g., "gemini", "openai").
    pub provider: String,
    /// Provider-specific reference (file URI or `file_id`).
    pub reference: String,
    /// When this reference expires (None if it never expires).
    #[serde(with = "system_time_option")]
    pub expires_at: Option<SystemTime>,
}

impl CacheEntry {
    /// Create a new cache entry.
    #[must_use]
    pub const fn new(
        content_hash: String,
        provider: String,
        reference: String,
        expires_at: Option<SystemTime>,
    ) -> Self {
        Self {
            content_hash,
            provider,
            reference,
            expires_at,
        }
    }

    /// Check if this cache entry is still valid (not expired and hash matches).
    #[must_use]
    pub fn is_valid(&self, current_hash: &str) -> bool {
        if self.content_hash != current_hash {
            return false;
        }
        self.expires_at
            .is_none_or(|expires| SystemTime::now() < expires)
    }
}

/// File cache for tracking uploaded files across providers.
///
/// The cache stores file references keyed by `(file_path, provider)` pairs.
/// Each entry includes the content hash to detect file changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCache {
    /// Cache directory for persistence.
    #[serde(skip)]
    cache_dir: PathBuf,
    /// Map of (`canonical_path`, provider) -> `CacheEntry`.
    entries: HashMap<String, CacheEntry>,
}

impl FileCache {
    /// Open a file cache with the given cache directory.
    ///
    /// Loads existing cache data if the cache file exists.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache file cannot be read or parsed.
    pub async fn open(cache_dir: PathBuf) -> std::io::Result<Self> {
        let cache_file = cache_dir.join(CACHE_FILE_NAME);
        if async_fs::metadata(&cache_file).await.is_ok() {
            let contents = async_fs::read_to_string(&cache_file).await?;
            let cache: Self = serde_json::from_str(&contents)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            return Ok(Self {
                cache_dir,
                entries: cache.entries,
            });
        }
        Ok(Self {
            cache_dir,
            entries: HashMap::new(),
        })
    }

    /// Save cache to disk.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache directory cannot be created or file cannot be written.
    pub async fn save(&self) -> std::io::Result<()> {
        async_fs::create_dir_all(&self.cache_dir).await?;
        let cache_file = self.cache_dir.join(CACHE_FILE_NAME);
        let contents = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        async_fs::write(cache_file, contents).await?;
        Ok(())
    }

    /// Generate a cache key from file path and provider.
    fn cache_key(file_path: &Path, provider: &str) -> String {
        format!("{}::{}", file_path.display(), provider)
    }

    /// Get a cached entry for a file if it exists and is valid.
    ///
    /// Returns None if:
    /// - No cache entry exists
    /// - The file has changed (hash mismatch)
    /// - The cache entry has expired
    ///
    /// # Errors
    ///
    /// Returns an error when file hashing fails.
    pub async fn get(
        &self,
        file_path: &Path,
        provider: &str,
    ) -> std::io::Result<Option<CacheEntry>> {
        let key = Self::cache_key(file_path, provider);
        let Some(entry) = self.entries.get(&key) else {
            return Ok(None);
        };

        let current_hash = hash_file(file_path).await?;
        if entry.is_valid(&current_hash) {
            Ok(Some(entry.clone()))
        } else {
            Ok(None)
        }
    }

    /// Insert a new cache entry.
    ///
    /// The content hash is computed automatically from the file.
    ///
    /// # Errors
    ///
    /// Returns an error when file hashing fails.
    pub async fn insert(
        &mut self,
        file_path: &Path,
        provider: &str,
        reference: String,
        expires_at: Option<SystemTime>,
    ) -> std::io::Result<()> {
        let content_hash = hash_file(file_path).await?;
        let key = Self::cache_key(file_path, provider);
        let entry = CacheEntry::new(content_hash, provider.to_string(), reference, expires_at);
        self.entries.insert(key, entry);
        Ok(())
    }

    /// Remove a cache entry.
    pub fn remove(&mut self, file_path: &Path, provider: &str) -> Option<CacheEntry> {
        let key = Self::cache_key(file_path, provider);
        self.entries.remove(&key)
    }

    /// Remove all expired entries from the cache.
    ///
    /// Returns true if any entries were removed.
    pub fn prune_expired(&mut self) -> bool {
        let now = SystemTime::now();
        let before = self.entries.len();
        self.entries
            .retain(|_, entry| entry.expires_at.is_none_or(|expires| now < expires));
        before != self.entries.len()
    }

    /// Calculate expiration time from a duration.
    #[must_use]
    pub fn expires_in(duration: Duration) -> Option<SystemTime> {
        SystemTime::now().checked_add(duration)
    }
}

async fn hash_file(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = file.read(&mut buffer).await?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let result = hasher.finalize();
    Ok(format!("{result:x}"))
}

/// Serde support for Option<SystemTime>.
mod system_time_option {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[allow(clippy::ref_option)]
    pub fn serialize<S>(time: &Option<SystemTime>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match time {
            Some(t) => {
                let duration = t.duration_since(UNIX_EPOCH).unwrap_or_default();
                Some(duration.as_secs()).serialize(serializer)
            }
            None => None::<u64>.serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<SystemTime>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs: Option<u64> = Option::deserialize(deserializer)?;
        Ok(secs.map(|s| UNIX_EPOCH + Duration::from_secs(s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_cache_insert_get() {
        tokio_test::block_on(async {
            let dir = tempfile::tempdir().expect("create temp dir");
            let cache_dir = dir.path().join("cache");
            let file_path = dir.path().join("file.txt");
            async_fs::write(&file_path, b"hello")
                .await
                .expect("write file");

            let mut cache = FileCache::open(cache_dir).await.expect("open cache");
            cache
                .insert(&file_path, "openai", "file-1".to_string(), None)
                .await
                .expect("insert");
            cache.save().await.expect("save");

            let cache = FileCache::open(dir.path().join("cache"))
                .await
                .expect("reload");
            let entry = cache
                .get(&file_path, "openai")
                .await
                .expect("get")
                .expect("entry");
            assert_eq!(entry.reference, "file-1");
        });
    }
}
