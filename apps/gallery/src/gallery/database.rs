//! Inspectable SQLite catalog for capture identities, thumbnail files, and
//! successful exports.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::source::CaptureLocator;

const SCHEMA_VERSION: u32 = 1;
const CAPTURE_IDENTITY_VERSION: u32 = 1;

/// Stable-enough identity available without reading a large LRI payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureIdentity {
    pub hash: String,
    pub source_path: String,
    pub name: String,
}

impl CaptureIdentity {
    pub fn for_capture(capture: &CaptureLocator) -> Option<Self> {
        let mut hasher = Sha256::new();
        hasher.update(CAPTURE_IDENTITY_VERSION.to_le_bytes());
        let (source_path, name) = match capture {
            CaptureLocator::Local(path) => {
                let metadata = fs::metadata(path).ok()?;
                let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                let source_path = canonical.to_string_lossy().into_owned();
                let name = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(&source_path)
                    .to_owned();
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |duration| duration.as_nanos());
                hasher.update(b"local");
                hasher.update(source_path.as_bytes());
                hasher.update(metadata.len().to_le_bytes());
                hasher.update(modified.to_le_bytes());
                (source_path, name)
            }
            CaptureLocator::Device(object) => {
                let device = object.device();
                let camera = device
                    .serial_number
                    .clone()
                    .unwrap_or_else(|| format!("usb-{}", device.location_id));
                let name = object.name.clone();
                let source_path = format!("l16://{camera}/{name}");
                hasher.update(b"device");
                hasher.update(camera.as_bytes());
                hasher.update(name.to_ascii_lowercase().as_bytes());
                hasher.update(object.size.to_le_bytes());
                (source_path, name)
            }
        };
        Some(Self {
            hash: hex(&hasher.finalize()),
            source_path,
            name,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ThumbnailRecord {
    pub image_path: PathBuf,
    pub metadata_path: PathBuf,
    pub capture_hash: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ThumbnailIndexEntry {
    pub key: String,
    pub image_path: PathBuf,
    pub metadata_path: PathBuf,
    pub size_bytes: u64,
}

/// One process-local connection protected for preview worker access.
#[derive(Debug)]
pub struct GalleryDatabase {
    path: PathBuf,
    connection: Mutex<Connection>,
}

impl GalleryDatabase {
    pub fn open(path: impl Into<PathBuf>) -> rusqlite::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
        }
        let connection = Connection::open(&path)?;
        connection.busy_timeout(Duration::from_secs(2))?;
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS captures (
                 capture_hash TEXT PRIMARY KEY,
                 source_path TEXT NOT NULL,
                 capture_name TEXT NOT NULL,
                 last_seen INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS thumbnails (
                 preview_key TEXT PRIMARY KEY,
                 capture_hash TEXT,
                 source_path TEXT,
                 image_path TEXT NOT NULL,
                 metadata_path TEXT NOT NULL,
                 size_bytes INTEGER NOT NULL,
                 last_accessed INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS thumbnails_capture_hash
                 ON thumbnails(capture_hash);
             CREATE INDEX IF NOT EXISTS thumbnails_last_accessed
                 ON thumbnails(last_accessed);
             CREATE TABLE IF NOT EXISTS exports (
                 capture_hash TEXT NOT NULL,
                 source_path TEXT NOT NULL,
                 capture_name TEXT NOT NULL,
                 pipeline TEXT NOT NULL,
                 output_path TEXT NOT NULL,
                 exported_at INTEGER NOT NULL,
                 PRIMARY KEY (capture_hash, pipeline, output_path)
             );
             CREATE INDEX IF NOT EXISTS exports_capture_hash
                 ON exports(capture_hash);
             CREATE TABLE IF NOT EXISTS app_metadata (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );",
        )?;
        connection.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(Self {
            path,
            connection: Mutex::new(connection),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn remember_capture(&self, identity: &CaptureIdentity) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute(
                "INSERT INTO captures (capture_hash, source_path, capture_name, last_seen)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(capture_hash) DO UPDATE SET
                 source_path = excluded.source_path,
                 capture_name = excluded.capture_name,
                 last_seen = excluded.last_seen",
                params![
                    identity.hash,
                    identity.source_path,
                    identity.name,
                    unix_time(SystemTime::now())
                ],
            )?;
        Ok(())
    }

    pub fn exported_hashes(&self) -> rusqlite::Result<HashSet<String>> {
        let connection = self.connection.lock().expect("gallery database poisoned");
        let mut query = connection.prepare("SELECT DISTINCT capture_hash FROM exports")?;
        query
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<HashSet<_>>>()
    }

    pub fn record_export(
        &self,
        identity: &CaptureIdentity,
        pipeline: &str,
        output_path: &Path,
    ) -> rusqlite::Result<()> {
        self.remember_capture(identity)?;
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute(
                "INSERT INTO exports
                 (capture_hash, source_path, capture_name, pipeline, output_path, exported_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(capture_hash, pipeline, output_path) DO UPDATE SET
                 source_path = excluded.source_path,
                 capture_name = excluded.capture_name,
                 exported_at = excluded.exported_at",
                params![
                    identity.hash,
                    identity.source_path,
                    identity.name,
                    pipeline,
                    output_path.to_string_lossy(),
                    unix_time(SystemTime::now())
                ],
            )?;
        Ok(())
    }

    pub fn thumbnail(&self, key: &str) -> rusqlite::Result<Option<ThumbnailRecord>> {
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .query_row(
                "SELECT image_path, metadata_path, capture_hash
                 FROM thumbnails WHERE preview_key = ?1",
                [key],
                |row| {
                    Ok(ThumbnailRecord {
                        image_path: PathBuf::from(row.get::<_, String>(0)?),
                        metadata_path: PathBuf::from(row.get::<_, String>(1)?),
                        capture_hash: row.get(2)?,
                    })
                },
            )
            .optional()
    }

    pub fn upsert_thumbnail(
        &self,
        key: &str,
        identity: Option<&CaptureIdentity>,
        image_path: &Path,
        metadata_path: &Path,
        size_bytes: u64,
        last_accessed: SystemTime,
    ) -> rusqlite::Result<()> {
        if let Some(identity) = identity {
            self.remember_capture(identity)?;
        }
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute(
                "INSERT INTO thumbnails
                 (preview_key, capture_hash, source_path, image_path, metadata_path,
                  size_bytes, last_accessed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(preview_key) DO UPDATE SET
                 capture_hash = excluded.capture_hash,
                 source_path = excluded.source_path,
                 image_path = excluded.image_path,
                 metadata_path = excluded.metadata_path,
                 size_bytes = excluded.size_bytes,
                 last_accessed = excluded.last_accessed",
                params![
                    key,
                    identity.map(|value| value.hash.as_str()),
                    identity.map(|value| value.source_path.as_str()),
                    image_path.to_string_lossy(),
                    metadata_path.to_string_lossy(),
                    i64::try_from(size_bytes).unwrap_or(i64::MAX),
                    unix_time(last_accessed)
                ],
            )?;
        Ok(())
    }

    pub fn touch_thumbnail(&self, key: &str) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute(
                "UPDATE thumbnails SET last_accessed = ?2 WHERE preview_key = ?1",
                params![key, unix_time(SystemTime::now())],
            )?;
        Ok(())
    }

    pub fn link_thumbnail(&self, key: &str, identity: &CaptureIdentity) -> rusqlite::Result<()> {
        self.remember_capture(identity)?;
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute(
                "UPDATE thumbnails
                 SET capture_hash = ?2, source_path = ?3
                 WHERE preview_key = ?1",
                params![key, identity.hash, identity.source_path],
            )?;
        Ok(())
    }

    pub fn thumbnail_usage(&self) -> rusqlite::Result<(u64, usize)> {
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .query_row(
                "SELECT COALESCE(SUM(size_bytes), 0), COUNT(*) FROM thumbnails",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?.max(0) as u64,
                        row.get::<_, i64>(1)?.max(0) as usize,
                    ))
                },
            )
    }

    pub fn thumbnails_oldest_first(&self) -> rusqlite::Result<Vec<ThumbnailIndexEntry>> {
        let connection = self.connection.lock().expect("gallery database poisoned");
        let mut query = connection.prepare(
            "SELECT preview_key, image_path, metadata_path, size_bytes
             FROM thumbnails ORDER BY last_accessed ASC",
        )?;
        query
            .query_map([], |row| {
                Ok(ThumbnailIndexEntry {
                    key: row.get(0)?,
                    image_path: PathBuf::from(row.get::<_, String>(1)?),
                    metadata_path: PathBuf::from(row.get::<_, String>(2)?),
                    size_bytes: row.get::<_, i64>(3)?.max(0) as u64,
                })
            })?
            .collect()
    }

    pub fn remove_thumbnail(&self, key: &str) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute("DELETE FROM thumbnails WHERE preview_key = ?1", [key])?;
        Ok(())
    }

    pub fn clear_thumbnails(&self) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute("DELETE FROM thumbnails", [])?;
        Ok(())
    }

    pub fn thumbnail_index_initialized(&self) -> rusqlite::Result<bool> {
        Ok(self
            .connection
            .lock()
            .expect("gallery database poisoned")
            .query_row(
                "SELECT value FROM app_metadata WHERE key = 'thumbnail_index_initialized'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .as_deref()
            == Some("1"))
    }

    pub fn set_thumbnail_index_initialized(&self) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("gallery database poisoned")
            .execute(
                "INSERT INTO app_metadata (key, value)
             VALUES ('thumbnail_index_initialized', '1')
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [],
            )?;
        Ok(())
    }
}

fn unix_time(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |duration| {
        i64::try_from(duration.as_secs()).unwrap_or(i64::MAX)
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn database_indexes_thumbnails_and_exports() {
        let dir = tempfile::tempdir().unwrap();
        let database = GalleryDatabase::open(dir.path().join("gallery.sqlite3")).unwrap();
        let identity = CaptureIdentity {
            hash: "ab".repeat(32),
            source_path: "/captures/L16_00001.lri".to_owned(),
            name: "L16_00001.lri".to_owned(),
        };
        let image = dir.path().join("thumb.jpg");
        let metadata = dir.path().join("thumb.json");
        database
            .upsert_thumbnail(
                "cdcd",
                Some(&identity),
                &image,
                &metadata,
                123,
                UNIX_EPOCH + Duration::from_secs(10),
            )
            .unwrap();
        assert_eq!(database.thumbnail_usage().unwrap(), (123, 1));
        assert_eq!(
            database.thumbnail("cdcd").unwrap().unwrap().image_path,
            image
        );

        database
            .record_export(&identity, "Fused high-resolution frame", dir.path())
            .unwrap();
        assert!(database.exported_hashes().unwrap().contains(&identity.hash));
        database.clear_thumbnails().unwrap();
        assert_eq!(database.thumbnail_usage().unwrap(), (0, 0));
    }
}
