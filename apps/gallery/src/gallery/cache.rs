//! On-disk thumbnail cache.
//!
//! Decoding a gallery preview from a camera means downloading a JPEG or the
//! LRI prefix plus sparse metadata reads: seconds per capture over USB, and a
//! long wait for a full card. Decoded previews are therefore persisted under
//! the platform cache directory (`$XDG_CACHE_HOME/chiaro/thumbnails`, or the
//! Windows/macOS equivalents) as a JPEG (quality 92, ~100 KB) plus a JSON
//! sidecar with the card metadata. A later session that lists the same captures is served from
//! disk without touching the camera's image data.
//!
//! Entries are keyed by a SHA-256 of what identifies a capture without
//! reading it: the source (camera serial, or folder path), file name, size,
//! and for local files the modification time, plus the decode parameters and
//! a format version so changes to the preview pipeline invalidate old entries.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use chiaro::lri::{CaptureDateTime, CaptureMetadata, PreviewImage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::source::{CaptureLocator, ObjectLocator, PreviewLocator};

/// Bump when the preview pipeline changes what a thumbnail looks like.
const FORMAT_VERSION: u32 = 3;
/// Thumbnails are display aids; at 720 px this keeps them around 60-120 KB.
const JPEG_QUALITY: u8 = 80;
/// Default size cap of the cache.
pub const DEFAULT_LIMIT_BYTES: u64 = 500 * 1_000_000;
/// Planning figure for "how many captures fit": a typical entry.
pub const TYPICAL_ENTRY_BYTES: u64 = 110 * 1000;

/// Where thumbnails live: `$XDG_CACHE_HOME/chiaro/thumbnails` or the platform
/// equivalent. `None` when no cache location can be determined.
pub fn thumbnail_dir() -> Option<PathBuf> {
    let base = if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg)
    } else if cfg!(target_os = "windows") {
        PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
    } else if cfg!(target_os = "macos") {
        std::env::home_dir()?.join("Library/Caches")
    } else {
        std::env::home_dir()?.join(".cache")
    };
    Some(base.join("chiaro").join("thumbnails"))
}

/// Identity of a cached preview.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ThumbnailKey(String);

impl ThumbnailKey {
    /// Key for a gallery preview decoded at `max_edge` pixels. Returns `None`
    /// when the source cannot be identified cheaply.
    pub fn for_preview(
        source: &PreviewLocator,
        capture: &CaptureLocator,
        max_edge: usize,
    ) -> Option<Self> {
        let mut hasher = Sha256::new();
        hasher.update(FORMAT_VERSION.to_le_bytes());
        hasher.update(max_edge.to_le_bytes());
        match capture {
            CaptureLocator::Local(path) => {
                let metadata = fs::metadata(path).ok()?;
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                hasher.update(b"local");
                hasher.update(
                    path.canonicalize()
                        .unwrap_or_else(|_| path.clone())
                        .to_string_lossy()
                        .as_bytes(),
                );
                hasher.update(metadata.len().to_le_bytes());
                hasher.update(modified.to_le_bytes());
            }
            CaptureLocator::Device(object) => {
                hasher.update(b"device");
                hasher.update(
                    object
                        .device()
                        .serial_number
                        .as_deref()
                        .unwrap_or("unknown-serial")
                        .as_bytes(),
                );
                hasher.update(object.name.to_ascii_lowercase().as_bytes());
                hasher.update(object.size.to_le_bytes());
            }
        }
        // The preview source matters too: a companion JPEG and the LRI's own
        // reference camera are different pictures.
        match source {
            PreviewLocator::Lri(_) => hasher.update(b"lri"),
            PreviewLocator::Jpeg(ObjectLocator::Device(jpeg)) => {
                hasher.update(b"jpeg");
                hasher.update(jpeg.size.to_le_bytes());
            }
        }
        Some(Self(hex(&hasher.finalize())))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Sidecar written next to every thumbnail: everything the card displays.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ThumbnailSidecar {
    pub version: u32,
    pub camera: String,
    pub color_calibrated: bool,
    pub width: usize,
    pub height: usize,
    pub iso: Option<u32>,
    pub shutter_ns: Option<u64>,
    pub focal_length_mm: Option<i32>,
    pub night_mode: bool,
    pub tripod: Option<bool>,
    /// `[year, month, day, hour, minute, second]` and the timezone offset.
    pub captured_at: Option<[u32; 6]>,
    pub timezone_offset_minutes: Option<i32>,
    pub orientation: u64,
}

impl ThumbnailSidecar {
    fn from_preview(preview: &PreviewImage) -> Self {
        let m = &preview.metadata;
        Self {
            version: FORMAT_VERSION,
            camera: preview.camera.clone(),
            color_calibrated: preview.color_calibrated,
            width: preview.size[0],
            height: preview.size[1],
            iso: m.iso,
            shutter_ns: m.shutter_ns,
            focal_length_mm: m.focal_length_mm,
            night_mode: m.night_mode,
            tripod: m.tripod,
            captured_at: m
                .captured_at
                .as_ref()
                .map(|t| [t.year, t.month, t.day, t.hour, t.minute, t.second]),
            timezone_offset_minutes: m
                .captured_at
                .as_ref()
                .and_then(|t| t.timezone_offset_minutes),
            orientation: m.orientation,
        }
    }

    fn metadata(&self) -> CaptureMetadata {
        CaptureMetadata {
            iso: self.iso,
            shutter_ns: self.shutter_ns,
            focal_length_mm: self.focal_length_mm,
            night_mode: self.night_mode,
            tripod: self.tripod,
            captured_at: self.captured_at.map(|t| CaptureDateTime {
                year: t[0],
                month: t[1],
                day: t[2],
                hour: t[3],
                minute: t[4],
                second: t[5],
                timezone_offset_minutes: self.timezone_offset_minutes,
            }),
            orientation: self.orientation,
        }
    }
}

/// Size and count of what is on disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheUsage {
    pub bytes: u64,
    pub entries: usize,
}

/// A thumbnail cache rooted at one directory, shared between the UI (which
/// configures it) and the preview workers (which read and write it).
#[derive(Debug)]
pub struct ThumbnailCache {
    root: PathBuf,
    enabled: AtomicBool,
    limit_bytes: AtomicU64,
    /// Usage as last scanned plus everything stored since; `None` until the
    /// first scan.
    usage: Mutex<Option<CacheUsage>>,
}

impl ThumbnailCache {
    /// The default platform cache, or `None` when it cannot be located.
    pub fn platform() -> Option<Self> {
        thumbnail_dir().map(Self::at)
    }

    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            enabled: AtomicBool::new(true),
            limit_bytes: AtomicU64::new(DEFAULT_LIMIT_BYTES),
            usage: Mutex::new(None),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes.load(Ordering::Relaxed)
    }

    /// Change the cap; evicts immediately if the cache is already larger.
    pub fn set_limit_bytes(&self, limit: u64) {
        self.limit_bytes.store(limit, Ordering::Relaxed);
        self.evict_to_limit();
    }

    /// Current usage, scanning the directory on first call.
    pub fn usage(&self) -> CacheUsage {
        let mut usage = self.usage.lock().expect("cache usage poisoned");
        *usage.get_or_insert_with(|| {
            let mut total = CacheUsage::default();
            for (_, _, size) in self.entries() {
                total.bytes += size;
                total.entries += 1;
            }
            total
        })
    }

    /// Delete every entry.
    pub fn clear(&self) -> std::io::Result<()> {
        if self.root.exists() {
            fs::remove_dir_all(&self.root)?;
        }
        *self.usage.lock().expect("cache usage poisoned") = Some(CacheUsage::default());
        Ok(())
    }

    /// `(image path, last use, bytes of image + sidecar)` for every entry.
    fn entries(&self) -> Vec<(PathBuf, SystemTime, u64)> {
        let mut out = Vec::new();
        let Ok(shards) = fs::read_dir(&self.root) else {
            return out;
        };
        for shard in shards.flatten() {
            let Ok(files) = fs::read_dir(shard.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().is_none_or(|e| e != "jpg") {
                    continue;
                }
                let Ok(meta) = fs::metadata(&path) else {
                    continue;
                };
                let sidecar = fs::metadata(path.with_extension("json")).map_or(0, |m| m.len());
                let used = meta.modified().unwrap_or(UNIX_EPOCH);
                out.push((path, used, meta.len() + sidecar));
            }
        }
        out
    }

    /// Remove least recently used entries until the cache fits its cap.
    fn evict_to_limit(&self) {
        let limit = self.limit_bytes();
        if self.usage().bytes <= limit {
            return;
        }
        let mut entries = self.entries();
        entries.sort_by_key(|(_, used, _)| *used);
        let mut total = CacheUsage {
            bytes: entries.iter().map(|e| e.2).sum(),
            entries: entries.len(),
        };
        for (path, _, size) in entries {
            if total.bytes <= limit {
                break;
            }
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(path.with_extension("json"));
            total.bytes = total.bytes.saturating_sub(size);
            total.entries = total.entries.saturating_sub(1);
        }
        *self.usage.lock().expect("cache usage poisoned") = Some(total);
    }

    fn paths(&self, key: &ThumbnailKey) -> (PathBuf, PathBuf) {
        // Two-character fan-out keeps directories small with thousands of entries.
        let dir = self.root.join(&key.as_str()[..2]);
        (
            dir.join(format!("{}.jpg", key.as_str())),
            dir.join(format!("{}.json", key.as_str())),
        )
    }

    /// Stored preview, or `None` when absent, unreadable, or the cache is
    /// disabled. A hit refreshes the entry's timestamp for LRU eviction.
    pub fn load(&self, key: &ThumbnailKey) -> Option<PreviewImage> {
        if !self.is_enabled() {
            return None;
        }
        let (image_path, json_path) = self.paths(key);
        if let Ok(file) = fs::File::options().append(true).open(&image_path) {
            let _ = file.set_modified(SystemTime::now());
        }
        let sidecar: ThumbnailSidecar = serde_json::from_slice(&fs::read(json_path).ok()?).ok()?;
        if sidecar.version != FORMAT_VERSION {
            return None;
        }
        let image = image::load_from_memory_with_format(
            &fs::read(image_path).ok()?,
            image::ImageFormat::Jpeg,
        )
        .ok()?
        .into_rgb8();
        if image.width() as usize != sidecar.width || image.height() as usize != sidecar.height {
            return None;
        }
        Some(PreviewImage {
            size: [sidecar.width, sidecar.height],
            rgb: image.into_raw(),
            camera: sidecar.camera.clone(),
            color_calibrated: sidecar.color_calibrated,
            metadata: sidecar.metadata(),
        })
    }

    /// Persist a decoded preview. Failures are reported but never fatal: the
    /// cache is an accelerator, not a store of record.
    pub fn store(&self, key: &ThumbnailKey, preview: &PreviewImage) -> std::io::Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        let (image_path, json_path) = self.paths(key);
        fs::create_dir_all(image_path.parent().expect("cache entry has a parent"))?;
        let temporary = image_path.with_extension("jpg.part");
        {
            let file = fs::File::create(&temporary)?;
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                std::io::BufWriter::new(file),
                JPEG_QUALITY,
            );
            encoder
                .encode(
                    &preview.rgb,
                    preview.size[0] as u32,
                    preview.size[1] as u32,
                    image::ExtendedColorType::Rgb8,
                )
                .map_err(std::io::Error::other)?;
        }
        fs::write(
            &json_path,
            serde_json::to_vec_pretty(&ThumbnailSidecar::from_preview(preview))?,
        )?;
        fs::rename(&temporary, &image_path)?;
        let added = fs::metadata(&image_path).map_or(0, |m| m.len())
            + fs::metadata(&json_path).map_or(0, |m| m.len());
        {
            let mut usage = self.usage.lock().expect("cache usage poisoned");
            let current = usage.get_or_insert_with(CacheUsage::default);
            current.bytes += added;
            current.entries += 1;
        }
        self.evict_to_limit();
        Ok(())
    }
}

/// Decode edge that keeps a framed crop at `max_edge` pixels: `None` when the
/// capture was framed wide (no crop) or the crop would not lose resolution.
pub fn framed_decode_edge(preview: &PreviewImage, max_edge: usize) -> Option<usize> {
    let focal = preview.metadata.focal_length_mm.filter(|f| *f > 28)?;
    let factor = (focal as f32 / 28.0).min(6.0);
    let edge = (max_edge as f32 * factor).round() as usize;
    (edge > max_edge).then_some(edge)
}

/// Crop a reference-camera preview to the field of view the photographer
/// framed: the wide modules see 28 mm equivalent and the recorded zoom is a
/// centred crop of that view. Companion JPEGs are already framed and are not
/// passed through here.
pub fn crop_to_framing(preview: &mut PreviewImage) {
    const WIDE_EQUIVALENT_FOCAL_MM: f32 = 28.0;
    let Some(focal) = preview.metadata.focal_length_mm.filter(|f| *f > 28) else {
        return;
    };
    let fraction = (WIDE_EQUIVALENT_FOCAL_MM / focal as f32).clamp(0.05, 1.0);
    let [width, height] = preview.size;
    let new_width = ((width as f32 * fraction).round() as usize).clamp(1, width);
    let new_height = ((height as f32 * fraction).round() as usize).clamp(1, height);
    let x0 = (width - new_width) / 2;
    let y0 = (height - new_height) / 2;
    let mut rgb = Vec::with_capacity(new_width * new_height * 3);
    for y in y0..y0 + new_height {
        let row = &preview.rgb[(y * width + x0) * 3..(y * width + x0 + new_width) * 3];
        rgb.extend_from_slice(row);
    }
    preview.rgb = rgb;
    preview.size = [new_width, new_height];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_preview() -> PreviewImage {
        // A smooth gradient: JPEG keeps it within a few codes.
        PreviewImage {
            size: [8, 6],
            rgb: (0..8 * 6 * 3).map(|i| (40 + i / 3 * 4) as u8).collect(),
            camera: "A1".to_owned(),
            color_calibrated: true,
            metadata: CaptureMetadata {
                iso: Some(800),
                shutter_ns: Some(6_000_000_000),
                focal_length_mm: Some(87),
                night_mode: false,
                tripod: Some(true),
                captured_at: Some(CaptureDateTime {
                    year: 2026,
                    month: 8,
                    day: 13,
                    hour: 1,
                    minute: 22,
                    second: 21,
                    timezone_offset_minutes: Some(-180),
                }),
                orientation: 0,
            },
        }
    }

    #[test]
    fn store_and_load_round_trip_pixels_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::at(dir.path());
        let key = ThumbnailKey("ab".repeat(32));
        assert!(cache.load(&key).is_none());
        let preview = sample_preview();
        cache.store(&key, &preview).unwrap();
        let loaded = cache.load(&key).unwrap();
        assert_eq!(loaded.size, preview.size);
        // JPEG is lossy; the content must stay close.
        let error = loaded
            .rgb
            .iter()
            .zip(&preview.rgb)
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .unwrap();
        assert!(error <= 12, "max pixel error {error}");
        assert_eq!(loaded.camera, "A1");
        assert_eq!(loaded.metadata, preview.metadata);
        assert!(
            dir.path()
                .join("ab")
                .join(format!("{}.json", key.as_str()))
                .is_file()
        );
    }

    #[test]
    fn eviction_drops_the_least_recently_used_entries_and_clear_empties() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::at(dir.path());
        let preview = sample_preview();
        let keys = (0..6)
            .map(|i| ThumbnailKey(format!("{i:02x}").repeat(32)))
            .collect::<Vec<_>>();
        cache.store(&keys[0], &preview).unwrap();
        let one = cache.usage();
        assert_eq!(one.entries, 1);
        assert!(one.bytes > 0);
        for key in &keys[1..] {
            // Distinct timestamps make the LRU order deterministic.
            std::thread::sleep(std::time::Duration::from_millis(15));
            cache.store(key, &preview).unwrap();
        }
        assert_eq!(cache.usage().entries, 6);
        // Use the first entry again so it survives, then cap at three entries.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(cache.load(&keys[0]).is_some());
        cache.set_limit_bytes(one.bytes * 3);
        let usage = cache.usage();
        assert!(usage.entries <= 3, "{usage:?}");
        assert!(cache.load(&keys[0]).is_some(), "recently used entry kept");
        assert!(cache.load(&keys[1]).is_none(), "oldest entry evicted");
        cache.set_enabled(false);
        assert!(cache.load(&keys[0]).is_none());
        cache.store(&keys[1], &preview).unwrap();
        cache.set_enabled(true);
        assert!(
            cache.load(&keys[1]).is_none(),
            "nothing stored while disabled"
        );
        cache.clear().unwrap();
        assert_eq!(cache.usage(), CacheUsage::default());
        assert!(cache.load(&keys[0]).is_none());
    }

    #[test]
    fn local_keys_change_with_file_contents_and_decode_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("L16_00001.lri");
        fs::write(&path, b"one").unwrap();
        let source = PreviewLocator::Lri(CaptureLocator::Local(path.clone()));
        let capture = CaptureLocator::Local(path.clone());
        let a = ThumbnailKey::for_preview(&source, &capture, 720).unwrap();
        let b = ThumbnailKey::for_preview(&source, &capture, 360).unwrap();
        assert_ne!(a, b);
        fs::write(&path, b"longer contents").unwrap();
        let c = ThumbnailKey::for_preview(&source, &capture, 720).unwrap();
        assert_ne!(a, c);
        assert!(
            ThumbnailKey::for_preview(
                &source,
                &CaptureLocator::Local(dir.path().join("missing.lri")),
                720
            )
            .is_none()
        );
    }

    #[test]
    fn framing_crop_keeps_the_centre() {
        let mut preview = sample_preview();
        preview.size = [100, 75];
        preview.rgb = vec![0; 100 * 75 * 3];
        // Mark the centre pixel.
        let centre = (37 * 100 + 50) * 3;
        preview.rgb[centre] = 255;
        crop_to_framing(&mut preview);
        // 28/87 of 100 x 75 is 32 x 24.
        assert_eq!(preview.size, [32, 24]);
        assert_eq!(preview.rgb.len(), 32 * 24 * 3);
        assert!(preview.rgb.contains(&255));
        let mut wide = sample_preview();
        wide.metadata.focal_length_mm = Some(28);
        crop_to_framing(&mut wide);
        assert_eq!(wide.size, [8, 6]);
    }

    #[test]
    fn framed_decode_edge_scales_with_focal_length() {
        let mut preview = sample_preview();
        assert_eq!(framed_decode_edge(&preview, 720), Some(2237));
        preview.metadata.focal_length_mm = Some(28);
        assert_eq!(framed_decode_edge(&preview, 720), None);
        preview.metadata.focal_length_mm = None;
        assert_eq!(framed_decode_edge(&preview, 720), None);
    }

    #[test]
    fn cache_dir_follows_xdg_when_set() {
        // Only the structure is asserted; the environment is not modified.
        if let Some(dir) = thumbnail_dir() {
            assert!(dir.ends_with(Path::new("chiaro").join("thumbnails")));
        }
    }
}
