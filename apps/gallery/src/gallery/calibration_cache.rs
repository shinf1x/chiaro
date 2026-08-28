//! Persistent, device-identity-safe copies of factory camera calibration.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use chiaro_fusion::calibration::{DeviceId, LriMessages};
use sha2::{Digest, Sha256};

use crate::source::DEVICE_CALIBRATION_FILES;

#[derive(Clone, Debug)]
pub struct CachedCalibration {
    pub device_id: DeviceId,
    pub files: HashMap<String, PathBuf>,
}

/// `$XDG_CACHE_HOME/chiaro/calibration` (or the platform equivalent).
pub fn calibration_dir() -> Option<PathBuf> {
    let base = if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg)
    } else if cfg!(target_os = "windows") {
        PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
    } else if cfg!(target_os = "macos") {
        std::env::home_dir()?.join("Library/Caches")
    } else {
        std::env::home_dir()?.join(".cache")
    };
    Some(base.join("chiaro").join("calibration"))
}

/// Store a newly encountered camera's files under its USB serial label. Every
/// LRI that declares an identity must agree before any file becomes eligible
/// for automatic use.
pub fn store_device_files(
    label: &str,
    bytes: HashMap<String, Vec<u8>>,
) -> Result<CachedCalibration, String> {
    let root = calibration_dir().ok_or_else(|| "No platform cache directory".to_owned())?;
    store_device_files_at(&root, label, bytes)
}

/// Reuse a previous persistent copy for a connected USB serial.
pub fn load_for_label(label: &str) -> Result<Option<CachedCalibration>, String> {
    let Some(root) = calibration_dir() else {
        return Ok(None);
    };
    load_folder(&root.join(label_key(label)))
}

/// Find calibration belonging to the exact physical device recorded by a
/// capture. Cache contents with missing or different ids are ignored.
pub fn load_for_device_id(device_id: DeviceId) -> Result<Option<CachedCalibration>, String> {
    let Some(root) = calibration_dir() else {
        return Ok(None);
    };
    load_for_device_id_at(&root, device_id)
}

fn store_device_files_at(
    root: &Path,
    label: &str,
    bytes: HashMap<String, Vec<u8>>,
) -> Result<CachedCalibration, String> {
    identify_bytes(&bytes)?.ok_or_else(|| {
        "The camera calibration files do not record a physical device id".to_owned()
    })?;
    let folder = root.join(label_key(label));
    fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    for (name, data) in bytes {
        if !is_calibration_name(&name) {
            continue;
        }
        let name = name.to_ascii_lowercase();
        let destination = folder.join(&name);
        let temporary = folder.join(format!(".{name}.tmp"));
        fs::write(&temporary, data).map_err(|error| error.to_string())?;
        fs::rename(&temporary, &destination).map_err(|error| error.to_string())?;
    }
    load_folder(&folder)?.ok_or_else(|| "Calibration cache write was incomplete".to_owned())
}

fn load_for_device_id_at(
    root: &Path,
    device_id: DeviceId,
) -> Result<Option<CachedCalibration>, String> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        if !entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
        {
            continue;
        }
        if let Some(cached) = load_folder(&entry.path())?
            && cached.device_id == device_id
        {
            return Ok(Some(cached));
        }
    }
    Ok(None)
}

fn load_folder(folder: &Path) -> Result<Option<CachedCalibration>, String> {
    let mut files = HashMap::new();
    for name in DEVICE_CALIBRATION_FILES {
        let path = folder.join(name);
        if path.is_file() {
            files.insert(name.to_owned(), path);
        }
    }
    if files.is_empty() {
        return Ok(None);
    }
    let device_id = identify_paths(&files)?.ok_or_else(|| {
        format!(
            "Cached calibration in {} has no physical device id",
            folder.display()
        )
    })?;
    Ok(Some(CachedCalibration { device_id, files }))
}

fn identify_bytes(files: &HashMap<String, Vec<u8>>) -> Result<Option<DeviceId>, String> {
    identify(files.iter().filter_map(|(name, bytes)| {
        name.to_ascii_lowercase()
            .ends_with(".lri")
            .then_some(bytes.as_slice())
    }))
}

fn identify_paths(files: &HashMap<String, PathBuf>) -> Result<Option<DeviceId>, String> {
    let contents = files
        .iter()
        .filter(|(name, _)| name.ends_with(".lri"))
        .map(|(_, path)| fs::read(path).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    identify(contents.iter().map(Vec::as_slice))
}

fn identify<'a>(lris: impl IntoIterator<Item = &'a [u8]>) -> Result<Option<DeviceId>, String> {
    let mut identity = None;
    for bytes in lris {
        let messages = LriMessages::parse(bytes).map_err(|error| error.to_string())?;
        let Some(candidate) = messages.device_id() else {
            continue;
        };
        if identity.is_some_and(|existing| existing != candidate) {
            return Err("Calibration files belong to different physical cameras".to_owned());
        }
        identity = Some(candidate);
    }
    Ok(identity)
}

fn is_calibration_name(name: &str) -> bool {
    DEVICE_CALIBRATION_FILES
        .iter()
        .any(|known| name.eq_ignore_ascii_case(known))
}

fn sanitize(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "unknown-camera".to_owned()
    } else {
        sanitized
    }
}

fn label_key(label: &str) -> String {
    let digest = Sha256::digest(label.as_bytes());
    let suffix = digest[..6]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{}-{suffix}", sanitize(label))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../crates/chiaro-fusion/tests/fixtures")
                .join(name),
        )
        .unwrap()
    }

    #[test]
    fn persistent_files_are_found_only_by_matching_device_id() {
        let directory = tempfile::tempdir().unwrap();
        let mut files = HashMap::new();
        files.insert("calibration.lri".to_owned(), fixture("calibration.lri"));
        files.insert("zoom_calib_v0.lri".to_owned(), fixture("zoom_calib_v0.lri"));
        files.insert("hotpixel.rec".to_owned(), b"hot pixels".to_vec());
        let stored = store_device_files_at(directory.path(), "L16 04366", files).unwrap();
        let loaded = load_for_device_id_at(directory.path(), stored.device_id)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.device_id, stored.device_id);
        assert_eq!(loaded.files.len(), 3);
        assert!(
            load_for_device_id_at(
                directory.path(),
                DeviceId {
                    low: stored.device_id.low ^ 1,
                    high: stored.device_id.high,
                },
            )
            .unwrap()
            .is_none()
        );
    }
}
