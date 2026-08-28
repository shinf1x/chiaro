//! Filesystem helpers shared by the extraction pipeline, the cleanup trainer,
//! and any host application that needs to locate and map `.lri` captures.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use walkdir::WalkDir;

use chiaro::lri::SensorPattern;

/// Memory-map a file read-only.
pub fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: the mapping is read-only and the File remains valid during creation.
    unsafe { Mmap::map(&file) }.with_context(|| format!("memory-map {}", path.display()))
}

/// Return every `.lri` file under `root`, sorted by path.
///
/// With `recursive` set, subdirectories are searched as well; symbolic links
/// are never followed. An error is returned when `root` is not a directory or
/// when no capture is found.
pub fn discover_lri_files(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!("input is not a directory: {}", root.display());
    }
    let walker = if recursive {
        WalkDir::new(root)
    } else {
        WalkDir::new(root).max_depth(1)
    };
    let mut files = Vec::new();
    for entry in walker.follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() && !entry.path().is_file() {
            continue;
        }
        if is_lri_path(entry.path()) {
            files.push(entry.into_path());
        }
    }
    files.sort();
    if files.is_empty() {
        bail!("no .lri files found under {}", root.display());
    }
    Ok(files)
}

/// `true` when the path carries a case-insensitive `.lri` extension.
pub fn is_lri_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("lri"))
}

/// Parse `CAMERA=PATTERN` overrides (for example `A2=MONO` or `B1=RGGB`) into
/// the map accepted by `chiaro::lri::parse_raw_layout`. Camera names are
/// upper-cased.
pub fn parse_pattern_overrides(values: &[String]) -> Result<HashMap<String, SensorPattern>> {
    let mut result = HashMap::new();
    for value in values {
        let (camera, pattern) = value
            .split_once('=')
            .with_context(|| format!("pattern override must be CAMERA=PATTERN: {value}"))?;
        result.insert(camera.to_ascii_uppercase(), SensorPattern::parse(pattern)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn discovers_only_lri_files_and_respects_recursion() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("b.LRI"), b"").unwrap();
        fs::write(dir.path().join("a.lri"), b"").unwrap();
        fs::write(dir.path().join("notes.txt"), b"").unwrap();
        fs::create_dir(dir.path().join("nested")).unwrap();
        fs::write(dir.path().join("nested/c.lri"), b"").unwrap();

        let flat = discover_lri_files(dir.path(), false).unwrap();
        assert_eq!(
            flat,
            vec![dir.path().join("a.lri"), dir.path().join("b.LRI")]
        );

        let deep = discover_lri_files(dir.path(), true).unwrap();
        assert_eq!(deep.len(), 3);
        assert!(deep.contains(&dir.path().join("nested/c.lri")));
    }

    #[test]
    fn pattern_overrides_are_upper_cased_and_validated() {
        let overrides = parse_pattern_overrides(&["a2=mono".to_owned()]).unwrap();
        assert_eq!(overrides.get("A2"), Some(&SensorPattern::Mono));
        assert!(parse_pattern_overrides(&["A2".to_owned()]).is_err());
    }
}
