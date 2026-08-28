//! Free-space queries for export destinations.

use std::{
    io,
    path::{Path, PathBuf},
};

/// Bytes available to this user on the volume that will hold `path`.
///
/// `path` does not need to exist yet: the nearest existing ancestor is used,
/// which matches where a new export folder would be created.
pub fn available_space(path: &Path) -> io::Result<u64> {
    let probe = nearest_existing(path)?;
    available_space_at(&probe)
}

fn nearest_existing(path: &Path) -> io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut candidate = absolute.as_path();
    loop {
        if candidate.exists() {
            return Ok(candidate.to_path_buf());
        }
        match candidate.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => candidate = parent,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("no existing ancestor for {}", path.display()),
                ));
            }
        }
    }
}

#[cfg(unix)]
fn available_space_at(path: &Path) -> io::Result<u64> {
    use std::{ffi::CString, mem::MaybeUninit, os::unix::ffi::OsStrExt};

    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
    let mut stats = MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is a valid NUL-terminated string and `stats` is a
    // writable buffer of the exact type statvfs fills in.
    let result = unsafe { libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: statvfs returned success, so the struct is initialised.
    let stats = unsafe { stats.assume_init() };
    // Field widths differ between libc targets; widen explicitly.
    #[allow(clippy::useless_conversion, clippy::unnecessary_cast)]
    let available = (stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64);
    Ok(available)
}

#[cfg(windows)]
fn available_space_at(path: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<u16>>();
    let mut available = 0u64;
    // SAFETY: `wide` is NUL-terminated and `available` is a valid out pointer;
    // the optional totals are allowed to be null.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(available)
}

#[cfg(not(any(unix, windows)))]
fn available_space_at(_path: &Path) -> io::Result<u64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "free-space query is not supported on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_space_for_a_not_yet_created_folder() {
        let dir = tempfile::tempdir().unwrap();
        let existing = available_space(dir.path()).unwrap();
        let future = available_space(&dir.path().join("new/export/frames")).unwrap();
        assert!(existing > 0);
        // Both probes resolve to the same volume; allow for concurrent writes.
        let delta = existing.abs_diff(future);
        assert!(delta < existing / 2 + 1);
    }

    #[test]
    fn relative_paths_resolve_against_the_working_directory() {
        assert!(available_space(Path::new("some-new-folder")).is_ok());
    }
}
