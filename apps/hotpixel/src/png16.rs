use anyhow::{Context, Result};
use png::{BitDepth, ColorType, Encoder};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|value| value.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    path.with_file_name(name)
}

fn scaled_be_bytes(samples: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        // Exact 10-bit-to-16-bit linear scaling. No stretch, gamma, or normalization.
        bytes.extend_from_slice(&(sample << 6).to_be_bytes());
    }
    bytes
}

pub fn write_gray16_atomic(
    path: &Path,
    width: usize,
    height: usize,
    samples: &[u16],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(path);
    let file =
        File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = Encoder::new(writer, width as u32, height as u32);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::Sixteen);
    let mut png = encoder.write_header()?;
    png.write_image_data(&scaled_be_bytes(samples))?;
    drop(png);
    fs::rename(&temporary, path)
        .with_context(|| format!("rename {} to {}", temporary.display(), path.display()))?;
    Ok(())
}

pub fn write_rgb16_atomic(path: &Path, width: usize, height: usize, samples: &[u16]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(path);
    let file =
        File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = Encoder::new(writer, width as u32, height as u32);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Sixteen);
    let mut png = encoder.write_header()?;
    png.write_image_data(&scaled_be_bytes(samples))?;
    drop(png);
    fs::rename(&temporary, path)
        .with_context(|| format!("rename {} to {}", temporary.display(), path.display()))?;
    Ok(())
}
