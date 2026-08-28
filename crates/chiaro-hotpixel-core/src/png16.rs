//! Linear 16-bit PNG output with band-parallel compression.
//!
//! A frame is written as a stream of row bands. Worker threads render a band
//! (demosaic, byte order), apply the PNG `Sub` filter, and deflate it
//! independently; the writer thread stitches the results into one ordinary
//! zlib stream. Independent bands are possible because each band is flushed
//! at a deflate block boundary (the same trick `pigz` uses), and the Adler-32
//! checksums of the bands are combined arithmetically. Any PNG decoder reads
//! the result; nothing about the format is non-standard.
//!
//! Only a few bands are alive at once, so a full-resolution RGB frame costs a
//! few megabytes of working memory instead of two 78 MB buffers. Files are
//! written to a `.part` sibling and renamed into place.

use anyhow::{Context, Result, bail};
use flate2::{Compress, Compression, FlushCompress, Status};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use crate::parallel;

/// Rows per band. 64 rows of 16-bit RGB at L16 width are about 1.6 MB, large
/// enough that independent compression costs almost nothing in file size.
const BAND_ROWS: usize = 64;
/// Output buffer in front of the file.
const WRITE_BUFFER: usize = 1 << 20;
/// Largest IDAT chunk written; a compressed band may span several.
const IDAT_CHUNK_BYTES: usize = 1 << 20;
/// Deflate level. Measured on real L16 frames: level 1 writes ~52 MB in the
/// same time level 2 writes ~33 MB; levels above 3 are several times slower
/// for about 1 MB more saving.
pub const DEFAULT_DEFLATE_LEVEL: u32 = 2;
const ADLER_BASE: u32 = 65521;

/// PNG colour types this writer produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PngColor {
    Gray16,
    Rgb16,
}

impl PngColor {
    fn channels(self) -> usize {
        match self {
            Self::Gray16 => 1,
            Self::Rgb16 => 3,
        }
    }

    fn ihdr_type(self) -> u8 {
        match self {
            Self::Gray16 => 0,
            Self::Rgb16 => 2,
        }
    }
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|value| value.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    path.with_file_name(name)
}

/// Convert 16-bit samples to PNG's big-endian byte order.
pub fn samples_to_be_bytes(samples: &[u16], bytes: &mut [u8]) {
    assert_eq!(bytes.len(), samples.len() * 2);
    for (chunk, &sample) in bytes.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
        *chunk = sample.to_be_bytes();
    }
}

/// Write a 16-bit PNG whose rows are produced on demand.
///
/// `render` fills the big-endian sample bytes for the rows `range`; it is
/// called from several threads for different bands. `threads` is the number
/// of workers (`0` = all cores). Samples are written as-is: pass values
/// already scaled to 16 bits.
pub fn write_png16_streaming_atomic(
    path: &Path,
    width: usize,
    height: usize,
    color: PngColor,
    threads: usize,
    render: impl Fn(Range<usize>, &mut [u8]) + Sync,
) -> Result<()> {
    write_png16_streaming_atomic_with_level(
        path,
        width,
        height,
        color,
        threads,
        DEFAULT_DEFLATE_LEVEL,
        render,
    )
}

/// [`write_png16_streaming_atomic`] with an explicit deflate level (0-9).
pub fn write_png16_streaming_atomic_with_level(
    path: &Path,
    width: usize,
    height: usize,
    color: PngColor,
    threads: usize,
    level: u32,
    render: impl Fn(Range<usize>, &mut [u8]) + Sync,
) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("PNG dimensions must be non-zero");
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(path);
    let file =
        File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    let mut out = BufWriter::with_capacity(WRITE_BUFFER, file);
    let result = write_png16_body(&mut out, width, height, color, threads, level, &render)
        .and_then(|()| out.flush().context("flush PNG"));
    drop(out);
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    fs::rename(&temporary, path)
        .with_context(|| format!("rename {} to {}", temporary.display(), path.display()))?;
    Ok(())
}

/// One compressed band ready for the writer.
struct CompressedBand {
    index: usize,
    deflate: Vec<u8>,
    adler: u32,
    filtered_len: usize,
}

fn write_png16_body(
    out: &mut impl Write,
    width: usize,
    height: usize,
    color: PngColor,
    threads: usize,
    level: u32,
    render: &(impl Fn(Range<usize>, &mut [u8]) + Sync),
) -> Result<()> {
    let bytes_per_pixel = color.channels() * 2;
    let row_bytes = width * bytes_per_pixel;
    let bands = parallel::row_bands(height, height.div_ceil(BAND_ROWS).max(1), BAND_ROWS);
    let workers = parallel::resolve_threads(threads).min(bands.len()).max(1);
    let last_band = bands.len() - 1;

    write_signature_and_header(out, width, height, color)?;

    // Each worker owns its render and filter scratch buffers for the whole
    // run; only compressed bands travel to the writer, so memory is bounded by
    // `workers` times a few megabytes regardless of frame size.
    let (band_tx, band_rx) = mpsc::channel::<Result<CompressedBand>>();
    let next_band = AtomicUsize::new(0);

    std::thread::scope(|scope| -> Result<()> {
        let bands = &bands;
        let next_band = &next_band;
        for _ in 0..workers {
            let band_tx = band_tx.clone();
            scope.spawn(move || {
                let mut pixels = vec![0u8; row_bytes * BAND_ROWS];
                let mut filtered = Vec::with_capacity((row_bytes + 1) * BAND_ROWS);
                loop {
                    let index = next_band.fetch_add(1, Ordering::Relaxed);
                    let Some(range) = bands.get(index) else {
                        break;
                    };
                    let pixels = &mut pixels[..range.len() * row_bytes];
                    render(range.clone(), pixels);
                    sub_filter_rows(pixels, row_bytes, bytes_per_pixel, &mut filtered);
                    let flush = if index == last_band {
                        FlushCompress::Finish
                    } else {
                        FlushCompress::Sync
                    };
                    let result =
                        deflate_band(&filtered, level, flush).map(|deflate| CompressedBand {
                            index,
                            deflate,
                            adler: adler32(&filtered),
                            filtered_len: filtered.len(),
                        });
                    if band_tx.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        drop(band_tx);

        // Writer: emit IDAT chunks in band order. The first carries the zlib
        // header; the last is followed by the combined Adler-32.
        let mut pending = BTreeMap::new();
        let mut expected = 0usize;
        let mut adler = 1u32;
        let mut idat = Vec::with_capacity(row_bytes * BAND_ROWS);
        idat.extend_from_slice(&[0x78, 0x9c]);
        for received in band_rx {
            let band = received?;
            pending.insert(band.index, band);
            while let Some(band) = pending.remove(&expected) {
                idat.extend_from_slice(&band.deflate);
                adler = adler32_combine(adler, band.adler, band.filtered_len);
                if expected == last_band {
                    idat.extend_from_slice(&adler.to_be_bytes());
                }
                // Some readers warn about IDAT chunks above a few MB; split.
                for piece in idat.chunks(IDAT_CHUNK_BYTES) {
                    write_chunk(out, b"IDAT", piece)?;
                }
                idat.clear();
                expected += 1;
            }
        }
        if expected != bands.len() {
            bail!("PNG writer received {expected} of {} bands", bands.len());
        }
        write_chunk(out, b"IEND", &[])?;
        Ok(())
    })
}

fn write_signature_and_header(
    out: &mut impl Write,
    width: usize,
    height: usize,
    color: PngColor,
) -> Result<()> {
    out.write_all(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a])?;
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&u32::try_from(width).context("PNG width")?.to_be_bytes());
    ihdr.extend_from_slice(&u32::try_from(height).context("PNG height")?.to_be_bytes());
    ihdr.push(16); // bit depth
    ihdr.push(color.ihdr_type());
    ihdr.extend_from_slice(&[0, 0, 0]); // deflate, adaptive filtering, no interlace
    write_chunk(out, b"IHDR", &ihdr)
}

fn write_chunk(out: &mut impl Write, kind: &[u8; 4], data: &[u8]) -> Result<()> {
    out.write_all(
        &u32::try_from(data.len())
            .context("PNG chunk length")?
            .to_be_bytes(),
    )?;
    out.write_all(kind)?;
    out.write_all(data)?;
    let mut crc = crc32fast::Hasher::new();
    crc.update(kind);
    crc.update(data);
    out.write_all(&crc.finalize().to_be_bytes())?;
    Ok(())
}

/// Prefix every row with the `Sub` filter byte and subtract the pixel to the
/// left. `Sub` is cheap and suits smooth 16-bit astrophotography data well.
fn sub_filter_rows(
    pixels: &[u8],
    row_bytes: usize,
    bytes_per_pixel: usize,
    filtered: &mut Vec<u8>,
) {
    filtered.clear();
    for row in pixels.chunks_exact(row_bytes) {
        filtered.push(1);
        filtered.extend_from_slice(&row[..bytes_per_pixel]);
        filtered.extend(
            row[bytes_per_pixel..]
                .iter()
                .zip(row)
                .map(|(current, left)| current.wrapping_sub(*left)),
        );
    }
}

/// Raw deflate of one band. `Sync` ends on a byte boundary so the next band's
/// independent stream can follow; `Finish` writes the final block.
fn deflate_band(input: &[u8], level: u32, flush: FlushCompress) -> Result<Vec<u8>> {
    let mut compressor = Compress::new(Compression::new(level.min(9)), false);
    let mut output = Vec::with_capacity(input.len() / 2 + 1024);
    loop {
        let consumed = compressor.total_in() as usize;
        if output.capacity() == output.len() {
            output.reserve(output.len().max(4096));
        }
        let status = compressor
            .compress_vec(&input[consumed..], &mut output, flush)
            .context("deflate band")?;
        let done = compressor.total_in() as usize == input.len()
            && match flush {
                FlushCompress::Finish => status == Status::StreamEnd,
                // A sync flush is complete once the compressor has spare
                // output room left over, i.e. it did not stop for lack of space.
                _ => output.len() < output.capacity(),
            };
        if done {
            return Ok(output);
        }
    }
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    // 5552 is the largest block that cannot overflow u32 between reductions.
    for block in data.chunks(5552) {
        for &byte in block {
            a += u32::from(byte);
            b += a;
        }
        a %= ADLER_BASE;
        b %= ADLER_BASE;
    }
    (b << 16) | a
}

/// Adler-32 of `A ++ B` from `adler32(A)`, `adler32(B)`, and `len(B)`.
fn adler32_combine(first: u32, second: u32, second_len: usize) -> u32 {
    let remainder = (second_len % ADLER_BASE as usize) as u32;
    let mut sum1 = first & 0xffff;
    let mut sum2 = (remainder * sum1) % ADLER_BASE;
    sum1 += (second & 0xffff) + ADLER_BASE - 1;
    sum2 += ((first >> 16) & 0xffff) + ((second >> 16) & 0xffff) + ADLER_BASE - remainder;
    if sum1 >= ADLER_BASE {
        sum1 -= ADLER_BASE;
    }
    if sum1 >= ADLER_BASE {
        sum1 -= ADLER_BASE;
    }
    if sum2 >= ADLER_BASE << 1 {
        sum2 -= ADLER_BASE << 1;
    }
    if sum2 >= ADLER_BASE {
        sum2 -= ADLER_BASE;
    }
    (sum2 << 16) | sum1
}

fn write_samples_atomic(
    path: &Path,
    width: usize,
    height: usize,
    color: PngColor,
    samples: &[u16],
) -> Result<()> {
    let channels = color.channels();
    if samples.len() != width * height * channels {
        bail!(
            "PNG sample count {} does not match {width}x{height}x{channels}",
            samples.len()
        );
    }
    let row_len = width * channels;
    write_png16_streaming_atomic(path, width, height, color, 0, |rows, bytes| {
        samples_to_be_bytes(&samples[rows.start * row_len..rows.end * row_len], bytes);
    })
}

/// Write 10-bit RAW codes as a grayscale PNG scaled exactly to 16 bits (`<< 6`).
pub fn write_gray16_atomic(
    path: &Path,
    width: usize,
    height: usize,
    samples: &[u16],
) -> Result<()> {
    let scaled = samples.iter().map(|s| s << 6).collect::<Vec<_>>();
    write_gray16_native_atomic(path, width, height, &scaled)
}

/// Write 10-bit interleaved RGB codes as a PNG scaled exactly to 16 bits.
pub fn write_rgb16_atomic(path: &Path, width: usize, height: usize, samples: &[u16]) -> Result<()> {
    let scaled = samples.iter().map(|s| s << 6).collect::<Vec<_>>();
    write_rgb16_native_atomic(path, width, height, &scaled)
}

/// Write 16-bit grayscale samples unchanged.
pub fn write_gray16_native_atomic(
    path: &Path,
    width: usize,
    height: usize,
    samples: &[u16],
) -> Result<()> {
    write_samples_atomic(path, width, height, PngColor::Gray16, samples)
}

/// Write 16-bit interleaved RGB samples unchanged.
pub fn write_rgb16_native_atomic(
    path: &Path,
    width: usize,
    height: usize,
    samples: &[u16],
) -> Result<()> {
    write_samples_atomic(path, width, height, PngColor::Rgb16, samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(path: &Path) -> (png::OutputInfo, Vec<u8>) {
        let decoder = png::Decoder::new(File::open(path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buffer = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buffer).unwrap();
        buffer.truncate(info.buffer_size());
        (info, buffer)
    }

    fn noisy(count: usize) -> Vec<u16> {
        (0..count)
            .map(|index| ((index as u64 * 2654435761) % 65536) as u16)
            .collect()
    }

    #[test]
    fn banded_rgb_png_decodes_to_the_exact_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rgb.png");
        let (width, height) = (37, 3 * BAND_ROWS + 5);
        let samples = noisy(width * height * 3);
        write_rgb16_native_atomic(&path, width, height, &samples).unwrap();
        let (info, bytes) = decode(&path);
        assert_eq!((info.width, info.height), (width as u32, height as u32));
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        let mut expected = vec![0u8; samples.len() * 2];
        samples_to_be_bytes(&samples, &mut expected);
        assert_eq!(bytes, expected);
        assert!(!temporary_path(&path).exists());
    }

    #[test]
    fn every_level_and_thread_count_produces_a_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let (width, height) = (9, 2 * BAND_ROWS + 1);
        let samples = noisy(width * height);
        let mut expected = vec![0u8; samples.len() * 2];
        samples_to_be_bytes(&samples, &mut expected);
        for level in [0, 1, 2, 6, 9] {
            for threads in [1, 3] {
                let path = dir.path().join(format!("gray-{level}-{threads}.png"));
                write_png16_streaming_atomic_with_level(
                    &path,
                    width,
                    height,
                    PngColor::Gray16,
                    threads,
                    level,
                    |rows, bytes| {
                        samples_to_be_bytes(&samples[rows.start * width..rows.end * width], bytes)
                    },
                )
                .unwrap();
                let (info, bytes) = decode(&path);
                assert_eq!(info.color_type, png::ColorType::Grayscale);
                assert_eq!(bytes, expected, "level {level} threads {threads}");
            }
        }
    }

    #[test]
    fn adler_combine_matches_a_single_pass() {
        let data = noisy(40_000)
            .into_iter()
            .flat_map(u16::to_be_bytes)
            .collect::<Vec<_>>();
        for split in [0, 1, 5551, 5552, 12_345, data.len()] {
            let (left, right) = data.split_at(split);
            assert_eq!(
                adler32_combine(adler32(left), adler32(right), right.len()),
                adler32(&data),
                "split {split}"
            );
        }
    }

    #[test]
    fn wrong_sample_count_is_rejected_and_leaves_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.png");
        assert!(write_rgb16_native_atomic(&path, 4, 4, &[0; 10]).is_err());
        assert!(!path.exists());
        assert!(!temporary_path(&path).exists());
    }
}
