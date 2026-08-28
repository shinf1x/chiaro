use anyhow::{Result, bail};
use serde::Serialize;

use crate::lri::SensorPattern;

#[derive(Clone, Copy, Debug)]
pub enum CorrectionMode {
    Adaptive,
    Replace,
}

#[derive(Clone, Debug)]
pub struct CorrectionConfig {
    pub mode: CorrectionMode,
    pub severity_threshold: u8,
    pub sigma_threshold: f64,
    pub absolute_threshold: i32,
    pub kernel: usize,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            mode: CorrectionMode::Adaptive,
            severity_threshold: 16,
            sigma_threshold: 6.0,
            absolute_threshold: 4,
            kernel: 5,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CorrectionStats {
    pub candidates: usize,
    pub corrected: usize,
    pub positive_corrected: usize,
    pub negative_corrected: usize,
    pub forced_corrected: usize,
    pub mean_absolute_change: f64,
    pub maximum_absolute_change: u16,
}

fn reflect_index(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let length = length as isize;
    while index < 0 || index >= length {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * length - 2 - index;
        }
    }
    index as usize
}

fn median_u16(values: &mut [u16]) -> u16 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn local_prediction_and_mad(
    raw: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    pattern: SensorPattern,
    kernel: usize,
) -> (u16, u16) {
    let step = if pattern == SensorPattern::Mono { 1 } else { 2 };
    let parity_x = x % step;
    let parity_y = y % step;
    let plane_x = x / step;
    let plane_y = y / step;
    let plane_width = (width - parity_x).div_ceil(step);
    let plane_height = (height - parity_y).div_ceil(step);
    let radius = (kernel / 2) as isize;

    let mut values = [0u16; 49];
    let mut count = 0usize;
    for dy in -radius..=radius {
        let py = reflect_index(plane_y as isize + dy, plane_height);
        let source_y = parity_y + py * step;
        for dx in -radius..=radius {
            let px = reflect_index(plane_x as isize + dx, plane_width);
            let source_x = parity_x + px * step;
            values[count] = raw[source_y * width + source_x];
            count += 1;
        }
    }

    let prediction = median_u16(&mut values[..count]);
    let mut deviations = [0u16; 49];
    for index in 0..count {
        deviations[index] = values[index].abs_diff(prediction);
    }
    let mad = median_u16(&mut deviations[..count]);
    (prediction, mad)
}

pub fn correct_hot_pixels(
    raw: &mut [u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    severity_map: &[u8],
    config: &CorrectionConfig,
) -> Result<CorrectionStats> {
    correct_hot_pixels_with_forced_map(raw, width, height, pattern, severity_map, None, config)
}

pub fn correct_hot_pixels_with_forced_map(
    raw: &mut [u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    severity_map: &[u8],
    forced_map: Option<&[bool]>,
    config: &CorrectionConfig,
) -> Result<CorrectionStats> {
    correct_hot_pixels_threaded(
        raw,
        width,
        height,
        pattern,
        severity_map,
        forced_map,
        config,
        1,
    )
}

/// [`correct_hot_pixels_with_forced_map`] with the candidate scan split across
/// `threads` row bands (`0` = all cores). Every candidate is evaluated against
/// the untouched source frame, so the result does not depend on the split.
#[allow(clippy::too_many_arguments)]
pub fn correct_hot_pixels_threaded(
    raw: &mut [u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    severity_map: &[u8],
    forced_map: Option<&[bool]>,
    config: &CorrectionConfig,
    threads: usize,
) -> Result<CorrectionStats> {
    if raw.len() != width * height || severity_map.len() != raw.len() {
        bail!("RAW and hotpixel map dimensions differ");
    }
    if forced_map.is_some_and(|map| map.len() != raw.len()) {
        bail!("forced hotpixel map dimensions differ");
    }
    if !matches!(config.kernel, 3 | 5 | 7) {
        bail!("correction kernel must be 3, 5, or 7");
    }
    if config.sigma_threshold < 0.0 || config.absolute_threshold < 0 {
        bail!("correction thresholds must be non-negative");
    }

    let source = raw.to_vec();
    let source = &source;
    let band_results = crate::parallel::map_row_bands(height, threads, 1, |rows| {
        let mut replacements = Vec::<(usize, u16, i32, bool)>::new();
        let mut candidates = 0usize;
        for index in rows.start * width..rows.end * width {
            let severity = severity_map[index];
            if severity < config.severity_threshold && severity != 255 {
                continue;
            }
            candidates += 1;
            let x = index % width;
            let y = index / width;
            let (prediction, mad) =
                local_prediction_and_mad(source, width, height, x, y, pattern, config.kernel);
            let delta = source[index] as i32 - prediction as i32;

            let forced = forced_map.is_some_and(|map| map[index]);
            let replace = forced
                || match config.mode {
                    CorrectionMode::Replace => true,
                    CorrectionMode::Adaptive => {
                        let robust_threshold = config.sigma_threshold * f64::from(mad.max(1));
                        let threshold = robust_threshold.max(config.absolute_threshold as f64);
                        if severity == 255 {
                            f64::from(delta.abs()) > threshold
                        } else {
                            f64::from(delta) > threshold
                        }
                    }
                };

            if replace {
                replacements.push((index, prediction, delta, forced));
            }
        }
        (candidates, replacements)
    });
    let candidates = band_results.iter().map(|(count, _)| count).sum::<usize>();
    let replacements = band_results
        .into_iter()
        .flat_map(|(_, replacements)| replacements)
        .collect::<Vec<_>>();

    let mut total_change = 0u64;
    let mut maximum_change = 0u16;
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut forced_corrected = 0usize;
    for (index, prediction, delta, forced) in replacements.iter().copied() {
        let change = source[index].abs_diff(prediction);
        total_change += change as u64;
        maximum_change = maximum_change.max(change);
        if delta > 0 {
            positive += 1;
        } else if delta < 0 {
            negative += 1;
        }
        if forced {
            forced_corrected += 1;
        }
        raw[index] = prediction;
    }

    Ok(CorrectionStats {
        candidates,
        corrected: replacements.len(),
        positive_corrected: positive,
        negative_corrected: negative,
        forced_corrected,
        mean_absolute_change: if replacements.is_empty() {
            0.0
        } else {
            total_change as f64 / replacements.len() as f64
        },
        maximum_absolute_change: maximum_change,
    })
}

/// Simple linear bilinear demosaic to interleaved RGB, same sample scale as
/// the input. Every missing channel is the rounded mean of the same-colour
/// pixels in the 3x3 neighbourhood; at the frame border only the neighbours
/// that exist are averaged.
pub fn demosaic_bilinear(
    raw: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
) -> Result<Vec<u16>> {
    demosaic_bilinear_threaded(raw, width, height, pattern, 1)
}

/// [`demosaic_bilinear`] with the rows split across `threads` workers
/// (`0` = all cores). Output is identical for every thread count.
pub fn demosaic_bilinear_threaded(
    raw: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    threads: usize,
) -> Result<Vec<u16>> {
    if pattern == SensorPattern::Mono {
        bail!("monochrome RAW does not require demosaicing");
    }
    if raw.len() != width * height {
        bail!("RAW dimensions do not match sample count");
    }
    let mut rgb = vec![0u16; raw.len() * 3];
    crate::parallel::map_row_bands_mut(&mut rgb, width * 3, threads, 2, |rows, band| {
        demosaic_rows(raw, width, height, pattern, rows, band);
    });
    Ok(rgb)
}

crate::simd::multiversion! {
/// Demosaic rows `rows` of the mosaic into `rgb`, which must hold exactly
/// `rows.len() * width * 3` samples. This is the building block for streaming
/// encoders that never hold the whole RGB frame in memory.
///
/// Interior pixels use a phase-specialised kernel: in a 2x2 Bayer cell the
/// neighbours of each missing colour sit at fixed offsets, so each output row
/// is produced from three input rows without per-pixel colour lookups. The
/// one-pixel border falls back to the generic neighbourhood search, so both
/// paths compute exactly the same rounded means.
pub fn demosaic_rows(
    raw: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    rows: std::ops::Range<usize>,
    rgb: &mut [u16],
) {
    assert_eq!(rgb.len(), rows.len() * width * 3, "RGB band size");
    assert!(raw.len() == width * height && pattern != SensorPattern::Mono);
    for (row_offset, y) in rows.enumerate() {
        let out = &mut rgb[row_offset * width * 3..(row_offset + 1) * width * 3];
        if y == 0 || y + 1 == height || width < 3 {
            for x in 0..width {
                out[x * 3..x * 3 + 3]
                    .copy_from_slice(&generic_pixel(raw, width, height, pattern, x, y));
            }
            continue;
        }
        let above = &raw[(y - 1) * width..y * width];
        let current = &raw[y * width..(y + 1) * width];
        let below = &raw[(y + 1) * width..(y + 2) * width];
        out[..3].copy_from_slice(&generic_pixel(raw, width, height, pattern, 0, y));
        out[(width - 1) * 3..].copy_from_slice(&generic_pixel(
            raw,
            width,
            height,
            pattern,
            width - 1,
            y,
        ));

        for x in 1..width - 1 {
            let colour = pattern.color_at(y, x);
            let (a, c, b) = (above, current, below);
            let pixel = &mut out[x * 3..x * 3 + 3];
            if colour == 1 {
                // Green pixel: one chroma lives left/right, the other above/below.
                let horizontal = pattern.color_at(y, x + 1);
                let vertical = pattern.color_at(y + 1, x);
                pixel[1] = c[x];
                pixel[horizontal] = mean2(c[x - 1], c[x + 1]);
                pixel[vertical] = mean2(a[x], b[x]);
            } else {
                // Red or blue pixel: green on the cross, the other chroma on the diagonals.
                pixel[colour] = c[x];
                pixel[1] = mean4(a[x], c[x - 1], c[x + 1], b[x]);
                pixel[2 - colour] = mean4(a[x - 1], a[x + 1], b[x - 1], b[x + 1]);
            }
        }
    }
}
}

#[inline]
fn mean2(first: u16, second: u16) -> u16 {
    (u32::from(first) + u32::from(second)).div_ceil(2) as u16
}

#[inline]
fn mean4(first: u16, second: u16, third: u16, fourth: u16) -> u16 {
    ((u32::from(first) + u32::from(second) + u32::from(third) + u32::from(fourth) + 2) / 4) as u16
}

/// Reference per-pixel kernel: rounded mean over the existing same-colour
/// neighbours in the 3x3 window. Used for borders and in tests.
fn generic_pixel(
    raw: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    x: usize,
    y: usize,
) -> [u16; 3] {
    let current_color = pattern.color_at(y, x);
    let mut pixel = [0u16; 3];
    for (channel, value) in pixel.iter_mut().enumerate() {
        *value = if current_color == channel {
            raw[y * width + x]
        } else {
            let mut sum = 0u32;
            let mut count = 0u32;
            for dy in -1isize..=1 {
                let ny = y as isize + dy;
                if ny < 0 || ny >= height as isize {
                    continue;
                }
                for dx in -1isize..=1 {
                    let nx = x as isize + dx;
                    if nx < 0 || nx >= width as isize {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if pattern.color_at(ny, nx) == channel {
                        sum += raw[ny * width + nx] as u32;
                        count += 1;
                    }
                }
            }
            (sum + count / 2)
                .checked_div(count)
                .map_or(raw[y * width + x], |value| value as u16)
        };
    }
    pixel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_kernel_matches_the_generic_neighbourhood_mean_for_every_pattern() {
        let (width, height) = (13, 9);
        let raw = (0..width * height)
            .map(|index| ((index * 2654435761usize) % 1024) as u16)
            .collect::<Vec<_>>();
        for pattern in [
            SensorPattern::Rggb,
            SensorPattern::Grbg,
            SensorPattern::Gbrg,
            SensorPattern::Bggr,
        ] {
            let mut expected = Vec::with_capacity(raw.len() * 3);
            for y in 0..height {
                for x in 0..width {
                    expected.extend_from_slice(&generic_pixel(&raw, width, height, pattern, x, y));
                }
            }
            for threads in [1, 2, 5] {
                let actual =
                    demosaic_bilinear_threaded(&raw, width, height, pattern, threads).unwrap();
                assert_eq!(actual, expected, "{pattern:?} threads={threads}");
            }
        }
    }

    #[test]
    fn corrects_only_factory_guided_positive_bayer_outlier() {
        let width = 10;
        let height = 10;
        let mut raw = vec![100u16; width * height];
        let hot = 4 * width + 4;
        raw[hot] = 500;
        raw[5 * width + 5] = 500; // Real bright structure, absent from factory map.
        let mut map = vec![0u8; raw.len()];
        map[hot] = 64;
        let stats = correct_hot_pixels(
            &mut raw,
            width,
            height,
            SensorPattern::Rggb,
            &map,
            &CorrectionConfig::default(),
        )
        .unwrap();
        assert_eq!(stats.corrected, 1);
        assert_eq!(raw[hot], 100);
        assert_eq!(raw[5 * width + 5], 500);
    }

    #[test]
    fn sentinel_255_can_replace_a_dead_pixel() {
        let width = 9;
        let height = 9;
        let mut raw = vec![200u16; width * height];
        let index = 4 * width + 4;
        raw[index] = 0;
        let mut map = vec![0u8; raw.len()];
        map[index] = 255;
        let stats = correct_hot_pixels(
            &mut raw,
            width,
            height,
            SensorPattern::Mono,
            &map,
            &CorrectionConfig::default(),
        )
        .unwrap();
        assert_eq!(stats.corrected, 1);
        assert_eq!(raw[index], 200);
    }

    #[test]
    fn temperature_active_factory_pixel_forces_local_replacement() {
        let width = 9;
        let height = 9;
        let mut raw = vec![200u16; width * height];
        let index = 4 * width + 4;
        raw[index] = 202;
        let mut severity = vec![0u8; raw.len()];
        severity[index] = 64;
        let mut active = vec![false; raw.len()];
        active[index] = true;
        let stats = correct_hot_pixels_with_forced_map(
            &mut raw,
            width,
            height,
            SensorPattern::Mono,
            &severity,
            Some(&active),
            &CorrectionConfig::default(),
        )
        .unwrap();
        assert_eq!(stats.corrected, 1);
        assert_eq!(stats.forced_corrected, 1);
        assert_eq!(raw[index], 200);
    }

    #[test]
    fn bilinear_demosaic_preserves_constant_channels() {
        let width = 8;
        let height = 8;
        let pattern = SensorPattern::Rggb;
        let mut raw = vec![0u16; width * height];
        for y in 0..height {
            for x in 0..width {
                raw[y * width + x] = match pattern.color_at(y, x) {
                    0 => 100,
                    1 => 200,
                    2 => 300,
                    _ => unreachable!(),
                };
            }
        }
        let rgb = demosaic_bilinear(&raw, width, height, pattern).unwrap();
        for pixel in rgb.chunks_exact(3) {
            assert_eq!(pixel, &[100, 200, 300]);
        }
    }
}
