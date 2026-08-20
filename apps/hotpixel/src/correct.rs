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
    let mut replacements = Vec::<(usize, u16, i32, bool)>::new();
    let mut candidates = 0usize;

    for index in 0..source.len() {
        let severity = severity_map[index];
        if severity < config.severity_threshold && severity != 255 {
            continue;
        }
        candidates += 1;
        let x = index % width;
        let y = index / width;
        let (prediction, mad) =
            local_prediction_and_mad(&source, width, height, x, y, pattern, config.kernel);
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

pub fn demosaic_bilinear(
    raw: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
) -> Result<Vec<u16>> {
    if pattern == SensorPattern::Mono {
        bail!("monochrome RAW does not require demosaicing");
    }
    if raw.len() != width * height {
        bail!("RAW dimensions do not match sample count");
    }

    let mut rgb = vec![0u16; raw.len() * 3];
    for y in 0..height {
        for x in 0..width {
            let current_color = pattern.color_at(y, x);
            for channel in 0..3usize {
                let value = if current_color == channel {
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
                rgb[(y * width + x) * 3 + channel] = value;
            }
        }
    }
    Ok(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

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
