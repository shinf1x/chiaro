//! Selectable Bayer demosaicing.
//!
//! The implementations in this module are clean-room Rust implementations
//! based on the published algorithm descriptions. They do not incorporate the
//! GPL implementations shipped by RawTherapee, darktable, or LibRaw.

use std::{fmt, str::FromStr};

use anyhow::{Result, bail};
use chiaro::lri::SensorPattern;
use serde::{Deserialize, Serialize};

use crate::{correct::demosaic_bilinear_threaded, parallel::map_row_bands_mut};

/// Bayer reconstruction method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DemosaicMethod {
    /// Existing 3x3 same-colour bilinear interpolation.
    Simple,
    /// Edge-directed, alias-suppressing reconstruction for ordinary captures.
    #[default]
    Amaze,
    /// Ratio-corrected chroma reconstruction for individual night photos.
    Rcd,
    /// Directional local minimum-mean-square-error reconstruction for noisy
    /// frames used in night stacks.
    Lmmse,
    /// Integrated Gaussian-vector colour-difference reconstruction for noisy
    /// frames used in night stacks and moire-prone detail.
    Igv,
}

impl DemosaicMethod {
    pub const ALL: [Self; 5] = [Self::Simple, Self::Amaze, Self::Rcd, Self::Lmmse, Self::Igv];

    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "Simple",
            Self::Amaze => "AMaZE",
            Self::Rcd => "RCD",
            Self::Lmmse => "LMMSE",
            Self::Igv => "IGV",
        }
    }

    pub fn recommendation(self) -> &'static str {
        match self {
            Self::Simple => "fast previews and compatibility",
            Self::Amaze => "general photography (default)",
            Self::Rcd => "individual night photos",
            Self::Lmmse => "night stacks; noise-tolerant",
            Self::Igv => "night stacks; noise and moire-tolerant",
        }
    }
}

impl fmt::Display for DemosaicMethod {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

impl FromStr for DemosaicMethod {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "simple" | "bilinear" => Ok(Self::Simple),
            "amaze" => Ok(Self::Amaze),
            "rcd" => Ok(Self::Rcd),
            "lmmse" => Ok(Self::Lmmse),
            "igv" => Ok(Self::Igv),
            _ => Err(format!(
                "unknown demosaicing method {value:?}; expected simple, amaze, rcd, lmmse, or igv"
            )),
        }
    }
}

/// Reconstruct one Bayer mosaic as interleaved linear RGB16.
pub fn demosaic(
    raw: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    method: DemosaicMethod,
    threads: usize,
) -> Result<Vec<u16>> {
    if pattern == SensorPattern::Mono {
        bail!("monochrome RAW does not require demosaicing");
    }
    if width == 0 || height == 0 || raw.len() != width.saturating_mul(height) {
        bail!("RAW dimensions do not match sample count");
    }
    if method == DemosaicMethod::Simple || width < 9 || height < 9 {
        return demosaic_bilinear_threaded(raw, width, height, pattern, threads);
    }

    let mut rgb = demosaic_bilinear_threaded(raw, width, height, pattern, threads)?;
    let mut green = vec![0.0f32; raw.len()];
    map_row_bands_mut(&mut green, width, threads, 2, |rows, band| {
        for (local_y, y) in rows.enumerate() {
            for x in 0..width {
                let index = y * width + x;
                band[local_y * width + x] = if pattern.color_at(y, x) == 1 {
                    f32::from(raw[index])
                } else if x < 4 || y < 4 || x + 4 >= width || y + 4 >= height {
                    f32::from(rgb[index * 3 + 1])
                } else {
                    estimate_green(raw, width, x, y, method)
                };
            }
        }
    });

    let reconstruct = |channel: usize| {
        let mut plane = vec![0.0f32; raw.len()];
        map_row_bands_mut(&mut plane, width, threads, 2, |rows, band| {
            for (local_y, y) in rows.enumerate() {
                for x in 0..width {
                    let index = y * width + x;
                    band[local_y * width + x] = if pattern.color_at(y, x) == channel {
                        f32::from(raw[index])
                    } else if x < 4 || y < 4 || x + 4 >= width || y + 4 >= height {
                        f32::from(rgb[index * 3 + channel])
                    } else {
                        estimate_chroma(raw, &green, width, height, pattern, x, y, channel, method)
                    };
                }
            }
        });
        plane
    };
    let red = reconstruct(0);
    let blue = reconstruct(2);

    map_row_bands_mut(&mut rgb, width * 3, threads, 2, |rows, band| {
        for (local_y, y) in rows.enumerate() {
            for x in 0..width {
                let index = y * width + x;
                let output = local_y * width * 3 + x * 3;
                band[output] = quantize(red[index]);
                band[output + 1] = quantize(green[index]);
                band[output + 2] = quantize(blue[index]);
            }
        }
    });
    Ok(rgb)
}

#[inline]
fn raw_at(raw: &[u16], width: usize, x: isize, y: isize) -> f32 {
    f32::from(raw[y as usize * width + x as usize])
}

fn directional_candidates(raw: &[u16], width: usize, x: usize, y: usize) -> (f32, f32, f32, f32) {
    let (x, y) = (x as isize, y as isize);
    let center = raw_at(raw, width, x, y);
    let left = raw_at(raw, width, x - 1, y);
    let right = raw_at(raw, width, x + 1, y);
    let up = raw_at(raw, width, x, y - 1);
    let down = raw_at(raw, width, x, y + 1);
    let left2 = raw_at(raw, width, x - 2, y);
    let right2 = raw_at(raw, width, x + 2, y);
    let up2 = raw_at(raw, width, x, y - 2);
    let down2 = raw_at(raw, width, x, y + 2);
    let gh = 0.5 * (left + right) + 0.25 * (2.0 * center - left2 - right2);
    let gv = 0.5 * (up + down) + 0.25 * (2.0 * center - up2 - down2);
    let dh =
        (left - right).abs() + 0.5 * (left2 - right2).abs() + (2.0 * center - left2 - right2).abs();
    let dv = (up - down).abs() + 0.5 * (up2 - down2).abs() + (2.0 * center - up2 - down2).abs();
    (gh, gv, dh, dv)
}

fn estimate_green(raw: &[u16], width: usize, x: usize, y: usize, method: DemosaicMethod) -> f32 {
    let (gh, gv, mut dh, mut dv) = directional_candidates(raw, width, x, y);
    match method {
        DemosaicMethod::Amaze => {
            // Extend the decision over the neighbouring same-colour sites.
            for offset in [-2isize, 2] {
                let (_, _, h, _) =
                    directional_candidates(raw, width, (x as isize + offset) as usize, y);
                let (_, _, _, v) =
                    directional_candidates(raw, width, x, (y as isize + offset) as usize);
                dh += 0.25 * h;
                dv += 0.25 * v;
            }
            variance_fusion(gh, gv, dh * dh, dv * dv)
        }
        DemosaicMethod::Rcd => {
            // RCD benefits from a decisive but bounded green estimate around
            // compact high-contrast objects.
            let value = variance_fusion(gh, gv, dh * dh, dv * dv);
            let (x, y) = (x as isize, y as isize);
            let low = raw_at(raw, width, x - 1, y)
                .min(raw_at(raw, width, x + 1, y))
                .min(raw_at(raw, width, x, y - 1))
                .min(raw_at(raw, width, x, y + 1));
            let high = raw_at(raw, width, x - 1, y)
                .max(raw_at(raw, width, x + 1, y))
                .max(raw_at(raw, width, x, y - 1))
                .max(raw_at(raw, width, x, y + 1));
            value.clamp(low, high)
        }
        DemosaicMethod::Lmmse => {
            let mut error_h = 0.0;
            let mut error_v = 0.0;
            for offset in -2isize..=2 {
                let nx = (x as isize + offset) as usize;
                let ny = (y as isize + offset) as usize;
                let (eh, _, _, _) = directional_candidates(raw, width, nx, y);
                let (_, ev, _, _) = directional_candidates(raw, width, x, ny);
                error_h += (eh - gh).powi(2);
                error_v += (ev - gv).powi(2);
            }
            variance_fusion(gh, gv, error_h + dh * dh, error_v + dv * dv)
        }
        DemosaicMethod::Igv => {
            let (x, y) = (x as isize, y as isize);
            let mut horizontal = 0.0;
            let mut vertical = 0.0;
            let mut wh = 0.0;
            let mut wv = 0.0;
            for (distance, gaussian) in [(1isize, 0.606_530_67), (3, 0.011_109)] {
                horizontal += gaussian
                    * (raw_at(raw, width, x - distance, y) + raw_at(raw, width, x + distance, y));
                vertical += gaussian
                    * (raw_at(raw, width, x, y - distance) + raw_at(raw, width, x, y + distance));
                wh += 2.0 * gaussian;
                wv += 2.0 * gaussian;
            }
            let integrated_h = 0.65 * gh + 0.35 * horizontal / wh;
            let integrated_v = 0.65 * gv + 0.35 * vertical / wv;
            variance_fusion(integrated_h, integrated_v, dh * dh, dv * dv)
        }
        DemosaicMethod::Simple => unreachable!(),
    }
    .clamp(0.0, 65535.0)
}

#[inline]
fn variance_fusion(horizontal: f32, vertical: f32, error_h: f32, error_v: f32) -> f32 {
    let floor = 1.0;
    let weight_h = 1.0 / (error_h + floor);
    let weight_v = 1.0 / (error_v + floor);
    (horizontal * weight_h + vertical * weight_v) / (weight_h + weight_v)
}

#[allow(clippy::too_many_arguments)]
fn estimate_chroma(
    raw: &[u16],
    green: &[f32],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    x: usize,
    y: usize,
    channel: usize,
    method: DemosaicMethod,
) -> f32 {
    let center = y * width + x;
    let center_green = green[center];
    let radius = match method {
        DemosaicMethod::Lmmse | DemosaicMethod::Igv => 4isize,
        _ => 2,
    };
    let mut candidates = Vec::with_capacity(16);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            if pattern.color_at(ny, nx) != channel {
                continue;
            }
            let index = ny * width + nx;
            let distance2 = (dx * dx + dy * dy) as f32;
            let green_delta = (green[index] - center_green).abs();
            let spatial = match method {
                DemosaicMethod::Lmmse => (-distance2 / 8.0).exp(),
                DemosaicMethod::Igv => (-distance2 / 5.0).exp(),
                _ => 1.0 / (1.0 + distance2),
            };
            let edge = 1.0 / (1.0 + green_delta / 256.0);
            candidates.push((f32::from(raw[index]), green[index], spatial * edge));
        }
    }
    if candidates.is_empty() {
        return center_green;
    }

    match method {
        DemosaicMethod::Rcd => {
            let signal_floor = 256.0;
            let (ratio_sum, difference_sum, weight_sum) = candidates.iter().fold(
                (0.0, 0.0, 0.0),
                |(ratio, difference, weight), &(value, neighbour_green, w)| {
                    (
                        ratio + w * value / neighbour_green.max(signal_floor),
                        difference + w * (value - neighbour_green),
                        weight + w,
                    )
                },
            );
            let ratio_estimate = center_green * ratio_sum / weight_sum;
            let difference_estimate = center_green + difference_sum / weight_sum;
            let ratio_weight = (center_green / 4096.0).clamp(0.0, 1.0);
            ratio_estimate * ratio_weight + difference_estimate * (1.0 - ratio_weight)
        }
        DemosaicMethod::Lmmse => robust_difference(center_green, &candidates, 2.5),
        DemosaicMethod::Igv => robust_difference(center_green, &candidates, 1.75),
        DemosaicMethod::Amaze => robust_difference(center_green, &candidates, 3.0),
        DemosaicMethod::Simple => unreachable!(),
    }
    .clamp(0.0, 65535.0)
}

fn robust_difference(center_green: f32, candidates: &[(f32, f32, f32)], limit: f32) -> f32 {
    let mut differences = candidates
        .iter()
        .map(|&(value, green, _)| value - green)
        .collect::<Vec<_>>();
    differences.sort_by(f32::total_cmp);
    let median = differences[differences.len() / 2];
    let mut deviations = differences
        .iter()
        .map(|difference| (difference - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f32::total_cmp);
    let scale = deviations[deviations.len() / 2].max(1.0);
    let (sum, weight) = candidates.iter().fold((0.0, 0.0), |(sum, weight), sample| {
        let difference =
            (sample.0 - sample.1).clamp(median - limit * scale, median + limit * scale);
        (sum + difference * sample.2, weight + sample.2)
    });
    center_green + sum / weight.max(f32::EPSILON)
}

#[inline]
fn quantize(value: f32) -> u16 {
    value.round().clamp(0.0, 65535.0) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_mosaic(width: usize, height: usize, pattern: SensorPattern) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| match pattern.color_at(y, x) {
                    0 => 12000,
                    1 => 22000,
                    2 => 32000,
                    _ => unreachable!(),
                })
            })
            .collect()
    }

    #[test]
    fn every_method_preserves_constant_channels_and_measured_samples() {
        let (width, height, pattern) = (18, 16, SensorPattern::Rggb);
        let raw = constant_mosaic(width, height, pattern);
        for method in DemosaicMethod::ALL {
            let rgb = demosaic(&raw, width, height, pattern, method, 3).unwrap();
            for y in 0..height {
                for x in 0..width {
                    let index = y * width + x;
                    assert_eq!(rgb[index * 3], 12000, "{method} red at {x},{y}");
                    assert_eq!(rgb[index * 3 + 1], 22000, "{method} green at {x},{y}");
                    assert_eq!(rgb[index * 3 + 2], 32000, "{method} blue at {x},{y}");
                    let measured = pattern.color_at(y, x);
                    assert_eq!(rgb[index * 3 + measured], raw[index]);
                }
            }
        }
    }

    #[test]
    fn output_is_independent_of_thread_count() {
        let (width, height, pattern) = (24, 20, SensorPattern::Gbrg);
        let raw = (0..width * height)
            .map(|index| ((index * 977 + index / width * 311) & 0xffff) as u16)
            .collect::<Vec<_>>();
        for method in DemosaicMethod::ALL {
            assert_eq!(
                demosaic(&raw, width, height, pattern, method, 1).unwrap(),
                demosaic(&raw, width, height, pattern, method, 4).unwrap(),
                "{method}"
            );
        }
    }

    #[test]
    fn method_names_round_trip() {
        assert_eq!(DemosaicMethod::default(), DemosaicMethod::Amaze);
        for method in DemosaicMethod::ALL {
            assert_eq!(method.label().parse::<DemosaicMethod>().unwrap(), method);
        }
        assert_eq!("bilinear".parse(), Ok(DemosaicMethod::Simple));
        assert!("invented".parse::<DemosaicMethod>().is_err());
    }
}
