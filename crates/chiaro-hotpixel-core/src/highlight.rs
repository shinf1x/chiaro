//! Highlight reconstruction on the Bayer mosaic, before crosstalk correction
//! and demosaicing.
//!
//! The local pass repairs small, partially clipped structures with
//! edge-directed interpolation.  The multiscale pass builds a clipping-aware
//! pyramid of half-resolution RGB observations and uses coarser surviving
//! colour ratios for samples which the local pass could not reconstruct
//! confidently.

use std::{fmt, str::FromStr};

use anyhow::{Result, bail};
use chiaro::lri::SensorPattern;
use serde::{Deserialize, Serialize};

/// RAW-domain highlight recovery policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HighlightRecovery {
    /// Preserve the corrected mosaic exactly.
    None,
    /// Repair only small, partially clipped Bayer structures.
    LocalBayer,
    /// Add clipping-aware coarse-to-fine reconstruction for larger regions.
    MultiscaleBayer,
    /// Use both spatial passes and permit a later geometry-gated donor pass.
    #[default]
    MultiCamera,
}

impl HighlightRecovery {
    pub const ALL: [Self; 4] = [
        Self::None,
        Self::LocalBayer,
        Self::MultiscaleBayer,
        Self::MultiCamera,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::LocalBayer => "Local Bayer",
            Self::MultiscaleBayer => "Multiscale Bayer",
            Self::MultiCamera => "Multi-camera",
        }
    }

    pub fn uses_local(self) -> bool {
        self != Self::None
    }

    pub fn uses_multiscale(self) -> bool {
        matches!(self, Self::MultiscaleBayer | Self::MultiCamera)
    }

    pub fn uses_multi_camera(self) -> bool {
        self == Self::MultiCamera
    }
}

impl fmt::Display for HighlightRecovery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "none",
            Self::LocalBayer => "local-bayer",
            Self::MultiscaleBayer => "multiscale-bayer",
            Self::MultiCamera => "multi-camera",
        })
    }
}

impl FromStr for HighlightRecovery {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "local" | "local-bayer" => Ok(Self::LocalBayer),
            "multiscale" | "multiscale-bayer" => Ok(Self::MultiscaleBayer),
            "multi-camera" | "multicamera" => Ok(Self::MultiCamera),
            _ => Err(format!(
                "unknown RAW highlight recovery {value:?}; expected none, local-bayer, \
                 multiscale-bayer, or multi-camera"
            )),
        }
    }
}

/// Per-module accounting for RAW highlight reconstruction.
#[derive(Clone, Debug, Default, Serialize)]
pub struct HighlightRecoveryReport {
    pub mode: HighlightRecovery,
    pub clipped_samples: usize,
    /// Original hard-clipped measurements grouped as R, G, and B.
    pub clipped_by_channel: [usize; 3],
    /// Valid measurements in the soft transition immediately below clipping.
    pub feathered_samples: usize,
    pub local_recovered: usize,
    pub multiscale_recovered: usize,
    pub multi_camera_recovered: usize,
    pub neutralized_samples: usize,
    pub unresolved_samples: usize,
    pub mean_recovery_confidence: f32,
}

/// Confidence retained for the optional cross-camera pass.  `255` means the
/// sample was measured safely below clipping, `1..=254` means near-clipped or
/// spatially recovered, and zero means no trustworthy colour estimate was
/// found.
#[derive(Clone, Debug)]
pub struct HighlightRecoveryState {
    pub confidence: Vec<u8>,
    pub report: HighlightRecoveryReport,
}

impl HighlightRecoveryState {
    /// Whether a spatially recovered sample is uncertain enough to benefit
    /// from an independently measured camera value.
    pub fn needs_donor(&self, index: usize) -> bool {
        self.confidence
            .get(index)
            .is_some_and(|&confidence| confidence < 208)
    }

    /// Record a geometry-verified donor replacement.
    pub fn mark_multi_camera(&mut self, index: usize, confidence: u8) {
        if let Some(value) = self.confidence.get_mut(index) {
            *value = (*value).max(confidence.min(254));
            self.report.multi_camera_recovered += 1;
        }
    }

    /// Recalculate aggregate fields after a batch of donor replacements.
    pub fn finish_multi_camera(&mut self) {
        finish_report(&self.confidence, &mut self.report);
    }
}

/// Recover clipped samples in a Q6 Bayer mosaic in place.
pub fn recover_bayer_highlights(
    raw: &mut [u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    black_q6: f32,
    white_q6: f32,
    mode: HighlightRecovery,
) -> Result<HighlightRecoveryState> {
    if width == 0 || height == 0 || raw.len() != width.saturating_mul(height) {
        bail!("RAW dimensions do not match sample count");
    }
    let mut report = HighlightRecoveryReport {
        mode,
        ..Default::default()
    };
    if pattern == SensorPattern::Mono || mode == HighlightRecovery::None {
        return Ok(HighlightRecoveryState {
            confidence: Vec::new(),
            report,
        });
    }

    // Leave a small guard below sensor white: clipped photos often contain a
    // few codes of roll-off/non-linearity before the literal 10-bit maximum.
    let range = (white_q6 - black_q6).max(64.0);
    let clip = black_q6 + range * 0.995;
    let feather_start = black_q6 + range * 0.970;
    let original = raw.to_vec();
    let mut confidence = original
        .iter()
        .map(|&value| {
            let value = f32::from(value);
            if value >= clip {
                0
            } else if mode.uses_multiscale() && value >= feather_start {
                (255.0 * (clip - value) / (clip - feather_start))
                    .round()
                    .clamp(1.0, 254.0) as u8
            } else {
                255
            }
        })
        .collect::<Vec<_>>();
    report.clipped_samples = confidence.iter().filter(|&&value| value == 0).count();
    report.feathered_samples = confidence
        .iter()
        .filter(|&&value| value > 0 && value < 255)
        .count();
    for y in 0..height {
        for x in 0..width {
            if confidence[y * width + x] == 0 {
                report.clipped_by_channel[pattern.color_at(y, x)] += 1;
            }
        }
    }
    if report.clipped_samples == 0 && report.feathered_samples == 0 {
        return Ok(HighlightRecoveryState { confidence, report });
    }

    if mode.uses_local() {
        for y in 2..height.saturating_sub(2) {
            for x in 2..width.saturating_sub(2) {
                let index = y * width + x;
                if confidence[index] != 0 {
                    continue;
                }
                if let Some((value, quality)) =
                    local_reconstruction(&original, width, height, pattern, x, y, black_q6, clip)
                {
                    raw[index] = value.max(clip).round().clamp(0.0, 65535.0) as u16;
                    confidence[index] = quality;
                    report.local_recovered += 1;
                }
            }
        }
    }

    if mode.uses_multiscale() {
        multiscale_reconstruction(
            raw,
            &original,
            &mut confidence,
            width,
            height,
            pattern,
            black_q6,
            clip,
            &mut report,
        );
    }

    finish_report(&confidence, &mut report);
    Ok(HighlightRecoveryState { confidence, report })
}

fn finish_report(confidence: &[u8], report: &mut HighlightRecoveryReport) {
    report.unresolved_samples = confidence.iter().filter(|&&value| value == 0).count();
    let recovered = confidence
        .iter()
        .filter(|&&value| value > 0 && value < 255)
        .copied()
        .collect::<Vec<_>>();
    report.mean_recovery_confidence = if recovered.is_empty() {
        0.0
    } else {
        recovered.iter().map(|&value| f32::from(value)).sum::<f32>()
            / (recovered.len() as f32 * 255.0)
    };
}

#[allow(clippy::too_many_arguments)]
fn local_reconstruction(
    raw: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    x: usize,
    y: usize,
    black: f32,
    clip: f32,
) -> Option<(f32, u8)> {
    let channel = pattern.color_at(y, x);
    if channel == 1 {
        return directional_same_phase(raw, width, height, x, y, black, clip);
    }

    // At a clipped red/blue site, reconstruct green first from the two
    // cardinal directions, choosing the smoother edge direction.
    let mut green_candidates = Vec::with_capacity(2);
    for ((ax, ay), (bx, by)) in [((-1, 0), (1, 0)), ((0, -1), (0, 1))] {
        let a = sample_offset(raw, width, height, x, y, ax, ay)?;
        let b = sample_offset(raw, width, height, x, y, bx, by)?;
        if a < clip && b < clip {
            green_candidates.push(((a + b) * 0.5, (a - b).abs()));
        }
    }
    let &(green, _) = green_candidates.iter().min_by(|a, b| a.1.total_cmp(&b.1))?;

    // Interpolate C-G, not absolute C. This preserves a coloured edge when
    // the local brightness changes more quickly than chroma.
    let mut weighted_difference = 0.0;
    let mut total_weight = 0.0;
    let mut support = 0usize;
    for (dx, dy) in [
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-2, -2),
        (2, -2),
        (-2, 2),
        (2, 2),
    ] {
        let Some(colour) = sample_offset(raw, width, height, x, y, dx, dy) else {
            continue;
        };
        if colour >= clip {
            continue;
        }
        let nx = (x as isize + dx) as usize;
        let ny = (y as isize + dy) as usize;
        let Some((neighbour_green, _)) = green_at_colour_site(raw, width, height, nx, ny, clip)
        else {
            continue;
        };
        let distance = ((dx * dx + dy * dy) as f32).sqrt();
        let edge = (neighbour_green - green).abs();
        let weight = 1.0 / (distance * (1.0 + edge / 256.0));
        weighted_difference += weight * ((colour - black) - (neighbour_green - black));
        total_weight += weight;
        support += 1;
    }
    if support < 2 || total_weight <= 0.0 {
        return None;
    }
    let value = green + weighted_difference / total_weight;
    let quality = (128 + support.min(5) * 20 + green_candidates.len() * 8).min(232) as u8;
    Some((value.max(black), quality))
}

fn green_at_colour_site(
    raw: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    clip: f32,
) -> Option<(f32, f32)> {
    let mut candidates = Vec::with_capacity(2);
    for ((ax, ay), (bx, by)) in [((-1, 0), (1, 0)), ((0, -1), (0, 1))] {
        let a = sample_offset(raw, width, height, x, y, ax, ay)?;
        let b = sample_offset(raw, width, height, x, y, bx, by)?;
        if a < clip && b < clip {
            candidates.push(((a + b) * 0.5, (a - b).abs()));
        }
    }
    candidates.into_iter().min_by(|a, b| a.1.total_cmp(&b.1))
}

fn directional_same_phase(
    raw: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    _black: f32,
    clip: f32,
) -> Option<(f32, u8)> {
    let mut weighted = 0.0;
    let mut total = 0.0;
    let mut support = 0usize;
    for ((ax, ay), (bx, by)) in [
        ((-2, 0), (2, 0)),
        ((0, -2), (0, 2)),
        ((-2, -2), (2, 2)),
        ((2, -2), (-2, 2)),
    ] {
        let Some(a) = sample_offset(raw, width, height, x, y, ax, ay) else {
            continue;
        };
        let Some(b) = sample_offset(raw, width, height, x, y, bx, by) else {
            continue;
        };
        if a >= clip || b >= clip {
            continue;
        }
        let gradient = (a - b).abs();
        let weight = 1.0 / (1.0 + gradient / 128.0).powi(2);
        weighted += weight * (a + b) * 0.5;
        total += weight;
        support += 1;
    }
    if support == 0 || total <= 0.0 {
        None
    } else {
        Some((weighted / total, (148 + support.min(4) * 20) as u8))
    }
}

fn sample_offset(
    raw: &[u16],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
) -> Option<f32> {
    let nx = x.checked_add_signed(dx)?;
    let ny = y.checked_add_signed(dy)?;
    (nx < width && ny < height).then(|| f32::from(raw[ny * width + nx]))
}

#[derive(Clone)]
struct PyramidLevel {
    width: usize,
    height: usize,
    values: Vec<[f32; 3]>,
    /// Fraction of the Gaussian support containing complete, unclipped RGB
    /// cells. A Bayer 2x2 cell becomes one half-resolution RGB observation.
    weights: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn multiscale_reconstruction(
    raw: &mut [u16],
    original: &[u16],
    confidence: &mut [u8],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    black: f32,
    clip: f32,
    report: &mut HighlightRecoveryReport,
) {
    let cell_width = width.div_ceil(2);
    let cell_height = height.div_ceil(2);
    let mut base = PyramidLevel {
        width: cell_width,
        height: cell_height,
        values: vec![[0.0; 3]; cell_width * cell_height],
        weights: vec![0.0; cell_width * cell_height],
    };
    let mut measured = vec![[false; 3]; cell_width * cell_height];
    for cell_y in 0..cell_height {
        for cell_x in 0..cell_width {
            let cell = cell_y * cell_width + cell_x;
            let mut sum = [0.0f32; 3];
            let mut count = [0usize; 3];
            for dy in 0..2 {
                for dx in 0..2 {
                    let (x, y) = (cell_x * 2 + dx, cell_y * 2 + dy);
                    if x >= width || y >= height {
                        continue;
                    }
                    let channel = pattern.color_at(y, x);
                    let index = y * width + x;
                    // If only one green is clipped, the other green is the
                    // best same-cell estimate and should remain an anchor.
                    if confidence[index] == 255 {
                        sum[channel] += (f32::from(original[index]) - black).max(0.0);
                        count[channel] += 1;
                    }
                }
            }
            for channel in 0..3 {
                if count[channel] > 0 {
                    base.values[cell][channel] = sum[channel] / count[channel] as f32;
                    measured[cell][channel] = true;
                } else {
                    // Retain the clipped lower bound for brightness matching;
                    // this value is never included in Gaussian reduction.
                    base.values[cell][channel] = clip - black;
                }
            }
            base.weights[cell] = f32::from(measured[cell].iter().all(|value| *value));
        }
    }

    let mut pyramid = vec![base];
    while pyramid.len() < 9 {
        let current = pyramid.last().expect("base pyramid level");
        if current.width <= 2 || current.height <= 2 {
            break;
        }
        pyramid.push(reduce_level(current));
    }

    // Fill holes in the complete-RGB pyramid from coarse to fine. Reduction
    // never averages a clipped cell, so colour ratios cannot be polluted by
    // the very magenta/green failure this stage is intended to remove.
    for level_index in (0..pyramid.len().saturating_sub(1)).rev() {
        let coarse = pyramid[level_index + 1].clone();
        let level = &mut pyramid[level_index];
        for y in 0..level.height {
            for x in 0..level.width {
                let index = y * level.width + x;
                if level.weights[index] <= 0.0 {
                    level.values[index] = sample_level(&coarse, x as f32 * 0.5, y as f32 * 0.5);
                    level.weights[index] =
                        sample_level_weight(&coarse, x as f32 * 0.5, y as f32 * 0.5) * 0.8;
                }
            }
        }
    }

    let base = &pyramid[0];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let cell = (y / 2) * cell_width + x / 2;
            let old_quality = confidence[index];
            if old_quality == 255 {
                continue;
            }
            let channel = pattern.color_at(y, x);
            let surviving = measured[cell].iter().filter(|value| **value).count();
            let support = base.weights[cell].clamp(0.0, 1.0);
            let lower_bound = f32::from(original[index]).min(clip);
            let mut ratios = Vec::with_capacity(3);
            for candidate in 0..3 {
                if measured[cell][candidate] && base.values[cell][candidate] > 1.0 {
                    let observed = measured_cell_value(
                        original,
                        width,
                        height,
                        pattern,
                        x / 2,
                        y / 2,
                        candidate,
                        black,
                        clip,
                    );
                    if let Some(observed) = observed {
                        ratios.push((observed / base.values[cell][candidate]).clamp(0.25, 4.0));
                    }
                }
            }
            ratios.sort_by(f32::total_cmp);
            // A partial cell has a real radiometric anchor. A fully clipped
            // cell supplies only a lower bound, so scale the smooth coarse
            // colour field just enough for every channel to remain clipped.
            let scale = ratios.get(ratios.len() / 2).copied().unwrap_or_else(|| {
                base.values[cell]
                    .iter()
                    .filter(|value| **value > 1.0)
                    .map(|value| (clip - black) / value)
                    .fold(1.0f32, f32::max)
            });
            let coarse = if base.values[cell][channel] > 1.0 {
                (black + base.values[cell][channel] * scale).max(lower_bound)
            } else {
                lower_bound
            };

            // `support` is propagated through the clipping-aware pyramid and
            // decays at every missing scale. Squaring it gives a continuous
            // feather: nearby colour evidence survives, while deep saturated
            // interiors approach a neutral sensor-white lower bound without a
            // binary cell boundary.
            let colour_weight = support * support;
            let field_value = lower_bound + (coarse - lower_bound) * colour_weight;

            // Local edge reconstruction remains useful around thin detail.
            // Blend it continuously with the field rather than selecting one
            // algorithm at a confidence threshold.
            let local_weight = (f32::from(old_quality) / 255.0).powi(2);
            let value = (field_value * (1.0 - local_weight) + f32::from(raw[index]) * local_weight)
                .max(lower_bound);
            raw[index] = value.round().clamp(0.0, 65535.0) as u16;
            let quality = (32.0 + support * 144.0 + surviving as f32 * 12.0)
                .round()
                .clamp(32.0, 204.0) as u8;
            confidence[index] = old_quality.max(quality);
            report.multiscale_recovered += usize::from(old_quality == 0);
            report.neutralized_samples += usize::from(old_quality == 0 && colour_weight < 0.05);
        }
    }

    debug_assert!(pattern != SensorPattern::Mono);
}

#[allow(clippy::too_many_arguments)]
fn measured_cell_value(
    original: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    cell_x: usize,
    cell_y: usize,
    channel: usize,
    black: f32,
    clip: f32,
) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0;
    for dy in 0..2 {
        for dx in 0..2 {
            let (x, y) = (cell_x * 2 + dx, cell_y * 2 + dy);
            if x >= width || y >= height || pattern.color_at(y, x) != channel {
                continue;
            }
            let value = f32::from(original[y * width + x]);
            if value < clip {
                sum += (value - black).max(0.0);
                count += 1;
            }
        }
    }
    (count > 0).then_some(sum / count as f32)
}

fn reduce_level(input: &PyramidLevel) -> PyramidLevel {
    let width = input.width.div_ceil(2);
    let height = input.height.div_ceil(2);
    let mut output = PyramidLevel {
        width,
        height,
        values: vec![[0.0; 3]; width * height],
        weights: vec![0.0; width * height],
    };
    const KERNEL: [f32; 3] = [1.0, 2.0, 1.0];
    for y in 0..height {
        for x in 0..width {
            let output_index = y * width + x;
            let mut kernel_sum = 0.0;
            let mut weight_sum = 0.0;
            for (ky, &wy) in KERNEL.iter().enumerate() {
                for (kx, &wx) in KERNEL.iter().enumerate() {
                    let sx = (x * 2 + kx).saturating_sub(1).min(input.width - 1);
                    let sy = (y * 2 + ky).saturating_sub(1).min(input.height - 1);
                    let index = sy * input.width + sx;
                    let kernel = wx * wy;
                    weight_sum += input.weights[index] * kernel;
                    kernel_sum += kernel;
                }
            }
            if weight_sum > 0.0 {
                for channel in 0..3 {
                    let mut sum = 0.0;
                    for (ky, &wy) in KERNEL.iter().enumerate() {
                        for (kx, &wx) in KERNEL.iter().enumerate() {
                            let sx = (x * 2 + kx).saturating_sub(1).min(input.width - 1);
                            let sy = (y * 2 + ky).saturating_sub(1).min(input.height - 1);
                            let index = sy * input.width + sx;
                            let kernel = wx * wy;
                            let weight = input.weights[index] * kernel;
                            sum += input.values[index][channel] * weight;
                        }
                    }
                    output.values[output_index][channel] = sum / weight_sum;
                }
                output.weights[output_index] = (weight_sum / kernel_sum).clamp(0.0, 1.0);
            }
        }
    }
    output
}

fn sample_level(level: &PyramidLevel, x: f32, y: f32) -> [f32; 3] {
    std::array::from_fn(|channel| sample_level_channel(level, x, y, channel, false))
}

fn sample_level_weight(level: &PyramidLevel, x: f32, y: f32) -> f32 {
    sample_level_channel(level, x, y, 0, true)
}

fn sample_level_channel(
    level: &PyramidLevel,
    x: f32,
    y: f32,
    channel: usize,
    weights: bool,
) -> f32 {
    let x0 = x.floor().clamp(0.0, (level.width - 1) as f32) as usize;
    let y0 = y.floor().clamp(0.0, (level.height - 1) as f32) as usize;
    let x1 = (x0 + 1).min(level.width - 1);
    let y1 = (y0 + 1).min(level.height - 1);
    let tx = (x - x0 as f32).clamp(0.0, 1.0);
    let ty = (y - y0 as f32).clamp(0.0, 1.0);
    let value = |px: usize, py: usize| {
        let index = py * level.width + px;
        if weights {
            level.weights[index]
        } else {
            level.values[index][channel]
        }
    };
    let top = value(x0, y0) * (1.0 - tx) + value(x1, y0) * tx;
    let bottom = value(x0, y1) * (1.0 - tx) + value(x1, y1) * tx;
    top * (1.0 - ty) + bottom * ty
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_with_constant_phases(width: usize, height: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| match (y & 1, x & 1) {
                    (0, 0) => 48_000,
                    (1, 1) => 28_000,
                    _ => 36_000,
                })
            })
            .collect()
    }

    #[test]
    fn local_reconstruction_never_darkens_a_clipped_measurement() {
        let (width, height) = (16, 16);
        let mut raw = raw_with_constant_phases(width, height);
        raw[8 * width + 8] = 65_535;
        let state = recover_bayer_highlights(
            &mut raw,
            width,
            height,
            SensorPattern::Rggb,
            0.0,
            65_535.0,
            HighlightRecovery::LocalBayer,
        )
        .unwrap();
        assert!(raw[8 * width + 8] >= 65_000);
        assert_eq!(state.report.local_recovered, 1);
        assert!(state.confidence[8 * width + 8] > 128);
    }

    #[test]
    fn multiscale_reaches_the_middle_of_a_large_clipped_region() {
        let (width, height) = (64, 64);
        let mut raw = raw_with_constant_phases(width, height);
        for y in 20..44 {
            for x in 20..44 {
                raw[y * width + x] = 65_535;
            }
        }
        let state = recover_bayer_highlights(
            &mut raw,
            width,
            height,
            SensorPattern::Rggb,
            0.0,
            65_535.0,
            HighlightRecovery::MultiscaleBayer,
        )
        .unwrap();
        let centre = 32 * width + 32;
        assert!(state.confidence[centre] > 0);
        assert!(state.report.multiscale_recovered > 0);
        assert!(raw[centre] >= 65_000);
    }

    #[test]
    fn multiscale_feathers_measurements_below_hard_clipping() {
        let (width, height) = (64, 64);
        let mut raw = raw_with_constant_phases(width, height);
        let index = 32 * width + 32;
        raw[index] = 64_200;
        let state = recover_bayer_highlights(
            &mut raw,
            width,
            height,
            SensorPattern::Rggb,
            0.0,
            65_535.0,
            HighlightRecovery::MultiscaleBayer,
        )
        .unwrap();
        assert_eq!(state.report.clipped_samples, 0);
        assert_eq!(state.report.feathered_samples, 1);
        assert!((1..255).contains(&state.confidence[index]));
        assert!(raw[index] >= 64_200);
    }

    #[test]
    fn none_is_bit_exact() {
        let mut raw = raw_with_constant_phases(8, 8);
        raw[27] = 65_535;
        let expected = raw.clone();
        let state = recover_bayer_highlights(
            &mut raw,
            8,
            8,
            SensorPattern::Rggb,
            0.0,
            65_535.0,
            HighlightRecovery::None,
        )
        .unwrap();
        assert_eq!(raw, expected);
        assert!(state.confidence.is_empty());
    }
}
