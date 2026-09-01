//! Selectable Bayer demosaicing.
//!
//! The implementations in this module are clean-room Rust implementations
//! based on the published algorithm descriptions. They do not incorporate the
//! GPL implementations shipped by RawTherapee, darktable, or LibRaw.

use std::{fmt, str::FromStr};

use anyhow::{Result, bail};
use chiaro::lri::SensorPattern;
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
        estimate_green_rows(raw, &rgb, width, height, pattern, method, rows, band);
    });

    map_row_bands_mut(&mut rgb, width * 3, threads, 2, |rows, band| {
        reconstruct_rows(raw, &green, width, height, pattern, method, rows, band);
    });
    Ok(rgb)
}

#[allow(clippy::too_many_arguments)]
fn estimate_green_rows(
    raw: &[u16],
    rgb: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    method: DemosaicMethod,
    rows: std::ops::Range<usize>,
    green: &mut [f32],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if method == DemosaicMethod::Amaze && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked above; the vector loop observes the
        // same four-pixel border as the scalar estimator.
        return unsafe {
            estimate_green_rows_amaze_avx2(raw, rgb, width, height, pattern, rows, green)
        };
    }
    estimate_green_rows_scalar(raw, rgb, width, height, pattern, method, rows, green);
}

#[allow(clippy::too_many_arguments)]
fn estimate_green_rows_scalar(
    raw: &[u16],
    rgb: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    method: DemosaicMethod,
    rows: std::ops::Range<usize>,
    green: &mut [f32],
) {
    for (local_y, y) in rows.enumerate() {
        for x in 0..width {
            let index = y * width + x;
            green[local_y * width + x] = if pattern.color_at(y, x) == 1 {
                f32::from(raw[index])
            } else if x < 4 || y < 4 || x + 4 >= width || y + 4 >= height {
                f32::from(rgb[index * 3 + 1])
            } else {
                estimate_green(raw, width, x, y, method)
            };
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn estimate_green_rows_amaze_avx2(
    raw: &[u16],
    rgb: &[u16],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    rows: std::ops::Range<usize>,
    green: &mut [f32],
) {
    for (local_y, y) in rows.enumerate() {
        for x in 0..width {
            let index = y * width + x;
            green[local_y * width + x] = if pattern.color_at(y, x) == 1 {
                f32::from(raw[index])
            } else {
                f32::from(rgb[index * 3 + 1])
            };
        }
        if y < 4 || y + 4 >= height {
            continue;
        }
        let parity = usize::from(pattern.color_at(y, 0) == 1);
        let mut x = 4 + parity;
        while x + 18 < width {
            let values = unsafe { amaze_green_batch_avx2(raw, width, x, y) };
            for (lane, value) in values.into_iter().enumerate() {
                green[local_y * width + x + lane * 2] = value;
            }
            x += 16;
        }
        while x + 4 < width {
            green[local_y * width + x] = estimate_green(raw, width, x, y, DemosaicMethod::Amaze);
            x += 2;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_rows(
    raw: &[u16],
    green: &[f32],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    method: DemosaicMethod,
    rows: std::ops::Range<usize>,
    rgb: &mut [u16],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if method == DemosaicMethod::Amaze && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked above, and the row kernel only
        // accesses the same four-pixel interior used by the scalar path.
        return unsafe {
            reconstruct_rows_amaze_avx2(raw, green, width, height, pattern, rows, rgb)
        };
    }
    reconstruct_rows_scalar(raw, green, width, height, pattern, method, rows, rgb);
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_rows_scalar(
    raw: &[u16],
    green: &[f32],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    method: DemosaicMethod,
    rows: std::ops::Range<usize>,
    rgb: &mut [u16],
) {
    for (local_y, y) in rows.enumerate() {
        for x in 0..width {
            let index = y * width + x;
            let output = local_y * width * 3 + x * 3;
            let colour = pattern.color_at(y, x);
            let border = x < 4 || y < 4 || x + 4 >= width || y + 4 >= height;
            let red = if colour == 0 {
                f32::from(raw[index])
            } else if border {
                f32::from(rgb[output])
            } else {
                estimate_chroma(raw, green, width, height, pattern, x, y, 0, method)
            };
            let blue = if colour == 2 {
                f32::from(raw[index])
            } else if border {
                f32::from(rgb[output + 2])
            } else {
                estimate_chroma(raw, green, width, height, pattern, x, y, 2, method)
            };
            rgb[output] = quantize(red);
            rgb[output + 1] = quantize(green[index]);
            rgb[output + 2] = quantize(blue);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn reconstruct_rows_amaze_avx2(
    raw: &[u16],
    green: &[f32],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    rows: std::ops::Range<usize>,
    rgb: &mut [u16],
) {
    const DIAGONAL: [(isize, isize); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
    const HORIZONTAL: [(isize, isize); 6] = [(-1, -2), (1, -2), (-1, 0), (1, 0), (-1, 2), (1, 2)];
    const VERTICAL: [(isize, isize); 6] = [(-2, -1), (0, -1), (2, -1), (-2, 1), (0, 1), (2, 1)];

    for (local_y, y) in rows.enumerate() {
        for x in 0..width {
            rgb[(local_y * width + x) * 3 + 1] = quantize(green[y * width + x]);
        }
        if y < 4 || y + 4 >= height {
            continue;
        }
        for parity in 0..2 {
            let mut x = 4 + parity;
            let colour = pattern.color_at(y, x);
            while x + 18 < width {
                if colour == 1 {
                    let red_offsets = if pattern.color_at(y, x + 1) == 0 {
                        &HORIZONTAL
                    } else {
                        &VERTICAL
                    };
                    let blue_offsets = if pattern.color_at(y, x + 1) == 2 {
                        &HORIZONTAL
                    } else {
                        &VERTICAL
                    };
                    let red =
                        unsafe { amaze_chroma_batch_avx2(raw, green, width, x, y, red_offsets) };
                    let blue =
                        unsafe { amaze_chroma_batch_avx2(raw, green, width, x, y, blue_offsets) };
                    write_chroma_batch(rgb, width, local_y, x, 0, red);
                    write_chroma_batch(rgb, width, local_y, x, 2, blue);
                } else {
                    let channel = if colour == 0 { 2 } else { 0 };
                    let values =
                        unsafe { amaze_chroma_batch_avx2(raw, green, width, x, y, &DIAGONAL) };
                    write_chroma_batch(rgb, width, local_y, x, channel, values);
                }
                x += 16;
            }
            while x + 4 < width {
                if colour == 1 {
                    rgb[(local_y * width + x) * 3] =
                        quantize(estimate_chroma_amaze(raw, green, width, pattern, x, y, 0));
                    rgb[(local_y * width + x) * 3 + 2] =
                        quantize(estimate_chroma_amaze(raw, green, width, pattern, x, y, 2));
                } else {
                    let channel = if colour == 0 { 2 } else { 0 };
                    rgb[(local_y * width + x) * 3 + channel] = quantize(estimate_chroma_amaze(
                        raw, green, width, pattern, x, y, channel,
                    ));
                }
                x += 2;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn write_chroma_batch(
    rgb: &mut [u16],
    width: usize,
    local_y: usize,
    x: usize,
    channel: usize,
    values: [f32; 8],
) {
    for (lane, value) in values.into_iter().enumerate() {
        rgb[(local_y * width + x + lane * 2) * 3 + channel] = quantize(value);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn gather_raw_avx2(raw: &[u16], indices: __m256i, offset: i32) -> __m256 {
    let indices = _mm256_add_epi32(indices, _mm256_set1_epi32(offset));
    let packed = unsafe { _mm256_i32gather_epi32(raw.as_ptr().cast::<i32>(), indices, 2) };
    _mm256_cvtepi32_ps(_mm256_and_si256(packed, _mm256_set1_epi32(0xffff)))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn abs_avx2(values: __m256) -> __m256 {
    _mm256_andnot_ps(_mm256_set1_ps(-0.0), values)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn directional_candidates_avx2(
    raw: &[u16],
    width: usize,
    indices: __m256i,
) -> (__m256, __m256, __m256, __m256) {
    unsafe {
        let center = gather_raw_avx2(raw, indices, 0);
        let left = gather_raw_avx2(raw, indices, -1);
        let right = gather_raw_avx2(raw, indices, 1);
        let up = gather_raw_avx2(raw, indices, -(width as i32));
        let down = gather_raw_avx2(raw, indices, width as i32);
        let left2 = gather_raw_avx2(raw, indices, -2);
        let right2 = gather_raw_avx2(raw, indices, 2);
        let up2 = gather_raw_avx2(raw, indices, -2 * width as i32);
        let down2 = gather_raw_avx2(raw, indices, 2 * width as i32);
        let half = _mm256_set1_ps(0.5);
        let quarter = _mm256_set1_ps(0.25);
        let horizontal_correction =
            _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(center, center), left2), right2);
        let vertical_correction =
            _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(center, center), up2), down2);
        let gh = _mm256_add_ps(
            _mm256_mul_ps(half, _mm256_add_ps(left, right)),
            _mm256_mul_ps(quarter, horizontal_correction),
        );
        let gv = _mm256_add_ps(
            _mm256_mul_ps(half, _mm256_add_ps(up, down)),
            _mm256_mul_ps(quarter, vertical_correction),
        );
        let dh = _mm256_add_ps(
            _mm256_add_ps(
                abs_avx2(_mm256_sub_ps(left, right)),
                _mm256_mul_ps(half, abs_avx2(_mm256_sub_ps(left2, right2))),
            ),
            abs_avx2(horizontal_correction),
        );
        let dv = _mm256_add_ps(
            _mm256_add_ps(
                abs_avx2(_mm256_sub_ps(up, down)),
                _mm256_mul_ps(half, abs_avx2(_mm256_sub_ps(up2, down2))),
            ),
            abs_avx2(vertical_correction),
        );
        (gh, gv, dh, dv)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_gradient_avx2(raw: &[u16], indices: __m256i) -> __m256 {
    unsafe {
        let center = gather_raw_avx2(raw, indices, 0);
        let left = gather_raw_avx2(raw, indices, -1);
        let right = gather_raw_avx2(raw, indices, 1);
        let left2 = gather_raw_avx2(raw, indices, -2);
        let right2 = gather_raw_avx2(raw, indices, 2);
        let correction = _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(center, center), left2), right2);
        _mm256_add_ps(
            _mm256_add_ps(
                abs_avx2(_mm256_sub_ps(left, right)),
                _mm256_mul_ps(_mm256_set1_ps(0.5), abs_avx2(_mm256_sub_ps(left2, right2))),
            ),
            abs_avx2(correction),
        )
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn vertical_gradient_avx2(raw: &[u16], width: usize, indices: __m256i) -> __m256 {
    unsafe {
        let row = width as i32;
        let center = gather_raw_avx2(raw, indices, 0);
        let up = gather_raw_avx2(raw, indices, -row);
        let down = gather_raw_avx2(raw, indices, row);
        let up2 = gather_raw_avx2(raw, indices, -2 * row);
        let down2 = gather_raw_avx2(raw, indices, 2 * row);
        let correction = _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(center, center), up2), down2);
        _mm256_add_ps(
            _mm256_add_ps(
                abs_avx2(_mm256_sub_ps(up, down)),
                _mm256_mul_ps(_mm256_set1_ps(0.5), abs_avx2(_mm256_sub_ps(up2, down2))),
            ),
            abs_avx2(correction),
        )
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn amaze_green_batch_avx2(raw: &[u16], width: usize, x: usize, y: usize) -> [f32; 8] {
    unsafe {
        let base = (y * width + x) as i32;
        let indices = _mm256_setr_epi32(
            base,
            base + 2,
            base + 4,
            base + 6,
            base + 8,
            base + 10,
            base + 12,
            base + 14,
        );
        let (gh, gv, mut dh, mut dv) = directional_candidates_avx2(raw, width, indices);
        let quarter = _mm256_set1_ps(0.25);
        dh = _mm256_add_ps(
            dh,
            _mm256_mul_ps(
                quarter,
                horizontal_gradient_avx2(raw, _mm256_sub_epi32(indices, _mm256_set1_epi32(2))),
            ),
        );
        dv = _mm256_add_ps(
            dv,
            _mm256_mul_ps(
                quarter,
                vertical_gradient_avx2(
                    raw,
                    width,
                    _mm256_sub_epi32(indices, _mm256_set1_epi32(2 * width as i32)),
                ),
            ),
        );
        dh = _mm256_add_ps(
            dh,
            _mm256_mul_ps(
                quarter,
                horizontal_gradient_avx2(raw, _mm256_add_epi32(indices, _mm256_set1_epi32(2))),
            ),
        );
        dv = _mm256_add_ps(
            dv,
            _mm256_mul_ps(
                quarter,
                vertical_gradient_avx2(
                    raw,
                    width,
                    _mm256_add_epi32(indices, _mm256_set1_epi32(2 * width as i32)),
                ),
            ),
        );
        let one = _mm256_set1_ps(1.0);
        let weight_h = _mm256_div_ps(one, _mm256_add_ps(_mm256_mul_ps(dh, dh), one));
        let weight_v = _mm256_div_ps(one, _mm256_add_ps(_mm256_mul_ps(dv, dv), one));
        let value = _mm256_div_ps(
            _mm256_add_ps(_mm256_mul_ps(gh, weight_h), _mm256_mul_ps(gv, weight_v)),
            _mm256_add_ps(weight_h, weight_v),
        );
        let value = _mm256_min_ps(
            _mm256_max_ps(value, _mm256_setzero_ps()),
            _mm256_set1_ps(65535.0),
        );
        let mut values = [0.0f32; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), value);
        values
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn amaze_chroma_batch_avx2<const N: usize>(
    raw: &[u16],
    green: &[f32],
    width: usize,
    x: usize,
    y: usize,
    offsets: &[(isize, isize); N],
) -> [f32; 8] {
    unsafe {
        let base = (y * width + x) as i32;
        let lane_indices = _mm256_setr_epi32(
            base,
            base + 2,
            base + 4,
            base + 6,
            base + 8,
            base + 10,
            base + 12,
            base + 14,
        );
        let center_green = _mm256_i32gather_ps(green.as_ptr(), lane_indices, 4);
        let zero = _mm256_setzero_ps();
        let mut differences = [zero; N];
        let mut weights = [zero; N];
        let one = _mm256_set1_ps(1.0);
        let inv_edge_scale = _mm256_set1_ps(1.0 / 256.0);
        let sign = _mm256_set1_ps(-0.0);
        let raw_base = raw.as_ptr().cast::<i32>();

        for (slot, &(dx, dy)) in offsets.iter().enumerate() {
            let offset = (dy * width as isize + dx) as i32;
            let indices = _mm256_add_epi32(lane_indices, _mm256_set1_epi32(offset));
            let packed = _mm256_i32gather_epi32(raw_base, indices, 2);
            let samples = _mm256_cvtepi32_ps(_mm256_and_si256(packed, _mm256_set1_epi32(0xffff)));
            let neighbour_green = _mm256_i32gather_ps(green.as_ptr(), indices, 4);
            let difference = _mm256_sub_ps(samples, neighbour_green);
            let green_delta = _mm256_andnot_ps(sign, _mm256_sub_ps(neighbour_green, center_green));
            let edge = _mm256_div_ps(
                one,
                _mm256_add_ps(one, _mm256_mul_ps(green_delta, inv_edge_scale)),
            );
            let distance2 = (dx * dx + dy * dy) as f32;
            let spatial = _mm256_set1_ps(1.0 / (1.0 + distance2));
            differences[slot] = difference;
            weights[slot] = _mm256_mul_ps(spatial, edge);
        }

        let median = upper_median_avx2(differences);
        let mut deviations = [zero; N];
        for slot in 0..N {
            deviations[slot] = _mm256_andnot_ps(sign, _mm256_sub_ps(differences[slot], median));
        }
        let scale = _mm256_max_ps(upper_median_avx2(deviations), one);
        let spread = _mm256_mul_ps(_mm256_set1_ps(3.0), scale);
        let low = _mm256_sub_ps(median, spread);
        let high = _mm256_add_ps(median, spread);
        let mut sum = zero;
        let mut weight = zero;
        for slot in 0..N {
            let bounded = _mm256_min_ps(_mm256_max_ps(differences[slot], low), high);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(bounded, weights[slot]));
            weight = _mm256_add_ps(weight, weights[slot]);
        }
        let result = _mm256_add_ps(
            center_green,
            _mm256_div_ps(sum, _mm256_max_ps(weight, _mm256_set1_ps(f32::EPSILON))),
        );
        let mut values = [0.0f32; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), result);
        values
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn upper_median_avx2<const N: usize>(mut values: [__m256; N]) -> __m256 {
    macro_rules! compare_swap {
        ($left:expr, $right:expr) => {{
            let a = values[$left];
            let b = values[$right];
            values[$left] = _mm256_min_ps(a, b);
            values[$right] = _mm256_max_ps(a, b);
        }};
    }

    if N == 4 {
        compare_swap!(0, 1);
        compare_swap!(2, 3);
        compare_swap!(0, 2);
        compare_swap!(1, 3);
        compare_swap!(1, 2);
        values[2]
    } else {
        debug_assert_eq!(N, 6);
        compare_swap!(1, 2);
        compare_swap!(4, 5);
        compare_swap!(0, 2);
        compare_swap!(3, 5);
        compare_swap!(0, 1);
        compare_swap!(3, 4);
        compare_swap!(2, 5);
        compare_swap!(0, 3);
        compare_swap!(1, 4);
        compare_swap!(2, 4);
        compare_swap!(1, 3);
        compare_swap!(2, 3);
        values[3]
    }
}

#[inline]
fn raw_at(raw: &[u16], width: usize, x: isize, y: isize) -> f32 {
    f32::from(raw[y as usize * width + x as usize])
}

#[inline(always)]
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

#[inline(always)]
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
#[inline(always)]
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
    if method == DemosaicMethod::Amaze {
        return estimate_chroma_amaze(raw, green, width, pattern, x, y, channel)
            .clamp(0.0, 65535.0);
    }
    estimate_chroma_generic(raw, green, width, height, pattern, x, y, channel, method)
}

/// AMaZE only considers target-colour sites in a 5x5 Bayer neighbourhood.
/// Depending on the centre phase that is exactly four diagonal samples or six
/// horizontal/vertical samples. Enumerating those sites directly avoids a
/// heap allocation and 25 colour lookups for every missing output sample.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn estimate_chroma_amaze(
    raw: &[u16],
    green: &[f32],
    width: usize,
    pattern: SensorPattern,
    x: usize,
    y: usize,
    channel: usize,
) -> f32 {
    const DIAGONAL: [(isize, isize); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
    const HORIZONTAL: [(isize, isize); 6] = [(-1, -2), (1, -2), (-1, 0), (1, 0), (-1, 2), (1, 2)];
    const VERTICAL: [(isize, isize); 6] = [(-2, -1), (0, -1), (2, -1), (-2, 1), (0, 1), (2, 1)];

    let center = y * width + x;
    let center_green = green[center];
    let center_colour = pattern.color_at(y, x);
    let offsets: &[(isize, isize)] = if center_colour != 1 {
        &DIAGONAL
    } else if pattern.color_at(y, x + 1) == channel {
        &HORIZONTAL
    } else {
        &VERTICAL
    };
    let mut differences = [0.0f32; 6];
    let mut weights = [0.0f32; 6];
    for (slot, &(dx, dy)) in offsets.iter().enumerate() {
        let nx = (x as isize + dx) as usize;
        let ny = (y as isize + dy) as usize;
        let index = ny * width + nx;
        let neighbour_green = green[index];
        let distance2 = (dx * dx + dy * dy) as f32;
        let spatial = 1.0 / (1.0 + distance2);
        let edge = 1.0 / (1.0 + (neighbour_green - center_green).abs() / 256.0);
        differences[slot] = f32::from(raw[index]) - neighbour_green;
        weights[slot] = spatial * edge;
    }

    robust_difference_from_arrays(center_green, &differences, &weights, offsets.len(), 3.0)
}

#[inline(always)]
fn robust_difference_from_arrays(
    center_green: f32,
    differences: &[f32; 6],
    weights: &[f32; 6],
    len: usize,
    limit: f32,
) -> f32 {
    let median = upper_median_4_or_6(*differences, len);
    let mut deviations = [0.0f32; 6];
    for (deviation, &difference) in deviations[..len].iter_mut().zip(&differences[..len]) {
        *deviation = (difference - median).abs();
    }
    let scale = upper_median_4_or_6(deviations, len).max(1.0);
    let low = median - limit * scale;
    let high = median + limit * scale;
    let mut sum = 0.0;
    let mut weight = 0.0;
    for (&difference, &sample_weight) in differences[..len].iter().zip(&weights[..len]) {
        sum += difference.clamp(low, high) * sample_weight;
        weight += sample_weight;
    }
    center_green + sum / weight.max(f32::EPSILON)
}

#[inline(always)]
fn upper_median_4_or_6(mut values: [f32; 6], len: usize) -> f32 {
    #[inline(always)]
    fn compare_swap(values: &mut [f32; 6], left: usize, right: usize) {
        let a = values[left];
        let b = values[right];
        values[left] = a.min(b);
        values[right] = a.max(b);
    }

    match len {
        4 => {
            compare_swap(&mut values, 0, 1);
            compare_swap(&mut values, 2, 3);
            compare_swap(&mut values, 0, 2);
            compare_swap(&mut values, 1, 3);
            compare_swap(&mut values, 1, 2);
            values[2]
        }
        6 => {
            compare_swap(&mut values, 1, 2);
            compare_swap(&mut values, 4, 5);
            compare_swap(&mut values, 0, 2);
            compare_swap(&mut values, 3, 5);
            compare_swap(&mut values, 0, 1);
            compare_swap(&mut values, 3, 4);
            compare_swap(&mut values, 2, 5);
            compare_swap(&mut values, 0, 3);
            compare_swap(&mut values, 1, 4);
            compare_swap(&mut values, 2, 4);
            compare_swap(&mut values, 1, 3);
            compare_swap(&mut values, 2, 3);
            values[3]
        }
        _ => unreachable!("AMaZE uses four or six chroma neighbours"),
    }
}

#[allow(clippy::too_many_arguments)]
fn estimate_chroma_generic(
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

    fn generic_amaze_reference(
        raw: &[u16],
        width: usize,
        height: usize,
        pattern: SensorPattern,
    ) -> Vec<u16> {
        let mut rgb = demosaic_bilinear_threaded(raw, width, height, pattern, 1).unwrap();
        let green = (0..raw.len())
            .map(|index| {
                let (x, y) = (index % width, index / width);
                if pattern.color_at(y, x) == 1 {
                    f32::from(raw[index])
                } else if x < 4 || y < 4 || x + 4 >= width || y + 4 >= height {
                    f32::from(rgb[index * 3 + 1])
                } else {
                    estimate_green(raw, width, x, y, DemosaicMethod::Amaze)
                }
            })
            .collect::<Vec<_>>();
        let reconstruct = |channel| {
            (0..raw.len())
                .map(|index| {
                    let (x, y) = (index % width, index / width);
                    if pattern.color_at(y, x) == channel {
                        f32::from(raw[index])
                    } else if x < 4 || y < 4 || x + 4 >= width || y + 4 >= height {
                        f32::from(rgb[index * 3 + channel])
                    } else {
                        estimate_chroma_generic(
                            raw,
                            &green,
                            width,
                            height,
                            pattern,
                            x,
                            y,
                            channel,
                            DemosaicMethod::Amaze,
                        )
                    }
                })
                .collect::<Vec<_>>()
        };
        let red = reconstruct(0);
        let blue = reconstruct(2);
        for index in 0..raw.len() {
            rgb[index * 3] = quantize(red[index]);
            rgb[index * 3 + 1] = quantize(green[index]);
            rgb[index * 3 + 2] = quantize(blue[index]);
        }
        rgb
    }

    fn scalar_amaze(raw: &[u16], width: usize, height: usize, pattern: SensorPattern) -> Vec<u16> {
        let mut rgb = demosaic_bilinear_threaded(raw, width, height, pattern, 1).unwrap();
        let mut green = vec![0.0; raw.len()];
        estimate_green_rows_scalar(
            raw,
            &rgb,
            width,
            height,
            pattern,
            DemosaicMethod::Amaze,
            0..height,
            &mut green,
        );
        reconstruct_rows_scalar(
            raw,
            &green,
            width,
            height,
            pattern,
            DemosaicMethod::Amaze,
            0..height,
            &mut rgb,
        );
        rgb
    }

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
    fn amaze_chroma_kernel_matches_generic_neighbour_search() {
        let (width, height) = (24, 20);
        let raw = (0..width * height)
            .map(|index| ((index * 977 + index / width * 311) & 0xffff) as u16)
            .collect::<Vec<_>>();
        for pattern in [
            SensorPattern::Rggb,
            SensorPattern::Grbg,
            SensorPattern::Gbrg,
            SensorPattern::Bggr,
        ] {
            let bilinear = demosaic_bilinear_threaded(&raw, width, height, pattern, 1).unwrap();
            let green = (0..raw.len())
                .map(|index| {
                    let (x, y) = (index % width, index / width);
                    if pattern.color_at(y, x) == 1 {
                        f32::from(raw[index])
                    } else if x < 4 || y < 4 || x + 4 >= width || y + 4 >= height {
                        f32::from(bilinear[index * 3 + 1])
                    } else {
                        estimate_green(&raw, width, x, y, DemosaicMethod::Amaze)
                    }
                })
                .collect::<Vec<_>>();
            for y in 4..height - 4 {
                for x in 4..width - 4 {
                    for channel in [0, 2] {
                        if pattern.color_at(y, x) == channel {
                            continue;
                        }
                        let optimized =
                            estimate_chroma_amaze(&raw, &green, width, pattern, x, y, channel)
                                .clamp(0.0, 65535.0);
                        let generic = estimate_chroma_generic(
                            &raw,
                            &green,
                            width,
                            height,
                            pattern,
                            x,
                            y,
                            channel,
                            DemosaicMethod::Amaze,
                        );
                        assert_eq!(
                            optimized.to_bits(),
                            generic.to_bits(),
                            "{pattern:?} channel={channel} at {x},{y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn optimized_amaze_matches_generic_full_frame_reference() {
        let (width, height) = (24, 20);
        let raw = (0..width * height)
            .map(|index| ((index * 977 + index / width * 311) & 0xffff) as u16)
            .collect::<Vec<_>>();
        for pattern in [
            SensorPattern::Rggb,
            SensorPattern::Grbg,
            SensorPattern::Gbrg,
            SensorPattern::Bggr,
        ] {
            let dispatched =
                demosaic(&raw, width, height, pattern, DemosaicMethod::Amaze, 3).unwrap();
            let scalar = scalar_amaze(&raw, width, height, pattern);
            assert_eq!(
                dispatched, scalar,
                "runtime dispatch versus scalar fallback for {pattern:?}"
            );
            assert_eq!(
                scalar,
                generic_amaze_reference(&raw, width, height, pattern),
                "optimized versus generic reference for {pattern:?}"
            );
        }
    }

    #[test]
    fn fixed_amaze_median_networks_select_the_upper_median() {
        fn next_permutation(values: &mut [f32]) -> bool {
            let Some(pivot) = (0..values.len() - 1)
                .rev()
                .find(|&index| values[index].total_cmp(&values[index + 1]).is_lt())
            else {
                return false;
            };
            let successor = (pivot + 1..values.len())
                .rev()
                .find(|&index| values[pivot].total_cmp(&values[index]).is_lt())
                .unwrap();
            values.swap(pivot, successor);
            values[pivot + 1..].reverse();
            true
        }

        let mut values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        loop {
            assert_eq!(upper_median_4_or_6(values, 6), 3.0);
            if !next_permutation(&mut values) {
                break;
            }
        }
        let mut values = [0.0, 1.0, 2.0, 3.0];
        loop {
            let input = [values[0], values[1], values[2], values[3], 99.0, -99.0];
            assert_eq!(upper_median_4_or_6(input, 4), 2.0);
            if !next_permutation(&mut values) {
                break;
            }
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
