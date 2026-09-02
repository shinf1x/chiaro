//! Image containers and sampling used by alignment and synthesis.
//!
//! Two representations are used:
//!
//! - [`Mosaic`]: a module's corrected Q6 samples in *calibration raster*
//!   order with its CFA pattern. Synthesis samples RGB from it by bilinear
//!   interpolation inside each colour plane (a demosaic at sample time, so the
//!   full RGB frame is never materialised).
//! - [`Plane`]: a single-channel `f32` image (log luminance at reduced
//!   resolution) with a box pyramid, used for correlation-based alignment.

use chiaro::lri::SensorPattern;
use chiaro_hotpixel_core::demosaic::{DemosaicMethod, demosaic};
use chiaro_hotpixel_core::highlight::HighlightRecoveryState;

use crate::calibration::{CrosstalkMesh, VignettingMesh};

/// Q6 samples of one module in calibration raster order.
pub struct Mosaic {
    pub width: usize,
    pub height: usize,
    pub pattern: SensorPattern,
    /// Row-major, `width * height` samples, Q6 (RAW code `<< 6`).
    pub samples: Vec<u16>,
    /// Black level in Q6 units.
    pub black_q6: f32,
    /// White level in Q6 units.
    pub white_q6: f32,
    /// Flat-field mesh in calibration-raster orientation, if calibrated.
    pub vignetting: Option<VignettingMesh>,
    /// Colour-crosstalk mesh, if calibrated (colour modules only).
    pub crosstalk: Option<CrosstalkMesh>,
    /// Full-resolution interleaved RGB cache for advanced demosaicing. Simple
    /// mode leaves this empty and retains the original on-demand sampler.
    pub demosaiced_rgb: Option<Vec<u16>>,
}

impl Mosaic {
    /// Build from decoded RAW-stream order by rotating 180 degrees into the
    /// calibration raster. `pattern` is the camera's recorded CFA layout,
    /// which Light expresses in calibration-raster coordinates
    /// (`RawCamera::pattern`), so it applies to the rotated samples as is.
    pub fn from_stream_q6(
        mut samples: Vec<u16>,
        width: usize,
        height: usize,
        pattern: SensorPattern,
        black_level: f32,
        white_level: f32,
    ) -> Self {
        samples.reverse();
        Self {
            width,
            height,
            pattern,
            samples,
            black_q6: black_level * 64.0,
            white_q6: white_level * 64.0,
            vignetting: None,
            crosstalk: None,
            demosaiced_rgb: None,
        }
    }

    /// Trade one fractional RAW bit for one stop of reconstruction headroom.
    /// A 10-bit sensor still retains Q5 precision, while demosaic and
    /// crosstalk can carry estimates above the physical clipping point rather
    /// than saturating immediately at `u16::MAX`.
    pub fn reserve_highlight_headroom(&mut self) {
        for sample in &mut self.samples {
            *sample = (*sample + 1) / 2;
        }
        self.black_q6 *= 0.5;
        self.white_q6 *= 0.5;
    }

    /// Prepare the selected Bayer reconstruction for repeated warped samples.
    /// Advanced methods materialise one RGB16 image; Simple retains the
    /// low-memory on-demand bilinear path.
    pub fn prepare_demosaic(
        &mut self,
        method: DemosaicMethod,
        threads: usize,
    ) -> anyhow::Result<()> {
        self.demosaiced_rgb = if self.is_mono() || method == DemosaicMethod::Simple {
            None
        } else {
            // The factory mesh mixes the two green phases independently, so
            // apply it while the four CFA lattices are still distinct. Doing
            // this after RGB reconstruction would incorrectly collapse both
            // green inputs into one value.
            let corrected;
            let input = if let Some(crosstalk) = &self.crosstalk {
                let (red_row, red_col) = self.red_position();
                let mut values_out = Vec::with_capacity(self.samples.len());
                for y in 0..self.height {
                    for x in 0..self.width {
                        let values = [
                            self.bilinear_plane(x as f32, y as f32, red_col, red_row, 2),
                            self.bilinear_plane(x as f32, y as f32, 1 - red_col, red_row, 2),
                            self.bilinear_plane(x as f32, y as f32, red_col, 1 - red_row, 2),
                            self.bilinear_plane(x as f32, y as f32, 1 - red_col, 1 - red_row, 2),
                        ]
                        .map(|value| value - self.black_q6);
                        let phase = match (y & 1 == red_row, x & 1 == red_col) {
                            (true, true) => 0,
                            (true, false) => 1,
                            (false, true) => 2,
                            (false, false) => 3,
                        };
                        let matrix = crosstalk.matrix(x as f32, y as f32, self.width, self.height);
                        let value = (0..4)
                            .map(|column| matrix[phase * 4 + column] * values[column])
                            .sum::<f32>()
                            + self.black_q6;
                        values_out.push(value.round().clamp(0.0, 65535.0) as u16);
                    }
                }
                corrected = values_out;
                &corrected
            } else {
                &self.samples
            };
            Some(demosaic(
                input,
                self.width,
                self.height,
                self.pattern,
                method,
                threads,
            )?)
        };
        Ok(())
    }

    fn red_position(&self) -> (usize, usize) {
        (0..2usize)
            .flat_map(|row| (0..2usize).map(move |column| (row, column)))
            .find(|&(row, column)| self.pattern.color_at(row, column) == 0)
            .unwrap_or((0, 0))
    }

    /// Flat-field gain at a raster position (1 without calibration).
    #[inline]
    pub fn flat_field(&self, x: f32, y: f32) -> f32 {
        self.vignetting
            .as_ref()
            .map_or(1.0, |mesh| mesh.gain(x, y, self.width, self.height))
    }

    pub fn is_mono(&self) -> bool {
        self.pattern == SensorPattern::Mono
    }

    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        f32::from(self.samples[y * self.width + x])
    }

    /// Linear RGB and the corresponding per-channel sensor-white response at
    /// a fractional calibration-raster position. Both include the same
    /// flat-field and crosstalk corrections, which lets colour processing
    /// distinguish genuine colour from unequal channel clipping after white
    /// balance. Mono modules return equal RGB and white values.
    pub fn sample_rgb_with_white(&self, x: f32, y: f32) -> Option<([f32; 3], [f32; 3])> {
        if !(x >= 0.0 && y >= 0.0 && x <= (self.width - 1) as f32 && y <= (self.height - 1) as f32)
        {
            return None;
        }
        let range = (self.white_q6 - self.black_q6).max(1.0);
        let flat = self.flat_field(x, y);
        let normalise = |v: f32| ((v - self.black_q6) / range).max(0.0) * flat;
        if self.is_mono() {
            let v = normalise(self.bilinear_plane(x, y, 0, 0, 1));
            return Some(([v, v, v], [flat; 3]));
        }
        // Each colour lives on a stride-2 lattice: red, the green in the red
        // row, the green in the blue row, and blue.
        let (red_row, red_col) = self.red_position();
        let mut planes = if let Some(rgb) = &self.demosaiced_rgb {
            let [red, green, blue] = self.bilinear_rgb(rgb, x, y);
            [red, green, green, blue]
        } else {
            [
                self.bilinear_plane(x, y, red_col, red_row, 2),
                self.bilinear_plane(x, y, 1 - red_col, red_row, 2),
                self.bilinear_plane(x, y, red_col, 1 - red_row, 2),
                self.bilinear_plane(x, y, 1 - red_col, 1 - red_row, 2),
            ]
        }
        .map(|v| v - self.black_q6);
        let mut white_planes = [range; 4];
        if let Some(crosstalk) = &self.crosstalk {
            let m = crosstalk.matrix(x, y, self.width, self.height);
            let correct = |values: &mut [f32; 4]| {
                let input = *values;
                for (row, value) in values.iter_mut().enumerate() {
                    *value = m[row * 4] * input[0]
                        + m[row * 4 + 1] * input[1]
                        + m[row * 4 + 2] * input[2]
                        + m[row * 4 + 3] * input[3];
                }
            };
            if self.demosaiced_rgb.is_none() {
                correct(&mut planes);
            }
            correct(&mut white_planes);
        }
        let scale = flat / range;
        let to_rgb = |values: [f32; 4]| {
            [
                (values[0] * scale).max(0.0),
                ((values[1] + values[2]) * 0.5 * scale).max(0.0),
                (values[3] * scale).max(0.0),
            ]
        };
        Some((to_rgb(planes), to_rgb(white_planes)))
    }

    /// Linear RGB (black-subtracted and normalised) at a fractional
    /// calibration-raster position, or `None` outside the sensor.
    pub fn sample_rgb(&self, x: f32, y: f32) -> Option<[f32; 3]> {
        self.sample_rgb_with_white(x, y).map(|(rgb, _)| rgb)
    }

    /// Sample one normalised CFA colour plane without demosaicing. This is
    /// useful for temporal RAW stacking: alignment may be sub-pixel, while
    /// each output mosaic position must remain a measurement of one colour.
    pub fn sample_channel(&self, x: f32, y: f32, channel: usize) -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0 && x <= (self.width - 1) as f32 && y <= (self.height - 1) as f32)
        {
            return None;
        }
        let range = (self.white_q6 - self.black_q6).max(1.0);
        let value = if self.is_mono() {
            self.bilinear_plane(x, y, 0, 0, 1)
        } else {
            let (row, column) = (0..2usize)
                .flat_map(|row| (0..2usize).map(move |column| (row, column)))
                .find(|&(row, column)| self.pattern.color_at(row, column) == channel)
                .unwrap_or((0, 0));
            self.bilinear_plane(x, y, column, row, 2)
        };
        Some(((value - self.black_q6) / range).max(0.0) * self.flat_field(x, y))
    }

    /// Sample a black-subtracted CFA plane before flat-field/crosstalk, along
    /// with the minimum recovery confidence of the contributing lattice
    /// points. Cross-camera highlight recovery uses only confidence 255,
    /// ensuring donor radiance came from sensor measurements rather than from
    /// another spatial reconstruction.
    pub fn sample_raw_channel(
        &self,
        x: f32,
        y: f32,
        channel: usize,
        highlight: &HighlightRecoveryState,
    ) -> Option<(f32, u8)> {
        if !(x >= 0.0 && y >= 0.0 && x <= (self.width - 1) as f32 && y <= (self.height - 1) as f32)
        {
            return None;
        }
        let (row, column) = (0..2usize)
            .flat_map(|row| (0..2usize).map(move |column| (row, column)))
            .find(|&(row, column)| self.pattern.color_at(row, column) == channel)?;
        let value = self.bilinear_plane(x, y, column, row, 2);
        let confidence =
            self.bilinear_plane_confidence(x, y, column, row, 2, &highlight.confidence);
        let range = (self.white_q6 - self.black_q6).max(1.0);
        Some((((value - self.black_q6) / range).max(0.0), confidence))
    }

    /// Convert a normalized, black-subtracted measurement back to this
    /// module's Q6 sensor encoding.
    pub fn normalized_raw_to_q6(&self, value: f32) -> u16 {
        let range = (self.white_q6 - self.black_q6).max(1.0);
        (self.black_q6 + value.max(0.0) * range)
            .round()
            .clamp(0.0, 65535.0) as u16
    }

    /// Bilinear interpolation on the lattice `(ox + i*step, oy + j*step)`.
    fn bilinear_plane(&self, x: f32, y: f32, ox: usize, oy: usize, step: usize) -> f32 {
        let step_f = step as f32;
        let lx = (x - ox as f32) / step_f;
        let ly = (y - oy as f32) / step_f;
        let max_i = (self.width - 1 - ox) / step;
        let max_j = (self.height - 1 - oy) / step;
        let i0 = lx.floor().clamp(0.0, max_i as f32) as usize;
        let j0 = ly.floor().clamp(0.0, max_j as f32) as usize;
        let i1 = (i0 + 1).min(max_i);
        let j1 = (j0 + 1).min(max_j);
        let tx = (lx - i0 as f32).clamp(0.0, 1.0);
        let ty = (ly - j0 as f32).clamp(0.0, 1.0);
        let px = |i: usize, j: usize| self.at(ox + i * step, oy + j * step);
        let top = px(i0, j0) * (1.0 - tx) + px(i1, j0) * tx;
        let bottom = px(i0, j1) * (1.0 - tx) + px(i1, j1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    fn bilinear_plane_confidence(
        &self,
        x: f32,
        y: f32,
        ox: usize,
        oy: usize,
        step: usize,
        confidence: &[u8],
    ) -> u8 {
        if confidence.len() != self.samples.len() {
            return 255;
        }
        let step_f = step as f32;
        let lx = (x - ox as f32) / step_f;
        let ly = (y - oy as f32) / step_f;
        let max_i = (self.width - 1 - ox) / step;
        let max_j = (self.height - 1 - oy) / step;
        let i0 = lx.floor().clamp(0.0, max_i as f32) as usize;
        let j0 = ly.floor().clamp(0.0, max_j as f32) as usize;
        let i1 = (i0 + 1).min(max_i);
        let j1 = (j0 + 1).min(max_j);
        [(i0, j0), (i1, j0), (i0, j1), (i1, j1)]
            .into_iter()
            .map(|(i, j)| confidence[(oy + j * step) * self.width + ox + i * step])
            .min()
            .unwrap_or(0)
    }

    fn bilinear_rgb(&self, rgb: &[u16], x: f32, y: f32) -> [f32; 3] {
        let x0 = x.floor().clamp(0.0, (self.width - 1) as f32) as usize;
        let y0 = y.floor().clamp(0.0, (self.height - 1) as f32) as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = (x - x0 as f32).clamp(0.0, 1.0);
        let ty = (y - y0 as f32).clamp(0.0, 1.0);
        let pixel = |px: usize, py: usize, channel: usize| {
            f32::from(rgb[(py * self.width + px) * 3 + channel])
        };
        std::array::from_fn(|channel| {
            let top = pixel(x0, y0, channel) * (1.0 - tx) + pixel(x1, y0, channel) * tx;
            let bottom = pixel(x0, y1, channel) * (1.0 - tx) + pixel(x1, y1, channel) * tx;
            top * (1.0 - ty) + bottom * ty
        })
    }

    /// Half-resolution log-luminance plane for alignment: each 2x2 CFA cell
    /// becomes one value (its mean), so every pattern and mono yield the same
    /// geometry: plane pixel `(i, j)` is centred at raster `(2i + 0.5, 2j + 0.5)`.
    pub fn luminance_half(&self) -> Plane {
        let width = self.width / 2;
        let height = self.height / 2;
        let mut data = Vec::with_capacity(width * height);
        let range = (self.white_q6 - self.black_q6).max(1.0);
        for j in 0..height {
            let row0 = &self.samples[2 * j * self.width..(2 * j + 1) * self.width];
            let row1 = &self.samples[(2 * j + 1) * self.width..(2 * j + 2) * self.width];
            for i in 0..width {
                let sum = f32::from(row0[2 * i])
                    + f32::from(row0[2 * i + 1])
                    + f32::from(row1[2 * i])
                    + f32::from(row1[2 * i + 1]);
                let linear = ((sum * 0.25 - self.black_q6) / range).max(0.0)
                    * self.flat_field(2.0 * i as f32 + 0.5, 2.0 * j as f32 + 0.5);
                // Log compression equalises contrast between bright and dark
                // regions so correlation is driven by structure, not exposure.
                data.push((1.0 + 1000.0 * linear).ln());
            }
        }
        Plane {
            width,
            height,
            data,
        }
    }
}

/// Single-channel float image.
#[derive(Clone, Debug)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Bilinear sample; `None` outside the image.
    pub fn sample(&self, x: f32, y: f32) -> Option<f32> {
        if !(x >= 0.0 && y >= 0.0 && x <= (self.width - 1) as f32 && y <= (self.height - 1) as f32)
        {
            return None;
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let top = self.at(x0, y0) * (1.0 - tx) + self.at(x1, y0) * tx;
        let bottom = self.at(x0, y1) * (1.0 - tx) + self.at(x1, y1) * tx;
        Some(top * (1.0 - ty) + bottom * ty)
    }

    /// 2x box downsample (odd trailing rows/columns are dropped).
    pub fn downsample(&self) -> Plane {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut out = Plane::new(width, height);
        for j in 0..height {
            for i in 0..width {
                let (x, y) = (2 * i, 2 * j);
                let x1 = (x + 1).min(self.width - 1);
                let y1 = (y + 1).min(self.height - 1);
                out.data[j * width + i] =
                    0.25 * (self.at(x, y) + self.at(x1, y) + self.at(x, y1) + self.at(x1, y1));
            }
        }
        out
    }

    /// Pyramid `[self, half, quarter, ...]` down to `min_size` pixels on the
    /// short side.
    pub fn pyramid(&self, min_size: usize) -> Vec<Plane> {
        let mut levels = vec![self.clone()];
        while levels
            .last()
            .is_some_and(|p| p.width.min(p.height) / 2 >= min_size.max(8))
        {
            let next = levels.last().unwrap().downsample();
            levels.push(next);
        }
        levels
    }

    /// Standard deviation of a `size x size` window at `(x0, y0)`.
    pub fn window_std(&self, x0: usize, y0: usize, size: usize) -> f32 {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for y in y0..y0 + size {
            for x in x0..x0 + size {
                let v = f64::from(self.at(x, y));
                sum += v;
                sum_sq += v * v;
            }
        }
        let n = (size * size) as f64;
        let mean = sum / n;
        ((sum_sq / n - mean * mean).max(0.0)).sqrt() as f32
    }
}

/// Result of matching one patch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchMatch {
    /// Sub-pixel shift `(dx, dy)` of the target patch relative to the reference.
    pub shift: [f32; 2],
    /// Peak normalised cross-correlation in `[-1, 1]`.
    pub score: f32,
}

/// Normalised cross-correlation between a `size x size` reference window at
/// `(rx, ry)` and the same-sized window of `target` displaced by every integer
/// shift within `radius`. The peak is refined to sub-pixel precision with a
/// parabola through its neighbours.
///
/// Returns `None` when either window lacks contrast or leaves the images.
pub fn match_patch(
    reference: &Plane,
    target: &Plane,
    rx: usize,
    ry: usize,
    size: usize,
    radius: usize,
) -> Option<PatchMatch> {
    if rx + size > reference.width
        || ry + size > reference.height
        || rx < radius
        || ry < radius
        || rx + size + radius > target.width
        || ry + size + radius > target.height
        || rx + size > target.width
        || ry + size > target.height
    {
        return None;
    }
    let n = (size * size) as f32;
    // Reference window statistics.
    let mut ref_values = Vec::with_capacity(size * size);
    for y in ry..ry + size {
        ref_values.extend_from_slice(
            &reference.data[y * reference.width + rx..y * reference.width + rx + size],
        );
    }
    let ref_mean = ref_values.iter().sum::<f32>() / n;
    let ref_std = (ref_values
        .iter()
        .map(|v| (v - ref_mean).powi(2))
        .sum::<f32>()
        / n)
        .sqrt();
    if ref_std < 1e-4 {
        return None;
    }
    let ref_centered = ref_values.iter().map(|v| v - ref_mean).collect::<Vec<_>>();

    let span = 2 * radius + 1;
    let mut scores = vec![f32::NEG_INFINITY; span * span];
    for dy in 0..span {
        for dx in 0..span {
            let tx = rx + dx - radius;
            let ty = ry + dy - radius;
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            let mut cross = 0.0f32;
            for y in 0..size {
                let row =
                    &target.data[(ty + y) * target.width + tx..(ty + y) * target.width + tx + size];
                let reference_row = &ref_centered[y * size..(y + 1) * size];
                for (t, r) in row.iter().zip(reference_row) {
                    sum += t;
                    sum_sq += t * t;
                    cross += t * r;
                }
            }
            let mean = sum / n;
            let var = (sum_sq / n - mean * mean).max(0.0);
            if var < 1e-8 {
                continue;
            }
            // cross already excludes the reference mean; subtract the target mean term.
            let covariance = cross / n;
            scores[dy * span + dx] = covariance / (var.sqrt() * ref_std);
        }
    }
    let (best, &score) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if !score.is_finite() {
        return None;
    }
    let (bx, by) = (best % span, best / span);
    let refine = |minus: f32, center: f32, plus: f32| -> f32 {
        let denominator = minus - 2.0 * center + plus;
        if denominator.abs() < 1e-9 {
            0.0
        } else {
            (0.5 * (minus - plus) / denominator).clamp(-0.5, 0.5)
        }
    };
    let sub_x = if bx > 0 && bx + 1 < span {
        refine(
            scores[by * span + bx - 1],
            score,
            scores[by * span + bx + 1],
        )
    } else {
        0.0
    };
    let sub_y = if by > 0 && by + 1 < span {
        refine(
            scores[(by - 1) * span + bx],
            score,
            scores[(by + 1) * span + bx],
        )
    } else {
        0.0
    };
    Some(PatchMatch {
        shift: [
            bx as f32 - radius as f32 + sub_x,
            by as f32 - radius as f32 + sub_y,
        ],
        score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(width: usize, height: usize, shift: (f32, f32)) -> Plane {
        let mut plane = Plane::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let (fx, fy) = (x as f32 - shift.0, y as f32 - shift.1);
                plane.data[y * width + x] =
                    (fx * 0.31).sin() * (fy * 0.17).cos() + 0.3 * ((fx * 0.05 + fy * 0.07).sin());
            }
        }
        plane
    }

    #[test]
    fn patch_match_recovers_a_known_shift_with_subpixel_precision() {
        let reference = textured(200, 160, (0.0, 0.0));
        let target = textured(200, 160, (3.4, -2.6));
        let matched = match_patch(&reference, &target, 60, 50, 48, 8).unwrap();
        assert!((matched.shift[0] - 3.4).abs() < 0.25, "{matched:?}");
        assert!((matched.shift[1] + 2.6).abs() < 0.25, "{matched:?}");
        assert!(matched.score > 0.9);
        let flat = Plane::new(200, 160);
        assert!(match_patch(&flat, &target, 60, 50, 48, 8).is_none());
    }

    #[test]
    fn mosaic_sampling_demosaics_each_plane_and_rotates_the_stream() {
        // A 4x4 stream whose calibration-raster layout is RGGB: in stream
        // order the colours sit on the diagonally opposite cells (BGGR).
        let width = 4;
        let height = 4;
        let mut samples = vec![0u16; 16];
        for y in 0..height {
            for x in 0..width {
                samples[y * width + x] = match SensorPattern::Rggb.rotated_180().color_at(y, x) {
                    0 => 1000 << 6,
                    1 => 500 << 6,
                    _ => 200 << 6,
                };
            }
        }
        let mosaic =
            Mosaic::from_stream_q6(samples, width, height, SensorPattern::Rggb, 0.0, 1000.0);
        assert_eq!(mosaic.pattern, SensorPattern::Rggb);
        let rgb = mosaic.sample_rgb(1.5, 1.5).unwrap();
        assert!(
            (rgb[0] - 1.0).abs() < 1e-6
                && (rgb[1] - 0.5).abs() < 1e-6
                && (rgb[2] - 0.2).abs() < 1e-6
        );
        assert!(mosaic.sample_rgb(-0.1, 0.0).is_none());
        let luma = mosaic.luminance_half();
        assert_eq!((luma.width, luma.height), (2, 2));
        assert!(!luma.pyramid(1).is_empty());
    }
}
