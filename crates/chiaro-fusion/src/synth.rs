//! Stage 3: synthesise one high-resolution frame from the aligned modules.
//!
//! The canvas is the reference module's raster scaled by an integer factor.
//! Every canvas pixel is mapped through each module's [`Warp`] and the module
//! is sampled (bilinear, per colour plane) where the mapping lands inside its
//! sensor. Contributions are blended with weights that
//!
//! - grow with the module's magnification (a B module has ~2.5x the linear
//!   resolution of an A module, so it carries 6x the weight where present);
//! - fall off smoothly towards the module's border (feathering), so seams
//!   between the narrow-field modules and the wide reference do not show;
//! - apply the alignment's photometric gain, so exposure differences between
//!   modules do not create brightness steps.
//!
//! Rendering is streamed in row bands straight into the PNG writer, so the
//! canvas is never held in memory.

use anyhow::{Result, bail};
use chiaro_hotpixel_core::png16::{
    PngColor, samples_to_be_bytes, write_png16_streaming_atomic_with_level,
};
use std::path::Path;

use crate::align::ModuleAlignment;
use crate::image::Mosaic;

/// Output colour handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OutputColor {
    /// Linear camera RGB after white balance, scaled so sensor white is 1.
    Linear,
    /// White balance, forward matrix to XYZ, XYZ to sRGB, exposure to the
    /// reference's 99.5th percentile and the sRGB transfer curve.
    #[default]
    Display,
}

/// Size of the output canvas.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum CanvasMode {
    /// The sensor's native 13 MP (4160 px across the framed view).
    #[default]
    Native,
    /// As many pixels as the finest module covering the view justifies, capped
    /// at this many megapixels.
    Maximum { max_megapixels: f32 },
    /// Explicit canvas pixels per reference pixel.
    Scale(f32),
}

#[derive(Clone, Debug)]
pub struct SynthOptions {
    pub canvas: CanvasMode,
    /// Feather width at module borders, in module pixels.
    pub feather_px: f32,
    pub color: OutputColor,
    /// Include monochrome modules (as luminance).
    pub include_mono: bool,
    /// Worker threads for rendering (`0` = all cores).
    pub threads: usize,
    /// PNG deflate level.
    pub png_level: u32,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            canvas: CanvasMode::Native,
            feather_px: 120.0,
            color: OutputColor::Display,
            include_mono: true,
            threads: 0,
            png_level: chiaro_hotpixel_core::png16::DEFAULT_DEFLATE_LEVEL,
        }
    }
}

/// The part of the reference raster that is rendered, in reference pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CropWindow {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CropWindow {
    pub fn full(width: usize, height: usize) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
        }
    }

    /// Centred crop covering `fraction` of each side.
    pub fn centred(width: usize, height: usize, fraction: f32) -> Self {
        let fraction = fraction.clamp(0.05, 1.0);
        let (w, h) = (width as f32 * fraction, height as f32 * fraction);
        Self {
            x: (width as f32 - w) / 2.0,
            y: (height as f32 - h) / 2.0,
            width: w,
            height: h,
        }
    }
}

/// Colour handling of one module: white balance in its own camera space and
/// its forward matrix to XYZ.
#[derive(Clone, Copy, Debug)]
pub struct ModuleColor {
    pub wb_gains: [f32; 3],
    /// Camera RGB -> XYZ (D65), or identity.
    pub forward: [[f32; 3]; 3],
}

impl Default for ModuleColor {
    fn default() -> Self {
        Self {
            wb_gains: [1.0; 3],
            forward: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

impl ModuleColor {
    #[inline]
    pub fn to_xyz(&self, rgb: [f32; 3]) -> [f32; 3] {
        let b = [
            rgb[0] * self.wb_gains[0],
            rgb[1] * self.wb_gains[1],
            rgb[2] * self.wb_gains[2],
        ];
        let f = &self.forward;
        [
            f[0][0] * b[0] + f[0][1] * b[1] + f[0][2] * b[2],
            f[1][0] * b[0] + f[1][1] * b[1] + f[1][2] * b[2],
            f[2][0] * b[0] + f[2][1] * b[1] + f[2][2] * b[2],
        ]
    }

    /// Luminance (XYZ Y) of a camera RGB sample.
    #[inline]
    pub fn luminance(&self, rgb: [f32; 3]) -> f32 {
        self.to_xyz(rgb)[1]
    }
}

/// Output colour constants.
#[derive(Clone, Debug)]
pub struct ColorPipeline {
    /// Exposure multiplier (applied before the transfer curve).
    pub exposure: f32,
}

impl Default for ColorPipeline {
    fn default() -> Self {
        Self { exposure: 1.0 }
    }
}

const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266, 1.876_010_8, 0.041_556],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];
/// XYZ of the D65 white (Y = 1), used for luminance-only content.
const D65_WHITE: [f32; 3] = [0.950_47, 1.0, 1.088_83];

fn srgb_transfer(linear: f32) -> f32 {
    let v = linear.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

impl ColorPipeline {
    fn apply(&self, xyz: [f32; 3], color: OutputColor) -> [f32; 3] {
        let m = &XYZ_TO_SRGB;
        let linear = [
            m[0][0] * xyz[0] + m[0][1] * xyz[1] + m[0][2] * xyz[2],
            m[1][0] * xyz[0] + m[1][1] * xyz[1] + m[1][2] * xyz[2],
            m[2][0] * xyz[0] + m[2][1] * xyz[1] + m[2][2] * xyz[2],
        ];
        match color {
            OutputColor::Linear => linear.map(|v| (v * self.exposure).clamp(0.0, 1.0)),
            OutputColor::Display => linear.map(|v| srgb_transfer(v * self.exposure)),
        }
    }
}

/// Low-frequency correction of one module towards the reference: a coarse
/// grid over the module's raster of per-XYZ-channel gains (reference over
/// module), bilinearly interpolated. It absorbs what a global match cannot:
/// mirror-path veiling glare, residual vignetting, and slow colour shading.
#[derive(Clone, Debug, PartialEq)]
pub struct GainField {
    pub columns: usize,
    pub rows: usize,
    pub gains: Vec<[f32; 3]>,
}

impl GainField {
    pub fn identity() -> Self {
        Self {
            columns: 1,
            rows: 1,
            gains: vec![[1.0; 3]],
        }
    }

    /// Gain at a module raster position of a `width x height` sensor.
    #[inline]
    pub fn at(&self, x: f32, y: f32, width: usize, height: usize) -> [f32; 3] {
        if self.columns == 1 && self.rows == 1 {
            return self.gains[0];
        }
        // Cell centres at ((c + 0.5) * width / columns, ...).
        let fx =
            (x / width as f32 * self.columns as f32 - 0.5).clamp(0.0, (self.columns - 1) as f32);
        let fy = (y / height as f32 * self.rows as f32 - 0.5).clamp(0.0, (self.rows - 1) as f32);
        let c0 = fx.floor() as usize;
        let r0 = fy.floor() as usize;
        let c1 = (c0 + 1).min(self.columns - 1);
        let r1 = (r0 + 1).min(self.rows - 1);
        let tx = fx - c0 as f32;
        let ty = fy - r0 as f32;
        let g = |c: usize, r: usize| self.gains[r * self.columns + c];
        let (a, b, c, d) = (g(c0, r0), g(c1, r0), g(c0, r1), g(c1, r1));
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            let top = a[k] * (1.0 - tx) + b[k] * tx;
            let bottom = c[k] * (1.0 - tx) + d[k] * tx;
            out[k] = top * (1.0 - ty) + bottom * ty;
        }
        out
    }
}

/// One module ready for synthesis.
pub struct SynthSource<'a> {
    pub mosaic: &'a Mosaic,
    pub alignment: &'a ModuleAlignment,
    /// Linear resolution relative to the reference (focal length ratio).
    pub magnification: f32,
    /// Alignment confidence, used to stop an uncertain high-resolution
    /// module from dominating a well-supported lower-resolution one.
    pub confidence: f32,
    pub color: ModuleColor,
    pub gain_field: GainField,
}

/// Per-module statistics of a synthesis run.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SynthReport {
    pub canvas_width: usize,
    pub canvas_height: usize,
    /// Canvas pixels per reference pixel.
    pub scale: f32,
    /// Reference-raster window that was rendered: x, y, width, height.
    pub crop: [f32; 4],
    /// Modules that contributed, in weight order.
    pub modules: Vec<String>,
    /// Fraction of canvas pixels with at least one contribution.
    pub covered: f32,
}

/// Canvas pixels per reference pixel for a crop and canvas mode.
/// `finest_magnification` is the largest module magnification covering the
/// crop (1 when only the reference does).
pub fn canvas_scale(
    crop: &CropWindow,
    reference_width: usize,
    mode: CanvasMode,
    finest_magnification: f32,
) -> f32 {
    match mode {
        CanvasMode::Native => reference_width as f32 / crop.width,
        CanvasMode::Maximum { max_megapixels } => {
            let wanted = finest_magnification.max(reference_width as f32 / crop.width);
            let pixels = crop.width * crop.height * wanted * wanted;
            let cap = (max_megapixels * 1.0e6).max(1.0);
            if pixels > cap {
                wanted * (cap / pixels).sqrt()
            } else {
                wanted
            }
        }
        CanvasMode::Scale(scale) => scale.max(0.05),
    }
}

/// Render the canvas and write it as a 16-bit RGB PNG.
pub fn synthesize(
    output: &Path,
    crop: CropWindow,
    scale: f32,
    sources: &[SynthSource<'_>],
    color: &ColorPipeline,
    options: &SynthOptions,
) -> Result<SynthReport> {
    let usable = sources
        .iter()
        .filter(|source| options.include_mono || !source.mosaic.is_mono())
        .collect::<Vec<_>>();
    if usable.is_empty() {
        bail!("no modules to synthesise from");
    }
    if scale.is_nan() || scale <= 0.0 || crop.width < 1.0 || crop.height < 1.0 {
        bail!("invalid canvas geometry");
    }
    let width = (crop.width * scale).round().max(1.0) as usize;
    let height = (crop.height * scale).round().max(1.0) as usize;
    let covered = std::sync::atomic::AtomicUsize::new(0);

    write_png16_streaming_atomic_with_level(
        output,
        width,
        height,
        PngColor::Rgb16,
        options.threads,
        options.png_level,
        |rows, bytes| {
            let mut band = vec![0u16; rows.len() * width * 3];
            let mut band_covered = 0usize;
            // Modules whose footprint can reach this band. Narrow modules
            // cover a small part of the canvas, so most bands skip them.
            let band_sources = usable
                .iter()
                .filter(|source| band_touches_module(source, rows.clone(), &crop, scale))
                .collect::<Vec<_>>();
            for (row_offset, v) in rows.clone().enumerate() {
                let ry = crop.y + (v as f32 + 0.5) / scale - 0.5;
                for u in 0..width {
                    let rx = crop.x + (u as f32 + 0.5) / scale - 0.5;
                    // Chroma comes from colour modules (XYZ), luminance from
                    // every module including panchromatic ones.
                    let mut xyz = [0.0f32; 3];
                    let mut color_weight = 0.0f32;
                    let mut luminance = 0.0f32;
                    let mut luminance_weight = 0.0f32;
                    for source in &band_sources {
                        let Some(q) = source.alignment.warp.map(rx, ry) else {
                            continue;
                        };
                        let mosaic = source.mosaic;
                        let Some(rgb) = mosaic.sample_rgb(q[0], q[1]) else {
                            continue;
                        };
                        let border = q[0]
                            .min(q[1])
                            .min((mosaic.width - 1) as f32 - q[0])
                            .min((mosaic.height - 1) as f32 - q[1]);
                        let feather = smoothstep(border / options.feather_px.max(1.0));
                        if feather <= 0.0 {
                            continue;
                        }
                        let weight = feather
                            * source.magnification
                            * source.magnification
                            * source.confidence;
                        let gain = source.alignment.gain;
                        let offset = source.alignment.offset;
                        let field = source
                            .gain_field
                            .at(q[0], q[1], mosaic.width, mosaic.height);
                        if mosaic.is_mono() {
                            let y = (gain * (rgb[1] - offset)).max(0.0) * field[1];
                            luminance += weight * y;
                            luminance_weight += weight;
                        } else {
                            let mut matched = source
                                .color
                                .to_xyz(rgb.map(|v| (gain * (v - offset)).max(0.0)));
                            for c in 0..3 {
                                matched[c] *= field[c];
                                xyz[c] += weight * matched[c];
                            }
                            color_weight += weight;
                            luminance += weight * matched[1];
                            luminance_weight += weight;
                        }
                    }
                    let pixel =
                        &mut band[(row_offset * width + u) * 3..(row_offset * width + u) * 3 + 3];
                    if luminance_weight > 0.0 {
                        band_covered += 1;
                        let target_luminance = luminance / luminance_weight;
                        let blended = if color_weight > 0.0 {
                            let mean = xyz.map(|v| v / color_weight);
                            if mean[1] > 1e-6 {
                                mean.map(|v| v * target_luminance / mean[1])
                            } else {
                                D65_WHITE.map(|v| v * target_luminance)
                            }
                        } else {
                            D65_WHITE.map(|v| v * target_luminance)
                        };
                        let rgb = color.apply(blended, options.color);
                        for c in 0..3 {
                            pixel[c] = (rgb[c].clamp(0.0, 1.0) * 65535.0).round() as u16;
                        }
                    }
                }
            }
            covered.fetch_add(band_covered, std::sync::atomic::Ordering::Relaxed);
            samples_to_be_bytes(&band, bytes);
        },
    )?;

    let mut modules = usable
        .iter()
        .map(|s| (s.alignment.name.clone(), s.magnification))
        .collect::<Vec<_>>();
    modules.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(SynthReport {
        canvas_width: width,
        canvas_height: height,
        scale,
        crop: [crop.x, crop.y, crop.width, crop.height],
        modules: modules.into_iter().map(|(name, _)| name).collect(),
        covered: covered.load(std::sync::atomic::Ordering::Relaxed) as f32
            / (width * height) as f32,
    })
}

/// Conservative test whether any canvas pixel of `rows` can map inside the
/// module: the warp is sampled densely along the band's first and last rows
/// and a few interior rows, with a margin of one warp cell.
fn band_touches_module(
    source: &SynthSource<'_>,
    rows: std::ops::Range<usize>,
    crop: &CropWindow,
    scale: f32,
) -> bool {
    let warp = &source.alignment.warp;
    let mosaic = source.mosaic;
    let margin = (warp.step * 2) as f32 * source.magnification.max(1.0);
    let inside = |x: f32, y: f32| {
        warp.map(x, y).is_some_and(|q| {
            q[0] >= -margin
                && q[1] >= -margin
                && q[0] <= mosaic.width as f32 + margin
                && q[1] <= mosaic.height as f32 + margin
        })
    };
    let row_samples = [
        rows.start,
        (rows.start + rows.end) / 2,
        rows.end.saturating_sub(1),
    ];
    for v in row_samples {
        let ry = crop.y + (v as f32 + 0.5) / scale - 0.5;
        let samples = ((crop.width / warp.step as f32) as usize).max(8) + 1;
        for i in 0..=samples {
            let rx = crop.x + (i as f32 / samples as f32) * (crop.width - 1.0);
            if inside(rx, ry) {
                return true;
            }
        }
    }
    false
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Exposure that maps the reference module's bright percentile to 0.92, as
/// the gallery previews do, from a subsample of the mosaic.
pub fn auto_exposure(reference: &Mosaic, color: &ModuleColor) -> f32 {
    let mut luminance = Vec::new();
    let step = 16;
    let mut y = 1;
    while y < reference.height - 2 {
        let mut x = 1;
        while x < reference.width - 2 {
            if let Some(rgb) = reference.sample_rgb(x as f32, y as f32) {
                luminance.push(color.luminance(rgb));
            }
            x += step;
        }
        y += step;
    }
    if luminance.is_empty() {
        return 1.0;
    }
    luminance.sort_by(f32::total_cmp);
    let percentile = luminance[((luminance.len() - 1) as f32 * 0.995) as usize].max(1e-4);
    0.92 / percentile
}

/// Luminance match `reference_Y ~= gain * (target_Y - offset)` where both
/// modules see the same scene. Colour is left to each module's own
/// calibration; only exposure, transmission, and flare differences are
/// equalised. Pairs are collected on a grid of mapped positions (ignoring
/// near-black and near-saturated values); the slope comes from the medians of
/// the darker and brighter halves, which is robust to misaligned or moving
/// content. Returns `(gain, offset)` in the module's linear sample units.
pub fn photometric_match(
    reference: &Mosaic,
    reference_color: &ModuleColor,
    target: &Mosaic,
    target_color: &ModuleColor,
    warp: &crate::align::Warp,
) -> (f32, f32) {
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    let step = 24usize;
    let mut y = step;
    while y + step < reference.height {
        let mut x = step;
        while x + step < reference.width {
            if let Some(q) = warp.map(x as f32, y as f32)
                && let Some(t) = target.sample_rgb(q[0], q[1])
                && let Some(r) = reference.sample_rgb(x as f32, y as f32)
            {
                let rv = reference_color.luminance(r);
                // A mono sample is already luminance in its own units.
                let tv = if target.is_mono() {
                    t[1]
                } else {
                    target_color.luminance(t)
                };
                if rv > 0.005 && tv > 0.005 && rv < 0.9 && tv < 0.9 {
                    pairs.push((tv, rv));
                }
            }
            x += step;
        }
        y += step;
    }
    if pairs.len() < 64 {
        return (1.0, 0.0);
    }
    let median = |values: &mut Vec<f32>| {
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    };
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let half = pairs.len() / 2;
    let (mut t_low, mut r_low): (Vec<f32>, Vec<f32>) = pairs[..half].iter().copied().unzip();
    let (mut t_high, mut r_high): (Vec<f32>, Vec<f32>) = pairs[half..].iter().copied().unzip();
    let (tl, rl) = (median(&mut t_low), median(&mut r_low));
    let (th, rh) = (median(&mut t_high), median(&mut r_high));
    if th - tl < 0.01 {
        return (((rl + rh) / (tl + th).max(1e-4)).clamp(0.2, 5.0), 0.0);
    }
    let slope = ((rh - rl) / (th - tl)).clamp(0.2, 5.0);
    let intercept = ((tl + th) / 2.0) - ((rl + rh) / 2.0) / slope;
    (slope, intercept.clamp(-0.2, 0.2))
}

/// Fit a [`GainField`] for `target` against `reference` after the global
/// luminance match `(gain, offset)`: per grid cell of the module's raster, the
/// median reference/target ratio of each XYZ channel over mapped samples (mono
/// modules: luminance only). Cells without overlap inherit from their
/// neighbours and the field is smoothed once, so it only carries slow trends.
#[allow(clippy::too_many_arguments)]
pub fn photometric_field(
    reference: &Mosaic,
    reference_color: &ModuleColor,
    target: &Mosaic,
    target_color: &ModuleColor,
    warp: &crate::align::Warp,
    gain: f32,
    offset: f32,
    columns: usize,
    rows: usize,
) -> GainField {
    let (columns, rows) = (columns.max(1), rows.max(1));
    let mut ratios: Vec<[Vec<f32>; 3]> = (0..columns * rows)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    let step = 12usize;
    let mut y = step;
    while y + step < reference.height {
        let mut x = step;
        while x + step < reference.width {
            if let Some(q) = warp.map(x as f32, y as f32)
                && let Some(t) = target.sample_rgb(q[0], q[1])
                && let Some(r) = reference.sample_rgb(x as f32, y as f32)
            {
                let column = ((q[0] / target.width as f32) * columns as f32)
                    .clamp(0.0, (columns - 1) as f32) as usize;
                let row = ((q[1] / target.height as f32) * rows as f32)
                    .clamp(0.0, (rows - 1) as f32) as usize;
                let cell = &mut ratios[row * columns + column];
                let r_xyz = reference_color.to_xyz(r);
                if target.is_mono() {
                    let tv = (gain * (t[1] - offset)).max(0.0);
                    if r_xyz[1] > 0.01 && tv > 0.01 && r_xyz[1] < 0.95 && tv < 0.95 {
                        cell[1].push(r_xyz[1] / tv);
                    }
                } else {
                    let t_xyz = target_color.to_xyz(t.map(|v| (gain * (v - offset)).max(0.0)));
                    for c in 0..3 {
                        if r_xyz[c] > 0.01 && t_xyz[c] > 0.01 && r_xyz[c] < 0.95 && t_xyz[c] < 0.95
                        {
                            cell[c].push(r_xyz[c] / t_xyz[c]);
                        }
                    }
                }
            }
            x += step;
        }
        y += step;
    }
    let mut gains: Vec<Option<[f32; 3]>> = ratios
        .iter_mut()
        .map(|cell| {
            let mut out = [1.0f32; 3];
            let mut any = false;
            for c in 0..3 {
                if cell[c].len() >= 12 {
                    cell[c].sort_by(f32::total_cmp);
                    out[c] = cell[c][cell[c].len() / 2].clamp(0.5, 2.0);
                    any = true;
                }
            }
            if target.is_mono() {
                out[0] = out[1];
                out[2] = out[1];
            }
            any.then_some(out)
        })
        .collect();
    if gains.iter().all(Option::is_none) {
        return GainField::identity();
    }
    // Fill empty cells from the mean of filled neighbours, repeatedly.
    while gains.iter().any(Option::is_none) {
        let snapshot = gains.clone();
        let mut progressed = false;
        for r in 0..rows {
            for c in 0..columns {
                if snapshot[r * columns + c].is_some() {
                    continue;
                }
                let mut sum = [0.0f32; 3];
                let mut n = 0.0;
                for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nr, nc) = (r as i32 + dr, c as i32 + dc);
                    if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= columns as i32 {
                        continue;
                    }
                    if let Some(g) = snapshot[nr as usize * columns + nc as usize] {
                        for k in 0..3 {
                            sum[k] += g[k];
                        }
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    gains[r * columns + c] = Some(sum.map(|v| v / n));
                    progressed = true;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    let filled = gains
        .into_iter()
        .map(|g| g.unwrap_or([1.0; 3]))
        .collect::<Vec<_>>();
    // One 3x3 smoothing pass keeps only slow trends.
    let mut smooth = filled.clone();
    for r in 0..rows {
        for c in 0..columns {
            let mut sum = [0.0f32; 3];
            let mut n = 0.0;
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    let (nr, nc) = (r as i32 + dr, c as i32 + dc);
                    if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= columns as i32 {
                        continue;
                    }
                    let g = filled[nr as usize * columns + nc as usize];
                    for k in 0..3 {
                        sum[k] += g[k];
                    }
                    n += 1.0;
                }
            }
            smooth[r * columns + c] = sum.map(|v| v / n);
        }
    }
    GainField {
        columns,
        rows,
        gains: smooth,
    }
}
