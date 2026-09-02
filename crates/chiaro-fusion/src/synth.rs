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
//! - retain fine structure from the reference unless another module reproduces
//!   the same edge direction at least as sharply, preventing a stack of mildly
//!   defocused or sub-pixel-misaligned samples from softening the result;
//! - apply the alignment's photometric gain, so exposure differences between
//!   modules do not create brightness steps.
//!
//! Rendering is streamed in row bands straight into the PNG writer, so the
//! canvas is never held in memory.

use anyhow::{Result, bail};
use chiaro_hotpixel_core::demosaic::DemosaicMethod;
use chiaro_hotpixel_core::highlight::HighlightRecovery;
use chiaro_hotpixel_core::png16::{
    PngColor, samples_to_be_bytes, write_png16_streaming_atomic_with_level,
    write_rgb16_native_atomic,
};
use std::path::Path;

use crate::align::ModuleAlignment;
use crate::depth::DenseDepthMap;
use crate::image::Mosaic;

/// Output colour handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OutputColor {
    /// Linear sRGB after white balance and D50-to-D65 adaptation.
    Linear,
    /// White balance, forward matrix to D50 XYZ, chromatic adaptation to D65,
    /// XYZ to sRGB, exposure to the reference's 99.5th percentile and the
    /// sRGB transfer curve.
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
    /// Bayer reconstruction used before repeated warped sampling.
    pub demosaic: DemosaicMethod,
    /// Clipped-sample reconstruction applied to the RAW Bayer mosaic before
    /// crosstalk correction and demosaicing.
    pub highlight_recovery: HighlightRecovery,
    /// Include monochrome modules (as luminance).
    pub include_mono: bool,
    /// Smoothly reconstruct false colour caused by unequal raw-channel clipping
    /// after white balance. Disable when preserving the unmodified channel
    /// response for a downstream raw processor.
    pub highlight_correction: bool,
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
            demosaic: DemosaicMethod::default(),
            highlight_recovery: HighlightRecovery::default(),
            include_mono: true,
            highlight_correction: true,
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
/// its DNG forward matrix to the D50 XYZ profile connection space.
#[derive(Clone, Copy, Debug)]
pub struct ModuleColor {
    pub wb_gains: [f32; 3],
    /// Camera RGB -> XYZ (D50), or identity when calibration is unavailable.
    pub forward: [[f32; 3]; 3],
    /// Whether `forward` is a real camera colour calibration. Uncalibrated
    /// Bayer modules may contribute luminance, but never unreliable chroma.
    pub calibrated: bool,
}

impl Default for ModuleColor {
    fn default() -> Self {
        Self {
            wb_gains: [1.0; 3],
            forward: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            calibrated: false,
        }
    }
}

impl ModuleColor {
    #[inline]
    fn balanced_to_xyz(&self, balanced: [f32; 3]) -> [f32; 3] {
        let f = &self.forward;
        [
            f[0][0] * balanced[0] + f[0][1] * balanced[1] + f[0][2] * balanced[2],
            f[1][0] * balanced[0] + f[1][1] * balanced[1] + f[1][2] * balanced[2],
            f[2][0] * balanced[0] + f[2][1] * balanced[1] + f[2][2] * balanced[2],
        ]
    }

    #[inline]
    pub fn to_xyz(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.balanced_to_xyz([
            rgb[0] * self.wb_gains[0],
            rgb[1] * self.wb_gains[1],
            rgb[2] * self.wb_gains[2],
        ])
    }

    /// Convert camera RGB to XYZ with a smooth highlight shoulder. Raw channels
    /// saturate before white balance at different scene intensities; without
    /// reconstruction, a neutral highlight becomes magenta when green reaches
    /// sensor white first. Progressively blending camera RGB towards its median
    /// white-balanced level removes that false chroma without the sharp boundary
    /// and flat low ceiling produced by independently clamping channels.
    #[inline]
    pub fn to_xyz_clipped(&self, rgb: [f32; 3], sensor_white: [f32; 3]) -> [f32; 3] {
        let mut balanced = [
            rgb[0] * self.wb_gains[0],
            rgb[1] * self.wb_gains[1],
            rgb[2] * self.wb_gains[2],
        ];
        let proximity = rgb
            .into_iter()
            .zip(sensor_white)
            .filter_map(|(value, white)| {
                (value.is_finite() && white.is_finite() && white > 0.0).then_some(value / white)
            })
            .fold(0.0f32, f32::max);
        let reconstruction = smoothstep((proximity - 0.94) / 0.06);
        if reconstruction > 0.0 {
            let mut levels = balanced;
            levels.sort_by(f32::total_cmp);
            let neutral = levels[1];
            balanced = balanced.map(|value| value + (neutral - value) * reconstruction);
        }
        self.balanced_to_xyz(balanced)
    }

    /// Luminance (XYZ Y) of a camera RGB sample.
    #[inline]
    pub fn luminance(&self, rgb: [f32; 3]) -> f32 {
        self.to_xyz(rgb)[1]
    }

    #[inline]
    pub fn luminance_clipped(&self, rgb: [f32; 3], sensor_white: [f32; 3]) -> f32 {
        self.to_xyz_clipped(rgb, sensor_white)[1]
    }

    #[inline]
    fn xyz_for_output(
        &self,
        rgb: [f32; 3],
        sensor_white: [f32; 3],
        highlight_correction: bool,
    ) -> [f32; 3] {
        if highlight_correction {
            self.to_xyz_clipped(rgb, sensor_white)
        } else {
            self.to_xyz(rgb)
        }
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
/// ICC/DNG profile connection-space white (Y = 1).
const D50_WHITE: [f32; 3] = [0.964_22, 1.0, 0.825_21];

/// Bradford chromatic adaptation from D50 XYZ to the D65 white used by sRGB.
const D50_TO_D65: [[f32; 3]; 3] = [
    [0.955_576_6, -0.023_039_3, 0.063_163_6],
    [-0.028_289_5, 1.009_941_6, 0.021_007_7],
    [0.012_298_2, -0.020_483, 1.329_909_8],
];

fn adapt_d50_to_d65(xyz: [f32; 3]) -> [f32; 3] {
    let m = &D50_TO_D65;
    [
        m[0][0] * xyz[0] + m[0][1] * xyz[1] + m[0][2] * xyz[2],
        m[1][0] * xyz[0] + m[1][1] * xyz[1] + m[1][2] * xyz[2],
        m[2][0] * xyz[0] + m[2][1] * xyz[1] + m[2][2] * xyz[2],
    ]
}

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
        let xyz = adapt_d50_to_d65(xyz);
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
    /// The reference module anchors edge ownership during robust blending.
    pub reference: bool,
    /// Linear resolution relative to the reference (focal length ratio).
    pub magnification: f32,
    /// Alignment confidence, used to stop an uncertain high-resolution
    /// module from dominating a well-supported lower-resolution one.
    pub confidence: f32,
    /// Object-space focus distance inferred from the captured lens position.
    /// Used only as a local prior when the depth map independently places an
    /// object substantially closer than a magnified source's focus plane.
    pub focus_distance: Option<f64>,
    pub color: ModuleColor,
    pub gain_field: GainField,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SourceContributionReport {
    pub camera: String,
    /// RGB legend colour used by the optional ownership diagnostics.
    pub diagnostic_rgb: [u8; 3],
    /// Fraction of covered output pixels where this source supplied the
    /// largest pre-normalisation luminance weight.
    pub luminance_owner_fraction: f32,
    /// Equivalent ownership fraction for colour.
    pub color_owner_fraction: f32,
    /// Fraction of sampled pixels where known scene depth strongly disagreed
    /// with this magnified source's focus plane.
    pub focus_suppressed_fraction: f32,
    /// Fraction of colour samples suppressed by reference chromaticity.
    pub chroma_suppressed_fraction: f32,
}

#[derive(Default)]
struct SourceCounters {
    sampled: std::sync::atomic::AtomicUsize,
    luminance_owner: std::sync::atomic::AtomicUsize,
    color_owner: std::sync::atomic::AtomicUsize,
    focus_suppressed: std::sync::atomic::AtomicUsize,
    color_sampled: std::sync::atomic::AtomicUsize,
    chroma_suppressed: std::sync::atomic::AtomicUsize,
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
    /// Whether unequal clipped-channel colour received smooth reconstruction.
    pub highlight_correction: bool,
    pub raw_highlight_recovery: HighlightRecovery,
    pub demosaic: DemosaicMethod,
    /// Fraction of non-reference samples rejected as strong photometric or
    /// local-detail contradictions. Agreeing modules remain fully blended.
    pub edge_rejected_fraction: f32,
    pub source_contributions: Vec<SourceContributionReport>,
    /// Reference/output pixel stride of the ownership diagnostics.
    pub ownership_diagnostic_step: usize,
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
    depth_map: Option<&DenseDepthMap>,
    diagnostic_dir: Option<&Path>,
    color: &ColorPipeline,
    options: &SynthOptions,
) -> Result<SynthReport> {
    let usable = sources
        .iter()
        .filter(|source| options.include_mono || !source.mosaic.is_mono())
        .enumerate()
        .collect::<Vec<_>>();
    if usable.is_empty() {
        bail!("no modules to synthesise from");
    }
    if scale.is_nan() || scale <= 0.0 || crop.width < 1.0 || crop.height < 1.0 {
        bail!("invalid canvas geometry");
    }
    let width = (crop.width * scale).round().max(1.0) as usize;
    let height = (crop.height * scale).round().max(1.0) as usize;
    const OWNERSHIP_STEP: usize = 8;
    let ownership_columns = width.div_ceil(OWNERSHIP_STEP);
    let ownership_rows = height.div_ceil(OWNERSHIP_STEP);
    let ownership_len = if diagnostic_dir.is_some() {
        ownership_columns * ownership_rows
    } else {
        0
    };
    let luminance_ownership = (0..ownership_len)
        .map(|_| std::sync::atomic::AtomicUsize::new(usize::MAX))
        .collect::<Vec<_>>();
    let color_ownership = (0..ownership_len)
        .map(|_| std::sync::atomic::AtomicUsize::new(usize::MAX))
        .collect::<Vec<_>>();
    let covered = std::sync::atomic::AtomicUsize::new(0);
    let edge_checked = std::sync::atomic::AtomicUsize::new(0);
    let edge_rejected = std::sync::atomic::AtomicUsize::new(0);
    let source_counters = (0..usable.len())
        .map(|_| SourceCounters::default())
        .collect::<Vec<_>>();

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
            let mut band_edge_checked = 0usize;
            let mut band_edge_rejected = 0usize;
            // Modules whose footprint can reach this band. Narrow modules
            // cover a small part of the canvas, so most bands skip them.
            let band_sources = usable
                .iter()
                .filter(|(_, source)| band_touches_module(source, rows.clone(), &crop, scale))
                .collect::<Vec<_>>();
            for (row_offset, v) in rows.clone().enumerate() {
                let ry = crop.y + (v as f32 + 0.5) / scale - 0.5;
                for u in 0..width {
                    let rx = crop.x + (u as f32 + 0.5) / scale - 0.5;
                    let reference_luminance = band_sources
                        .iter()
                        .find(|(_, source)| source.reference)
                        .and_then(|(_, source)| source_luminance(source, rx, ry, options));
                    let reference_structure = band_sources
                        .iter()
                        .find(|(_, source)| source.reference)
                        .and_then(|(_, source)| {
                            source_log_luminance_structure(source, rx, ry, options)
                        });
                    let reference_color = band_sources
                        .iter()
                        .find(|(_, source)| source.reference)
                        .and_then(|(_, source)| source_xyz(source, rx, ry, options));
                    let scene_depth = depth_map
                        .and_then(|map| map.sample_nearest(rx, ry))
                        .and_then(|node| node.depth.map(|depth| (depth, node.confidence)));
                    // Chroma comes from colour modules (XYZ), luminance from
                    // every module including panchromatic ones.
                    let mut xyz = [0.0f32; 3];
                    let mut color_weight = 0.0f32;
                    let mut luminance = 0.0f32;
                    let mut luminance_weight = 0.0f32;
                    let mut reference_xyz = [0.0f32; 3];
                    let mut reference_color_weight = 0.0f32;
                    let mut reference_luminance_sum = 0.0f32;
                    let mut reference_luminance_weight = 0.0f32;
                    let mut sharpest_agreeing_detail = 1.0f32;
                    let mut luminance_owner = None::<(usize, f32)>;
                    let mut color_owner = None::<(usize, f32)>;
                    let mut reference_source_index = None;
                    for (source_index, source) in &band_sources {
                        let Some(q) = source.alignment.warp.map(rx, ry) else {
                            continue;
                        };
                        let mosaic = source.mosaic;
                        let Some((rgb, sensor_white)) = mosaic.sample_rgb_with_white(q[0], q[1])
                        else {
                            continue;
                        };
                        let border = q[0]
                            .min(q[1])
                            .min((mosaic.width - 1) as f32 - q[0])
                            .min((mosaic.height - 1) as f32 - q[1]);
                        let feather = smoothstep(border / options.feather_px.max(1.0));
                        let local_confidence = source.alignment.warp.confidence(rx, ry);
                        if feather <= 0.0 || local_confidence <= 0.0 {
                            continue;
                        }
                        let source_index = *source_index;
                        if source.reference {
                            reference_source_index = Some(source_index);
                        }
                        source_counters[source_index]
                            .sampled
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let base_weight = feather
                            * local_confidence
                            * source.magnification
                            * source.magnification
                            * source.confidence;
                        let focus_weight = focus_consistency_weight(
                            scene_depth,
                            source.focus_distance,
                            source.magnification,
                            source.reference,
                        );
                        if focus_weight < 0.5 {
                            source_counters[source_index]
                                .focus_suppressed
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        let gain = source.alignment.gain;
                        let offset = source.alignment.offset;
                        let field = source
                            .gain_field
                            .at(q[0], q[1], mosaic.width, mosaic.height);
                        if mosaic.is_mono() || !source.color.calibrated {
                            let y = (gain * (rgb[1] - offset)).max(0.0) * field[1];
                            let sample_structure = (!source.reference)
                                .then(|| source_log_luminance_structure(source, rx, ry, options))
                                .flatten();
                            let mut edge_weight =
                                edge_consistency_weight(reference_luminance, y, source.reference)
                                    * detail_consistency_weight(
                                        reference_structure,
                                        sample_structure,
                                        source.reference,
                                    );
                            sharpest_agreeing_detail = sharpest_agreeing_detail.max(
                                agreeing_detail_gain(reference_structure, sample_structure)
                                    * focus_weight,
                            );
                            let rejected = edge_weight < 0.1;
                            if rejected {
                                edge_weight = 0.0;
                            }
                            if !source.reference && reference_luminance.is_some() {
                                band_edge_checked += 1;
                                band_edge_rejected += usize::from(rejected);
                            }
                            edge_weight *= focus_weight;
                            let weight = base_weight * edge_weight;
                            luminance += weight * y;
                            luminance_weight += weight;
                            if luminance_owner.is_none_or(|(_, best)| weight > best) {
                                luminance_owner = Some((source_index, weight));
                            }
                            if source.reference {
                                reference_luminance_sum += weight * y;
                                reference_luminance_weight += weight;
                            }
                        } else {
                            let matched_rgb = rgb.map(|v| (gain * (v - offset)).max(0.0));
                            let matched_white =
                                sensor_white.map(|v| (gain * (v - offset)).max(0.0));
                            let mut matched = source.color.xyz_for_output(
                                matched_rgb,
                                matched_white,
                                options.highlight_correction,
                            );
                            for c in 0..3 {
                                matched[c] *= field[c];
                            }
                            let sample_structure = (!source.reference)
                                .then(|| source_log_luminance_structure(source, rx, ry, options))
                                .flatten();
                            let mut edge_weight = edge_consistency_weight(
                                reference_luminance,
                                matched[1],
                                source.reference,
                            ) * detail_consistency_weight(
                                reference_structure,
                                sample_structure,
                                source.reference,
                            );
                            sharpest_agreeing_detail = sharpest_agreeing_detail.max(
                                agreeing_detail_gain(reference_structure, sample_structure)
                                    * focus_weight,
                            );
                            let rejected = edge_weight < 0.1;
                            if rejected {
                                edge_weight = 0.0;
                            }
                            if !source.reference && reference_luminance.is_some() {
                                band_edge_checked += 1;
                                band_edge_rejected += usize::from(rejected);
                            }
                            edge_weight *= focus_weight;
                            let luminance_source_weight = base_weight * edge_weight;
                            let chroma_weight = chroma_consistency_weight(
                                reference_color,
                                Some(matched),
                                source.reference,
                            );
                            source_counters[source_index]
                                .color_sampled
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if chroma_weight < 0.5 {
                                source_counters[source_index]
                                    .chroma_suppressed
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            let color_source_weight = base_weight * edge_weight * chroma_weight;
                            for c in 0..3 {
                                xyz[c] += color_source_weight * matched[c];
                            }
                            color_weight += color_source_weight;
                            luminance += luminance_source_weight * matched[1];
                            luminance_weight += luminance_source_weight;
                            if luminance_owner
                                .is_none_or(|(_, best)| luminance_source_weight > best)
                            {
                                luminance_owner = Some((source_index, luminance_source_weight));
                            }
                            if color_owner.is_none_or(|(_, best)| color_source_weight > best) {
                                color_owner = Some((source_index, color_source_weight));
                            }
                            if source.reference {
                                for c in 0..3 {
                                    reference_xyz[c] += color_source_weight * matched[c];
                                }
                                reference_color_weight += color_source_weight;
                                reference_luminance_sum += luminance_source_weight * matched[1];
                                reference_luminance_weight += luminance_source_weight;
                            }
                        }
                    }
                    let other_scale = reference_detail_protection_scale(
                        reference_structure,
                        reference_luminance_weight,
                        luminance_weight,
                        sharpest_agreeing_detail,
                    );
                    if other_scale < 1.0 {
                        luminance = reference_luminance_sum
                            + (luminance - reference_luminance_sum) * other_scale;
                        luminance_weight = reference_luminance_weight
                            + (luminance_weight - reference_luminance_weight) * other_scale;
                        if reference_color_weight > 0.0 {
                            for c in 0..3 {
                                xyz[c] =
                                    reference_xyz[c] + (xyz[c] - reference_xyz[c]) * other_scale;
                            }
                            color_weight = reference_color_weight
                                + (color_weight - reference_color_weight) * other_scale;
                        }
                    }
                    if let Some(reference_index) = reference_source_index {
                        if let Some((owner, weight)) = luminance_owner
                            && owner != reference_index
                            && reference_luminance_weight > weight * other_scale
                        {
                            luminance_owner = Some((reference_index, reference_luminance_weight));
                        }
                        if let Some((owner, weight)) = color_owner
                            && owner != reference_index
                            && reference_color_weight > weight * other_scale
                        {
                            color_owner = Some((reference_index, reference_color_weight));
                        }
                    }
                    let pixel =
                        &mut band[(row_offset * width + u) * 3..(row_offset * width + u) * 3 + 3];
                    if luminance_weight > 0.0 {
                        band_covered += 1;
                        if let Some((owner, _)) = luminance_owner {
                            source_counters[owner]
                                .luminance_owner
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        if let Some((owner, _)) = color_owner {
                            source_counters[owner]
                                .color_owner
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        if !luminance_ownership.is_empty()
                            && u % OWNERSHIP_STEP == OWNERSHIP_STEP / 2
                            && v % OWNERSHIP_STEP == OWNERSHIP_STEP / 2
                        {
                            let diagnostic_index =
                                (v / OWNERSHIP_STEP) * ownership_columns + u / OWNERSHIP_STEP;
                            if diagnostic_index < luminance_ownership.len() {
                                luminance_ownership[diagnostic_index].store(
                                    luminance_owner.map_or(usize::MAX, |(owner, _)| owner),
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                color_ownership[diagnostic_index].store(
                                    color_owner.map_or(usize::MAX, |(owner, _)| owner),
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                        }
                        let target_luminance = luminance / luminance_weight;
                        let blended = if color_weight > 0.0 {
                            let mean = xyz.map(|v| v / color_weight);
                            if mean[1] > 1e-6 {
                                mean.map(|v| v * target_luminance / mean[1])
                            } else {
                                D50_WHITE.map(|v| v * target_luminance)
                            }
                        } else {
                            D50_WHITE.map(|v| v * target_luminance)
                        };
                        let rgb = color.apply(blended, options.color);
                        for c in 0..3 {
                            pixel[c] = (rgb[c].clamp(0.0, 1.0) * 65535.0).round() as u16;
                        }
                    }
                }
            }
            covered.fetch_add(band_covered, std::sync::atomic::Ordering::Relaxed);
            edge_checked.fetch_add(band_edge_checked, std::sync::atomic::Ordering::Relaxed);
            edge_rejected.fetch_add(band_edge_rejected, std::sync::atomic::Ordering::Relaxed);
            samples_to_be_bytes(&band, bytes);
        },
    )?;

    if let Some(directory) = diagnostic_dir {
        write_ownership_diagnostic(
            &directory.join("source-luminance-ownership.png"),
            ownership_columns,
            ownership_rows,
            &luminance_ownership,
        )?;
        write_ownership_diagnostic(
            &directory.join("source-color-ownership.png"),
            ownership_columns,
            ownership_rows,
            &color_ownership,
        )?;
    }

    let mut modules = usable
        .iter()
        .map(|(_, source)| (source.alignment.name.clone(), source.magnification))
        .collect::<Vec<_>>();
    modules.sort_by(|a, b| b.1.total_cmp(&a.1));
    let covered_pixels = covered.load(std::sync::atomic::Ordering::Relaxed);
    let source_contributions = usable
        .iter()
        .zip(&source_counters)
        .enumerate()
        .map(|(source_index, ((_, source), counters))| {
            let sampled = counters.sampled.load(std::sync::atomic::Ordering::Relaxed);
            let color_sampled = counters
                .color_sampled
                .load(std::sync::atomic::Ordering::Relaxed);
            SourceContributionReport {
                camera: source.alignment.name.clone(),
                diagnostic_rgb: ownership_color(source_index),
                luminance_owner_fraction: fraction(
                    counters
                        .luminance_owner
                        .load(std::sync::atomic::Ordering::Relaxed),
                    covered_pixels,
                ),
                color_owner_fraction: fraction(
                    counters
                        .color_owner
                        .load(std::sync::atomic::Ordering::Relaxed),
                    covered_pixels,
                ),
                focus_suppressed_fraction: fraction(
                    counters
                        .focus_suppressed
                        .load(std::sync::atomic::Ordering::Relaxed),
                    sampled,
                ),
                chroma_suppressed_fraction: fraction(
                    counters
                        .chroma_suppressed
                        .load(std::sync::atomic::Ordering::Relaxed),
                    color_sampled,
                ),
            }
        })
        .collect();
    Ok(SynthReport {
        canvas_width: width,
        canvas_height: height,
        scale,
        crop: [crop.x, crop.y, crop.width, crop.height],
        modules: modules.into_iter().map(|(name, _)| name).collect(),
        covered: covered_pixels as f32 / (width * height) as f32,
        highlight_correction: options.highlight_correction,
        raw_highlight_recovery: options.highlight_recovery,
        demosaic: options.demosaic,
        edge_rejected_fraction: fraction(
            edge_rejected.load(std::sync::atomic::Ordering::Relaxed),
            edge_checked.load(std::sync::atomic::Ordering::Relaxed),
        ),
        source_contributions,
        ownership_diagnostic_step: OWNERSHIP_STEP,
    })
}

fn ownership_color(index: usize) -> [u8; 3] {
    const PALETTE: [[u8; 3]; 16] = [
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
    ];
    PALETTE[index % PALETTE.len()]
}

fn write_ownership_diagnostic(
    path: &Path,
    width: usize,
    height: usize,
    ownership: &[std::sync::atomic::AtomicUsize],
) -> Result<()> {
    let mut labels = ownership
        .iter()
        .map(|owner| owner.load(std::sync::atomic::Ordering::Relaxed))
        .collect::<Vec<_>>();
    // Ownership is decided per sampled pixel and is intentionally sensitive
    // to fine texture. Two small majority passes make the diagnostic readable
    // as regions without changing any synthesis weights.
    for _ in 0..2 {
        let source = labels.clone();
        for row in 0..height {
            for column in 0..width {
                let mut counts = [0u8; 16];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (x, y) = (column as i32 + dx, row as i32 + dy);
                        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                            continue;
                        }
                        let label = source[y as usize * width + x as usize];
                        if label < counts.len() {
                            counts[label] += 1;
                        }
                    }
                }
                if let Some((label, &count)) = counts.iter().enumerate().max_by_key(|(_, n)| *n)
                    && count >= 3
                {
                    labels[row * width + column] = label;
                }
            }
        }
    }
    let mut pixels = Vec::with_capacity(width * height * 3);
    for index in labels {
        let color = if index == usize::MAX {
            [0; 3]
        } else {
            ownership_color(index).map(|channel| u16::from(channel) * 257)
        };
        pixels.extend(color);
    }
    write_rgb16_native_atomic(path, width, height, &pixels)
}

fn fraction(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}

fn source_luminance(
    source: &SynthSource<'_>,
    rx: f32,
    ry: f32,
    options: &SynthOptions,
) -> Option<f32> {
    let q = source.alignment.warp.map(rx, ry)?;
    let (rgb, sensor_white) = source.mosaic.sample_rgb_with_white(q[0], q[1])?;
    let gain = source.alignment.gain;
    let offset = source.alignment.offset;
    let field = source
        .gain_field
        .at(q[0], q[1], source.mosaic.width, source.mosaic.height);
    if source.mosaic.is_mono() || !source.color.calibrated {
        Some((gain * (rgb[1] - offset)).max(0.0) * field[1])
    } else {
        let matched_rgb = rgb.map(|value| (gain * (value - offset)).max(0.0));
        let matched_white = sensor_white.map(|value| (gain * (value - offset)).max(0.0));
        Some(
            source
                .color
                .xyz_for_output(matched_rgb, matched_white, options.highlight_correction)[1]
                * field[1],
        )
    }
}

fn source_xyz(
    source: &SynthSource<'_>,
    rx: f32,
    ry: f32,
    options: &SynthOptions,
) -> Option<[f32; 3]> {
    if source.mosaic.is_mono() || !source.color.calibrated {
        return None;
    }
    let q = source.alignment.warp.map(rx, ry)?;
    let (rgb, sensor_white) = source.mosaic.sample_rgb_with_white(q[0], q[1])?;
    let gain = source.alignment.gain;
    let offset = source.alignment.offset;
    let matched_rgb = rgb.map(|value| (gain * (value - offset)).max(0.0));
    let matched_white = sensor_white.map(|value| (gain * (value - offset)).max(0.0));
    let mut xyz =
        source
            .color
            .xyz_for_output(matched_rgb, matched_white, options.highlight_correction);
    let field = source
        .gain_field
        .at(q[0], q[1], source.mosaic.width, source.mosaic.height);
    for channel in 0..3 {
        xyz[channel] *= field[channel];
    }
    Some(xyz)
}

/// Conservative close-side defocus prior. Dense-depth labels describe
/// residual inverse-depth parallax on top of the global warp, which already
/// absorbs an unknown dominant scene plane. Treating a label as an absolute
/// subject distance would therefore be wrong. If autofocus selected that
/// dominant plane, `focus / residual_depth` is the additional relative
/// dioptric displacement towards the camera; magnified sources are more
/// sensitive to it. Image-measured sharpness remains authoritative.
fn focus_consistency_weight(
    residual: Option<(f64, f32)>,
    focus_distance: Option<f64>,
    magnification: f32,
    reference: bool,
) -> f32 {
    if reference || magnification <= 1.05 {
        return 1.0;
    }
    let (Some((residual_depth, confidence)), Some(focus)) = (residual, focus_distance) else {
        return 1.0;
    };
    if !residual_depth.is_finite() || !focus.is_finite() || residual_depth <= 0.0 || focus <= 0.0 {
        return 1.0;
    }
    let mismatch = (focus / residual_depth) as f32 * (magnification - 1.0);
    let relative = mismatch / 0.75;
    let supported = 1.0 / (1.0 + relative.powi(4));
    1.0 - confidence.clamp(0.0, 1.0) * (1.0 - supported)
}

/// Compare colour independently of luminance. A defocused chromatic halo may
/// preserve total luminance and therefore pass the structural checks; D50 XYZ
/// chromaticity exposes that disagreement without rejecting exposure changes.
fn chroma_consistency_weight(
    reference: Option<[f32; 3]>,
    sample: Option<[f32; 3]>,
    is_reference: bool,
) -> f32 {
    if is_reference {
        return 1.0;
    }
    let (Some(reference), Some(sample)) = (reference, sample) else {
        return 1.0;
    };
    let chromaticity = |xyz: [f32; 3]| {
        let sum = (xyz[0] + xyz[1] + xyz[2]).max(1.0e-5);
        [xyz[0] / sum, xyz[2] / sum]
    };
    let reference_chroma = chromaticity(reference);
    let sample_chroma = chromaticity(sample);
    let difference =
        (reference_chroma[0] - sample_chroma[0]).hypot(reference_chroma[1] - sample_chroma[1]);
    let relative = difference / 0.055;
    let robust = 1.0 / (1.0 + relative.powi(4));
    // Chroma is unstable in deep shadows. Fade the decision in using the
    // darker of the two samples instead of manufacturing coloured speckle.
    let reliability = smoothstep(reference[1].min(sample[1]) / 0.02);
    1.0 - reliability * (1.0 - robust)
}

/// Reference-space log-luminance structure sampled through one module's warp.
/// Mapping all four neighbours independently accounts for local scale,
/// rotation, and distortion instead of assuming sensor axes match the canvas.
/// The third component is centre-surround contrast: unlike a gradient, it is
/// strong at the middle of a one-pixel dark twig surrounded by bright sky.
fn source_log_luminance_structure(
    source: &SynthSource<'_>,
    rx: f32,
    ry: f32,
    options: &SynthOptions,
) -> Option<[f32; 3]> {
    const RADIUS: f32 = 1.5;
    let log_sample = |x, y| {
        source_luminance(source, x, y, options).map(|value| (1.0 + 64.0 * value.max(0.0)).ln())
    };
    let (centre, left, right, above, below) = (
        log_sample(rx, ry)?,
        log_sample(rx - RADIUS, ry)?,
        log_sample(rx + RADIUS, ry)?,
        log_sample(rx, ry - RADIUS)?,
        log_sample(rx, ry + RADIUS)?,
    );
    Some([
        (right - left) / (2.0 * RADIUS),
        (below - above) / (2.0 * RADIUS),
        (left + right + above + below) * 0.25 - centre,
    ])
}

/// Preserve reference detail while still accepting genuinely sharper samples.
/// A source is penalised when its edge is weaker than the reference or points
/// in a different direction. Extra contrast in the same direction is allowed,
/// so a finer module can add detail instead of being forced to match the
/// reference lens's modulation exactly.
fn detail_consistency_weight(
    reference: Option<[f32; 3]>,
    sample: Option<[f32; 3]>,
    is_reference: bool,
) -> f32 {
    if is_reference {
        return 1.0;
    }
    let Some(reference) = reference else {
        return 1.0;
    };
    let reference_gradient = reference[0].hypot(reference[1]);
    let reference_ridge = reference[2].abs();
    let reference_strength = reference_gradient.max(reference_ridge);
    if reference_strength < 0.01 {
        return 1.0;
    }
    let Some(sample) = sample else {
        return 0.0;
    };
    let gradient_error = if reference_gradient >= 0.01 {
        let unit = [
            reference[0] / reference_gradient,
            reference[1] / reference_gradient,
        ];
        let parallel = sample[0] * unit[0] + sample[1] * unit[1];
        let perpendicular = sample[0] * unit[1] - sample[1] * unit[0];
        (reference_gradient - parallel)
            .max(0.0)
            .hypot(perpendicular)
    } else {
        0.0
    };
    let ridge_error = if reference_ridge >= 0.01 {
        let agreeing = sample[2] * reference[2].signum();
        (reference_ridge - agreeing).max(0.0)
    } else {
        0.0
    };
    let error = gradient_error.hypot(ridge_error);
    let tolerance = 0.012 + 0.30 * reference_strength;
    let relative = error / tolerance;
    1.0 / (1.0 + relative.powi(4))
}

/// Relative strength of a sharper source that reproduces the reference edge
/// direction or ridge sign. Such a source may own high-frequency detail rather
/// than being capped merely because it is not the reference camera.
fn agreeing_detail_gain(reference: Option<[f32; 3]>, sample: Option<[f32; 3]>) -> f32 {
    let (Some(reference), Some(sample)) = (reference, sample) else {
        return 1.0;
    };
    let reference_gradient = reference[0].hypot(reference[1]);
    let reference_ridge = reference[2].abs();
    let reference_strength = reference_gradient.max(reference_ridge);
    if reference_strength < 0.01 {
        return 1.0;
    }
    if reference_gradient >= reference_ridge {
        let sample_gradient = sample[0].hypot(sample[1]);
        if sample_gradient <= reference_gradient {
            return 1.0;
        }
        let cosine = (sample[0] * reference[0] + sample[1] * reference[1])
            / (sample_gradient * reference_gradient).max(1.0e-6);
        if cosine < 0.94 {
            return 1.0;
        }
        (sample_gradient / reference_gradient).clamp(1.0, 4.0)
    } else {
        let agreeing = sample[2] * reference[2].signum();
        if agreeing <= reference_ridge {
            1.0
        } else {
            (agreeing / reference_ridge).clamp(1.0, 4.0)
        }
    }
}

/// Cap the *combined* weight of other lenses around strong reference detail.
/// Per-source robust weights are insufficient when many small residual weights
/// collectively outvote a one-pixel twig. Flat regions retain their full
/// denoising average. A strong edge stays reference-owned unless a finer source
/// reproduces its direction with measurably greater contrast.
fn reference_detail_protection_scale(
    reference: Option<[f32; 3]>,
    reference_weight: f32,
    total_weight: f32,
    sharpest_agreeing_gain: f32,
) -> f32 {
    let Some(reference) = reference else {
        return 1.0;
    };
    if reference_weight <= 0.0 || total_weight <= reference_weight {
        return 1.0;
    }
    let strength = reference[0].hypot(reference[1]).max(reference[2].abs());
    if strength <= 0.01 {
        return 1.0;
    }
    let protection = smoothstep((strength - 0.01) / 0.08);
    let ownership = smoothstep((sharpest_agreeing_gain - 1.0) / 0.75);
    let protected_ratio = 0.5 + 2.5 * ownership;
    let maximum_other_ratio = 4.0 * (1.0 - protection) + protected_ratio * protection;
    let other_weight = total_weight - reference_weight;
    (maximum_other_ratio * reference_weight / other_weight).min(1.0)
}

/// Preserve agreeing high-resolution samples while rejecting the opposite
/// side of a misregistered edge. Log luminance makes the test relative across
/// shadows and highlights; a 20% difference keeps almost all weight, whereas
/// a twofold contradiction contributes only a few percent.
fn edge_consistency_weight(reference: Option<f32>, sample: f32, is_reference: bool) -> f32 {
    if is_reference {
        return 1.0;
    }
    let Some(reference) = reference else {
        return 1.0;
    };
    let difference =
        ((1.0 + 64.0 * reference.max(0.0)).ln() - (1.0 + 64.0 * sample.max(0.0)).ln()).abs();
    let relative = difference / 0.30;
    1.0 / (1.0 + relative.powi(4))
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
pub fn auto_exposure(reference: &Mosaic, color: &ModuleColor, highlight_correction: bool) -> f32 {
    let mut luminance = Vec::new();
    let step = 16;
    let mut y = 1;
    while y < reference.height - 2 {
        let mut x = 1;
        while x < reference.width - 2 {
            if let Some((rgb, sensor_white)) = reference.sample_rgb_with_white(x as f32, y as f32) {
                luminance.push(color.xyz_for_output(rgb, sensor_white, highlight_correction)[1]);
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
    highlight_correction: bool,
) -> (f32, f32) {
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    let step = 24usize;
    let mut y = step;
    while y + step < reference.height {
        let mut x = step;
        while x + step < reference.width {
            if let Some(q) = warp.map(x as f32, y as f32)
                && let Some((t, t_white)) = target.sample_rgb_with_white(q[0], q[1])
                && let Some((r, r_white)) = reference.sample_rgb_with_white(x as f32, y as f32)
            {
                let rv = reference_color.xyz_for_output(r, r_white, highlight_correction)[1];
                // A mono sample is already luminance in its own units.
                let tv = if target.is_mono() || !target_color.calibrated {
                    t[1]
                } else {
                    target_color.xyz_for_output(t, t_white, highlight_correction)[1]
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
    highlight_correction: bool,
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
                && let Some((t, t_white)) = target.sample_rgb_with_white(q[0], q[1])
                && let Some((r, r_white)) = reference.sample_rgb_with_white(x as f32, y as f32)
            {
                let column = ((q[0] / target.width as f32) * columns as f32)
                    .clamp(0.0, (columns - 1) as f32) as usize;
                let row = ((q[1] / target.height as f32) * rows as f32)
                    .clamp(0.0, (rows - 1) as f32) as usize;
                let cell = &mut ratios[row * columns + column];
                let r_xyz = reference_color.xyz_for_output(r, r_white, highlight_correction);
                if target.is_mono() || !target_color.calibrated {
                    let tv = (gain * (t[1] - offset)).max(0.0);
                    if r_xyz[1] > 0.01 && tv > 0.01 && r_xyz[1] < 0.95 && tv < 0.95 {
                        cell[1].push(r_xyz[1] / tv);
                    }
                } else {
                    let matched = t.map(|v| (gain * (v - offset)).max(0.0));
                    let matched_white = t_white.map(|v| (gain * (v - offset)).max(0.0));
                    let t_xyz =
                        target_color.xyz_for_output(matched, matched_white, highlight_correction);
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
            if target.is_mono() || !target_color.calibrated {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_consistency_keeps_matches_and_rejects_contradictions() {
        assert_eq!(edge_consistency_weight(Some(0.2), 0.2, false), 1.0);
        assert!(edge_consistency_weight(Some(0.2), 0.24, false) > 0.8);
        assert!(edge_consistency_weight(Some(0.1), 0.5, false) < 0.05);
        assert_eq!(edge_consistency_weight(Some(0.1), 0.5, true), 1.0);
    }

    #[test]
    fn detail_weight_preserves_reference_edges_but_accepts_sharper_agreement() {
        let reference = Some([0.1, 0.0, 0.0]);
        assert_eq!(
            detail_consistency_weight(reference, Some([0.1, 0.0, 0.0]), true),
            1.0
        );
        assert!(detail_consistency_weight(reference, Some([0.2, 0.0, 0.0]), false) > 0.99);
        assert!(detail_consistency_weight(reference, Some([0.03, 0.0, 0.0]), false) < 0.2);
        assert!(detail_consistency_weight(reference, Some([0.0, 0.1, 0.0]), false) < 0.1);
        assert!(detail_consistency_weight(reference, Some([-0.1, 0.0, 0.0]), false) < 0.05);
    }

    #[test]
    fn detail_weight_protects_a_thin_dark_line_with_zero_centre_gradient() {
        let dark_line = Some([0.0, 0.0, 0.12]);
        assert!(detail_consistency_weight(dark_line, Some([0.0, 0.0, 0.13]), false) > 0.99);
        assert!(detail_consistency_weight(dark_line, Some([0.0, 0.0, 0.0]), false) < 0.05);
        assert!(detail_consistency_weight(dark_line, Some([0.0, 0.0, -0.12]), false) < 0.01);
    }

    #[test]
    fn combined_non_reference_weight_cannot_erase_strong_thin_detail() {
        assert_eq!(
            reference_detail_protection_scale(Some([0.0, 0.0, 0.0]), 1.0, 10.0, 1.0),
            1.0
        );
        let scale = reference_detail_protection_scale(Some([0.0, 0.0, 0.2]), 1.0, 10.0, 1.0);
        assert!((scale - 0.5 / 9.0).abs() < 1.0e-6);

        let sharper = reference_detail_protection_scale(Some([0.0, 0.0, 0.2]), 1.0, 10.0, 2.0);
        assert!(sharper > scale * 5.0);
    }

    #[test]
    fn focus_prior_rejects_close_content_only_for_magnified_sources() {
        let close = Some((500.0, 1.0));
        assert!(focus_consistency_weight(close, Some(2_500.0), 2.25, false) < 0.01);
        assert_eq!(
            focus_consistency_weight(close, Some(2_500.0), 1.0, false),
            1.0
        );
        assert!(focus_consistency_weight(Some((10_000.0, 1.0)), Some(2_500.0), 2.25, false) > 0.97);
        assert_eq!(
            focus_consistency_weight(None, Some(2_500.0), 2.25, false),
            1.0
        );
    }

    #[test]
    fn chroma_is_checked_independently_of_luminance() {
        let neutral = Some([0.2, 0.2, 0.2]);
        assert!(chroma_consistency_weight(neutral, Some([0.4, 0.4, 0.4]), false) > 0.99);
        assert!(chroma_consistency_weight(neutral, Some([0.38, 0.2, 0.02]), false) < 0.1);
        assert_eq!(
            chroma_consistency_weight(neutral, Some([0.38, 0.2, 0.02]), true),
            1.0
        );
    }

    #[test]
    fn sharper_agreeing_detail_can_own_an_edge() {
        assert_eq!(
            agreeing_detail_gain(Some([0.1, 0.0, 0.0]), Some([0.25, 0.0, 0.0])),
            2.5
        );
        assert_eq!(
            agreeing_detail_gain(Some([0.1, 0.0, 0.0]), Some([0.0, 0.25, 0.0])),
            1.0
        );
    }

    #[test]
    fn common_balanced_white_removes_magenta_from_clipped_neutral() {
        let color = ModuleColor {
            wb_gains: [2.0, 1.0, 1.5],
            ..ModuleColor::default()
        };

        // Green has reached sensor white while red and blue continue towards
        // their higher post-WB white points: the unhandled result is magenta.
        let raw = [0.8, 1.0, 0.9];
        assert_eq!(color.to_xyz(raw), [1.6, 1.0, 1.349_999_9]);
        assert_eq!(color.to_xyz_clipped(raw, [1.0; 3]), [1.349_999_9; 3]);
        assert_eq!(color.xyz_for_output(raw, [1.0; 3], true), [1.349_999_9; 3]);
        assert_eq!(
            color.xyz_for_output(raw, [1.0; 3], false),
            color.to_xyz(raw)
        );
    }

    #[test]
    fn common_balanced_white_preserves_unclipped_color() {
        let color = ModuleColor {
            wb_gains: [2.0, 1.0, 1.5],
            ..ModuleColor::default()
        };
        let raw = [0.2, 0.35, 0.3];
        assert_eq!(color.to_xyz_clipped(raw, [1.0; 3]), color.to_xyz(raw));
    }

    #[test]
    fn highlight_reconstruction_has_a_smooth_shoulder() {
        let color = ModuleColor::default();
        let below = color.to_xyz_clipped([0.939, 0.5, 0.5], [1.0; 3]);
        let entering = color.to_xyz_clipped([0.941, 0.5, 0.5], [1.0; 3]);
        let clipped = color.to_xyz_clipped([1.0, 0.5, 0.5], [1.0; 3]);
        assert!((below[0] - entering[0]).abs() < 0.003);
        assert_eq!(clipped, [0.5; 3]);
    }

    #[test]
    fn d50_profile_white_maps_to_neutral_srgb() {
        let pipeline = ColorPipeline::default();
        let rgb = pipeline.apply(D50_WHITE, OutputColor::Linear);
        for channel in rgb {
            assert!((channel - 1.0).abs() < 2e-4, "channel was {channel}");
        }
    }
}
