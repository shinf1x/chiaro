//! The fusion pipeline: hot-pixel removal -> alignment -> synthesis.
//!
//! Each stage consumes and produces plain data (`Mosaic`s, `ModuleAlignment`s,
//! a PNG on disk plus a report), so stages can be swapped or inspected
//! independently. Progress is reported through a callback; nothing here knows
//! about a UI.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use chiaro::lri::{RawCamera, parse_raw_layout};
use chiaro_hotpixel_core::{
    highlight::{HighlightRecoveryReport, HighlightRecoveryState, recover_bayer_highlights},
    hotpixel::HotpixelRec,
    pipeline::{CleanupStage, FramePipeline, extract_raw_plane_threaded},
    thermal::ThermalProfile,
    universal_hotpixel::UniversalHotpixelProfile,
};
use serde::Serialize;

use crate::align::{AlignInput, AlignOptions, AlignmentReport, ModuleAlignment, align_module};
use crate::calibration::{
    CalibrationDatabase, CameraCalibration, IntrinsicsMode, LriMessages, ModuleFocusState,
    awb_gains, image_focal_length_mm, module_states,
};
use crate::depth::refine_multiview_depth;
use crate::geometry::{CameraRefinement, ResolvedCamera};
use crate::image::{Mosaic, Plane};
use crate::synth::{
    ColorPipeline, CropWindow, GainField, ModuleColor, SynthOptions, SynthReport, SynthSource,
    auto_exposure, canvas_scale, photometric_field, photometric_match, synthesize,
};

/// Hot-pixel stage settings. `None` skips the stage.
#[derive(Clone, Debug)]
pub struct HotpixelStage {
    pub rec: PathBuf,
    pub universal_model: bool,
    pub glow_correction: bool,
}

#[derive(Clone, Debug)]
pub struct FusionOptions {
    /// Reference module; defaults to the capture's own reference camera.
    pub reference: Option<String>,
    /// Extra calibration files (`calibration.lri`, `zoom_calib_v0.lri`).
    pub overlays: Vec<PathBuf>,
    pub intrinsics_mode: IntrinsicsMode,
    pub hotpixel: Option<HotpixelStage>,
    /// Modules to use; empty means every RAW module in the capture.
    pub cameras: Vec<String>,
    pub align: AlignOptions,
    pub synth: SynthOptions,
    /// Apply the factory vignetting meshes as flat-field gains.
    pub flat_field: bool,
    /// Fit a coarse per-module gain field (in addition to the global match)
    /// that removes slow brightness and colour differences across a module.
    pub local_photometric: bool,
    /// Crop the output to the field of view the photographer framed
    /// (`image_focal_length`, 35 mm equivalent) instead of the full reference
    /// frame.
    pub crop_to_framing: bool,
    /// Write per-module alignment checkerboards (`<module>_check.png`) here.
    pub debug_dir: Option<PathBuf>,
    /// Threads for per-frame kernels (`0` = all cores).
    pub threads: usize,
}

impl Default for FusionOptions {
    fn default() -> Self {
        Self {
            reference: None,
            overlays: Vec::new(),
            intrinsics_mode: IntrinsicsMode::LinearHall,
            hotpixel: None,
            cameras: Vec::new(),
            align: AlignOptions::default(),
            synth: SynthOptions::default(),
            flat_field: true,
            local_photometric: true,
            crop_to_framing: true,
            debug_dir: None,
            threads: 0,
        }
    }
}

/// Progress of a run, for status displays.
#[derive(Clone, Debug)]
pub struct Progress {
    pub stage: &'static str,
    pub detail: String,
    /// Overall fraction, 0..=1.
    pub fraction: f32,
}

/// Everything worth knowing about one run, written next to the output.
#[derive(Clone, Debug, Serialize)]
pub struct FusionReport {
    pub reference: String,
    pub calibration_modules: usize,
    /// 35 mm-equivalent focal length recorded for the framing, if any.
    pub framed_focal_length_mm: Option<i32>,
    pub modules: Vec<AlignmentReport>,
    /// RAW-domain clipped-sample reconstruction performed per module.
    pub highlights: Vec<(String, HighlightRecoveryReport)>,
    /// Per module: `(name, luminance gain, luminance offset)`.
    pub gains: Vec<(String, f32, f32)>,
    pub synthesis: SynthReport,
    pub seconds: FusionTimings,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionTimings {
    pub load: f32,
    pub hotpixel: f32,
    pub align: f32,
    pub synthesize: f32,
}

/// 35 mm-equivalent focal length of the wide (A) modules, the reference view.
pub const WIDE_EQUIVALENT_FOCAL_MM: f32 = 28.0;
/// 35 mm-equivalent native fields of view for the medium (B) and tele (C)
/// groups. A B+C capture uses a B module as its reference, so treating that
/// raster as a 28 mm view would crop it a second time and discard most of it.
pub const MEDIUM_EQUIVALENT_FOCAL_MM: f32 = 70.0;
pub const TELE_EQUIVALENT_FOCAL_MM: f32 = 150.0;
/// Grid of the per-module photometric gain field (cells of ~350 x 350 px).
const GAIN_FIELD_COLUMNS: usize = 12;
const GAIN_FIELD_ROWS: usize = 9;

/// White balance and forward matrix for one module. The recorded AWB gains
/// describe the reference module; a module with different D65 grey ratios
/// gets them rescaled so a grey object stays grey in its own camera space.
fn module_color(
    module: Option<&CameraCalibration>,
    reference: Option<&CameraCalibration>,
    recorded_wb: Option<[f32; 3]>,
) -> ModuleColor {
    fn d65(cal: Option<&CameraCalibration>) -> Option<&crate::calibration::ColorProfile> {
        cal.and_then(|c| {
            c.color
                .iter()
                .find(|p| p.illuminant == 2)
                .or(c.color.first())
        })
    }
    let (profile, reference_profile) = (d65(module), d65(reference));
    let mut color = ModuleColor::default();
    if let Some(profile) = profile {
        color.forward = profile.forward_matrix.map(|row| row.map(|v| v as f32));
        color.calibrated = true;
    }
    color.wb_gains = match (recorded_wb, profile, reference_profile) {
        (Some(wb), Some(p), Some(r)) => [
            wb[0] * (r.rg_ratio / p.rg_ratio.max(1e-3)) as f32,
            wb[1],
            wb[2] * (r.bg_ratio / p.bg_ratio.max(1e-3)) as f32,
        ],
        (Some(wb), _, _) => wb,
        (None, Some(p), _) => [
            (1.0 / p.rg_ratio.max(0.01)) as f32,
            1.0,
            (1.0 / p.bg_ratio.max(0.01)) as f32,
        ],
        (None, None, _) => [1.0; 3],
    };
    color
}

/// Whether a module's warp lands inside its sensor anywhere on a 5x5 grid of
/// probes over the crop (a cheap "contributes to the framed view" test).
fn intersects_crop(alignment: &ModuleAlignment, module: &LoadedModule, crop: &CropWindow) -> bool {
    (0..5).any(|i| {
        (0..5).any(|j| {
            let x = crop.x + crop.width * (0.1 + 0.2 * i as f32);
            let y = crop.y + crop.height * (0.1 + 0.2 * j as f32);
            alignment.warp.map(x, y).is_some_and(|q| {
                q[0] >= 0.0
                    && q[1] >= 0.0
                    && q[0] <= (module.raw.width - 1) as f32
                    && q[1] <= (module.raw.height - 1) as f32
            })
        })
    })
}

/// Nominal focal length (pixels) of each L16 focal group, used only when no
/// calibration is available for a module.
pub fn nominal_focal_px(camera: &str) -> f64 {
    match camera.chars().next().map(|c| c.to_ascii_uppercase()) {
        Some('B') => 8300.0,
        Some('C') => 18700.0,
        _ => 3380.0,
    }
}

/// Native 35 mm-equivalent field of view of the reference camera's focal
/// group. The recorded framing focal length is a crop relative to this value,
/// not always relative to the A group's 28 mm view.
pub fn group_equivalent_focal_mm(camera: &str) -> f32 {
    match camera.chars().next().map(|c| c.to_ascii_uppercase()) {
        Some('B') => MEDIUM_EQUIVALENT_FOCAL_MM,
        Some('C') => TELE_EQUIVALENT_FOCAL_MM,
        _ => WIDE_EQUIVALENT_FOCAL_MM,
    }
}

fn framing_crop(
    width: usize,
    height: usize,
    reference_camera: &str,
    framed_focal_length_mm: f32,
) -> CropWindow {
    CropWindow::centred(
        width,
        height,
        group_equivalent_focal_mm(reference_camera) / framed_focal_length_mm,
    )
}

/// Blend confidence for an accepted refined alignment. The synthesis stage
/// already rewards resolution by magnification squared; this counterweight
/// stops a barely supported tele frame from overwhelming the reference.
fn correspondence_confidence(inlier_ratio: f32, minimum: f32) -> f32 {
    let minimum = minimum.clamp(0.0, 1.0);
    let reliable = (minimum + 0.25).min(1.0);
    let t = ((inlier_ratio - minimum) / (reliable - minimum).max(1e-3)).clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)).max(0.05)
}

struct LoadedModule {
    raw: RawCamera,
    mosaic: Mosaic,
    camera: Option<ResolvedCamera>,
    focus: ModuleFocusState,
    highlight: HighlightRecoveryState,
}

/// Replace only low-confidence spatial reconstructions for which at least two
/// other modules provide geometrically consistent, genuinely measured RAW
/// samples. Per-channel overlap ratios account for exposure/transmission
/// differences without applying white balance or a colour matrix prematurely.
#[derive(Clone, Copy)]
pub struct RawHighlightSource<'a> {
    pub mosaic: &'a Mosaic,
    pub highlight: &'a HighlightRecoveryState,
    pub alignment: &'a ModuleAlignment,
}

#[derive(Clone, Copy, Debug)]
pub struct RawHighlightUpdate {
    pub index: usize,
    pub value: u16,
    pub confidence: u8,
}

/// Calculate conservative donor replacements for one module without mutating
/// any source. Keeping calculation and application separate lets callers own
/// mosaics in different pipeline-specific containers.
pub fn cross_camera_highlight_updates(
    sources: &[RawHighlightSource<'_>],
    target_index: usize,
    reference_width: usize,
    reference_height: usize,
) -> Vec<RawHighlightUpdate> {
    if sources.len() < 3 || target_index >= sources.len() {
        return Vec::new();
    }
    let target = &sources[target_index];
    if !target.alignment.report.accepted
        || target.mosaic.is_mono()
        || target.highlight.confidence.is_empty()
    {
        return Vec::new();
    }
    let ratios = sources
        .iter()
        .enumerate()
        .map(|(donor_index, donor)| {
            if target_index == donor_index || !donor.alignment.report.accepted {
                [None; 3]
            } else {
                raw_channel_ratios(target, donor, reference_width, reference_height)
            }
        })
        .collect::<Vec<_>>();
    // Build a donor radiance field first. Applying accepted estimates during
    // this pass would turn the binary geometry/consensus decision into a
    // salt-and-pepper CFA mask after demosaic.
    let mut candidates = vec![None; target.mosaic.samples.len()];
    for y in 0..target.mosaic.height {
        for x in 0..target.mosaic.width {
            let index = y * target.mosaic.width + x;
            if !target.highlight.needs_donor(index) {
                continue;
            }
            let Some(reference) = invert_warp(
                &target.alignment.warp,
                [x as f32, y as f32],
                reference_width,
                reference_height,
            ) else {
                continue;
            };
            if target.alignment.warp.confidence(reference[0], reference[1]) < 0.7 {
                continue;
            }
            let channel = target.mosaic.pattern.color_at(y, x);
            let mut estimates = Vec::with_capacity(sources.len() - 1);
            for (donor_index, donor) in sources.iter().enumerate() {
                let Some(ratio) = ratios[donor_index][channel] else {
                    continue;
                };
                if donor.alignment.warp.confidence(reference[0], reference[1]) < 0.7 {
                    continue;
                }
                let Some(q) = donor.alignment.warp.map(reference[0], reference[1]) else {
                    continue;
                };
                let Some((sample, confidence)) =
                    donor
                        .mosaic
                        .sample_raw_channel(q[0], q[1], channel, donor.highlight)
                else {
                    continue;
                };
                if confidence == 255 && sample < 0.985 {
                    estimates.push(sample * ratio);
                }
            }
            if let Some((estimate, confidence)) = consistent_donor_estimate(&mut estimates) {
                candidates[index] = Some((estimate.max(0.995), confidence));
            }
        }
    }

    // Regularise each CFA phase independently, preserving radiance edges with
    // a range weight. Neighbourhood support becomes a continuous feather, so
    // donor coverage and occlusion boundaries fade into the spatial estimate
    // rather than making hard per-pixel replacements.
    let mut updates = Vec::new();
    const OFFSETS: [(isize, isize, f32); 9] = [
        (-2, -2, 1.0),
        (0, -2, 2.0),
        (2, -2, 1.0),
        (-2, 0, 2.0),
        (0, 0, 4.0),
        (2, 0, 2.0),
        (-2, 2, 1.0),
        (0, 2, 2.0),
        (2, 2, 1.0),
    ];
    for y in 0..target.mosaic.height {
        for x in 0..target.mosaic.width {
            let index = y * target.mosaic.width + x;
            let Some((centre, donor_confidence)) = candidates[index] else {
                continue;
            };
            let mut weighted = 0.0;
            let mut total_weight = 0.0;
            let mut support_weight = 0.0;
            let mut possible_weight = 0.0;
            for (dx, dy, spatial_weight) in OFFSETS {
                let Some(nx) = x.checked_add_signed(dx) else {
                    continue;
                };
                let Some(ny) = y.checked_add_signed(dy) else {
                    continue;
                };
                if nx >= target.mosaic.width || ny >= target.mosaic.height {
                    continue;
                }
                possible_weight += spatial_weight;
                let Some((neighbour, _)) = candidates[ny * target.mosaic.width + nx] else {
                    continue;
                };
                support_weight += spatial_weight;
                let relative_difference = (neighbour - centre).abs() / centre.max(0.05);
                let range_weight = 1.0 / (1.0 + relative_difference / 0.08).powi(2);
                let weight = spatial_weight * range_weight;
                weighted += neighbour * weight;
                total_weight += weight;
            }
            if total_weight <= 0.0 || possible_weight <= 0.0 {
                continue;
            }
            let support = (support_weight / possible_weight).clamp(0.0, 1.0);
            let feather = smoothstep((support - 0.2) / 0.65);
            let donor_strength = feather * (f32::from(donor_confidence) / 255.0);
            if donor_strength < 0.02 {
                continue;
            }
            let estimate = weighted / total_weight;
            let range = (target.mosaic.white_q6 - target.mosaic.black_q6).max(1.0);
            let spatial = ((f32::from(target.mosaic.samples[index]) - target.mosaic.black_q6)
                / range)
                .max(0.0);
            let blended = spatial + (estimate - spatial) * donor_strength;
            let old_confidence = target.highlight.confidence[index];
            let confidence = f32::from(old_confidence)
                + (f32::from(donor_confidence) - f32::from(old_confidence)) * donor_strength;
            updates.push(RawHighlightUpdate {
                index,
                value: target.mosaic.normalized_raw_to_q6(blended.max(spatial)),
                confidence: confidence.round().clamp(1.0, 254.0) as u8,
            });
        }
    }
    updates
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn recover_cross_camera_highlights(
    modules: &mut [LoadedModule],
    alignments: &[ModuleAlignment],
    reference_width: usize,
    reference_height: usize,
) {
    if modules.len() < 3 || modules.len() != alignments.len() {
        return;
    }
    let updates = {
        let sources = modules
            .iter()
            .zip(alignments)
            .map(|(module, alignment)| RawHighlightSource {
                mosaic: &module.mosaic,
                highlight: &module.highlight,
                alignment,
            })
            .collect::<Vec<_>>();
        (0..modules.len())
            .map(|target| {
                cross_camera_highlight_updates(&sources, target, reference_width, reference_height)
            })
            .collect::<Vec<_>>()
    };
    for target_index in 0..modules.len() {
        let target = &mut modules[target_index];
        for update in &updates[target_index] {
            target.mosaic.samples[update.index] = update.value;
            target
                .highlight
                .mark_multi_camera(update.index, update.confidence);
        }
        target.highlight.finish_multi_camera();
    }
}

/// Robust target/donor response ratios from unclipped overlap samples.
fn raw_channel_ratios(
    target: &RawHighlightSource<'_>,
    donor: &RawHighlightSource<'_>,
    reference_width: usize,
    reference_height: usize,
) -> [Option<f32>; 3] {
    let mut samples: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    for y in (24..reference_height.saturating_sub(24)).step_by(48) {
        for x in (24..reference_width.saturating_sub(24)).step_by(48) {
            let (x, y) = (x as f32, y as f32);
            if target.alignment.warp.confidence(x, y) < 0.75
                || donor.alignment.warp.confidence(x, y) < 0.75
            {
                continue;
            }
            let (Some(tq), Some(dq)) = (
                target.alignment.warp.map(x, y),
                donor.alignment.warp.map(x, y),
            ) else {
                continue;
            };
            for (channel, ratios) in samples.iter_mut().enumerate() {
                let Some((target_value, target_confidence)) =
                    target
                        .mosaic
                        .sample_raw_channel(tq[0], tq[1], channel, target.highlight)
                else {
                    continue;
                };
                let Some((donor_value, donor_confidence)) =
                    donor
                        .mosaic
                        .sample_raw_channel(dq[0], dq[1], channel, donor.highlight)
                else {
                    continue;
                };
                if target_confidence == 255
                    && donor_confidence == 255
                    && (0.03..0.94).contains(&target_value)
                    && (0.03..0.94).contains(&donor_value)
                {
                    let ratio = target_value / donor_value;
                    if (0.2..5.0).contains(&ratio) {
                        ratios.push(ratio);
                    }
                }
            }
        }
    }
    std::array::from_fn(|channel| robust_median(&mut samples[channel], 32))
}

fn robust_median(values: &mut [f32], minimum: usize) -> Option<f32> {
    if values.len() < minimum {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let trim = values.len() / 10;
    let retained = &values[trim..values.len() - trim];
    Some(retained[retained.len() / 2])
}

fn consistent_donor_estimate(values: &mut [f32]) -> Option<(f32, u8)> {
    if values.len() < 2 {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let median = values[values.len() / 2];
    if !median.is_finite() || median <= 0.0 {
        return None;
    }
    let mut deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f32::total_cmp);
    let relative_mad = deviations[deviations.len() / 2] / median;
    if relative_mad > 0.12 {
        return None;
    }
    let confidence = (220 + values.len().saturating_sub(2).min(4) * 6) as u8;
    Some((median, confidence))
}

/// Numerically invert the reference-to-module warp near the corresponding
/// sensor coordinate. The L16 rasters share dimensions, so `[qx,qy]` is a
/// useful initial guess even for tele modules; Newton updates handle the
/// calibrated/refined displacement.
fn invert_warp(
    warp: &crate::align::Warp,
    target: [f32; 2],
    reference_width: usize,
    reference_height: usize,
) -> Option<[f32; 2]> {
    let mut point = [
        target[0].clamp(0.0, (reference_width - 1) as f32),
        target[1].clamp(0.0, (reference_height - 1) as f32),
    ];
    for _ in 0..8 {
        let mapped = warp.map(point[0], point[1])?;
        let error = [mapped[0] - target[0], mapped[1] - target[1]];
        if error[0].abs().max(error[1].abs()) < 0.35 {
            return Some(point);
        }
        let dx = warp.map((point[0] + 1.0).min((reference_width - 1) as f32), point[1])?;
        let dy = warp.map(
            point[0],
            (point[1] + 1.0).min((reference_height - 1) as f32),
        )?;
        let j00 = dx[0] - mapped[0];
        let j10 = dx[1] - mapped[1];
        let j01 = dy[0] - mapped[0];
        let j11 = dy[1] - mapped[1];
        let determinant = j00 * j11 - j01 * j10;
        if determinant.abs() < 1e-5 {
            return None;
        }
        let update_x = (error[0] * j11 - error[1] * j01) / determinant;
        let update_y = (j00 * error[1] - j10 * error[0]) / determinant;
        point[0] =
            (point[0] - update_x.clamp(-64.0, 64.0)).clamp(0.0, (reference_width - 1) as f32);
        point[1] =
            (point[1] - update_y.clamp(-64.0, 64.0)).clamp(0.0, (reference_height - 1) as f32);
    }
    let mapped = warp.map(point[0], point[1])?;
    ((mapped[0] - target[0])
        .abs()
        .max((mapped[1] - target[1]).abs())
        < 0.75)
        .then_some(point)
}

/// Run the whole pipeline on an in-memory LRI and write `output` (16-bit PNG)
/// plus `<output>.fusion.json`.
pub fn fuse(
    lri: &[u8],
    options: &FusionOptions,
    output: &Path,
    progress: &mut dyn FnMut(Progress),
) -> Result<FusionReport> {
    let started = Instant::now();
    let mut timings = FusionTimings::default();
    progress(Progress {
        stage: "load",
        detail: "parsing capture".to_owned(),
        fraction: 0.0,
    });

    // Capture metadata and calibration.
    let messages = LriMessages::parse(lri)?;
    let overlays = options
        .overlays
        .iter()
        .map(|path| {
            LriMessages::parse(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
                .with_context(|| format!("parse {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let calibration = CalibrationDatabase::from_capture_and_overlays(&messages, &overlays);
    let states = module_states(&messages)
        .into_iter()
        .map(|state| (state.name.clone(), state))
        .collect::<HashMap<_, _>>();
    let layout = parse_raw_layout(lri, &HashMap::new()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let selected = layout
        .cameras
        .iter()
        .filter(|camera| {
            options.cameras.is_empty()
                || options
                    .cameras
                    .iter()
                    .any(|wanted| wanted.eq_ignore_ascii_case(&camera.name))
        })
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("capture has no selected RAW modules");
    }
    let reference_name = options
        .reference
        .clone()
        .or_else(|| {
            messages
                .headers
                .iter()
                .find_map(|h| h.image_reference_camera)
                .map(|id| crate::calibration::camera_name(id.value()))
        })
        .unwrap_or_else(|| "A1".to_owned())
        .to_ascii_uppercase();
    if !selected.iter().any(|camera| camera.name == reference_name) {
        bail!("reference module {reference_name} is not among the selected modules");
    }

    // Stage 1: hot-pixel removal per module, producing calibration-raster mosaics.
    let hotpixel_models = match &options.hotpixel {
        Some(stage) => Some((
            HotpixelRec::open(&stage.rec).map_err(|e| anyhow::anyhow!("hotpixel.rec: {e:#}"))?,
            stage
                .universal_model
                .then(UniversalHotpixelProfile::bundled)
                .transpose()?,
            stage
                .glow_correction
                .then(ThermalProfile::bundled)
                .transpose()?,
        )),
        None => None,
    };
    timings.load = started.elapsed().as_secs_f32();
    let stage_started = Instant::now();
    let mut modules = Vec::with_capacity(selected.len());
    for (index, raw) in selected.iter().enumerate() {
        progress(Progress {
            stage: "hotpixel",
            detail: raw.name.clone(),
            fraction: 0.05 + 0.25 * index as f32 / selected.len() as f32,
        });
        let samples_q6 = match &hotpixel_models {
            Some((rec, universal, thermal)) => {
                let map = rec
                    .load_rotated_map(raw.id, raw.width, raw.height)
                    .map_err(|e| anyhow::anyhow!("{}: {e:#}", raw.name))?;
                let pipeline = FramePipeline {
                    universal_hotpixel: universal.as_ref(),
                    thermal: thermal.as_ref(),
                    cleanup: CleanupStage::Disabled,
                    threads: options.threads,
                    ..FramePipeline::default()
                };
                pipeline
                    .correct_lri(lri, raw, &map)
                    .map_err(|e| anyhow::anyhow!("{}: {e:#}", raw.name))?
                    .samples_q6
            }
            None => extract_raw_plane_threaded(lri, raw, options.threads)
                .map_err(|e| anyhow::anyhow!("{}: {e:#}", raw.name))?
                .into_iter()
                .map(|s| s << 6)
                .collect(),
        };
        let mut mosaic = Mosaic::from_stream_q6(
            samples_q6,
            raw.width,
            raw.height,
            raw.pattern,
            raw.black_level,
            raw.white_level,
        );
        if options.synth.highlight_recovery
            != chiaro_hotpixel_core::highlight::HighlightRecovery::None
            && !mosaic.is_mono()
        {
            mosaic.reserve_highlight_headroom();
        }
        let highlight = recover_bayer_highlights(
            &mut mosaic.samples,
            mosaic.width,
            mosaic.height,
            mosaic.pattern,
            mosaic.black_q6,
            mosaic.white_q6,
            options.synth.highlight_recovery,
        )
        .with_context(|| format!("RAW highlight recovery {}", raw.name))?;
        let state = states.get(&raw.name).cloned();
        if options.flat_field
            && let Some(vignetting) = calibration
                .cameras
                .get(&raw.name)
                .and_then(|c| c.vignetting.as_ref())
        {
            let mirror_hall = state.as_ref().map_or(0.0, |s| s.mirror_hall);
            mosaic.vignetting = vignetting.mesh_for_hall(mirror_hall);
            if !mosaic.is_mono() {
                mosaic.crosstalk = vignetting.crosstalk.clone();
            }
        }
        let camera = match (&state, calibration.cameras.get(&raw.name)) {
            (Some(state), Some(cal)) => ResolvedCamera::new(
                cal,
                state,
                options.intrinsics_mode,
                &CameraRefinement::default(),
            )
            .ok(),
            _ => None,
        };
        let focus = state.map_or_else(ModuleFocusState::default, |state| state.focus);
        modules.push(LoadedModule {
            raw: raw.clone(),
            mosaic,
            camera,
            focus,
            highlight,
        });
    }
    timings.hotpixel = stage_started.elapsed().as_secs_f32();

    // Stage 2: alignment to the reference, modules in parallel.
    let stage_started = Instant::now();
    progress(Progress {
        stage: "align",
        detail: "building luminance pyramids".to_owned(),
        fraction: 0.3,
    });
    let luminance = modules
        .iter()
        .map(|module| module.mosaic.luminance_half())
        .collect::<Vec<Plane>>();
    let reference_index = modules
        .iter()
        .position(|module| module.raw.name == reference_name)
        .expect("reference selected");
    let inputs = modules
        .iter()
        .zip(&luminance)
        .map(|(module, luminance)| AlignInput {
            name: &module.raw.name,
            luminance,
            width: module.raw.width,
            height: module.raw.height,
            camera: module.camera.as_ref(),
            nominal_focal_px: module
                .camera
                .as_ref()
                .map(|c| c.focal_px)
                .unwrap_or_else(|| nominal_focal_px(&module.raw.name)),
        })
        .collect::<Vec<_>>();
    let reference_input = &inputs[reference_index];
    let mut alignments = std::thread::scope(|scope| {
        let handles = inputs
            .iter()
            .map(|input| {
                let options = &options.align;
                scope.spawn(move || align_module(reference_input, input, options))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("alignment worker panicked"))
            .collect::<Result<Vec<ModuleAlignment>>>()
    })?;
    for (module, alignment) in modules.iter().zip(&mut alignments) {
        alignment.report.focus_achieved = module.focus.achieved;
        alignment.report.calibrated_focus_distance = module
            .camera
            .as_ref()
            .and_then(|camera| camera.focus_distance);
        alignment.report.disparity_focus_distance = module.focus.disparity_distance;
        alignment.report.contrast_focus_distance = module.focus.contrast_distance;
        alignment.report.focus_roi = module.focus.roi;
        alignment.report.lens_timeout = module.focus.lens_timeout;
        alignment.report.mirror_timeout = module.focus.mirror_timeout;
    }
    let depth_map = if options.align.refine && options.align.depth.enabled {
        progress(Progress {
            stage: "align",
            detail: "multi-view local depth refinement".to_owned(),
            fraction: 0.45,
        });
        refine_multiview_depth(
            &inputs,
            reference_index,
            &mut alignments,
            &options.align.depth,
        )
    } else {
        None
    };
    if options.synth.highlight_recovery.uses_multi_camera() {
        progress(Progress {
            stage: "highlight",
            detail: "geometry-gated cross-camera recovery".to_owned(),
            fraction: 0.52,
        });
        let reference_dimensions = (
            modules[reference_index].raw.width,
            modules[reference_index].raw.height,
        );
        recover_cross_camera_highlights(
            &mut modules,
            &alignments,
            reference_dimensions.0,
            reference_dimensions.1,
        );
    }
    if let Some(debug_dir) = &options.debug_dir {
        fs::create_dir_all(debug_dir).with_context(|| format!("create {}", debug_dir.display()))?;
        if let Some(depth_map) = &depth_map {
            depth_map.write_diagnostics(
                &debug_dir.join("depth-inverse.png"),
                &debug_dir.join("depth-provenance.png"),
            )?;
            depth_map.write_visualization(&debug_dir.join("depth-visualization.png"))?;
        }
        for module in &modules {
            if module.highlight.confidence.is_empty() {
                continue;
            }
            let uncertainty = module
                .highlight
                .confidence
                .iter()
                .map(|&confidence| {
                    if confidence == 255 {
                        0
                    } else {
                        u16::from(255 - confidence) * 257
                    }
                })
                .collect::<Vec<_>>();
            chiaro_hotpixel_core::png16::write_gray16_native_atomic(
                &debug_dir.join(format!("{}_highlight-uncertainty.png", module.raw.name)),
                module.raw.width,
                module.raw.height,
                &uncertainty,
            )?;
        }
        for (module, alignment) in modules.iter().zip(&alignments) {
            if module.raw.name == reference_name {
                continue;
            }
            let (samples, width, height) = crate::align::debug_checkerboard(
                &luminance[reference_index],
                &luminance[modules
                    .iter()
                    .position(|m| m.raw.name == module.raw.name)
                    .unwrap()],
                &alignment.warp,
                64,
            );
            chiaro_hotpixel_core::png16::write_gray16_native_atomic(
                &debug_dir.join(format!("{}_check.png", module.raw.name)),
                width,
                height,
                &samples,
            )?;
        }
    }

    // Colour per module: its own D50-output DNG forward matrix, and the
    // recorded white balance transferred from the reference through the D65
    // calibration grey ratios. Synthesis adapts the blended D50 XYZ to D65.
    let reference_calibration = calibration.cameras.get(&reference_name);
    let recorded_wb = awb_gains(&messages).map(|g| [g[0] as f32, g[1] as f32, g[2] as f32]);
    let module_colors = modules
        .iter()
        .map(|module| {
            module_color(
                calibration.cameras.get(&module.raw.name),
                reference_calibration,
                recorded_wb,
            )
        })
        .collect::<Vec<_>>();

    // Advanced demosaicing is prepared only for geometrically accepted colour
    // modules. This avoids allocating an RGB cache for rejected cameras.
    for (module, alignment) in modules.iter_mut().zip(&alignments) {
        if alignment.report.accepted && !module.mosaic.is_mono() {
            module
                .mosaic
                .prepare_demosaic(options.synth.demosaic, options.threads)
                .with_context(|| format!("demosaic {}", module.raw.name))?;
        }
    }

    // Photometric matching: a global luminance gain and offset against the
    // reference, then a coarse per-module gain field for the slow remainder
    // (mirror-path glare, colour shading).
    let mut gain_fields = vec![GainField::identity(); modules.len()];
    for index in 0..modules.len() {
        if modules[index].raw.name != reference_name && alignments[index].report.accepted {
            let (gain, offset) = photometric_match(
                &modules[reference_index].mosaic,
                &module_colors[reference_index],
                &modules[index].mosaic,
                &module_colors[index],
                &alignments[index].warp,
                options.synth.highlight_correction,
            );
            alignments[index].gain = gain;
            alignments[index].offset = offset;
            if options.local_photometric {
                // A narrow module sees too little of the reference to
                // constrain a full-frame gain grid. Use one robust XYZ gain
                // over its measured overlap instead of extrapolating sparse
                // cells, which produced strong magenta/green blocks.
                let (columns, rows) = if alignments[index].report.coverage >= 0.5 {
                    (GAIN_FIELD_COLUMNS, GAIN_FIELD_ROWS)
                } else {
                    (1, 1)
                };
                gain_fields[index] = photometric_field(
                    &modules[reference_index].mosaic,
                    &module_colors[reference_index],
                    &modules[index].mosaic,
                    &module_colors[index],
                    &alignments[index].warp,
                    gain,
                    offset,
                    columns,
                    rows,
                    options.synth.highlight_correction,
                );
            }
        }
    }
    timings.align = stage_started.elapsed().as_secs_f32();

    // Stage 3: framing, canvas, and synthesis.
    let stage_started = Instant::now();
    let reference = &modules[reference_index];
    let framed_focal_length_mm = image_focal_length_mm(&messages);
    let crop = match framed_focal_length_mm {
        Some(focal) if options.crop_to_framing && focal > 0 => {
            // Framing is relative to the native field of view of the capture's
            // reference group: 28 mm for A, 70 mm for B, and 150 mm for C.
            framing_crop(
                reference.raw.width,
                reference.raw.height,
                &reference_name,
                focal as f32,
            )
        }
        _ => CropWindow::full(reference.raw.width, reference.raw.height),
    };
    let reference_focal = reference
        .camera
        .as_ref()
        .map(|c| c.focal_px)
        .unwrap_or_else(|| nominal_focal_px(&reference_name));
    let magnification = |module: &LoadedModule| {
        (module
            .camera
            .as_ref()
            .map(|c| c.focal_px)
            .unwrap_or_else(|| nominal_focal_px(&module.raw.name))
            / reference_focal) as f32
    };
    let synthesis_confidence = |alignment: &ModuleAlignment| {
        if !options.align.refine || alignment.name == reference_name {
            return 1.0;
        }
        // Smoothly suppress barely accepted modules. A small floor lets them
        // fill otherwise uncovered areas without allowing a high-resolution
        // but low-consensus tele frame to dominate the reference.
        correspondence_confidence(
            alignment.report.inlier_ratio,
            options.align.min_inlier_ratio,
        )
    };
    // The finest module that intersects the framed view decides the maximum
    // useful canvas resolution ("as much detail as any module provides").
    let finest = modules
        .iter()
        .zip(&alignments)
        .filter(|(module, alignment)| {
            alignment.report.accepted
                && (options.synth.include_mono || !module.mosaic.is_mono())
                && intersects_crop(alignment, module, &crop)
        })
        .map(|(module, _)| magnification(module))
        .fold(1.0f32, f32::max);
    let scale = canvas_scale(&crop, reference.raw.width, options.synth.canvas, finest);
    progress(Progress {
        stage: "synthesize",
        detail: format!(
            "{}x{} canvas",
            (crop.width * scale).round() as usize,
            (crop.height * scale).round() as usize
        ),
        fraction: 0.55,
    });
    let color = ColorPipeline {
        exposure: auto_exposure(
            &reference.mosaic,
            &module_colors[reference_index],
            options.synth.highlight_correction,
        ),
    };
    let sources = modules
        .iter()
        .zip(&alignments)
        .zip(module_colors.iter().zip(&gain_fields))
        .filter(|((_, alignment), _)| alignment.report.accepted)
        .map(|((module, alignment), (color, gain_field))| SynthSource {
            mosaic: &module.mosaic,
            alignment,
            reference: alignment.name == reference_name,
            magnification: magnification(module),
            confidence: synthesis_confidence(alignment)
                * if module.focus.mirror_timeout {
                    0.1
                } else if module.focus.lens_timeout {
                    0.25
                } else {
                    1.0
                },
            focus_distance: module
                .camera
                .as_ref()
                .and_then(|camera| camera.focus_distance),
            color: *color,
            gain_field: gain_field.clone(),
        })
        .collect::<Vec<_>>();
    let synthesis = synthesize(
        output,
        crop,
        scale,
        &sources,
        depth_map.as_ref(),
        options.debug_dir.as_deref(),
        &color,
        &options.synth,
    )?;
    timings.synthesize = stage_started.elapsed().as_secs_f32();

    let report = FusionReport {
        reference: reference_name,
        calibration_modules: calibration.cameras.len(),
        framed_focal_length_mm,
        modules: alignments.iter().map(|a| a.report.clone()).collect(),
        highlights: modules
            .iter()
            .map(|module| (module.raw.name.clone(), module.highlight.report.clone()))
            .collect(),
        gains: alignments
            .iter()
            .map(|a| (a.name.clone(), a.gain, a.offset))
            .collect(),
        synthesis,
        seconds: timings,
    };
    let report_path = output.with_extension("fusion.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    progress(Progress {
        stage: "done",
        detail: output.display().to_string(),
        fraction: 1.0,
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chiaro::lri::SensorPattern;
    use chiaro_hotpixel_core::highlight::{HighlightRecovery, HighlightRecoveryReport};

    #[test]
    fn framing_crop_is_relative_to_the_reference_camera_group() {
        let a = framing_crop(4_000, 3_000, "A1", 100.0);
        let b = framing_crop(4_000, 3_000, "B4", 100.0);
        let c = framing_crop(4_000, 3_000, "C2", 150.0);

        assert_eq!(a.width, 1_120.0);
        assert_eq!(b.width, 2_800.0);
        assert_eq!(b.height, 2_100.0);
        assert_eq!(c, CropWindow::full(4_000, 3_000));
    }

    #[test]
    fn correspondence_confidence_only_rewards_clear_consensus() {
        let minimum = 0.45;
        assert_eq!(correspondence_confidence(minimum, minimum), 0.05);
        assert!((correspondence_confidence(0.575, minimum) - 0.5).abs() < 1e-6);
        assert_eq!(correspondence_confidence(0.70, minimum), 1.0);
        assert_eq!(correspondence_confidence(0.90, minimum), 1.0);
    }

    #[test]
    fn cross_camera_recovery_requires_agreeing_measured_donors() {
        let (width, height) = (512, 512);
        let centre = 256 * width + 256;
        let make_mosaic = |clipped: bool| {
            let mut samples = (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| match (y & 1, x & 1) {
                        (0, 0) => 30_000,
                        (1, 1) => 18_000,
                        _ => 24_000,
                    })
                })
                .collect::<Vec<_>>();
            if clipped {
                for y in 250..=262 {
                    for x in 250..=262 {
                        samples[y * width + x] = 65_535;
                    }
                }
            }
            Mosaic {
                width,
                height,
                pattern: SensorPattern::Rggb,
                samples,
                black_q6: 0.0,
                white_q6: 65_535.0,
                vignetting: None,
                crosstalk: None,
                demosaiced_rgb: None,
            }
        };
        let mosaics = [make_mosaic(true), make_mosaic(false), make_mosaic(false)];
        let states = [
            HighlightRecoveryState {
                confidence: {
                    let mut confidence = vec![255; width * height];
                    for y in 250..=262 {
                        for x in 250..=262 {
                            confidence[y * width + x] = 0;
                        }
                    }
                    confidence
                },
                report: HighlightRecoveryReport {
                    mode: HighlightRecovery::MultiCamera,
                    clipped_samples: 13 * 13,
                    ..Default::default()
                },
            },
            HighlightRecoveryState {
                confidence: vec![255; width * height],
                report: HighlightRecoveryReport::default(),
            },
            HighlightRecoveryState {
                confidence: vec![255; width * height],
                report: HighlightRecoveryReport::default(),
            },
        ];
        let alignments = (0..3)
            .map(|index| ModuleAlignment {
                name: format!("B{index}"),
                warp: crate::align::Warp::from_fn(width, height, 32, Some),
                gain: 1.0,
                offset: 0.0,
                report: AlignmentReport {
                    accepted: true,
                    ..Default::default()
                },
            })
            .collect::<Vec<_>>();
        let sources = (0..3)
            .map(|index| RawHighlightSource {
                mosaic: &mosaics[index],
                highlight: &states[index],
                alignment: &alignments[index],
            })
            .collect::<Vec<_>>();

        let mut isolated_confidence = vec![255; width * height];
        isolated_confidence[centre] = 0;
        let isolated_state = HighlightRecoveryState {
            confidence: isolated_confidence,
            report: HighlightRecoveryReport::default(),
        };
        let isolated_target = RawHighlightSource {
            mosaic: &mosaics[0],
            highlight: &isolated_state,
            alignment: &alignments[0],
        };
        let isolated_sources = [isolated_target, sources[1], sources[2]];
        assert!(
            cross_camera_highlight_updates(&isolated_sources, 0, width, height)
                .iter()
                .all(|update| update.index != centre)
        );

        let updates = cross_camera_highlight_updates(&sources, 0, width, height);
        let recovered = updates
            .iter()
            .find(|update| update.index == centre)
            .unwrap();
        assert!(recovered.value >= 65_000);
        assert!(recovered.confidence >= 180);
    }
}
