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
use chiaro::lri::{
    NoiseModel, RawCamera, parse_frame_layout, parse_raw_layout, sensor_characterization_type,
};
use chiaro_hotpixel_core::{
    cleanup::{CleanupCameraProfile, CleanupDiagnostics, CleanupProfile},
    highlight::{HighlightRecoveryReport, HighlightRecoveryState, recover_bayer_highlights},
    hotpixel::HotpixelRec,
    pipeline::{CleanupStage, FramePipeline, extract_raw_plane_threaded},
    thermal::ThermalProfile,
    universal_hotpixel::UniversalHotpixelProfile,
};
use serde::Serialize;

use crate::align::{AlignInput, AlignOptions, AlignmentReport, ModuleAlignment, align_module};
use crate::array_color::{
    ArrayColorSelectionReport, ArrayColorSource, ColorProfileMode, ProfileBlend,
    blended_profile as blended_array_profile, module_color_for_blend, select_array_profile,
};
use crate::calibration::{
    CalibrationDatabase, CameraCalibration, IntrinsicsMode, LriMessages, ModuleFocusState,
    ModuleState, awb_gains, image_focal_length_mm, module_states,
};
use crate::crosstalk::{
    AdaptiveCrosstalkReport, CrosstalkFitSource, CrosstalkMode, fit_adaptive_crosstalk,
};
use crate::depth::refine_multiview_depth;
use crate::geometry::{CameraRefinement, ResolvedCamera};
use crate::image::{Mosaic, Plane};
use crate::resolution::refine_resolution_warp;
use crate::rig::{
    RigCameraInput, RigRefinementOptions, RigRefinementReport, gate_on_image_space_alignment,
    refine_capture_rig,
};
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
    /// Optional camera-specific learned defect and line calibration. The
    /// archive is validated against `rec` and opened once per fusion run.
    pub cleanup_profile: Option<PathBuf>,
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
    /// Physical modules retained for geometry and real-CFA validation but
    /// excluded completely from reconstruction.
    pub cfa_held_out: Vec<String>,
    pub align: AlignOptions,
    /// Capture-specific bounded physical rig refinement, accepted only on an
    /// independently held-out correspondence subset.
    pub rig_refinement: RigRefinementOptions,
    pub synth: SynthOptions,
    /// Factory-only, disabled, or capture-adaptive CFA-phase crosstalk.
    pub crosstalk: CrosstalkMode,
    /// Factory colour-profile selection strategy.
    pub color_profile: ColorProfileMode,
    /// Apply the factory vignetting meshes as flat-field gains.
    pub flat_field: bool,
    /// Fit a coarse per-module gain field (in addition to the global match)
    /// that removes slow brightness and colour differences across a module.
    pub local_photometric: bool,
    /// Crop the output to the field of view the photographer framed
    /// (`image_focal_length`, 35 mm equivalent) instead of the full reference
    /// frame.
    pub crop_to_framing: bool,
    /// Explicit reference-raster crop for diagnostics and matched experiments.
    /// When present this takes precedence over `crop_to_framing`.
    pub crop: Option<CropWindow>,
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
            cfa_held_out: Vec::new(),
            align: AlignOptions::default(),
            rig_refinement: RigRefinementOptions::default(),
            synth: SynthOptions::default(),
            crosstalk: CrosstalkMode::default(),
            color_profile: ColorProfileMode::default(),
            flat_field: true,
            local_photometric: true,
            crop_to_framing: true,
            crop: None,
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
    /// Capture-specific physical orientation/mirror refinement performed
    /// before the downstream residual image-space warp.
    pub rig_refinement: RigRefinementReport,
    /// RAW-domain clipped-sample reconstruction performed per module.
    pub highlights: Vec<(String, HighlightRecoveryReport)>,
    /// Camera-specific learned cleanup availability and correction results.
    pub cleanup: Vec<(String, CleanupDiagnostics)>,
    /// Per-module factory-prior and capture-adaptive crosstalk fit.
    pub crosstalk: Vec<(String, AdaptiveCrosstalkReport)>,
    /// Illuminant estimate and factory colour-profile interpolation per module.
    pub color: Vec<ColorSelectionReport>,
    /// Sparse aligned-overlap evidence used for the common profile blend.
    pub array_color: ArrayColorSelectionReport,
    /// Per module: `(name, luminance gain, luminance offset)`.
    pub gains: Vec<(String, f32, f32)>,
    pub synthesis: SynthReport,
    pub seconds: FusionTimings,
    pub resources: FusionResources,
}

#[derive(Clone, Debug, Serialize)]
pub struct ColorSelectionReport {
    pub module: String,
    pub available_illuminants: Vec<String>,
    pub selected_illuminants: Vec<(String, f32)>,
    pub estimated_mired: Option<f32>,
    pub profile_source: &'static str,
    pub confidence: f32,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionTimings {
    pub load: f32,
    pub hotpixel: f32,
    pub align: f32,
    pub synthesize: f32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FusionResources {
    /// Process high-water resident set where the host exposes it (Linux
    /// `/proc/self/status`). This includes alignment and all synthesis stages.
    pub peak_resident_bytes: Option<u64>,
    pub output_megapixels: f32,
    pub total_seconds_per_megapixel: f32,
    pub synthesis_seconds_per_megapixel: f32,
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
#[derive(Clone, Copy, Debug)]
#[cfg(test)]
struct IlluminantSelection {
    first: i32,
    second: i32,
    second_weight: f64,
    estimated_mired: f64,
    confidence: f64,
}

#[cfg(test)]
fn illuminant_mired(illuminant: i32) -> Option<f64> {
    match illuminant {
        0 => Some(1_000_000.0 / 2_856.0),     // A
        1 => Some(1_000_000.0 / 5_003.0),     // D50
        2 => Some(1_000_000.0 / 6_504.0),     // D65
        3 => Some(1_000_000.0 / 7_504.0),     // D75
        4 => Some(1_000_000.0 / 4_230.0),     // F2
        5 => Some(1_000_000.0 / 6_500.0),     // F7
        6 | 7 => Some(1_000_000.0 / 4_000.0), // F11 / TL84
        _ => None,
    }
}

#[cfg(test)]
fn illuminant_selection(
    reference: Option<&CameraCalibration>,
    recorded_wb: Option<[f32; 3]>,
) -> Option<IlluminantSelection> {
    let reference = reference?;
    let Some(wb) = recorded_wb else {
        let profile = reference
            .color
            .iter()
            .find(|profile| profile.illuminant == 2)
            .or(reference.color.first())?;
        return Some(IlluminantSelection {
            first: profile.illuminant,
            second: profile.illuminant,
            second_weight: 0.0,
            estimated_mired: illuminant_mired(profile.illuminant).unwrap_or(0.0),
            confidence: 0.5,
        });
    };
    if wb[0] <= 0.0 || wb[1] <= 0.0 || wb[2] <= 0.0 {
        return None;
    }
    let target = [f64::from(wb[1] / wb[0]).ln(), f64::from(wb[1] / wb[2]).ln()];
    let mut anchors = reference
        .color
        .iter()
        .filter_map(|profile| {
            Some((
                illuminant_mired(profile.illuminant)?,
                profile.illuminant,
                [profile.rg_ratio.ln(), profile.bg_ratio.ln()],
            ))
        })
        .filter(|(_, _, ratio)| ratio.iter().all(|value| value.is_finite()))
        .collect::<Vec<_>>();
    anchors.sort_by(|a, b| a.0.total_cmp(&b.0));
    let first = *anchors.first()?;
    if anchors.len() == 1 {
        return Some(IlluminantSelection {
            first: first.1,
            second: first.1,
            second_weight: 0.0,
            estimated_mired: first.0,
            confidence: 0.25,
        });
    }
    anchors
        .windows(2)
        .map(|pair| {
            let (left, right) = (pair[0], pair[1]);
            let direction = [right.2[0] - left.2[0], right.2[1] - left.2[1]];
            let relative = [target[0] - left.2[0], target[1] - left.2[1]];
            let denominator = direction[0] * direction[0] + direction[1] * direction[1];
            let weight = if denominator > 1e-12 {
                ((relative[0] * direction[0] + relative[1] * direction[1]) / denominator)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            let projected = [
                left.2[0] + direction[0] * weight,
                left.2[1] + direction[1] * weight,
            ];
            let distance = (target[0] - projected[0]).hypot(target[1] - projected[1]);
            (
                distance,
                IlluminantSelection {
                    first: left.1,
                    second: right.1,
                    second_weight: weight,
                    estimated_mired: left.0 + (right.0 - left.0) * weight,
                    confidence: 1.0 / (1.0 + 4.0 * distance),
                },
            )
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, selection)| selection)
}

#[cfg(test)]
fn blended_profile(
    calibration: Option<&CameraCalibration>,
    selection: IlluminantSelection,
) -> Option<(crate::math::Mat3, f64, f64, bool)> {
    let calibration = calibration?;
    let first = calibration
        .color
        .iter()
        .find(|profile| profile.illuminant == selection.first)?;
    let second = calibration
        .color
        .iter()
        .find(|profile| profile.illuminant == selection.second)
        .unwrap_or(first);
    let first_matrix = first
        .validated_matrix
        .as_ref()
        .unwrap_or(&first.forward_matrix);
    let second_matrix = second
        .validated_matrix
        .as_ref()
        .unwrap_or(&second.forward_matrix);
    let weight = selection.second_weight;
    let matrix = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            first_matrix[row][column] * (1.0 - weight) + second_matrix[row][column] * weight
        })
    });
    let interpolate_ratio = |a: f64, b: f64| (a.ln() * (1.0 - weight) + b.ln() * weight).exp();
    Some((
        matrix,
        interpolate_ratio(first.rg_ratio, second.rg_ratio),
        interpolate_ratio(first.bg_ratio, second.bg_ratio),
        first.validated_matrix.is_some() || second.validated_matrix.is_some(),
    ))
}

#[cfg(test)]
fn module_color(
    module_name: &str,
    module: Option<&CameraCalibration>,
    reference: Option<&CameraCalibration>,
    recorded_wb: Option<[f32; 3]>,
) -> (ModuleColor, ColorSelectionReport) {
    let selection = illuminant_selection(reference, recorded_wb);
    let profile = selection.and_then(|value| blended_profile(module, value));
    let reference_profile = selection.and_then(|value| blended_profile(reference, value));
    let mut color = ModuleColor::default();
    if let Some((matrix, _, _, _)) = profile {
        color.forward = matrix.map(|row| row.map(|value| value as f32));
        color.calibrated = true;
    }
    color.wb_gains = match (recorded_wb, profile, reference_profile) {
        (Some(wb), Some((_, p_rg, p_bg, _)), Some((_, r_rg, r_bg, _))) => [
            wb[0] * (r_rg / p_rg.max(1e-3)) as f32,
            wb[1],
            wb[2] * (r_bg / p_bg.max(1e-3)) as f32,
        ],
        (Some(wb), _, _) => wb,
        (None, Some((_, rg, bg, _)), _) => [
            (1.0 / rg.max(0.01)) as f32,
            1.0,
            (1.0 / bg.max(0.01)) as f32,
        ],
        (None, None, _) => [1.0; 3],
    };
    let available_illuminants = module
        .map(|calibration| {
            calibration
                .color
                .iter()
                .map(|profile| {
                    crate::color_profile::illuminant_name(Some(profile.illuminant)).to_owned()
                })
                .collect()
        })
        .unwrap_or_default();
    let report = if let Some(selection) = selection {
        let second_weight = selection.second_weight as f32;
        let mut selected = vec![(
            crate::color_profile::illuminant_name(Some(selection.first)).to_owned(),
            1.0 - second_weight,
        )];
        if selection.second != selection.first && second_weight > 0.0 {
            selected.push((
                crate::color_profile::illuminant_name(Some(selection.second)).to_owned(),
                second_weight,
            ));
        }
        ColorSelectionReport {
            module: module_name.to_owned(),
            available_illuminants,
            selected_illuminants: selected,
            estimated_mired: Some(selection.estimated_mired as f32),
            profile_source: if !color.calibrated {
                "uncalibrated_luminance_only"
            } else if profile.is_some_and(|(_, _, _, validated)| validated) {
                if selection.first == selection.second {
                    "validated_macbeth_matrix"
                } else {
                    "interpolated_validated_and_factory_matrices"
                }
            } else if selection.first == selection.second {
                "factory_forward_matrix"
            } else {
                "interpolated_factory_forward_matrices"
            },
            confidence: if color.calibrated {
                selection.confidence as f32
            } else {
                0.0
            },
            fallback_reason: (!color.calibrated)
                .then_some("selected illuminant is unavailable for this module".to_owned()),
        }
    } else {
        ColorSelectionReport {
            module: module_name.to_owned(),
            available_illuminants,
            selected_illuminants: Vec::new(),
            estimated_mired: None,
            profile_source: "uncalibrated_luminance_only",
            confidence: 0.0,
            fallback_reason: Some("no usable colour calibration or white balance".to_owned()),
        }
    };
    (color, report)
}

fn module_color_for_selection(
    module_name: &str,
    module: Option<&CameraCalibration>,
    reference: Option<&CameraCalibration>,
    recorded_wb: Option<[f32; 3]>,
    blend: ProfileBlend,
    estimated_mired: Option<f32>,
    confidence: f32,
) -> (ModuleColor, ColorSelectionReport) {
    let profile = blended_array_profile(module, blend);
    let color = module_color_for_blend(module, reference, recorded_wb, blend).unwrap_or_default();
    let available_illuminants = module
        .map(|calibration| {
            calibration
                .color
                .iter()
                .map(|profile| {
                    crate::color_profile::illuminant_name(Some(profile.illuminant)).to_owned()
                })
                .collect()
        })
        .unwrap_or_default();
    let report = ColorSelectionReport {
        module: module_name.to_owned(),
        available_illuminants,
        selected_illuminants: blend.named_weights(),
        estimated_mired,
        profile_source: if !color.calibrated {
            "uncalibrated_luminance_only"
        } else if profile.is_some_and(|profile| profile.uses_validated_matrix) {
            "blended_validated_and_factory_matrices"
        } else if blend.named_weights().len() == 1 {
            "factory_forward_matrix"
        } else {
            "interpolated_factory_forward_matrices"
        },
        confidence: if color.calibrated { confidence } else { 0.0 },
        fallback_reason: (!color.calibrated)
            .then_some("selected profile blend is unavailable for this module".to_owned()),
    };
    (color, report)
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
    state: Option<ModuleState>,
    focus: ModuleFocusState,
    highlight: HighlightRecoveryState,
    cleanup: CleanupDiagnostics,
    capture_gain: f32,
    exposure_ns: u64,
    noise_model: Option<NoiseModel>,
}

struct LoadedHotpixelModels {
    rec: HotpixelRec,
    universal: Option<UniversalHotpixelProfile>,
    thermal: Option<ThermalProfile>,
    cleanup_requested: bool,
    cleanup_cameras: HashMap<usize, CleanupCameraProfile>,
}

fn alignment_inputs<'a>(
    modules: &'a [LoadedModule],
    luminance: &'a [Plane],
) -> Vec<AlignInput<'a>> {
    modules
        .iter()
        .zip(luminance)
        .map(|(module, luminance)| AlignInput {
            name: &module.raw.name,
            luminance,
            width: module.raw.width,
            height: module.raw.height,
            camera: module.camera.as_ref(),
            nominal_focal_px: module
                .camera
                .as_ref()
                .map(|camera| camera.focal_px)
                .unwrap_or_else(|| nominal_focal_px(&module.raw.name)),
        })
        .collect()
}

fn align_all_modules(
    inputs: &[AlignInput<'_>],
    reference_index: usize,
    options: &AlignOptions,
) -> Result<Vec<ModuleAlignment>> {
    let reference = &inputs[reference_index];
    std::thread::scope(|scope| {
        let handles = inputs
            .iter()
            .map(|input| scope.spawn(move || align_module(reference, input, options)))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("alignment worker panicked"))
            .collect::<Result<Vec<_>>>()
    })
}

fn correct_fusion_raw(
    lri: &[u8],
    raw: &RawCamera,
    models: &LoadedHotpixelModels,
    threads: usize,
) -> Result<(Vec<u16>, CleanupDiagnostics)> {
    let map = models
        .rec
        .load_rotated_map(raw.id, raw.width, raw.height)
        .map_err(|e| anyhow::anyhow!("{}: {e:#}", raw.name))?;
    let cleanup_camera = models.cleanup_cameras.get(&raw.id);
    let pipeline = FramePipeline {
        universal_hotpixel: models.universal.as_ref(),
        thermal: models.thermal.as_ref(),
        cleanup: CleanupStage::from_loaded(models.cleanup_requested, cleanup_camera),
        threads,
        ..FramePipeline::default()
    };
    let corrected = pipeline
        .correct_lri(lri, raw, &map)
        .map_err(|e| anyhow::anyhow!("{}: {e:#}", raw.name))?;
    Ok((
        corrected.samples_q6,
        CleanupDiagnostics::new(
            models.cleanup_requested,
            cleanup_camera.is_some(),
            corrected.cleanup,
        ),
    ))
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
    eligible: &[bool],
    reference_width: usize,
    reference_height: usize,
) {
    if modules.len() != alignments.len() || modules.len() != eligible.len() {
        return;
    }
    let updates = {
        let selected = eligible
            .iter()
            .enumerate()
            .filter_map(|(index, &enabled)| enabled.then_some(index))
            .collect::<Vec<_>>();
        let sources = selected
            .iter()
            .map(|&index| RawHighlightSource {
                mosaic: &modules[index].mosaic,
                highlight: &modules[index].highlight,
                alignment: &alignments[index],
            })
            .collect::<Vec<_>>();
        selected
            .iter()
            .enumerate()
            .map(|(target, &module_index)| {
                (
                    module_index,
                    cross_camera_highlight_updates(
                        &sources,
                        target,
                        reference_width,
                        reference_height,
                    ),
                )
            })
            .collect::<Vec<_>>()
    };
    for (target_index, updates) in updates {
        let target = &mut modules[target_index];
        for update in &updates {
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
    let frame_layout = parse_frame_layout(lri, &HashMap::new())
        .map_err(|e| anyhow::anyhow!("noise metadata: {e}"))?;
    // A held-out experiment loads every camera so geometry, crop, and scale
    // stay fixed across contributor ablations. `options.cameras` is applied
    // later as an admission mask; held-out and unselected radiance never enter
    // reconstruction or scene-fitted colour operations.
    let selected = layout
        .cameras
        .iter()
        .filter(|camera| {
            !options.cfa_held_out.is_empty()
                || options.cameras.is_empty()
                || options
                    .cameras
                    .iter()
                    .any(|wanted| wanted.eq_ignore_ascii_case(&camera.name))
                || options
                    .cfa_held_out
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
    if options
        .cfa_held_out
        .iter()
        .any(|camera| camera.eq_ignore_ascii_case(&reference_name))
    {
        bail!("reference module {reference_name} cannot be held out");
    }

    // Stage 1: hot-pixel removal per module, producing calibration-raster mosaics.
    let hotpixel_models = match &options.hotpixel {
        Some(stage) => {
            let rec = HotpixelRec::open(&stage.rec)
                .map_err(|e| anyhow::anyhow!("hotpixel.rec: {e:#}"))?;
            let cleanup = stage
                .cleanup_profile
                .as_ref()
                .map(|path| CleanupProfile::open(path, &rec))
                .transpose()
                .map_err(|e| anyhow::anyhow!("cleanup profile: {e:#}"))?;
            let mut cleanup_cameras = HashMap::new();
            if let Some(profile) = &cleanup {
                for camera in &selected {
                    if let Some(camera_profile) = profile
                        .load_camera(camera)
                        .with_context(|| format!("load cleanup profile for {}", camera.name))?
                    {
                        cleanup_cameras.insert(camera.id, camera_profile);
                    }
                }
            }
            Some(LoadedHotpixelModels {
                rec,
                universal: stage
                    .universal_model
                    .then(UniversalHotpixelProfile::bundled)
                    .transpose()?,
                thermal: stage
                    .glow_correction
                    .then(ThermalProfile::bundled)
                    .transpose()?,
                cleanup_requested: cleanup.is_some(),
                cleanup_cameras,
            })
        }
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
        let (samples_q6, cleanup) = match &hotpixel_models {
            Some(models) => correct_fusion_raw(lri, raw, models, options.threads)?,
            None => (
                extract_raw_plane_threaded(lri, raw, options.threads)
                    .map_err(|e| anyhow::anyhow!("{}: {e:#}", raw.name))?
                    .into_iter()
                    .map(|s| s << 6)
                    .collect(),
                CleanupDiagnostics::default(),
            ),
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
        let focus = state
            .as_ref()
            .map_or_else(ModuleFocusState::default, |state| state.focus.clone());
        let capture_gain = state.as_ref().map_or(1.0, |state| state.gain as f32);
        let exposure_ns = state.as_ref().map_or(0, |state| state.exposure_ns);
        let noise_model = frame_layout
            .frames
            .iter()
            .find(|frame| frame.camera.id == raw.id)
            .and_then(|frame| {
                calibration
                    .sensor_noise_profiles
                    .get(&frame.sensor_type)
                    .or_else(|| {
                        calibration
                            .sensor_noise_profiles
                            .get(&sensor_characterization_type(frame.sensor_type))
                    })
            })
            .and_then(|profile| profile.model_for_gain(raw.analog_gain, raw.digital_gain));
        modules.push(LoadedModule {
            raw: raw.clone(),
            mosaic,
            camera,
            state,
            focus,
            highlight,
            cleanup,
            capture_gain,
            exposure_ns,
            noise_model,
        });
    }
    timings.hotpixel = stage_started.elapsed().as_secs_f32();
    for held_out in &options.cfa_held_out {
        let Some(module) = modules
            .iter()
            .find(|module| module.raw.name.eq_ignore_ascii_case(held_out))
        else {
            bail!("held-out module {held_out} is not present in this capture");
        };
        if module.mosaic.is_mono() {
            bail!(
                "held-out module {} is monochrome; joint-CFA validation requires a Bayer module",
                module.raw.name
            );
        }
    }

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
    let inputs = alignment_inputs(&modules, &luminance);
    let factory_alignments = align_all_modules(&inputs, reference_index, &options.align)?;
    drop(inputs);

    progress(Progress {
        stage: "align",
        detail: "validating capture-specific physical rig".to_owned(),
        fraction: 0.42,
    });
    let rig_inputs = modules
        .iter()
        .map(|module| RigCameraInput {
            name: &module.raw.name,
            calibration: calibration.cameras.get(&module.raw.name),
            state: module.state.as_ref(),
        })
        .collect::<Vec<_>>();
    let mut rig_outcome = refine_capture_rig(
        &rig_inputs,
        reference_index,
        &factory_alignments,
        options.intrinsics_mode,
        &options.rig_refinement,
    );
    drop(rig_inputs);
    let factory_cameras = modules
        .iter()
        .map(|module| module.camera.clone())
        .collect::<Vec<_>>();
    let mut alignments = if rig_outcome.report.accepted {
        for (module, refinement) in modules.iter_mut().zip(&rig_outcome.refinements) {
            module.camera = match (
                calibration.cameras.get(&module.raw.name),
                module.state.as_ref(),
            ) {
                (Some(calibration), Some(state)) => {
                    ResolvedCamera::new(calibration, state, options.intrinsics_mode, refinement)
                        .ok()
                }
                _ => None,
            };
        }
        progress(Progress {
            stage: "align",
            detail: "refining residual warp from accepted physical rig".to_owned(),
            fraction: 0.44,
        });
        let refined_inputs = alignment_inputs(&modules, &luminance);
        let refined_alignments =
            align_all_modules(&refined_inputs, reference_index, &options.align)?;
        if gate_on_image_space_alignment(
            &mut rig_outcome.report,
            &factory_alignments,
            &refined_alignments,
            options
                .rig_refinement
                .min_image_space_correction_improvement,
        ) {
            refined_alignments
        } else {
            for (module, factory_camera) in modules.iter_mut().zip(factory_cameras) {
                module.camera = factory_camera;
            }
            factory_alignments
        }
    } else {
        factory_alignments
    };
    let inputs = alignment_inputs(&modules, &luminance);
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
    let resolution_warps = if options
        .synth
        .resolution_reconstruction
        .uses_resolution_warps()
    {
        progress(Progress {
            stage: "align",
            detail: "resolution-domain local refinement".to_owned(),
            fraction: 0.50,
        });
        inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                if index == reference_index {
                    None
                } else {
                    Some(refine_resolution_warp(
                        inputs[reference_index].luminance,
                        input.luminance,
                        &alignments[index].warp,
                        inputs[reference_index].width,
                        inputs[reference_index].height,
                    ))
                }
            })
            .collect::<Vec<_>>()
    } else {
        vec![None; alignments.len()]
    };
    let contributor_enabled = modules
        .iter()
        .map(|module| {
            !options
                .cfa_held_out
                .iter()
                .any(|camera| camera.eq_ignore_ascii_case(&module.raw.name))
                && (options.cameras.is_empty()
                    || options
                        .cameras
                        .iter()
                        .any(|camera| camera.eq_ignore_ascii_case(&module.raw.name)))
        })
        .collect::<Vec<_>>();
    // In a held-out admission ablation, keep the fitted contributor-side
    // radiometry fixed across camera subsets while still excluding the target
    // camera completely. Outside that protocol this is identical to ordinary
    // contributor admission.
    let radiometry_enabled = modules
        .iter()
        .enumerate()
        .map(|(index, module)| {
            if options.cfa_held_out.is_empty() {
                contributor_enabled[index]
            } else {
                !options
                    .cfa_held_out
                    .iter()
                    .any(|camera| camera.eq_ignore_ascii_case(&module.raw.name))
            }
        })
        .collect::<Vec<_>>();
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
            &contributor_enabled,
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

    // Colour per module: use sparse, reliable aligned overlap to select one
    // common A/F11/D65 blend for the array. Recorded neutral gains remain a
    // soft prior and the unconditional fallback when evidence is weak.
    let reference_calibration = calibration.cameras.get(&reference_name);
    let recorded_wb = awb_gains(&messages).map(|g| [g[0] as f32, g[1] as f32, g[2] as f32]);
    progress(Progress {
        stage: "color",
        detail: "scoring sparse aligned factory-profile blends".to_owned(),
        fraction: 0.53,
    });
    let array_indices = radiometry_enabled
        .iter()
        .enumerate()
        .filter_map(|(index, &enabled)| enabled.then_some(index))
        .collect::<Vec<_>>();
    let array_sources = array_indices
        .iter()
        .map(|&index| ArrayColorSource {
            name: &modules[index].raw.name,
            mosaic: &modules[index].mosaic,
            highlight: &modules[index].highlight,
            alignment: &alignments[index],
            calibration: calibration.cameras.get(&modules[index].raw.name),
        })
        .collect::<Vec<_>>();
    let array_reference_index = array_indices
        .iter()
        .position(|&index| index == reference_index)
        .expect("reference contributor selected");
    let array_selection = select_array_profile(
        &array_sources,
        array_reference_index,
        modules[reference_index].raw.width,
        modules[reference_index].raw.height,
        depth_map.as_ref(),
        recorded_wb,
        options.color_profile,
    );
    let selected_blend = array_selection.blend;
    let selected_mired = array_selection.report.estimated_mired;
    let selection_confidence = array_selection.report.confidence;
    let (module_colors, color_reports): (Vec<_>, Vec<_>) = modules
        .iter()
        .map(|module| {
            module_color_for_selection(
                &module.raw.name,
                calibration.cameras.get(&module.raw.name),
                reference_calibration,
                recorded_wb,
                selected_blend,
                selected_mired,
                selection_confidence,
            )
        })
        .unzip();

    progress(Progress {
        stage: "crosstalk",
        detail: "fitting capture-adaptive factory residuals".to_owned(),
        fraction: 0.54,
    });
    let crosstalk_fits = {
        let active_indices = radiometry_enabled
            .iter()
            .enumerate()
            .filter_map(|(index, &enabled)| enabled.then_some(index))
            .collect::<Vec<_>>();
        let sources = active_indices
            .iter()
            .map(|&index| CrosstalkFitSource {
                mosaic: &modules[index].mosaic,
                highlight: &modules[index].highlight,
                alignment: &alignments[index],
                color: module_colors[index],
                capture_gain: modules[index].capture_gain,
                exposure_ns: modules[index].exposure_ns,
            })
            .collect::<Vec<_>>();
        let dimensions = (
            modules[reference_index].mosaic.width,
            modules[reference_index].mosaic.height,
        );
        let active_reference = active_indices
            .iter()
            .position(|&index| index == reference_index)
            .expect("reference contributor selected");
        let active_fits = fit_adaptive_crosstalk(
            &sources,
            active_reference,
            options.crosstalk,
            dimensions.0,
            dimensions.1,
        );
        let mut active_fits = active_indices.into_iter().zip(active_fits);
        (0..modules.len())
            .map(|index| {
                if radiometry_enabled[index] {
                    let (fit_index, fit) = active_fits.next().expect("one fit per contributor");
                    debug_assert_eq!(fit_index, index);
                    fit
                } else {
                    let source = CrosstalkFitSource {
                        mosaic: &modules[index].mosaic,
                        highlight: &modules[index].highlight,
                        alignment: &alignments[index],
                        color: module_colors[index],
                        capture_gain: modules[index].capture_gain,
                        exposure_ns: modules[index].exposure_ns,
                    };
                    fit_adaptive_crosstalk(
                        std::slice::from_ref(&source),
                        0,
                        CrosstalkMode::Factory,
                        dimensions.0,
                        dimensions.1,
                    )
                    .pop()
                    .expect("one factory fit")
                }
            })
            .collect::<Vec<_>>()
    };
    let crosstalk_reports = modules
        .iter_mut()
        .zip(crosstalk_fits)
        .map(|(module, fit)| {
            module.mosaic.crosstalk = fit.mesh;
            (module.raw.name.clone(), fit.report)
        })
        .collect::<Vec<_>>();

    // Advanced demosaicing is prepared only for geometrically accepted colour
    // modules. This avoids allocating an RGB cache for rejected cameras.
    for (index, (module, alignment)) in modules.iter_mut().zip(&alignments).enumerate() {
        if contributor_enabled[index] && alignment.report.accepted && !module.mosaic.is_mono() {
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
        if contributor_enabled[index]
            && modules[index].raw.name != reference_name
            && alignments[index].report.accepted
        {
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
    let crop = match (options.crop, framed_focal_length_mm) {
        (Some(crop), _) => {
            if ![crop.x, crop.y, crop.width, crop.height]
                .into_iter()
                .all(f32::is_finite)
                || crop.x < 0.0
                || crop.y < 0.0
                || crop.width < 1.0
                || crop.height < 1.0
                || crop.x + crop.width > reference.raw.width as f32
                || crop.y + crop.height > reference.raw.height as f32
            {
                bail!(
                    "explicit crop [{:.1}, {:.1}, {:.1}, {:.1}] is outside the {}x{} reference raster",
                    crop.x,
                    crop.y,
                    crop.width,
                    crop.height,
                    reference.raw.width,
                    reference.raw.height
                );
            }
            crop
        }
        (None, Some(focal)) if options.crop_to_framing && focal > 0 => {
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
        .enumerate()
        .filter(|(index, (module, alignment))| {
            (if options.cfa_held_out.is_empty() {
                contributor_enabled[*index]
            } else {
                radiometry_enabled[*index]
            }) && alignment.report.accepted
                && (options.synth.include_mono || !module.mosaic.is_mono())
                && intersects_crop(alignment, module, &crop)
        })
        .map(|(_, (module, _))| magnification(module))
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
        .zip(&resolution_warps)
        .enumerate()
        .filter(|(index, (((module, alignment), _), resolution_warp))| {
            let held_out = options
                .cfa_held_out
                .iter()
                .any(|camera| camera.eq_ignore_ascii_case(&module.raw.name));
            (contributor_enabled[*index] || held_out)
                && (held_out
                    || alignment.report.accepted
                    || resolution_warp.as_ref().is_some_and(|refined| {
                        refined.report.supported_fraction >= 0.005
                            && refined.report.mean_confidence >= 0.5
                    }))
        })
        .map(
            |(index, (((module, alignment), (color, gain_field)), resolution_warp))| SynthSource {
                camera_id: module.raw.id,
                mosaic: &module.mosaic,
                highlight: &module.highlight,
                noise_model: module.noise_model,
                held_out: options
                    .cfa_held_out
                    .iter()
                    .any(|camera| camera.eq_ignore_ascii_case(&module.raw.name)),
                alignment,
                resolution_warp: resolution_warp.as_ref(),
                fusion_enabled: contributor_enabled[index]
                    && alignment.report.accepted
                    && !options
                        .cfa_held_out
                        .iter()
                        .any(|camera| camera.eq_ignore_ascii_case(&module.raw.name)),
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
            },
        )
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

    let output_megapixels = (synthesis.canvas_width * synthesis.canvas_height) as f32 / 1_000_000.0;
    let total_seconds = timings.load + timings.hotpixel + timings.align + timings.synthesize;
    let resources = FusionResources {
        peak_resident_bytes: peak_resident_bytes(),
        output_megapixels,
        total_seconds_per_megapixel: total_seconds / output_megapixels.max(1.0e-6),
        synthesis_seconds_per_megapixel: timings.synthesize / output_megapixels.max(1.0e-6),
    };
    let report = FusionReport {
        reference: reference_name,
        calibration_modules: calibration.cameras.len(),
        framed_focal_length_mm,
        modules: alignments.iter().map(|a| a.report.clone()).collect(),
        rig_refinement: rig_outcome.report,
        highlights: modules
            .iter()
            .map(|module| (module.raw.name.clone(), module.highlight.report.clone()))
            .collect(),
        cleanup: modules
            .iter()
            .map(|module| (module.raw.name.clone(), module.cleanup.clone()))
            .collect(),
        crosstalk: crosstalk_reports,
        color: color_reports,
        array_color: array_selection.report,
        gains: alignments
            .iter()
            .map(|a| (a.name.clone(), a.gain, a.offset))
            .collect(),
        synthesis,
        seconds: timings,
        resources,
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

fn peak_resident_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kibibytes = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kibibytes.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chiaro::lri::SensorPattern;
    use chiaro::mock::{MockCamera, MockCapture};
    use chiaro_hotpixel_core::{
        cleanup::{BuildCleanupProfileOptions, CleanupProfile, build_cleanup_profile},
        highlight::{HighlightRecovery, HighlightRecoveryReport},
        hotpixel::write_hotpixel_rec,
    };
    use std::{collections::HashSet, fs};

    fn colour_profile(
        illuminant: i32,
        rg: f64,
        bg: f64,
        diagonal: f64,
    ) -> crate::calibration::ColorProfile {
        crate::calibration::ColorProfile {
            illuminant,
            forward_matrix: [
                [diagonal, 0.0, 0.0],
                [0.0, diagonal, 0.0],
                [0.0, 0.0, diagonal],
            ],
            validated_matrix: None,
            color_matrix: None,
            rg_ratio: rg,
            bg_ratio: bg,
            macbeth_data: Vec::new(),
            illuminant_spd: Vec::new(),
            spectral_data: None,
            provenance: crate::calibration::ColorProfileProvenance::Module,
        }
    }

    #[test]
    fn colour_profile_interpolates_from_recorded_neutral_in_mired_space() {
        let calibration = CameraCalibration {
            name: "B1".to_owned(),
            color: vec![
                colour_profile(2, 0.48, 0.67, 1.0),
                colour_profile(6, 0.58, 0.53, 2.0),
                colour_profile(0, 0.75, 0.45, 3.0),
            ],
            ..Default::default()
        };
        let target_rg = (0.48_f64 * 0.58).sqrt() as f32;
        let target_bg = (0.67_f64 * 0.53).sqrt() as f32;
        let selection = illuminant_selection(
            Some(&calibration),
            Some([1.0 / target_rg, 1.0, 1.0 / target_bg]),
        )
        .unwrap();
        assert_eq!((selection.first, selection.second), (2, 6));
        assert!((selection.second_weight - 0.5).abs() < 1e-5);
        let (matrix, rg, bg, validated) = blended_profile(Some(&calibration), selection).unwrap();
        assert!((matrix[0][0] - 1.5).abs() < 1e-5);
        assert!((rg - f64::from(target_rg)).abs() < 1e-5);
        assert!((bg - f64::from(target_bg)).abs() < 1e-5);
        assert!(!validated);
    }

    #[test]
    fn missing_recorded_white_balance_preserves_d65_fallback() {
        let calibration = CameraCalibration {
            name: "B1".to_owned(),
            color: vec![
                colour_profile(0, 0.75, 0.45, 3.0),
                colour_profile(2, 0.48, 0.67, 1.0),
            ],
            ..Default::default()
        };
        let selection = illuminant_selection(Some(&calibration), None).unwrap();
        assert_eq!((selection.first, selection.second), (2, 2));
        let (colour, report) = module_color("B1", Some(&calibration), Some(&calibration), None);
        assert_eq!(colour.forward[0][0], 1.0);
        assert_eq!(report.profile_source, "factory_forward_matrix");
    }

    fn row_biased_camera(temperature: i32) -> MockCamera {
        let mut camera = MockCamera::gradient("A1", 64, 48, SensorPattern::Bggr, 80, 180);
        camera.sensor_temperature_c = Some(temperature);
        for (index, sample) in camera.samples.iter_mut().enumerate() {
            if index / camera.width == 24 {
                *sample = sample.saturating_add(48).min(1023);
            }
        }
        camera
    }

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
                physical_code_range: 65_535.0,
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
                correspondences: Vec::new(),
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

    #[test]
    fn fuse_cleanup_matches_the_shared_hotpixel_frame_pipeline() {
        let temporary = tempfile::tempdir().unwrap();
        let rec_path = temporary.path().join("hotpixel.rec");
        write_hotpixel_rec(
            &rec_path,
            &(0..16)
                .map(|_| (64, 48, vec![0; 64 * 48]))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let rec = HotpixelRec::open(&rec_path).unwrap();
        let training = temporary.path().join("training");
        fs::create_dir(&training).unwrap();
        for (index, temperature) in [20, 30, 40].into_iter().enumerate() {
            let data = MockCapture {
                cameras: vec![row_biased_camera(temperature)],
                reference_camera: Some("A1".to_owned()),
                ..MockCapture::default()
            }
            .encode()
            .unwrap();
            fs::write(training.join(format!("dark-{index}.lri")), data).unwrap();
        }
        let profile_path = temporary.path().join("camera.chiaro-cleanup");
        build_cleanup_profile(
            &BuildCleanupProfileOptions {
                input: training,
                output: profile_path.clone(),
                recursive: false,
                selected_cameras: HashSet::from(["A1".to_owned()]),
                pattern_overrides: HashMap::new(),
                overwrite: false,
                severity_threshold: 16,
                line_neighborhood_radius: 4,
                max_frames_per_camera: None,
            },
            &rec,
            |_| {},
        )
        .unwrap();
        let cleanup = CleanupProfile::open(profile_path, &rec).unwrap();
        let lri = MockCapture {
            cameras: vec![row_biased_camera(30)],
            reference_camera: Some("A1".to_owned()),
            ..MockCapture::default()
        }
        .encode()
        .unwrap();
        let camera = parse_raw_layout(&lri, &HashMap::new()).unwrap().cameras[0].clone();
        let loaded = cleanup.load_camera(&camera).unwrap().unwrap();
        let mut models = LoadedHotpixelModels {
            rec,
            universal: None,
            thermal: None,
            cleanup_requested: true,
            cleanup_cameras: HashMap::from([(camera.id, loaded)]),
        };

        let (fusion_samples, diagnostics) = correct_fusion_raw(&lri, &camera, &models, 1).unwrap();
        let severity = models
            .rec
            .load_rotated_map(camera.id, camera.width, camera.height)
            .unwrap();
        let direct = FramePipeline {
            cleanup: CleanupStage::Profile(models.cleanup_cameras.get(&camera.id).unwrap()),
            threads: 1,
            ..FramePipeline::default()
        }
        .correct_lri(&lri, &camera, &severity)
        .unwrap();
        assert_eq!(fusion_samples, direct.samples_q6);
        assert_eq!(
            diagnostics.correction.mean_absolute_change,
            direct.cleanup.mean_absolute_change
        );
        assert!(diagnostics.profile_supplied);
        assert!(diagnostics.profile_available);

        models.cleanup_cameras.clear();
        let (_, missing) = correct_fusion_raw(&lri, &camera, &models, 1).unwrap();
        assert!(missing.profile_supplied);
        assert!(!missing.profile_available);
        assert!(
            missing
                .correction
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("no entry for this camera"))
        );
    }
}
