//! The fusion pipeline: hot-pixel removal -> alignment -> synthesis.
//!
//! Each stage consumes and produces plain data (`Mosaic`s, `ModuleAlignment`s,
//! a PNG on disk plus a report), so stages can be swapped or inspected
//! independently. Progress is reported through a callback; nothing here knows
//! about a UI.

use std::{
    collections::HashMap,
    fmt::Write as _,
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

use crate::align::{
    AlignInput, AlignOptions, AlignPyramidCache, AlignmentReport, ModuleAlignment, Warp,
    WarpVisibility, align_module_seeded_cached,
};
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
use crate::depth::{
    DenseDepthMap, DepthAlignmentReport, DepthGeometryMode, refine_multiview_depth,
};
use crate::geometry::{CameraRefinement, ResolvedCamera};
use crate::image::{Mosaic, Plane};
use crate::resolution::{ResolutionReconstruction, refine_resolution_warp};
use crate::rig::{
    RigCameraInput, RigRefinementOptions, RigRefinementReport, evaluate_image_space_alignment,
    refine_capture_rig,
};
use crate::synth::{
    ColorPipeline, CropWindow, GainField, ModuleColor, SynthOptions, SynthReport, SynthSource,
    auto_exposure, canvas_scale, photometric_field, photometric_match, synthesize,
};

/// The same effectively-infinite distance used by the ordinary alignment
/// initializer. Keeping it here makes the debug projection an exact picture of
/// the physical seed, before image evidence adds a residual correction.
const DEBUG_FAR_DEPTH: f64 = 1.0e8;

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
    /// Capture-specific bounded physical rig refinement. Debug-report runs
    /// reserve an independently held-out subset; production uses all tracks.
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
    /// Write a human-readable visual trace of every geometry handoff here.
    /// Legacy `<module>_check.png` files remain aliases of the final warp.
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
    /// Capture-specific physical pose/raster/mirror refinement performed
    /// before the downstream residual image-space warp.
    pub rig_refinement: RigRefinementReport,
    /// Debug-mode A/B dense-depth solve. Both branches use the same measured
    /// perpendicular epipolar proposal/output fallback; neither branch is
    /// allowed to use the scene-fitted warp as physical depth evidence or fall
    /// back to WarpSeeded matching.
    pub dense_depth_audit: Option<DenseDepthAuditReport>,
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
pub struct DenseDepthAuditReport {
    pub selected_path: String,
    pub common_anchor: String,
    pub factory: DenseDepthAuditBranchReport,
    pub candidate: DenseDepthAuditBranchReport,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DenseDepthAuditBranchReport {
    pub depth_available: bool,
    pub tested_nodes: usize,
    pub direct_selected_nodes: usize,
    pub neighbour_consistent_nodes: usize,
    pub component_consistent_nodes: usize,
    pub far_supported_nodes: usize,
    pub measured_nodes: usize,
    pub regularized_nodes: usize,
    pub reconstructed_fraction: f32,
    pub accepted_views: usize,
    pub cameras: Vec<DenseDepthAuditCameraReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DenseDepthAuditCameraReport {
    pub camera: String,
    pub geometry_accepted: bool,
    pub depth: Option<DepthAlignmentReport>,
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
            depth_evidence_enabled: true,
            nominal_focal_px: module
                .camera
                .as_ref()
                .map(|camera| camera.focal_px)
                .unwrap_or_else(|| nominal_focal_px(&module.raw.name)),
        })
        .collect()
}

fn alignment_inputs_with_cameras<'a>(
    modules: &'a [LoadedModule],
    luminance: &'a [Plane],
    cameras: &'a [Option<ResolvedCamera>],
) -> Vec<AlignInput<'a>> {
    modules
        .iter()
        .zip(luminance)
        .zip(cameras)
        .map(|((module, luminance), camera)| AlignInput {
            name: &module.raw.name,
            luminance,
            width: module.raw.width,
            height: module.raw.height,
            camera: camera.as_ref(),
            depth_evidence_enabled: true,
            nominal_focal_px: camera
                .as_ref()
                .map(|camera| camera.focal_px)
                .unwrap_or_else(|| nominal_focal_px(&module.raw.name)),
        })
        .collect()
}

fn disable_held_out_depth_evidence(inputs: &mut [AlignInput<'_>], held_out: &[String]) {
    for input in inputs {
        if held_out
            .iter()
            .any(|camera| camera.eq_ignore_ascii_case(input.name))
        {
            input.depth_evidence_enabled = false;
        }
    }
}

fn align_all_modules(
    inputs: &[AlignInput<'_>],
    pyramids: &[AlignPyramidCache],
    reference_index: usize,
    options: &AlignOptions,
    threads: usize,
) -> Result<Vec<ModuleAlignment>> {
    let reference = &inputs[reference_index];
    debug_assert_eq!(inputs.len(), pyramids.len());
    let automatic_workers = std::thread::available_parallelism().map_or(1, usize::from);
    let requested_workers = if threads == 0 {
        automatic_workers
    } else {
        threads
    };
    let worker_count = requested_workers.clamp(1, inputs.len().max(1));
    let inputs_per_worker = inputs.len().div_ceil(worker_count);
    std::thread::scope(|scope| {
        let handles = (0..inputs.len())
            .step_by(inputs_per_worker)
            .map(|first_index| {
                let last_index = (first_index + inputs_per_worker).min(inputs.len());
                scope.spawn(move || {
                    (first_index..last_index)
                        .map(|index| {
                            (
                                index,
                                align_module_seeded_cached(
                                    reference,
                                    &inputs[index],
                                    options,
                                    None,
                                    &pyramids[reference_index],
                                    &pyramids[index],
                                ),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let mut outputs = std::iter::repeat_with(|| None)
            .take(inputs.len())
            .collect::<Vec<_>>();
        for handle in handles {
            for (index, result) in handle.join().expect("alignment worker panicked") {
                outputs[index] = Some(result);
            }
        }
        outputs
            .into_iter()
            .map(|result| result.expect("alignment worker omitted a module"))
            .collect()
    })
}

fn alignment_debug_warps(alignments: &[ModuleAlignment]) -> Vec<Option<Warp>> {
    alignments
        .iter()
        .map(|alignment| Some(alignment.warp.clone()))
        .collect()
}

fn physical_debug_warps(
    cameras: &[Option<ResolvedCamera>],
    reference_index: usize,
    width: usize,
    height: usize,
) -> Vec<Option<Warp>> {
    let Some(reference) = cameras.get(reference_index).and_then(Option::as_ref) else {
        return vec![None; cameras.len()];
    };
    cameras
        .iter()
        .map(|camera| {
            let camera = camera.as_ref()?;
            Some(Warp::from_fn(width, height, 8, |pixel| {
                camera
                    .map_from(reference, pixel, DEBUG_FAR_DEPTH)
                    .filter(|point| camera.contains(*point))
            }))
        })
        .collect()
}

fn candidate_debug_cameras(
    modules: &[LoadedModule],
    calibration: &CalibrationDatabase,
    report: &RigRefinementReport,
    intrinsics_mode: IntrinsicsMode,
) -> Vec<Option<ResolvedCamera>> {
    modules
        .iter()
        .map(|module| {
            let correction = report
                .corrections
                .iter()
                .find(|correction| correction.camera == module.raw.name);
            let refinement = correction.map_or_else(CameraRefinement::default, |correction| {
                let orientation = correction.orientation_offset_degrees;
                CameraRefinement {
                    mirror_angle_offset_degrees: correction.mirror_angle_offset_degrees,
                    orientation_offset_degrees: (orientation != [0.0; 3]).then_some(orientation),
                    center_offset_world: (correction.center_offset_world != [0.0; 3])
                        .then_some(correction.center_offset_world),
                    sensor_offset_px: (correction.sensor_offset_px != [0.0; 2])
                        .then_some(correction.sensor_offset_px),
                }
            });
            ResolvedCamera::new(
                calibration.cameras.get(&module.raw.name)?,
                module.state.as_ref()?,
                intrinsics_mode,
                &refinement,
            )
            .ok()
        })
        .collect()
}

fn diagnostic_error_color(error_px: f32) -> [u16; 3] {
    if !error_px.is_finite() {
        return [0; 3];
    }
    // Log-like landmarks: <=1 px green, 4 px yellow, 16 px orange and
    // >=32 px magenta. These are deliberately fixed across runs so two debug
    // directories can be compared without auto-scaling hiding a regression.
    let stops = [
        (0.0, [0.0, 0.15, 0.0]),
        (1.0, [0.0, 1.0, 0.0]),
        (4.0, [1.0, 1.0, 0.0]),
        (16.0, [1.0, 0.25, 0.0]),
        (32.0, [1.0, 0.0, 1.0]),
    ];
    let error_px = error_px.clamp(0.0, 32.0);
    let (left, right) = stops
        .windows(2)
        .find_map(|pair| (error_px <= pair[1].0).then_some((pair[0], pair[1])))
        .unwrap_or((stops[stops.len() - 2], stops[stops.len() - 1]));
    let t = ((error_px - left.0) / (right.0 - left.0).max(f32::EPSILON)).clamp(0.0, 1.0);
    std::array::from_fn(|channel| {
        ((left.1[channel] + (right.1[channel] - left.1[channel]) * t) * 65_535.0).round() as u16
    })
}

#[inline]
fn diagnostic_point_inside_raster(point: [f32; 2], width: usize, height: usize) -> bool {
    width > 0
        && height > 0
        && point[0].is_finite()
        && point[1].is_finite()
        && point[0] >= 0.0
        && point[1] >= 0.0
        && point[0] <= width.saturating_sub(1) as f32
        && point[1] <= height.saturating_sub(1) as f32
}

fn write_warp_maps(
    directory: &Path,
    camera: &str,
    warp: &Warp,
    measured: Option<&Warp>,
    target_width: usize,
    target_height: usize,
) -> Result<()> {
    let confidence = warp
        .points
        .iter()
        .zip(&warp.confidence)
        .map(|(&point, value)| {
            if diagnostic_point_inside_raster(point, target_width, target_height) {
                (value.clamp(0.0, 1.0) * 65_535.0).round() as u16
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    chiaro_hotpixel_core::png16::write_gray16_native_atomic(
        &directory.join(format!("{camera}-confidence.png")),
        warp.columns,
        warp.rows,
        &confidence,
    )?;

    let mut visibility = Vec::with_capacity(warp.points.len() * 3);
    for (point, state) in warp.points.iter().zip(&warp.visibility) {
        let color = if !diagnostic_point_inside_raster(*point, target_width, target_height) {
            [0, 0, 0]
        } else {
            match state {
                WarpVisibility::Visible => [0, 65_535, 0],
                WarpVisibility::Unknown => [65_535, 49_152, 0],
                WarpVisibility::Occluded => [65_535, 0, 0],
                WarpVisibility::Boundary => [65_535, 0, 65_535],
            }
        };
        visibility.extend(color);
    }
    chiaro_hotpixel_core::png16::write_rgb16_native_atomic(
        &directory.join(format!("{camera}-visibility.png")),
        warp.columns,
        warp.rows,
        &visibility,
    )?;

    if let Some(measured) = measured {
        let mut disagreement = Vec::with_capacity(warp.points.len() * 3);
        for row in 0..warp.rows {
            for column in 0..warp.columns {
                let pixel = [(column * warp.step) as f32, (row * warp.step) as f32];
                let color = match (
                    warp.map(pixel[0], pixel[1]),
                    measured.map(pixel[0], pixel[1]),
                ) {
                    (Some(first), Some(second))
                        if diagnostic_point_inside_raster(first, target_width, target_height)
                            && diagnostic_point_inside_raster(
                                second,
                                target_width,
                                target_height,
                            ) =>
                    {
                        diagnostic_error_color((first[0] - second[0]).hypot(first[1] - second[1]))
                    }
                    _ => [0; 3],
                };
                disagreement.extend(color);
            }
        }
        chiaro_hotpixel_core::png16::write_rgb16_native_atomic(
            &directory.join(format!("{camera}-vs-measured-error.png")),
            warp.columns,
            warp.rows,
            &disagreement,
        )?;
    }
    Ok(())
}

fn write_debug_warp_stage(
    debug_dir: &Path,
    stage: &str,
    modules: &[LoadedModule],
    luminance: &[Plane],
    reference_index: usize,
    warps: &[Option<Warp>],
    measured: Option<&[Option<Warp>]>,
    write_maps: bool,
) -> Result<()> {
    let directory = debug_dir.join(stage);
    fs::create_dir_all(&directory)
        .with_context(|| format!("create debug stage {}", directory.display()))?;
    for index in 0..modules.len() {
        if index == reference_index {
            continue;
        }
        let Some(warp) = warps.get(index).and_then(Option::as_ref) else {
            continue;
        };
        let camera = &modules[index].raw.name;
        let (samples, width, height) = crate::align::debug_checkerboard(
            &luminance[reference_index],
            &luminance[index],
            warp,
            64,
        );
        chiaro_hotpixel_core::png16::write_gray16_native_atomic(
            &directory.join(format!("{camera}-checkerboard.png")),
            width,
            height,
            &samples,
        )?;
        if write_maps {
            write_warp_maps(
                &directory,
                camera,
                warp,
                measured
                    .and_then(|warps| warps.get(index))
                    .and_then(Option::as_ref),
                modules[index].raw.width,
                modules[index].raw.height,
            )?;
        }
    }
    Ok(())
}

fn dense_depth_audit_branch_report(
    depth_map: Option<&DenseDepthMap>,
    alignments: &[ModuleAlignment],
    reference_index: usize,
) -> DenseDepthAuditBranchReport {
    let shared = alignments
        .iter()
        .find_map(|alignment| alignment.report.depth.as_ref());
    DenseDepthAuditBranchReport {
        depth_available: depth_map.is_some(),
        tested_nodes: shared.map_or(0, |depth| depth.tested_nodes),
        direct_selected_nodes: shared.map_or(0, |depth| depth.direct_selected_nodes),
        neighbour_consistent_nodes: shared.map_or(0, |depth| depth.neighbour_consistent_nodes),
        component_consistent_nodes: shared.map_or(0, |depth| depth.component_consistent_nodes),
        far_supported_nodes: shared.map_or(0, |depth| depth.far_supported_nodes),
        measured_nodes: shared.map_or(0, |depth| depth.measured_nodes),
        regularized_nodes: shared.map_or(0, |depth| depth.regularized_nodes),
        reconstructed_fraction: shared.map_or(0.0, |depth| depth.reconstructed_fraction),
        accepted_views: alignments
            .iter()
            .enumerate()
            .filter(|(index, alignment)| *index != reference_index && alignment.geometry_accepted())
            .count(),
        cameras: alignments
            .iter()
            .map(|alignment| DenseDepthAuditCameraReport {
                camera: alignment.name.clone(),
                geometry_accepted: alignment.geometry_accepted(),
                depth: alignment.report.depth.clone(),
            })
            .collect(),
    }
}

fn run_dense_depth_audit_branch(
    debug_dir: &Path,
    stage: &str,
    modules: &[LoadedModule],
    luminance: &[Plane],
    cameras: &[Option<ResolvedCamera>],
    reference_index: usize,
    measured_seed: &[ModuleAlignment],
    measured_warps: &[Option<Warp>],
    held_out: &[String],
    options: &crate::depth::DepthOptions,
) -> Result<DenseDepthAuditBranchReport> {
    let mut alignments = measured_seed.to_vec();
    let mut inputs = alignment_inputs_with_cameras(modules, luminance, cameras);
    disable_held_out_depth_evidence(&mut inputs, held_out);
    let depth_map = refine_multiview_depth(
        &inputs,
        reference_index,
        &mut alignments,
        options,
        DepthGeometryMode::PhysicalRig,
    );
    write_dense_depth_audit_outputs(
        debug_dir,
        stage,
        modules,
        luminance,
        reference_index,
        &alignments,
        depth_map.as_ref(),
        measured_warps,
    )
}

fn write_dense_depth_audit_outputs(
    debug_dir: &Path,
    stage: &str,
    modules: &[LoadedModule],
    luminance: &[Plane],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    depth_map: Option<&DenseDepthMap>,
    measured_warps: &[Option<Warp>],
) -> Result<DenseDepthAuditBranchReport> {
    let warps = alignment_debug_warps(&alignments);
    write_debug_warp_stage(
        debug_dir,
        stage,
        modules,
        luminance,
        reference_index,
        &warps,
        Some(measured_warps),
        true,
    )?;
    if let Some(depth_map) = depth_map {
        let directory = debug_dir.join(stage);
        depth_map.write_diagnostics(
            &directory.join("depth-inverse.png"),
            &directory.join("depth-provenance.png"),
        )?;
        depth_map.write_visualization(&directory.join("depth-visualization.png"))?;
    }
    Ok(dense_depth_audit_branch_report(
        depth_map,
        alignments,
        reference_index,
    ))
}

fn write_rig_residual_field_svg(debug_dir: &Path, report: &RigRefinementReport) -> Result<()> {
    let cameras = report
        .residual_field
        .iter()
        .map(|field| field.camera.as_str())
        .fold(Vec::<&str>::new(), |mut cameras, camera| {
            if !cameras.contains(&camera) {
                cameras.push(camera);
            }
            cameras
        });
    if cameras.is_empty() {
        return Ok(());
    }
    const PANEL_WIDTH: usize = 360;
    const PANEL_HEIGHT: usize = 275;
    const COLUMNS: usize = 3;
    let rows = cameras.len().div_ceil(COLUMNS);
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        PANEL_WIDTH * COLUMNS,
        PANEL_HEIGHT * rows + 45,
        PANEL_WIDTH * COLUMNS,
        PANEL_HEIGHT * rows + 45,
    );
    svg.push_str(
        r##"<rect width="100%" height="100%" fill="#11151b"/><style>text{font-family:monospace;fill:#e8edf2}.grid{stroke:#34404c;stroke-width:1}.before{stroke:#ff5d5d;stroke-width:2}.after{stroke:#46e0e0;stroke-width:2}</style>"##,
    );
    svg.push_str(
        r#"<text x="18" y="27" font-size="15">Held-out reprojection residuals: red=factory, cyan=candidate; arrow scale 5 SVG px per sensor px</text>"#,
    );
    for (camera_index, camera) in cameras.iter().enumerate() {
        let origin_x = (camera_index % COLUMNS) * PANEL_WIDTH + 25;
        let origin_y = (camera_index / COLUMNS) * PANEL_HEIGHT + 70;
        let plot_width = 300.0;
        let plot_height = 210.0;
        let _ = writeln!(
            svg,
            r##"<g><text x="{}" y="{}" font-size="18">{}</text><rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#6b7885"/>"##,
            origin_x,
            origin_y - 12,
            camera,
            origin_x,
            origin_y,
            plot_width,
            plot_height,
        );
        for column in 1..4 {
            let x = origin_x as f64 + plot_width * column as f64 / 4.0;
            let _ = writeln!(
                svg,
                r#"<line class="grid" x1="{x}" y1="{}" x2="{x}" y2="{}"/>"#,
                origin_y,
                origin_y as f64 + plot_height,
            );
        }
        for row in 1..3 {
            let y = origin_y as f64 + plot_height * row as f64 / 3.0;
            let _ = writeln!(
                svg,
                r#"<line class="grid" x1="{}" y1="{y}" x2="{}" y2="{y}"/>"#,
                origin_x,
                origin_x as f64 + plot_width,
            );
        }
        for field in report
            .residual_field
            .iter()
            .filter(|field| field.camera == **camera)
        {
            let x = origin_x as f64 + plot_width * (field.cell[0] as f64 + 0.5) / 4.0;
            let y = origin_y as f64 + plot_height * (field.cell[1] as f64 + 0.5) / 3.0;
            let arrow = |residual: [f64; 2]| {
                let length = residual[0].hypot(residual[1]);
                let scale = if length > 12.0 { 60.0 / length } else { 5.0 };
                [x + residual[0] * scale, y + residual[1] * scale]
            };
            let before = arrow(field.mean_before);
            let after = arrow(field.mean_after);
            let _ = writeln!(
                svg,
                r##"<circle cx="{x}" cy="{y}" r="2" fill="#fff"/><line class="before" x1="{x}" y1="{y}" x2="{}" y2="{}"/><line class="after" x1="{x}" y1="{y}" x2="{}" y2="{}"/><text x="{}" y="{}" font-size="9">n={}</text>"##,
                before[0],
                before[1],
                after[0],
                after[1],
                x + 4.0,
                y - 4.0,
                field.samples,
            );
        }
        svg.push_str("</g>");
    }
    svg.push_str("</svg>");
    fs::write(debug_dir.join("rig-held-out-residual-field.svg"), svg)
        .with_context(|| format!("write debug residual field in {}", debug_dir.display()))
}

fn write_rig_held_out_observations(debug_dir: &Path, report: &RigRefinementReport) -> Result<()> {
    let observations = &report.held_out_observations;
    if observations.is_empty() {
        return Ok(());
    }
    let cameras = observations
        .iter()
        .map(|observation| observation.camera.as_str())
        .fold(Vec::<&str>::new(), |mut cameras, camera| {
            if !cameras.contains(&camera) {
                cameras.push(camera);
            }
            cameras
        });
    const PANEL_WIDTH: usize = 420;
    const PANEL_HEIGHT: usize = 320;
    const COLUMNS: usize = 3;
    let rows = cameras.len().div_ceil(COLUMNS);
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        PANEL_WIDTH * COLUMNS,
        PANEL_HEIGHT * rows + 80,
        PANEL_WIDTH * COLUMNS,
        PANEL_HEIGHT * rows + 80,
    );
    svg.push_str(
        r##"<rect width="100%" height="100%" fill="#11151b"/><style>text{font-family:monospace;fill:#e8edf2}.grid{stroke:#34404c;stroke-width:1}.before{stroke:#ff5d5d;stroke-width:1.4;opacity:.72}.after{stroke:#46e0e0;stroke-width:1.4;opacity:.82}.factory-tail{fill:none;stroke:#ff8c42;stroke-width:1.5;stroke-dasharray:2 2}.candidate-tail{fill:none;stroke:#ffd166;stroke-width:2.5}.tail-label{fill:#ffd166;font-size:9px}</style>"##,
    );
    let _ = writeln!(
        svg,
        r#"<text x="18" y="25" font-size="15">Individual held-out observations: red=factory, cyan=candidate, yellow ring=candidate p95 tail (&gt;={:.3} reference px)</text>"#,
        report
            .held_out_residuals_after
            .reference_equivalent_pixels
            .p95,
    );
    svg.push_str(
        r#"<text x="18" y="47" font-size="12">Arrow scale 3 SVG px per sensor px; arrows above 15 sensor px are capped. Hover a point for exact values. Orange dashed ring=factory p95 tail.</text>"#,
    );
    for (camera_index, camera) in cameras.iter().enumerate() {
        let origin_x = (camera_index % COLUMNS) * PANEL_WIDTH + 30;
        let origin_y = (camera_index / COLUMNS) * PANEL_HEIGHT + 100;
        let plot_width = 350.0;
        let plot_height = 245.0;
        let camera_observations = observations
            .iter()
            .filter(|observation| observation.camera == **camera)
            .collect::<Vec<_>>();
        let [sensor_width, sensor_height] = camera_observations[0].sensor_size;
        let _ = writeln!(
            svg,
            r##"<g><text x="{}" y="{}" font-size="18">{} — n={}</text><rect x="{}" y="{}" width="{}" height="{}" fill="#080a0d" stroke="#6b7885"/>"##,
            origin_x,
            origin_y - 12,
            camera,
            camera_observations.len(),
            origin_x,
            origin_y,
            plot_width,
            plot_height,
        );
        for division in 1..4 {
            let x = origin_x as f64 + plot_width * division as f64 / 4.0;
            let y = origin_y as f64 + plot_height * division as f64 / 4.0;
            let _ = writeln!(
                svg,
                r#"<line class="grid" x1="{x}" y1="{}" x2="{x}" y2="{}"/><line class="grid" x1="{}" y1="{y}" x2="{}" y2="{y}"/>"#,
                origin_y,
                origin_y as f64 + plot_height,
                origin_x,
                origin_x as f64 + plot_width,
            );
        }
        for observation in camera_observations {
            let x =
                origin_x as f64 + plot_width * observation.pixel[0] / sensor_width.max(1) as f64;
            let y =
                origin_y as f64 + plot_height * observation.pixel[1] / sensor_height.max(1) as f64;
            let arrow = |residual: [f64; 2]| {
                let length = residual[0].hypot(residual[1]);
                let scale = if length > 15.0 { 45.0 / length } else { 3.0 };
                [x + residual[0] * scale, y + residual[1] * scale]
            };
            let before = arrow(observation.factory_residual);
            let after = arrow(observation.candidate_residual);
            let _ = writeln!(
                svg,
                r##"<g><title>{} ({:.1},{:.1}) factory {:.3} ref px/{:.4} deg; candidate {:.3} ref px/{:.4} deg</title><line class="before" x1="{x}" y1="{y}" x2="{}" y2="{}"/><line class="after" x1="{x}" y1="{y}" x2="{}" y2="{}"/><circle cx="{x}" cy="{y}" r="1.7" fill="#f4f7fa"/>"##,
                observation.camera,
                observation.pixel[0],
                observation.pixel[1],
                observation.factory_reference_pixels,
                observation.factory_angular_degrees,
                observation.candidate_reference_pixels,
                observation.candidate_angular_degrees,
                before[0],
                before[1],
                after[0],
                after[1],
            );
            if observation.factory_p95_tail {
                let _ = write!(
                    svg,
                    r#"<circle class="factory-tail" cx="{x}" cy="{y}" r="5"/>"#,
                );
            }
            if observation.candidate_p95_tail {
                let _ = write!(
                    svg,
                    r#"<circle class="candidate-tail" cx="{x}" cy="{y}" r="7"/><text class="tail-label" x="{}" y="{}">{:.1}</text>"#,
                    x + 8.0,
                    y - 7.0,
                    observation.candidate_reference_pixels,
                );
            }
            svg.push_str("</g>");
        }
        svg.push_str("</g>");
    }
    svg.push_str("</svg>");
    fs::write(debug_dir.join("rig-held-out-observations.svg"), svg)
        .with_context(|| format!("write held-out observation map in {}", debug_dir.display()))?;

    let mut csv = String::from(
        "camera,x,y,factory_dx,factory_dy,candidate_dx,candidate_dy,factory_sensor_px,candidate_sensor_px,factory_reference_px,candidate_reference_px,factory_angular_deg,candidate_angular_deg,factory_p95_tail,candidate_p95_tail\n",
    );
    for observation in observations {
        let _ = writeln!(
            csv,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.9},{:.9},{},{}",
            observation.camera,
            observation.pixel[0],
            observation.pixel[1],
            observation.factory_residual[0],
            observation.factory_residual[1],
            observation.candidate_residual[0],
            observation.candidate_residual[1],
            observation.factory_sensor_pixels,
            observation.candidate_sensor_pixels,
            observation.factory_reference_pixels,
            observation.candidate_reference_pixels,
            observation.factory_angular_degrees,
            observation.candidate_angular_degrees,
            observation.factory_p95_tail,
            observation.candidate_p95_tail,
        );
    }
    fs::write(debug_dir.join("rig-held-out-observations.csv"), csv).with_context(|| {
        format!(
            "write held-out observation table in {}",
            debug_dir.display()
        )
    })
}

fn write_pipeline_trace(
    debug_dir: &Path,
    reference: &str,
    rig: &RigRefinementReport,
    geometry_mode: DepthGeometryMode,
    dense_depth_audit: Option<&DenseDepthAuditReport>,
    alignments: &[ModuleAlignment],
    synthesis: &SynthReport,
) -> Result<()> {
    let mut trace = String::new();
    let _ = writeln!(trace, "Chiaro visual pipeline trace");
    let _ = writeln!(trace, "============================");
    let _ = writeln!(trace, "Reference camera: {reference}");
    let _ = writeln!(trace, "Dense geometry mode: {geometry_mode:?}");
    let _ = writeln!(trace, "Rig candidate accepted: {}", rig.accepted);
    if let Some(reason) = &rig.fallback_reason {
        let _ = writeln!(trace, "Rig fallback reason: {reason}");
    }
    if rig.validation_evaluated {
        let _ = writeln!(
            trace,
            "Rig held-out RMS: {:.3} -> {:.3} px ({:+.2}%)",
            rig.held_out_rms_before,
            rig.held_out_rms_after,
            rig.held_out_relative_improvement * 100.0,
        );
        let _ = writeln!(
            trace,
            "Rig tracks: {} fit, {} held out; physical matcher used: {}",
            rig.fit_tracks, rig.validation_tracks, rig.physical_match_used,
        );
        let _ = writeln!(
            trace,
            "Rig optimization: {} coordinate sweeps, {} robust membership passes",
            rig.optimizer_iterations, rig.membership_iterations,
        );
        let _ = writeln!(
            trace,
            "Rig bootstrap: factory-first, then stage-02 perpendicular epipolar proposal +/-{:.1} target px, followed by <= {:.1} reference px proposal-normalized physical consistency; {} observations/{} tracks rejected before optimization",
            rig.physical_match_residual_radius_px,
            rig.physical_match_pre_solve_reprojection_px,
            rig.rejected_inconsistent_observations,
            rig.rejected_inconsistent_tracks,
        );
        if rig.physical_match_candidates > 0 {
            let _ = writeln!(
                trace,
                "Rig matcher yield: {} candidates -> {} physical tracks; rejected: {} no supported depth, {} ambiguous depth, {} insufficient native-resolution views, {} failed physical consistency",
                rig.physical_match_candidates,
                rig.physical_match_tracks,
                rig.physical_match_rejected_no_supported_depth,
                rig.physical_match_rejected_ambiguous_depth,
                rig.physical_match_rejected_insufficient_views,
                rig.rejected_inconsistent_tracks,
            );
            let _ = writeln!(
                trace,
                "Rig depth hierarchy: {:.1} evaluated hypotheses/reference candidate, up to {} refinements, {:.2} target px worst final neighbouring-depth motion ({:.1} px limit)",
                rig.physical_match_depth_hypotheses as f64
                    / rig.physical_match_candidates.max(1) as f64,
                rig.physical_match_max_depth_refinement_levels,
                rig.physical_match_observed_max_projected_step_px,
                rig.physical_match_max_projected_step_px,
            );
            let _ = writeln!(
                trace,
                "Rig depth levels (level:candidates): {}",
                rig.physical_match_depth_refinement_histogram
                    .iter()
                    .enumerate()
                    .filter(|(_, count)| **count > 0)
                    .map(|(level, count)| format!("{level}:{count}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    } else {
        let _ = writeln!(
            trace,
            "Rig held-out RMS: not evaluated (production uses all usable tracks)"
        );
        if rig.enabled {
            let _ = writeln!(
                trace,
                "Rig tracks: {} all-track fit, 0 held out; accepted: {}; physical matcher used: {}",
                rig.fit_tracks, rig.accepted, rig.physical_match_used,
            );
            if rig.physical_match_candidates > 0 {
                let _ = writeln!(
                    trace,
                    "Rig matcher yield: {} candidates -> {} physical tracks; rejected: {} no supported depth, {} ambiguous depth, {} insufficient native-resolution views, {} failed physical consistency",
                    rig.physical_match_candidates,
                    rig.physical_match_tracks,
                    rig.physical_match_rejected_no_supported_depth,
                    rig.physical_match_rejected_ambiguous_depth,
                    rig.physical_match_rejected_insufficient_views,
                    rig.rejected_inconsistent_tracks,
                );
                let _ = writeln!(
                    trace,
                    "Rig depth hierarchy: {:.1} evaluated hypotheses/reference candidate, up to {} refinements, {:.2} target px worst final neighbouring-depth motion ({:.1} px limit)",
                    rig.physical_match_depth_hypotheses as f64
                        / rig.physical_match_candidates.max(1) as f64,
                    rig.physical_match_max_depth_refinement_levels,
                    rig.physical_match_observed_max_projected_step_px,
                    rig.physical_match_max_projected_step_px,
                );
                let _ = writeln!(
                    trace,
                    "Rig depth levels (level:candidates): {}",
                    rig.physical_match_depth_refinement_histogram
                        .iter()
                        .enumerate()
                        .filter(|(_, count)| **count > 0)
                        .map(|(level, count)| format!("{level}:{count}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        } else {
            let _ = writeln!(trace, "Rig tracks/optimization: not run (disabled)");
        }
    }
    if let Some(passed) = rig.image_space_validation_passed {
        let _ = writeln!(
            trace,
            "Downstream residual diagnostic: {} across {} optimized cameras; median correction {:.3} -> {:.3} px ({:+.2}%). This diagnostic does not select factory versus candidate geometry.",
            if passed {
                "passed reference threshold"
            } else {
                "warning"
            },
            rig.image_space_evaluated_cameras,
            rig.image_space_median_correction_before_px,
            rig.image_space_median_correction_after_px,
            rig.image_space_relative_improvement * 100.0,
        );
    }
    if let Some(warning) = &rig.image_space_warning {
        let _ = writeln!(trace, "Downstream residual warning: {warning}");
    }
    if !rig.corrections.is_empty() {
        trace.push_str("\nRig candidate corrections\n-------------------------\n");
        trace.push_str(
            "camera  optimized  rotation(x,y,z) deg          centre(x,y,z)       sensor(x,y) px  mirror deg  bound\n",
        );
        for correction in &rig.corrections {
            let _ = writeln!(
                trace,
                "{:<7} {:<9} ({:+7.4},{:+7.4},{:+7.4})  ({:+6.3},{:+6.3},{:+6.3})  ({:+6.2},{:+6.2})   {:+7.4}    {}",
                correction.camera,
                correction.optimized,
                correction.orientation_offset_degrees[0],
                correction.orientation_offset_degrees[1],
                correction.orientation_offset_degrees[2],
                correction.center_offset_world[0],
                correction.center_offset_world[1],
                correction.center_offset_world[2],
                correction.sensor_offset_px[0],
                correction.sensor_offset_px[1],
                correction.mirror_angle_offset_degrees,
                correction.reached_bound,
            );
        }
    }
    trace.push_str("\nHow to read the stage folders\n-----------------------------\n");
    trace.push_str(
        "01-factory-physical: factory camera projection at effectively infinite depth, before image evidence.\n",
    );
    trace.push_str(
        "02-measured-residual: correlation-refined image alignment; this is the measured baseline.\n",
    );
    trace.push_str(
        "03-rig-candidate-physical: proposed pose/raster/mirror correction without a residual image warp.\n",
    );
    trace.push_str(
        "04-active-pre-depth: the accepted/fallback image alignment immediately before dense depth.\n",
    );
    trace.push_str(
        "05-final-depth: the warp actually sent downstream after PhysicalRig or WarpSeeded depth.\n",
    );
    if dense_depth_audit.is_some() {
        trace.push_str(
            "05a-factory-depth-audit: diagnostic factory PhysicalRig dense solve with no WarpSeeded fallback.\n",
        );
        trace.push_str(
            "05b-candidate-depth-audit: diagnostic proposed-rig PhysicalRig dense solve, run even when the rig candidate is rejected.\n",
        );
    }
    trace.push_str(
        "Checkerboards should have continuous edges across tile boundaries. Factory/candidate physical checkerboards use an effectively infinite scene plane; finite-scene parallax is therefore expected until stage 05. In *-vs-measured-error.png, the displacement between that stage and the measured capture warp is green <=1 px, yellow ~4 px, orange ~16 px, magenta >=32 px, black=no shared in-sensor domain. It is not the held-out rig RMS and, before depth, includes scene parallax.\n",
    );
    trace.push_str(
        "Visibility colors: green=directly visible, amber=in-sensor but visibility unknown, red=occluded, magenta=depth boundary, black=outside the target sensor/undefined. In stages 01 and 03, confidence is only binary in-sensor projection validity (white=in sensor); it is not image evidence or calibration confidence.\n",
    );
    trace.push_str(
        "rig-held-out-residual-field.svg shows spatial residual direction before/after the candidate; coherent position-dependent arrows suggest a missing camera-model degree of freedom.\n",
    );
    if !rig.held_out_observations.is_empty() {
        let factory_tail = rig
            .held_out_observations
            .iter()
            .filter(|observation| observation.factory_p95_tail)
            .count();
        let candidate_tail = rig
            .held_out_observations
            .iter()
            .filter(|observation| observation.candidate_p95_tail)
            .count();
        let _ = writeln!(
            trace,
            "rig-held-out-observations.svg/csv contain all {} paired held-out observations; {factory_tail} factory and {candidate_tail} candidate samples are at or above their respective global p95 thresholds.",
            rig.held_out_observations.len(),
        );
    }

    if let Some(depth) = alignments
        .iter()
        .find_map(|alignment| alignment.report.depth.as_ref())
    {
        let _ = writeln!(
            trace,
            "\nShared reference-space depth\n----------------------------\n{} / {} tested nodes directly measured ({:.2}%); {} regularized.\nDense acceptance funnel: {} direct finite selections -> {} neighbour-consistent -> {} component-consistent; {} nodes directly support the stage-2 output fallback.\nThis one field is projected into every camera, so it is intentionally not a per-camera percentage.",
            depth.measured_nodes,
            depth.tested_nodes,
            depth.reconstructed_fraction * 100.0,
            depth.regularized_nodes,
            depth.direct_selected_nodes,
            depth.neighbour_consistent_nodes,
            depth.component_consistent_nodes,
            depth.far_supported_nodes,
        );
    }

    if let Some(audit) = dense_depth_audit {
        trace.push_str("\nDense-depth A/B audit\n---------------------\n");
        let _ = writeln!(trace, "Selected production path: {}", audit.selected_path);
        let _ = writeln!(trace, "Common comparison anchor: {}", audit.common_anchor);
        for (name, branch) in [
            ("factory PhysicalRig", &audit.factory),
            ("candidate PhysicalRig", &audit.candidate),
        ] {
            let _ = writeln!(
                trace,
                "{name}: available={}, {}/{} finite measured ({:.2}%), {} regularized, {} accepted target views; funnel {} selected -> {} neighbour -> {} component, {} supported fallback",
                branch.depth_available,
                branch.measured_nodes,
                branch.tested_nodes,
                branch.reconstructed_fraction * 100.0,
                branch.regularized_nodes,
                branch.accepted_views,
                branch.direct_selected_nodes,
                branch.neighbour_consistent_nodes,
                branch.component_consistent_nodes,
                branch.far_supported_nodes,
            );
        }
        let measured_delta =
            audit.candidate.measured_nodes as i64 - audit.factory.measured_nodes as i64;
        let _ = writeln!(
            trace,
            "Candidate - factory: {measured_delta:+} measured nodes, {:+.2} percentage points reconstructed",
            (audit.candidate.reconstructed_fraction - audit.factory.reconstructed_fraction) * 100.0,
        );
        trace.push_str(
            "These branches are diagnostics only: both use stage 02 only as the same perpendicular epipolar proposal/output fallback, both force PhysicalRig without WarpSeeded matching, and acceptance still controls the production path.\n",
        );
        trace.push_str(
            "camera  factory-defined  candidate-defined  factory-finite  candidate-finite  factory-far  candidate-far  accepted(F/C)\n",
        );
        for factory_camera in &audit.factory.cameras {
            let candidate_camera = audit
                .candidate
                .cameras
                .iter()
                .find(|candidate| candidate.camera == factory_camera.camera);
            let factory_depth = factory_camera.depth.as_ref();
            let candidate_depth = candidate_camera.and_then(|camera| camera.depth.as_ref());
            let _ = writeln!(
                trace,
                "{:<7} {:>7.2}%          {:>7.2}%          {:>7}          {:>7}       {:>7}        {:>7}       {}/{}",
                factory_camera.camera,
                factory_depth.map_or(0.0, |depth| depth.defined_fraction * 100.0),
                candidate_depth.map_or(0.0, |depth| depth.defined_fraction * 100.0),
                factory_depth.map_or(0, |depth| depth.refined_nodes),
                candidate_depth.map_or(0, |depth| depth.refined_nodes),
                factory_depth.map_or(0, |depth| depth.fallback_nodes),
                candidate_depth.map_or(0, |depth| depth.fallback_nodes),
                factory_camera.geometry_accepted,
                candidate_camera.is_some_and(|camera| camera.geometry_accepted),
            );
        }
    }

    trace.push_str("\nPer-camera geometry\n-------------------\n");
    let _ = writeln!(
        trace,
        "camera  legacy  physical  overlap  warp-defined  direct-support  correction(x,y) px  view-refined  far-fallback  occluded"
    );
    for alignment in alignments {
        let depth = alignment.report.depth.as_ref();
        let defined = depth
            .map(|depth| format!("{:>7.2}%", depth.defined_fraction * 100.0))
            .unwrap_or_else(|| "    n/a".to_owned());
        let direct = depth
            .map(|depth| format!("{:>7.2}%", depth.directly_supported_fraction * 100.0))
            .unwrap_or_else(|| "    n/a".to_owned());
        let _ = writeln!(
            trace,
            "{:<7} {:<7} {:<8} {:>6.2}%    {}        {}       ({:>7.2},{:>7.2})  {:>7.2}%      {:>7}    {:>7}",
            alignment.name,
            if alignment.report.accepted {
                "accept"
            } else {
                "reject"
            },
            if alignment.geometry_accepted() {
                "accept"
            } else {
                "reject"
            },
            alignment.report.coverage * 100.0,
            defined,
            direct,
            alignment.report.correction_median_px[0],
            alignment.report.correction_median_px[1],
            depth.map_or(0.0, |depth| depth.refined_fraction * 100.0),
            depth.map_or(0, |depth| depth.fallback_nodes),
            depth.map_or(0, |depth| depth.occluded_nodes),
        );
    }

    trace.push_str("\nSynthesis\n---------\n");
    let _ = writeln!(
        trace,
        "Canvas: {}x{}, {:.3} canvas pixels/reference pixel, {:.2}% covered",
        synthesis.canvas_width,
        synthesis.canvas_height,
        synthesis.scale,
        synthesis.covered * 100.0,
    );
    let _ = writeln!(
        trace,
        "Non-reference edge/detail rejection: {:.2}%",
        synthesis.edge_rejected_fraction * 100.0,
    );
    let resolution = &synthesis.resolution_reconstruction;
    if resolution.mode == ResolutionReconstruction::Resample {
        trace.push_str("Resolution reconstruction: disabled by resample mode.\n");
    } else {
        let _ = writeln!(
            trace,
            "Resolution reconstruction: {:.2}% candidates, {:.4}% applied, mean confidence {:.3}",
            resolution.candidate_fraction * 100.0,
            resolution.reconstructed_fraction * 100.0,
            resolution.mean_confidence,
        );
    }
    if let Some(joint) = &synthesis.joint_cfa {
        let _ = writeln!(
            trace,
            "Joint CFA: {} / {} candidates reconstructed ({:.4}%), mean {:.2} cameras",
            joint.reconstructed_pixels,
            joint.attempted_pixels,
            joint.reconstructed_fraction * 100.0,
            joint.mean_cameras_per_pixel,
        );
    }
    trace.push_str("\nSource ownership and rejection\n------------------------------\n");
    trace.push_str(
        "Weight share is the actual normalized blend contribution. Owner is only the largest single weight at each pixel; it does not mean other cameras were excluded.\n",
    );
    let _ = writeln!(
        trace,
        "camera  enabled  mag    luma-share  color-share  luma-owner  color-owner  visibility-reject  chroma-reject  resolution-use"
    );
    for source in &synthesis.source_contributions {
        let _ = writeln!(
            trace,
            "{:<7} {:<7} {:>4.2}x  {:>7.2}%     {:>7.2}%     {:>7.2}%     {:>7.2}%      {:>7.2}%            {:>7.2}%       {:>7.3}%",
            source.camera,
            source.fusion_enabled,
            source.magnification,
            source.luminance_weight_fraction * 100.0,
            source.color_weight_fraction * 100.0,
            source.luminance_owner_fraction * 100.0,
            source.color_owner_fraction * 100.0,
            source.visibility_suppressed_fraction * 100.0,
            source.chroma_suppressed_fraction * 100.0,
            source.resolution_contributor_fraction * 100.0,
        );
    }
    trace.push_str("\nFast warning signs\n------------------\n");
    trace.push_str(
        "- Physical checkerboard worse than measured checkerboard: the physical model is not yet an adequate correspondence model.\n",
    );
    trace.push_str(
        "- Mostly amber/black final visibility: dense depth has little directly verified support.\n",
    );
    trace.push_str("- High edge rejection: downstream safety gates are discarding much of the aligned evidence.\n");
    if resolution.mode != ResolutionReconstruction::Resample {
        trace.push_str(
            "- Near-zero resolution use: reconstruction could not establish sufficiently distinct, locally registered sampling phases.\n",
        );
    }
    trace.push_str(
        "- Large canvas scale beyond the finest optical magnification: apparent softness is expected because output pixels exceed measured sampling density.\n",
    );
    fs::write(debug_dir.join("PIPELINE_TRACE.txt"), trace)
        .with_context(|| format!("write pipeline trace in {}", debug_dir.display()))?;

    let mut stages = vec![
        ("01-factory-physical", "Factory physical (infinity plane)"),
        ("02-measured-residual", "Measured residual"),
        (
            "03-rig-candidate-physical",
            "Rig candidate (infinity plane)",
        ),
        ("04-active-pre-depth", "Active pre-depth"),
        ("05-final-depth", "Final depth"),
    ];
    if debug_dir.join("05a-factory-depth-audit").is_dir() {
        stages.push(("05a-factory-depth-audit", "Audit: factory dense depth"));
    }
    if debug_dir.join("05b-candidate-depth-audit").is_dir() {
        stages.push(("05b-candidate-depth-audit", "Audit: candidate dense depth"));
    }
    let mut html = String::from(
        r##"<!doctype html><html><head><meta charset="utf-8"><title>Chiaro pipeline trace</title><style>
body{margin:0;padding:24px;background:#11151b;color:#e8edf2;font:15px system-ui,sans-serif}a{color:#65c9ff}h1,h2{margin:.4em 0}.hint{color:#aab6c2;max-width:1000px}.camera{margin:24px 0;padding:16px;background:#1a2028;border-radius:10px;overflow:auto}.stages{display:flex;gap:12px;min-width:max-content}.stage{background:#0c1015;padding:8px;border-radius:6px;min-width:260px;flex:1}.stage img{display:block;width:100%;height:auto;background:#000}.maps{display:flex;gap:10px;flex-wrap:wrap;margin-top:7px;font-size:12px}.overview{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}.overview img,.overview object{width:100%;max-height:700px;object-fit:contain;background:#080a0d}code{color:#ffd166}</style></head><body>"##,
    );
    let _ = writeln!(
        html,
        "<h1>Chiaro pipeline trace — reference {reference}</h1>"
    );
    html.push_str(
        "<p class=\"hint\">Read each row left to right. Continuous edges across checkerboard tiles mean the mapping agrees with the images. The crucial comparison is measured residual versus final depth: a degradation there identifies the geometry handoff, independently of final colour or sharpening.</p>",
    );
    html.push_str(
        "<p class=\"hint\"><strong>Physical-stage maps:</strong> stages 01 and 03 project an effectively infinite plane because dense scene depth is not available yet. Their error map is displacement from the measured capture warp, so it includes finite-depth parallax and is not the held-out rig RMS. White confidence means only that the projection lands inside the target sensor. Amber means in-sensor with visibility not yet known; black means outside the target sensor or undefined.</p>",
    );
    html.push_str(
        "<p><a href=\"PIPELINE_TRACE.txt\">Open the numerical/plain-language trace</a></p>",
    );
    for alignment in alignments {
        if alignment.name == reference {
            continue;
        }
        let _ = writeln!(
            html,
            "<section class=\"camera\"><h2>{}</h2><div class=\"stages\">",
            alignment.name
        );
        for &(directory, title) in &stages {
            let checkerboard = format!("{directory}/{}-checkerboard.png", alignment.name);
            let _ = writeln!(
                html,
                "<div class=\"stage\"><strong>{title}</strong><a href=\"{checkerboard}\"><img loading=\"lazy\" src=\"{checkerboard}\"></a>"
            );
            if matches!(
                directory,
                "01-factory-physical"
                    | "03-rig-candidate-physical"
                    | "05-final-depth"
                    | "05a-factory-depth-audit"
                    | "05b-candidate-depth-audit"
            ) {
                let (error_label, confidence_label, visibility_label) = if matches!(
                    directory,
                    "01-factory-physical" | "03-rig-candidate-physical"
                ) {
                    (
                        "far-plane displacement",
                        "in-sensor domain",
                        "pre-depth visibility",
                    )
                } else {
                    ("vs measured displacement", "confidence", "visibility")
                };
                let _ = writeln!(
                    html,
                    "<div class=\"maps\"><a href=\"{directory}/{}-vs-measured-error.png\">{error_label}</a><a href=\"{directory}/{}-confidence.png\">{confidence_label}</a><a href=\"{directory}/{}-visibility.png\">{visibility_label}</a></div>",
                    alignment.name, alignment.name, alignment.name,
                );
            }
            html.push_str("</div>");
        }
        html.push_str("</div></section>");
    }
    html.push_str("<h2>Whole-run diagnostics</h2><div class=\"overview\">");
    if !rig.held_out_observations.is_empty() {
        html.push_str(
            "<div><h3>Individual held-out observations</h3><object data=\"rig-held-out-observations.svg\" type=\"image/svg+xml\"></object><p><a href=\"rig-held-out-observations.svg\">Open full-size map</a> · <a href=\"rig-held-out-observations.csv\">Open exact values (CSV)</a></p></div>",
        );
    }
    if !rig.residual_field.is_empty() {
        html.push_str(
            "<div><h3>Held-out rig residual field</h3><object data=\"rig-held-out-residual-field.svg\" type=\"image/svg+xml\"></object></div>",
        );
    }
    if alignments
        .iter()
        .any(|alignment| alignment.report.depth.is_some())
    {
        html.push_str(
            "<div><h3>Depth</h3><a href=\"depth-visualization.png\"><img loading=\"lazy\" src=\"depth-visualization.png\"></a></div>",
        );
    }
    for (directory, title) in [
        (
            "05a-factory-depth-audit",
            "Factory PhysicalRig depth provenance",
        ),
        (
            "05b-candidate-depth-audit",
            "Candidate PhysicalRig depth provenance",
        ),
    ] {
        if debug_dir
            .join(directory)
            .join("depth-provenance.png")
            .is_file()
        {
            let _ = writeln!(
                html,
                "<div><h3>{title}</h3><a href=\"{directory}/depth-provenance.png\"><img loading=\"lazy\" src=\"{directory}/depth-provenance.png\"></a></div>"
            );
        }
    }
    html.push_str(
        "<div><h3>Luminance ownership</h3><a href=\"source-luminance-ownership.png\"><img loading=\"lazy\" src=\"source-luminance-ownership.png\"></a></div><div><h3>Colour ownership</h3><a href=\"source-color-ownership.png\"><img loading=\"lazy\" src=\"source-color-ownership.png\"></a></div>",
    );
    html.push_str("</div></body></html>");
    fs::write(debug_dir.join("index.html"), html)
        .with_context(|| format!("write debug gallery in {}", debug_dir.display()))
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
    if !target.alignment.geometry_accepted()
        || target.mosaic.is_mono()
        || target.highlight.confidence.is_empty()
    {
        return Vec::new();
    }
    let ratios = sources
        .iter()
        .enumerate()
        .map(|(donor_index, donor)| {
            if target_index == donor_index || !donor.alignment.geometry_accepted() {
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
            if target
                .alignment
                .warp
                .visibility(reference[0], reference[1])
                .blocks_sampling()
                || target.alignment.warp.confidence(reference[0], reference[1]) < 0.7
            {
                continue;
            }
            let channel = target.mosaic.pattern.color_at(y, x);
            let mut estimates = Vec::with_capacity(sources.len() - 1);
            for (donor_index, donor) in sources.iter().enumerate() {
                let Some(ratio) = ratios[donor_index][channel] else {
                    continue;
                };
                if donor
                    .alignment
                    .warp
                    .visibility(reference[0], reference[1])
                    .blocks_sampling()
                    || donor.alignment.warp.confidence(reference[0], reference[1]) < 0.7
                {
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
            if target.alignment.warp.visibility(x, y).blocks_sampling()
                || donor.alignment.warp.visibility(x, y).blocks_sampling()
                || target.alignment.warp.confidence(x, y) < 0.75
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
    let alignment_pyramids = luminance
        .iter()
        .map(AlignPyramidCache::new)
        .collect::<Vec<_>>();
    let reference_index = modules
        .iter()
        .position(|module| module.raw.name == reference_name)
        .expect("reference selected");
    if options
        .cfa_held_out
        .iter()
        .any(|camera| camera.eq_ignore_ascii_case(&modules[reference_index].raw.name))
    {
        bail!(
            "reference module {} cannot be held out: its image defines the reconstruction coordinate/evidence frame",
            modules[reference_index].raw.name
        );
    }
    let inputs = alignment_inputs(&modules, &luminance);
    let factory_alignments = align_all_modules(
        &inputs,
        &alignment_pyramids,
        reference_index,
        &options.align,
        options.threads,
    )?;
    drop(inputs);
    let factory_measured_debug_warps = options
        .debug_dir
        .as_ref()
        .map(|_| alignment_debug_warps(&factory_alignments));
    let factory_depth_audit_seed = options
        .debug_dir
        .as_ref()
        .map(|_| factory_alignments.clone());
    if let (Some(debug_dir), Some(measured_warps)) =
        (&options.debug_dir, &factory_measured_debug_warps)
    {
        fs::create_dir_all(debug_dir).with_context(|| format!("create {}", debug_dir.display()))?;
        let factory_cameras = modules
            .iter()
            .map(|module| module.camera.clone())
            .collect::<Vec<_>>();
        let factory_physical = physical_debug_warps(
            &factory_cameras,
            reference_index,
            modules[reference_index].raw.width,
            modules[reference_index].raw.height,
        );
        write_debug_warp_stage(
            debug_dir,
            "01-factory-physical",
            &modules,
            &luminance,
            reference_index,
            &factory_physical,
            Some(measured_warps),
            true,
        )?;
        write_debug_warp_stage(
            debug_dir,
            "02-measured-residual",
            &modules,
            &luminance,
            reference_index,
            measured_warps,
            None,
            false,
        )?;
    }

    progress(Progress {
        stage: "align",
        detail: "validating capture-specific physical rig".to_owned(),
        fraction: 0.42,
    });
    let rig_inputs = modules
        .iter()
        .enumerate()
        .map(|(index, module)| RigCameraInput {
            name: &module.raw.name,
            calibration: calibration.cameras.get(&module.raw.name),
            state: module.state.as_ref(),
            match_evidence_enabled: !options
                .cfa_held_out
                .iter()
                .any(|camera| camera.eq_ignore_ascii_case(&module.raw.name)),
            luminance: luminance.get(index),
        })
        .collect::<Vec<_>>();
    let mut rig_options = options.rig_refinement.clone();
    rig_options.threads = options.threads;
    rig_options.held_out_validation = options.debug_dir.is_some();
    let mut rig_outcome = refine_capture_rig(
        &rig_inputs,
        reference_index,
        &factory_alignments,
        options.intrinsics_mode,
        &rig_options,
    );
    drop(rig_inputs);
    let candidate_audit_cameras = options.debug_dir.as_ref().map(|_| {
        candidate_debug_cameras(
            &modules,
            &calibration,
            &rig_outcome.report,
            options.intrinsics_mode,
        )
    });
    if let (Some(debug_dir), Some(measured_warps), Some(candidate_cameras)) = (
        &options.debug_dir,
        &factory_measured_debug_warps,
        &candidate_audit_cameras,
    ) {
        let candidate_physical = physical_debug_warps(
            candidate_cameras,
            reference_index,
            modules[reference_index].raw.width,
            modules[reference_index].raw.height,
        );
        write_debug_warp_stage(
            debug_dir,
            "03-rig-candidate-physical",
            &modules,
            &luminance,
            reference_index,
            &candidate_physical,
            Some(measured_warps),
            true,
        )?;
        write_rig_residual_field_svg(debug_dir, &rig_outcome.report)?;
        write_rig_held_out_observations(debug_dir, &rig_outcome.report)?;
    }
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
        let refined_alignments = align_all_modules(
            &refined_inputs,
            &alignment_pyramids,
            reference_index,
            &options.align,
            options.threads,
        )?;
        evaluate_image_space_alignment(
            &mut rig_outcome.report,
            &factory_alignments,
            &refined_alignments,
            options
                .rig_refinement
                .min_image_space_correction_improvement,
        );
        refined_alignments
    } else {
        factory_alignments
    };
    let active_pre_depth_debug_warps = options
        .debug_dir
        .as_ref()
        .map(|_| alignment_debug_warps(&alignments));
    if let (Some(debug_dir), Some(active_warps)) =
        (&options.debug_dir, &active_pre_depth_debug_warps)
    {
        write_debug_warp_stage(
            debug_dir,
            "04-active-pre-depth",
            &modules,
            &luminance,
            reference_index,
            active_warps,
            None,
            false,
        )?;
    }
    let mut inputs = alignment_inputs(&modules, &luminance);
    let mut depth_options = options.align.depth.clone();
    depth_options.threads = options.threads;
    disable_held_out_depth_evidence(&mut inputs, &options.cfa_held_out);
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
    // Dense correspondence is a property of the calibrated physical rig, not
    // of whether a capture-specific correction happened to improve that rig.
    // If refinement is rejected the module cameras above have already fallen
    // back to factory calibration, which remains the correct physical model.
    // Use the legacy warp-seeded path only when there are not enough calibrated
    // cameras to form a physical multi-view depth hypothesis at all.
    let calibrated_depth_views = inputs
        .iter()
        .enumerate()
        .filter(|(index, input)| {
            *index != reference_index && input.camera.is_some() && input.depth_evidence_enabled
        })
        .count();
    let mut depth_geometry_mode = if inputs[reference_index].camera.is_some()
        && calibrated_depth_views >= options.align.depth.minimum_support
    {
        DepthGeometryMode::PhysicalRig
    } else {
        DepthGeometryMode::WarpSeeded
    };
    let depth_map = if options.align.refine && options.align.depth.enabled {
        progress(Progress {
            stage: "align",
            detail: "joint physical multi-view depth/correspondence".to_owned(),
            fraction: 0.45,
        });
        if matches!(depth_geometry_mode, DepthGeometryMode::PhysicalRig) {
            // Keep a deterministic compatibility fallback for captures where
            // factory/refined physical geometry cannot produce enough usable
            // target views.  Physical geometry remains the primary path; the
            // old warp is restored only when the physical solve itself fails.
            let warp_seeded_alignments = alignments.clone();
            let physical = refine_multiview_depth(
                &inputs,
                reference_index,
                &mut alignments,
                &depth_options,
                DepthGeometryMode::PhysicalRig,
            );
            let physical_views = inputs
                .iter()
                .enumerate()
                .filter(|(index, input)| {
                    *index != reference_index
                        && input.depth_evidence_enabled
                        && alignments[*index].geometry_accepted()
                })
                .count();
            if physical.is_some() && physical_views >= options.align.depth.minimum_support {
                physical
            } else {
                alignments = warp_seeded_alignments;
                depth_geometry_mode = DepthGeometryMode::WarpSeeded;
                refine_multiview_depth(
                    &inputs,
                    reference_index,
                    &mut alignments,
                    &depth_options,
                    DepthGeometryMode::WarpSeeded,
                )
            }
        } else {
            refine_multiview_depth(
                &inputs,
                reference_index,
                &mut alignments,
                &depth_options,
                DepthGeometryMode::WarpSeeded,
            )
        }
    } else {
        None
    };
    let dense_depth_audit = if options.align.refine && options.align.depth.enabled {
        match (
            &options.debug_dir,
            &factory_depth_audit_seed,
            &factory_measured_debug_warps,
            &candidate_audit_cameras,
        ) {
            (
                Some(debug_dir),
                Some(measured_seed),
                Some(measured_warps),
                Some(candidate_cameras),
            ) => {
                progress(Progress {
                    stage: "align",
                    detail: "auditing factory and candidate physical depth independently"
                        .to_owned(),
                    fraction: 0.48,
                });
                let factory = if matches!(depth_geometry_mode, DepthGeometryMode::PhysicalRig)
                    && !rig_outcome.report.accepted
                    && depth_map.is_some()
                {
                    // The production branch is already exactly the forced
                    // factory PhysicalRig solve in this case. Reuse it rather
                    // than rebuilding the same full cost volume a second time.
                    write_dense_depth_audit_outputs(
                        debug_dir,
                        "05a-factory-depth-audit",
                        &modules,
                        &luminance,
                        reference_index,
                        &alignments,
                        depth_map.as_ref(),
                        measured_warps,
                    )?
                } else {
                    run_dense_depth_audit_branch(
                        debug_dir,
                        "05a-factory-depth-audit",
                        &modules,
                        &luminance,
                        &factory_cameras,
                        reference_index,
                        measured_seed,
                        measured_warps,
                        &options.cfa_held_out,
                        &depth_options,
                    )?
                };
                let candidate = run_dense_depth_audit_branch(
                    debug_dir,
                    "05b-candidate-depth-audit",
                    &modules,
                    &luminance,
                    candidate_cameras,
                    reference_index,
                    measured_seed,
                    measured_warps,
                    &options.cfa_held_out,
                    &depth_options,
                )?;
                let selected_path = match depth_geometry_mode {
                    DepthGeometryMode::WarpSeeded => "WarpSeeded compatibility fallback",
                    DepthGeometryMode::PhysicalRig if rig_outcome.report.accepted => {
                        "accepted candidate PhysicalRig"
                    }
                    DepthGeometryMode::PhysicalRig => "factory PhysicalRig",
                };
                Some(DenseDepthAuditReport {
                    selected_path: selected_path.to_owned(),
                    common_anchor:
                        "02-measured-residual perpendicular epipolar proposal/output fallback"
                            .to_owned(),
                    factory,
                    candidate,
                })
            }
            _ => None,
        }
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
        let final_depth_warps = alignment_debug_warps(&alignments);
        write_debug_warp_stage(
            debug_dir,
            "05-final-depth",
            &modules,
            &luminance,
            reference_index,
            &final_depth_warps,
            active_pre_depth_debug_warps.as_deref(),
            true,
        )?;
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
        if contributor_enabled[index] && alignment.geometry_accepted() && !module.mosaic.is_mono() {
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
            && alignments[index].geometry_accepted()
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
        if alignment
            .report
            .depth
            .as_ref()
            .is_some_and(|depth| depth.physical_geometry)
        {
            // Physical depth carries its confidence per warp node. Do not
            // re-apply the old homography inlier ratio as a global veto after
            // that 2-D model has ceased to define correspondence.
            return 1.0;
        }
        // Compatibility path: retain the legacy global correspondence gate.
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
            }) && alignment.geometry_accepted()
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
                    || alignment.geometry_accepted()
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
                    && alignment.geometry_accepted()
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
    if let Some(debug_dir) = &options.debug_dir {
        write_pipeline_trace(
            debug_dir,
            &reference_name,
            &rig_outcome.report,
            depth_geometry_mode,
            dense_depth_audit.as_ref(),
            &alignments,
            &synthesis,
        )?;
    }
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
        dense_depth_audit,
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

    #[test]
    fn diagnostic_domain_excludes_finite_points_outside_the_target_sensor() {
        assert!(diagnostic_point_inside_raster([0.0, 0.0], 10, 8));
        assert!(diagnostic_point_inside_raster([9.0, 7.0], 10, 8));
        assert!(!diagnostic_point_inside_raster([-0.01, 4.0], 10, 8));
        assert!(!diagnostic_point_inside_raster([10.0, 4.0], 10, 8));
        assert!(!diagnostic_point_inside_raster([4.0, 8.0], 10, 8));
        assert!(!diagnostic_point_inside_raster([f32::NAN, 4.0], 10, 8));
        assert!(!diagnostic_point_inside_raster([4.0, 4.0], 0, 8));
    }

    #[test]
    fn dense_depth_audit_keeps_finite_fill_separate_from_defined_warp_support() {
        let alignment = |name: &str, reference: bool| ModuleAlignment {
            name: name.to_owned(),
            warp: Warp::from_fn(8, 8, 4, Some),
            correspondences: Vec::new(),
            gain: 1.0,
            offset: 0.0,
            report: AlignmentReport {
                accepted: true,
                geometry_accepted: Some(true),
                depth: (!reference).then_some(DepthAlignmentReport {
                    physical_geometry: true,
                    tested_nodes: 100,
                    measured_nodes: 10,
                    regularized_nodes: 0,
                    fallback_nodes: 80,
                    refined_nodes: 8,
                    defined_nodes: 90,
                    defined_fraction: 0.90,
                    reconstructed_fraction: 0.10,
                    ..Default::default()
                }),
                ..Default::default()
            },
        };
        let alignments = [alignment("B4", true), alignment("B5", false)];
        let depth_map = DenseDepthMap {
            columns: 1,
            rows: 1,
            step: 4,
            near_depth: 500.0,
            far_depth: 10_000_000.0,
            nodes: vec![crate::depth::DenseDepthNode {
                depth: Some(2_000.0),
                confidence: 0.9,
                provenance: crate::depth::DepthProvenance::Measured,
            }],
        };
        let report = dense_depth_audit_branch_report(Some(&depth_map), &alignments, 0);
        assert!(report.depth_available);
        assert_eq!(report.measured_nodes, 10);
        assert_eq!(report.regularized_nodes, 0);
        assert_eq!(report.reconstructed_fraction, 0.10);
        assert_eq!(report.accepted_views, 1);
        assert_eq!(
            report.cameras[1].depth.as_ref().unwrap().defined_fraction,
            0.90
        );
    }

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
