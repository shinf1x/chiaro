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
    hotpixel::HotpixelRec,
    pipeline::{CleanupStage, FramePipeline, extract_raw_plane_threaded},
    thermal::ThermalProfile,
    universal_hotpixel::UniversalHotpixelProfile,
};
use serde::Serialize;

use crate::align::{AlignInput, AlignOptions, AlignmentReport, ModuleAlignment, align_module};
use crate::calibration::{
    CalibrationDatabase, CameraCalibration, IntrinsicsMode, LriMessages, awb_gains,
    image_focal_length_mm, module_states,
};
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
        modules.push(LoadedModule {
            raw: raw.clone(),
            mosaic,
            camera,
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
    if let Some(debug_dir) = &options.debug_dir {
        fs::create_dir_all(debug_dir).with_context(|| format!("create {}", debug_dir.display()))?;
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

    // Colour per module: its own D65 forward matrix, and the recorded white
    // balance transferred from the reference through the D65 grey ratios.
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
            magnification: magnification(module),
            confidence: synthesis_confidence(alignment),
            color: *color,
            gain_field: gain_field.clone(),
        })
        .collect::<Vec<_>>();
    let synthesis = synthesize(output, crop, scale, &sources, &color, &options.synth)?;
    timings.synthesize = stage_started.elapsed().as_secs_f32();

    let report = FusionReport {
        reference: reference_name,
        calibration_modules: calibration.cameras.len(),
        framed_focal_length_mm,
        modules: alignments.iter().map(|a| a.report.clone()).collect(),
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
}
