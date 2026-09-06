//! Combined temporal and calibrated multi-module night fusion.

use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use chiaro::lri::{RawCamera, parse_frame_layout};
use chiaro_fusion::{
    align::{AlignInput, AlignOptions, AlignmentReport, ModuleAlignment, Warp, align_module},
    calibration::{
        CalibrationDatabase, CameraCalibration, IntrinsicsMode, LriMessages, awb_gains,
        camera_name, image_focal_length_mm, module_states,
    },
    crosstalk::{
        AdaptiveCrosstalkReport, CrosstalkFitSource, CrosstalkMode, fit_adaptive_crosstalk,
    },
    depth::{DenseDepthMap, refine_multiview_depth},
    geometry::{CameraRefinement, ResolvedCamera},
    image::{Mosaic, Plane},
    pipeline::{
        RawHighlightSource, cross_camera_highlight_updates, group_equivalent_focal_mm,
        nominal_focal_px,
    },
    resolution::{ResolutionReconstruction, refine_resolution_warp},
    synth::{
        CanvasMode, ColorPipeline, CropWindow, GainField, ModuleColor, OutputColor, SynthOptions,
        SynthReport, SynthSource, auto_exposure, canvas_scale, photometric_field,
        photometric_match, synthesize,
    },
};
use chiaro_hotpixel_core::{
    cleanup::CleanupProfile,
    demosaic::DemosaicMethod,
    highlight::{HighlightRecoveryReport, HighlightRecoveryState},
    hotpixel::HotpixelRec,
    thermal::ThermalProfile,
    universal_hotpixel::UniversalHotpixelProfile,
};
use serde::Serialize;

use crate::{StackOptions, StackReport, stack_mosaic_burst};

#[derive(Clone, Debug)]
pub struct NightFusionOptions {
    pub reference: Option<String>,
    pub cameras: Vec<String>,
    pub overlays: Vec<PathBuf>,
    pub hotpixel_rec: Option<PathBuf>,
    pub cleanup_profile: Option<PathBuf>,
    pub intrinsics_mode: IntrinsicsMode,
    pub temporal_align: AlignOptions,
    pub module_align: AlignOptions,
    pub motion_sigma: f32,
    pub gyro_seed: bool,
    pub flat_field: bool,
    pub local_photometric: bool,
    pub crop_to_framing: bool,
    pub synth: SynthOptions,
    pub crosstalk: CrosstalkMode,
    pub threads: usize,
}

impl Default for NightFusionOptions {
    fn default() -> Self {
        Self {
            reference: None,
            cameras: Vec::new(),
            overlays: Vec::new(),
            hotpixel_rec: None,
            cleanup_profile: None,
            intrinsics_mode: IntrinsicsMode::LinearHall,
            temporal_align: AlignOptions {
                min_inlier_ratio: 0.30,
                ..AlignOptions::default()
            },
            module_align: AlignOptions::default(),
            motion_sigma: 4.0,
            gyro_seed: true,
            flat_field: true,
            local_photometric: true,
            crop_to_framing: true,
            synth: SynthOptions {
                demosaic: DemosaicMethod::Lmmse,
                // Joint CFA still lacks propagated temporal variance and
                // provenance for merged Night mosaics (CFA-13).
                resolution_reconstruction: ResolutionReconstruction::MultiCamera,
                ..SynthOptions::default()
            },
            crosstalk: CrosstalkMode::default(),
            threads: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NightFusionReport {
    pub reference_camera: String,
    pub reference_frame: u64,
    pub temporal: Vec<StackReport>,
    pub modules: Vec<AlignmentReport>,
    pub highlights: Vec<(String, HighlightRecoveryReport)>,
    pub crosstalk: Vec<(String, AdaptiveCrosstalkReport)>,
    /// Per-module synthesis confidence contributed by the calibrated dark
    /// noise estimate, relative to the reference module.
    pub dark_noise_weights: Vec<(String, f32)>,
    pub synthesis: SynthReport,
    /// Exact dense reconstruction control field, retained for optional image
    /// diagnostics but omitted from the JSON report.
    #[serde(skip)]
    pub depth_map: Option<DenseDepthMap>,
}

struct LoadedModule {
    raw: RawCamera,
    mosaic: Mosaic,
    camera: Option<ResolvedCamera>,
    dark_noise_variance: Option<f32>,
    highlight: HighlightRecoveryState,
    capture_gain: f32,
    exposure_ns: u64,
}

pub fn fuse_night(
    lri: &[u8],
    options: &NightFusionOptions,
    output: &Path,
    progress: &mut dyn FnMut(&str),
) -> Result<NightFusionReport> {
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
    let frame_layout = parse_frame_layout(lri, &HashMap::new()).map_err(|e| anyhow::anyhow!(e))?;
    let mut counts = HashMap::<String, usize>::new();
    for frame in &frame_layout.frames {
        *counts.entry(frame.camera.name.clone()).or_default() += 1;
    }
    let available = counts
        .into_iter()
        .filter_map(|(name, count)| (count >= 2).then_some(name))
        .collect::<BTreeSet<_>>();
    let reference_name = options
        .reference
        .clone()
        .or_else(|| {
            messages
                .headers
                .iter()
                .find_map(|header| header.image_reference_camera)
                .map(|id| camera_name(id.value()))
        })
        .unwrap_or_else(|| "A1".to_owned())
        .to_ascii_uppercase();
    if !available.contains(&reference_name) {
        bail!("reference module {reference_name} has no temporal burst");
    }
    let mut names = available
        .into_iter()
        .filter(|name| {
            options.cameras.is_empty()
                || options
                    .cameras
                    .iter()
                    .any(|wanted| wanted.eq_ignore_ascii_case(name))
        })
        .collect::<Vec<_>>();
    if !names.iter().any(|name| name == &reference_name) {
        bail!("reference module {reference_name} is not selected");
    }
    names.sort_by_key(|name| (name != &reference_name, name.clone()));
    if options.cleanup_profile.is_some() && options.hotpixel_rec.is_none() {
        bail!("--cleanup-profile requires the corresponding --hotpixel-rec");
    }
    let hotpixel = options
        .hotpixel_rec
        .as_ref()
        .map(HotpixelRec::open)
        .transpose()?;
    let cleanup = match (&options.cleanup_profile, &hotpixel) {
        (Some(path), Some(rec)) => Some(
            CleanupProfile::open(path, rec)
                .with_context(|| format!("open cleanup profile {}", path.display()))?,
        ),
        _ => None,
    };
    let universal = hotpixel
        .is_some()
        .then(UniversalHotpixelProfile::bundled)
        .transpose()?
        .map(Arc::new);
    let thermal = hotpixel
        .is_some()
        .then(ThermalProfile::bundled)
        .transpose()?
        .map(Arc::new);

    let mut modules = Vec::with_capacity(names.len());
    let mut temporal_reports = Vec::with_capacity(names.len());
    let mut shared_reference_frame = None;
    let mut shared_motion = HashMap::<u64, Warp>::new();
    let mut shared_raw = None::<RawCamera>;
    let mut shared_focal = 0.0f64;
    for name in &names {
        progress(&format!("temporal stack {name}"));
        let state = states.get(name);
        let resolved = match (state, calibration.cameras.get(name)) {
            (Some(state), Some(camera)) => ResolvedCamera::new(
                camera,
                state,
                options.intrinsics_mode,
                &CameraRefinement::default(),
            )
            .ok(),
            _ => None,
        };
        let focal = resolved
            .as_ref()
            .map(|camera| camera.focal_px)
            .unwrap_or_else(|| nominal_focal_px(name));
        let raw = frame_layout
            .frames
            .iter()
            .find(|frame| frame.camera.name == *name)
            .map(|frame| &frame.camera)
            .with_context(|| format!("capture has no {name}"))?;
        let severity_map = hotpixel
            .as_ref()
            .map(|rec| rec.load_rotated_map(raw.id, raw.width, raw.height))
            .transpose()
            .with_context(|| format!("load hotpixel map for {name}"))?;
        let cleanup_camera = cleanup
            .as_ref()
            .map(|profile| profile.load_camera(raw))
            .transpose()
            .with_context(|| format!("load cleanup profile for {name}"))?
            .flatten();
        let motion_seeds = match (&shared_raw, shared_motion.is_empty()) {
            (Some(source), false) => scale_motion_warps(
                &shared_motion,
                source,
                shared_focal,
                raw,
                focal,
                options.temporal_align.grid_step,
            ),
            _ => HashMap::new(),
        };
        let stack_options = StackOptions {
            camera: name.clone(),
            align: options.temporal_align.clone(),
            motion_sigma: options.motion_sigma,
            severity_map,
            cleanup_profile_supplied: cleanup.is_some(),
            cleanup_profile: cleanup_camera,
            universal_hotpixel_model: universal.clone(),
            thermal_model: thermal.clone(),
            reference_frame: shared_reference_frame,
            gyro_seed: options.gyro_seed,
            focal_px: Some(focal),
            motion_seeds,
            noise_profiles: calibration.sensor_noise_profiles.clone(),
            demosaic: options.synth.demosaic,
            highlight_recovery: options.synth.highlight_recovery,
            threads: options.threads,
        };
        let stack = stack_mosaic_burst(lri, &stack_options)?;
        if shared_reference_frame.is_none() {
            shared_reference_frame = Some(stack.report.reference_frame);
            shared_motion = stack.temporal_warps.clone();
            shared_raw = Some(stack.camera.clone());
            shared_focal = focal;
        }
        let mut mosaic = Mosaic {
            width: stack.camera.width,
            height: stack.camera.height,
            pattern: stack.camera.pattern,
            samples: stack.mosaic16,
            black_q6: 0.0,
            white_q6: 65535.0,
            physical_code_range: 65535.0,
            vignetting: None,
            crosstalk: None,
            demosaiced_rgb: None,
        };
        if options.synth.highlight_recovery
            != chiaro_hotpixel_core::highlight::HighlightRecovery::None
            && !mosaic.is_mono()
        {
            mosaic.reserve_highlight_headroom();
        }
        if options.flat_field
            && let Some(vignetting) = calibration
                .cameras
                .get(name)
                .and_then(|camera| camera.vignetting.as_ref())
        {
            let mirror_hall = state.map_or(0.0, |state| state.mirror_hall);
            mosaic.vignetting = vignetting.mesh_for_hall(mirror_hall);
            if !mosaic.is_mono() {
                mosaic.crosstalk = vignetting.crosstalk.clone();
            }
        }
        let dark_noise_variance = stack.report.dark_noise_variance;
        temporal_reports.push(stack.report);
        modules.push(LoadedModule {
            raw: stack.camera,
            mosaic,
            camera: resolved,
            dark_noise_variance,
            highlight: stack.highlight,
            capture_gain: state.map_or(1.0, |state| state.gain as f32),
            exposure_ns: state.map_or(0, |state| state.exposure_ns),
        });
    }

    progress("align denoised modules");
    let luminance = modules
        .iter()
        .map(|module| module.mosaic.luminance_half())
        .collect::<Vec<Plane>>();
    let reference_index = modules
        .iter()
        .position(|module| module.raw.name == reference_name)
        .expect("reference was selected");
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
                .map(|camera| camera.focal_px)
                .unwrap_or_else(|| nominal_focal_px(&module.raw.name)),
        })
        .collect::<Vec<_>>();
    let reference_input = &inputs[reference_index];
    let mut alignments = std::thread::scope(|scope| {
        inputs
            .iter()
            .map(|input| {
                scope.spawn(|| align_module(reference_input, input, &options.module_align))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("module alignment worker panicked"))
            .collect::<Result<Vec<_>>>()
    })?;
    let depth_map = if options.module_align.refine && options.module_align.depth.enabled {
        progress("refine shared multi-view depth");
        refine_multiview_depth(
            &inputs,
            reference_index,
            &mut alignments,
            &options.module_align.depth,
        )
    } else {
        None
    };
    let resolution_warps = if options
        .synth
        .resolution_reconstruction
        .uses_resolution_warps()
    {
        progress("refine local resolution alignment");
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
    if options.synth.highlight_recovery.uses_multi_camera() {
        progress("recover cross-camera RAW highlights");
        let reference_dimensions = (
            modules[reference_index].raw.width,
            modules[reference_index].raw.height,
        );
        let updates = {
            let sources = modules
                .iter()
                .zip(&alignments)
                .map(|(module, alignment)| RawHighlightSource {
                    mosaic: &module.mosaic,
                    highlight: &module.highlight,
                    alignment,
                })
                .collect::<Vec<_>>();
            (0..sources.len())
                .map(|target| {
                    cross_camera_highlight_updates(
                        &sources,
                        target,
                        reference_dimensions.0,
                        reference_dimensions.1,
                    )
                })
                .collect::<Vec<_>>()
        };
        for (module, updates) in modules.iter_mut().zip(updates) {
            for update in updates {
                module.mosaic.samples[update.index] = update.value;
                module
                    .highlight
                    .mark_multi_camera(update.index, update.confidence);
            }
            module.highlight.finish_multi_camera();
        }
    }

    let reference_calibration = calibration.cameras.get(&reference_name);
    let recorded_wb = awb_gains(&messages).map(|gains| gains.map(|gain| gain as f32));
    let colors = modules
        .iter()
        .map(|module| {
            module_color(
                calibration.cameras.get(&module.raw.name),
                reference_calibration,
                recorded_wb,
            )
        })
        .collect::<Vec<_>>();
    progress("fit adaptive RAW crosstalk");
    let crosstalk_fits = {
        let sources = modules
            .iter()
            .zip(&alignments)
            .zip(&colors)
            .map(|((module, alignment), &color)| CrosstalkFitSource {
                mosaic: &module.mosaic,
                highlight: &module.highlight,
                alignment,
                color,
                capture_gain: module.capture_gain,
                exposure_ns: module.exposure_ns,
            })
            .collect::<Vec<_>>();
        fit_adaptive_crosstalk(
            &sources,
            reference_index,
            options.crosstalk,
            modules[reference_index].mosaic.width,
            modules[reference_index].mosaic.height,
        )
    };
    let crosstalk_reports = modules
        .iter_mut()
        .zip(crosstalk_fits)
        .map(|(module, fit)| {
            module.mosaic.crosstalk = fit.mesh;
            (module.raw.name.clone(), fit.report)
        })
        .collect::<Vec<_>>();
    for (module, alignment) in modules.iter_mut().zip(&alignments) {
        if alignment.report.accepted && !module.mosaic.is_mono() {
            module
                .mosaic
                .prepare_demosaic(options.synth.demosaic, options.threads)
                .with_context(|| format!("demosaic {}", module.raw.name))?;
        }
    }
    let mut gain_fields = vec![GainField::identity(); modules.len()];
    for index in 0..modules.len() {
        if index == reference_index || !alignments[index].report.accepted {
            continue;
        }
        let (gain, offset) = photometric_match(
            &modules[reference_index].mosaic,
            &colors[reference_index],
            &modules[index].mosaic,
            &colors[index],
            &alignments[index].warp,
            options.synth.highlight_correction,
        );
        alignments[index].gain = gain;
        alignments[index].offset = offset;
        if options.local_photometric {
            let (columns, rows) = if alignments[index].report.coverage >= 0.5 {
                (12, 9)
            } else {
                (1, 1)
            };
            gain_fields[index] = photometric_field(
                &modules[reference_index].mosaic,
                &colors[reference_index],
                &modules[index].mosaic,
                &colors[index],
                &alignments[index].warp,
                gain,
                offset,
                columns,
                rows,
                options.synth.highlight_correction,
            );
        }
    }

    progress("synthesise all-module night image");
    let reference = &modules[reference_index];
    let crop = match image_focal_length_mm(&messages) {
        Some(focal) if options.crop_to_framing && focal > 0 => CropWindow::centred(
            reference.raw.width,
            reference.raw.height,
            group_equivalent_focal_mm(&reference_name) / focal as f32,
        ),
        _ => CropWindow::full(reference.raw.width, reference.raw.height),
    };
    let reference_focal = module_focal(reference);
    let reference_noise = reference.dark_noise_variance;
    let magnifications = modules
        .iter()
        .map(|module| (module_focal(module) / reference_focal) as f32)
        .collect::<Vec<_>>();
    let finest = alignments
        .iter()
        .zip(&magnifications)
        .filter(|(alignment, _)| alignment.report.accepted)
        .map(|(_, magnification)| *magnification)
        .fold(1.0, f32::max);
    let scale = canvas_scale(&crop, reference.raw.width, options.synth.canvas, finest);
    let color_pipeline = ColorPipeline {
        exposure: auto_exposure(
            &reference.mosaic,
            &colors[reference_index],
            options.synth.highlight_correction,
        ),
    };
    let sources = modules
        .iter()
        .zip(&alignments)
        .zip(colors.iter().zip(&gain_fields))
        .zip(&magnifications)
        .zip(&resolution_warps)
        .filter(|((((_, alignment), _), _), resolution_warp)| {
            alignment.report.accepted
                || resolution_warp.as_ref().is_some_and(|refined| {
                    refined.report.supported_fraction >= 0.005
                        && refined.report.mean_confidence >= 0.5
                })
        })
        .map(
            |((((module, alignment), (color, gain_field)), magnification), resolution_warp)| {
                SynthSource {
                    camera_id: module.raw.id,
                    mosaic: &module.mosaic,
                    highlight: &module.highlight,
                    noise_model: None,
                    held_out: false,
                    alignment,
                    resolution_warp: resolution_warp.as_ref(),
                    fusion_enabled: alignment.report.accepted,
                    reference: alignment.name == reference_name,
                    magnification: *magnification,
                    confidence: alignment_confidence(
                        alignment,
                        &reference_name,
                        &options.module_align,
                    ) * noise_confidence(reference_noise, module.dark_noise_variance),
                    focus_distance: module
                        .camera
                        .as_ref()
                        .and_then(|camera| camera.focus_distance),
                    color: *color,
                    gain_field: gain_field.clone(),
                }
            },
        )
        .collect::<Vec<_>>();
    let dark_noise_weights = modules
        .iter()
        .map(|module| {
            (
                module.raw.name.clone(),
                noise_confidence(reference_noise, module.dark_noise_variance),
            )
        })
        .collect();
    let synthesis = synthesize(
        output,
        crop,
        scale,
        &sources,
        depth_map.as_ref(),
        None,
        &color_pipeline,
        &options.synth,
    )?;
    let report = NightFusionReport {
        reference_camera: reference_name,
        reference_frame: shared_reference_frame.unwrap_or(0),
        temporal: temporal_reports,
        modules: alignments
            .iter()
            .map(|alignment| alignment.report.clone())
            .collect(),
        highlights: modules
            .iter()
            .map(|module| (module.raw.name.clone(), module.highlight.report.clone()))
            .collect(),
        crosstalk: crosstalk_reports,
        dark_noise_weights,
        synthesis,
        depth_map,
    };
    let report_path = output.with_extension("night-fusion.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    Ok(report)
}

fn scale_motion_warps(
    source: &HashMap<u64, Warp>,
    source_raw: &RawCamera,
    source_focal: f64,
    target_raw: &RawCamera,
    target_focal: f64,
    step: usize,
) -> HashMap<u64, Warp> {
    source
        .iter()
        .map(|(&frame, warp)| {
            let scaled = Warp::from_fn(target_raw.width, target_raw.height, step, |point| {
                let source_point = [
                    point[0] * source_raw.width as f64 / target_raw.width as f64,
                    point[1] * source_raw.height as f64 / target_raw.height as f64,
                ];
                let mapped = warp.map(source_point[0] as f32, source_point[1] as f32)?;
                let focal_scale = target_focal / source_focal.max(1.0);
                Some([
                    point[0] + (f64::from(mapped[0]) - source_point[0]) * focal_scale,
                    point[1] + (f64::from(mapped[1]) - source_point[1]) * focal_scale,
                ])
            });
            (frame, scaled)
        })
        .collect()
}

fn module_focal(module: &LoadedModule) -> f64 {
    module
        .camera
        .as_ref()
        .map(|camera| camera.focal_px)
        .unwrap_or_else(|| nominal_focal_px(&module.raw.name))
}

fn alignment_confidence(
    alignment: &ModuleAlignment,
    reference: &str,
    options: &AlignOptions,
) -> f32 {
    if !options.refine || alignment.name == reference {
        return 1.0;
    }
    let minimum = options.min_inlier_ratio.clamp(0.0, 1.0);
    let reliable = (minimum + 0.25).min(1.0);
    let t = ((alignment.report.inlier_ratio - minimum) / (reliable - minimum).max(1e-3))
        .clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)).max(0.05)
}

fn noise_confidence(reference: Option<f32>, module: Option<f32>) -> f32 {
    match (reference, module) {
        (Some(reference), Some(module)) if reference > 0.0 && module > 0.0 => {
            (reference / module).clamp(0.1, 4.0)
        }
        _ => 1.0,
    }
}

fn module_color(
    module: Option<&CameraCalibration>,
    reference: Option<&CameraCalibration>,
    recorded_wb: Option<[f32; 3]>,
) -> ModuleColor {
    fn d65(
        camera: Option<&CameraCalibration>,
    ) -> Option<&chiaro_fusion::calibration::ColorProfile> {
        camera.and_then(|camera| {
            camera
                .color
                .iter()
                .find(|profile| profile.illuminant == 2)
                .or(camera.color.first())
        })
    }
    let (profile, reference_profile) = (d65(module), d65(reference));
    let mut color = ModuleColor::default();
    if let Some(profile) = profile {
        color.forward = profile
            .forward_matrix
            .map(|row| row.map(|value| value as f32));
        color.calibrated = true;
    }
    color.wb_gains = match (recorded_wb, profile, reference_profile) {
        (Some(wb), Some(profile), Some(reference)) => [
            wb[0] * (reference.rg_ratio / profile.rg_ratio.max(1e-3)) as f32,
            wb[1],
            wb[2] * (reference.bg_ratio / profile.bg_ratio.max(1e-3)) as f32,
        ],
        (Some(wb), _, _) => wb,
        (None, Some(profile), _) => [
            (1.0 / profile.rg_ratio.max(0.01)) as f32,
            1.0,
            (1.0 / profile.bg_ratio.max(0.01)) as f32,
        ],
        _ => [1.0; 3],
    };
    color
}

pub fn set_output_color(options: &mut NightFusionOptions, linear: bool) {
    options.synth.color = if linear {
        OutputColor::Linear
    } else {
        OutputColor::Display
    };
}

pub fn set_canvas(options: &mut NightFusionOptions, maximum: bool, max_megapixels: f32) {
    options.synth.canvas = if maximum {
        CanvasMode::Maximum { max_megapixels }
    } else {
        CanvasMode::Native
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use chiaro::lri::SensorPattern;

    #[test]
    fn night_fusion_keeps_temporally_validated_defaults() {
        let options = NightFusionOptions::default();
        assert_eq!(options.synth.demosaic, DemosaicMethod::Lmmse);
        assert_eq!(
            options.synth.resolution_reconstruction,
            ResolutionReconstruction::MultiCamera
        );
    }

    fn raw(name: &str) -> RawCamera {
        RawCamera {
            id: 0,
            name: name.to_owned(),
            width: 100,
            height: 80,
            row_stride: 0,
            absolute_offset: 0,
            byte_len: 0,
            pattern: SensorPattern::Rggb,
            sensor_temperature_c: None,
            analog_gain: 1.0,
            digital_gain: 1.0,
            exposure_ns: 10_000_000,
            black_level: 0.0,
            white_level: 255.0,
        }
    }

    #[test]
    fn shared_motion_displacement_scales_with_focal_length() {
        let source = Warp::from_fn(100, 80, 10, |point| Some([point[0] + 2.0, point[1] - 1.0]));
        let scaled = scale_motion_warps(
            &HashMap::from([(1, source)]),
            &raw("A1"),
            100.0,
            &raw("B1"),
            200.0,
            10,
        );
        let mapped = scaled[&1].map(50.0, 40.0).unwrap();
        assert!((mapped[0] - 54.0).abs() < 1e-5);
        assert!((mapped[1] - 38.0).abs() < 1e-5);
    }
}
