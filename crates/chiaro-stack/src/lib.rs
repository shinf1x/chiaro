//! Motion-aware stacking of repeated RAW frames from one physical L16 camera.
//!
//! Frames are corrected independently, a sharp reference is selected, and
//! every other frame is aligned to it. Fusion happens in the CFA domain. A
//! noise-normalised robust weight rejects motion, occlusion, and alignment
//! failures per pixel; rejected areas therefore fall back to the reference
//! instead of producing ghosts.

pub mod fusion;

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result, bail};
use chiaro::lri::{
    MotionSequence, NoiseChannelModel, NoiseModel, RawCamera, RawFrame, SensorNoiseProfile,
    SensorPattern, decode_raw_frame, parse_frame_layout, sensor_characterization_type,
};
use chiaro_fusion::{
    align::{AlignInput, AlignOptions, AlignmentReport, AlignmentSeed, Warp, align_module_seeded},
    image::{Mosaic, Plane},
    math::{self, Mat3},
};
use chiaro_hotpixel_core::{
    cleanup::{CleanupCameraProfile, CleanupDiagnostics},
    demosaic::{DemosaicMethod, demosaic},
    highlight::{
        HighlightRecovery, HighlightRecoveryReport, HighlightRecoveryState,
        recover_bayer_highlights,
    },
    parallel::map_row_bands,
    pipeline::{CleanupStage, FramePipeline},
    thermal::ThermalProfile,
    universal_hotpixel::UniversalHotpixelProfile,
};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct StackOptions {
    pub camera: String,
    pub align: AlignOptions,
    /// Robust rejection cutoff measured in predicted standard deviations.
    pub motion_sigma: f32,
    /// Factory severity map in decoded stream order. An all-zero map is used
    /// when omitted, leaving factory hot-pixel correction inactive.
    pub severity_map: Option<Vec<u8>>,
    /// Whether a `.chiaro-cleanup` archive was supplied for this run. This is
    /// kept separately so a profile without this selected camera is reported
    /// as `NotCalibrated` instead of looking disabled.
    pub cleanup_profile_supplied: bool,
    /// Camera entry loaded once from the validated cleanup archive.
    pub cleanup_profile: Option<CleanupCameraProfile>,
    /// Preloaded bundled models. Night fusion shares these across physical
    /// camera stacks so they are decoded only once per processing run.
    pub universal_hotpixel_model: Option<Arc<UniversalHotpixelProfile>>,
    pub thermal_model: Option<Arc<ThermalProfile>>,
    /// Force every physical module to use the same temporal reference.
    pub reference_frame: Option<u64>,
    /// Use the ordered row-indexed gyroscope packets as a soft rotation seed.
    pub gyro_seed: bool,
    /// Calibrated focal length for the selected module, when available.
    pub focal_px: Option<f64>,
    /// Refined rig-motion seeds from another module, keyed by temporal frame.
    pub motion_seeds: HashMap<u64, Warp>,
    /// Device-matched noise profiles supplied by calibration overlays. The
    /// capture's embedded points retain priority; these fill missing sensors
    /// or gain entries.
    pub noise_profiles: HashMap<u64, SensorNoiseProfile>,
    pub demosaic: DemosaicMethod,
    /// RAW-domain reconstruction applied to the merged Bayer mosaic before
    /// demosaicing. Multi-camera mode performs its spatial portion here; the
    /// all-module fusion stage may subsequently add aligned donor samples.
    pub highlight_recovery: HighlightRecovery,
    pub threads: usize,
}

impl Default for StackOptions {
    fn default() -> Self {
        Self {
            camera: "A1".to_owned(),
            align: AlignOptions {
                min_inlier_ratio: 0.30,
                ..AlignOptions::default()
            },
            motion_sigma: 4.0,
            severity_map: None,
            cleanup_profile_supplied: false,
            cleanup_profile: None,
            universal_hotpixel_model: None,
            thermal_model: None,
            reference_frame: None,
            gyro_seed: true,
            focal_px: None,
            motion_seeds: HashMap::new(),
            noise_profiles: HashMap::new(),
            demosaic: DemosaicMethod::Lmmse,
            highlight_recovery: HighlightRecovery::MultiscaleBayer,
            threads: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameReport {
    pub frame_index: u64,
    pub sharpness: f32,
    pub noise_model_gain: Option<u32>,
    pub is_reference: bool,
    pub accepted: bool,
    pub alignment: AlignmentReport,
    pub cleanup: CleanupDiagnostics,
}

#[derive(Clone, Debug, Serialize)]
pub struct StackReport {
    pub camera: String,
    pub dimensions: [usize; 2],
    pub input_frames: usize,
    pub accepted_frames: usize,
    pub reference_frame: u64,
    /// Cleanup result for the selected reference frame. Per-frame values
    /// follow in `frames` when temperatures or exposure settings differ.
    pub cleanup: CleanupDiagnostics,
    pub cleanup_frames_applied: usize,
    pub motion_sigma: f32,
    pub noise_model_gain: Option<u32>,
    /// Predicted variance around 5% linear signal after temporal averaging.
    pub dark_noise_variance: Option<f32>,
    pub mean_effective_frames: f32,
    pub fallback_fraction: f32,
    pub imu_sequences: usize,
    pub gyro_seeded_frames: usize,
    pub highlight: HighlightRecoveryReport,
    pub reference_highlight: HighlightRecoveryReport,
    pub frames: Vec<FrameReport>,
}

pub struct StackResult {
    pub width: usize,
    pub height: usize,
    /// Linear, black-subtracted, normalised RGB16 in calibration orientation.
    pub rgb16: Vec<u16>,
    /// The selected single reference frame, in the same representation.
    pub reference_rgb16: Vec<u16>,
    /// Effective contributing-frame count, scaled so one frame is 16384.
    pub effective_count: Vec<u16>,
    pub report: StackReport,
}

pub struct MosaicStackResult {
    pub camera: RawCamera,
    /// Normalised CFA/mono mosaic in calibration-raster orientation.
    pub mosaic16: Vec<u16>,
    pub reference_mosaic16: Vec<u16>,
    pub highlight: HighlightRecoveryState,
    pub reference_highlight: HighlightRecoveryState,
    pub effective_count: Vec<u16>,
    pub temporal_warps: HashMap<u64, Warp>,
    pub report: StackReport,
}

struct PreparedFrame {
    frame: RawFrame,
    mosaic: Mosaic,
    luminance: Plane,
    sharpness: f32,
    noise_model: Option<NoiseModel>,
    cleanup: CleanupDiagnostics,
}

struct AlignedFrame {
    frame_index: u64,
    warp: Warp,
    gain: f32,
    accepted: bool,
    report: AlignmentReport,
}

pub fn stack_burst(data: &[u8], options: &StackOptions) -> Result<StackResult> {
    let stacked = stack_mosaic_burst(data, options)?;
    let width = stacked.camera.width;
    let height = stacked.camera.height;
    let pattern = stacked.camera.pattern;
    let (rgb16, reference_rgb16) = if pattern == SensorPattern::Mono {
        (
            gray_to_rgb(&stacked.mosaic16),
            gray_to_rgb(&stacked.reference_mosaic16),
        )
    } else {
        (
            demosaic(
                &stacked.mosaic16,
                width,
                height,
                pattern,
                options.demosaic,
                options.threads,
            )?,
            demosaic(
                &stacked.reference_mosaic16,
                width,
                height,
                pattern,
                options.demosaic,
                options.threads,
            )?,
        )
    };
    Ok(StackResult {
        width,
        height,
        rgb16,
        reference_rgb16,
        effective_count: stacked.effective_count,
        report: stacked.report,
    })
}

pub fn stack_mosaic_burst(data: &[u8], options: &StackOptions) -> Result<MosaicStackResult> {
    if options.motion_sigma <= 0.0 || !options.motion_sigma.is_finite() {
        bail!("motion sigma must be finite and greater than zero");
    }
    let mut layout =
        parse_frame_layout(data, &HashMap::new()).map_err(|error| anyhow::anyhow!(error))?;
    for (sensor, profile) in &options.noise_profiles {
        match layout.sensor_profiles.entry(*sensor) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().merge_missing(profile);
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(profile.clone());
            }
        }
    }
    let motion_sequences = layout.motion_sequences.clone();
    let mut selected = layout
        .frames
        .into_iter()
        .filter(|frame| frame.camera.name.eq_ignore_ascii_case(&options.camera))
        .collect::<Vec<_>>();
    selected.sort_by_key(|frame| frame.frame_index);
    if selected.len() < 2 {
        bail!(
            "camera {} has {} temporal frame(s); stacking needs at least two",
            options.camera,
            selected.len()
        );
    }
    let first = &selected[0].camera;
    if selected.iter().any(|frame| {
        frame.camera.width != first.width
            || frame.camera.height != first.height
            || frame.camera.pattern != first.pattern
    }) {
        bail!("temporal frames do not share dimensions and sensor pattern");
    }
    let sample_count = first
        .width
        .checked_mul(first.height)
        .context("frame dimensions overflow")?;
    let severity = options.severity_map.as_deref().unwrap_or(&[]);
    if options.cleanup_profile_supplied && options.severity_map.is_none() {
        bail!("a cleanup profile requires the corresponding hotpixel.rec factory map");
    }
    if !severity.is_empty() && severity.len() != sample_count {
        bail!("hotpixel severity map dimensions do not match the selected camera");
    }
    let zero_severity = severity.is_empty().then(|| vec![0; sample_count]);
    let severity = zero_severity.as_deref().unwrap_or(severity);
    // Match the established single-frame fusion pipeline: supplying the
    // camera's factory map also enables the bundled sensor-family corrections.
    // Long, high-gain night frames are precisely where uncorrected corner glow
    // can turn a global colour transform into a strong magenta cast.
    let universal = if options.severity_map.is_some() {
        Some(match &options.universal_hotpixel_model {
            Some(profile) => Arc::clone(profile),
            None => Arc::new(UniversalHotpixelProfile::bundled()?),
        })
    } else {
        None
    };
    let thermal = if options.severity_map.is_some() {
        Some(match &options.thermal_model {
            Some(profile) => Arc::clone(profile),
            None => Arc::new(ThermalProfile::bundled()?),
        })
    } else {
        None
    };
    let pipeline = FramePipeline {
        universal_hotpixel: universal.as_deref(),
        thermal: thermal.as_deref(),
        cleanup: CleanupStage::from_loaded(
            options.cleanup_profile_supplied,
            options.cleanup_profile.as_ref(),
        ),
        threads: options.threads,
        ..FramePipeline::default()
    };

    let mut prepared = Vec::with_capacity(selected.len());
    for frame in selected {
        let noise_model = layout
            .sensor_profiles
            .get(&frame.sensor_type)
            .or_else(|| {
                layout
                    .sensor_profiles
                    .get(&sensor_characterization_type(frame.sensor_type))
            })
            .and_then(|profile| {
                profile.model_for_gain(frame.camera.analog_gain, frame.camera.digital_gain)
            });
        let raw = decode_raw_frame(data, &frame).map_err(|error| anyhow::anyhow!(error))?;
        let corrected = pipeline
            .correct_raw(&frame.camera, raw, severity)
            .with_context(|| {
                format!("correct {} frame {}", frame.camera.name, frame.frame_index)
            })?;
        let cleanup = CleanupDiagnostics::new(
            options.cleanup_profile_supplied,
            options.cleanup_profile.is_some(),
            corrected.cleanup,
        );
        let mosaic = Mosaic::from_stream_q6(
            corrected.samples_q6,
            frame.camera.width,
            frame.camera.height,
            frame.camera.pattern,
            frame.camera.black_level,
            frame.camera.white_level,
        );
        let luminance = mosaic.luminance_half();
        let sharpness = laplacian_energy(&luminance);
        prepared.push(PreparedFrame {
            frame,
            mosaic,
            luminance,
            sharpness,
            noise_model,
            cleanup,
        });
    }
    let reference_index = match options.reference_frame {
        Some(wanted) => prepared
            .iter()
            .position(|frame| frame.frame.frame_index == wanted)
            .with_context(|| format!("{} has no temporal frame {wanted}", options.camera))?,
        None => prepared
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.sharpness.total_cmp(&right.sharpness))
            .map(|(index, _)| index)
            .context("no frames prepared")?,
    };
    let reference = &prepared[reference_index];
    let names = prepared
        .iter()
        .map(|frame| format!("{}#{}", frame.frame.camera.name, frame.frame.frame_index))
        .collect::<Vec<_>>();
    let reference_input = AlignInput {
        name: &names[reference_index],
        luminance: &reference.luminance,
        width: reference.frame.camera.width,
        height: reference.frame.camera.height,
        camera: None,
        nominal_focal_px: options
            .focal_px
            .unwrap_or(reference.frame.camera.width as f64),
    };
    let mut alignments = Vec::with_capacity(prepared.len());
    for (index, frame) in prepared.iter().enumerate() {
        let target = AlignInput {
            name: &names[index],
            luminance: &frame.luminance,
            width: frame.frame.camera.width,
            height: frame.frame.camera.height,
            camera: None,
            nominal_focal_px: options.focal_px.unwrap_or(frame.frame.camera.width as f64),
        };
        let (seed, seed_name) =
            if let Some(seed) = options.motion_seeds.get(&frame.frame.frame_index) {
                (Some(seed.clone()), "shared rig motion")
            } else {
                (
                    options
                        .gyro_seed
                        .then(|| {
                            gyro_seed_warp(
                                &motion_sequences,
                                reference_index,
                                index,
                                GyroSeedGeometry {
                                    width: frame.frame.camera.width,
                                    height: frame.frame.camera.height,
                                    focal_px: options
                                        .focal_px
                                        .unwrap_or(frame.frame.camera.width as f64),
                                    exposure_ns: frame.frame.camera.exposure_ns,
                                    grid_step: options.align.grid_step,
                                },
                            )
                        })
                        .flatten(),
                    "IMU rotation",
                )
            };
        let mut alignment = align_module_seeded(
            &reference_input,
            &target,
            &options.align,
            seed.as_ref().map(|warp| AlignmentSeed {
                warp,
                name: seed_name,
            }),
        )
        .with_context(|| format!("align temporal frame {}", frame.frame.frame_index))?;
        // Metadata conventions vary between firmware versions. A rejected IMU
        // seed is never allowed to make the image path worse: retry from the
        // established correlation-only initialisation.
        if seed.is_some() && !alignment.report.accepted {
            alignment = align_module_seeded(&reference_input, &target, &options.align, None)
                .with_context(|| {
                    format!(
                        "retry temporal frame {} without IMU",
                        frame.frame.frame_index
                    )
                })?;
        }
        alignments.push(AlignedFrame {
            frame_index: frame.frame.frame_index,
            warp: alignment.warp,
            gain: alignment.gain,
            accepted: alignment.report.accepted,
            report: alignment.report,
        });
    }

    let noise_model = reference.noise_model;
    let width = reference.frame.camera.width;
    let height = reference.frame.camera.height;
    let pattern = reference.frame.camera.pattern;
    let sigma2 = options.motion_sigma * options.motion_sigma;
    let bands = map_row_bands(height, options.threads, 2, |rows| {
        let mut merged = Vec::with_capacity(rows.len() * width);
        let mut counts = Vec::with_capacity(rows.len() * width);
        for y in rows {
            for x in 0..width {
                let channel = if pattern == SensorPattern::Mono {
                    0
                } else {
                    pattern.color_at(y, x)
                };
                let reference_value = reference
                    .mosaic
                    .sample_channel(x as f32, y as f32, channel)
                    .unwrap_or(0.0);
                let mut sum = reference_value;
                let mut weight_sum = 1.0f32;
                for (frame, alignment) in prepared.iter().zip(&alignments) {
                    if frame.frame.frame_index == reference.frame.frame_index || !alignment.accepted
                    {
                        continue;
                    }
                    let Some(position) = alignment.warp.map(x as f32, y as f32) else {
                        continue;
                    };
                    let Some(target_value) =
                        frame
                            .mosaic
                            .sample_channel(position[0], position[1], channel)
                    else {
                        continue;
                    };
                    let value = target_value * alignment.gain;
                    let variance = pair_variance(
                        reference_value,
                        noise_channel(noise_model, pattern, channel),
                        raw_code_range(&reference.frame.camera),
                        target_value,
                        noise_channel(frame.noise_model, pattern, channel),
                        raw_code_range(&frame.frame.camera),
                        alignment.gain,
                    );
                    let residual = (value - reference_value).powi(2);
                    let ratio = residual / (sigma2 * variance).max(1e-9);
                    let weight = (1.0 - ratio).clamp(0.0, 1.0).powi(2);
                    sum += value * weight;
                    weight_sum += weight;
                }
                merged.push(((sum / weight_sum).clamp(0.0, 1.0) * 65535.0).round() as u16);
                counts.push((weight_sum.clamp(0.0, 4.0) * 16384.0).round().min(65535.0) as u16);
            }
        }
        (merged, counts)
    });
    let mut merged = Vec::with_capacity(sample_count);
    let mut effective_count = Vec::with_capacity(sample_count);
    for (samples, counts) in bands {
        merged.extend(samples);
        effective_count.extend(counts);
    }
    let mut reference_mosaic = reference_mosaic_u16(reference);
    let highlight = recover_bayer_highlights(
        &mut merged,
        width,
        height,
        pattern,
        0.0,
        65535.0,
        options.highlight_recovery,
    )?;
    let reference_highlight = recover_bayer_highlights(
        &mut reference_mosaic,
        width,
        height,
        pattern,
        0.0,
        65535.0,
        options.highlight_recovery,
    )?;
    let highlight_report = highlight.report.clone();
    let reference_highlight_report = reference_highlight.report.clone();
    let mean_effective_frames = effective_count
        .iter()
        .map(|count| *count as f64 / 16384.0)
        .sum::<f64>()
        / effective_count.len() as f64;
    let dark_channel = noise_channel(noise_model, pattern, 1);
    let dark_noise_variance = dark_channel.map(|channel| {
        single_variance(0.05, Some(channel), raw_code_range(&reference.frame.camera))
            / (mean_effective_frames as f32).max(1.0)
    });
    let fallback_fraction = effective_count
        .iter()
        .filter(|count| **count < 18432)
        .count() as f32
        / effective_count.len() as f32;
    let accepted_frames = alignments.iter().filter(|frame| frame.accepted).count();
    let gyro_seeded_frames = alignments
        .iter()
        .filter(|frame| frame.report.initialised_from == "IMU rotation")
        .count();
    let temporal_warps = alignments
        .iter()
        .map(|alignment| (alignment.frame_index, alignment.warp.clone()))
        .collect();
    let cleanup = reference.cleanup.clone();
    let cleanup_frames_applied = prepared
        .iter()
        .filter(|frame| frame.cleanup.correction.applied)
        .count();
    let frames = prepared
        .iter()
        .zip(alignments)
        .map(|(frame, alignment)| FrameReport {
            frame_index: alignment.frame_index,
            sharpness: frame.sharpness,
            noise_model_gain: frame.noise_model.map(|model| model.gain),
            is_reference: frame.frame.frame_index == reference.frame.frame_index,
            accepted: alignment.accepted,
            alignment: alignment.report,
            cleanup: frame.cleanup.clone(),
        })
        .collect();
    Ok(MosaicStackResult {
        camera: reference.frame.camera.clone(),
        mosaic16: merged,
        reference_mosaic16: reference_mosaic,
        highlight,
        reference_highlight,
        effective_count,
        temporal_warps,
        report: StackReport {
            camera: reference.frame.camera.name.clone(),
            dimensions: [width, height],
            input_frames: prepared.len(),
            accepted_frames,
            reference_frame: reference.frame.frame_index,
            cleanup,
            cleanup_frames_applied,
            motion_sigma: options.motion_sigma,
            noise_model_gain: noise_model.map(|model| model.gain),
            dark_noise_variance,
            mean_effective_frames: mean_effective_frames as f32,
            fallback_fraction,
            imu_sequences: motion_sequences.len(),
            gyro_seeded_frames,
            highlight: highlight_report,
            reference_highlight: reference_highlight_report,
            frames,
        },
    })
}

#[derive(Clone, Copy)]
struct GyroSeedGeometry {
    width: usize,
    height: usize,
    focal_px: f64,
    exposure_ns: u64,
    grid_step: usize,
}

fn gyro_seed_warp(
    sequences: &[MotionSequence],
    reference_index: usize,
    target_index: usize,
    geometry: GyroSeedGeometry,
) -> Option<Warp> {
    if reference_index >= sequences.len()
        || target_index >= sequences.len()
        || sequences[reference_index].gyroscope.is_empty()
        || sequences[target_index].gyroscope.is_empty()
    {
        return None;
    }
    let exposure_seconds = geometry.exposure_ns as f64 * 1e-9;
    if exposure_seconds <= 0.0 || !exposure_seconds.is_finite() {
        return None;
    }
    let centre = [
        (geometry.width as f64 - 1.0) * 0.5,
        (geometry.height as f64 - 1.0) * 0.5,
    ];
    let k: Mat3 = [
        [geometry.focal_px, 0.0, centre[0]],
        [0.0, geometry.focal_px, centre[1]],
        [0.0, 0.0, 1.0],
    ];
    let k_inverse = math::inverse(&k)?;
    Some(Warp::from_fn(
        geometry.width,
        geometry.height,
        geometry.grid_step,
        |point| {
            let fraction = (point[1] / geometry.height.max(1) as f64).clamp(0.0, 1.0);
            let reference_pose = gyro_pose_at(
                sequences,
                reference_index,
                fraction,
                geometry.height,
                exposure_seconds,
            )?;
            let target_pose = gyro_pose_at(
                sequences,
                target_index,
                fraction,
                geometry.height,
                exposure_seconds,
            )?;
            // The IMU measures physical camera rotation. Static-scene rays move
            // through the inverse rotation in camera coordinates.
            let relative = [
                reference_pose[0] - target_pose[0],
                reference_pose[1] - target_pose[1],
                reference_pose[2] - target_pose[2],
            ];
            let rotation = math::rotation_from_axis_angle(relative);
            let homography = math::mul(&math::mul(&k, &rotation), &k_inverse);
            math::apply_homography(&homography, point)
        },
    ))
}

fn gyro_pose_at(
    sequences: &[MotionSequence],
    frame: usize,
    row_fraction: f64,
    sensor_height: usize,
    exposure_seconds: f64,
) -> Option<[f64; 3]> {
    let mut pose = [0.0f64; 3];
    for sequence in sequences.iter().take(frame) {
        let delta = integrate_gyro(sequence, 1.0, sensor_height, exposure_seconds)?;
        for axis in 0..3 {
            pose[axis] += delta[axis];
        }
    }
    let delta = integrate_gyro(
        sequences.get(frame)?,
        row_fraction,
        sensor_height,
        exposure_seconds,
    )?;
    for axis in 0..3 {
        pose[axis] += delta[axis];
    }
    Some(pose)
}

fn integrate_gyro(
    sequence: &MotionSequence,
    fraction: f64,
    sensor_height: usize,
    exposure_seconds: f64,
) -> Option<[f64; 3]> {
    let samples = &sequence.gyroscope;
    let first = samples.first()?;
    let end = fraction.clamp(0.0, 1.0);
    if end == 0.0 {
        return Some([0.0; 3]);
    }
    let row_scale = sensor_height.max(1) as f64;
    let mut previous_t = 0.0;
    let mut previous = first.vector.map(f64::from);
    let mut area = [0.0f64; 3];
    for sample in samples {
        let t = (sample.row as f64 / row_scale).clamp(previous_t, 1.0);
        let next_t = t.min(end);
        if next_t > previous_t {
            let current = sample.vector.map(f64::from);
            for axis in 0..3 {
                area[axis] += 0.5 * (previous[axis] + current[axis]) * (next_t - previous_t);
            }
            previous = current;
            previous_t = next_t;
        }
        if t >= end {
            break;
        }
    }
    if previous_t < end {
        for axis in 0..3 {
            area[axis] += previous[axis] * (end - previous_t);
        }
    }
    Some(area.map(|value| value * exposure_seconds))
}

fn laplacian_energy(plane: &Plane) -> f32 {
    if plane.width < 3 || plane.height < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in (1..plane.height - 1).step_by(2) {
        for x in (1..plane.width - 1).step_by(2) {
            let centre = plane.at(x, y);
            let laplacian = 4.0 * centre
                - plane.at(x - 1, y)
                - plane.at(x + 1, y)
                - plane.at(x, y - 1)
                - plane.at(x, y + 1);
            sum += f64::from(laplacian * laplacian);
            count += 1;
        }
    }
    (sum / count.max(1) as f64) as f32
}

fn noise_channel(
    model: Option<NoiseModel>,
    pattern: SensorPattern,
    channel: usize,
) -> Option<NoiseChannelModel> {
    model.map(|model| {
        if pattern == SensorPattern::Mono {
            model.panchromatic
        } else {
            [model.red, model.green, model.blue][channel]
        }
    })
}

fn raw_code_range(camera: &RawCamera) -> f32 {
    (camera.white_level - camera.black_level).max(1.0)
}

#[allow(clippy::too_many_arguments)]
fn pair_variance(
    reference_signal: f32,
    reference_model: Option<NoiseChannelModel>,
    reference_code_range: f32,
    target_signal: f32,
    target_model: Option<NoiseChannelModel>,
    target_code_range: f32,
    target_gain: f32,
) -> f32 {
    // Two independently noisy observations are compared. The target is
    // photometrically scaled into reference units, so its variance scales by
    // the square of the same gain. Quantisation remains a conservative floor,
    // especially for 8-bit Bayer-JPEG burst frames.
    single_variance(reference_signal, reference_model, reference_code_range)
        + target_gain.powi(2) * single_variance(target_signal, target_model, target_code_range)
}

fn single_variance(signal: f32, model: Option<NoiseChannelModel>, code_range: f32) -> f32 {
    let quantisation = 1.0 / code_range.max(1.0);
    model
        .map(|model| model.a * signal + model.b)
        .unwrap_or(quantisation * quantisation)
        .max(quantisation * quantisation)
}

fn reference_mosaic_u16(frame: &PreparedFrame) -> Vec<u16> {
    let pattern = frame.frame.camera.pattern;
    (0..frame.frame.camera.height)
        .flat_map(|y| {
            (0..frame.frame.camera.width).map(move |x| {
                let channel = if pattern == SensorPattern::Mono {
                    0
                } else {
                    pattern.color_at(y, x)
                };
                (frame
                    .mosaic
                    .sample_channel(x as f32, y as f32, channel)
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0)
                    * 65535.0)
                    .round() as u16
            })
        })
        .collect()
}

fn gray_to_rgb(gray: &[u16]) -> Vec<u16> {
    gray.iter().flat_map(|value| [*value; 3]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chiaro::mock::{MockCamera, MockCapture};
    use chiaro_hotpixel_core::{
        cleanup::{BuildCleanupProfileOptions, CleanupProfile, build_cleanup_profile},
        hotpixel::{HotpixelRec, write_hotpixel_rec},
    };
    use std::{collections::HashSet, fs, path::Path};

    fn row_biased_camera(temperature: i32, frame_delta: u16) -> MockCamera {
        let mut camera = MockCamera::gradient("A1", 64, 48, SensorPattern::Bggr, 80, 180);
        camera.sensor_temperature_c = Some(temperature);
        for (index, sample) in camera.samples.iter_mut().enumerate() {
            let row_bias = if index / camera.width == 24 { 48 } else { 0 };
            *sample = sample.saturating_add(row_bias + frame_delta).min(1023);
        }
        camera
    }

    fn build_cleanup_fixture(root: &Path) -> (HotpixelRec, CleanupProfile) {
        let rec_path = root.join("hotpixel.rec");
        write_hotpixel_rec(
            &rec_path,
            &(0..16)
                .map(|_| (64, 48, vec![0; 64 * 48]))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let rec = HotpixelRec::open(&rec_path).unwrap();
        let training = root.join("training");
        fs::create_dir(&training).unwrap();
        for (index, temperature) in [20, 30, 40].into_iter().enumerate() {
            let capture = MockCapture {
                cameras: vec![row_biased_camera(temperature, 0)],
                reference_camera: Some("A1".to_owned()),
                ..MockCapture::default()
            };
            fs::write(
                training.join(format!("dark-{index}.lri")),
                capture.encode().unwrap(),
            )
            .unwrap();
        }
        let profile_path = root.join("camera.chiaro-cleanup");
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
        let profile = CleanupProfile::open(profile_path, &rec).unwrap();
        (rec, profile)
    }

    #[test]
    fn temporal_stack_defaults_to_lmmse() {
        assert_eq!(StackOptions::default().demosaic, DemosaicMethod::Lmmse);
        assert_eq!(
            StackOptions::default().highlight_recovery,
            HighlightRecovery::MultiscaleBayer
        );
    }

    #[test]
    fn noise_variance_has_a_quantisation_floor() {
        let model = NoiseChannelModel { a: 0.01, b: -1.0 };
        let actual = pair_variance(0.5, Some(model), 255.0, 0.5, Some(model), 255.0, 1.0);
        let expected = 2.0 / 255.0f32.powi(2);
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn a_four_sigma_outlier_gets_zero_weight() {
        let variance = pair_variance(0.2, None, 255.0, 0.2, None, 255.0, 1.0);
        let residual = 16.0 * variance;
        let ratio = residual / (16.0 * variance);
        assert_eq!((1.0 - ratio).clamp(0.0, 1.0).powi(2), 0.0);
    }

    #[test]
    fn target_noise_variance_follows_photometric_gain() {
        let model = NoiseChannelModel { a: 0.01, b: 0.0 };
        let actual = pair_variance(0.2, Some(model), 1023.0, 0.1, Some(model), 1023.0, 2.0);
        assert!((actual - 0.006).abs() < 1e-8);
    }

    #[test]
    fn monochrome_uses_the_panchromatic_noise_channel() {
        let channel = |a| NoiseChannelModel { a, b: 0.0 };
        let model = NoiseModel {
            gain: 100,
            threshold: 0.0,
            scale: 1.0,
            red: channel(1.0),
            green: channel(2.0),
            blue: channel(3.0),
            panchromatic: channel(4.0),
        };
        assert_eq!(
            noise_channel(Some(model), SensorPattern::Mono, 0),
            Some(channel(4.0))
        );
        assert_eq!(
            noise_channel(Some(model), SensorPattern::Rggb, 0),
            Some(channel(1.0))
        );
    }

    #[test]
    fn constant_gyro_integrates_over_the_requested_rows() {
        let sequence = MotionSequence {
            declared_frame_index: 0,
            accelerometer: Vec::new(),
            gyroscope: vec![
                chiaro::lri::MotionSample {
                    row: 0,
                    vector: [1.0, 2.0, 3.0],
                },
                chiaro::lri::MotionSample {
                    row: 100,
                    vector: [1.0, 2.0, 3.0],
                },
            ],
        };
        let integrated = integrate_gyro(&sequence, 0.5, 100, 0.02).unwrap();
        assert_eq!(integrated, [0.01, 0.02, 0.03]);
    }

    #[test]
    fn mock_capture_can_represent_repeated_physical_frames() {
        let camera = MockCamera::gradient("A1", 64, 48, SensorPattern::Bggr, 64, 700);
        let capture = MockCapture {
            cameras: vec![camera.clone(), camera.clone(), camera],
            reference_camera: Some("A1".to_owned()),
            ..MockCapture::default()
        }
        .encode()
        .unwrap();
        let layout = parse_frame_layout(&capture, &HashMap::new()).unwrap();
        assert_eq!(
            layout
                .frames
                .iter()
                .filter(|frame| frame.camera.name == "A1")
                .count(),
            3
        );
    }

    #[test]
    fn cleanup_is_applied_to_every_temporal_frame() {
        let temporary = tempfile::tempdir().unwrap();
        let (rec, profile) = build_cleanup_fixture(temporary.path());
        let capture = MockCapture {
            cameras: vec![
                row_biased_camera(30, 0),
                row_biased_camera(30, 1),
                row_biased_camera(30, 2),
            ],
            reference_camera: Some("A1".to_owned()),
            ..MockCapture::default()
        }
        .encode()
        .unwrap();
        let layout = parse_frame_layout(&capture, &HashMap::new()).unwrap();
        let camera = &layout.frames[0].camera;
        let options = StackOptions {
            camera: "A1".to_owned(),
            severity_map: Some(
                rec.load_rotated_map(camera.id, camera.width, camera.height)
                    .unwrap(),
            ),
            cleanup_profile_supplied: true,
            cleanup_profile: profile.load_camera(camera).unwrap(),
            highlight_recovery: HighlightRecovery::None,
            ..StackOptions::default()
        };

        let result = stack_mosaic_burst(&capture, &options).unwrap();
        assert_eq!(result.report.frames.len(), 3);
        assert!(result.report.frames.iter().all(|frame| {
            frame.cleanup.profile_supplied
                && frame.cleanup.profile_available
                && frame.cleanup.correction.applied
                && frame.cleanup.correction.mean_absolute_change > 0.0
        }));
    }
}
