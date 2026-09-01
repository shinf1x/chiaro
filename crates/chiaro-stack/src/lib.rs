//! Motion-aware stacking of repeated RAW frames from one physical L16 camera.
//!
//! Frames are corrected independently, a sharp reference is selected, and
//! every other frame is aligned to it. Fusion happens in the CFA domain. A
//! noise-normalised robust weight rejects motion, occlusion, and alignment
//! failures per pixel; rejected areas therefore fall back to the reference
//! instead of producing ghosts.

pub mod fusion;

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use chiaro::lri::{
    MotionSequence, NoiseChannelModel, RawCamera, RawFrame, SensorPattern, decode_raw_frame,
    parse_frame_layout,
};
use chiaro_fusion::{
    align::{AlignInput, AlignOptions, AlignmentReport, AlignmentSeed, Warp, align_module_seeded},
    image::{Mosaic, Plane},
    math::{self, Mat3},
};
use chiaro_hotpixel_core::{
    demosaic::{DemosaicMethod, demosaic},
    parallel::map_row_bands,
    pipeline::FramePipeline,
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
    /// Force every physical module to use the same temporal reference.
    pub reference_frame: Option<u64>,
    /// Use the ordered row-indexed gyroscope packets as a soft rotation seed.
    pub gyro_seed: bool,
    /// Calibrated focal length for the selected module, when available.
    pub focal_px: Option<f64>,
    /// Refined rig-motion seeds from another module, keyed by temporal frame.
    pub motion_seeds: HashMap<u64, Warp>,
    pub demosaic: DemosaicMethod,
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
            reference_frame: None,
            gyro_seed: true,
            focal_px: None,
            motion_seeds: HashMap::new(),
            demosaic: DemosaicMethod::Lmmse,
            threads: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameReport {
    pub frame_index: u64,
    pub sharpness: f32,
    pub is_reference: bool,
    pub accepted: bool,
    pub alignment: AlignmentReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct StackReport {
    pub camera: String,
    pub dimensions: [usize; 2],
    pub input_frames: usize,
    pub accepted_frames: usize,
    pub reference_frame: u64,
    pub motion_sigma: f32,
    pub noise_model_gain: Option<u32>,
    /// Predicted variance around 5% linear signal after temporal averaging.
    pub dark_noise_variance: Option<f32>,
    pub mean_effective_frames: f32,
    pub fallback_fraction: f32,
    pub imu_sequences: usize,
    pub gyro_seeded_frames: usize,
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
    pub effective_count: Vec<u16>,
    pub temporal_warps: HashMap<u64, Warp>,
    pub report: StackReport,
}

struct PreparedFrame {
    frame: RawFrame,
    mosaic: Mosaic,
    luminance: Plane,
    sharpness: f32,
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
    let layout =
        parse_frame_layout(data, &HashMap::new()).map_err(|error| anyhow::anyhow!(error))?;
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
    if !severity.is_empty() && severity.len() != sample_count {
        bail!("hotpixel severity map dimensions do not match the selected camera");
    }
    let zero_severity = severity.is_empty().then(|| vec![0; sample_count]);
    let severity = zero_severity.as_deref().unwrap_or(severity);
    // Match the established single-frame fusion pipeline: supplying the
    // camera's factory map also enables the bundled sensor-family corrections.
    // Long, high-gain night frames are precisely where uncorrected corner glow
    // can turn a global colour transform into a strong magenta cast.
    let universal = options
        .severity_map
        .is_some()
        .then(UniversalHotpixelProfile::bundled)
        .transpose()?;
    let thermal = options
        .severity_map
        .is_some()
        .then(ThermalProfile::bundled)
        .transpose()?;
    let pipeline = FramePipeline {
        universal_hotpixel: universal.as_ref(),
        thermal: thermal.as_ref(),
        threads: options.threads,
        ..FramePipeline::default()
    };

    let mut prepared = Vec::with_capacity(selected.len());
    for frame in selected {
        let raw = decode_raw_frame(data, &frame).map_err(|error| anyhow::anyhow!(error))?;
        let corrected = pipeline
            .correct_raw(&frame.camera, raw, severity)
            .with_context(|| {
                format!("correct {} frame {}", frame.camera.name, frame.frame_index)
            })?;
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

    let noise_model = layout
        .sensor_profiles
        .get(&reference.frame.sensor_type)
        .and_then(|profile| {
            profile.nearest_model(
                reference.frame.camera.analog_gain,
                reference.frame.camera.digital_gain,
            )
        });
    let channel_noise = noise_model.map(|model| [model.red, model.green, model.blue]);
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
                    let Some(value) = frame
                        .mosaic
                        .sample_channel(position[0], position[1], channel)
                        .map(|value| value * alignment.gain)
                    else {
                        continue;
                    };
                    let variance = pair_variance(
                        reference_value.max(value),
                        channel_noise.map(|models| models[channel]),
                        frame.frame.camera.white_level,
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
    let reference_mosaic = reference_mosaic_u16(reference);
    let mean_effective_frames = effective_count
        .iter()
        .map(|count| *count as f64 / 16384.0)
        .sum::<f64>()
        / effective_count.len() as f64;
    let dark_channel = noise_model.map(|model| {
        if pattern == SensorPattern::Mono {
            model.panchromatic
        } else {
            model.green
        }
    });
    let dark_noise_variance = Some(
        single_variance(0.05, dark_channel, reference.frame.camera.white_level)
            / (mean_effective_frames as f32).max(1.0),
    );
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
    let frames = prepared
        .iter()
        .zip(alignments)
        .map(|(frame, alignment)| FrameReport {
            frame_index: alignment.frame_index,
            sharpness: frame.sharpness,
            is_reference: frame.frame.frame_index == reference.frame.frame_index,
            accepted: alignment.accepted,
            alignment: alignment.report,
        })
        .collect();
    Ok(MosaicStackResult {
        camera: reference.frame.camera.clone(),
        mosaic16: merged,
        reference_mosaic16: reference_mosaic,
        effective_count,
        temporal_warps,
        report: StackReport {
            camera: reference.frame.camera.name.clone(),
            dimensions: [width, height],
            input_frames: prepared.len(),
            accepted_frames,
            reference_frame: reference.frame.frame_index,
            motion_sigma: options.motion_sigma,
            noise_model_gain: noise_model.map(|model| model.gain),
            dark_noise_variance,
            mean_effective_frames: mean_effective_frames as f32,
            fallback_fraction,
            imu_sequences: motion_sequences.len(),
            gyro_seeded_frames,
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

fn pair_variance(signal: f32, model: Option<NoiseChannelModel>, white_level: f32) -> f32 {
    // Two independently noisy observations are compared. Quantisation is a
    // conservative floor, especially important for 8-bit Bayer-JPEG bursts.
    2.0 * single_variance(signal, model, white_level)
}

fn single_variance(signal: f32, model: Option<NoiseChannelModel>, white_level: f32) -> f32 {
    let quantisation = 1.0 / white_level.max(1.0);
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

    #[test]
    fn temporal_stack_defaults_to_lmmse() {
        assert_eq!(StackOptions::default().demosaic, DemosaicMethod::Lmmse);
    }

    #[test]
    fn noise_variance_has_a_quantisation_floor() {
        let model = NoiseChannelModel { a: 0.01, b: -1.0 };
        let actual = pair_variance(0.5, Some(model), 255.0);
        let expected = 2.0 / 255.0f32.powi(2);
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn a_four_sigma_outlier_gets_zero_weight() {
        let variance = pair_variance(0.2, None, 255.0);
        let residual = 16.0 * variance;
        let ratio = residual / (16.0 * variance);
        assert_eq!((1.0 - ratio).clamp(0.0, 1.0).powi(2), 0.0);
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
}
