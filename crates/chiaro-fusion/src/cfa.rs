//! Experimental cross-camera reconstruction from physical CFA observations.
//!
//! This module is deliberately independent of the production selector. It
//! provides a compact observation record and a tileable three-variable robust
//! solve in the common D50 XYZ space. Real-capture held-out validation decides
//! whether the path is worth further production work.

use chiaro::lri::{NoiseChannelModel, NoiseModel};
use serde::Serialize;

use crate::{
    image::{CfaPhase, CorrectedCfaSample},
    math::inverse,
    synth::ModuleColor,
};

const HUBER_SIGMA: f32 = 2.5;
const SOLVER_ITERATIONS: usize = 3;
const MODEL_SIZE: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HighlightProvenance {
    Measured,
    Recovered,
    Unresolved,
}

impl HighlightProvenance {
    pub fn from_confidence(confidence: u8) -> Self {
        match confidence {
            255 => Self::Measured,
            1..=254 => Self::Recovered,
            0 => Self::Unresolved,
        }
    }

    fn weight(self, confidence: u8) -> f32 {
        match self {
            Self::Measured => 1.0,
            Self::Recovered => (f32::from(confidence) / 255.0).powi(2),
            Self::Unresolved => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Visible,
    Occluded,
    Unknown,
}

/// One real sensor measurement projected near an output reconstruction point.
#[derive(Clone, Debug, Serialize)]
pub struct CfaObservation {
    pub camera_index: usize,
    /// Stable physical L16 module id from capture metadata.
    pub camera_id: usize,
    pub sensor_xy: [u16; 2],
    pub output_offset: [f32; 2],
    pub phase: CfaPhase,
    pub value: f32,
    pub noise_variance: f32,
    pub highlight_provenance: HighlightProvenance,
    pub highlight_confidence: u8,
    pub geometry_confidence: f32,
    pub visibility: Visibility,
    /// Row mapping common D50 XYZ to this corrected camera-CFA measurement.
    pub response: [f32; 3],
    /// Compact spatial window weight around the reconstruction point.
    pub spatial_weight: f32,
    /// Prediction made by the production baseline at this observation's
    /// projected location. Contributor diagnostics use this spatially matched
    /// value instead of comparing an affine fit with a constant centre pixel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_prediction: Option<f32>,
    /// Physical sensor sites reused by this corrected observation. These are
    /// omitted from JSON but used to conservatively account for covariance
    /// between neighboring corrected samples.
    #[serde(skip)]
    pub noise_dependencies: [NoiseDependency; 16],
    #[serde(skip)]
    pub noise_dependency_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoiseDependency {
    pub key: u64,
    pub coefficient: f32,
    pub physical_variance: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct JointCfaSolveReport {
    pub observations: usize,
    pub cameras: usize,
    pub phase_mask: u8,
    /// RMS distance of the closest physical sampling position from its
    /// cross-camera centroid, in output pixels.
    pub phase_spread: f32,
    pub iterations: usize,
    pub weighted_residual: f32,
    /// In-sample diagnostic on the observations used by the fit. This is not
    /// independent evidence of output quality.
    pub in_sample_baseline_loss: f32,
    pub in_sample_affine_loss: f32,
    /// Effective rank of the unregularized XYZ response information.
    pub data_rank: usize,
    /// Smallest-to-largest elimination pivot ratio of that information. This
    /// is a conservative conditioning score in `[0, 1]`.
    pub information_confidence: f32,
    /// Rank of the complete unregularized affine design (XYZ plus both
    /// spatial derivatives). A value below nine means the centre estimate can
    /// still depend on ridge priors through an unsupported slope direction.
    pub model_rank: usize,
    pub spatial_rank: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct JointCfaEstimate {
    pub xyz: [f32; 3],
    /// Confidence used by synthesis when blending this estimate with its
    /// production baseline. The algebraic solver itself returns one.
    pub application_weight: f32,
    pub report: JointCfaSolveReport,
    model: [f32; MODEL_SIZE],
    preserve_baseline_luminance: bool,
}

impl JointCfaEstimate {
    /// Evaluate the fitted local affine field at an output-pixel offset. This
    /// is used only by spatially matched held-out diagnostics; rendered output
    /// continues to consume the centre estimate in `xyz`.
    pub fn fitted_xyz_at(&self, offset: [f32; 2]) -> [f32; 3] {
        std::array::from_fn(|channel| {
            self.model[channel]
                + offset[0] * self.model[3 + channel]
                + offset[1] * self.model[6 + channel]
        })
    }

    pub fn apply_over_baseline(
        &mut self,
        baseline_xyz: [f32; 3],
        structure_weight: f32,
        preserve_baseline_luminance: bool,
    ) {
        self.application_weight *= structure_weight.clamp(0.0, 1.0);
        self.preserve_baseline_luminance = preserve_baseline_luminance;
        self.xyz = self.applied_xyz_at([0.0, 0.0], baseline_xyz);
    }

    pub fn applied_xyz_at(&self, offset: [f32; 2], baseline_xyz: [f32; 3]) -> [f32; 3] {
        let mut fitted = self.fitted_xyz_at(offset).map(|value| value.max(0.0));
        if self.preserve_baseline_luminance && fitted[1] > 1.0e-8 {
            let scale = baseline_xyz[1].max(0.0) / fitted[1];
            for value in &mut fitted {
                *value *= scale;
            }
        }
        std::array::from_fn(|channel| {
            baseline_xyz[channel]
                + self.application_weight * (fitted[channel] - baseline_xyz[channel])
        })
    }
}

/// Robust local weighted least squares with a weak baseline prior. In addition
/// to centre XYZ, the model contains an X/Y slope for every channel. Treating
/// every nearby CFA site as one constant colour smears precisely the edges and
/// fine texture this experiment is intended to recover; the local affine field
/// preserves first-order structure while remaining independently tileable.
pub fn solve_joint_xyz(
    observations: &[CfaObservation],
    prior_xyz: [f32; 3],
    prior_weight: f32,
) -> Option<JointCfaEstimate> {
    let valid = observations
        .iter()
        .filter(|observation| observation_weight(observation) > 0.0)
        .collect::<Vec<_>>();
    if valid.len() < 3 {
        return None;
    }
    let observation_scale = valid
        .iter()
        .map(|observation| observation_weight(observation))
        .sum::<f32>()
        / valid.len() as f32;
    let (data_rank, initial_information_confidence) = response_information(&valid, None);
    let (spatial_rank, initial_spatial_confidence) = spatial_information(&valid, None);
    if data_rank < 3 || spatial_rank < 2 {
        return None;
    }
    let regularization = prior_weight.max(1.0e-8) * observation_scale.max(1.0e-8);
    let mut estimate = [0.0_f32; MODEL_SIZE];
    estimate[..3].copy_from_slice(&prior_xyz);
    let mut iterations = 0;
    for _ in 0..SOLVER_ITERATIONS {
        let mut normal = [[0.0_f64; MODEL_SIZE]; MODEL_SIZE];
        let mut rhs = [0.0_f64; MODEL_SIZE];
        for parameter in 0..MODEL_SIZE {
            // Derivatives receive a zero-centred ridge prior. It makes sparse
            // or nearly collinear real-camera sample layouts well-conditioned
            // without imposing a flat radiance field.
            let ridge = if parameter < 3 {
                regularization
            } else {
                regularization * 0.10
            };
            normal[parameter][parameter] += f64::from(ridge);
            if parameter < 3 {
                rhs[parameter] += f64::from(ridge * prior_xyz[parameter]);
            }
        }
        for observation in &valid {
            let design = design_row(observation);
            let predicted = dot_model(design, estimate);
            let normalized_residual = (observation.value - predicted).abs()
                / observation.noise_variance.max(1.0e-10).sqrt();
            let robust = (HUBER_SIGMA / normalized_residual.max(HUBER_SIGMA)).min(1.0);
            let weight = observation_weight(observation) * robust;
            for row in 0..MODEL_SIZE {
                rhs[row] += f64::from(weight * design[row] * observation.value);
                for column in 0..MODEL_SIZE {
                    normal[row][column] += f64::from(weight * design[row] * design[column]);
                }
            }
        }
        estimate = solve_linear(normal, rhs)?.map(|value| value as f32);
        if estimate.iter().any(|value| !value.is_finite()) {
            return None;
        }
        iterations += 1;
    }
    let mut phase_mask = 0_u8;
    let mut residual_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut baseline_loss = 0.0;
    let mut joint_loss = 0.0;
    let mut closest = [None::<(f32, [f32; 2])>; u32::BITS as usize];
    let mut robust_weights = Vec::with_capacity(valid.len());
    for observation in &valid {
        let weight = observation_weight(observation);
        let baseline_residual = observation.value
            - observation.baseline_prediction.unwrap_or_else(|| {
                observation.response[0] * prior_xyz[0]
                    + observation.response[1] * prior_xyz[1]
                    + observation.response[2] * prior_xyz[2]
            });
        let joint_residual = observation.value - dot_model(design_row(observation), estimate);
        let sigma = observation.noise_variance.max(1.0e-10).sqrt();
        let normalized_residual = joint_residual.abs() / sigma;
        robust_weights.push((HUBER_SIGMA / normalized_residual.max(HUBER_SIGMA)).min(1.0));
        residual_sum += weight * joint_residual.abs();
        baseline_loss += weight * robust_noise_loss(baseline_residual / sigma);
        joint_loss += weight * robust_noise_loss(joint_residual / sigma);
        weight_sum += weight;
        if observation.camera_index < u32::BITS as usize {
            let distance = observation.output_offset[0].hypot(observation.output_offset[1]);
            let slot = &mut closest[observation.camera_index];
            if slot.is_none_or(|(best, _)| distance < best) {
                *slot = Some((distance, observation.output_offset));
            }
        }
        phase_mask |= 1 << observation.phase.index();
    }
    let (robust_rank, robust_information_confidence) =
        response_information(&valid, Some(&robust_weights));
    let (robust_model_rank, _) = model_information(&valid, Some(&robust_weights));
    let (robust_spatial_rank, robust_spatial_confidence) =
        spatial_information(&valid, Some(&robust_weights));
    if robust_rank < 3 || robust_spatial_rank < 2 {
        return None;
    }
    let mut camera_weights = [0.0_f32; u32::BITS as usize];
    for (observation, robust) in valid.iter().zip(&robust_weights) {
        if observation.camera_index < camera_weights.len() {
            camera_weights[observation.camera_index] += observation_weight(observation) * robust;
        }
    }
    let strongest_camera = camera_weights.into_iter().fold(0.0_f32, f32::max);
    let retained_camera_mask = camera_weights
        .into_iter()
        .enumerate()
        .filter(|(_, weight)| *weight >= strongest_camera * 0.02)
        .fold(0_u32, |mask, (camera, _)| mask | (1_u32 << camera));
    if retained_camera_mask.count_ones() < 2 {
        return None;
    }
    let closest = closest.into_iter().flatten().collect::<Vec<_>>();
    let centroid = closest.iter().fold([0.0_f32; 2], |mut total, (_, offset)| {
        total[0] += offset[0];
        total[1] += offset[1];
        total
    });
    let centroid = centroid.map(|value| value / closest.len().max(1) as f32);
    let phase_spread = (closest
        .iter()
        .map(|(_, offset)| (offset[0] - centroid[0]).powi(2) + (offset[1] - centroid[1]).powi(2))
        .sum::<f32>()
        / closest.len().max(1) as f32)
        .sqrt();
    Some(JointCfaEstimate {
        xyz: [
            estimate[0].max(0.0),
            estimate[1].max(0.0),
            estimate[2].max(0.0),
        ],
        application_weight: initial_information_confidence
            .min(robust_information_confidence)
            .min(initial_spatial_confidence)
            .min(robust_spatial_confidence)
            .sqrt(),
        report: JointCfaSolveReport {
            observations: valid.len(),
            cameras: retained_camera_mask.count_ones() as usize,
            phase_mask,
            phase_spread,
            iterations,
            weighted_residual: residual_sum / weight_sum.max(1.0e-8),
            in_sample_baseline_loss: baseline_loss / weight_sum.max(1.0e-8),
            in_sample_affine_loss: joint_loss / weight_sum.max(1.0e-8),
            data_rank: robust_rank,
            information_confidence: initial_information_confidence
                .min(robust_information_confidence)
                .min(initial_spatial_confidence)
                .min(robust_spatial_confidence),
            model_rank: robust_model_rank,
            spatial_rank: robust_spatial_rank,
        },
        model: estimate,
        preserve_baseline_luminance: false,
    })
}

/// Rank and a scale-free conditioning score for the unregularized centre-XYZ
/// response information. Ridge priors deliberately do not participate: they
/// may stabilize unsupported directions, but must not be reported as sensor
/// evidence.
fn response_information(
    observations: &[&CfaObservation],
    robust_weights: Option<&[f32]>,
) -> (usize, f32) {
    let mut rows = Vec::with_capacity(observations.len());
    for (index, observation) in observations.iter().enumerate() {
        let robust = robust_weights.map_or(1.0, |weights| weights[index]);
        let weight = observation_weight(observation) * robust;
        rows.push((observation.response, weight));
    }
    information_rank(&rows)
}

fn model_information(
    observations: &[&CfaObservation],
    robust_weights: Option<&[f32]>,
) -> (usize, f32) {
    let mut rows = Vec::with_capacity(observations.len());
    for (index, observation) in observations.iter().enumerate() {
        let robust = robust_weights.map_or(1.0, |weights| weights[index]);
        rows.push((
            design_row(observation),
            observation_weight(observation) * robust,
        ));
    }
    information_rank(&rows)
}

fn spatial_information(
    observations: &[&CfaObservation],
    robust_weights: Option<&[f32]>,
) -> (usize, f32) {
    let total_weight = observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            observation_weight(observation) * robust_weights.map_or(1.0, |weights| weights[index])
        })
        .sum::<f32>()
        .max(1.0e-10);
    let mut mean = [0.0_f32; 2];
    for (index, observation) in observations.iter().enumerate() {
        let weight =
            observation_weight(observation) * robust_weights.map_or(1.0, |weights| weights[index]);
        mean[0] += weight * observation.output_offset[0] / total_weight;
        mean[1] += weight * observation.output_offset[1] / total_weight;
    }
    let rows = observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            let robust = robust_weights.map_or(1.0, |weights| weights[index]);
            (
                [
                    observation.output_offset[0] - mean[0],
                    observation.output_offset[1] - mean[1],
                ],
                observation_weight(observation) * robust,
            )
        })
        .collect::<Vec<_>>();
    information_rank(&rows)
}

fn information_rank<const N: usize>(rows: &[([f32; N], f32)]) -> (usize, f32) {
    let mut information = [[0.0_f64; N]; N];
    for (design, weight) in rows {
        for row in 0..N {
            for column in 0..N {
                information[row][column] += f64::from(*weight * design[row] * design[column]);
            }
        }
    }
    let scale = information
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale <= 0.0 {
        return (0, 0.0);
    }
    let mut rank = 0;
    let mut minimum_pivot = f64::INFINITY;
    let mut maximum_pivot = 0.0_f64;
    for column in 0..N {
        let pivot = (column..N).max_by(|left, right| {
            information[*left][column]
                .abs()
                .total_cmp(&information[*right][column].abs())
        });
        let Some(pivot) = pivot else { continue };
        let magnitude = information[pivot][column].abs();
        if magnitude <= scale * 1.0e-5 {
            continue;
        }
        information.swap(column, pivot);
        let divisor = information[column][column];
        for row in column + 1..N {
            let factor = information[row][column] / divisor;
            for entry in column..N {
                information[row][entry] -= factor * information[column][entry];
            }
        }
        rank += 1;
        minimum_pivot = minimum_pivot.min(magnitude);
        maximum_pivot = maximum_pivot.max(magnitude);
    }
    let confidence = if rank == N {
        (minimum_pivot / maximum_pivot.max(f64::MIN_POSITIVE)).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    (rank, confidence)
}

fn robust_noise_loss(residual: f32) -> f32 {
    let absolute = residual.abs();
    if absolute <= HUBER_SIGMA {
        0.5 * absolute * absolute
    } else {
        HUBER_SIGMA * (absolute - 0.5 * HUBER_SIGMA)
    }
}

fn design_row(observation: &CfaObservation) -> [f32; MODEL_SIZE] {
    let [dx, dy] = observation.output_offset;
    let [x, y, z] = observation.response;
    [x, y, z, dx * x, dx * y, dx * z, dy * x, dy * y, dy * z]
}

fn dot_model(first: [f32; MODEL_SIZE], second: [f32; MODEL_SIZE]) -> f32 {
    first
        .into_iter()
        .zip(second)
        .map(|(left, right)| left * right)
        .sum()
}

fn solve_linear<const N: usize>(mut matrix: [[f64; N]; N], mut rhs: [f64; N]) -> Option<[f64; N]> {
    for column in 0..N {
        let pivot = (column..N).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if !matrix[pivot][column].is_finite() || matrix[pivot][column].abs() < 1.0e-14 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let divisor = matrix[column][column];
        for entry in &mut matrix[column][column..] {
            *entry /= divisor;
        }
        rhs[column] /= divisor;
        let pivot_row = matrix[column];
        for row in 0..N {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor == 0.0 {
                continue;
            }
            for (target, pivot) in matrix[row][column..].iter_mut().zip(&pivot_row[column..]) {
                *target -= factor * pivot;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    Some(rhs)
}

fn observation_weight(observation: &CfaObservation) -> f32 {
    if observation.visibility != Visibility::Visible
        || !observation.value.is_finite()
        || !observation.noise_variance.is_finite()
        || observation.noise_variance <= 0.0
    {
        return 0.0;
    }
    observation.spatial_weight.max(0.0)
        * observation.geometry_confidence.clamp(0.0, 1.0)
        * observation
            .highlight_provenance
            .weight(observation.highlight_confidence)
        / observation.noise_variance.max(1.0e-10)
}

pub fn noise_variance(
    signal: f32,
    phase: CfaPhase,
    model: Option<NoiseModel>,
    code_range: f32,
) -> f32 {
    let quantization = code_range.max(1.0).recip();
    let channel: Option<NoiseChannelModel> = model.map(|model| match phase {
        CfaPhase::R => model.red,
        CfaPhase::Gr | CfaPhase::Gb => model.green,
        CfaPhase::B => model.blue,
    });
    channel
        .map(|channel| channel.a * signal.max(0.0) + channel.b)
        .unwrap_or(quantization * quantization)
        .max(quantization * quantization)
}

/// Inflate diagonal variances by the absolute correlation mass caused by
/// reused physical sensor sites. This is a conservative diagonal
/// approximation to generalized least squares: it prevents overlapping
/// crosstalk/interpolation footprints from being counted as independent while
/// keeping the small robust solver tractable.
pub fn account_shared_sample_dependence(observations: &mut [CfaObservation]) {
    let original = observations
        .iter()
        .map(|observation| observation.noise_variance.max(1.0e-10))
        .collect::<Vec<_>>();
    let mut correlation_mass = vec![0.0_f32; observations.len()];
    for left in 0..observations.len() {
        for right in left + 1..observations.len() {
            let mut covariance = 0.0_f32;
            for a in
                &observations[left].noise_dependencies[..observations[left].noise_dependency_count]
            {
                for b in &observations[right].noise_dependencies
                    [..observations[right].noise_dependency_count]
                {
                    if a.key == b.key {
                        covariance += a.coefficient
                            * b.coefficient
                            * 0.5
                            * (a.physical_variance + b.physical_variance);
                    }
                }
            }
            let correlation = (covariance / (original[left] * original[right]).sqrt())
                .abs()
                .min(1.0);
            correlation_mass[left] += correlation;
            correlation_mass[right] += correlation;
        }
    }
    for ((observation, variance), mass) in
        observations.iter_mut().zip(original).zip(correlation_mass)
    {
        observation.noise_variance = variance * (1.0 + mass);
    }
}

/// Propagate calibrated independent CFA-plane noise through the exact local
/// four-phase crosstalk row and flat-field gain used for this measurement.
pub fn corrected_noise_variance(
    sample: &CorrectedCfaSample,
    model: Option<NoiseModel>,
    code_range: f32,
) -> f32 {
    let variance = sample.noise_components[..sample.noise_component_count]
        .iter()
        .map(|component| {
            component.coefficient.powi(2)
                * noise_variance(component.signal, component.phase, model, code_range)
        })
        .sum::<f32>();
    variance * sample.flat_field.powi(2)
}

/// Measurement row which predicts one camera CFA value from common D50 XYZ.
/// This is the inverse of `diag(flat_field) * forward * diag(white_balance)`.
pub fn camera_response(
    color: &ModuleColor,
    xyz_field_gain: [f32; 3],
    phase: CfaPhase,
) -> Option<[f32; 3]> {
    if !color.calibrated {
        return None;
    }
    let mut camera_to_xyz = [[0.0_f64; 3]; 3];
    for (row, values) in camera_to_xyz.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = f64::from(
                xyz_field_gain[row] * color.forward[row][column] * color.wb_gains[column],
            );
        }
    }
    let xyz_to_camera = inverse(&camera_to_xyz)?;
    Some(xyz_to_camera[phase.color_channel()].map(|value| value as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(response: [f32; 3], value: f32, phase: CfaPhase) -> CfaObservation {
        CfaObservation {
            camera_index: phase.index(),
            camera_id: phase.index(),
            sensor_xy: [0, 0],
            output_offset: [0.0, 0.0],
            phase,
            value,
            noise_variance: 1.0,
            highlight_provenance: HighlightProvenance::Measured,
            highlight_confidence: 255,
            geometry_confidence: 1.0,
            visibility: Visibility::Visible,
            response,
            spatial_weight: 1.0,
            baseline_prediction: None,
            noise_dependencies: [NoiseDependency::default(); 16],
            noise_dependency_count: 0,
        }
    }

    #[test]
    fn spatially_independent_cfa_equations_recover_xyz() {
        let mut observations = Vec::new();
        for (channel, (value, phase)) in
            [(0.2, CfaPhase::R), (0.4, CfaPhase::Gr), (0.6, CfaPhase::B)]
                .into_iter()
                .enumerate()
        {
            let mut response = [0.0; 3];
            response[channel] = 1.0;
            for offset in [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
                let mut sample = observation(response, value, phase);
                sample.output_offset = offset;
                observations.push(sample);
            }
        }
        let estimate = solve_joint_xyz(&observations, [0.0; 3], 1.0e-6).unwrap();
        assert!((estimate.xyz[0] - 0.2).abs() < 1.0e-4);
        assert!((estimate.xyz[1] - 0.4).abs() < 1.0e-4);
        assert!((estimate.xyz[2] - 0.6).abs() < 1.0e-4);
        assert_eq!(estimate.report.cameras, 3);
        assert_eq!(estimate.report.model_rank, 9);
    }

    #[test]
    fn unresolved_highlight_is_not_an_observation() {
        let mut observations = Vec::new();
        for (channel, phase) in [CfaPhase::R, CfaPhase::Gr, CfaPhase::B]
            .into_iter()
            .enumerate()
        {
            let mut response = [0.0; 3];
            response[channel] = 1.0;
            for offset in [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
                let mut sample = observation(response, 0.2 * (channel + 1) as f32, phase);
                sample.output_offset = offset;
                if channel == 2 {
                    sample.highlight_provenance = HighlightProvenance::Unresolved;
                    sample.highlight_confidence = 0;
                }
                observations.push(sample);
            }
        }
        assert!(solve_joint_xyz(&observations, [0.0; 3], 0.0).is_none());
    }

    #[test]
    fn affine_field_recovers_centre_without_averaging_gradient() {
        let centres = [0.2, 0.4, 0.6];
        let slopes_x = [0.08, -0.03, 0.05];
        let slopes_y = [-0.04, 0.06, 0.02];
        let phases = [CfaPhase::R, CfaPhase::Gr, CfaPhase::B];
        let mut observations = Vec::new();
        for channel in 0..3 {
            let mut response = [0.0; 3];
            response[channel] = 1.0;
            for offset in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
                let mut sample = observation(
                    response,
                    centres[channel]
                        + slopes_x[channel] * offset[0]
                        + slopes_y[channel] * offset[1],
                    phases[channel],
                );
                sample.output_offset = offset;
                observations.push(sample);
            }
        }
        let estimate = solve_joint_xyz(&observations, [0.1; 3], 1.0e-8).unwrap();
        for (actual, expected) in estimate.xyz.into_iter().zip(centres) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn regularization_does_not_turn_rank_deficiency_into_sensor_evidence() {
        let observations = [
            observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
            observation([0.0, 1.0, 0.0], 0.4, CfaPhase::Gr),
            observation([1.0, 1.0, 0.0], 0.6, CfaPhase::Gb),
        ];
        assert!(solve_joint_xyz(&observations, [0.1, 0.1, 0.9], 1.0).is_none());
    }

    #[test]
    fn coincident_sampling_positions_do_not_claim_spatial_support() {
        let observations = (0..3)
            .flat_map(|_| {
                [
                    observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
                    observation([0.0, 1.0, 0.0], 0.4, CfaPhase::Gr),
                    observation([0.0, 0.0, 1.0], 0.6, CfaPhase::B),
                ]
            })
            .collect::<Vec<_>>();
        assert!(solve_joint_xyz(&observations, [0.1; 3], 0.1).is_none());
    }

    #[test]
    fn unsupported_baseline_luminance_is_preserved_during_application() {
        let centres = [0.2, 0.4, 0.6];
        let mut observations = Vec::new();
        for (channel, phase) in [CfaPhase::R, CfaPhase::Gr, CfaPhase::B]
            .into_iter()
            .enumerate()
        {
            let mut response = [0.0; 3];
            response[channel] = 1.0;
            for offset in [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
                let mut sample = observation(response, centres[channel], phase);
                sample.output_offset = offset;
                observations.push(sample);
            }
        }
        let baseline = [0.1, 0.3, 0.5];
        let mut estimate = solve_joint_xyz(&observations, baseline, 1.0e-6).unwrap();
        estimate.apply_over_baseline(baseline, 1.0, true);
        assert!((estimate.xyz[1] - baseline[1]).abs() < 1.0e-6);
        assert!((estimate.xyz[0] - baseline[0]).abs() > 1.0e-3);
    }

    #[test]
    fn in_sample_baseline_loss_uses_the_spatially_matched_prediction() {
        let mut observations = Vec::new();
        for (channel, phase) in [CfaPhase::R, CfaPhase::Gr, CfaPhase::B]
            .into_iter()
            .enumerate()
        {
            let mut response = [0.0; 3];
            response[channel] = 1.0;
            for offset in [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
                let value = 0.2 * (channel + 1) as f32 + 0.05 * offset[0];
                let mut sample = observation(response, value, phase);
                sample.output_offset = offset;
                sample.baseline_prediction = Some(value);
                observations.push(sample);
            }
        }
        let estimate = solve_joint_xyz(&observations, [0.2, 0.4, 0.6], 1.0e-6).unwrap();
        assert!(estimate.report.in_sample_baseline_loss < 1.0e-10);
    }

    #[test]
    fn corrected_noise_follows_crosstalk_coefficients_and_flat_field() {
        let sample = CorrectedCfaSample {
            phase: CfaPhase::R,
            value: 0.2,
            white: 1.0,
            highlight_confidence: 255,
            source_values: [0.2, 0.3, 0.4, 0.5],
            crosstalk_row: [1.0, 2.0, 0.0, 0.0],
            flat_field: 2.0,
            noise_components: {
                let mut components = [crate::image::CfaNoiseComponent::default(); 16];
                components[0] = crate::image::CfaNoiseComponent {
                    phase: CfaPhase::R,
                    sensor_index: 0,
                    signal: 0.2,
                    coefficient: 1.0,
                };
                components[1] = crate::image::CfaNoiseComponent {
                    phase: CfaPhase::Gr,
                    sensor_index: 1,
                    signal: 0.3,
                    coefficient: 2.0,
                };
                components
            },
            noise_component_count: 2,
        };
        let variance = corrected_noise_variance(&sample, None, 1_000.0);
        assert!((variance - 20.0e-6).abs() < 1.0e-10);
    }

    #[test]
    fn shared_physical_samples_are_not_counted_as_independent() {
        let mut observations = [
            observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
            observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
        ];
        for observation in &mut observations {
            observation.noise_dependencies[0] = NoiseDependency {
                key: 42,
                coefficient: 1.0,
                physical_variance: 1.0,
            };
            observation.noise_dependency_count = 1;
        }
        account_shared_sample_dependence(&mut observations);
        assert_eq!(observations[0].noise_variance, 2.0);
        assert_eq!(observations[1].noise_variance, 2.0);
    }
}
