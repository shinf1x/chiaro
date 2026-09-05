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
    pub contributor_baseline_loss: f32,
    pub contributor_joint_loss: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct JointCfaEstimate {
    pub xyz: [f32; 3],
    /// Confidence used by synthesis when blending this estimate with its
    /// production baseline. The algebraic solver itself returns one.
    pub application_weight: f32,
    pub report: JointCfaSolveReport,
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
    let mut camera_mask = 0_u32;
    let mut phase_mask = 0_u8;
    let mut residual_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut baseline_loss = 0.0;
    let mut joint_loss = 0.0;
    let mut closest = [None::<(f32, [f32; 2])>; u32::BITS as usize];
    for observation in &valid {
        let weight = observation_weight(observation);
        let baseline_residual = observation.value
            - observation.response[0] * prior_xyz[0]
            - observation.response[1] * prior_xyz[1]
            - observation.response[2] * prior_xyz[2];
        let joint_residual = observation.value - dot_model(design_row(observation), estimate);
        let sigma = observation.noise_variance.max(1.0e-10).sqrt();
        residual_sum += weight * joint_residual.abs();
        baseline_loss += weight * robust_noise_loss(baseline_residual / sigma);
        joint_loss += weight * robust_noise_loss(joint_residual / sigma);
        weight_sum += weight;
        if observation.camera_index < u32::BITS as usize {
            camera_mask |= 1 << observation.camera_index;
            let distance = observation.output_offset[0].hypot(observation.output_offset[1]);
            let slot = &mut closest[observation.camera_index];
            if slot.is_none_or(|(best, _)| distance < best) {
                *slot = Some((distance, observation.output_offset));
            }
        }
        phase_mask |= 1 << observation.phase.index();
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
        application_weight: 1.0,
        report: JointCfaSolveReport {
            observations: valid.len(),
            cameras: camera_mask.count_ones() as usize,
            phase_mask,
            phase_spread,
            iterations,
            weighted_residual: residual_sum / weight_sum.max(1.0e-8),
            contributor_baseline_loss: baseline_loss / weight_sum.max(1.0e-8),
            contributor_joint_loss: joint_loss / weight_sum.max(1.0e-8),
        },
    })
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

/// Propagate calibrated independent CFA-plane noise through the exact local
/// four-phase crosstalk row and flat-field gain used for this measurement.
pub fn corrected_noise_variance(
    sample: &CorrectedCfaSample,
    model: Option<NoiseModel>,
    code_range: f32,
) -> f32 {
    let phases = [CfaPhase::R, CfaPhase::Gr, CfaPhase::Gb, CfaPhase::B];
    let variance = sample
        .source_values
        .into_iter()
        .zip(sample.crosstalk_row)
        .zip(phases)
        .map(|((signal, coefficient), phase)| {
            coefficient.powi(2) * noise_variance(signal, phase, model, code_range)
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
        }
    }

    #[test]
    fn three_independent_cfa_equations_recover_xyz() {
        let observations = [
            observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
            observation([0.0, 1.0, 0.0], 0.4, CfaPhase::Gr),
            observation([0.0, 0.0, 1.0], 0.6, CfaPhase::B),
        ];
        let estimate = solve_joint_xyz(&observations, [0.0; 3], 1.0e-6).unwrap();
        assert!((estimate.xyz[0] - 0.2).abs() < 1.0e-4);
        assert!((estimate.xyz[1] - 0.4).abs() < 1.0e-4);
        assert!((estimate.xyz[2] - 0.6).abs() < 1.0e-4);
        assert_eq!(estimate.report.cameras, 3);
    }

    #[test]
    fn unresolved_highlight_is_not_an_observation() {
        let mut observations = vec![
            observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
            observation([0.0, 1.0, 0.0], 0.4, CfaPhase::Gr),
            observation([0.0, 0.0, 1.0], 0.6, CfaPhase::B),
        ];
        observations[2].highlight_provenance = HighlightProvenance::Unresolved;
        observations[2].highlight_confidence = 0;
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
    fn corrected_noise_follows_crosstalk_coefficients_and_flat_field() {
        let sample = CorrectedCfaSample {
            phase: CfaPhase::R,
            value: 0.2,
            white: 1.0,
            highlight_confidence: 255,
            source_values: [0.2, 0.3, 0.4, 0.5],
            crosstalk_row: [1.0, 2.0, 0.0, 0.0],
            flat_field: 2.0,
        };
        let variance = corrected_noise_variance(&sample, None, 1_000.0);
        assert!((variance - 20.0e-6).abs() < 1.0e-10);
    }
}
