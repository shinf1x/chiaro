//! Experimental cross-camera reconstruction from physical CFA observations.
//!
//! This module is deliberately independent of the production selector. It
//! provides a compact observation record and tileable robust solves in the
//! common D50 XYZ space: a cheap constant-XYZ tier and a full local-affine tier. Real-capture held-out validation decides
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

/// Complexity of the local Joint-CFA model.  Flat/weakly structured regions
/// can use the three-parameter constant-XYZ model; edge/texture regions retain
/// the full nine-parameter affine field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JointCfaSolveMode {
    Constant,
    Affine,
}

impl Default for JointCfaSolveMode {
    fn default() -> Self {
        Self::Affine
    }
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

#[derive(Clone, Copy, Debug, Default)]
struct DependencyUse {
    key: u64,
    observation: usize,
    coefficient: f32,
    physical_variance: f32,
}

/// Reusable hot-loop storage for Joint-CFA solving and covariance accounting.
/// A renderer can keep one instance per worker/band and avoid allocator traffic
/// for every output pixel.
#[derive(Debug, Default)]
pub struct CfaSolverScratch {
    robust_weights: Vec<f32>,
    /// Precomputed design rows and immutable base weights for the current pixel.
    /// The solver used to rebuild both for every IRLS pass and for several
    /// diagnostics; keeping them here removes a large amount of repeated scalar
    /// work in the hottest loop.
    design_rows: Vec<[f32; MODEL_SIZE]>,
    base_weights: Vec<f32>,
    inverse_sigmas: Vec<f32>,
    original_variance: Vec<f32>,
    correlation_mass: Vec<f32>,
    dependency_uses: Vec<DependencyUse>,
    pair_covariance: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct JointCfaSolveReport {
    pub observations: usize,
    pub mode: JointCfaSolveMode,
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
    /// Fraction of baseline robust loss removed by the solved model, further
    /// tempered by its absolute residual. Unlike matrix conditioning this
    /// measures whether the local model actually explains the sensor data.
    pub fit_confidence: f32,
    /// Rank of the complete unregularized affine design (XYZ plus both
    /// spatial derivatives). A value below nine means the centre estimate can
    /// still depend on ridge priors through an unsupported slope direction.
    pub model_rank: usize,
    pub spatial_rank: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct JointCfaEstimate {
    pub xyz: [f32; 3],
    /// Luminance/detail confidence used by synthesis when blending this
    /// estimate with its trusted production colour baseline.
    pub application_weight: f32,
    /// Chroma correction weight. Production Joint-CFA intentionally keeps this
    /// at zero: the raw multi-camera solve supplies high-frequency luminance
    /// while the established synthesis path owns chromaticity. Held-out
    /// diagnostics may opt into a non-zero value without affecting rendering.
    pub chroma_application_weight: f32,
    pub report: JointCfaSolveReport,
    model: [f32; MODEL_SIZE],
    /// Solver-only confidence before the output structure gate is applied.
    /// It combines matrix conditioning with actual robust fit improvement.
    base_application_weight: f32,
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
        chroma_weight: f32,
    ) {
        self.application_weight =
            self.base_application_weight * structure_weight.clamp(0.0, 1.0);
        self.chroma_application_weight =
            self.application_weight * chroma_weight.clamp(0.0, 1.0);
        self.preserve_baseline_luminance = preserve_baseline_luminance;
        self.xyz = self.applied_xyz_at([0.0, 0.0], baseline_xyz);
    }

    /// Re-evaluate this solved local model at a nearby output-pixel offset.
    /// The spatial lattice is deliberately luminance-only. Affine chroma
    /// slopes are not propagated into neighbouring output pixels because a
    /// tiny colour-model error becomes a conspicuous magenta/green impulse.
    pub fn reapplied_luminance_at(
        &self,
        model_offset: [f32; 2],
        baseline_xyz: [f32; 3],
        structure_weight: f32,
        preserve_baseline_luminance: bool,
    ) -> Self {
        let mut estimate = *self;
        estimate.application_weight =
            estimate.base_application_weight * structure_weight.clamp(0.0, 1.0);
        estimate.chroma_application_weight = 0.0;
        estimate.preserve_baseline_luminance = preserve_baseline_luminance;
        estimate.xyz = estimate.applied_xyz_at(model_offset, baseline_xyz);
        estimate
    }

    pub fn base_application_weight(&self) -> f32 {
        self.base_application_weight
    }

    /// Apply the raw-CFA model as a detail reconstruction over a trusted colour
    /// baseline. Luminance is taken from the fitted model, but chromaticity is
    /// retained from the baseline unless an explicit diagnostic chroma weight
    /// is supplied. This mirrors the useful part of Lumen's ResAmp philosophy:
    /// aggressive high-frequency evidence is allowed to sharpen intensity
    /// without inventing high-frequency colour.
    pub fn applied_xyz_at(&self, offset: [f32; 2], baseline_xyz: [f32; 3]) -> [f32; 3] {
        let fitted = self.fitted_xyz_at(offset).map(|value| value.max(0.0));
        let baseline_y = baseline_xyz[1].max(0.0);
        let fitted_y = fitted[1].max(0.0);

        // Scale the trusted baseline colour to the fitted luminance. This is
        // the Joint-CFA high-frequency detail candidate. In near-black regions
        // do not manufacture a colour direction from an ill-conditioned ratio.
        let luminance_candidate = if baseline_y > 1.0e-6 && fitted_y.is_finite() {
            let scale = (fitted_y / baseline_y).clamp(0.0, 8.0);
            baseline_xyz.map(|value| (value * scale).max(0.0))
        } else {
            baseline_xyz
        };

        let luma_weight = if self.preserve_baseline_luminance {
            0.0
        } else {
            self.application_weight.clamp(0.0, 1.0)
        };
        let chroma_weight = self.chroma_application_weight.clamp(0.0, 1.0);
        std::array::from_fn(|channel| {
            let luma_delta = luminance_candidate[channel] - baseline_xyz[channel];
            let chroma_delta = fitted[channel] - luminance_candidate[channel];
            (baseline_xyz[channel] + luma_weight * luma_delta + chroma_weight * chroma_delta)
                .max(0.0)
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
    let mut scratch = CfaSolverScratch::default();
    solve_joint_xyz_with_scratch(observations, prior_xyz, prior_weight, &mut scratch)
}

pub fn solve_joint_xyz_with_scratch(
    observations: &[CfaObservation],
    prior_xyz: [f32; 3],
    prior_weight: f32,
    scratch: &mut CfaSolverScratch,
) -> Option<JointCfaEstimate> {
    solve_joint_xyz_mode_with_scratch(
        observations,
        prior_xyz,
        prior_weight,
        JointCfaSolveMode::Affine,
        scratch,
    )
}

/// Tiered Joint-CFA solve.  `Constant` keeps only centre XYZ (3 variables),
/// while `Affine` retains the historical XYZ + dXYZ/dx + dXYZ/dy model
/// (9 variables).  Both paths share the same robust weighting, covariance
/// accounting, camera-rank checks and report semantics.
pub fn solve_joint_xyz_mode_with_scratch(
    observations: &[CfaObservation],
    prior_xyz: [f32; 3],
    prior_weight: f32,
    mode: JointCfaSolveMode,
    scratch: &mut CfaSolverScratch,
) -> Option<JointCfaEstimate> {
    scratch.design_rows.clear();
    scratch.base_weights.clear();
    scratch.inverse_sigmas.clear();
    scratch.design_rows.reserve(observations.len());
    scratch.base_weights.reserve(observations.len());
    scratch.inverse_sigmas.reserve(observations.len());

    let mut valid_count = 0usize;
    let mut observation_weight_sum = 0.0f32;
    for observation in observations {
        let weight = observation_weight(observation);
        scratch.design_rows.push(design_row(observation));
        scratch.base_weights.push(weight);
        scratch
            .inverse_sigmas
            .push(observation.noise_variance.max(1.0e-10).sqrt().recip());
        if weight > 0.0 {
            valid_count += 1;
            observation_weight_sum += weight;
        }
    }
    if valid_count < 3 {
        return None;
    }
    let observation_scale = observation_weight_sum / valid_count as f32;
    let (data_rank, initial_information_confidence) =
        response_information_prepared(observations, &scratch.base_weights, None);
    if data_rank < 3 {
        return None;
    }
    let (_spatial_rank, initial_spatial_confidence) = if mode == JointCfaSolveMode::Affine {
        let result = spatial_information_prepared(observations, &scratch.base_weights, None);
        if result.0 < 2 {
            return None;
        }
        result
    } else {
        (0, 1.0)
    };

    let parameter_count = match mode {
        JointCfaSolveMode::Constant => 3,
        JointCfaSolveMode::Affine => MODEL_SIZE,
    };
    let regularization = prior_weight.max(1.0e-8) * observation_scale.max(1.0e-8);
    let mut estimate = [0.0_f32; MODEL_SIZE];
    estimate[..3].copy_from_slice(&prior_xyz);
    let mut iterations = 0;
    for _ in 0..SOLVER_ITERATIONS {
        let mut normal = [[0.0_f64; MODEL_SIZE]; MODEL_SIZE];
        let mut rhs = [0.0_f64; MODEL_SIZE];
        for parameter in 0..parameter_count {
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
        for (index, observation) in observations.iter().enumerate() {
            let base_weight = scratch.base_weights[index];
            if base_weight <= 0.0 {
                continue;
            }
            let design = scratch.design_rows[index];
            let predicted = dot_model_prefix(design, estimate, parameter_count);
            let normalized_residual =
                (observation.value - predicted).abs() * scratch.inverse_sigmas[index];
            let robust = (HUBER_SIGMA / normalized_residual.max(HUBER_SIGMA)).min(1.0);
            let weight = base_weight * robust;
            accumulate_normal_upper(
                &mut normal,
                &mut rhs,
                design,
                observation.value,
                weight,
                parameter_count,
            );
        }
        mirror_upper_triangle(&mut normal, parameter_count);
        let solved = solve_spd_prefix(normal, rhs, parameter_count)?;
        for parameter in 0..parameter_count {
            estimate[parameter] = solved[parameter] as f32;
        }
        for parameter in parameter_count..MODEL_SIZE {
            estimate[parameter] = 0.0;
        }
        if estimate[..parameter_count]
            .iter()
            .any(|value| !value.is_finite())
        {
            return None;
        }
        iterations += 1;
    }

    scratch.robust_weights.clear();
    scratch.robust_weights.resize(observations.len(), 0.0);
    let mut phase_mask = 0_u8;
    let mut residual_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut baseline_loss = 0.0;
    let mut joint_loss = 0.0;
    let mut closest = [None::<(f32, [f32; 2])>; u32::BITS as usize];
    for (index, observation) in observations.iter().enumerate() {
        let weight = scratch.base_weights[index];
        if weight <= 0.0 {
            continue;
        }
        let baseline_residual = observation.value
            - observation.baseline_prediction.unwrap_or_else(|| {
                observation.response[0] * prior_xyz[0]
                    + observation.response[1] * prior_xyz[1]
                    + observation.response[2] * prior_xyz[2]
            });
        let joint_residual = observation.value
            - dot_model_prefix(scratch.design_rows[index], estimate, parameter_count);
        let normalized_residual = joint_residual.abs() * scratch.inverse_sigmas[index];
        let robust = (HUBER_SIGMA / normalized_residual.max(HUBER_SIGMA)).min(1.0);
        scratch.robust_weights[index] = robust;
        residual_sum += weight * joint_residual.abs();
        baseline_loss += weight
            * robust_noise_loss(baseline_residual * scratch.inverse_sigmas[index]);
        joint_loss +=
            weight * robust_noise_loss(joint_residual * scratch.inverse_sigmas[index]);
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
    let robust_weights = Some(scratch.robust_weights.as_slice());
    let (robust_rank, robust_information_confidence) = response_information_prepared(
        observations,
        &scratch.base_weights,
        robust_weights,
    );
    let (robust_model_rank, _) = if mode == JointCfaSolveMode::Affine {
        model_information_prepared(
            observations,
            &scratch.design_rows,
            &scratch.base_weights,
            robust_weights,
        )
    } else {
        (robust_rank, robust_information_confidence)
    };
    let (robust_spatial_rank, robust_spatial_confidence) =
        if mode == JointCfaSolveMode::Affine {
            spatial_information_prepared(observations, &scratch.base_weights, robust_weights)
        } else {
            (0, 1.0)
        };
    if robust_rank < 3
        || (mode == JointCfaSolveMode::Affine && robust_spatial_rank < 2)
    {
        return None;
    }
    let mut camera_weights = [0.0_f32; u32::BITS as usize];
    for (index, observation) in observations.iter().enumerate() {
        if observation.camera_index < camera_weights.len() {
            camera_weights[observation.camera_index] +=
                scratch.base_weights[index] * scratch.robust_weights[index];
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

    let mut centroid = [0.0_f32; 2];
    let mut closest_count = 0usize;
    for (_, offset) in closest.iter().flatten() {
        centroid[0] += offset[0];
        centroid[1] += offset[1];
        closest_count += 1;
    }
    centroid = centroid.map(|value| value / closest_count.max(1) as f32);
    let mut phase_spread_sum = 0.0_f32;
    for (_, offset) in closest.iter().flatten() {
        phase_spread_sum += (offset[0] - centroid[0]).powi(2) + (offset[1] - centroid[1]).powi(2);
    }
    let phase_spread = (phase_spread_sum / closest_count.max(1) as f32).sqrt();

    let information_confidence = initial_information_confidence
        .min(robust_information_confidence)
        .min(initial_spatial_confidence)
        .min(robust_spatial_confidence);
    let mean_baseline_loss = baseline_loss / weight_sum.max(1.0e-8);
    let mean_joint_loss = joint_loss / weight_sum.max(1.0e-8);
    let relative_fit = if mean_baseline_loss <= 1.0e-8 && mean_joint_loss <= 1.0e-8 {
        // The spatially matched baseline can already be exact in diagnostic
        // tests. That is a good fit, not a zero-confidence divide-by-zero case.
        1.0
    } else {
        ((mean_baseline_loss - mean_joint_loss) / mean_baseline_loss.max(1.0e-8))
            .clamp(0.0, 1.0)
    };
    // Conditioning says whether the equations constrain XYZ; fit confidence
    // says whether the constrained model actually explains these measurements.
    // The absolute-loss term is intentionally gentle for good fits but pushes
    // noisy/contradictory solves down before they can become visible detail.
    let absolute_fit = 1.0 / (1.0 + 0.25 * mean_joint_loss.max(0.0));
    let fit_confidence = (relative_fit * absolute_fit).clamp(0.0, 1.0);
    if fit_confidence < 0.05 {
        return None;
    }
    let base_application_weight = information_confidence.sqrt() * fit_confidence;
    Some(JointCfaEstimate {
        xyz: [
            estimate[0].max(0.0),
            estimate[1].max(0.0),
            estimate[2].max(0.0),
        ],
        application_weight: base_application_weight,
        chroma_application_weight: 0.0,
        report: JointCfaSolveReport {
            observations: valid_count,
            mode,
            cameras: retained_camera_mask.count_ones() as usize,
            phase_mask,
            phase_spread,
            iterations,
            weighted_residual: residual_sum / weight_sum.max(1.0e-8),
            in_sample_baseline_loss: mean_baseline_loss,
            in_sample_affine_loss: mean_joint_loss,
            data_rank: robust_rank,
            information_confidence,
            fit_confidence,
            model_rank: robust_model_rank,
            spatial_rank: robust_spatial_rank,
        },
        model: estimate,
        base_application_weight,
        preserve_baseline_luminance: false,
    })
}

/// Rank and a scale-free conditioning score for the unregularized centre-XYZ
/// response information. Ridge priors deliberately do not participate: they
/// may stabilize unsupported directions, but must not be reported as sensor
/// evidence.
fn response_information_prepared(
    observations: &[CfaObservation],
    base_weights: &[f32],
    robust_weights: Option<&[f32]>,
) -> (usize, f32) {
    let mut information = [[0.0_f64; 3]; 3];
    for (index, observation) in observations.iter().enumerate() {
        let robust = robust_weights.map_or(1.0, |weights| weights[index]);
        let weight = base_weights[index] * robust;
        if weight <= 0.0 {
            continue;
        }
        accumulate_information_upper(&mut information, observation.response, weight);
    }
    mirror_upper_triangle(&mut information, 3);
    information_rank_matrix(information)
}

fn model_information_prepared(
    observations: &[CfaObservation],
    designs: &[[f32; MODEL_SIZE]],
    base_weights: &[f32],
    robust_weights: Option<&[f32]>,
) -> (usize, f32) {
    let mut information = [[0.0_f64; MODEL_SIZE]; MODEL_SIZE];
    for index in 0..observations.len() {
        let robust = robust_weights.map_or(1.0, |weights| weights[index]);
        let weight = base_weights[index] * robust;
        if weight <= 0.0 {
            continue;
        }
        accumulate_information_upper(&mut information, designs[index], weight);
    }
    mirror_upper_triangle(&mut information, MODEL_SIZE);
    information_rank_matrix(information)
}

fn spatial_information_prepared(
    observations: &[CfaObservation],
    base_weights: &[f32],
    robust_weights: Option<&[f32]>,
) -> (usize, f32) {
    let total_weight = observations
        .iter()
        .enumerate()
        .map(|(index, _)| {
            base_weights[index] * robust_weights.map_or(1.0, |weights| weights[index])
        })
        .sum::<f32>()
        .max(1.0e-10);
    let mut mean = [0.0_f32; 2];
    for (index, observation) in observations.iter().enumerate() {
        let weight =
            base_weights[index] * robust_weights.map_or(1.0, |weights| weights[index]);
        mean[0] += weight * observation.output_offset[0] / total_weight;
        mean[1] += weight * observation.output_offset[1] / total_weight;
    }
    let mut information = [[0.0_f64; 2]; 2];
    for (index, observation) in observations.iter().enumerate() {
        let robust = robust_weights.map_or(1.0, |weights| weights[index]);
        let weight = base_weights[index] * robust;
        if weight <= 0.0 {
            continue;
        }
        accumulate_information_upper(
            &mut information,
            [
                observation.output_offset[0] - mean[0],
                observation.output_offset[1] - mean[1],
            ],
            weight,
        );
    }
    mirror_upper_triangle(&mut information, 2);
    information_rank_matrix(information)
}

#[inline]
fn accumulate_information_upper<const N: usize>(
    information: &mut [[f64; N]; N],
    design: [f32; N],
    weight: f32,
) {
    for row in 0..N {
        let weighted = f64::from(weight * design[row]);
        for column in row..N {
            information[row][column] += weighted * f64::from(design[column]);
        }
    }
}

#[inline]
fn accumulate_normal_upper(
    normal: &mut [[f64; MODEL_SIZE]; MODEL_SIZE],
    rhs: &mut [f64; MODEL_SIZE],
    design: [f32; MODEL_SIZE],
    value: f32,
    weight: f32,
    parameter_count: usize,
) {
    let value = f64::from(value);
    for row in 0..parameter_count {
        let weighted = f64::from(weight * design[row]);
        rhs[row] += weighted * value;
        for column in row..parameter_count {
            normal[row][column] += weighted * f64::from(design[column]);
        }
    }
}

#[inline]
fn mirror_upper_triangle<const N: usize>(matrix: &mut [[f64; N]; N], parameter_count: usize) {
    for row in 0..parameter_count {
        for column in row + 1..parameter_count {
            matrix[column][row] = matrix[row][column];
        }
    }
}

fn information_rank_matrix<const N: usize>(mut information: [[f64; N]; N]) -> (usize, f32) {
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

#[inline]
fn dot_model_prefix(
    first: [f32; MODEL_SIZE],
    second: [f32; MODEL_SIZE],
    parameter_count: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for index in 0..parameter_count {
        sum += first[index] * second[index];
    }
    sum
}

/// Solve the positive-definite normal equations with a Cholesky factorisation.
/// Ridge regularisation guarantees a strictly positive diagonal for supported
/// systems, so this is both cheaper and more cache-friendly than the previous
/// full Gauss-Jordan elimination.
fn solve_spd_prefix(
    matrix: [[f64; MODEL_SIZE]; MODEL_SIZE],
    rhs: [f64; MODEL_SIZE],
    parameter_count: usize,
) -> Option<[f64; MODEL_SIZE]> {
    let mut lower = [[0.0_f64; MODEL_SIZE]; MODEL_SIZE];
    for row in 0..parameter_count {
        for column in 0..=row {
            let mut sum = matrix[row][column];
            for k in 0..column {
                sum -= lower[row][k] * lower[column][k];
            }
            if row == column {
                if !sum.is_finite() || sum <= 1.0e-18 {
                    return None;
                }
                lower[row][column] = sum.sqrt();
            } else {
                let diagonal = lower[column][column];
                if !diagonal.is_finite() || diagonal.abs() <= 1.0e-18 {
                    return None;
                }
                lower[row][column] = sum / diagonal;
            }
        }
    }

    let mut y = [0.0_f64; MODEL_SIZE];
    for row in 0..parameter_count {
        let mut sum = rhs[row];
        for column in 0..row {
            sum -= lower[row][column] * y[column];
        }
        y[row] = sum / lower[row][row];
    }
    let mut solution = [0.0_f64; MODEL_SIZE];
    for row in (0..parameter_count).rev() {
        let mut sum = y[row];
        for column in row + 1..parameter_count {
            sum -= lower[column][row] * solution[column];
        }
        solution[row] = sum / lower[row][row];
    }
    Some(solution)
}

fn observation_weight(observation: &CfaObservation) -> f32 {
    let visibility_weight = match observation.visibility {
        Visibility::Visible => 1.0,
        // A globally aligned location without independent depth support is
        // still useful evidence, but it must not carry the same authority as
        // a view whose selected scene surface was explicitly verified.
        Visibility::Unknown => 0.35,
        Visibility::Occluded => 0.0,
    };
    if visibility_weight <= 0.0
        || !observation.value.is_finite()
        || !observation.noise_variance.is_finite()
        || observation.noise_variance <= 0.0
    {
        return 0.0;
    }
    visibility_weight
        * observation.spatial_weight.max(0.0)
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
    let mut scratch = CfaSolverScratch::default();
    account_shared_sample_dependence_with_scratch(observations, &mut scratch);
}

pub fn account_shared_sample_dependence_with_scratch(
    observations: &mut [CfaObservation],
    scratch: &mut CfaSolverScratch,
) {
    let count = observations.len();
    scratch.original_variance.clear();
    scratch.original_variance.extend(
        observations
            .iter()
            .map(|observation| observation.noise_variance.max(1.0e-10)),
    );
    scratch.correlation_mass.clear();
    scratch.correlation_mass.resize(count, 0.0);
    scratch.pair_covariance.clear();
    scratch
        .pair_covariance
        .resize(count.saturating_mul(count), 0.0);
    scratch.dependency_uses.clear();
    for (observation, item) in observations.iter().enumerate() {
        for dependency in &item.noise_dependencies[..item.noise_dependency_count] {
            scratch.dependency_uses.push(DependencyUse {
                key: dependency.key,
                observation,
                coefficient: dependency.coefficient,
                physical_variance: dependency.physical_variance,
            });
        }
    }
    scratch
        .dependency_uses
        .sort_unstable_by_key(|dependency| dependency.key);
    let mut start = 0usize;
    while start < scratch.dependency_uses.len() {
        let key = scratch.dependency_uses[start].key;
        let mut end = start + 1;
        while end < scratch.dependency_uses.len() && scratch.dependency_uses[end].key == key {
            end += 1;
        }
        let group = &scratch.dependency_uses[start..end];
        for left_index in 0..group.len() {
            let left = group[left_index];
            for &right in &group[left_index + 1..] {
                if left.observation == right.observation {
                    continue;
                }
                let (a, b) = if left.observation < right.observation {
                    (left, right)
                } else {
                    (right, left)
                };
                scratch.pair_covariance[a.observation * count + b.observation] += a.coefficient
                    * b.coefficient
                    * 0.5
                    * (a.physical_variance + b.physical_variance);
            }
        }
        start = end;
    }
    for left in 0..count {
        for right in left + 1..count {
            let covariance = scratch.pair_covariance[left * count + right];
            let correlation = (covariance
                / (scratch.original_variance[left] * scratch.original_variance[right]).sqrt())
            .abs()
            .min(1.0);
            scratch.correlation_mass[left] += correlation;
            scratch.correlation_mass[right] += correlation;
        }
    }
    for index in 0..count {
        observations[index].noise_variance =
            scratch.original_variance[index] * (1.0 + scratch.correlation_mass[index]);
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
pub type CameraResponseBase = [[f32; 3]; 3];

/// Camera-response factor independent of the spatial flat-field gain.
/// `(diag(field) * forward * diag(wb))^-1` factors into
/// `diag(wb)^-1 * forward^-1 * diag(field)^-1`, so the expensive 3x3 inverse
/// is required only once per camera rather than once per CFA observation.
pub fn camera_response_base(color: &ModuleColor) -> Option<CameraResponseBase> {
    if !color.calibrated
        || color
            .wb_gains
            .iter()
            .any(|gain| !gain.is_finite() || gain.abs() < 1.0e-12)
    {
        return None;
    }
    let forward = color.forward.map(|row| row.map(f64::from));
    let inverse_forward = inverse(&forward)?;
    Some(std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (inverse_forward[row][column] / f64::from(color.wb_gains[row])) as f32
        })
    }))
}

#[inline]
pub fn camera_response_from_base(
    base: &CameraResponseBase,
    xyz_field_gain: [f32; 3],
    phase: CfaPhase,
) -> Option<[f32; 3]> {
    if xyz_field_gain
        .iter()
        .any(|gain| !gain.is_finite() || gain.abs() < 1.0e-12)
    {
        return None;
    }
    let row = base[phase.color_channel()];
    Some(std::array::from_fn(|column| {
        row[column] / xyz_field_gain[column]
    }))
}

pub fn camera_response(
    color: &ModuleColor,
    xyz_field_gain: [f32; 3],
    phase: CfaPhase,
) -> Option<[f32; 3]> {
    let base = camera_response_base(color)?;
    camera_response_from_base(&base, xyz_field_gain, phase)
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
    fn visibility_controls_measurement_authority() {
        let visible = observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R);
        let mut unknown = visible.clone();
        unknown.visibility = Visibility::Unknown;
        let mut occluded = visible.clone();
        occluded.visibility = Visibility::Occluded;
        assert!(observation_weight(&visible) > observation_weight(&unknown));
        assert!(observation_weight(&unknown) > 0.0);
        assert_eq!(observation_weight(&occluded), 0.0);
    }

    #[test]
    fn factored_camera_response_matches_direct_matrix_inverse() {
        let color = ModuleColor {
            wb_gains: [1.7, 1.05, 1.35],
            forward: [[0.72, 0.18, 0.04], [0.12, 0.81, 0.09], [0.03, 0.16, 0.77]],
            calibrated: true,
        };
        let field = [0.83, 1.12, 0.94];
        let mut camera_to_xyz = [[0.0_f64; 3]; 3];
        for (row, values) in camera_to_xyz.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value =
                    f64::from(field[row] * color.forward[row][column] * color.wb_gains[column]);
            }
        }
        let direct = inverse(&camera_to_xyz).unwrap();
        let base = camera_response_base(&color).unwrap();
        for phase in [CfaPhase::R, CfaPhase::Gr, CfaPhase::Gb, CfaPhase::B] {
            let factored = camera_response_from_base(&base, field, phase).unwrap();
            for (actual, expected) in factored.into_iter().zip(direct[phase.color_channel()]) {
                assert!((f64::from(actual) - expected).abs() < 2.0e-7);
            }
        }
    }

    #[test]
    fn joint_cfa_scratch_can_be_reused_without_state_leakage() {
        let make_observations = |scale: f32| {
            let mut observations = Vec::new();
            for (channel, (value, phase)) in
                [(0.2, CfaPhase::R), (0.4, CfaPhase::Gr), (0.6, CfaPhase::B)]
                    .into_iter()
                    .enumerate()
            {
                let mut response = [0.0; 3];
                response[channel] = 1.0;
                for offset in [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
                    let mut sample = observation(response, value * scale, phase);
                    sample.output_offset = offset;
                    observations.push(sample);
                }
            }
            observations
        };
        let mut scratch = CfaSolverScratch::default();
        let first = make_observations(1.0);
        solve_joint_xyz_with_scratch(&first, [0.0; 3], 1.0e-6, &mut scratch).unwrap();

        let second = make_observations(0.5);
        let reused = solve_joint_xyz_with_scratch(&second, [0.0; 3], 1.0e-6, &mut scratch).unwrap();
        let fresh = solve_joint_xyz(&second, [0.0; 3], 1.0e-6).unwrap();
        assert_eq!(reused.xyz, fresh.xyz);
        assert_eq!(reused.report.observations, fresh.report.observations);
        assert_eq!(reused.report.cameras, fresh.report.cameras);
        assert_eq!(reused.report.phase_mask, fresh.report.phase_mask);
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
    fn constant_tier_requires_colour_rank_but_not_spatial_rank() {
        let observations = [
            observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
            observation([0.0, 1.0, 0.0], 0.4, CfaPhase::Gr),
            observation([0.0, 0.0, 1.0], 0.6, CfaPhase::B),
            observation([1.0, 0.0, 0.0], 0.2, CfaPhase::R),
            observation([0.0, 1.0, 0.0], 0.4, CfaPhase::Gr),
            observation([0.0, 0.0, 1.0], 0.6, CfaPhase::B),
        ];
        let mut scratch = CfaSolverScratch::default();
        let estimate = solve_joint_xyz_mode_with_scratch(
            &observations,
            [0.0; 3],
            1.0e-6,
            JointCfaSolveMode::Constant,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(estimate.report.mode, JointCfaSolveMode::Constant);
        assert_eq!(estimate.report.model_rank, 3);
        assert_eq!(estimate.report.spatial_rank, 0);
        for (actual, expected) in estimate.xyz.into_iter().zip([0.2, 0.4, 0.6]) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
        // The full affine model correctly rejects the same coincident sites.
        assert!(solve_joint_xyz(&observations, [0.0; 3], 1.0e-6).is_none());
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
        estimate.apply_over_baseline(baseline, 1.0, true, 1.0);
        assert!((estimate.xyz[1] - baseline[1]).abs() < 1.0e-6);
        assert!((estimate.xyz[0] - baseline[0]).abs() > 1.0e-3);
    }

    #[test]
    fn production_application_changes_luminance_without_chromaticity() {
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
        let baseline = [0.12, 0.30, 0.48];
        let mut estimate = solve_joint_xyz(&observations, baseline, 1.0e-6).unwrap();
        estimate.apply_over_baseline(baseline, 1.0, false, 0.0);
        assert!((estimate.xyz[1] - baseline[1]).abs() > 1.0e-3);
        assert!((estimate.xyz[0] / estimate.xyz[1] - baseline[0] / baseline[1]).abs() < 1.0e-6);
        assert!((estimate.xyz[2] / estimate.xyz[1] - baseline[2] / baseline[1]).abs() < 1.0e-6);
        assert_eq!(estimate.chroma_application_weight, 0.0);
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
