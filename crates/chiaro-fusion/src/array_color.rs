//! Sparse, array-aware selection of the L16 factory colour profiles.
//!
//! Correlated colour errors matter to a camera array differently from absolute
//! single-camera error.  This module keeps capture AWB/CCT as a soft prior but
//! scores A/F11/D65 blends from aligned, reliable overlap observations before
//! the expensive demosaic and synthesis stages.

use chiaro_hotpixel_core::highlight::HighlightRecoveryState;
use serde::Serialize;

use crate::{
    align::ModuleAlignment,
    calibration::{CameraCalibration, ColorProfile},
    depth::{DenseDepthMap, DepthProvenance},
    image::Mosaic,
    synth::{ModuleColor, chroma_distance},
};

pub const ILLUMINANT_A: i32 = 0;
pub const ILLUMINANT_D65: i32 = 2;
pub const ILLUMINANT_F11: i32 = 6;
const ILLUMINANTS: [i32; 3] = [ILLUMINANT_A, ILLUMINANT_F11, ILLUMINANT_D65];
const SIMPLEX_STEPS: usize = 20;
const PRIOR_WEIGHT: f32 = 0.000_20;
const MINIMUM_SAMPLES: usize = 384;
const MINIMUM_TARGET_MODULES: usize = 2;
const MINIMUM_SELECTION_CONFIDENCE: f32 = 0.05;
const MINIMUM_RELATIVE_IMPROVEMENT: f32 = 0.01;
const COVERAGE_COLUMNS: usize = 8;
const COVERAGE_ROWS: usize = 6;
const CHROMA_RATIO_EPSILON: f32 = 1.0e-4;
const CHROMATIC_RATIO_FLOOR: f32 = 0.01;
const MAX_SCATTER_POINTS: usize = 1_024;
const D50_WHITE: [f32; 3] = [0.964_22, 1.0, 0.825_21];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorProfileMode {
    ArrayAware,
    CctOnly,
    ForceA,
    ForceF11,
    #[default]
    ForceD65,
}

impl ColorProfileMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::ArrayAware => "array_aware",
            Self::CctOnly => "cct_only",
            Self::ForceA => "forced_a",
            Self::ForceF11 => "forced_f11",
            Self::ForceD65 => "forced_d65",
        }
    }
}

/// Non-negative A/F11/D65 weights which sum to one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfileBlend {
    pub weights: [f32; 3],
}

impl ProfileBlend {
    pub const A: Self = Self {
        weights: [1.0, 0.0, 0.0],
    };
    pub const F11: Self = Self {
        weights: [0.0, 1.0, 0.0],
    };
    pub const D65: Self = Self {
        weights: [0.0, 0.0, 1.0],
    };

    fn new(weights: [f32; 3]) -> Self {
        let sum = weights.iter().sum::<f32>();
        if sum <= 0.0 || !sum.is_finite() {
            return Self::D65;
        }
        Self {
            weights: weights.map(|weight| (weight / sum).max(0.0)),
        }
    }

    pub fn named_weights(self) -> Vec<(String, f32)> {
        ILLUMINANTS
            .into_iter()
            .zip(self.weights)
            .filter(|(_, weight)| *weight > 1.0e-5)
            .map(|(illuminant, weight)| {
                (
                    crate::color_profile::illuminant_name(Some(illuminant)).to_owned(),
                    weight,
                )
            })
            .collect()
    }

    fn distance(self, other: Self) -> f32 {
        self.weights
            .into_iter()
            .zip(other.weights)
            .map(|(first, second)| (first - second).abs())
            .sum()
    }

    fn prior_penalty(self, prior: Self) -> f32 {
        self.weights
            .into_iter()
            .zip(prior.weights)
            .map(|(candidate, expected)| (candidate - expected).powi(2))
            .sum::<f32>()
            * 0.5
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfileCandidateReport {
    pub weights: Vec<(String, f32)>,
    pub array_disagreement: f32,
    pub cct_prior_penalty: f32,
    pub total_score: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChromaDistributionReport {
    pub count: usize,
    pub mean: f32,
    pub p10: f32,
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p90: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RelativeChromaReport {
    pub count: usize,
    pub p10: f32,
    pub p50: f32,
    pub p90: f32,
}

/// Report-only measurements used to detect a profile winning by flattening
/// colour. These values never participate in profile selection.
#[derive(Clone, Debug, Serialize)]
pub struct ProfileChromaDiagnostic {
    pub illuminant: &'static str,
    pub chroma: ChromaDistributionReport,
    pub relative_to_d65: RelativeChromaReport,
    pub relative_to_d65_chromatic: RelativeChromaReport,
    pub mean_inter_module_disagreement: f32,
    pub disagreement_normalized_by_mean_chroma: f32,
    pub selector_disagreement: Option<f32>,
    /// `[D65 chroma, candidate chroma]`, deterministically subsampled.
    pub scatter_vs_d65: Vec<[f32; 2]>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArrayColorSelectionReport {
    pub mode: &'static str,
    pub prior_weights: Vec<(String, f32)>,
    pub selected_weights: Vec<(String, f32)>,
    pub estimated_mired: Option<f32>,
    pub evaluated_candidates: usize,
    pub best_candidate: Option<ProfileCandidateReport>,
    pub second_best_candidate: Option<ProfileCandidateReport>,
    pub score_difference: Option<f32>,
    pub prior_array_disagreement: Option<f32>,
    pub selected_array_disagreement: Option<f32>,
    pub sample_count: usize,
    pub target_modules: usize,
    pub module_samples: Vec<(String, usize)>,
    pub spatial_cells: usize,
    pub spatial_coverage: f32,
    pub confidence: f32,
    pub used_array_evidence: bool,
    pub fallback_reason: Option<String>,
    pub profile_chroma_diagnostics: Vec<ProfileChromaDiagnostic>,
}

pub struct ArrayColorSource<'a> {
    pub name: &'a str,
    pub mosaic: &'a Mosaic,
    pub highlight: &'a HighlightRecoveryState,
    pub alignment: &'a ModuleAlignment,
    pub calibration: Option<&'a CameraCalibration>,
}

pub struct ArrayColorSelection {
    pub blend: ProfileBlend,
    pub report: ArrayColorSelectionReport,
}

#[derive(Clone, Copy, Debug)]
pub struct BlendedProfile {
    pub matrix: [[f64; 3]; 3],
    pub rg_ratio: f64,
    pub bg_ratio: f64,
    pub uses_validated_matrix: bool,
}

#[derive(Clone, Copy)]
struct Observation {
    target: usize,
    reference_rgb: [f32; 3],
    target_rgb: [f32; 3],
    cell: usize,
}

#[derive(Clone)]
struct ScoredCandidate {
    blend: ProfileBlend,
    array: f32,
    prior: f32,
    total: f32,
}

impl ScoredCandidate {
    fn report(&self) -> ProfileCandidateReport {
        ProfileCandidateReport {
            weights: self.blend.named_weights(),
            array_disagreement: self.array,
            cct_prior_penalty: self.prior,
            total_score: self.total,
        }
    }
}

fn illuminant_mired(illuminant: i32) -> Option<f64> {
    match illuminant {
        ILLUMINANT_A => Some(1_000_000.0 / 2_856.0),
        ILLUMINANT_D65 => Some(1_000_000.0 / 6_504.0),
        ILLUMINANT_F11 => Some(1_000_000.0 / 4_000.0),
        _ => None,
    }
}

/// Existing neutral-ratio/CCT estimate, represented in the same simplex as
/// the array search.  It remains the unconditional fallback.
pub fn cct_prior(
    reference: Option<&CameraCalibration>,
    recorded_wb: Option<[f32; 3]>,
) -> (ProfileBlend, Option<f32>, f32) {
    let Some(reference) = reference else {
        return (ProfileBlend::D65, None, 0.0);
    };
    let available = |illuminant| {
        reference
            .color
            .iter()
            .any(|profile| profile.illuminant == illuminant)
    };
    let fallback = if available(ILLUMINANT_D65) {
        ProfileBlend::D65
    } else if available(ILLUMINANT_F11) {
        ProfileBlend::F11
    } else {
        ProfileBlend::A
    };
    let Some(wb) = recorded_wb else {
        let mired =
            ILLUMINANTS
                .into_iter()
                .zip(fallback.weights)
                .find_map(|(illuminant, weight)| {
                    (weight > 0.0)
                        .then(|| illuminant_mired(illuminant))
                        .flatten()
                });
        return (fallback, mired.map(|value| value as f32), 0.5);
    };
    if wb.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return (fallback, None, 0.0);
    }
    let target = [f64::from(wb[1] / wb[0]).ln(), f64::from(wb[1] / wb[2]).ln()];
    let mut anchors = ILLUMINANTS
        .into_iter()
        .enumerate()
        .filter_map(|(index, illuminant)| {
            let profile = reference
                .color
                .iter()
                .find(|profile| profile.illuminant == illuminant)?;
            Some((
                illuminant_mired(illuminant)?,
                index,
                [profile.rg_ratio.ln(), profile.bg_ratio.ln()],
            ))
        })
        .collect::<Vec<_>>();
    anchors.sort_by(|first, second| first.0.total_cmp(&second.0));
    if anchors.is_empty() {
        return (fallback, None, 0.0);
    }
    if anchors.len() == 1 {
        let mut weights = [0.0; 3];
        weights[anchors[0].1] = 1.0;
        return (ProfileBlend::new(weights), Some(anchors[0].0 as f32), 0.25);
    }
    anchors
        .windows(2)
        .map(|pair| {
            let (left, right) = (pair[0], pair[1]);
            let direction = [right.2[0] - left.2[0], right.2[1] - left.2[1]];
            let relative = [target[0] - left.2[0], target[1] - left.2[1]];
            let denominator = direction[0].powi(2) + direction[1].powi(2);
            let weight = if denominator > 1.0e-12 {
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
            let mut weights = [0.0; 3];
            weights[left.1] = (1.0 - weight) as f32;
            weights[right.1] = weight as f32;
            (
                distance,
                ProfileBlend::new(weights),
                (left.0 + (right.0 - left.0) * weight) as f32,
                (1.0 / (1.0 + 4.0 * distance)) as f32,
            )
        })
        .min_by(|first, second| first.0.total_cmp(&second.0))
        .map(|(_, blend, mired, confidence)| (blend, Some(mired), confidence))
        .unwrap_or((fallback, None, 0.0))
}

pub fn blended_profile(
    calibration: Option<&CameraCalibration>,
    blend: ProfileBlend,
) -> Option<BlendedProfile> {
    let calibration = calibration?;
    let mut profiles = Vec::<(&ColorProfile, f64)>::new();
    for (illuminant, weight) in ILLUMINANTS.into_iter().zip(blend.weights) {
        if weight <= 1.0e-5 {
            continue;
        }
        let profile = calibration
            .color
            .iter()
            .find(|profile| profile.illuminant == illuminant)?;
        profiles.push((profile, f64::from(weight)));
    }
    if profiles.is_empty() {
        return None;
    }
    let mut matrix = [[0.0; 3]; 3];
    let mut log_rg = 0.0;
    let mut log_bg = 0.0;
    let mut validated = false;
    for (profile, weight) in profiles {
        let source = profile
            .validated_matrix
            .as_ref()
            .unwrap_or(&profile.forward_matrix);
        for row in 0..3 {
            for column in 0..3 {
                matrix[row][column] += source[row][column] * weight;
            }
        }
        log_rg += profile.rg_ratio.max(1.0e-6).ln() * weight;
        log_bg += profile.bg_ratio.max(1.0e-6).ln() * weight;
        validated |= profile.validated_matrix.is_some();
    }
    Some(BlendedProfile {
        matrix,
        rg_ratio: log_rg.exp(),
        bg_ratio: log_bg.exp(),
        uses_validated_matrix: validated,
    })
}

pub fn module_color_for_blend(
    module: Option<&CameraCalibration>,
    reference: Option<&CameraCalibration>,
    recorded_wb: Option<[f32; 3]>,
    blend: ProfileBlend,
) -> Option<ModuleColor> {
    let profile = blended_profile(module, blend)?;
    let reference_profile = blended_profile(reference, blend);
    let mut color = ModuleColor {
        forward: profile.matrix.map(|row| row.map(|value| value as f32)),
        calibrated: true,
        ..ModuleColor::default()
    };
    color.wb_gains = match (recorded_wb, reference_profile) {
        (Some(wb), Some(reference)) => [
            wb[0] * (reference.rg_ratio / profile.rg_ratio.max(1.0e-3)) as f32,
            wb[1],
            wb[2] * (reference.bg_ratio / profile.bg_ratio.max(1.0e-3)) as f32,
        ],
        (Some(wb), None) => wb,
        (None, _) => [
            (1.0 / profile.rg_ratio.max(0.01)) as f32,
            1.0,
            (1.0 / profile.bg_ratio.max(0.01)) as f32,
        ],
    };
    Some(color)
}

pub fn select_array_profile(
    sources: &[ArrayColorSource<'_>],
    reference_index: usize,
    reference_width: usize,
    reference_height: usize,
    depth: Option<&DenseDepthMap>,
    recorded_wb: Option<[f32; 3]>,
    mode: ColorProfileMode,
) -> ArrayColorSelection {
    let reference_calibration = sources
        .get(reference_index)
        .and_then(|source| source.calibration);
    let (prior, estimated_mired, prior_confidence) = cct_prior(reference_calibration, recorded_wb);
    let observations = collect_observations(
        sources,
        reference_index,
        reference_width,
        reference_height,
        depth,
    );
    let profile_chroma_diagnostics =
        profile_chroma_diagnostics(sources, reference_index, recorded_wb, &observations);
    let target_modules = {
        let mut present = vec![false; sources.len()];
        for observation in &observations {
            present[observation.target] = true;
        }
        present.into_iter().filter(|present| *present).count()
    };
    let module_samples = sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| {
            let count = observations
                .iter()
                .filter(|observation| observation.target == index)
                .count();
            (count > 0).then(|| (source.name.to_owned(), count))
        })
        .collect::<Vec<_>>();
    let spatial_cells = {
        let mut cells = [false; COVERAGE_COLUMNS * COVERAGE_ROWS];
        for observation in &observations {
            cells[observation.cell] = true;
        }
        cells.into_iter().filter(|present| *present).count()
    };
    let spatial_coverage = spatial_cells as f32 / (COVERAGE_COLUMNS * COVERAGE_ROWS) as f32;

    let forced = match mode {
        ColorProfileMode::ForceA => Some(ProfileBlend::A),
        ColorProfileMode::ForceF11 => Some(ProfileBlend::F11),
        ColorProfileMode::ForceD65 => Some(ProfileBlend::D65),
        _ => None,
    };
    if let Some(blend) = forced {
        let array = score_blend(blend, sources, reference_index, recorded_wb, &observations);
        let prior_array = score_blend(prior, sources, reference_index, recorded_wb, &observations);
        return selection_result(
            blend,
            prior,
            estimated_mired,
            mode,
            Vec::new(),
            prior_array,
            array,
            observations.len(),
            target_modules,
            module_samples.clone(),
            spatial_cells,
            spatial_coverage,
            1.0,
            false,
            None,
            profile_chroma_diagnostics,
        );
    }
    if mode == ColorProfileMode::CctOnly {
        let array = score_blend(prior, sources, reference_index, recorded_wb, &observations);
        return selection_result(
            prior,
            prior,
            estimated_mired,
            mode,
            Vec::new(),
            array,
            array,
            observations.len(),
            target_modules,
            module_samples.clone(),
            spatial_cells,
            spatial_coverage,
            prior_confidence,
            false,
            Some("array-aware selection disabled".to_owned()),
            profile_chroma_diagnostics,
        );
    }

    let candidates = simplex_grid()
        .into_iter()
        .filter_map(|blend| {
            let array = score_blend(blend, sources, reference_index, recorded_wb, &observations)?;
            let prior_penalty = blend.prior_penalty(prior);
            Some(ScoredCandidate {
                blend,
                array,
                prior: prior_penalty,
                total: array + PRIOR_WEIGHT * prior_penalty,
            })
        })
        .collect::<Vec<_>>();
    let prior_array = score_blend(prior, sources, reference_index, recorded_wb, &observations);
    let enough_evidence = observations.len() >= MINIMUM_SAMPLES
        && target_modules >= MINIMUM_TARGET_MODULES
        && spatial_cells >= 8;
    if candidates.is_empty() || !enough_evidence {
        let reason = if candidates.is_empty() {
            "no common A/F11/D65 calibration across aligned colour modules"
        } else if observations.len() < MINIMUM_SAMPLES {
            "too few reliable aligned colour samples"
        } else if target_modules < MINIMUM_TARGET_MODULES {
            "too few independently aligned colour modules"
        } else {
            "insufficient spatial overlap coverage"
        };
        return selection_result(
            prior,
            prior,
            estimated_mired,
            mode,
            candidates,
            prior_array,
            prior_array,
            observations.len(),
            target_modules,
            module_samples.clone(),
            spatial_cells,
            spatial_coverage,
            0.0,
            false,
            Some(reason.to_owned()),
            profile_chroma_diagnostics,
        );
    }

    let mut ranked = candidates;
    ranked.sort_by(|first, second| first.total.total_cmp(&second.total));
    let best = &ranked[0];
    // The literal runner-up is normally an adjacent 5% grid point. Confidence
    // therefore uses the strongest materially different competitor instead of
    // pretending that a smooth optimum should have a discontinuous score gap.
    let distinct = ranked
        .iter()
        .skip(1)
        .find(|candidate| candidate.blend.distance(best.blend) >= 0.30)
        .unwrap_or_else(|| ranked.get(1).unwrap_or(best));
    let distinct_gap = (distinct.array - best.array).max(0.0);
    let improvement = prior_array.map_or(0.0, |score| (score - best.array).max(0.0));
    let support = (observations.len() as f32 / 2_000.0).min(1.0)
        * (target_modules as f32 / 4.0).min(1.0)
        * (spatial_coverage / 0.40).min(1.0);
    let separation = (distinct_gap / (best.array * 0.04).max(0.000_15)).clamp(0.0, 1.0);
    let relative_improvement = improvement / best.array.max(1.0e-5);
    let improvement_strength =
        (relative_improvement / (MINIMUM_RELATIVE_IMPROVEMENT * 5.0)).clamp(0.0, 1.0);
    let confidence = support * separation.max(improvement_strength);
    let moved_materially = best.blend.distance(prior) >= 0.20;
    let improvement_is_real = relative_improvement >= MINIMUM_RELATIVE_IMPROVEMENT;
    let (selected, used_array, fallback) =
        if !array_override_supported(confidence, moved_materially, improvement_is_real) {
            (
                prior,
                false,
                Some("candidate scores are too similar to override the CCT prior".to_owned()),
            )
        } else {
            (best.blend, true, None)
        };
    let selected_array = if selected == best.blend {
        Some(best.array)
    } else {
        prior_array
    };
    selection_result(
        selected,
        prior,
        estimated_mired,
        mode,
        ranked,
        prior_array,
        selected_array,
        observations.len(),
        target_modules,
        module_samples,
        spatial_cells,
        spatial_coverage,
        confidence,
        used_array,
        fallback,
        profile_chroma_diagnostics,
    )
}

fn array_override_supported(
    confidence: f32,
    moved_materially: bool,
    improvement_is_real: bool,
) -> bool {
    confidence >= MINIMUM_SELECTION_CONFIDENCE && (!moved_materially || improvement_is_real)
}

#[allow(clippy::too_many_arguments)]
fn selection_result(
    blend: ProfileBlend,
    prior: ProfileBlend,
    estimated_mired: Option<f32>,
    mode: ColorProfileMode,
    candidates: Vec<ScoredCandidate>,
    prior_array: Option<f32>,
    selected_array: Option<f32>,
    sample_count: usize,
    target_modules: usize,
    module_samples: Vec<(String, usize)>,
    spatial_cells: usize,
    spatial_coverage: f32,
    confidence: f32,
    used_array_evidence: bool,
    fallback_reason: Option<String>,
    profile_chroma_diagnostics: Vec<ProfileChromaDiagnostic>,
) -> ArrayColorSelection {
    let best = candidates.first();
    let second = candidates.get(1);
    ArrayColorSelection {
        blend,
        report: ArrayColorSelectionReport {
            mode: mode.name(),
            prior_weights: prior.named_weights(),
            selected_weights: blend.named_weights(),
            estimated_mired,
            evaluated_candidates: candidates.len(),
            best_candidate: best.map(ScoredCandidate::report),
            second_best_candidate: second.map(ScoredCandidate::report),
            score_difference: best
                .zip(second)
                .map(|(best, second)| (second.total - best.total).max(0.0)),
            prior_array_disagreement: prior_array,
            selected_array_disagreement: selected_array,
            sample_count,
            target_modules,
            module_samples,
            spatial_cells,
            spatial_coverage,
            confidence,
            used_array_evidence,
            fallback_reason,
            profile_chroma_diagnostics,
        },
    }
}

#[derive(Clone, Copy)]
struct ChromaMeasurement {
    endpoint_chroma: [f32; 2],
    disagreement: f32,
}

fn profile_chroma_diagnostics(
    sources: &[ArrayColorSource<'_>],
    reference_index: usize,
    recorded_wb: Option<[f32; 3]>,
    observations: &[Observation],
) -> Vec<ProfileChromaDiagnostic> {
    let d65 = chroma_measurements(
        ProfileBlend::D65,
        sources,
        reference_index,
        recorded_wb,
        observations,
    );
    [
        ("A", ProfileBlend::A),
        ("F11", ProfileBlend::F11),
        ("D65", ProfileBlend::D65),
    ]
    .into_iter()
    .map(|(illuminant, blend)| {
        let measurements = if blend == ProfileBlend::D65 {
            d65.clone()
        } else {
            chroma_measurements(blend, sources, reference_index, recorded_wb, observations)
        };
        make_chroma_diagnostic(
            illuminant,
            blend,
            &measurements,
            &d65,
            sources,
            reference_index,
            recorded_wb,
            observations,
        )
    })
    .collect()
}

fn chroma_measurements(
    blend: ProfileBlend,
    sources: &[ArrayColorSource<'_>],
    reference_index: usize,
    recorded_wb: Option<[f32; 3]>,
    observations: &[Observation],
) -> Vec<Option<ChromaMeasurement>> {
    let reference_calibration = sources
        .get(reference_index)
        .and_then(|source| source.calibration);
    let colors = sources
        .iter()
        .map(|source| {
            module_color_for_blend(
                source.calibration,
                reference_calibration,
                recorded_wb,
                blend,
            )
        })
        .collect::<Vec<_>>();
    let Some(reference_color) = colors.get(reference_index).and_then(Option::as_ref) else {
        return vec![None; observations.len()];
    };
    observations
        .iter()
        .map(|observation| {
            let target_color = colors.get(observation.target)?.as_ref()?;
            let reference_xyz = reference_color.to_xyz(observation.reference_rgb);
            let target_xyz = target_color.to_xyz(observation.target_rgb);
            let measurement = ChromaMeasurement {
                endpoint_chroma: [
                    chroma_distance(reference_xyz, D50_WHITE),
                    chroma_distance(target_xyz, D50_WHITE),
                ],
                disagreement: chroma_distance(reference_xyz, target_xyz),
            };
            measurement
                .endpoint_chroma
                .iter()
                .chain(std::iter::once(&measurement.disagreement))
                .all(|value| value.is_finite())
                .then_some(measurement)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn make_chroma_diagnostic(
    illuminant: &'static str,
    blend: ProfileBlend,
    measurements: &[Option<ChromaMeasurement>],
    d65: &[Option<ChromaMeasurement>],
    sources: &[ArrayColorSource<'_>],
    reference_index: usize,
    recorded_wb: Option<[f32; 3]>,
    observations: &[Observation],
) -> ProfileChromaDiagnostic {
    let mut chroma = Vec::new();
    let mut disagreements = Vec::new();
    let mut ratios = Vec::new();
    let mut chromatic_ratios = Vec::new();
    let mut scatter = Vec::new();
    for (candidate, baseline) in measurements.iter().zip(d65) {
        let (Some(candidate), Some(baseline)) = (candidate, baseline) else {
            continue;
        };
        disagreements.push(candidate.disagreement);
        for endpoint in 0..2 {
            let candidate_chroma = candidate.endpoint_chroma[endpoint];
            let d65_chroma = baseline.endpoint_chroma[endpoint];
            chroma.push(candidate_chroma);
            ratios.push(candidate_chroma / (d65_chroma + CHROMA_RATIO_EPSILON));
            if d65_chroma >= CHROMATIC_RATIO_FLOOR {
                chromatic_ratios.push(candidate_chroma / d65_chroma);
            }
            scatter.push([d65_chroma, candidate_chroma]);
        }
    }
    let mean_chroma = mean(&chroma);
    let mean_disagreement = mean(&disagreements);
    ProfileChromaDiagnostic {
        illuminant,
        chroma: distribution(&mut chroma),
        relative_to_d65: relative_distribution(&mut ratios),
        relative_to_d65_chromatic: relative_distribution(&mut chromatic_ratios),
        mean_inter_module_disagreement: mean_disagreement,
        disagreement_normalized_by_mean_chroma: mean_disagreement
            / (mean_chroma + CHROMA_RATIO_EPSILON),
        selector_disagreement: score_blend(
            blend,
            sources,
            reference_index,
            recorded_wb,
            observations,
        ),
        scatter_vs_d65: subsample_scatter(scatter),
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn percentile(sorted: &[f32], quantile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = quantile * (sorted.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f32;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn distribution(values: &mut [f32]) -> ChromaDistributionReport {
    let average = mean(values);
    values.sort_by(f32::total_cmp);
    ChromaDistributionReport {
        count: values.len(),
        mean: average,
        p10: percentile(values, 0.10),
        p25: percentile(values, 0.25),
        p50: percentile(values, 0.50),
        p75: percentile(values, 0.75),
        p90: percentile(values, 0.90),
    }
}

fn relative_distribution(values: &mut [f32]) -> RelativeChromaReport {
    values.sort_by(f32::total_cmp);
    RelativeChromaReport {
        count: values.len(),
        p10: percentile(values, 0.10),
        p50: percentile(values, 0.50),
        p90: percentile(values, 0.90),
    }
}

fn subsample_scatter(values: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if values.len() <= MAX_SCATTER_POINTS {
        return values;
    }
    let stride = values.len().div_ceil(MAX_SCATTER_POINTS);
    values.into_iter().step_by(stride).collect()
}

fn simplex_grid() -> Vec<ProfileBlend> {
    let mut candidates = Vec::with_capacity(231);
    for a in 0..=SIMPLEX_STEPS {
        for f11 in 0..=SIMPLEX_STEPS - a {
            let d65 = SIMPLEX_STEPS - a - f11;
            candidates.push(ProfileBlend::new([
                a as f32 / SIMPLEX_STEPS as f32,
                f11 as f32 / SIMPLEX_STEPS as f32,
                d65 as f32 / SIMPLEX_STEPS as f32,
            ]));
        }
    }
    candidates
}

fn score_blend(
    blend: ProfileBlend,
    sources: &[ArrayColorSource<'_>],
    reference_index: usize,
    recorded_wb: Option<[f32; 3]>,
    observations: &[Observation],
) -> Option<f32> {
    let reference_calibration = sources.get(reference_index)?.calibration;
    let colors = sources
        .iter()
        .map(|source| {
            module_color_for_blend(
                source.calibration,
                reference_calibration,
                recorded_wb,
                blend,
            )
        })
        .collect::<Vec<_>>();
    let reference_color = colors.get(reference_index)?.as_ref()?;
    let mut per_module = vec![Vec::<f32>::new(); sources.len()];
    for observation in observations {
        let Some(target_color) = colors[observation.target].as_ref() else {
            continue;
        };
        let distance = chroma_distance(
            reference_color.to_xyz(observation.reference_rgb),
            target_color.to_xyz(observation.target_rgb),
        );
        if distance.is_finite() {
            // Bound remaining view-dependent/specular outliers without making
            // an already-large disagreement dominate the profile decision.
            per_module[observation.target].push(distance.min(0.12));
        }
    }
    let mut module_scores = per_module
        .iter_mut()
        .filter(|samples| samples.len() >= 24)
        .map(|samples| trimmed_mean(samples, 0.10))
        .collect::<Vec<_>>();
    if module_scores.len() < MINIMUM_TARGET_MODULES {
        return None;
    }
    module_scores.sort_by(f32::total_cmp);
    Some(trimmed_mean(&mut module_scores, 0.0))
}

fn trimmed_mean(values: &mut [f32], fraction: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    let trim = ((values.len() as f32 * fraction) as usize).min(values.len() / 3);
    let retained = &values[trim..values.len() - trim];
    retained.iter().sum::<f32>() / retained.len().max(1) as f32
}

fn collect_observations(
    sources: &[ArrayColorSource<'_>],
    reference_index: usize,
    width: usize,
    height: usize,
    depth: Option<&DenseDepthMap>,
) -> Vec<Observation> {
    let Some(reference) = sources.get(reference_index) else {
        return Vec::new();
    };
    if reference.mosaic.is_mono() || reference.calibration.is_none() {
        return Vec::new();
    }
    let step = (width.max(height) / 96).clamp(24, 96);
    let margin = step.max(12);
    let mut observations = Vec::new();
    for y in (margin..height.saturating_sub(margin)).step_by(step) {
        for x in (margin..width.saturating_sub(margin)).step_by(step) {
            let point = [x as f32, y as f32];
            if !depth_is_reliable(depth, point) {
                continue;
            }
            let Some(reference_rgb) = reliable_sample(reference, point, point, true) else {
                continue;
            };
            for (target, source) in sources.iter().enumerate() {
                if target == reference_index
                    || source.mosaic.is_mono()
                    || source.calibration.is_none()
                    || !source.alignment.report.accepted
                    || source.alignment.warp.confidence(point[0], point[1]) < 0.78
                    || !locally_confident(&source.alignment.warp, point, 8.0)
                {
                    continue;
                }
                let Some(mapped) = source.alignment.warp.map(point[0], point[1]) else {
                    continue;
                };
                let Some(target_rgb) = reliable_sample(source, point, mapped, false) else {
                    continue;
                };
                if !pair_structure_agrees(reference, source, point, mapped) {
                    continue;
                }
                let column = (x * COVERAGE_COLUMNS / width).min(COVERAGE_COLUMNS - 1);
                let row = (y * COVERAGE_ROWS / height).min(COVERAGE_ROWS - 1);
                observations.push(Observation {
                    target,
                    reference_rgb,
                    target_rgb,
                    cell: row * COVERAGE_COLUMNS + column,
                });
            }
        }
    }
    observations
}

fn depth_is_reliable(depth: Option<&DenseDepthMap>, point: [f32; 2]) -> bool {
    let Some(depth) = depth else {
        return true;
    };
    let Some(node) = depth.sample_nearest(point[0], point[1]) else {
        return false;
    };
    node.provenance != DepthProvenance::Unsupported && node.confidence >= 0.45
}

fn locally_confident(warp: &crate::align::Warp, point: [f32; 2], radius: f32) -> bool {
    [[-radius, 0.0], [radius, 0.0], [0.0, -radius], [0.0, radius]]
        .into_iter()
        .all(|offset| warp.confidence(point[0] + offset[0], point[1] + offset[1]) >= 0.70)
}

fn reliable_sample(
    source: &ArrayColorSource<'_>,
    reference_point: [f32; 2],
    mapped: [f32; 2],
    is_reference: bool,
) -> Option<[f32; 3]> {
    let (rgb, white) = source.mosaic.sample_rgb_with_white(mapped[0], mapped[1])?;
    let relative =
        std::array::from_fn::<_, 3, _>(|channel| rgb[channel] / white[channel].max(1.0e-5));
    let luminance = (relative[0] + 2.0 * relative[1] + relative[2]) * 0.25;
    if !(0.018..0.84).contains(&luminance)
        || relative
            .iter()
            .any(|value| *value >= 0.90 || !value.is_finite())
    {
        return None;
    }
    let confidence = highlight_confidence(source, mapped);
    if confidence < 220 {
        return None;
    }
    let raw_chroma = {
        let sum = relative.iter().sum::<f32>().max(1.0e-5);
        let minimum = relative.iter().copied().fold(f32::INFINITY, f32::min) / sum;
        let maximum = relative.iter().copied().fold(0.0, f32::max) / sum;
        maximum - minimum
    };
    // Extremely saturated colours and tiny speculars are precisely where
    // view angle and sensor metamerism can be least stable.
    if raw_chroma > 0.72 {
        return None;
    }
    if !is_reference
        && source
            .alignment
            .warp
            .map(reference_point[0], reference_point[1])
            .is_none()
    {
        return None;
    }
    Some(rgb)
}

fn highlight_confidence(source: &ArrayColorSource<'_>, point: [f32; 2]) -> u8 {
    if source.highlight.confidence.is_empty() {
        return 255;
    }
    let x = point[0]
        .round()
        .clamp(0.0, (source.mosaic.width - 1) as f32) as usize;
    let y = point[1]
        .round()
        .clamp(0.0, (source.mosaic.height - 1) as f32) as usize;
    source.highlight.confidence[y * source.mosaic.width + x]
}

fn pair_structure_agrees(
    reference: &ArrayColorSource<'_>,
    target: &ArrayColorSource<'_>,
    point: [f32; 2],
    mapped: [f32; 2],
) -> bool {
    let local_contrast = |mosaic: &Mosaic, centre: [f32; 2], radius: f32| -> Option<f32> {
        let centre_value = mosaic.sample_rgb(centre[0], centre[1])?[1].max(1.0e-4);
        let mut maximum = 0.0f32;
        for offset in [[-radius, 0.0], [radius, 0.0], [0.0, -radius], [0.0, radius]] {
            let value = mosaic.sample_rgb(centre[0] + offset[0], centre[1] + offset[1])?[1];
            maximum = maximum.max((value - centre_value).abs() / centre_value.max(value).max(0.01));
        }
        Some(maximum)
    };
    let reference_contrast = local_contrast(reference.mosaic, point, 4.0);
    let target_radius = target
        .alignment
        .warp
        .magnification(point[0], point[1])
        .unwrap_or(1.0)
        .clamp(0.5, 4.0)
        * 4.0;
    let target_contrast = local_contrast(target.mosaic, mapped, target_radius);
    matches!((reference_contrast, target_contrast), (Some(first), Some(second)) if first < 0.32 && second < 0.32 && (first - second).abs() < 0.18)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::ColorProfileProvenance;

    fn profile(illuminant: i32, diagonal: [f64; 3], rg: f64, bg: f64) -> ColorProfile {
        ColorProfile {
            illuminant,
            forward_matrix: [
                [diagonal[0], 0.0, 0.0],
                [0.0, diagonal[1], 0.0],
                [0.0, 0.0, diagonal[2]],
            ],
            validated_matrix: None,
            color_matrix: None,
            rg_ratio: rg,
            bg_ratio: bg,
            macbeth_data: Vec::new(),
            illuminant_spd: Vec::new(),
            spectral_data: None,
            provenance: ColorProfileProvenance::Module,
        }
    }

    #[test]
    fn coarse_simplex_has_231_candidates() {
        let candidates = simplex_grid();
        assert_eq!(candidates.len(), 231);
        assert!(candidates.iter().all(|candidate| {
            (candidate.weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6
                && candidate.weights.iter().all(|weight| *weight >= 0.0)
        }));
    }

    #[test]
    fn cct_prior_interpolates_adjacent_anchors() {
        let calibration = CameraCalibration {
            name: "B4".to_owned(),
            color: vec![
                profile(ILLUMINANT_A, [1.0; 3], 0.75, 0.45),
                profile(ILLUMINANT_F11, [1.0; 3], 0.58, 0.53),
                profile(ILLUMINANT_D65, [1.0; 3], 0.48, 0.67),
            ],
            ..CameraCalibration::default()
        };
        let rg = (0.75_f64 * 0.58).sqrt() as f32;
        let bg = (0.45_f64 * 0.53).sqrt() as f32;
        let (blend, _, _) = cct_prior(Some(&calibration), Some([1.0 / rg, 1.0, 1.0 / bg]));
        assert!((blend.weights[0] - 0.5).abs() < 1.0e-4);
        assert!((blend.weights[1] - 0.5).abs() < 1.0e-4);
        assert_eq!(blend.weights[2], 0.0);
    }

    #[test]
    fn blend_uses_all_three_factory_profiles() {
        let calibration = CameraCalibration {
            name: "B4".to_owned(),
            color: vec![
                profile(ILLUMINANT_A, [3.0; 3], 0.75, 0.45),
                profile(ILLUMINANT_F11, [2.0; 3], 0.58, 0.53),
                profile(ILLUMINANT_D65, [1.0; 3], 0.48, 0.67),
            ],
            ..CameraCalibration::default()
        };
        let blend = ProfileBlend::new([0.2, 0.3, 0.5]);
        let profile = blended_profile(Some(&calibration), blend).unwrap();
        assert!((profile.matrix[0][0] - 1.7).abs() < 1.0e-6);
        assert!(profile.rg_ratio > 0.48 && profile.rg_ratio < 0.75);
    }

    #[test]
    fn material_override_requires_clear_array_evidence() {
        assert!(array_override_supported(0.40, true, true));
        assert!(!array_override_supported(0.40, true, false));
        assert!(!array_override_supported(0.04, true, true));
        // A small adjustment near the prior still needs a resolved score
        // surface, but not a separate large-improvement test.
        assert!(array_override_supported(0.20, false, false));
    }

    #[test]
    fn production_default_remains_original_d65_path() {
        assert_eq!(ColorProfileMode::default(), ColorProfileMode::ForceD65);
    }

    #[test]
    fn chroma_distribution_uses_interpolated_percentiles() {
        let mut values = (1..=10).map(|value| value as f32).collect::<Vec<_>>();
        let report = distribution(&mut values);
        assert_eq!(report.count, 10);
        assert!((report.mean - 5.5).abs() < 1.0e-6);
        assert!((report.p10 - 1.9).abs() < 1.0e-6);
        assert!((report.p25 - 3.25).abs() < 1.0e-6);
        assert!((report.p50 - 5.5).abs() < 1.0e-6);
        assert!((report.p75 - 7.75).abs() < 1.0e-6);
        assert!((report.p90 - 9.1).abs() < 1.0e-6);
    }

    #[test]
    fn relative_chroma_exposes_uniform_contraction() {
        let d65 = [0.02_f32, 0.04, 0.08, 0.16];
        let mut ratios = d65
            .into_iter()
            .map(|baseline| (baseline * 0.5) / baseline)
            .collect::<Vec<_>>();
        let report = relative_distribution(&mut ratios);
        assert_eq!(report.count, 4);
        assert!((report.p10 - 0.5).abs() < 1.0e-6);
        assert!((report.p50 - 0.5).abs() < 1.0e-6);
        assert!((report.p90 - 0.5).abs() < 1.0e-6);
    }
}
