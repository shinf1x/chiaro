//! Capture-specific refinement of the physical multi-camera rig.
//!
//! The preferred matcher is physically guided and semi-dense: structured
//! reference patches are swept over calibrated inverse depth, projected through
//! every target camera, and admitted only when a common depth is supported by
//! several physical views. When the original factory-centred search finds no
//! supported match, the provisional image alignment supplies a fallback
//! proposal only for the perpendicular position of the residual band; physical
//! depth still defines motion along the epipolar locus, and no homography
//! supplies an accepted observation. Legacy alignment correspondences remain a
//! deterministic fallback for metadata-only/tests or captures where physical
//! matching cannot establish a sufficient population. The resulting
//! multi-view tracks are triangulated and used to fit small orientation,
//! virtual-centre translation, sensor-raster
//! offset, and movable-mirror state corrections. The reference camera is the
//! fixed gauge. Debug runs reserve an independently held-out track subset for
//! validation; production runs use every usable track for the final estimate.

use std::{collections::BTreeMap, thread};

use serde::Serialize;

use crate::{
    align::ModuleAlignment,
    calibration::{CameraCalibration, IntrinsicsMode, ModuleState},
    geometry::{CameraRefinement, ResolvedCamera},
    image::Plane,
    math::{self, Mat3, Vec2, Vec3, add, cross, dot, mul_vec, norm, scale, sub},
};

#[derive(Clone, Debug)]
pub struct RigRefinementOptions {
    /// Run capture-specific physical refinement before the residual image warp.
    pub enabled: bool,
    /// Worker threads used by independent physical reference candidates
    /// (`0` = all available cores).
    pub threads: usize,
    /// Reserve spatially separated tracks for diagnostic validation instead
    /// of using them to estimate the capture rig. The production pipeline
    /// enables this only when a debug report was requested.
    pub held_out_validation: bool,
    /// Minimum geometrically valid tracks required for an attempted fit.
    pub min_tracks: usize,
    /// Minimum independently held-out tracks required for acceptance.
    pub min_validation_tracks: usize,
    /// Minimum fit-set observations required before a camera receives free
    /// physical parameters. Sparse cameras remain on their factory model.
    pub min_camera_observations: usize,
    /// Deterministic fraction of tracks withheld from optimization.
    pub validation_fraction: f64,
    /// Full-resolution reference-raster block size used for the deterministic
    /// spatial fit/validation split. All tracks in one block stay together.
    pub validation_block_size_px: usize,
    /// Bounded coordinate-Newton sweeps.
    pub max_iterations: usize,
    /// Alternating fit-only rig/track-membership passes after the initial
    /// solve. These passes retriangulate stored observations without rereading
    /// image data.
    pub max_membership_iterations: usize,
    /// Strict world-frame bearing correction bound per axis.
    pub max_orientation_degrees: f64,
    /// Strict additive movable-mirror angle correction bound.
    pub max_mirror_degrees: f64,
    /// Strict optical-centre correction bound per world axis, in the factory
    /// calibration's distance units (millimetres on L16).
    pub max_center_offset: f64,
    /// Strict calibration-raster origin correction bound per sensor axis.
    pub max_sensor_offset_px: f64,
    /// Gaussian factory prior scale for each orientation component.
    pub orientation_prior_sigma_degrees: f64,
    /// Gaussian factory prior scale for the movable-mirror angle.
    pub mirror_prior_sigma_degrees: f64,
    /// Gaussian factory prior scale for each optical-centre axis.
    pub center_prior_sigma: f64,
    /// Gaussian factory prior scale for each sensor-raster axis.
    pub sensor_offset_prior_sigma_px: f64,
    /// Weight of the normalized quadratic factory prior.
    pub factory_prior_weight: f64,
    /// Huber transition in normalized reprojection units.
    pub huber_delta: f64,
    /// Maximum accepted triangulation normal-matrix condition number.
    pub max_triangulation_condition: f64,
    /// Minimum angle between any two track rays.
    pub min_ray_angle_degrees: f64,
    /// Reject numerically explosive or mismatched initial ray-line tracks
    /// before robust fitting. This is deliberately much wider than the final
    /// image-space inlier threshold.
    pub max_initial_track_rms: f64,
    /// Minimum relative validation RMS reduction required for acceptance.
    pub min_validation_improvement: f64,
    /// Minimum fraction of fit and held-out tracks that triangulate in front
    /// of every participating camera under the candidate rig.
    pub min_positive_depth_fraction: f64,
    /// Reference threshold for the diagnostic comparison of the later
    /// residual image-space correction. Falling short produces a warning but
    /// does not replace an accepted physical rig with factory geometry.
    pub min_image_space_correction_improvement: f64,
    /// Absolute held-out quality reference lines shown in diagnostics.
    /// Reference-equivalent pixels account for local cross-focal
    /// magnification; angular error accounts for camera focal length. These
    /// do not veto a candidate that improves the independent split.
    pub max_validation_p95_reference_px: f64,
    pub max_validation_p95_angular_degrees: f64,

    /// Build the capture-rig observations directly from the calibrated
    /// generalized-camera geometry instead of inheriting the legacy
    /// homography matcher. The old correspondence population remains only as
    /// a deterministic fallback when image evidence is unavailable.
    pub physical_matching: bool,
    /// Semi-dense sampling interval in full-resolution sensor pixels.
    pub physical_match_stride_px: usize,
    /// Maximum number of spatially distributed reference candidates.
    pub physical_match_max_candidates: usize,
    /// Radius of the ZNCC support in pixels of the active luminance-pyramid
    /// level.
    pub physical_match_patch_radius: usize,
    /// Number of coarse inverse-depth hypotheses searched for each candidate.
    /// The matcher refines promising intervals hierarchically when adjacent
    /// coarse hypotheses move farther than `physical_match_max_projected_step_px`.
    pub physical_match_planes: usize,
    /// Maximum target-sensor motion between adjacent final inverse-depth
    /// hypotheses. The image pyramid and depth hierarchy are chosen per
    /// reference candidate from the actual calibrated projections.
    pub physical_match_max_projected_step_px: f64,
    /// Number of separated depth modes retained between pyramid levels.
    pub physical_match_depth_beam_width: usize,
    /// Safety cap on per-candidate binary inverse-depth refinements.
    pub physical_match_max_depth_refinements: usize,
    /// Near/far limits of the physical match search, in calibration units.
    pub physical_match_near_depth: f64,
    pub physical_match_far_depth: f64,
    /// Coarse image-space tolerance around the physical epipolar/depth locus.
    /// This exists only to bootstrap the small capture-specific rig error; the
    /// resulting observations are subsequently required to triangulate and
    /// reproject coherently under one physical rig.
    pub physical_match_residual_radius_px: f64,
    /// Weakest per-camera ZNCC accepted as evidence for a physical track.
    pub physical_match_min_score: f32,
    /// Minimum reference-patch log-luminance standard deviation.
    pub physical_match_min_structure: f32,
    /// Minimum relative separation from the best non-adjacent depth
    /// hypothesis.
    pub physical_match_min_margin: f32,
    /// Minimum number of physical cameras in a semi-dense track, including
    /// the reference camera.
    pub physical_match_min_views: usize,
    /// Maximum per-observation reprojection error after triangulation and
    /// iterative outlier removal, expressed in reference-equivalent pixels.
    pub physical_match_max_reprojection_reference_px: f64,
}

impl Default for RigRefinementOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            threads: 0,
            held_out_validation: true,
            min_tracks: 80,
            min_validation_tracks: 20,
            min_camera_observations: 150,
            validation_fraction: 0.20,
            validation_block_size_px: 256,
            max_iterations: 6,
            max_membership_iterations: 3,
            max_orientation_degrees: 0.5,
            max_mirror_degrees: 0.35,
            max_center_offset: 5.0,
            max_sensor_offset_px: 64.0,
            orientation_prior_sigma_degrees: 0.20,
            mirror_prior_sigma_degrees: 0.08,
            center_prior_sigma: 1.0,
            sensor_offset_prior_sigma_px: 12.0,
            factory_prior_weight: 0.02,
            huber_delta: 2.5,
            max_triangulation_condition: 1.0e9,
            min_ray_angle_degrees: 0.003,
            max_initial_track_rms: 100.0,
            min_validation_improvement: 0.005,
            min_positive_depth_fraction: 0.80,
            min_image_space_correction_improvement: 0.05,
            max_validation_p95_reference_px: 6.0,
            max_validation_p95_angular_degrees: 0.10,
            physical_matching: true,
            physical_match_stride_px: 32,
            physical_match_max_candidates: 6000,
            physical_match_patch_radius: 3,
            physical_match_planes: 48,
            physical_match_max_projected_step_px: 8.0,
            physical_match_depth_beam_width: 6,
            physical_match_max_depth_refinements: 6,
            physical_match_near_depth: 500.0,
            physical_match_far_depth: 10_000_000.0,
            physical_match_residual_radius_px: 32.0,
            physical_match_min_score: 0.50,
            physical_match_min_structure: 0.008,
            physical_match_min_margin: 0.03,
            physical_match_min_views: 3,
            physical_match_max_reprojection_reference_px: 12.0,
        }
    }
}

pub struct RigCameraInput<'a> {
    pub name: &'a str,
    pub calibration: Option<&'a CameraCalibration>,
    pub state: Option<&'a ModuleState>,
    /// Whether this camera's image observations may influence capture-specific
    /// calibration. Held-out CFA cameras keep their physical model available
    /// for projection/evaluation but must not steer either matcher path.
    pub match_evidence_enabled: bool,
    /// Half-resolution log-luminance used only for physically guided sparse
    /// rig matching. `None` keeps synthetic/unit tests and metadata-only paths
    /// on the legacy correspondence fallback.
    pub luminance: Option<&'a Plane>,
}

#[derive(Clone, Debug)]
pub struct RigRefinementOutcome {
    /// Zero corrections when the validation gate rejects the candidate.
    pub refinements: Vec<CameraRefinement>,
    pub report: RigRefinementReport,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigRefinementReport {
    pub enabled: bool,
    pub accepted: bool,
    /// True only when the candidate reached fit and held-out geometric
    /// evaluation. Numeric validation fields are otherwise default storage,
    /// not measured zero error.
    pub validation_evaluated: bool,
    pub reference_camera: String,
    pub pairwise_matches: usize,
    /// Number of semi-dense reference locations considered by the physical
    /// matcher before texture/visibility/depth-consensus rejection.
    pub physical_match_candidates: usize,
    /// True when the rig solve consumed the physically guided semi-dense
    /// population rather than the legacy image-space correspondences.
    pub physical_match_used: bool,
    /// Number of tracks produced directly by rig+depth matching.
    pub physical_match_tracks: usize,
    /// Number of target-camera observations retained in those tracks.
    pub physical_match_observations: usize,
    /// Total depth hypotheses evaluated over every pyramid level and reference
    /// candidate. Dividing by `physical_match_candidates` gives the actual
    /// average search cost rather than the old nominal plane count.
    pub physical_match_depth_hypotheses: usize,
    /// Deepest per-candidate inverse-depth hierarchy used in this capture.
    pub physical_match_max_depth_refinement_levels: usize,
    /// Candidate count by refinement depth; index zero is the unchanged
    /// coarse grid, index N is N binary inverse-depth subdivisions.
    pub physical_match_depth_refinement_histogram: Vec<usize>,
    /// Configured maximum target-pixel displacement between adjacent final
    /// hypotheses (unless the explicit refinement safety cap is reached).
    pub physical_match_max_projected_step_px: f64,
    /// Largest final adjacent-hypothesis displacement actually measured over
    /// the sampled reference candidates and eligible target cameras.
    pub physical_match_observed_max_projected_step_px: f64,
    /// Target-sensor radius searched first around the factory physical locus
    /// and, if that produces no supported match, around a stage-2 perpendicular
    /// proposal. Motion along either locus remains determined by physical
    /// depth.
    pub physical_match_residual_radius_px: f64,
    /// Reference-equivalent reprojection threshold applied after removing the
    /// stage-2 perpendicular proposal from bootstrap observations. This tests
    /// the physical depth/localization residual without rejecting the
    /// capture-specific displacement that the optimizer is intended to fit.
    pub physical_match_pre_solve_reprojection_px: f64,
    pub physical_match_per_camera: Vec<RigCameraMatchSupportReport>,
    /// Reference candidates for which no depth hypothesis retained the
    /// configured minimum number of target-camera matches.
    pub physical_match_rejected_no_supported_depth: usize,
    /// Reference candidates whose strongest separated depth modes were too
    /// similar to identify one reliable depth.
    pub physical_match_rejected_ambiguous_depth: usize,
    /// Reference candidates that had a coarse depth consensus but lost too
    /// many observations during native-resolution localization.
    pub physical_match_rejected_insufficient_views: usize,
    pub tracks: usize,
    pub tracks_three_plus: usize,
    pub fit_tracks: usize,
    pub validation_tracks: usize,
    pub rejected_degenerate_tracks: usize,
    pub rejected_outlier_tracks: usize,
    pub rejected_nonpositive_observations: usize,
    /// Target observations removed by iterative triangulate/reproject pruning.
    pub rejected_inconsistent_observations: usize,
    /// Physical tracks that could not retain the configured minimum number of
    /// mutually consistent views after observation pruning.
    pub rejected_inconsistent_tracks: usize,
    pub optimizer_iterations: usize,
    pub membership_iterations: usize,
    pub fit_membership_rejected_observations: usize,
    pub fit_membership_rejected_tracks: usize,
    pub training_objective_before: f64,
    pub training_objective_after: f64,
    pub reprojection_rms_before: f64,
    pub reprojection_rms_after: f64,
    pub held_out_rms_before: f64,
    pub held_out_rms_after: f64,
    pub fit_positive_depth_fraction_before: f64,
    pub fit_positive_depth_fraction_after: f64,
    pub held_out_positive_depth_fraction_before: f64,
    pub held_out_positive_depth_fraction_after: f64,
    /// Positive means the independently held-out reprojection RMS decreased.
    pub held_out_relative_improvement: f64,
    pub fit_residuals_before: RigResidualDistributionReport,
    pub fit_residuals_after: RigResidualDistributionReport,
    pub held_out_residuals_before: RigResidualDistributionReport,
    pub held_out_residuals_after: RigResidualDistributionReport,
    pub median_triangulation_condition: f64,
    pub p90_triangulation_condition: f64,
    pub median_max_ray_angle_degrees: f64,
    pub per_camera: Vec<RigCameraResidualReport>,
    pub residual_field: Vec<RigResidualFieldReport>,
    /// Every paired held-out reprojection observation. These samples are
    /// exported for spatial diagnostics; neither population participates in
    /// physical-parameter optimization.
    pub held_out_observations: Vec<RigHeldOutObservationReport>,
    pub corrections: Vec<RigCameraCorrectionReport>,
    /// Finite-difference identifiability of each candidate physical parameter.
    /// Parameters with negligible sensitivity or near-collinearity are kept on
    /// the factory model instead of being released merely because the camera
    /// has many observations.
    pub parameter_observability: Vec<RigParameterObservabilityReport>,
    /// Filled by the pipeline after the accepted physical model is used as the
    /// seed for the ordinary residual image-space alignment.
    pub image_space_corrections: Vec<RigImageCorrectionReport>,
    pub image_space_evaluated_cameras: usize,
    pub image_space_median_correction_before_px: f64,
    pub image_space_median_correction_after_px: f64,
    /// Positive means the physical model reduced the amount of later
    /// image-space correction required.
    pub image_space_relative_improvement: f64,
    /// Diagnostic result of comparing the later patch-supported residual
    /// correction from factory and refined physical seeds. This is not an
    /// acceptance gate: an accepted physical rig remains selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_space_validation_passed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_space_warning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigCameraResidualReport {
    pub camera: String,
    pub fit_samples: usize,
    pub fit_rms_before: f64,
    pub fit_rms_after: f64,
    pub validation_samples: usize,
    pub validation_rms_before: f64,
    pub validation_rms_after: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigCameraMatchSupportReport {
    pub camera: String,
    pub observations: usize,
    pub track_fraction: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigResidualPercentiles {
    pub median: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigResidualDistributionReport {
    pub samples: usize,
    pub sensor_pixels: RigResidualPercentiles,
    pub reference_equivalent_pixels: RigResidualPercentiles,
    pub angular_degrees: RigResidualPercentiles,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigResidualFieldReport {
    pub camera: String,
    pub cell: [usize; 2],
    pub samples: usize,
    pub mean_pixel: [f64; 2],
    pub mean_before: [f64; 2],
    pub mean_after: [f64; 2],
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigHeldOutObservationReport {
    pub camera: String,
    pub sensor_size: [usize; 2],
    pub pixel: [f64; 2],
    pub factory_residual: [f64; 2],
    pub candidate_residual: [f64; 2],
    pub factory_sensor_pixels: f64,
    pub candidate_sensor_pixels: f64,
    pub factory_reference_pixels: f64,
    pub candidate_reference_pixels: f64,
    pub factory_angular_degrees: f64,
    pub candidate_angular_degrees: f64,
    /// True when this observation is at or above the global factory p95.
    pub factory_p95_tail: bool,
    /// True when this observation is at or above the global candidate p95.
    pub candidate_p95_tail: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigParameterObservabilityReport {
    pub camera: String,
    pub parameter: String,
    /// RMS change in normalized retriangulated reprojection residual for a
    /// one-prior-sigma perturbation of this parameter.
    pub sensitivity_rms: f64,
    /// Largest absolute Jacobian correlation with another candidate parameter.
    pub max_correlation: f64,
    pub optimized: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection_reason: Option<String>,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct RigCameraCorrectionReport {
    pub camera: String,
    /// False for the fixed gauge and cameras without enough fit observations.
    pub optimized: bool,
    pub orientation_offset_degrees: [f64; 3],
    pub mirror_angle_offset_degrees: f64,
    pub center_offset_world: [f64; 3],
    pub sensor_offset_px: [f64; 2],
    pub reached_bound: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigImageCorrectionReport {
    pub camera: String,
    pub factory_seed_correction_px: [f32; 2],
    pub refined_seed_correction_px: [f32; 2],
}

/// Populate the downstream residual-alignment comparison. This deliberately
/// does not mutate physical-rig acceptance: the residual warp is a downstream
/// image-space correction, not ground truth for the physical camera model.
pub fn evaluate_image_space_alignment(
    report: &mut RigRefinementReport,
    factory: &[ModuleAlignment],
    refined: &[ModuleAlignment],
    minimum_improvement: f64,
) -> bool {
    report.image_space_corrections = factory
        .iter()
        .zip(refined)
        .map(|(factory, refined)| RigImageCorrectionReport {
            camera: refined.name.clone(),
            factory_seed_correction_px: factory.report.correction_median_px,
            refined_seed_correction_px: refined.report.correction_median_px,
        })
        .collect();

    let active = report
        .corrections
        .iter()
        .filter(|correction| correction.optimized)
        .map(|correction| correction.camera.as_str())
        .collect::<Vec<_>>();
    let mut before = Vec::new();
    let mut after = Vec::new();
    let mut lost_measurement = None;
    for (factory, refined) in factory.iter().zip(refined) {
        if !active.contains(&factory.name.as_str()) {
            continue;
        }
        // This diagnostic measures how much *measured residual correction*
        // the downstream matcher needs. The legacy homography acceptance bit
        // is deliberately not a veto, but a zero/identity correction with no
        // patch support is not evidence of improvement either.
        let factory_measured =
            factory.report.inliers >= 4 && factory.report.residual_median_px.is_finite();
        let refined_measured =
            refined.report.inliers >= 4 && refined.report.residual_median_px.is_finite();
        if factory_measured && !refined_measured {
            lost_measurement = Some(factory.name.clone());
            continue;
        }
        if !factory_measured || !refined_measured {
            continue;
        }
        let factory_correction = correction_magnitude(factory.report.correction_median_px);
        let refined_correction = correction_magnitude(refined.report.correction_median_px);
        if factory_correction.is_finite() && refined_correction.is_finite() {
            before.push(factory_correction);
            after.push(refined_correction);
        }
    }
    before.sort_by(f64::total_cmp);
    after.sort_by(f64::total_cmp);
    report.image_space_evaluated_cameras = before.len();
    report.image_space_median_correction_before_px = percentile(&before, 0.5);
    report.image_space_median_correction_after_px = percentile(&after, 0.5);
    report.image_space_relative_improvement = (report.image_space_median_correction_before_px
        - report.image_space_median_correction_after_px)
        / report.image_space_median_correction_before_px.max(1.0e-12);

    let passed = lost_measurement.is_none()
        && !before.is_empty()
        && report.image_space_relative_improvement >= minimum_improvement;
    report.image_space_validation_passed = Some(passed);
    report.image_space_warning = (!passed).then(|| {
        if let Some(camera) = lost_measurement {
            format!(
                "residual correction for {camera} became unmeasurable from the refined physical seed"
            )
        } else if before.is_empty() {
            "no finite measured residual corrections available for downstream validation".to_owned()
        } else {
            format!(
                "later image-space correction improved by only {:+.2}% (need at least {:+.2}%)",
                report.image_space_relative_improvement * 100.0,
                minimum_improvement * 100.0
            )
        }
    });
    passed
}

fn correction_magnitude(correction: [f32; 2]) -> f64 {
    (f64::from(correction[0]).powi(2) + f64::from(correction[1]).powi(2)).sqrt()
}

#[derive(Clone, Debug)]
struct TrackObservation {
    camera: usize,
    pixel: Vec2,
    /// Proposal removed only while validating the pre-optimization physical
    /// bootstrap. Optimization and held-out evaluation always use `pixel`.
    bootstrap_residual_proposal: Vec2,
    /// Unit-determinant localization covariance in target sensor coordinates.
    /// The scalar photometric/structure uncertainty is applied separately.
    localization_covariance: [[f64; 2]; 2],
    fixed_gauge: bool,
    confidence: f64,
    local_scale: f64,
    structure: f64,
    depth_reliability: Option<f64>,
}

#[derive(Clone, Debug)]
struct Track {
    key: [i32; 2],
    observations: Vec<TrackObservation>,
    condition: f64,
    max_ray_angle_degrees: f64,
}

#[derive(Clone, Copy, Debug)]
enum ParameterKind {
    Orientation(usize),
    Mirror,
    Center(usize),
    Sensor(usize),
}

#[derive(Clone, Copy, Debug)]
struct ParameterSpec {
    camera: usize,
    kind: ParameterKind,
    bound: f64,
    prior_sigma: f64,
    difference_step: f64,
    maximum_update: f64,
}

#[derive(Clone, Debug)]
struct ResidualSample {
    camera: usize,
    pixel: Vec2,
    residual: Vec2,
    reference_equivalent_pixels: f64,
    angular_degrees: f64,
}

#[derive(Clone, Debug, Default)]
struct Evaluation {
    sum_squared: f64,
    samples: usize,
    tracks: usize,
    positive_depth_tracks: usize,
    residuals: Vec<ResidualSample>,
}

impl Evaluation {
    fn rms(&self) -> f64 {
        if self.samples == 0 {
            f64::NAN
        } else {
            (self.sum_squared / self.samples as f64).sqrt()
        }
    }

    fn positive_depth_fraction(&self) -> f64 {
        if self.tracks == 0 {
            f64::NAN
        } else {
            self.positive_depth_tracks as f64 / self.tracks as f64
        }
    }
}

/// Fit a bounded capture-specific physical model. Diagnostic runs reserve an
/// independent deterministic track split and require relative improvement;
/// production runs consume all usable tracks.
pub fn refine_capture_rig(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    provisional_alignments: &[ModuleAlignment],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> RigRefinementOutcome {
    let mut report = RigRefinementReport {
        enabled: options.enabled,
        physical_match_max_projected_step_px: options.physical_match_max_projected_step_px,
        physical_match_residual_radius_px: options.physical_match_residual_radius_px,
        physical_match_pre_solve_reprojection_px: options
            .physical_match_max_reprojection_reference_px,
        reference_camera: cameras
            .get(reference_index)
            .map_or_else(String::new, |camera| camera.name.to_owned()),
        ..Default::default()
    };
    let zero = vec![CameraRefinement::default(); cameras.len()];
    if !options.enabled {
        report.fallback_reason = Some("disabled".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }
    if cameras.len() != provisional_alignments.len() || reference_index >= cameras.len() {
        report.fallback_reason = Some("camera/alignment population mismatch".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }
    let Some(factory_cameras) = resolve_cameras(cameras, &zero, intrinsics_mode) else {
        report.fallback_reason =
            Some("one or more physical camera models are unavailable".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    };
    let physical = build_physical_tracks(
        cameras,
        reference_index,
        &factory_cameras,
        provisional_alignments,
        options,
    );
    let physical_tracks = physical.tracks;
    let physical_observations = physical.observations;
    report.physical_match_candidates = physical.candidates;
    report.physical_match_tracks = physical_tracks.len();
    report.physical_match_observations = physical_observations;
    report.physical_match_depth_hypotheses = physical.depth_hypotheses;
    report.physical_match_max_depth_refinement_levels = physical.max_depth_refinement_levels;
    report.physical_match_depth_refinement_histogram = physical.depth_refinement_histogram;
    report.physical_match_observed_max_projected_step_px = physical.observed_max_projected_step_px;
    report.physical_match_per_camera =
        physical_match_support_reports(cameras, &physical_tracks, reference_index);
    report.physical_match_rejected_no_supported_depth = physical.rejected_no_supported_depth;
    report.physical_match_rejected_ambiguous_depth = physical.rejected_ambiguous_depth;
    report.physical_match_rejected_insufficient_views = physical.rejected_insufficient_views;
    report.rejected_inconsistent_observations = physical.rejected_observations;
    report.rejected_inconsistent_tracks = physical.rejected_tracks;
    let validation_modulus = (1.0 / options.validation_fraction.clamp(0.05, 0.5)).round() as u64;
    let physical_validation_tracks = physical_tracks
        .iter()
        .filter(|track| {
            options.held_out_validation && is_validation_track(track, validation_modulus, options)
        })
        .count();
    let physical_fit_tracks = physical_tracks.len() - physical_validation_tracks;
    let physical_population_is_sufficient = physical_fit_tracks >= options.min_tracks
        && (!options.held_out_validation
            || physical_validation_tracks >= options.min_validation_tracks);
    let (raw_tracks, pairwise_matches) = if physical_population_is_sufficient {
        report.physical_match_used = true;
        (physical_tracks, physical_observations)
    } else {
        let legacy = build_tracks(
            cameras,
            reference_index,
            provisional_alignments,
            &factory_cameras,
        );
        report.physical_match_used = false;
        legacy
    };
    report.pairwise_matches = pairwise_matches;
    report.tracks = raw_tracks.len();
    report.tracks_three_plus = raw_tracks
        .iter()
        .filter(|track| track.observations.len() >= 3)
        .count();

    let mut tracks = Vec::new();
    let mut rejected = 0;
    let mut rejected_outliers = 0;
    let mut rejected_nonpositive_observations = 0;
    for mut track in raw_tracks {
        let Some(reference) = track
            .observations
            .iter()
            .find(|observation| observation.camera == reference_index)
            .cloned()
        else {
            rejected += 1;
            continue;
        };
        let reference_ray = factory_cameras[reference_index].pixel_to_ray(reference.pixel);
        let before = track.observations.len();
        track.observations.retain(|observation| {
            observation.camera == reference_index
                || pair_has_positive_depth(
                    reference_ray,
                    factory_cameras[observation.camera].pixel_to_ray(observation.pixel),
                )
        });
        rejected_nonpositive_observations += before - track.observations.len();
        let Some(triangulated) = triangulate(&track.observations, &factory_cameras, options) else {
            rejected += 1;
            continue;
        };
        if !triangulation_has_positive_depth(
            &track.observations,
            &factory_cameras,
            triangulated.point,
        ) {
            rejected += 1;
            continue;
        }
        let initial_rms = track_rms(&track.observations, &factory_cameras, triangulated.point);
        if !initial_rms.is_finite() || initial_rms > options.max_initial_track_rms {
            rejected_outliers += 1;
            continue;
        }
        track.condition = triangulated.condition;
        track.max_ray_angle_degrees = triangulated.max_ray_angle_degrees;
        tracks.push(track);
    }
    report.rejected_degenerate_tracks = rejected;
    report.rejected_outlier_tracks = rejected_outliers;
    report.rejected_nonpositive_observations = rejected_nonpositive_observations;
    if tracks.len() < options.min_tracks {
        report.fallback_reason = Some(format!(
            "only {} geometrically valid tracks (need {})",
            tracks.len(),
            options.min_tracks
        ));
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }

    let (preliminary_validation, preliminary_fit): (Vec<_>, Vec<_>) = if options.held_out_validation
    {
        tracks
            .iter()
            .partition(|track| is_validation_track(track, validation_modulus, options))
    } else {
        (Vec::new(), tracks.iter().collect())
    };
    if preliminary_fit.len() < options.min_tracks
        || (options.held_out_validation
            && preliminary_validation.len() < options.min_validation_tracks)
    {
        report.fallback_reason = Some(format!(
            "insufficient fit/validation split ({}/{})",
            preliminary_fit.len(),
            preliminary_validation.len()
        ));
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }

    let candidate_specs = parameter_specs(cameras, reference_index, &preliminary_fit, options);
    let (specs, parameter_observability) = filter_observable_parameter_specs(
        &candidate_specs,
        cameras,
        &preliminary_fit,
        intrinsics_mode,
        options,
    );
    report.parameter_observability = parameter_observability;
    if specs.is_empty() {
        report.fallback_reason = Some("no observable non-reference physical parameters".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }
    // A sparse camera that cannot support its own physical parameters must
    // not steer the nuisance 3-D points for well-observed cameras. Keep it on
    // the factory model and outside this solve; ordinary residual alignment
    // still processes it downstream.
    let observable_camera =
        |camera: usize| camera == reference_index || specs.iter().any(|spec| spec.camera == camera);
    let prepare = |source: &[&Track]| {
        source
            .iter()
            .filter_map(|track| {
                let mut track = (*track).clone();
                track
                    .observations
                    .retain(|observation| observable_camera(observation.camera));
                let triangulated = triangulate(&track.observations, &factory_cameras, options)?;
                if !triangulation_has_positive_depth(
                    &track.observations,
                    &factory_cameras,
                    triangulated.point,
                ) {
                    return None;
                }
                let initial_rms =
                    track_rms(&track.observations, &factory_cameras, triangulated.point);
                if !initial_rms.is_finite() || initial_rms > options.max_initial_track_rms {
                    return None;
                }
                track.condition = triangulated.condition;
                track.max_ray_angle_degrees = triangulated.max_ray_angle_degrees;
                Some(track)
            })
            .collect::<Vec<_>>()
    };
    let mut fit_tracks = prepare(&preliminary_fit);
    let validation_tracks = prepare(&preliminary_validation);
    let removed_after_observability = preliminary_fit.len() + preliminary_validation.len()
        - fit_tracks.len()
        - validation_tracks.len();
    report.rejected_degenerate_tracks += removed_after_observability;
    report.fit_tracks = fit_tracks.len();
    report.validation_tracks = validation_tracks.len();
    if fit_tracks.len() < options.min_tracks
        || (options.held_out_validation && validation_tracks.len() < options.min_validation_tracks)
    {
        report.fallback_reason = Some(format!(
            "insufficient observable fit/validation tracks ({}/{})",
            fit_tracks.len(),
            validation_tracks.len()
        ));
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }
    let validation = validation_tracks.iter().collect::<Vec<_>>();
    let factory_parameters = vec![0.0; specs.len()];
    let initial_fit = fit_tracks.iter().collect::<Vec<_>>();
    let initial_before_objective = objective(
        &factory_parameters,
        &specs,
        cameras,
        &initial_fit,
        intrinsics_mode,
        options,
    );
    // The factory rig can miss by more than the finite-depth parallax itself.
    // Obtain a fit-only epipolar initialization before alternating
    // triangulation and physical parameter updates. Validation tracks never
    // enter either optimization objective.
    let (angular_candidate, _, initialization_iterations) = coordinate_optimize(
        factory_parameters.clone(),
        &specs,
        options.max_iterations.min(4),
        |parameters| {
            epipolar_objective(
                parameters,
                &specs,
                cameras,
                reference_index,
                &initial_fit,
                intrinsics_mode,
                options,
            )
        },
    );
    // Retain the epipolar candidate only when it also improves the actual
    // finite-depth fit objective; otherwise the factory rig is the better BA
    // initializer.
    let initialized = if objective(
        &angular_candidate,
        &specs,
        cameras,
        &initial_fit,
        intrinsics_mode,
        options,
    ) < initial_before_objective
    {
        angular_candidate
    } else {
        vec![0.0; specs.len()]
    };
    let (mut parameters, _, bundle_iterations) =
        coordinate_optimize(initialized, &specs, options.max_iterations, |parameters| {
            objective(
                parameters,
                &specs,
                cameras,
                &initial_fit,
                intrinsics_mode,
                options,
            )
        });
    drop(initial_fit);
    let mut iterations = initialization_iterations + bundle_iterations;
    let mut membership_iterations = 0usize;
    let mut membership_rejected_observations = 0usize;
    let mut membership_rejected_tracks = 0usize;
    for _ in 0..options.max_membership_iterations {
        let refinements = refinements_from_parameters(cameras.len(), &parameters, &specs);
        let Some(current_cameras) = resolve_cameras(cameras, &refinements, intrinsics_mode) else {
            break;
        };
        let (rejected_observations, rejected_tracks) =
            update_fit_track_membership(&mut fit_tracks, &current_cameras, options);
        if rejected_observations == 0 && rejected_tracks == 0 {
            break;
        }
        membership_iterations += 1;
        membership_rejected_observations += rejected_observations;
        membership_rejected_tracks += rejected_tracks;
        if fit_tracks.len() < options.min_tracks {
            report.optimizer_iterations = iterations;
            report.membership_iterations = membership_iterations;
            report.fit_membership_rejected_observations = membership_rejected_observations;
            report.fit_membership_rejected_tracks = membership_rejected_tracks;
            report.fit_tracks = fit_tracks.len();
            report.fallback_reason = Some(format!(
                "only {} fit tracks remain after robust membership update (need {})",
                fit_tracks.len(),
                options.min_tracks,
            ));
            return RigRefinementOutcome {
                refinements: zero,
                report,
            };
        }
        let current_fit = fit_tracks.iter().collect::<Vec<_>>();
        let (next_parameters, _, next_iterations) =
            coordinate_optimize(parameters, &specs, options.max_iterations, |parameters| {
                objective(
                    parameters,
                    &specs,
                    cameras,
                    &current_fit,
                    intrinsics_mode,
                    options,
                )
            });
        parameters = next_parameters;
        iterations += next_iterations;
    }
    report.membership_iterations = membership_iterations;
    report.fit_membership_rejected_observations = membership_rejected_observations;
    report.fit_membership_rejected_tracks = membership_rejected_tracks;
    report.fit_tracks = fit_tracks.len();

    let candidate_refinements = refinements_from_parameters(cameras.len(), &parameters, &specs);
    let Some(candidate_cameras) = resolve_cameras(cameras, &candidate_refinements, intrinsics_mode)
    else {
        report.fallback_reason = Some("candidate physical model could not be resolved".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    };
    let fit = fit_tracks.iter().collect::<Vec<_>>();
    let before_objective = objective(
        &factory_parameters,
        &specs,
        cameras,
        &fit,
        intrinsics_mode,
        options,
    );
    let current_objective = objective(&parameters, &specs, cameras, &fit, intrinsics_mode, options);
    let fit_before = evaluate(&fit, &factory_cameras, options, true);
    let fit_after = evaluate(&fit, &candidate_cameras, options, true);
    let validation_before = evaluate(&validation, &factory_cameras, options, true);
    let validation_after = evaluate(&validation, &candidate_cameras, options, true);
    let fit_residuals_before = residual_distribution(&fit_before.residuals);
    let fit_residuals_after = residual_distribution(&fit_after.residuals);
    let held_out_residuals_before = options
        .held_out_validation
        .then(|| residual_distribution(&validation_before.residuals))
        .unwrap_or_default();
    let held_out_residuals_after = options
        .held_out_validation
        .then(|| residual_distribution(&validation_after.residuals))
        .unwrap_or_default();
    let held_out_factory_p95_reference_px =
        held_out_residuals_before.reference_equivalent_pixels.p95;
    let held_out_p95_reference_px = held_out_residuals_after.reference_equivalent_pixels.p95;
    let held_out_improvement = if options.held_out_validation {
        (validation_before.rms() - validation_after.rms()) / validation_before.rms().max(1.0e-12)
    } else {
        0.0
    };
    let reached_bound = parameters
        .iter()
        .zip(&specs)
        .any(|(&value, spec)| value.abs() >= spec.bound * 0.98);
    let per_camera = per_camera_reports(
        cameras,
        &fit_before,
        &fit_after,
        &validation_before,
        &validation_after,
    );
    let camera_regression = per_camera.iter().enumerate().any(|(camera_index, camera)| {
        let affects_acceptance =
            camera_index == reference_index || specs.iter().any(|spec| spec.camera == camera_index);
        affects_acceptance
            && camera.validation_samples >= 12
            && camera.validation_rms_after > camera.validation_rms_before * 1.05 + 0.02
    });
    let training_improved = fit_after.rms() < fit_before.rms();
    let validation_improved =
        !options.held_out_validation || held_out_improvement >= options.min_validation_improvement;
    let fit_positive_depth = fit_after.positive_depth_fraction();
    let validation_positive_depth = options
        .held_out_validation
        .then(|| validation_after.positive_depth_fraction())
        .unwrap_or(0.0);
    let physical_depth_valid = fit_positive_depth >= options.min_positive_depth_fraction
        && (!options.held_out_validation
            || validation_positive_depth >= options.min_positive_depth_fraction);
    // Absolute p95 remains visible as a model-quality warning, but relative
    // held-out improvement is the diagnostic selection criterion. These
    // observations are image-derived consistency evidence rather than
    // external metric-depth ground truth, so an arbitrary absolute cutoff
    // must not force the known-worse factory candidate.
    let accepted = training_improved
        && validation_improved
        && physical_depth_valid
        && !reached_bound
        && !camera_regression;

    let mut conditions = fit_tracks
        .iter()
        .chain(&validation_tracks)
        .map(|track| track.condition)
        .collect::<Vec<_>>();
    conditions.sort_by(f64::total_cmp);
    let mut ray_angles = fit_tracks
        .iter()
        .chain(&validation_tracks)
        .map(|track| track.max_ray_angle_degrees)
        .collect::<Vec<_>>();
    ray_angles.sort_by(f64::total_cmp);
    report.accepted = accepted;
    report.validation_evaluated = options.held_out_validation;
    report.optimizer_iterations = iterations;
    report.training_objective_before = before_objective;
    report.training_objective_after = current_objective;
    report.reprojection_rms_before = fit_before.rms();
    report.reprojection_rms_after = fit_after.rms();
    report.held_out_rms_before = options
        .held_out_validation
        .then(|| validation_before.rms())
        .unwrap_or(0.0);
    report.held_out_rms_after = options
        .held_out_validation
        .then(|| validation_after.rms())
        .unwrap_or(0.0);
    report.fit_positive_depth_fraction_before = fit_before.positive_depth_fraction();
    report.fit_positive_depth_fraction_after = fit_positive_depth;
    report.held_out_positive_depth_fraction_before = options
        .held_out_validation
        .then(|| validation_before.positive_depth_fraction())
        .unwrap_or(0.0);
    report.held_out_positive_depth_fraction_after = validation_positive_depth;
    report.held_out_relative_improvement = held_out_improvement;
    report.fit_residuals_before = fit_residuals_before;
    report.fit_residuals_after = fit_residuals_after;
    report.held_out_residuals_before = held_out_residuals_before;
    report.held_out_residuals_after = held_out_residuals_after;
    report.median_triangulation_condition = percentile(&conditions, 0.5);
    report.p90_triangulation_condition = percentile(&conditions, 0.9);
    report.median_max_ray_angle_degrees = percentile(&ray_angles, 0.5);
    report.corrections = correction_reports(cameras, &candidate_refinements, &parameters, &specs);
    report.residual_field = residual_field_reports(
        cameras,
        &validation_before.residuals,
        &validation_after.residuals,
    );
    report.held_out_observations = held_out_observation_reports(
        cameras,
        &validation_before.residuals,
        &validation_after.residuals,
        held_out_factory_p95_reference_px,
        held_out_p95_reference_px,
    );
    report.per_camera = per_camera;
    report.fallback_reason = (!accepted).then(|| {
        let mut reasons = Vec::new();
        if reached_bound {
            reasons.push("candidate reached a physical correction bound".to_owned());
        }
        if !physical_depth_valid {
            reasons.push(if options.held_out_validation {
                format!(
                    "positive-depth support is only {:.1}% fit/{:.1}% held out (need {:.1}%)",
                    fit_positive_depth * 100.0,
                    validation_positive_depth * 100.0,
                    options.min_positive_depth_fraction * 100.0,
                )
            } else {
                format!(
                    "positive-depth support is only {:.1}% (need {:.1}%)",
                    fit_positive_depth * 100.0,
                    options.min_positive_depth_fraction * 100.0,
                )
            });
        }
        if camera_regression {
            reasons.push("held-out residual regressed for an observed camera".to_owned());
        }
        if !training_improved {
            reasons.push("training reprojection RMS did not improve".to_owned());
        }
        if options.held_out_validation && !validation_improved {
            reasons.push(format!(
                "held-out improvement {:+.3}% is below required {:+.3}%",
                held_out_improvement * 100.0,
                options.min_validation_improvement * 100.0
            ));
        }
        reasons.join("; ")
    });

    RigRefinementOutcome {
        refinements: if accepted {
            candidate_refinements
        } else {
            zero
        },
        report,
    }
}

fn resolve_cameras(
    inputs: &[RigCameraInput<'_>],
    refinements: &[CameraRefinement],
    intrinsics_mode: IntrinsicsMode,
) -> Option<Vec<ResolvedCamera>> {
    inputs
        .iter()
        .zip(refinements)
        .map(|(input, refinement)| {
            ResolvedCamera::new(
                input.calibration?,
                input.state?,
                intrinsics_mode,
                refinement,
            )
            .ok()
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct PhysicalPatchProjection {
    centre: Vec2,
    target_centre: Vec2,
    target_dx: Vec2,
    target_dy: Vec2,
}

impl PhysicalPatchProjection {
    #[inline]
    fn map(self, point: Vec2) -> Vec2 {
        let dx = point[0] - self.centre[0];
        let dy = point[1] - self.centre[1];
        [
            self.target_centre[0] + self.target_dx[0] * dx + self.target_dy[0] * dy,
            self.target_centre[1] + self.target_dx[1] * dx + self.target_dy[1] * dy,
        ]
    }

    #[inline]
    fn local_scale(self) -> f64 {
        let determinant =
            self.target_dx[0] * self.target_dy[1] - self.target_dx[1] * self.target_dy[0];
        determinant.abs().sqrt().max(1.0e-6)
    }
}

#[derive(Clone, Copy, Debug)]
struct PhysicalViewMatch {
    camera: usize,
    score: f32,
    target_pixel: Vec2,
    residual: Vec2,
    /// Stage-2 proposal projected onto the direction perpendicular to the
    /// physical depth locus. Search/refinement bounds are relative to this
    /// proposal, not relative to the uncorrected factory projection.
    residual_proposal: Vec2,
    local_scale: f64,
}

#[derive(Clone, Debug)]
struct PhysicalDepthCandidate {
    label: usize,
    depth: f64,
    ranking: f32,
    mean_score: f32,
    matches: Vec<PhysicalViewMatch>,
}

#[derive(Debug, Default)]
struct PhysicalTrackBuild {
    tracks: Vec<Track>,
    observations: usize,
    candidates: usize,
    depth_hypotheses: usize,
    max_depth_refinement_levels: usize,
    depth_refinement_histogram: Vec<usize>,
    observed_max_projected_step_px: f64,
    rejected_no_supported_depth: usize,
    rejected_ambiguous_depth: usize,
    rejected_insufficient_views: usize,
    rejected_observations: usize,
    rejected_tracks: usize,
}

#[derive(Debug, Default)]
struct PhysicalLuminancePyramid {
    /// Level zero is borrowed directly from `RigCameraInput::luminance`;
    /// element zero here is pyramid level one.
    lower_levels: Vec<Plane>,
}

impl PhysicalLuminancePyramid {
    fn build(base: Option<&Plane>, maximum_lower_levels: usize, patch_radius: usize) -> Self {
        let Some(base) = base else {
            return Self::default();
        };
        let minimum_size = patch_radius.saturating_mul(2).saturating_add(5).max(8);
        let mut lower_levels = Vec::with_capacity(maximum_lower_levels);
        let mut previous = base;
        for _ in 0..maximum_lower_levels {
            if previous.width / 2 < minimum_size || previous.height / 2 < minimum_size {
                break;
            }
            lower_levels.push(previous.downsample());
            previous = lower_levels.last().unwrap();
        }
        Self { lower_levels }
    }

    fn level<'a>(&'a self, base: Option<&'a Plane>, level: usize) -> Option<&'a Plane> {
        if level == 0 {
            base
        } else {
            self.lower_levels.get(level - 1)
        }
    }
}

#[inline]
fn luminance_level_scale(level: usize) -> f64 {
    0.5 / (1usize << level.min(30)) as f64
}

#[inline]
fn sensor_to_luminance_coordinate(pixel: f64, level: usize) -> f32 {
    let scale = luminance_level_scale(level);
    let first_sensor_centre = 0.5 / scale - 0.5;
    ((pixel - first_sensor_centre) * scale) as f32
}

/// A candidate depth is the correspondence model.  The source patch is lifted
/// onto the local scene plane implied by that depth, then projected through the
/// complete target camera model.  Unlike the legacy aligner, no homography or
/// pre-existing image warp defines the match locus.
fn physical_patch_projection(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    depth: f64,
) -> Option<PhysicalPatchProjection> {
    const DERIVATIVE_STEP: f64 = 2.0;
    let centre_ray = reference_camera.pixel_to_ray(centre);
    let surface_point = add(centre_ray.origin, scale(centre_ray.direction, depth));
    let surface_normal = centre_ray.direction;
    let project = |point: Vec2| -> Option<Vec2> {
        let ray = reference_camera.pixel_to_ray(point);
        let denominator = dot(ray.direction, surface_normal);
        if denominator.abs() <= 1.0e-9 {
            return None;
        }
        let distance = dot(sub(surface_point, ray.origin), surface_normal) / denominator;
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        target_camera.project(add(ray.origin, scale(ray.direction, distance)))
    };

    let target_centre = project(centre)?;
    let left = project([centre[0] - DERIVATIVE_STEP, centre[1]])?;
    let right = project([centre[0] + DERIVATIVE_STEP, centre[1]])?;
    let above = project([centre[0], centre[1] - DERIVATIVE_STEP])?;
    let below = project([centre[0], centre[1] + DERIVATIVE_STEP])?;
    let derivative_scale = 1.0 / (2.0 * DERIVATIVE_STEP);
    Some(PhysicalPatchProjection {
        centre,
        target_centre,
        target_dx: [
            (right[0] - left[0]) * derivative_scale,
            (right[1] - left[1]) * derivative_scale,
        ],
        target_dy: [
            (below[0] - above[0]) * derivative_scale,
            (below[1] - above[1]) * derivative_scale,
        ],
    })
}

/// ZNCC of a physically projected patch.  `residual` is a bounded bootstrap
/// allowance for the capture-specific rig error.  It does not change the
/// depth-induced Jacobian and is subsequently converted into an ordinary
/// target-camera observation that must triangulate coherently with the other
/// views.
fn physical_patch_zncc(
    reference: &Plane,
    target: &Plane,
    projection: PhysicalPatchProjection,
    centre: Vec2,
    residual: Vec2,
    radius: usize,
    pyramid_level: usize,
) -> Option<f32> {
    let centre_reference = reference.sample(
        sensor_to_luminance_coordinate(centre[0], pyramid_level),
        sensor_to_luminance_coordinate(centre[1], pyramid_level),
    )?;
    let sensor_step = 1.0 / luminance_level_scale(pyramid_level);
    let sigma = (radius as f32 * 0.75).max(1.0);
    let mut weight_sum = 0.0f32;
    let mut sum_reference = 0.0f32;
    let mut sum_target = 0.0f32;
    let mut sum_reference_sq = 0.0f32;
    let mut sum_target_sq = 0.0f32;
    let mut sum_product = 0.0f32;

    for dy in -(radius as isize)..=radius as isize {
        for dx in -(radius as isize)..=radius as isize {
            let point = [
                centre[0] + dx as f64 * sensor_step,
                centre[1] + dy as f64 * sensor_step,
            ];
            let reference_value = reference.sample(
                sensor_to_luminance_coordinate(point[0], pyramid_level),
                sensor_to_luminance_coordinate(point[1], pyramid_level),
            )?;
            let mapped = projection.map(point);
            let target_value = target.sample(
                sensor_to_luminance_coordinate(mapped[0] + residual[0], pyramid_level),
                sensor_to_luminance_coordinate(mapped[1] + residual[1], pyramid_level),
            )?;
            let distance_sq = (dx * dx + dy * dy) as f32;
            let spatial = (-distance_sq / (2.0 * sigma * sigma)).exp();
            let range = (-1.2 * (reference_value - centre_reference).abs()).exp();
            let weight = spatial * range;
            weight_sum += weight;
            sum_reference += weight * reference_value;
            sum_target += weight * target_value;
            sum_reference_sq += weight * reference_value * reference_value;
            sum_target_sq += weight * target_value * target_value;
            sum_product += weight * reference_value * target_value;
        }
    }
    if weight_sum <= 1.0e-6 {
        return None;
    }
    let covariance = sum_product - sum_reference * sum_target / weight_sum;
    let reference_variance =
        (sum_reference_sq - sum_reference * sum_reference / weight_sum).max(0.0);
    let target_variance = (sum_target_sq - sum_target * sum_target / weight_sum).max(0.0);
    let denominator = (reference_variance * target_variance).sqrt();
    (denominator > 1.0e-8).then_some((covariance / denominator).clamp(-1.0, 1.0))
}

fn physical_epipolar_tangent(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    depth: f64,
    options: &RigRefinementOptions,
) -> Option<Vec2> {
    let inverse = 1.0 / depth;
    let min_inverse = 1.0 / options.physical_match_far_depth;
    let max_inverse = 1.0 / options.physical_match_near_depth;
    let inverse_step = (max_inverse - min_inverse)
        / (options.physical_match_planes.saturating_sub(1).max(1) as f64);
    let a_inverse = (inverse - inverse_step).clamp(min_inverse, max_inverse);
    let b_inverse = (inverse + inverse_step).clamp(min_inverse, max_inverse);
    match (
        physical_patch_projection(reference_camera, target_camera, centre, 1.0 / a_inverse),
        physical_patch_projection(reference_camera, target_camera, centre, 1.0 / b_inverse),
    ) {
        (Some(a), Some(b)) => {
            let delta = [
                b.target_centre[0] - a.target_centre[0],
                b.target_centre[1] - a.target_centre[1],
            ];
            let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
            (length > 1.0e-6).then_some([delta[0] / length, delta[1] / length])
        }
        _ => None,
    }
}

fn physical_localization_covariance(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    depth: f64,
    options: &RigRefinementOptions,
) -> [[f64; 2]; 2] {
    // The initial matcher deliberately has more freedom along the physical
    // depth locus than perpendicular to it. Preserve that anisotropy in the
    // persistent observation so BA does not treat both axes as equally known.
    let Some(tangent) =
        physical_epipolar_tangent(reference_camera, target_camera, centre, depth, options)
    else {
        return [[1.0, 0.0], [0.0, 1.0]];
    };
    let normal = [-tangent[1], tangent[0]];
    let tangent_variance = 4.0;
    let normal_variance = 0.25;
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            tangent_variance * tangent[row] * tangent[column]
                + normal_variance * normal[row] * normal[column]
        })
    })
}

fn epipolar_residual_grid(
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    depth: f64,
    radius: f64,
    options: &RigRefinementOptions,
) -> Vec<Vec2> {
    if !radius.is_finite() || radius <= 0.0 {
        return vec![[0.0, 0.0]];
    }
    let tangent =
        physical_epipolar_tangent(reference_camera, target_camera, centre, depth, options);
    let Some(tangent) = tangent else {
        // At effectively infinite depth the epipolar curve can collapse to a
        // point. Those observations still constrain orientation, so search a
        // small 2-D bootstrap box rather than arbitrarily choosing one normal
        // direction.
        let half = radius * 0.5;
        let coordinates = [-radius, -half, 0.0, half, radius];
        let mut offsets = Vec::with_capacity(25);
        offsets.push([0.0, 0.0]);
        for dy in coordinates {
            for dx in coordinates {
                if dx != 0.0 || dy != 0.0 {
                    offsets.push([dx, dy]);
                }
            }
        }
        return offsets;
    };
    let normal = [-tangent[1], tangent[0]];
    let tangential_tolerance = (radius / 8.0).clamp(1.0, 4.0);
    let normal_offsets = [-radius, -radius * 0.5, 0.0, radius * 0.5, radius];
    let tangent_offsets = [-tangential_tolerance, 0.0, tangential_tolerance];
    let mut offsets = Vec::with_capacity(normal_offsets.len() * tangent_offsets.len());
    offsets.push([0.0, 0.0]);
    for normal_distance in normal_offsets {
        for tangent_distance in tangent_offsets {
            if normal_distance == 0.0 && tangent_distance == 0.0 {
                continue;
            }
            offsets.push([
                normal[0] * normal_distance + tangent[0] * tangent_distance,
                normal[1] * normal_distance + tangent[1] * tangent_distance,
            ]);
        }
    }
    offsets
}

/// Convert the stage-2 image warp into a proposal for capture-specific rig
/// error without importing its unknown scene depth. The component along the
/// physical epipolar trajectory is deliberately discarded: using it would
/// place every depth hypothesis near the same measured pixel and erase the
/// parallax signal that the physical matcher is meant to estimate.
fn measured_epipolar_residual_proposal(
    alignment: &ModuleAlignment,
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    projection: PhysicalPatchProjection,
    centre: Vec2,
    depth: f64,
    options: &RigRefinementOptions,
) -> Vec2 {
    let Some(measured) = alignment.warp.map(centre[0] as f32, centre[1] as f32) else {
        return [0.0, 0.0];
    };
    let delta = [
        f64::from(measured[0]) - projection.target_centre[0],
        f64::from(measured[1]) - projection.target_centre[1],
    ];
    if !delta[0].is_finite() || !delta[1].is_finite() {
        return [0.0, 0.0];
    }
    let tangent =
        physical_epipolar_tangent(reference_camera, target_camera, centre, depth, options);
    perpendicular_residual_proposal(delta, tangent)
}

fn perpendicular_residual_proposal(delta: Vec2, tangent: Option<Vec2>) -> Vec2 {
    let Some(tangent) = tangent else {
        // With no observable depth direction there is no parallax component
        // to protect, so the complete measured displacement is the only useful
        // proposal for the small two-dimensional bootstrap search.
        return delta;
    };
    let normal = [-tangent[1], tangent[0]];
    let distance = dot2(delta, normal);
    [normal[0] * distance, normal[1] * distance]
}

fn match_physical_view_at_depth(
    reference: &Plane,
    target: &Plane,
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    measured_alignment: &ModuleAlignment,
    target_index: usize,
    centre: Vec2,
    depth: f64,
    pyramid_level: usize,
    options: &RigRefinementOptions,
) -> Option<PhysicalViewMatch> {
    let projection = physical_patch_projection(reference_camera, target_camera, centre, depth)?;
    let measured_proposal = measured_epipolar_residual_proposal(
        measured_alignment,
        reference_camera,
        target_camera,
        projection,
        centre,
        depth,
        options,
    );
    let offsets = epipolar_residual_grid(
        reference_camera,
        target_camera,
        centre,
        depth,
        options.physical_match_residual_radius_px,
        options,
    );
    let search = |proposal: Vec2| {
        let mut best: Option<(f32, f32, Vec2)> = None;
        for &offset in &offsets {
            let residual = [proposal[0] + offset[0], proposal[1] + offset[1]];
            let Some(score) = physical_patch_zncc(
                reference,
                target,
                projection,
                centre,
                residual,
                options.physical_match_patch_radius,
                pyramid_level,
            ) else {
                continue;
            };
            // Equal-score plateaus prefer the active proposal rather than the
            // edge of its local bootstrap search band.
            let penalty = ((offset[0] * offset[0] + offset[1] * offset[1]).sqrt()
                / options.physical_match_residual_radius_px.max(1.0))
                as f32
                * 0.001;
            let objective = score - penalty;
            if best.is_none_or(|(best_objective, _, _)| objective > best_objective) {
                best = Some((objective, score, residual));
            }
        }
        best
    };

    // Preserve the original factory-centred result whenever it is already a
    // supported match. The measured proposal is additive recovery for views
    // outside that band, never a replacement for known-good factory evidence.
    let factory = search([0.0, 0.0]);
    let (score, residual, residual_proposal) = if let Some((_, score, residual)) = factory
        && score >= options.physical_match_min_score
    {
        (score, residual, [0.0, 0.0])
    } else if dot2(measured_proposal, measured_proposal) > 1.0 {
        let (_, score, residual) = search(measured_proposal).or(factory)?;
        (score, residual, measured_proposal)
    } else {
        let (_, score, residual) = factory?;
        (score, residual, [0.0, 0.0])
    };
    Some(PhysicalViewMatch {
        camera: target_index,
        score,
        target_pixel: [
            projection.target_centre[0] + residual[0],
            projection.target_centre[1] + residual[1],
        ],
        residual,
        residual_proposal,
        local_scale: projection.local_scale(),
    })
}

fn refine_physical_view_match(
    reference_input: &RigCameraInput<'_>,
    target_input: &RigCameraInput<'_>,
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    depth: f64,
    mut matched: PhysicalViewMatch,
    options: &RigRefinementOptions,
) -> PhysicalViewMatch {
    let (Some(reference), Some(target), Some(projection)) = (
        reference_input.luminance,
        target_input.luminance,
        physical_patch_projection(reference_camera, target_camera, centre, depth),
    ) else {
        return matched;
    };
    let step = (options.physical_match_residual_radius_px / 8.0).clamp(1.0, 4.0);
    let base = matched.residual;
    for dy in [-step, 0.0, step] {
        for dx in [-step, 0.0, step] {
            let residual = [base[0] + dx, base[1] + dy];
            let proposal_delta = [
                residual[0] - matched.residual_proposal[0],
                residual[1] - matched.residual_proposal[1],
            ];
            if dot2(proposal_delta, proposal_delta).sqrt()
                > options.physical_match_residual_radius_px * 1.10
            {
                continue;
            }
            let Some(score) = physical_patch_zncc(
                reference,
                target,
                projection,
                centre,
                residual,
                options.physical_match_patch_radius,
                0,
            ) else {
                continue;
            };
            if score > matched.score {
                matched.score = score;
                matched.residual = residual;
                matched.target_pixel = [
                    projection.target_centre[0] + residual[0],
                    projection.target_centre[1] + residual[1],
                ];
            }
        }
    }
    matched
}

fn physical_depth_candidate(
    matches: Vec<PhysicalViewMatch>,
    label: usize,
    depth: f64,
    options: &RigRefinementOptions,
) -> Option<PhysicalDepthCandidate> {
    let supporting = matches
        .into_iter()
        .filter(|matched| matched.score >= options.physical_match_min_score)
        .collect::<Vec<_>>();
    if supporting.len() + 1 < options.physical_match_min_views.max(2) {
        return None;
    }
    let mean_score = supporting.iter().map(|m| m.score).sum::<f32>() / supporting.len() as f32;
    // Each independent camera contributes positive evidence above the minimum
    // match likelihood. This deliberately lets a broad moderate consensus beat
    // a small accidental set of excellent repeated-texture peaks, while views
    // below the support threshold are neutral rather than negative votes.
    let ranking = supporting
        .iter()
        .map(|matched| (matched.score - options.physical_match_min_score).max(0.0) + 0.05)
        .sum::<f32>();
    Some(PhysicalDepthCandidate {
        label,
        depth,
        ranking,
        mean_score,
        matches: supporting,
    })
}

fn evaluate_physical_depth_candidate(
    cameras: &[RigCameraInput<'_>],
    pyramids: &[PhysicalLuminancePyramid],
    reference_index: usize,
    resolved: &[ResolvedCamera],
    measured_alignments: &[ModuleAlignment],
    centre: Vec2,
    label: usize,
    inverse_depth: f64,
    pyramid_level: usize,
    options: &RigRefinementOptions,
) -> Option<PhysicalDepthCandidate> {
    if !inverse_depth.is_finite() || inverse_depth <= 0.0 {
        return None;
    }
    let depth = 1.0 / inverse_depth;
    let reference =
        pyramids[reference_index].level(cameras[reference_index].luminance, pyramid_level)?;
    let mut matches = Vec::new();
    for target_index in 0..cameras.len() {
        if target_index == reference_index
            || !cameras[target_index].match_evidence_enabled
            || cameras[target_index].luminance.is_none()
            || cameras[target_index].calibration.is_none()
            || cameras[target_index].state.is_none()
        {
            continue;
        }
        let Some(target) =
            pyramids[target_index].level(cameras[target_index].luminance, pyramid_level)
        else {
            continue;
        };
        if let Some(matched) = match_physical_view_at_depth(
            reference,
            target,
            &resolved[reference_index],
            &resolved[target_index],
            &measured_alignments[target_index],
            target_index,
            centre,
            depth,
            pyramid_level,
            options,
        ) {
            matches.push(matched);
        }
    }
    physical_depth_candidate(matches, label, depth, options)
}

fn semi_dense_reference_candidates(
    reference: &RigCameraInput<'_>,
    reference_camera: &ResolvedCamera,
    options: &RigRefinementOptions,
) -> Vec<(Vec2, f32)> {
    let Some(plane) = reference.luminance else {
        return Vec::new();
    };
    let radius = options.physical_match_patch_radius.max(1);
    let window = radius * 2 + 1;
    if plane.width <= window + 2 || plane.height <= window + 2 {
        return Vec::new();
    }
    let stride = options.physical_match_stride_px.div_ceil(2).max(radius + 1);
    let mut candidates = Vec::new();
    let mut y = radius + 1;
    while y + radius + 1 < plane.height {
        let mut x = radius + 1;
        while x + radius + 1 < plane.width {
            let structure = plane.window_std(x - radius, y - radius, window);
            if structure >= options.physical_match_min_structure {
                let centre = [x as f64 * 2.0 + 0.5, y as f64 * 2.0 + 0.5];
                if reference_camera.contains(centre) {
                    candidates.push((centre, structure));
                }
            }
            x += stride;
        }
        y += stride;
    }

    let maximum = options.physical_match_max_candidates.max(1);
    if candidates.len() <= maximum {
        return candidates;
    }
    // Preserve field coverage instead of taking only the globally strongest
    // edges, which would make the calibration information matrix spatially
    // degenerate on one convenient facade/railing.
    let source = candidates;
    (0..maximum)
        .map(|index| {
            let selected = index * source.len() / maximum;
            source[selected]
        })
        .collect()
}

/// Enforce that the final independently localized image observations describe
/// one 3-D point under the factory rig. Outlying target observations are
/// removed one at a time and the point is retriangulated after every removal;
/// the fixed reference observation is never discarded.
fn prune_physical_track(
    mut track: Track,
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Option<(Track, usize)> {
    // The stage-2 proposal represents the large capture-specific component we
    // are trying to fit. Requiring raw measured pixels to agree with the
    // uncorrected factory rig would reject precisely those useful tracks. For
    // bootstrap validation only, remove the proposal and test whether the
    // remaining local match residuals describe one factory-rig 3-D point.
    let mut normalized = track.clone();
    for observation in &mut normalized.observations {
        observation.pixel[0] -= observation.bootstrap_residual_proposal[0];
        observation.pixel[1] -= observation.bootstrap_residual_proposal[1];
    }
    let (normalized, removed) = prune_track_observations(
        normalized,
        cameras,
        options,
        options.physical_match_min_views.max(2),
        options.physical_match_max_reprojection_reference_px,
    )?;
    track.observations.retain(|observation| {
        normalized
            .observations
            .iter()
            .any(|retained| retained.camera == observation.camera)
    });
    track.condition = normalized.condition;
    track.max_ray_angle_degrees = normalized.max_ray_angle_degrees;
    Some((track, removed))
}

fn prune_track_observations(
    mut track: Track,
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
    minimum_views: usize,
    threshold: f64,
) -> Option<(Track, usize)> {
    if !threshold.is_finite() || threshold <= 0.0 {
        return None;
    }
    let original_observations = track.observations.len();
    loop {
        let triangulated = triangulate(&track.observations, cameras, options)?;
        if !triangulation_has_positive_depth(&track.observations, cameras, triangulated.point) {
            return None;
        }
        let mut all_consistent = true;
        let mut worst_target: Option<(usize, f64)> = None;
        for (index, observation) in track.observations.iter().enumerate() {
            let projected = project_observation(
                &cameras[observation.camera],
                observation,
                triangulated.point,
            )?;
            let pixel_residual = [
                projected[0] - observation.pixel[0],
                projected[1] - observation.pixel[1],
            ];
            let sensor_error = dot2(pixel_residual, pixel_residual).sqrt();
            let reference_error = sensor_error / observation.local_scale.clamp(0.25, 4.0);
            all_consistent &= reference_error <= threshold;
            if !observation.fixed_gauge
                && worst_target.is_none_or(|(_, worst)| reference_error > worst)
            {
                worst_target = Some((index, reference_error));
            }
        }
        if all_consistent {
            track.condition = triangulated.condition;
            track.max_ray_angle_degrees = triangulated.max_ray_angle_degrees;
            let removed = original_observations - track.observations.len();
            return Some((track, removed));
        }
        if track.observations.len() <= minimum_views {
            return None;
        }
        let (worst_index, _) = worst_target?;
        track.observations.remove(worst_index);
    }
}

fn update_fit_track_membership(
    tracks: &mut Vec<Track>,
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> (usize, usize) {
    let original_tracks = tracks.len();
    let mut rejected_observations = 0usize;
    let mut retained = Vec::with_capacity(tracks.len());
    for track in tracks.drain(..) {
        let original_observations = track.observations.len();
        if let Some((track, removed)) = prune_track_observations(
            track,
            cameras,
            options,
            options.physical_match_min_views.max(2),
            options.max_validation_p95_reference_px,
        ) {
            rejected_observations += removed;
            retained.push(track);
        } else {
            rejected_observations += original_observations;
        }
    }
    let rejected_tracks = original_tracks - retained.len();
    *tracks = retained;
    (rejected_observations, rejected_tracks)
}

/// Generate sparse calibration tracks by solving matching and depth jointly
/// from the factory physical rig.  The reference camera supplies a spatially
/// balanced semi-dense set of structured patches; every candidate inverse
/// depth predicts the patch in all other cameras. The coarse sweep runs at an
/// automatically selected pyramid level where neighbouring hypotheses are
/// close in projected pixels, then a small beam of separated depth modes is
/// refined back to the native matching level. A bounded residual band
/// bootstraps capture-specific calibration error, but no homography controls
/// the search or decides which observations are legal.
fn maximum_projected_depth_step(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    resolved: &[ResolvedCamera],
    centre: Vec2,
    intervals: usize,
    min_inverse: f64,
    max_inverse: f64,
) -> f64 {
    if intervals == 0 {
        return f64::INFINITY;
    }
    let reference_ray = resolved[reference_index].pixel_to_ray(centre);
    let mut maximum = 0.0f64;
    for target_index in 0..cameras.len() {
        if target_index == reference_index
            || !cameras[target_index].match_evidence_enabled
            || cameras[target_index].luminance.is_none()
            || cameras[target_index].calibration.is_none()
            || cameras[target_index].state.is_none()
        {
            continue;
        }
        let mut previous: Option<Vec2> = None;
        for index in 0..=intervals {
            let t = index as f64 / intervals as f64;
            let inverse_depth = min_inverse + t * (max_inverse - min_inverse);
            let point = add(
                reference_ray.origin,
                scale(reference_ray.direction, 1.0 / inverse_depth),
            );
            // Restrict motion to the actual image domain. `project()` has a
            // deliberately generous half-frame distortion guard, whereas a
            // projected centre outside the sensor cannot supply a luminance
            // patch and must not force useful candidates onto a coarser level.
            let projected = resolved[target_index]
                .project(point)
                .filter(|pixel| resolved[target_index].contains(*pixel));
            if let (Some(a), Some(b)) = (previous, projected) {
                let delta = [b[0] - a[0], b[1] - a[1]];
                maximum = maximum.max(dot2(delta, delta).sqrt());
            }
            previous = projected;
        }
    }
    maximum
}

fn physical_depth_refinement_levels(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    resolved: &[ResolvedCamera],
    centre: Vec2,
    available_levels: usize,
    options: &RigRefinementOptions,
) -> (usize, f64) {
    let min_inverse = 1.0 / options.physical_match_far_depth;
    let max_inverse = 1.0 / options.physical_match_near_depth;
    let mut levels = 0usize;
    let maximum = available_levels.min(options.physical_match_max_depth_refinements);
    loop {
        let intervals = options
            .physical_match_planes
            .saturating_sub(1)
            .saturating_mul(1usize << levels);
        let projected_step = maximum_projected_depth_step(
            cameras,
            reference_index,
            resolved,
            centre,
            intervals,
            min_inverse,
            max_inverse,
        );
        if projected_step <= options.physical_match_max_projected_step_px || levels >= maximum {
            return (levels, projected_step);
        }
        levels += 1;
    }
}

fn separated_depth_beam(
    mut candidates: Vec<PhysicalDepthCandidate>,
    width: usize,
) -> Vec<PhysicalDepthCandidate> {
    candidates.sort_by(|left, right| right.ranking.total_cmp(&left.ranking));
    let mut selected = Vec::<PhysicalDepthCandidate>::with_capacity(width.max(1));
    for candidate in candidates {
        if selected
            .iter()
            .all(|existing| existing.label.abs_diff(candidate.label) > 1)
        {
            selected.push(candidate);
            if selected.len() >= width.max(1) {
                break;
            }
        }
    }
    selected
}

fn hierarchical_physical_depth_candidates(
    cameras: &[RigCameraInput<'_>],
    pyramids: &[PhysicalLuminancePyramid],
    reference_index: usize,
    resolved: &[ResolvedCamera],
    measured_alignments: &[ModuleAlignment],
    centre: Vec2,
    available_levels: usize,
    options: &RigRefinementOptions,
) -> (Vec<PhysicalDepthCandidate>, usize, usize, f64) {
    let (refinement_levels, final_projected_step) = physical_depth_refinement_levels(
        cameras,
        reference_index,
        resolved,
        centre,
        available_levels,
        options,
    );
    let min_inverse = 1.0 / options.physical_match_far_depth;
    let max_inverse = 1.0 / options.physical_match_near_depth;
    let coarse_intervals = options.physical_match_planes - 1;
    let mut evaluated = 0usize;
    let mut candidates = Vec::with_capacity(options.physical_match_planes);
    for label in 0..=coarse_intervals {
        let t = label as f64 / coarse_intervals as f64;
        let inverse_depth = min_inverse + t * (max_inverse - min_inverse);
        evaluated += 1;
        if let Some(candidate) = evaluate_physical_depth_candidate(
            cameras,
            pyramids,
            reference_index,
            resolved,
            measured_alignments,
            centre,
            label,
            inverse_depth,
            refinement_levels,
            options,
        ) {
            candidates.push(candidate);
        }
    }

    for refinement in 1..=refinement_levels {
        let beam = separated_depth_beam(candidates, options.physical_match_depth_beam_width);
        if beam.is_empty() {
            return (
                Vec::new(),
                evaluated,
                refinement_levels,
                final_projected_step,
            );
        }
        let intervals = coarse_intervals.saturating_mul(1usize << refinement);
        let mut labels = BTreeMap::<usize, ()>::new();
        for candidate in beam {
            let centre_label = candidate.label.saturating_mul(2);
            let first = centre_label.saturating_sub(2);
            let last = centre_label.saturating_add(2).min(intervals);
            for label in first..=last {
                labels.insert(label, ());
            }
        }
        candidates = Vec::with_capacity(labels.len());
        let pyramid_level = refinement_levels - refinement;
        for label in labels.into_keys() {
            let t = label as f64 / intervals as f64;
            let inverse_depth = min_inverse + t * (max_inverse - min_inverse);
            evaluated += 1;
            if let Some(candidate) = evaluate_physical_depth_candidate(
                cameras,
                pyramids,
                reference_index,
                resolved,
                measured_alignments,
                centre,
                label,
                inverse_depth,
                pyramid_level,
                options,
            ) {
                candidates.push(candidate);
            }
        }
    }
    (
        candidates,
        evaluated,
        refinement_levels,
        final_projected_step,
    )
}

fn build_physical_track_chunk(
    candidates: &[(Vec2, f32)],
    cameras: &[RigCameraInput<'_>],
    pyramids: &[PhysicalLuminancePyramid],
    reference_index: usize,
    resolved: &[ResolvedCamera],
    measured_alignments: &[ModuleAlignment],
    available_pyramid_levels: usize,
    options: &RigRefinementOptions,
) -> PhysicalTrackBuild {
    let reference_camera = &resolved[reference_index];
    let mut tracks = Vec::new();
    let mut observations = 0usize;
    let mut depth_hypotheses = 0usize;
    let mut max_depth_refinement_levels = 0usize;
    let mut depth_refinement_histogram = vec![
        0usize;
        options
            .physical_match_max_depth_refinements
            .saturating_add(1)
    ];
    let mut observed_max_projected_step_px = 0.0f64;
    let mut rejected_no_supported_depth = 0usize;
    let mut rejected_ambiguous_depth = 0usize;
    let mut rejected_insufficient_views = 0usize;
    let mut rejected_observations = 0usize;
    let mut rejected_tracks = 0usize;

    for &(centre, structure) in candidates {
        let (mut depth_candidates, evaluated, refinement_levels, projected_step) =
            hierarchical_physical_depth_candidates(
                cameras,
                pyramids,
                reference_index,
                resolved,
                measured_alignments,
                centre,
                available_pyramid_levels,
                options,
            );
        depth_hypotheses += evaluated;
        max_depth_refinement_levels = max_depth_refinement_levels.max(refinement_levels);
        depth_refinement_histogram[refinement_levels] += 1;
        observed_max_projected_step_px = observed_max_projected_step_px.max(projected_step);
        if depth_candidates.is_empty() {
            rejected_no_supported_depth += 1;
            continue;
        }
        depth_candidates.sort_by(|left, right| right.ranking.total_cmp(&left.ranking));
        let best = depth_candidates.remove(0);
        let second = depth_candidates
            .iter()
            .filter(|candidate| candidate.label.abs_diff(best.label) > 1)
            .max_by(|a, b| a.ranking.total_cmp(&b.ranking));
        let relative_margin = second.map_or(1.0, |second| {
            (best.ranking - second.ranking) / best.ranking.max(1.0e-6)
        });
        if relative_margin < options.physical_match_min_margin {
            rejected_ambiguous_depth += 1;
            continue;
        }

        let depth_reliability = relative_margin.clamp(0.0, 1.0);
        let mut track_observations = Vec::with_capacity(best.matches.len() + 1);
        track_observations.push(TrackObservation {
            camera: reference_index,
            pixel: centre,
            bootstrap_residual_proposal: [0.0, 0.0],
            localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
            fixed_gauge: true,
            confidence: f64::from(best.mean_score),
            local_scale: 1.0,
            structure: f64::from(structure),
            depth_reliability: Some(f64::from(depth_reliability)),
        });
        for matched in best.matches {
            let matched = refine_physical_view_match(
                &cameras[reference_index],
                &cameras[matched.camera],
                reference_camera,
                &resolved[matched.camera],
                centre,
                best.depth,
                matched,
                options,
            );
            if matched.score < options.physical_match_min_score
                || !resolved[matched.camera].contains(matched.target_pixel)
            {
                continue;
            }
            track_observations.push(TrackObservation {
                camera: matched.camera,
                pixel: matched.target_pixel,
                bootstrap_residual_proposal: matched.residual_proposal,
                localization_covariance: physical_localization_covariance(
                    reference_camera,
                    &resolved[matched.camera],
                    centre,
                    best.depth,
                    options,
                ),
                fixed_gauge: false,
                confidence: f64::from(matched.score),
                local_scale: matched.local_scale,
                structure: f64::from(structure),
                depth_reliability: Some(f64::from(depth_reliability)),
            });
        }
        if track_observations.len() < options.physical_match_min_views.max(2) {
            rejected_insufficient_views += 1;
            continue;
        }
        let track = Track {
            key: [
                (centre[0] * 16.0).round() as i32,
                (centre[1] * 16.0).round() as i32,
            ],
            observations: track_observations,
            condition: f64::NAN,
            max_ray_angle_degrees: f64::NAN,
        };
        // Final localization is not sufficient evidence by itself. Iteratively
        // triangulate, reject incompatible target observations, and
        // retriangulate until every retained view agrees with one point.
        let Some((track, removed)) = prune_physical_track(track, resolved, options) else {
            rejected_tracks += 1;
            continue;
        };
        rejected_observations += removed;
        observations += track.observations.len() - 1;
        tracks.push(track);
    }
    PhysicalTrackBuild {
        tracks,
        observations,
        candidates: candidates.len(),
        depth_hypotheses,
        max_depth_refinement_levels,
        depth_refinement_histogram,
        observed_max_projected_step_px,
        rejected_no_supported_depth,
        rejected_ambiguous_depth,
        rejected_insufficient_views,
        rejected_observations,
        rejected_tracks,
    }
}

fn build_physical_tracks(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    resolved: &[ResolvedCamera],
    measured_alignments: &[ModuleAlignment],
    options: &RigRefinementOptions,
) -> PhysicalTrackBuild {
    if !options.physical_matching
        || reference_index >= cameras.len()
        || cameras.len() != resolved.len()
        || cameras.len() != measured_alignments.len()
        || !cameras[reference_index].match_evidence_enabled
        || cameras[reference_index].luminance.is_none()
        || options.physical_match_planes < 2
        || options.physical_match_depth_beam_width == 0
        || !options.physical_match_max_projected_step_px.is_finite()
        || options.physical_match_max_projected_step_px <= 0.0
        || !(options.physical_match_near_depth > 0.0
            && options.physical_match_far_depth > options.physical_match_near_depth)
    {
        return PhysicalTrackBuild::default();
    }

    let candidates = semi_dense_reference_candidates(
        &cameras[reference_index],
        &resolved[reference_index],
        options,
    );
    if candidates.is_empty() {
        return PhysicalTrackBuild::default();
    }
    let pyramids = cameras
        .iter()
        .map(|camera| {
            PhysicalLuminancePyramid::build(
                camera.luminance,
                options.physical_match_max_depth_refinements,
                options.physical_match_patch_radius,
            )
        })
        .collect::<Vec<_>>();
    let available_pyramid_levels = cameras
        .iter()
        .enumerate()
        .filter(|(camera, input)| {
            (*camera == reference_index || input.match_evidence_enabled)
                && input.luminance.is_some()
                && input.calibration.is_some()
                && input.state.is_some()
        })
        .map(|(camera, _)| pyramids[camera].lower_levels.len())
        .min()
        .unwrap_or(0);
    let automatic = thread::available_parallelism().map_or(1, usize::from);
    let workers = if options.threads == 0 {
        automatic
    } else {
        options.threads.min(automatic)
    }
    .clamp(1, candidates.len());
    let chunk_size = candidates.len().div_ceil(workers);
    let pyramids = &pyramids;
    let chunks = thread::scope(|scope| {
        candidates
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    build_physical_track_chunk(
                        chunk,
                        cameras,
                        pyramids,
                        reference_index,
                        resolved,
                        measured_alignments,
                        available_pyramid_levels,
                        options,
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("physical matcher worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut combined = PhysicalTrackBuild::default();
    for mut chunk in chunks {
        combined.tracks.append(&mut chunk.tracks);
        combined.observations += chunk.observations;
        combined.candidates += chunk.candidates;
        combined.depth_hypotheses += chunk.depth_hypotheses;
        combined.max_depth_refinement_levels = combined
            .max_depth_refinement_levels
            .max(chunk.max_depth_refinement_levels);
        if combined.depth_refinement_histogram.len() < chunk.depth_refinement_histogram.len() {
            combined
                .depth_refinement_histogram
                .resize(chunk.depth_refinement_histogram.len(), 0);
        }
        for (level, count) in chunk.depth_refinement_histogram.into_iter().enumerate() {
            combined.depth_refinement_histogram[level] += count;
        }
        combined.observed_max_projected_step_px = combined
            .observed_max_projected_step_px
            .max(chunk.observed_max_projected_step_px);
        combined.rejected_no_supported_depth += chunk.rejected_no_supported_depth;
        combined.rejected_ambiguous_depth += chunk.rejected_ambiguous_depth;
        combined.rejected_insufficient_views += chunk.rejected_insufficient_views;
        combined.rejected_observations += chunk.rejected_observations;
        combined.rejected_tracks += chunk.rejected_tracks;
    }
    combined
}

fn physical_match_support_reports(
    cameras: &[RigCameraInput<'_>],
    tracks: &[Track],
    reference_index: usize,
) -> Vec<RigCameraMatchSupportReport> {
    let denominator = tracks.len().max(1) as f64;
    cameras
        .iter()
        .enumerate()
        .filter(|(camera, _)| *camera != reference_index)
        .map(|(camera, input)| {
            let observations = tracks
                .iter()
                .filter(|track| {
                    track
                        .observations
                        .iter()
                        .any(|observation| observation.camera == camera)
                })
                .count();
            RigCameraMatchSupportReport {
                camera: input.name.to_owned(),
                observations,
                track_fraction: observations as f64 / denominator,
            }
        })
        .collect()
}

fn build_tracks(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    resolved: &[ResolvedCamera],
) -> (Vec<Track>, usize) {
    let mut tracks = BTreeMap::<[i32; 2], Vec<TrackObservation>>::new();
    let mut pairwise_matches = 0;
    for (camera, alignment) in alignments.iter().enumerate() {
        if camera == reference_index
            || !cameras[camera].match_evidence_enabled
            || cameras[camera].calibration.is_none()
        {
            continue;
        }
        for correspondence in &alignment.correspondences {
            if !resolved[camera].contains(correspondence.target_pixel)
                || !resolved[reference_index].contains(correspondence.reference_pixel)
            {
                continue;
            }
            pairwise_matches += 1;
            let key = [
                (correspondence.reference_pixel[0] * 16.0).round() as i32,
                (correspondence.reference_pixel[1] * 16.0).round() as i32,
            ];
            let observations = tracks.entry(key).or_insert_with(|| {
                vec![TrackObservation {
                    camera: reference_index,
                    pixel: correspondence.reference_pixel,
                    bootstrap_residual_proposal: [0.0, 0.0],
                    localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    fixed_gauge: true,
                    confidence: f64::from(correspondence.confidence),
                    local_scale: 1.0,
                    structure: f64::from(correspondence.structure),
                    depth_reliability: correspondence.depth_reliability.map(f64::from),
                }]
            });
            if observations
                .iter()
                .all(|observation| observation.camera != camera)
            {
                observations.push(TrackObservation {
                    camera,
                    pixel: correspondence.target_pixel,
                    bootstrap_residual_proposal: [0.0, 0.0],
                    localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    fixed_gauge: false,
                    confidence: f64::from(correspondence.confidence),
                    local_scale: f64::from(correspondence.local_scale),
                    structure: f64::from(correspondence.structure),
                    depth_reliability: correspondence.depth_reliability.map(f64::from),
                });
            }
        }
    }
    (
        tracks
            .into_iter()
            .filter(|(_, observations)| observations.len() >= 2)
            .map(|(key, observations)| Track {
                key,
                observations,
                condition: f64::NAN,
                max_ray_angle_degrees: f64::NAN,
            })
            .collect(),
        pairwise_matches,
    )
}

fn parameter_specs(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    tracks: &[&Track],
    options: &RigRefinementOptions,
) -> Vec<ParameterSpec> {
    let mut observations = vec![0usize; cameras.len()];
    for track in tracks {
        for observation in &track.observations {
            observations[observation.camera] += 1;
        }
    }
    let mut specs = Vec::new();
    for (camera, input) in cameras.iter().enumerate() {
        if camera == reference_index
            || input.calibration.is_none()
            || input.state.is_none()
            || observations[camera] < options.min_camera_observations
        {
            continue;
        }
        if options.max_orientation_degrees > 0.0 {
            for axis in 0..3 {
                specs.push(ParameterSpec {
                    camera,
                    kind: ParameterKind::Orientation(axis),
                    bound: options.max_orientation_degrees,
                    prior_sigma: options.orientation_prior_sigma_degrees,
                    difference_step: 0.005,
                    maximum_update: 0.12,
                });
            }
        }
        if options.max_center_offset > 0.0 {
            for axis in 0..3 {
                specs.push(ParameterSpec {
                    camera,
                    kind: ParameterKind::Center(axis),
                    bound: options.max_center_offset,
                    prior_sigma: options.center_prior_sigma,
                    difference_step: 0.05,
                    maximum_update: 0.5,
                });
            }
        }
        if options.max_sensor_offset_px > 0.0 {
            for axis in 0..2 {
                specs.push(ParameterSpec {
                    camera,
                    kind: ParameterKind::Sensor(axis),
                    bound: options.max_sensor_offset_px,
                    prior_sigma: options.sensor_offset_prior_sigma_px,
                    difference_step: 0.25,
                    maximum_update: 4.0,
                });
            }
        }
        if options.max_mirror_degrees > 0.0
            && input
                .calibration
                .is_some_and(|calibration| calibration.mirror.is_some())
        {
            specs.push(ParameterSpec {
                camera,
                kind: ParameterKind::Mirror,
                bound: options.max_mirror_degrees,
                prior_sigma: options.mirror_prior_sigma_degrees,
                difference_step: 0.0025,
                maximum_update: 0.04,
            });
        }
    }
    specs
}

fn parameter_name(kind: ParameterKind) -> String {
    match kind {
        ParameterKind::Orientation(0) => "orientation_x".to_owned(),
        ParameterKind::Orientation(1) => "orientation_y".to_owned(),
        ParameterKind::Orientation(2) => "orientation_z".to_owned(),
        ParameterKind::Orientation(axis) => format!("orientation_{axis}"),
        ParameterKind::Mirror => "mirror_angle".to_owned(),
        ParameterKind::Center(0) => "center_x".to_owned(),
        ParameterKind::Center(1) => "center_y".to_owned(),
        ParameterKind::Center(2) => "center_z".to_owned(),
        ParameterKind::Center(axis) => format!("center_{axis}"),
        ParameterKind::Sensor(0) => "sensor_x".to_owned(),
        ParameterKind::Sensor(1) => "sensor_y".to_owned(),
        ParameterKind::Sensor(axis) => format!("sensor_{axis}"),
    }
}

/// Retriangulated normalized reprojection Jacobian used only to determine
/// whether one candidate physical parameter is independently observable. The
/// nuisance 3-D point is re-estimated on both sides of the finite difference,
/// so a parameter is observable only when its residual cannot be absorbed by
/// moving the track point. Unstable gated tracks get a zero derivative while
/// retaining fixed row correspondence between all parameter columns.
fn observability_jacobian_column(
    minus_parameters: &[f64],
    plus_parameters: &[f64],
    specs: &[ParameterSpec],
    parameter: &ParameterSpec,
    difference_step: f64,
    inputs: &[RigCameraInput<'_>],
    tracks: &[&Track],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> Option<Vec<f64>> {
    let minus_refinements = refinements_from_parameters(inputs.len(), minus_parameters, specs);
    let plus_refinements = refinements_from_parameters(inputs.len(), plus_parameters, specs);
    let minus_cameras = resolve_cameras(inputs, &minus_refinements, intrinsics_mode)?;
    let plus_cameras = resolve_cameras(inputs, &plus_refinements, intrinsics_mode)?;
    let mut column = Vec::new();
    for track in tracks {
        let (Some(minus_point), Some(plus_point)) = (
            triangulate(&track.observations, &minus_cameras, options),
            triangulate(&track.observations, &plus_cameras, options),
        ) else {
            column.extend(std::iter::repeat_n(0.0, track.observations.len() * 2));
            continue;
        };
        for observation in &track.observations {
            let sigma = observation_sigma(observation).max(1.0e-6);
            let (Some(minus), Some(plus)) = (
                project_observation(
                    &minus_cameras[observation.camera],
                    observation,
                    minus_point.point,
                ),
                project_observation(
                    &plus_cameras[observation.camera],
                    observation,
                    plus_point.point,
                ),
            ) else {
                column.extend([0.0, 0.0]);
                continue;
            };
            let scale = parameter.prior_sigma / (2.0 * difference_step * sigma);
            column.push((plus[0] - minus[0]) * scale);
            column.push((plus[1] - minus[1]) * scale);
        }
    }
    (!column.is_empty()).then_some(column)
}

fn column_correlation(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let dot_product = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if left_norm <= 1.0e-12 || right_norm <= 1.0e-12 {
        0.0
    } else {
        (dot_product / (left_norm * right_norm))
            .abs()
            .clamp(0.0, 1.0)
    }
}

/// Release physical parameters only when the data can actually see them.
/// Observation count is still used as the cheap first gate; this second gate
/// measures finite-difference sensitivity and removes same-camera parameter
/// directions that are numerically indistinguishable.  When a mirror angle
/// and generic orientation explain the same bearing change, prefer the
/// physical mirror DOF unless its sensitivity is itself negligible.
fn filter_observable_parameter_specs(
    candidates: &[ParameterSpec],
    inputs: &[RigCameraInput<'_>],
    tracks: &[&Track],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> (Vec<ParameterSpec>, Vec<RigParameterObservabilityReport>) {
    const MIN_SENSITIVITY_RMS: f64 = 0.02;
    const MAX_DEGENERATE_CORRELATION: f64 = 0.995;

    if candidates.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let zero = vec![0.0; candidates.len()];
    let mut columns = Vec::with_capacity(candidates.len());
    let mut sensitivities = Vec::with_capacity(candidates.len());
    for (index, spec) in candidates.iter().enumerate() {
        let step = spec.difference_step.max(1.0e-6);
        let mut minus = zero.clone();
        let mut plus = zero.clone();
        minus[index] = -step;
        plus[index] = step;
        let column = observability_jacobian_column(
            &minus,
            &plus,
            candidates,
            spec,
            step,
            inputs,
            tracks,
            intrinsics_mode,
            options,
        )
        .unwrap_or_default();
        let sensitivity = if column.is_empty() {
            0.0
        } else {
            (column.iter().map(|value| value * value).sum::<f64>() / column.len() as f64).sqrt()
        };
        columns.push(column);
        sensitivities.push(sensitivity);
    }

    let mut max_correlations = vec![0.0f64; candidates.len()];
    let mut keep = sensitivities
        .iter()
        .map(|sensitivity| sensitivity.is_finite() && *sensitivity >= MIN_SENSITIVITY_RMS)
        .collect::<Vec<_>>();
    let mut reasons = vec![None::<String>; candidates.len()];
    for index in 0..candidates.len() {
        if !keep[index] {
            reasons[index] = Some("insufficient independent reprojection sensitivity".to_owned());
        }
    }

    for first in 0..candidates.len() {
        for second in first + 1..candidates.len() {
            if columns[first].is_empty() || columns[second].is_empty() {
                continue;
            }
            let correlation = column_correlation(&columns[first], &columns[second]);
            max_correlations[first] = max_correlations[first].max(correlation);
            max_correlations[second] = max_correlations[second].max(correlation);
            if correlation < MAX_DEGENERATE_CORRELATION
                || candidates[first].camera != candidates[second].camera
                || !keep[first]
                || !keep[second]
            {
                continue;
            }

            let first_is_mirror = matches!(candidates[first].kind, ParameterKind::Mirror);
            let second_is_mirror = matches!(candidates[second].kind, ParameterKind::Mirror);
            let first_is_sensor = matches!(candidates[first].kind, ParameterKind::Sensor(_));
            let second_is_sensor = matches!(candidates[second].kind, ParameterKind::Sensor(_));
            let first_is_orientation =
                matches!(candidates[first].kind, ParameterKind::Orientation(_));
            let second_is_orientation =
                matches!(candidates[second].kind, ParameterKind::Orientation(_));
            let loser = match (
                first_is_mirror,
                second_is_mirror,
                first_is_sensor,
                second_is_sensor,
                first_is_orientation,
                second_is_orientation,
            ) {
                // A sensor-raster shift and a small yaw/pitch can be almost
                // identical on a distortion-free or narrow field. Preserve
                // the established bearing correction unless field-dependent
                // evidence distinguishes the raster offset.
                (_, _, true, false, _, true) => first,
                (_, _, false, true, true, _) => second,
                (true, false, _, _, _, _)
                    if sensitivities[first] >= sensitivities[second] * 0.25 =>
                {
                    second
                }
                (false, true, _, _, _, _)
                    if sensitivities[second] >= sensitivities[first] * 0.25 =>
                {
                    first
                }
                _ if sensitivities[first] <= sensitivities[second] => first,
                _ => second,
            };
            keep[loser] = false;
            reasons[loser] = Some(format!(
                "degenerate with {} (|corr|={correlation:.5})",
                parameter_name(candidates[if loser == first { second } else { first }].kind)
            ));
        }
    }

    let reports = candidates
        .iter()
        .enumerate()
        .map(|(index, spec)| RigParameterObservabilityReport {
            camera: inputs[spec.camera].name.to_owned(),
            parameter: parameter_name(spec.kind),
            sensitivity_rms: sensitivities[index],
            max_correlation: max_correlations[index],
            optimized: keep[index],
            rejection_reason: reasons[index].clone(),
        })
        .collect::<Vec<_>>();
    let specs = candidates
        .iter()
        .copied()
        .zip(keep)
        .filter_map(|(spec, keep)| keep.then_some(spec))
        .collect::<Vec<_>>();
    (specs, reports)
}

fn refinements_from_parameters(
    camera_count: usize,
    parameters: &[f64],
    specs: &[ParameterSpec],
) -> Vec<CameraRefinement> {
    let mut orientations = vec![[0.0; 3]; camera_count];
    let mut mirrors = vec![0.0; camera_count];
    let mut centers = vec![[0.0; 3]; camera_count];
    let mut sensors = vec![[0.0; 2]; camera_count];
    for (&value, spec) in parameters.iter().zip(specs) {
        match spec.kind {
            ParameterKind::Orientation(axis) => orientations[spec.camera][axis] = value,
            ParameterKind::Mirror => mirrors[spec.camera] = value,
            ParameterKind::Center(axis) => centers[spec.camera][axis] = value,
            ParameterKind::Sensor(axis) => sensors[spec.camera][axis] = value,
        }
    }
    orientations
        .into_iter()
        .zip(mirrors)
        .zip(centers)
        .zip(sensors)
        .map(
            |(((orientation, mirror), center), sensor)| CameraRefinement {
                mirror_angle_offset_degrees: mirror,
                orientation_offset_degrees: (orientation != [0.0; 3]).then_some(orientation),
                center_offset_world: (center != [0.0; 3]).then_some(center),
                sensor_offset_px: (sensor != [0.0; 2]).then_some(sensor),
            },
        )
        .collect()
}

fn coordinate_optimize(
    mut parameters: Vec<f64>,
    specs: &[ParameterSpec],
    max_iterations: usize,
    objective: impl Fn(&[f64]) -> f64,
) -> (Vec<f64>, f64, usize) {
    let mut current_objective = objective(&parameters);
    let mut iterations = 0;
    for iteration in 0..max_iterations {
        let sweep_before = current_objective;
        for parameter in 0..parameters.len() {
            let spec = specs[parameter];
            let centre = parameters[parameter];
            let step = spec.difference_step;
            let mut minus = parameters.clone();
            let mut plus = parameters.clone();
            minus[parameter] = (centre - step).max(-spec.bound);
            plus[parameter] = (centre + step).min(spec.bound);
            let f_minus = objective(&minus);
            let f_plus = objective(&plus);
            let gradient = (f_plus - f_minus) / (2.0 * step);
            let curvature = (f_plus + f_minus - 2.0 * current_objective) / (step * step);
            let update = if curvature.is_finite() && curvature > 1.0e-9 {
                (-gradient / curvature).clamp(-spec.maximum_update, spec.maximum_update)
            } else if f_plus < f_minus {
                spec.maximum_update.min(step * 4.0)
            } else {
                -spec.maximum_update.min(step * 4.0)
            };
            let candidate_value = (centre + update).clamp(-spec.bound, spec.bound);
            if candidate_value == centre {
                continue;
            }
            let mut candidate = parameters.clone();
            candidate[parameter] = candidate_value;
            let candidate_objective = objective(&candidate);
            if candidate_objective < current_objective {
                parameters = candidate;
                current_objective = candidate_objective;
            } else if f_minus < current_objective || f_plus < current_objective {
                if f_minus <= f_plus {
                    parameters = minus;
                    current_objective = f_minus;
                } else {
                    parameters = plus;
                    current_objective = f_plus;
                }
            }
        }
        iterations = iteration + 1;
        if sweep_before - current_objective < 1.0e-6 {
            break;
        }
    }
    (parameters, current_objective, iterations)
}

fn epipolar_objective(
    parameters: &[f64],
    specs: &[ParameterSpec],
    inputs: &[RigCameraInput<'_>],
    reference_index: usize,
    tracks: &[&Track],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> f64 {
    let refinements = refinements_from_parameters(inputs.len(), parameters, specs);
    let Some(cameras) = resolve_cameras(inputs, &refinements, intrinsics_mode) else {
        return f64::INFINITY;
    };
    let mut cost = 0.0;
    let mut samples = 0;
    for track in tracks {
        let Some(reference) = track
            .observations
            .iter()
            .find(|observation| observation.camera == reference_index)
        else {
            continue;
        };
        let reference_ray = cameras[reference_index].pixel_to_ray(reference.pixel);
        for observation in track
            .observations
            .iter()
            .filter(|observation| observation.camera != reference_index)
        {
            let target_camera = &cameras[observation.camera];
            let target_ray = target_camera.pixel_to_ray(observation.pixel);
            let baseline = sub(target_ray.origin, reference_ray.origin);
            let baseline_length = norm(baseline);
            if baseline_length <= 1.0e-9 {
                continue;
            }
            // Calibrated epipolar error: corresponding world bearings and
            // their camera baseline must be coplanar. Express the angular
            // scalar-triple-product error in approximate pixels so the robust
            // scale remains comparable with the finite-depth objective.
            let normalised_baseline = math::scale(baseline, 1.0 / baseline_length);
            let angular_error = dot(
                normalised_baseline,
                cross(reference_ray.direction, target_ray.direction),
            )
            .abs();
            let focal = (cameras[reference_index].focal_px * target_camera.focal_px)
                .abs()
                .sqrt();
            let pair_sigma = (observation_sigma(reference).powi(2)
                + observation_sigma(observation).powi(2))
            .sqrt();
            let normalized = angular_error * focal / pair_sigma.max(1.0e-6);
            cost += huber(normalized, options.huber_delta);
            if !pair_has_positive_depth(reference_ray, target_ray) {
                // Epipolar coplanarity alone has a mirror ambiguity. This
                // discrete cheirality term selects the solution whose closest
                // ray intersection lies in front of both cameras.
                cost += 10.0;
            }
            samples += 1;
        }
    }
    let data = cost / samples.max(1) as f64;
    let prior = parameters
        .iter()
        .zip(specs)
        .map(|(&value, spec)| (value / spec.prior_sigma).powi(2))
        .sum::<f64>();
    data + options.factory_prior_weight * prior
}

fn pair_has_positive_depth(first: crate::geometry::Ray, second: crate::geometry::Ray) -> bool {
    let cosine = dot(first.direction, second.direction);
    let denominator = 1.0 - cosine * cosine;
    if denominator <= 1.0e-14 {
        return false;
    }
    let origins = sub(first.origin, second.origin);
    let first_origin = dot(first.direction, origins);
    let second_origin = dot(second.direction, origins);
    let first_depth = (cosine * second_origin - first_origin) / denominator;
    let second_depth = (second_origin - cosine * first_origin) / denominator;
    first_depth > 0.0 && second_depth > 0.0
}

fn triangulation_has_positive_depth(
    observations: &[TrackObservation],
    cameras: &[ResolvedCamera],
    point: Vec3,
) -> bool {
    observations.iter().all(|observation| {
        let ray = cameras[observation.camera].pixel_to_ray(observation.pixel);
        dot(sub(point, ray.origin), ray.direction) > 0.0
    })
}

fn objective(
    parameters: &[f64],
    specs: &[ParameterSpec],
    inputs: &[RigCameraInput<'_>],
    tracks: &[&Track],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> f64 {
    let refinements = refinements_from_parameters(inputs.len(), parameters, specs);
    let Some(cameras) = resolve_cameras(inputs, &refinements, intrinsics_mode) else {
        return f64::INFINITY;
    };
    let mut cost = 0.0;
    let mut samples = 0;
    for track in tracks {
        let Some(triangulated) = triangulate(&track.observations, &cameras, options) else {
            cost += 25.0;
            samples += track.observations.len();
            continue;
        };
        for observation in &track.observations {
            let ray = cameras[observation.camera].pixel_to_ray(observation.pixel);
            if dot(sub(triangulated.point, ray.origin), ray.direction) <= 0.0 {
                // Do not let the continuous ray-line projection make a
                // behind-camera intersection look like a good physical fit.
                cost += 10.0;
            }
            let Some(projected) = project_observation(
                &cameras[observation.camera],
                observation,
                triangulated.point,
            ) else {
                cost += 25.0;
                samples += 1;
                continue;
            };
            let residual = [
                projected[0] - observation.pixel[0],
                projected[1] - observation.pixel[1],
            ];
            let normalized = normalized_observation_residual(observation, residual);
            cost += observation_balance(observation, track.observations.len())
                * huber(normalized, options.huber_delta);
            samples += 1;
        }
    }
    let data = cost / samples.max(1) as f64;
    let prior = parameters
        .iter()
        .zip(specs)
        .map(|(&value, spec)| (value / spec.prior_sigma).powi(2))
        .sum::<f64>();
    data + options.factory_prior_weight * prior
}

fn evaluate(
    tracks: &[&Track],
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
    retain_residuals: bool,
) -> Evaluation {
    let mut evaluation = Evaluation::default();
    for track in tracks {
        let Some(triangulated) = triangulate(&track.observations, cameras, options) else {
            continue;
        };
        evaluation.tracks += 1;
        if track.observations.iter().all(|observation| {
            let ray = cameras[observation.camera].pixel_to_ray(observation.pixel);
            dot(sub(triangulated.point, ray.origin), ray.direction) > 0.0
        }) {
            evaluation.positive_depth_tracks += 1;
        }
        for observation in &track.observations {
            let Some(projected) = project_observation(
                &cameras[observation.camera],
                observation,
                triangulated.point,
            ) else {
                continue;
            };
            let residual = [
                projected[0] - observation.pixel[0],
                projected[1] - observation.pixel[1],
            ];
            evaluation.sum_squared += dot2(residual, residual);
            evaluation.samples += 1;
            if retain_residuals {
                let sensor_pixels = dot2(residual, residual).sqrt();
                let observed_direction = cameras[observation.camera]
                    .pixel_to_ray(observation.pixel)
                    .direction;
                let projected_direction = cameras[observation.camera]
                    .pixel_to_ray(projected)
                    .direction;
                evaluation.residuals.push(ResidualSample {
                    camera: observation.camera,
                    pixel: observation.pixel,
                    residual,
                    reference_equivalent_pixels: sensor_pixels
                        / observation.local_scale.clamp(0.25, 4.0),
                    angular_degrees: dot(observed_direction, projected_direction)
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees(),
                });
            }
        }
    }
    evaluation
}

#[derive(Clone, Copy)]
struct Triangulated {
    point: Vec3,
    condition: f64,
    max_ray_angle_degrees: f64,
}

fn triangulate(
    observations: &[TrackObservation],
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Option<Triangulated> {
    if observations.len() < 2 {
        return None;
    }
    let rays = observations
        .iter()
        .map(|observation| cameras[observation.camera].pixel_to_ray(observation.pixel))
        .collect::<Vec<_>>();
    let mut max_sine = 0.0f64;
    for first in 0..rays.len() {
        for second in first + 1..rays.len() {
            max_sine = max_sine.max(math::norm(math::cross(
                rays[first].direction,
                rays[second].direction,
            )));
        }
    }
    let max_ray_angle_degrees = max_sine.clamp(0.0, 1.0).asin().to_degrees();
    if max_ray_angle_degrees < options.min_ray_angle_degrees {
        return None;
    }
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for (observation, ray) in observations.iter().zip(&rays) {
        let sigma = observation_sigma(observation);
        let weight = observation_balance(observation, observations.len()) / (sigma * sigma);
        let projector: Mat3 = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                let identity = f64::from(row == column);
                identity - ray.direction[row] * ray.direction[column]
            })
        });
        for row in 0..3 {
            rhs[row] += weight * dot(projector[row], ray.origin);
            for column in 0..3 {
                normal[row][column] += weight * projector[row][column];
            }
        }
    }
    let mut eigenvalues = symmetric_eigenvalues(normal);
    eigenvalues.sort_by(f64::total_cmp);
    if eigenvalues[0] <= 1.0e-14 {
        return None;
    }
    let condition = eigenvalues[2] / eigenvalues[0];
    if !condition.is_finite() || condition > options.max_triangulation_condition {
        return None;
    }
    let point = mul_vec(&math::inverse(&normal)?, rhs);
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(Triangulated {
        point,
        condition,
        max_ray_angle_degrees,
    })
}

/// Reproject the triangulated *line* intersection along the observed ray's
/// forward half-line. Factory angular errors can put the least-squares line
/// intersection behind a camera before refinement; flipping that camera's
/// line direction supplies a continuous calibration residual without treating
/// the non-physical point as valid scene depth.
fn project_observation(
    camera: &ResolvedCamera,
    observation: &TrackObservation,
    point: Vec3,
) -> Option<Vec2> {
    let ray = camera.pixel_to_ray(observation.pixel);
    let displacement = sub(point, ray.origin);
    let forward_point = if dot(displacement, ray.direction) >= 0.0 {
        point
    } else {
        sub(ray.origin, displacement)
    };
    camera.project_unbounded(forward_point)
}

fn track_rms(observations: &[TrackObservation], cameras: &[ResolvedCamera], point: Vec3) -> f64 {
    let mut sum = 0.0;
    let mut samples = 0;
    for observation in observations {
        let Some(projected) = project_observation(&cameras[observation.camera], observation, point)
        else {
            return f64::INFINITY;
        };
        sum += (projected[0] - observation.pixel[0]).powi(2)
            + (projected[1] - observation.pixel[1]).powi(2);
        samples += 1;
    }
    (sum / samples.max(1) as f64).sqrt()
}

fn observation_sigma(observation: &TrackObservation) -> f64 {
    let score = observation.confidence.clamp(0.0, 1.0);
    let structure_support = (observation.structure / 0.08).clamp(0.25, 1.0);
    let depth_support = observation
        .depth_reliability
        .unwrap_or(1.0)
        .clamp(0.25, 1.0);
    let scale = observation.local_scale.clamp(0.5, 3.0);
    ((0.35 + 1.65 * (1.0 - score)) / (structure_support * depth_support).sqrt() * scale.sqrt())
        .clamp(0.30, 3.0)
}

fn normalized_observation_residual(observation: &TrackObservation, residual: Vec2) -> f64 {
    let covariance = observation.localization_covariance;
    let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
    let squared = if determinant.is_finite() && determinant > 1.0e-9 {
        let inverse = [
            [
                covariance[1][1] / determinant,
                -covariance[0][1] / determinant,
            ],
            [
                -covariance[1][0] / determinant,
                covariance[0][0] / determinant,
            ],
        ];
        residual[0] * (inverse[0][0] * residual[0] + inverse[0][1] * residual[1])
            + residual[1] * (inverse[1][0] * residual[0] + inverse[1][1] * residual[1])
    } else {
        dot2(residual, residual)
    };
    squared.max(0.0).sqrt() / observation_sigma(observation).max(1.0e-6)
}

fn observation_balance(observation: &TrackObservation, track_size: usize) -> f64 {
    if observation.fixed_gauge {
        track_size.saturating_sub(1).max(1) as f64
    } else {
        1.0
    }
}

fn huber(value: f64, delta: f64) -> f64 {
    if value <= delta {
        0.5 * value * value
    } else {
        delta * (value - 0.5 * delta)
    }
}

fn symmetric_eigenvalues(mut matrix: Mat3) -> [f64; 3] {
    for _ in 0..16 {
        let pairs = [(0, 1), (0, 2), (1, 2)];
        let &(p, q) = pairs
            .iter()
            .max_by(|&&(ap, aq), &&(bp, bq)| matrix[ap][aq].abs().total_cmp(&matrix[bp][bq].abs()))
            .expect("three off-diagonal pairs");
        if matrix[p][q].abs() < 1.0e-14 {
            break;
        }
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = angle.sin_cos();
        let rotation = match (p, q) {
            (0, 1) => [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            (0, 2) => [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
            (1, 2) => [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            _ => unreachable!(),
        };
        matrix = math::mul(&math::transpose(&rotation), &math::mul(&matrix, &rotation));
    }
    [matrix[0][0], matrix[1][1], matrix[2][2]]
}

fn per_camera_reports(
    cameras: &[RigCameraInput<'_>],
    fit_before: &Evaluation,
    fit_after: &Evaluation,
    validation_before: &Evaluation,
    validation_after: &Evaluation,
) -> Vec<RigCameraResidualReport> {
    cameras
        .iter()
        .enumerate()
        .map(|(camera, input)| {
            let (fit_samples, fit_before_sum) = camera_sum(&fit_before.residuals, camera);
            let (_, fit_after_sum) = camera_sum(&fit_after.residuals, camera);
            let (validation_samples, validation_before_sum) =
                camera_sum(&validation_before.residuals, camera);
            let (_, validation_after_sum) = camera_sum(&validation_after.residuals, camera);
            RigCameraResidualReport {
                camera: input.name.to_owned(),
                fit_samples,
                fit_rms_before: rms(fit_before_sum, fit_samples),
                fit_rms_after: rms(fit_after_sum, fit_samples),
                validation_samples,
                validation_rms_before: rms(validation_before_sum, validation_samples),
                validation_rms_after: rms(validation_after_sum, validation_samples),
            }
        })
        .collect()
}

fn camera_sum(residuals: &[ResidualSample], camera: usize) -> (usize, f64) {
    residuals
        .iter()
        .filter(|sample| sample.camera == camera)
        .fold((0, 0.0), |(count, sum), sample| {
            (count + 1, sum + dot2(sample.residual, sample.residual))
        })
}

fn rms(sum: f64, samples: usize) -> f64 {
    if samples == 0 {
        f64::NAN
    } else {
        (sum / samples as f64).sqrt()
    }
}

fn residual_field_reports(
    cameras: &[RigCameraInput<'_>],
    before: &[ResidualSample],
    after: &[ResidualSample],
) -> Vec<RigResidualFieldReport> {
    const COLUMNS: usize = 4;
    const ROWS: usize = 3;
    let mut reports = Vec::new();
    for (camera, input) in cameras.iter().enumerate() {
        let Some(state) = input.state else { continue };
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let matching = before.iter().enumerate().filter(|(_, sample)| {
                    sample.camera == camera
                        && ((sample.pixel[0] / state.width.max(1) as f64 * COLUMNS as f64) as usize)
                            .min(COLUMNS - 1)
                            == column
                        && ((sample.pixel[1] / state.height.max(1) as f64 * ROWS as f64) as usize)
                            .min(ROWS - 1)
                            == row
                });
                let mut samples = 0;
                let mut pixel = [0.0; 2];
                let mut before_sum = [0.0; 2];
                let mut after_sum = [0.0; 2];
                for (index, sample) in matching {
                    let Some(after_sample) = after.get(index).filter(|after_sample| {
                        after_sample.camera == sample.camera && after_sample.pixel == sample.pixel
                    }) else {
                        continue;
                    };
                    samples += 1;
                    for axis in 0..2 {
                        pixel[axis] += sample.pixel[axis];
                        before_sum[axis] += sample.residual[axis];
                        after_sum[axis] += after_sample.residual[axis];
                    }
                }
                if samples > 0 {
                    let denominator = samples as f64;
                    reports.push(RigResidualFieldReport {
                        camera: input.name.to_owned(),
                        cell: [column, row],
                        samples,
                        mean_pixel: pixel.map(|value| value / denominator),
                        mean_before: before_sum.map(|value| value / denominator),
                        mean_after: after_sum.map(|value| value / denominator),
                    });
                }
            }
        }
    }
    reports
}

fn held_out_observation_reports(
    cameras: &[RigCameraInput<'_>],
    before: &[ResidualSample],
    after: &[ResidualSample],
    factory_p95: f64,
    candidate_p95: f64,
) -> Vec<RigHeldOutObservationReport> {
    let mut used_after = vec![false; after.len()];
    before
        .iter()
        .filter_map(|factory| {
            let (after_index, candidate) =
                after.iter().enumerate().find(|(index, candidate)| {
                    !used_after[*index]
                        && candidate.camera == factory.camera
                        && candidate.pixel == factory.pixel
                })?;
            used_after[after_index] = true;
            let input = cameras.get(factory.camera)?;
            let state = input.state?;
            let factory_sensor_pixels = dot2(factory.residual, factory.residual).sqrt();
            let candidate_sensor_pixels = dot2(candidate.residual, candidate.residual).sqrt();
            Some(RigHeldOutObservationReport {
                camera: input.name.to_owned(),
                sensor_size: [state.width, state.height],
                pixel: factory.pixel,
                factory_residual: factory.residual,
                candidate_residual: candidate.residual,
                factory_sensor_pixels,
                candidate_sensor_pixels,
                factory_reference_pixels: factory.reference_equivalent_pixels,
                candidate_reference_pixels: candidate.reference_equivalent_pixels,
                factory_angular_degrees: factory.angular_degrees,
                candidate_angular_degrees: candidate.angular_degrees,
                factory_p95_tail: factory.reference_equivalent_pixels >= factory_p95,
                candidate_p95_tail: candidate.reference_equivalent_pixels >= candidate_p95,
            })
        })
        .collect()
}

fn correction_reports(
    cameras: &[RigCameraInput<'_>],
    refinements: &[CameraRefinement],
    parameters: &[f64],
    specs: &[ParameterSpec],
) -> Vec<RigCameraCorrectionReport> {
    cameras
        .iter()
        .zip(refinements)
        .enumerate()
        .map(|(camera, (input, refinement))| RigCameraCorrectionReport {
            camera: input.name.to_owned(),
            optimized: specs.iter().any(|spec| spec.camera == camera),
            orientation_offset_degrees: refinement.orientation_offset_degrees.unwrap_or([0.0; 3]),
            mirror_angle_offset_degrees: refinement.mirror_angle_offset_degrees,
            center_offset_world: refinement.center_offset_world.unwrap_or([0.0; 3]),
            sensor_offset_px: refinement.sensor_offset_px.unwrap_or([0.0; 2]),
            reached_bound: parameters
                .iter()
                .zip(specs)
                .any(|(&value, spec)| spec.camera == camera && value.abs() >= spec.bound * 0.98),
        })
        .collect()
}

fn stable_track_hash(key: [i32; 2]) -> u64 {
    let mut value = (key[0] as u32 as u64) << 32 | key[1] as u32 as u64;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn is_validation_track(
    track: &Track,
    validation_modulus: u64,
    options: &RigRefinementOptions,
) -> bool {
    // Track keys use sixteenth-pixel reference coordinates. Hash the spatial
    // block rather than the individual track so correlated samples from the
    // same edge/facade cannot appear in both fit and validation populations.
    let block_units = options.validation_block_size_px.max(1).saturating_mul(16) as i32;
    let block = [
        track.key[0].div_euclid(block_units),
        track.key[1].div_euclid(block_units),
    ];
    stable_track_hash(block).is_multiple_of(validation_modulus.max(2))
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        f64::NAN
    } else {
        sorted[((sorted.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize]
    }
}

fn residual_percentiles(mut values: Vec<f64>) -> RigResidualPercentiles {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    RigResidualPercentiles {
        median: percentile(&values, 0.50),
        p75: percentile(&values, 0.75),
        p90: percentile(&values, 0.90),
        p95: percentile(&values, 0.95),
    }
}

fn residual_distribution(samples: &[ResidualSample]) -> RigResidualDistributionReport {
    RigResidualDistributionReport {
        samples: samples.len(),
        sensor_pixels: residual_percentiles(
            samples
                .iter()
                .map(|sample| dot2(sample.residual, sample.residual).sqrt())
                .collect(),
        ),
        reference_equivalent_pixels: residual_percentiles(
            samples
                .iter()
                .map(|sample| sample.reference_equivalent_pixels)
                .collect(),
        ),
        angular_degrees: residual_percentiles(
            samples
                .iter()
                .map(|sample| sample.angular_degrees)
                .collect(),
        ),
    }
}

fn dot2(first: Vec2, second: Vec2) -> f64 {
    first[0] * second[0] + first[1] * second[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        align::{AlignmentCorrespondence, AlignmentReport, Warp},
        calibration::{CanonicalPose, IntrinsicsBundle, PolynomialDistortion},
    };

    fn calibration(name: &str, centre: Vec3) -> CameraCalibration {
        CameraCalibration {
            name: name.to_owned(),
            intrinsics: vec![IntrinsicsBundle {
                hall_code: Some(0.0),
                focus_distance: 10_000.0,
                k: [[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]],
            }],
            canonical_pose: Some(CanonicalPose {
                rotation_wc: math::IDENTITY,
                translation_wc: math::scale(centre, -1.0),
            }),
            ..Default::default()
        }
    }

    fn state(name: &str) -> ModuleState {
        ModuleState {
            name: name.to_owned(),
            lens_hall: 0.0,
            mirror_hall: 0.0,
            width: 1000,
            height: 800,
            gain: 1.0,
            exposure_ns: 1,
            focus: Default::default(),
        }
    }

    fn synthetic_alignments(truth: &[ResolvedCamera]) -> Vec<ModuleAlignment> {
        let mut alignments = (0..truth.len())
            .map(|camera| ModuleAlignment {
                name: format!("B{}", camera + 1),
                warp: Warp::from_fn(1000, 800, 32, Some),
                correspondences: Vec::new(),
                gain: 1.0,
                offset: 0.0,
                report: AlignmentReport {
                    accepted: true,
                    ..Default::default()
                },
            })
            .collect::<Vec<_>>();
        for row in 0..10 {
            for column in 0..12 {
                let point = [
                    (column as f64 - 5.5) * 70.0,
                    (row as f64 - 4.5) * 60.0,
                    2_000.0 + ((row * 17 + column * 31) % 11) as f64 * 350.0,
                ];
                let reference_pixel = truth[0].project(point).unwrap();
                for camera in 1..truth.len() {
                    alignments[camera]
                        .correspondences
                        .push(AlignmentCorrespondence {
                            reference_pixel,
                            target_pixel: truth[camera].project(point).unwrap(),
                            confidence: 0.95,
                            local_scale: 1.0,
                            structure: 0.1,
                            depth_reliability: None,
                        });
                }
            }
        }
        alignments
    }

    #[test]
    fn physical_refinement_uses_held_out_tracks_and_recovers_small_rotations() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
        ];
        let states = [state("B1"), state("B2"), state("B3")];
        let truth_refinements = [
            CameraRefinement::default(),
            CameraRefinement {
                orientation_offset_degrees: Some([0.12, -0.16, 0.04]),
                ..Default::default()
            },
            CameraRefinement {
                orientation_offset_degrees: Some([-0.10, 0.08, -0.03]),
                ..Default::default()
            },
        ];
        let truth = calibrations
            .iter()
            .zip(&states)
            .zip(&truth_refinements)
            .map(|((calibration, state), refinement)| {
                ResolvedCamera::new(calibration, state, IntrinsicsMode::Clamp, refinement).unwrap()
            })
            .collect::<Vec<_>>();
        let alignments = synthetic_alignments(&truth);
        let inputs = calibrations
            .iter()
            .zip(&states)
            .map(|(calibration, state)| RigCameraInput {
                name: &calibration.name,
                calibration: Some(calibration),
                state: Some(state),
                match_evidence_enabled: true,
                luminance: None,
            })
            .collect::<Vec<_>>();
        let options = RigRefinementOptions {
            min_tracks: 40,
            min_validation_tracks: 10,
            min_camera_observations: 24,
            max_iterations: 8,
            min_validation_improvement: 0.001,
            ..Default::default()
        };
        let result = refine_capture_rig(&inputs, 0, &alignments, IntrinsicsMode::Clamp, &options);
        assert!(result.report.accepted, "{:#?}", result.report);
        assert!(result.report.validation_evaluated);
        assert!(result.report.held_out_rms_after < result.report.held_out_rms_before * 0.5);
        assert_eq!(result.refinements[0].orientation_offset_degrees, None);
        for (camera, expected_refinement) in truth_refinements.iter().enumerate().skip(1) {
            let fitted = result.refinements[camera]
                .orientation_offset_degrees
                .expect("non-reference orientation");
            let expected = expected_refinement.orientation_offset_degrees.unwrap();
            for axis in 0..3 {
                assert!(
                    (fitted[axis] - expected[axis]).abs() < 0.08,
                    "camera {camera} axis {axis}: fitted {fitted:?}, expected {expected:?}"
                );
            }
        }

        let production_options = RigRefinementOptions {
            held_out_validation: false,
            ..options
        };
        let production = refine_capture_rig(
            &inputs,
            0,
            &alignments,
            IntrinsicsMode::Clamp,
            &production_options,
        );
        assert!(production.report.accepted, "{:#?}", production.report);
        assert!(!production.report.validation_evaluated);
        assert_eq!(production.report.validation_tracks, 0);
        assert!(production.report.fit_tracks > result.report.fit_tracks);
    }

    #[test]
    fn physical_refinement_uses_center_offsets_when_bearings_are_fixed() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
        ];
        let states = [state("B1"), state("B2"), state("B3")];
        let truth_refinements = [
            CameraRefinement::default(),
            CameraRefinement {
                center_offset_world: Some([0.0, -2.0, 1.0]),
                ..Default::default()
            },
            CameraRefinement {
                center_offset_world: Some([2.0, 0.0, -1.0]),
                ..Default::default()
            },
        ];
        let truth = calibrations
            .iter()
            .zip(&states)
            .zip(&truth_refinements)
            .map(|((calibration, state), refinement)| {
                ResolvedCamera::new(calibration, state, IntrinsicsMode::Clamp, refinement).unwrap()
            })
            .collect::<Vec<_>>();
        let alignments = synthetic_alignments(&truth);
        let inputs = calibrations
            .iter()
            .zip(&states)
            .map(|(calibration, state)| RigCameraInput {
                name: &calibration.name,
                calibration: Some(calibration),
                state: Some(state),
                match_evidence_enabled: true,
                luminance: None,
            })
            .collect::<Vec<_>>();
        let options = RigRefinementOptions {
            min_tracks: 40,
            min_validation_tracks: 10,
            min_camera_observations: 24,
            max_iterations: 12,
            max_orientation_degrees: 0.0,
            max_sensor_offset_px: 0.0,
            factory_prior_weight: 1.0e-5,
            min_validation_improvement: 0.001,
            ..Default::default()
        };
        let result = refine_capture_rig(&inputs, 0, &alignments, IntrinsicsMode::Clamp, &options);
        assert!(result.report.accepted, "{:#?}", result.report);
        assert!(result.report.held_out_rms_after < result.report.held_out_rms_before * 0.1);
        assert_eq!(result.refinements[0].center_offset_world, None);
        for camera in 1..3 {
            let fitted = result.refinements[camera]
                .center_offset_world
                .expect("non-reference center offset");
            assert!(norm(fitted) > 0.25, "camera {camera}: {fitted:?}");
        }
    }

    #[test]
    fn physical_refinement_uses_sensor_offsets_when_pose_is_fixed() {
        let mut calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
        ];
        for calibration in &mut calibrations {
            calibration.distortion = Some(PolynomialDistortion {
                center: [500.0, 400.0],
                normalization: [800.0, 800.0],
                coeffs: vec![0.12, -0.04, 0.002, -0.001, 0.01],
            });
        }
        let states = [state("B1"), state("B2"), state("B3")];
        let truth_refinements = [
            CameraRefinement::default(),
            CameraRefinement {
                sensor_offset_px: Some([8.0, -5.0]),
                ..Default::default()
            },
            CameraRefinement {
                sensor_offset_px: Some([-6.0, 7.0]),
                ..Default::default()
            },
        ];
        let truth = calibrations
            .iter()
            .zip(&states)
            .zip(&truth_refinements)
            .map(|((calibration, state), refinement)| {
                ResolvedCamera::new(calibration, state, IntrinsicsMode::Clamp, refinement).unwrap()
            })
            .collect::<Vec<_>>();
        let alignments = synthetic_alignments(&truth);
        let inputs = calibrations
            .iter()
            .zip(&states)
            .map(|(calibration, state)| RigCameraInput {
                name: &calibration.name,
                calibration: Some(calibration),
                state: Some(state),
                match_evidence_enabled: true,
                luminance: None,
            })
            .collect::<Vec<_>>();
        let options = RigRefinementOptions {
            min_tracks: 40,
            min_validation_tracks: 10,
            min_camera_observations: 24,
            max_iterations: 12,
            max_orientation_degrees: 0.0,
            max_center_offset: 0.0,
            factory_prior_weight: 1.0e-5,
            min_validation_improvement: 0.001,
            ..Default::default()
        };
        let result = refine_capture_rig(&inputs, 0, &alignments, IntrinsicsMode::Clamp, &options);
        assert!(result.report.accepted, "{:#?}", result.report);
        assert!(result.report.held_out_rms_after < result.report.held_out_rms_before * 0.1);
        assert_eq!(result.refinements[0].sensor_offset_px, None);
        for camera in 1..3 {
            let fitted = result.refinements[camera]
                .sensor_offset_px
                .expect("non-reference sensor offset");
            assert!(
                norm([fitted[0], fitted[1], 0.0]) > 1.0,
                "camera {camera}: fitted {fitted:?}; report {:#?}",
                result.report,
            );
        }
    }

    #[test]
    fn downstream_alignment_comparison_is_diagnostic_only() {
        let alignment = |name: &str, correction: [f32; 2]| ModuleAlignment {
            name: name.to_owned(),
            warp: Warp::from_fn(8, 8, 4, Some),
            correspondences: Vec::new(),
            gain: 1.0,
            offset: 0.0,
            report: AlignmentReport {
                camera: name.to_owned(),
                correction_median_px: correction,
                inliers: 20,
                residual_median_px: 0.5,
                accepted: true,
                ..Default::default()
            },
        };
        let mut report = RigRefinementReport {
            accepted: true,
            corrections: vec![RigCameraCorrectionReport {
                camera: "B2".to_owned(),
                optimized: true,
                orientation_offset_degrees: [0.1, 0.0, 0.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(evaluate_image_space_alignment(
            &mut report,
            &[alignment("B2", [10.0, 0.0])],
            &[alignment("B2", [4.0, 0.0])],
            0.05,
        ));
        assert_eq!(report.image_space_evaluated_cameras, 1);
        assert!((report.image_space_relative_improvement - 0.6).abs() < 1.0e-9);

        assert!(!evaluate_image_space_alignment(
            &mut report,
            &[alignment("B2", [4.0, 0.0])],
            &[alignment("B2", [5.0, 0.0])],
            0.05,
        ));
        assert!(report.accepted);
        assert_eq!(report.image_space_validation_passed, Some(false));
        assert!(report.image_space_warning.is_some());
        assert!(report.fallback_reason.is_none());
    }

    #[test]
    fn symmetric_eigenvalues_match_a_diagonal_matrix() {
        let mut values = symmetric_eigenvalues([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
        values.sort_by(f64::total_cmp);
        assert_eq!(values, [1.0, 2.0, 4.0]);
    }

    #[test]
    fn physical_track_ranking_rewards_broad_multiview_support() {
        let options = RigRefinementOptions {
            physical_match_min_score: 0.50,
            physical_match_min_views: 3,
            ..Default::default()
        };
        let view = |camera: usize, score: f32| PhysicalViewMatch {
            camera,
            score,
            target_pixel: [100.0 + camera as f64, 80.0],
            residual: [0.0, 0.0],
            residual_proposal: [0.0, 0.0],
            local_scale: 1.0,
        };
        let accidental = physical_depth_candidate(
            vec![view(1, 0.97), view(2, 0.95), view(3, 0.94)],
            10,
            2000.0,
            &options,
        )
        .unwrap();
        let broad = physical_depth_candidate(
            (1..=9).map(|camera| view(camera, 0.80)).collect(),
            20,
            1800.0,
            &options,
        )
        .unwrap();
        assert!(
            broad.ranking > accidental.ranking,
            "broad={} accidental={}",
            broad.ranking,
            accidental.ranking
        );
    }

    #[test]
    fn measured_proposal_discards_displacement_along_the_depth_locus() {
        let proposal = perpendicular_residual_proposal([7.0, 11.0], Some([1.0, 0.0]));
        assert_eq!(proposal, [0.0, 11.0]);
        assert_eq!(
            perpendicular_residual_proposal([7.0, 11.0], None),
            [7.0, 11.0]
        );
    }

    #[test]
    fn physical_match_defaults_to_projected_motion_bounded_hierarchy() {
        let options = RigRefinementOptions::default();
        assert_eq!(options.physical_match_planes, 48);
        assert_eq!(options.physical_match_max_projected_step_px, 8.0);
        assert_eq!(options.physical_match_depth_beam_width, 6);
        assert_eq!(options.physical_match_max_depth_refinements, 6);
    }

    #[test]
    fn luminance_pyramid_coordinates_follow_box_filter_centres() {
        assert_eq!(sensor_to_luminance_coordinate(0.5, 0), 0.0);
        assert_eq!(sensor_to_luminance_coordinate(2.5, 0), 1.0);
        assert_eq!(sensor_to_luminance_coordinate(1.5, 1), 0.0);
        assert_eq!(sensor_to_luminance_coordinate(5.5, 1), 1.0);
        assert_eq!(sensor_to_luminance_coordinate(3.5, 2), 0.0);
        assert_eq!(sensor_to_luminance_coordinate(11.5, 2), 1.0);
    }

    #[test]
    fn depth_hierarchy_subdivides_until_projected_motion_is_bounded() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [200.0, 0.0, 0.0]),
        ];
        let states = [state("B1"), state("B2")];
        let luminance = [Plane::new(500, 400), Plane::new(500, 400)];
        let inputs = calibrations
            .iter()
            .zip(&states)
            .zip(&luminance)
            .map(|((calibration, state), luminance)| RigCameraInput {
                name: &calibration.name,
                calibration: Some(calibration),
                state: Some(state),
                match_evidence_enabled: true,
                luminance: Some(luminance),
            })
            .collect::<Vec<_>>();
        let resolved = calibrations
            .iter()
            .zip(&states)
            .map(|(calibration, state)| {
                ResolvedCamera::new(
                    calibration,
                    state,
                    IntrinsicsMode::Clamp,
                    &CameraRefinement::default(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let options = RigRefinementOptions::default();
        let centre = [500.0, 400.0];
        let min_inverse = 1.0 / options.physical_match_far_depth;
        let max_inverse = 1.0 / options.physical_match_near_depth;
        let coarse_intervals = options.physical_match_planes - 1;
        let coarse_step = maximum_projected_depth_step(
            &inputs,
            0,
            &resolved,
            centre,
            coarse_intervals,
            min_inverse,
            max_inverse,
        );
        let (levels, selected_step) = physical_depth_refinement_levels(
            &inputs,
            0,
            &resolved,
            centre,
            options.physical_match_max_depth_refinements,
            &options,
        );
        let final_step = maximum_projected_depth_step(
            &inputs,
            0,
            &resolved,
            centre,
            coarse_intervals * (1usize << levels),
            min_inverse,
            max_inverse,
        );
        assert!(coarse_step > options.physical_match_max_projected_step_px);
        assert!(levels > 0);
        assert!((selected_step - final_step).abs() < 1.0e-9);
        assert!(
            final_step <= options.physical_match_max_projected_step_px,
            "coarse={coarse_step}, levels={levels}, final={final_step}"
        );
    }

    #[test]
    fn physical_track_pruning_removes_one_incompatible_camera_and_retriangulates() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
            calibration("B4", [-75.0, 0.0, 0.0]),
        ];
        let states = [state("B1"), state("B2"), state("B3"), state("B4")];
        let cameras = calibrations
            .iter()
            .zip(&states)
            .map(|(calibration, state)| {
                ResolvedCamera::new(
                    calibration,
                    state,
                    IntrinsicsMode::Clamp,
                    &CameraRefinement::default(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let point = [120.0, -40.0, 4200.0];
        let mut observations = cameras
            .iter()
            .enumerate()
            .map(|(camera, resolved)| TrackObservation {
                camera,
                pixel: resolved.project(point).unwrap(),
                bootstrap_residual_proposal: [0.0, 0.0],
                localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                fixed_gauge: camera == 0,
                confidence: 0.95,
                local_scale: 1.0,
                structure: 0.1,
                depth_reliability: Some(1.0),
            })
            .collect::<Vec<_>>();
        observations[3].pixel[0] += 28.0;
        observations[3].pixel[1] -= 17.0;
        let track = Track {
            key: [1000, 2000],
            observations,
            condition: f64::NAN,
            max_ray_angle_degrees: f64::NAN,
        };
        let options = RigRefinementOptions {
            physical_match_min_views: 3,
            physical_match_max_reprojection_reference_px: 2.0,
            ..Default::default()
        };
        let (pruned, removed) = prune_physical_track(track, &cameras, &options).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(pruned.observations.len(), 3);
        assert!(
            pruned
                .observations
                .iter()
                .all(|observation| observation.camera != 3)
        );
        assert!(
            track_rms(
                &pruned.observations,
                &cameras,
                triangulate(&pruned.observations, &cameras, &options)
                    .unwrap()
                    .point,
            ) < 1.0e-6
        );
    }

    #[test]
    fn physical_track_bootstrap_validates_after_removing_measured_proposals() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
        ];
        let states = [state("B1"), state("B2"), state("B3")];
        let cameras = calibrations
            .iter()
            .zip(&states)
            .map(|(calibration, state)| {
                ResolvedCamera::new(
                    calibration,
                    state,
                    IntrinsicsMode::Clamp,
                    &CameraRefinement::default(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let point = [120.0, -40.0, 4200.0];
        let observations = cameras
            .iter()
            .enumerate()
            .map(|(camera, resolved)| {
                let proposal = if camera == 0 {
                    [0.0, 0.0]
                } else {
                    [40.0 + camera as f64, -25.0]
                };
                let mut pixel = resolved.project(point).unwrap();
                pixel[0] += proposal[0];
                pixel[1] += proposal[1];
                TrackObservation {
                    camera,
                    pixel,
                    bootstrap_residual_proposal: proposal,
                    localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    fixed_gauge: camera == 0,
                    confidence: 0.95,
                    local_scale: 1.0,
                    structure: 0.1,
                    depth_reliability: Some(1.0),
                }
            })
            .collect();
        let track = Track {
            key: [1000, 2000],
            observations,
            condition: f64::NAN,
            max_ray_angle_degrees: f64::NAN,
        };
        let options = RigRefinementOptions {
            physical_match_min_views: 3,
            physical_match_max_reprojection_reference_px: 2.0,
            ..Default::default()
        };
        let (retained, removed) = prune_physical_track(track, &cameras, &options).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(retained.observations.len(), 3);
        assert_eq!(
            retained.observations[1].bootstrap_residual_proposal,
            [41.0, -25.0]
        );
    }

    #[test]
    fn validation_split_keeps_nearby_tracks_in_the_same_spatial_block() {
        let options = RigRefinementOptions {
            validation_block_size_px: 256,
            ..Default::default()
        };
        let track = |pixel: [i32; 2]| Track {
            key: pixel.map(|coordinate| coordinate * 16),
            observations: Vec::new(),
            condition: 1.0,
            max_ray_angle_degrees: 1.0,
        };
        assert_eq!(
            is_validation_track(&track([300, 500]), 5, &options),
            is_validation_track(&track([490, 510]), 5, &options),
        );
    }

    #[test]
    fn anisotropic_localization_covariance_weights_the_known_direction_more_strongly() {
        let observation = TrackObservation {
            camera: 0,
            pixel: [100.0, 80.0],
            bootstrap_residual_proposal: [0.0, 0.0],
            // Four times the variance along x and one quarter along y keeps
            // unit determinant while describing an edge/locus ambiguity.
            localization_covariance: [[4.0, 0.0], [0.0, 0.25]],
            fixed_gauge: false,
            confidence: 0.9,
            local_scale: 1.0,
            structure: 0.1,
            depth_reliability: Some(1.0),
        };
        let along_uncertain = normalized_observation_residual(&observation, [2.0, 0.0]);
        let across_uncertain = normalized_observation_residual(&observation, [0.0, 2.0]);
        assert!(across_uncertain > along_uncertain * 3.9);
    }

    #[test]
    fn pair_cheirality_distinguishes_forward_and_behind_intersections() {
        let first = crate::geometry::Ray {
            origin: [0.0, 0.0, 0.0],
            direction: math::normalize([0.0, 0.0, 10.0]),
        };
        let forward = crate::geometry::Ray {
            origin: [1.0, 0.0, 0.0],
            direction: math::normalize([-1.0, 0.0, 10.0]),
        };
        let behind = crate::geometry::Ray {
            origin: [1.0, 0.0, 0.0],
            direction: math::normalize([1.0, 0.0, -10.0]),
        };
        assert!(pair_has_positive_depth(first, forward));
        assert!(!pair_has_positive_depth(first, behind));
    }
}
