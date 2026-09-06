//! Capture-specific refinement of the physical multi-camera rig.
//!
//! The optimizer consumes the finest reliable patch observations already
//! produced by [`crate::align`]. It joins pairwise reference-camera matches
//! into multi-view tracks, triangulates a world point for each track, and fits
//! only small orientation and movable-mirror state corrections. The reference
//! camera is the fixed gauge. An independently held-out track subset decides
//! whether the candidate physical model is allowed downstream.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    align::ModuleAlignment,
    calibration::{CameraCalibration, IntrinsicsMode, ModuleState},
    geometry::{CameraRefinement, ResolvedCamera},
    math::{self, Mat3, Vec2, Vec3, cross, dot, mul_vec, norm, sub},
};

#[derive(Clone, Debug)]
pub struct RigRefinementOptions {
    /// Run capture-specific physical refinement before the residual image warp.
    pub enabled: bool,
    /// Minimum geometrically valid tracks required for an attempted fit.
    pub min_tracks: usize,
    /// Minimum independently held-out tracks required for acceptance.
    pub min_validation_tracks: usize,
    /// Minimum fit-set observations required before a camera receives free
    /// physical parameters. Sparse cameras remain on their factory model.
    pub min_camera_observations: usize,
    /// Deterministic fraction of tracks withheld from optimization.
    pub validation_fraction: f64,
    /// Bounded coordinate-Newton sweeps.
    pub max_iterations: usize,
    /// Strict world-frame bearing correction bound per axis.
    pub max_orientation_degrees: f64,
    /// Strict additive movable-mirror angle correction bound.
    pub max_mirror_degrees: f64,
    /// Gaussian factory prior scale for each orientation component.
    pub orientation_prior_sigma_degrees: f64,
    /// Gaussian factory prior scale for the movable-mirror angle.
    pub mirror_prior_sigma_degrees: f64,
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
    /// Minimum reduction in the median magnitude of the later residual
    /// image-space correction. This is a second, downstream acceptance gate.
    pub min_image_space_correction_improvement: f64,
}

impl Default for RigRefinementOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            min_tracks: 80,
            min_validation_tracks: 20,
            min_camera_observations: 150,
            validation_fraction: 0.20,
            max_iterations: 6,
            max_orientation_degrees: 0.5,
            max_mirror_degrees: 0.35,
            orientation_prior_sigma_degrees: 0.20,
            mirror_prior_sigma_degrees: 0.08,
            factory_prior_weight: 0.02,
            huber_delta: 2.5,
            max_triangulation_condition: 1.0e9,
            min_ray_angle_degrees: 0.003,
            max_initial_track_rms: 100.0,
            min_validation_improvement: 0.005,
            min_positive_depth_fraction: 0.80,
            min_image_space_correction_improvement: 0.05,
        }
    }
}

pub struct RigCameraInput<'a> {
    pub name: &'a str,
    pub calibration: Option<&'a CameraCalibration>,
    pub state: Option<&'a ModuleState>,
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
    pub reference_camera: String,
    pub pairwise_matches: usize,
    pub tracks: usize,
    pub tracks_three_plus: usize,
    pub fit_tracks: usize,
    pub validation_tracks: usize,
    pub rejected_degenerate_tracks: usize,
    pub rejected_outlier_tracks: usize,
    pub rejected_nonpositive_observations: usize,
    pub optimizer_iterations: usize,
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
    pub median_triangulation_condition: f64,
    pub p90_triangulation_condition: f64,
    pub median_max_ray_angle_degrees: f64,
    pub per_camera: Vec<RigCameraResidualReport>,
    pub residual_field: Vec<RigResidualFieldReport>,
    pub corrections: Vec<RigCameraCorrectionReport>,
    /// Filled by the pipeline after the accepted physical model is used as the
    /// seed for the ordinary residual image-space alignment.
    pub image_space_corrections: Vec<RigImageCorrectionReport>,
    pub image_space_evaluated_cameras: usize,
    pub image_space_median_correction_before_px: f64,
    pub image_space_median_correction_after_px: f64,
    /// Positive means the physical model reduced the amount of later
    /// image-space correction required.
    pub image_space_relative_improvement: f64,
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
pub struct RigResidualFieldReport {
    pub camera: String,
    pub cell: [usize; 2],
    pub samples: usize,
    pub mean_pixel: [f64; 2],
    pub mean_before: [f64; 2],
    pub mean_after: [f64; 2],
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigCameraCorrectionReport {
    pub camera: String,
    /// False for the fixed gauge and cameras without enough fit observations.
    pub optimized: bool,
    pub orientation_offset_degrees: [f64; 3],
    pub mirror_angle_offset_degrees: f64,
    pub reached_bound: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RigImageCorrectionReport {
    pub camera: String,
    pub factory_seed_correction_px: [f32; 2],
    pub refined_seed_correction_px: [f32; 2],
}

/// Populate the downstream residual-alignment comparison and retain an
/// otherwise valid physical candidate only when it reduces the amount of
/// later image-space correction. The physical fit and its diagnostics remain
/// in the report when this second gate rejects it.
pub fn gate_on_image_space_alignment(
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
    let mut lost_alignment = None;
    for (factory, refined) in factory.iter().zip(refined) {
        if !active.contains(&factory.name.as_str()) {
            continue;
        }
        if factory.report.accepted && !refined.report.accepted {
            lost_alignment = Some(factory.name.clone());
        }
        if factory.report.accepted && refined.report.accepted {
            before.push(correction_magnitude(factory.report.correction_median_px));
            after.push(correction_magnitude(refined.report.correction_median_px));
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

    let accepted = lost_alignment.is_none()
        && !before.is_empty()
        && report.image_space_relative_improvement >= minimum_improvement;
    if !accepted {
        report.accepted = false;
        report.fallback_reason = Some(if let Some(camera) = lost_alignment {
            format!("residual alignment for {camera} failed from the refined physical seed")
        } else if before.is_empty() {
            "no accepted residual alignments available for downstream validation".to_owned()
        } else {
            format!(
                "later image-space correction improved by only {:+.2}% (need at least {:+.2}%)",
                report.image_space_relative_improvement * 100.0,
                minimum_improvement * 100.0
            )
        });
    }
    accepted
}

fn correction_magnitude(correction: [f32; 2]) -> f64 {
    (f64::from(correction[0]).powi(2) + f64::from(correction[1]).powi(2)).sqrt()
}

#[derive(Clone, Debug)]
struct TrackObservation {
    camera: usize,
    pixel: Vec2,
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

/// Fit a bounded capture-specific physical model and apply it only when an
/// independent deterministic track split improves.
pub fn refine_capture_rig(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    provisional_alignments: &[ModuleAlignment],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> RigRefinementOutcome {
    let mut report = RigRefinementReport {
        enabled: options.enabled,
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
    let (raw_tracks, pairwise_matches) = build_tracks(
        cameras,
        reference_index,
        provisional_alignments,
        &factory_cameras,
    );
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

    let validation_modulus = (1.0 / options.validation_fraction.clamp(0.05, 0.5)).round() as u64;
    let (preliminary_validation, preliminary_fit): (Vec<_>, Vec<_>) = tracks
        .iter()
        .partition(|track| stable_track_hash(track.key).is_multiple_of(validation_modulus.max(2)));
    if preliminary_fit.len() < options.min_tracks
        || preliminary_validation.len() < options.min_validation_tracks
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

    let specs = parameter_specs(cameras, reference_index, &preliminary_fit, options);
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
    let fit_tracks = prepare(&preliminary_fit);
    let validation_tracks = prepare(&preliminary_validation);
    let removed_after_observability = preliminary_fit.len() + preliminary_validation.len()
        - fit_tracks.len()
        - validation_tracks.len();
    report.rejected_degenerate_tracks += removed_after_observability;
    report.fit_tracks = fit_tracks.len();
    report.validation_tracks = validation_tracks.len();
    if fit_tracks.len() < options.min_tracks
        || validation_tracks.len() < options.min_validation_tracks
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
    let fit = fit_tracks.iter().collect::<Vec<_>>();
    let validation = validation_tracks.iter().collect::<Vec<_>>();
    let factory_parameters = vec![0.0; specs.len()];
    let before_objective = objective(
        &factory_parameters,
        &specs,
        cameras,
        &fit,
        intrinsics_mode,
        options,
    );
    // The factory rig can miss by more than the finite-depth parallax itself.
    // Obtain a fit-only epipolar initialization before alternating
    // triangulation and physical parameter updates. Validation tracks never
    // enter either optimization objective.
    let (angular_candidate, _, initialization_iterations) = coordinate_optimize(
        factory_parameters,
        &specs,
        options.max_iterations.min(4),
        |parameters| {
            epipolar_objective(
                parameters,
                &specs,
                cameras,
                reference_index,
                &fit,
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
        &fit,
        intrinsics_mode,
        options,
    ) < before_objective
    {
        angular_candidate
    } else {
        vec![0.0; specs.len()]
    };
    let (parameters, current_objective, bundle_iterations) =
        coordinate_optimize(initialized, &specs, options.max_iterations, |parameters| {
            objective(parameters, &specs, cameras, &fit, intrinsics_mode, options)
        });
    let iterations = initialization_iterations + bundle_iterations;

    let candidate_refinements = refinements_from_parameters(cameras.len(), &parameters, &specs);
    let Some(candidate_cameras) = resolve_cameras(cameras, &candidate_refinements, intrinsics_mode)
    else {
        report.fallback_reason = Some("candidate physical model could not be resolved".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    };
    let fit_before = evaluate(&fit, &factory_cameras, options, true);
    let fit_after = evaluate(&fit, &candidate_cameras, options, true);
    let validation_before = evaluate(&validation, &factory_cameras, options, true);
    let validation_after = evaluate(&validation, &candidate_cameras, options, true);
    let held_out_improvement =
        (validation_before.rms() - validation_after.rms()) / validation_before.rms().max(1.0e-12);
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
    let validation_improved = held_out_improvement >= options.min_validation_improvement;
    let fit_positive_depth = fit_after.positive_depth_fraction();
    let validation_positive_depth = validation_after.positive_depth_fraction();
    let physical_depth_valid = fit_positive_depth >= options.min_positive_depth_fraction
        && validation_positive_depth >= options.min_positive_depth_fraction;
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
    report.optimizer_iterations = iterations;
    report.training_objective_before = before_objective;
    report.training_objective_after = current_objective;
    report.reprojection_rms_before = fit_before.rms();
    report.reprojection_rms_after = fit_after.rms();
    report.held_out_rms_before = validation_before.rms();
    report.held_out_rms_after = validation_after.rms();
    report.fit_positive_depth_fraction_before = fit_before.positive_depth_fraction();
    report.fit_positive_depth_fraction_after = fit_positive_depth;
    report.held_out_positive_depth_fraction_before = validation_before.positive_depth_fraction();
    report.held_out_positive_depth_fraction_after = validation_positive_depth;
    report.held_out_relative_improvement = held_out_improvement;
    report.median_triangulation_condition = percentile(&conditions, 0.5);
    report.p90_triangulation_condition = percentile(&conditions, 0.9);
    report.median_max_ray_angle_degrees = percentile(&ray_angles, 0.5);
    report.corrections = correction_reports(cameras, &candidate_refinements, &parameters, &specs);
    report.residual_field = residual_field_reports(
        cameras,
        &validation_before.residuals,
        &validation_after.residuals,
    );
    report.per_camera = per_camera;
    report.fallback_reason = (!accepted).then(|| {
        if reached_bound {
            "candidate reached a physical correction bound".to_owned()
        } else if !physical_depth_valid {
            format!(
                "positive-depth support is only {:.1}% fit/{:.1}% held out (need {:.1}%)",
                fit_positive_depth * 100.0,
                validation_positive_depth * 100.0,
                options.min_positive_depth_fraction * 100.0,
            )
        } else if camera_regression {
            "held-out residual regressed for an observed camera".to_owned()
        } else if !training_improved {
            "training reprojection RMS did not improve".to_owned()
        } else {
            format!(
                "held-out improvement {:+.3}% is below required {:+.3}%",
                held_out_improvement * 100.0,
                options.min_validation_improvement * 100.0
            )
        }
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
            || cameras[camera].calibration.is_none()
            || !alignment.report.accepted
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
        if input
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

fn refinements_from_parameters(
    camera_count: usize,
    parameters: &[f64],
    specs: &[ParameterSpec],
) -> Vec<CameraRefinement> {
    let mut orientations = vec![[0.0; 3]; camera_count];
    let mut mirrors = vec![0.0; camera_count];
    for (&value, spec) in parameters.iter().zip(specs) {
        match spec.kind {
            ParameterKind::Orientation(axis) => orientations[spec.camera][axis] = value,
            ParameterKind::Mirror => mirrors[spec.camera] = value,
        }
    }
    orientations
        .into_iter()
        .zip(mirrors)
        .map(|(orientation, mirror)| CameraRefinement {
            mirror_angle_offset_degrees: mirror,
            orientation_offset_degrees: (orientation != [0.0; 3]).then_some(orientation),
        })
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
            let sigma = observation_sigma(observation);
            let normalized = ((projected[0] - observation.pixel[0]).powi(2)
                + (projected[1] - observation.pixel[1]).powi(2))
            .sqrt()
                / sigma;
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
                evaluation.residuals.push(ResidualSample {
                    camera: observation.camera,
                    pixel: observation.pixel,
                    residual,
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

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        f64::NAN
    } else {
        sorted[((sorted.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize]
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
        calibration::{CanonicalPose, IntrinsicsBundle},
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
        let mut alignments = (0..3)
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
                    3500.0 + ((row * 17 + column * 31) % 11) as f64 * 180.0,
                ];
                let reference_pixel = truth[0].project(point).unwrap();
                for camera in 1..3 {
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
        let inputs = calibrations
            .iter()
            .zip(&states)
            .map(|(calibration, state)| RigCameraInput {
                name: &calibration.name,
                calibration: Some(calibration),
                state: Some(state),
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
    }

    #[test]
    fn downstream_gate_requires_less_residual_image_correction() {
        let alignment = |name: &str, correction: [f32; 2]| ModuleAlignment {
            name: name.to_owned(),
            warp: Warp::from_fn(8, 8, 4, Some),
            correspondences: Vec::new(),
            gain: 1.0,
            offset: 0.0,
            report: AlignmentReport {
                camera: name.to_owned(),
                correction_median_px: correction,
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
        assert!(gate_on_image_space_alignment(
            &mut report,
            &[alignment("B2", [10.0, 0.0])],
            &[alignment("B2", [4.0, 0.0])],
            0.05,
        ));
        assert_eq!(report.image_space_evaluated_cameras, 1);
        assert!((report.image_space_relative_improvement - 0.6).abs() < 1.0e-9);

        report.accepted = true;
        report.fallback_reason = None;
        assert!(!gate_on_image_space_alignment(
            &mut report,
            &[alignment("B2", [4.0, 0.0])],
            &[alignment("B2", [5.0, 0.0])],
            0.05,
        ));
        assert!(!report.accepted);
        assert!(report.fallback_reason.is_some());
    }

    #[test]
    fn symmetric_eigenvalues_match_a_diagonal_matrix() {
        let mut values = symmetric_eigenvalues([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
        values.sort_by(f64::total_cmp);
        assert_eq!(values, [1.0, 2.0, 4.0]);
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
