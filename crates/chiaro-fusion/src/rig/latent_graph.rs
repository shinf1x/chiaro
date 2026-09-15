//! Latent multi-hypothesis correspondence strategy.
//!
//! The ordinary sparse rig path commits to one target observation before the
//! physical solve.  That is a poor failure mode on repeated industrial
//! structure: several visually plausible target corners can lie on almost the
//! same epipolar locus, and a locally good but globally wrong winner becomes a
//! fixed measurement.  This strategy instead:
//!
//! 1. merges the independently verified reference->camera observations into
//!    genuine multi-view tracks;
//! 2. finds a small set of self-similar target-side alternatives for every
//!    observation from the other sub-pixel landmarks already localized in the
//!    same camera;
//! 3. keeps those alternatives latent while the ordinary bounded physical rig
//!    solver runs; and
//! 4. builds a bootstrap-only multi-camera cycle graph from independently
//!    image-verified B<->C / C<->C edges, excluding held-out spatial blocks; and
//! 5. between bundle passes, lets an observation switch identity when the
//!    leave-one-camera-out 3-D prediction, local landmark constellation, and
//!    available cross-camera cycle evidence support another candidate.
//!
//! Held-out tracks never call `update_assignments`; their initially measured
//! correspondence membership is therefore frozen exactly like the other rig
//! strategies' validation population.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use super::{
    FALLBACK_ROTATION_MIN_INLIERS, RigCameraInput, RigRefinementOptions, RigRefinementStrategy,
    Track, TrackObservation, anchor_graph::build_anchor_tracks, fallback_epipolar_inliers,
    fallback_rotation_inliers, pair_signed_depths, triangulate,
};
use crate::{
    align::{AlignmentCorrespondence, ModuleAlignment, detect_rig_corners},
    calibration::IntrinsicsMode,
    geometry::ResolvedCamera,
    image::Plane,
    math::{self, Vec2, Vec3, cross, dot, norm, scale, sub},
};

const DESCRIPTOR_SIDE: usize = 5;
const DESCRIPTOR_SAMPLES: usize = DESCRIPTOR_SIDE * DESCRIPTOR_SIDE;
const DESCRIPTOR_STEP_LUMA: f32 = 1.5;

#[derive(Clone, Debug, Default, Serialize)]
pub struct LatentMatchRoundReport {
    pub iteration: usize,
    pub switches: usize,
    pub geometry_supported_switches: usize,
    pub constellation_supported_switches: usize,
    pub cycle_supported_switches: usize,
    /// Proposed switches rejected because the bootstrap multi-camera graph
    /// strongly disagreed and leave-one-camera-out 3-D geometry did not rescue
    /// the candidate.
    pub cycle_rejected_switches: usize,
    /// Candidate switches rejected because another track already owns the
    /// same target-side landmark in this camera.
    pub collision_rejected_switches: usize,
    pub evaluated_observations: usize,
    pub mean_score_before: f64,
    pub mean_score_after: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LatentCameraSupportReport {
    pub camera: String,
    /// Response-sorted reference-camera corners offered to this pairwise match.
    pub detector_candidates: usize,
    /// Image-space matches retained by the pairwise matcher before the
    /// calibrated epipolar filter.
    pub image_matches: usize,
    /// Pairwise matches surviving image matching and epipolar filtering.
    pub seed_pairwise_matches: usize,
    /// Localized target-side landmarks available as latent alternatives.
    pub candidate_pool_landmarks: usize,
    /// Active fit observations before reversible membership starts.
    pub initial_fit_observations: usize,
    /// Active fit observations after the last reversible membership pass.
    pub final_fit_observations: usize,
    /// Fit observations belonging to intrinsically two-view tracks before and
    /// after pairwise whole-track membership. These tracks use calibrated
    /// epipolar consistency because leave-one-camera-out triangulation is
    /// mathematically unavailable.
    pub initial_pairwise_fit_observations: usize,
    pub final_pairwise_fit_observations: usize,
    /// Minimum active support protected for this camera.
    pub membership_floor: usize,
    /// Cumulative demotions to soft/dormant membership.
    pub soft_outlier_events: usize,
    /// Cumulative ordinary reactivations after geometry improved.
    pub reactivated_events: usize,
    /// Cumulative reactivations made specifically to restore camera support.
    pub floor_reactivated_events: usize,
    /// Demotion attempts skipped because the camera was already at its floor.
    pub floor_protected_events: usize,
    /// Whole intrinsically-two-view tracks made dormant/reactivated by the
    /// calibrated pairwise epipolar membership test.
    pub pairwise_track_demotions: usize,
    pub pairwise_track_reactivations: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LatentMatchReport {
    /// Pairwise image observations surviving the same independent epipolar
    /// filter used by the ordinary physical strategy.
    pub seed_pairwise_matches: usize,
    /// Multi-view tracks after reference-landmark merging.
    pub initial_tracks: usize,
    pub initial_three_plus_tracks: usize,
    /// Independently localized target-side landmarks searched for alternatives.
    pub candidate_pool_landmarks: usize,
    /// Target-camera observations for which more than one self-similar
    /// candidate was retained.
    pub ambiguous_observations: usize,
    pub total_candidates: usize,
    pub max_candidates_per_observation: usize,
    /// Assignment changes made only on the fit population.
    pub assignment_iterations: usize,
    pub assignment_switches: usize,
    pub geometry_supported_switches: usize,
    pub constellation_supported_switches: usize,
    pub cycle_supported_switches: usize,
    pub cycle_rejected_switches: usize,
    pub collision_rejected_switches: usize,
    /// Bootstrap-only anchor-graph evidence used as a correspondence
    /// constraint inside LatentGraph. The graph is image-derived and is not a
    /// second physical-rig solve.
    pub cycle_graph_candidate_edges: usize,
    pub cycle_graph_active_edges: usize,
    pub cycle_graph_pairs: usize,
    pub cycle_graph_anchor_tracks: usize,
    pub cycle_graph_anchor_observations: usize,
    pub cycle_graph_validation_anchors_excluded: usize,
    pub cycle_predictions_evaluated: usize,
    /// Image-only frozen validation labels rejected by the local-constellation
    /// consistency check before any physical rig is fitted.
    pub validation_constellation_rejected_observations: usize,
    /// Per-camera feature and reversible-membership support diagnostics.
    pub camera_support: Vec<LatentCameraSupportReport>,
    pub membership_soft_outlier_events: usize,
    pub membership_reactivated_events: usize,
    pub membership_floor_reactivated_events: usize,
    pub membership_floor_protected_events: usize,
    pub pairwise_track_demotions: usize,
    pub pairwise_track_reactivations: usize,
    pub rounds: Vec<LatentMatchRoundReport>,
}

#[derive(Clone, Debug)]
pub(super) struct LatentCandidate {
    pub pixel: Vec2,
    pub localization_covariance: [[f64; 2]; 2],
    pub confidence: f64,
    pub local_scale: f64,
    pub structure: f64,
    /// Self-similarity in the target camera.  The initially selected
    /// observation has similarity 1.0 by construction.
    pub appearance_similarity: f64,
}

#[derive(Clone, Debug)]
pub(super) struct LatentObservationCandidates {
    pub camera: usize,
    pub candidates: Vec<LatentCandidate>,
    /// Whether the initially selected image correspondence is strong enough to
    /// serve as a frozen validation label without consulting fitted geometry.
    /// Fit observations deliberately keep weaker labels because the latent
    /// solver is allowed to repair them.
    pub validation_reliable: bool,
}

#[derive(Clone, Copy, Debug)]
struct CycleAnchor {
    first: Vec2,
    second: Vec2,
    /// Reference-camera location of the 3+-view identity that generated this
    /// anchor. This lets the fit-side graph exclude frozen validation blocks.
    reference_pixel: Vec2,
}

/// Image-only bootstrap multi-camera correspondence graph used as a
/// higher-order identity constraint by LatentGraph.
#[derive(Clone, Debug, Default)]
struct LatentCycleGraph {
    pairs: HashMap<(usize, usize), Vec<CycleAnchor>>,
    anchor_tracks: usize,
    anchor_observations: usize,
}

#[inline]
fn ordered_camera_pair(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

impl LatentCycleGraph {
    fn from_anchor_tracks(
        tracks: &[Track],
        minimum_pair_anchors: usize,
        reference_index: usize,
    ) -> Self {
        let mut graph = Self::default();
        for track in tracks.iter().filter(|track| track.observations.len() >= 3) {
            let Some(reference_pixel) = track
                .observations
                .iter()
                .find(|observation| observation.camera == reference_index)
                .map(|observation| observation.pixel)
            else {
                continue;
            };
            graph.anchor_tracks += 1;
            graph.anchor_observations += track.observations.len();
            for first_index in 0..track.observations.len() {
                for second_index in first_index + 1..track.observations.len() {
                    let first = &track.observations[first_index];
                    let second = &track.observations[second_index];
                    let pair = ordered_camera_pair(first.camera, second.camera);
                    let anchor = if first.camera <= second.camera {
                        CycleAnchor {
                            first: first.pixel,
                            second: second.pixel,
                            reference_pixel,
                        }
                    } else {
                        CycleAnchor {
                            first: second.pixel,
                            second: first.pixel,
                            reference_pixel,
                        }
                    };
                    graph.pairs.entry(pair).or_default().push(anchor);
                }
            }
        }
        graph
            .pairs
            .retain(|_, anchors| anchors.len() >= minimum_pair_anchors.max(3));
        graph
    }

    fn reference_block(pixel: Vec2, block_size_px: usize) -> [i32; 2] {
        let block_units = block_size_px.max(1).saturating_mul(16) as i32;
        [
            ((pixel[0] * 16.0).round() as i32).div_euclid(block_units),
            ((pixel[1] * 16.0).round() as i32).div_euclid(block_units),
        ]
    }

    fn exclude_validation_blocks(
        &mut self,
        selected_blocks: &HashSet<[i32; 2]>,
        block_size_px: usize,
        minimum_pair_anchors: usize,
    ) -> usize {
        let mut removed = 0usize;
        for anchors in self.pairs.values_mut() {
            let before = anchors.len();
            anchors.retain(|anchor| {
                !selected_blocks.contains(&Self::reference_block(
                    anchor.reference_pixel,
                    block_size_px,
                ))
            });
            removed += before.saturating_sub(anchors.len());
        }
        self.pairs
            .retain(|_, anchors| anchors.len() >= minimum_pair_anchors.max(3));
        removed
    }

    fn prediction(
        &self,
        source_camera: usize,
        target_camera: usize,
        source_pixel: Vec2,
        neighbour_count: usize,
    ) -> Option<Vec2> {
        if source_camera == target_camera {
            return None;
        }
        let pair = ordered_camera_pair(source_camera, target_camera);
        let anchors = self.pairs.get(&pair)?;
        let reversed = source_camera > target_camera;
        let mut nearest = anchors
            .iter()
            .map(|anchor| {
                let source = if reversed {
                    anchor.second
                } else {
                    anchor.first
                };
                let target = if reversed {
                    anchor.first
                } else {
                    anchor.second
                };
                (distance_squared(source_pixel, source), source, target, 1.0)
            })
            .filter(|(distance, _, _, _)| distance.is_finite())
            .collect::<Vec<_>>();
        if nearest.len() < 3 {
            return None;
        }
        nearest.sort_by(|left, right| left.0.total_cmp(&right.0));
        nearest.truncate(neighbour_count.max(3));
        let distance_scale = nearest[nearest.len().min(4) - 1].0.sqrt().max(8.0);
        let robust_weights = vec![1.0; nearest.len()];
        let (beta_x, beta_y) =
            weighted_local_affine_fit(&nearest, source_pixel, distance_scale, &robust_weights)?;
        let predicted = [beta_x[0], beta_y[0]];
        predicted
            .iter()
            .all(|value| value.is_finite())
            .then_some(predicted)
    }

    fn predictions_for_track(
        &self,
        track: &Track,
        target_camera: usize,
        neighbour_count: usize,
    ) -> Vec<Vec2> {
        track
            .observations
            .iter()
            .filter(|observation| observation.camera != target_camera)
            .filter_map(|observation| {
                self.prediction(
                    observation.camera,
                    target_camera,
                    observation.pixel,
                    neighbour_count,
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct LatentCandidateState {
    by_track: HashMap<[i32; 2], Vec<LatentObservationCandidates>>,
    cycle_graph: LatentCycleGraph,
}

impl LatentCandidateState {
    /// Produce an image-only frozen validation track.  The spatial hold-out is
    /// chosen before any rig fitting; within that block we additionally remove
    /// target observations whose original pairwise identity is intrinsically
    /// ambiguous.  This is not geometry-based pruning: it uses only the
    /// forward/backward match quality and target-side self-similarity that were
    /// available before the candidate camera existed.
    pub(super) fn validation_track(
        &self,
        track: &Track,
        maximum_alternative_similarity: f64,
    ) -> Option<(Track, usize)> {
        let observations = self.by_track.get(&track.key)?;
        let mut filtered = track.clone();
        let before = filtered.observations.len();
        filtered.observations.retain(|observation| {
            if observation.fixed_gauge {
                return true;
            }
            observations
                .iter()
                .find(|set| set.camera == observation.camera)
                .is_some_and(|set| {
                    set.validation_reliable
                        && set.candidates.iter().skip(1).all(|candidate| {
                            candidate.appearance_similarity <= maximum_alternative_similarity
                        })
                })
        });
        let removed = before.saturating_sub(filtered.observations.len());
        // A two-view track can be triangulated to nearly zero residual by
        // construction and is therefore a poor independent camera-model test.
        // Require reference + at least two target views.
        (filtered.observations.len() >= 3).then_some((filtered, removed))
    }

    pub(super) fn exclude_cycle_validation_blocks(
        &mut self,
        selected_blocks: &HashSet<[i32; 2]>,
        block_size_px: usize,
        minimum_pair_anchors: usize,
    ) -> usize {
        self.cycle_graph.exclude_validation_blocks(
            selected_blocks,
            block_size_px,
            minimum_pair_anchors,
        )
    }

    pub(super) fn cycle_graph_pair_count(&self) -> usize {
        self.cycle_graph.pairs.len()
    }
}

pub(super) struct LatentBuildOutcome {
    pub tracks: Vec<Track>,
    pub pairwise_matches: usize,
    pub candidates: LatentCandidateState,
    pub report: LatentMatchReport,
}

#[derive(Clone, Debug)]
struct LatentMembershipTrack {
    template: Track,
    inactive_cameras: HashSet<usize>,
    /// Intrinsically two-view tracks are switched on/off as a whole because
    /// no leave-one-camera-out 3-D prediction exists for either observation.
    inactive_pairwise_track: bool,
}

/// Persistent fit membership for LatentGraph.
///
/// Unlike the legacy membership pass, observations are never destroyed. A bad
/// observation becomes dormant and can re-enter after a subsequent bundle
/// update. The template also keeps the latest latent identity for every active
/// camera, so correspondence switching and membership remain two independent
/// latent variables.
#[derive(Clone, Debug, Default)]
pub(super) struct LatentMembershipState {
    tracks: Vec<LatentMembershipTrack>,
    by_key: HashMap<[i32; 2], usize>,
    initial_per_camera: Vec<usize>,
    initial_pairwise_per_camera: Vec<usize>,
    floor_per_camera: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LatentMembershipUpdate {
    pub demoted: usize,
    pub reactivated: usize,
    pub floor_reactivated: usize,
    pub floor_protected: usize,
    pub pairwise_demoted: usize,
    pub pairwise_reactivated: usize,
}

impl LatentMembershipUpdate {
    pub(super) fn changed(self) -> bool {
        self.demoted > 0
            || self.reactivated > 0
            || self.floor_reactivated > 0
            || self.pairwise_demoted > 0
            || self.pairwise_reactivated > 0
    }
}

impl LatentMembershipState {
    pub(super) fn new(
        tracks: &[Track],
        camera_count: usize,
        options: &RigRefinementOptions,
        report: &mut LatentMatchReport,
    ) -> Self {
        let mut initial_per_camera = vec![0usize; camera_count];
        let mut initial_pairwise_per_camera = vec![0usize; camera_count];
        let mut state_tracks = Vec::with_capacity(tracks.len());
        let mut by_key = HashMap::with_capacity(tracks.len());
        for track in tracks {
            for observation in &track.observations {
                if !observation.fixed_gauge {
                    initial_per_camera[observation.camera] += 1;
                    if track.observations.len() == 2 {
                        initial_pairwise_per_camera[observation.camera] += 1;
                    }
                }
            }
            let index = state_tracks.len();
            by_key.insert(track.key, index);
            state_tracks.push(LatentMembershipTrack {
                template: track.clone(),
                inactive_cameras: HashSet::new(),
                inactive_pairwise_track: false,
            });
        }
        let fraction = options
            .latent_membership_min_camera_fraction
            .clamp(0.0, 1.0);
        let floor_per_camera = initial_per_camera
            .iter()
            .map(|&count| {
                if count == 0 {
                    0
                } else {
                    let fractional = ((count as f64) * fraction).ceil() as usize;
                    count.min(
                        options
                            .latent_membership_min_camera_observations
                            .max(fractional),
                    )
                }
            })
            .collect::<Vec<_>>();
        let state = Self {
            tracks: state_tracks,
            by_key,
            initial_per_camera,
            initial_pairwise_per_camera,
            floor_per_camera,
        };
        state.refresh_report(report);
        state
    }

    /// Copy latent identity changes from the active working set back into the
    /// persistent templates. Dormant observations retain their last identity
    /// and can be reconsidered later.
    pub(super) fn sync_from_active(&mut self, tracks: &[Track]) {
        for track in tracks {
            let Some(&state_index) = self.by_key.get(&track.key) else {
                continue;
            };
            let state_track = &mut self.tracks[state_index];
            for observation in &track.observations {
                if let Some(slot) = state_track
                    .template
                    .observations
                    .iter_mut()
                    .find(|candidate| candidate.camera == observation.camera)
                {
                    *slot = observation.clone();
                }
            }
            state_track.template.condition = track.condition;
            state_track.template.max_ray_angle_degrees = track.max_ray_angle_degrees;
        }
    }

    fn active_observation_count(track: &LatentMembershipTrack) -> usize {
        if track.inactive_pairwise_track {
            return 0;
        }
        track
            .template
            .observations
            .iter()
            .filter(|observation| {
                observation.fixed_gauge || !track.inactive_cameras.contains(&observation.camera)
            })
            .count()
    }

    fn active_per_camera(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.initial_per_camera.len()];
        for track in &self.tracks {
            if track.inactive_pairwise_track {
                continue;
            }
            for observation in &track.template.observations {
                if !observation.fixed_gauge && !track.inactive_cameras.contains(&observation.camera)
                {
                    counts[observation.camera] += 1;
                }
            }
        }
        counts
    }

    fn prediction_error(
        &self,
        track_index: usize,
        camera: usize,
        cameras: &[ResolvedCamera],
        options: &RigRefinementOptions,
    ) -> Option<f64> {
        let track = self.tracks.get(track_index)?;
        let target = track
            .template
            .observations
            .iter()
            .find(|observation| observation.camera == camera && !observation.fixed_gauge)?;
        let other_observations = track
            .template
            .observations
            .iter()
            .filter(|observation| {
                observation.camera != camera
                    && (observation.fixed_gauge
                        || !track.inactive_cameras.contains(&observation.camera))
            })
            .cloned()
            .collect::<Vec<_>>();
        if other_observations.len() < 2 {
            return None;
        }
        let triangulated = triangulate(&other_observations, cameras, options)?;
        let predicted = cameras[camera]
            .project(triangulated.point)
            .filter(|pixel| pixel.iter().all(|value| value.is_finite()))
            .filter(|&pixel| cameras[camera].contains(pixel))?;
        let dx = predicted[0] - target.pixel[0];
        let dy = predicted[1] - target.pixel[1];
        let sensor_error = (dx * dx + dy * dy).sqrt();
        let reference_error = sensor_error / target.local_scale.clamp(0.25, 4.0);
        reference_error.is_finite().then_some(reference_error)
    }

    /// Geometry-only consistency for an intrinsically two-view track.  Such a
    /// track cannot be leave-one-camera-out triangulated: removing either view
    /// leaves one ray.  Instead measure the symmetric calibrated epipolar-plane
    /// error under the current physical cameras.  The value is expressed in
    /// reference-equivalent pixels and includes a conservative cheirality
    /// penalty so an opposite-depth line intersection is not treated as a good
    /// pair merely because the rays are coplanar.
    fn pairwise_prediction_error(
        &self,
        track_index: usize,
        cameras: &[ResolvedCamera],
        options: &RigRefinementOptions,
    ) -> Option<f64> {
        let track = self.tracks.get(track_index)?;
        if track.template.observations.len() != 2 {
            return None;
        }
        let first = &track.template.observations[0];
        let second = &track.template.observations[1];
        let first_camera = cameras.get(first.camera)?;
        let second_camera = cameras.get(second.camera)?;
        let first_ray = first_camera.pixel_to_ray(first.pixel);
        let second_ray = second_camera.pixel_to_ray(second.pixel);
        let baseline = sub(second_ray.origin, first_ray.origin);
        let baseline_norm = norm(baseline);
        if !baseline_norm.is_finite() || baseline_norm <= 1.0e-9 {
            return None;
        }

        let error_one_way = |source_direction: Vec3,
                             baseline: Vec3,
                             target_direction: Vec3,
                             target_camera: &ResolvedCamera,
                             target_scale: f64|
         -> Option<f64> {
            let normal = cross(source_direction, baseline);
            let normal_norm = norm(normal);
            if !normal_norm.is_finite() || normal_norm <= 1.0e-12 {
                return None;
            }
            let unit_normal = scale(normal, 1.0 / normal_norm);
            let angular = dot(unit_normal, target_direction)
                .abs()
                .clamp(0.0, 1.0)
                .asin();
            let pixels = angular * target_camera.focal_scale_px();
            Some(pixels / target_scale.clamp(0.25, 4.0))
        };

        let forward = error_one_way(
            first_ray.direction,
            baseline,
            second_ray.direction,
            second_camera,
            second.local_scale,
        )?;
        let reverse = error_one_way(
            second_ray.direction,
            scale(baseline, -1.0),
            first_ray.direction,
            first_camera,
            first.local_scale,
        )?;
        let mut error = ((forward * forward + reverse * reverse) * 0.5).sqrt();
        if !pair_signed_depths(first_ray, second_ray)
            .is_some_and(|(first_depth, second_depth)| first_depth > 0.0 && second_depth > 0.0)
        {
            error = error.max(options.latent_pairwise_membership_max_reference_px * 2.0);
        }
        error.is_finite().then_some(error)
    }

    fn rebuild_active_tracks(&self, tracks: &mut Vec<Track>) {
        tracks.clear();
        tracks.reserve(self.tracks.len());
        for state_track in &self.tracks {
            if state_track.inactive_pairwise_track {
                continue;
            }
            let mut track = state_track.template.clone();
            track.observations.retain(|observation| {
                observation.fixed_gauge
                    || !state_track.inactive_cameras.contains(&observation.camera)
            });
            if track.observations.len() >= 2 {
                track.condition = f64::NAN;
                track.max_ray_angle_degrees = f64::NAN;
                tracks.push(track);
            }
        }
    }

    pub(super) fn refresh_report(&self, report: &mut LatentMatchReport) {
        let active = self.active_per_camera();
        let mut active_pairwise = vec![0usize; self.initial_per_camera.len()];
        for track in &self.tracks {
            if track.inactive_pairwise_track || track.template.observations.len() != 2 {
                continue;
            }
            for observation in &track.template.observations {
                if !observation.fixed_gauge {
                    active_pairwise[observation.camera] += 1;
                }
            }
        }
        for camera in 0..report.camera_support.len() {
            let support = &mut report.camera_support[camera];
            support.initial_fit_observations =
                self.initial_per_camera.get(camera).copied().unwrap_or(0);
            support.final_fit_observations = active.get(camera).copied().unwrap_or(0);
            support.initial_pairwise_fit_observations = self
                .initial_pairwise_per_camera
                .get(camera)
                .copied()
                .unwrap_or(0);
            support.final_pairwise_fit_observations =
                active_pairwise.get(camera).copied().unwrap_or(0);
            support.membership_floor = self.floor_per_camera.get(camera).copied().unwrap_or(0);
        }
    }

    /// Reversible, leave-one-camera-out membership update with camera support
    /// floors. The update is synchronous: errors are measured from one frozen
    /// active set, then promotions/demotions are applied together.
    pub(super) fn update(
        &mut self,
        tracks: &mut Vec<Track>,
        cameras: &[ResolvedCamera],
        options: &RigRefinementOptions,
        report: &mut LatentMatchReport,
    ) -> LatentMembershipUpdate {
        self.sync_from_active(tracks);
        let mut scored = Vec::<(usize, usize, f64, bool)>::new();
        let mut pairwise_scored = Vec::<(usize, f64, bool)>::new();
        for (track_index, track) in self.tracks.iter().enumerate() {
            if track.template.observations.len() == 2 {
                if let Some(error) = self.pairwise_prediction_error(track_index, cameras, options) {
                    pairwise_scored.push((track_index, error, track.inactive_pairwise_track));
                }
                continue;
            }
            for observation in &track.template.observations {
                if observation.fixed_gauge {
                    continue;
                }
                let inactive = track.inactive_cameras.contains(&observation.camera);
                if let Some(error) =
                    self.prediction_error(track_index, observation.camera, cameras, options)
                {
                    scored.push((track_index, observation.camera, error, inactive));
                }
            }
        }

        let mut update = LatentMembershipUpdate::default();

        // Hysteretic recovery first: after the continuous rig improves, a
        // previously dormant observation can become geometrically convincing.
        for &(track_index, camera, error, inactive) in &scored {
            if inactive && error <= options.latent_membership_recovery_reference_px {
                if self.tracks[track_index].inactive_cameras.remove(&camera) {
                    update.reactivated += 1;
                    report.membership_reactivated_events += 1;
                    if let Some(support) = report.camera_support.get_mut(camera) {
                        support.reactivated_events += 1;
                    }
                }
            }
        }

        // Intrinsically two-view tracks recover as a unit from their symmetric
        // epipolar score. This lets a track return when calibration moves, even
        // though it was absent from the previous bundle working set.
        for &(track_index, error, inactive) in &pairwise_scored {
            if !inactive || error > options.latent_pairwise_membership_recovery_reference_px {
                continue;
            }
            self.tracks[track_index].inactive_pairwise_track = false;
            update.pairwise_reactivated += 1;
            report.pairwise_track_reactivations += 1;
            for observation in &self.tracks[track_index].template.observations {
                if observation.fixed_gauge {
                    continue;
                }
                update.reactivated += 1;
                report.membership_reactivated_events += 1;
                if let Some(support) = report.camera_support.get_mut(observation.camera) {
                    support.reactivated_events += 1;
                    support.pairwise_track_reactivations += 1;
                }
            }
        }

        let mut active_per_camera = self.active_per_camera();
        let mut demotions = scored
            .iter()
            .filter_map(|&(track_index, camera, error, was_inactive)| {
                (!was_inactive && error > options.fit_membership_max_reference_px).then_some((
                    error,
                    track_index,
                    camera,
                ))
            })
            .collect::<Vec<_>>();
        demotions.sort_by(|left, right| right.0.total_cmp(&left.0));

        for (_, track_index, camera) in demotions {
            if self.tracks[track_index].inactive_cameras.contains(&camera) {
                continue;
            }
            // A 3+-view track can shed inconsistent cameras and later
            // recover them, but never destroys its last two-view core. Tracks
            // that were intrinsically pairwise from the start are handled by
            // the separate whole-track epipolar test below.
            if Self::active_observation_count(&self.tracks[track_index]) <= 2 {
                continue;
            }
            let floor = self.floor_per_camera.get(camera).copied().unwrap_or(0);
            if active_per_camera.get(camera).copied().unwrap_or(0) <= floor {
                update.floor_protected += 1;
                report.membership_floor_protected_events += 1;
                if let Some(support) = report.camera_support.get_mut(camera) {
                    support.floor_protected_events += 1;
                }
                continue;
            }
            self.tracks[track_index].inactive_cameras.insert(camera);
            active_per_camera[camera] = active_per_camera[camera].saturating_sub(1);
            update.demoted += 1;
            report.membership_soft_outlier_events += 1;
            if let Some(support) = report.camera_support.get_mut(camera) {
                support.soft_outlier_events += 1;
            }
        }

        // Two-view tracks no longer get an unconditional pass. If the current
        // physical cameras cannot put their two rays into one epipolar plane,
        // make the entire track dormant. Respect the same per-camera support
        // floor so this cannot starve a narrow C module.
        let mut pairwise_demotions = pairwise_scored
            .iter()
            .filter_map(|&(track_index, error, was_inactive)| {
                (!was_inactive && error > options.latent_pairwise_membership_max_reference_px)
                    .then_some((error, track_index))
            })
            .collect::<Vec<_>>();
        pairwise_demotions.sort_by(|left, right| right.0.total_cmp(&left.0));
        for (_, track_index) in pairwise_demotions {
            if self.tracks[track_index].inactive_pairwise_track {
                continue;
            }
            let affected = self.tracks[track_index]
                .template
                .observations
                .iter()
                .filter(|observation| !observation.fixed_gauge)
                .map(|observation| observation.camera)
                .collect::<Vec<_>>();
            let would_starve = affected.iter().any(|&camera| {
                active_per_camera.get(camera).copied().unwrap_or(0)
                    <= self.floor_per_camera.get(camera).copied().unwrap_or(0)
            });
            if would_starve {
                update.floor_protected += affected.len();
                report.membership_floor_protected_events += affected.len();
                for &camera in &affected {
                    if let Some(support) = report.camera_support.get_mut(camera) {
                        support.floor_protected_events += 1;
                    }
                }
                continue;
            }
            self.tracks[track_index].inactive_pairwise_track = true;
            update.pairwise_demoted += 1;
            report.pairwise_track_demotions += 1;
            for camera in affected {
                active_per_camera[camera] = active_per_camera[camera].saturating_sub(1);
                update.demoted += 1;
                report.membership_soft_outlier_events += 1;
                if let Some(support) = report.camera_support.get_mut(camera) {
                    support.soft_outlier_events += 1;
                    support.pairwise_track_demotions += 1;
                }
            }
        }

        // A sparse/narrow-FOV camera must not be destroyed by robust pruning.
        // Restore its best dormant observations, but only when they are still
        // within a finite, deliberately loose LOO gate.
        for camera in 0..active_per_camera.len() {
            let floor = self.floor_per_camera[camera];
            if active_per_camera[camera] >= floor {
                continue;
            }
            let mut recovery = scored
                .iter()
                .filter_map(|&(track_index, scored_camera, error, _)| {
                    (scored_camera == camera
                        && self.tracks[track_index].inactive_cameras.contains(&camera)
                        && error <= options.latent_membership_floor_max_reference_px)
                        .then_some((error, track_index))
                })
                .collect::<Vec<_>>();
            recovery.sort_by(|left, right| left.0.total_cmp(&right.0));
            for (_, track_index) in recovery {
                if active_per_camera[camera] >= floor {
                    break;
                }
                if self.tracks[track_index].inactive_cameras.remove(&camera) {
                    active_per_camera[camera] += 1;
                    update.floor_reactivated += 1;
                    report.membership_floor_reactivated_events += 1;
                    if let Some(support) = report.camera_support.get_mut(camera) {
                        support.floor_reactivated_events += 1;
                    }
                }
            }

            // If LOO-capable observations are insufficient, pairwise tracks
            // may restore support only when their current epipolar error is
            // still inside the same loose floor gate.
            if active_per_camera[camera] < floor {
                let mut pairwise_recovery = pairwise_scored
                    .iter()
                    .filter_map(|&(track_index, error, _)| {
                        (self.tracks[track_index].inactive_pairwise_track
                            && error <= options.latent_membership_floor_max_reference_px
                            && self.tracks[track_index].template.observations.iter().any(
                                |observation| {
                                    !observation.fixed_gauge && observation.camera == camera
                                },
                            ))
                        .then_some((error, track_index))
                    })
                    .collect::<Vec<_>>();
                pairwise_recovery.sort_by(|left, right| left.0.total_cmp(&right.0));
                for (_, track_index) in pairwise_recovery {
                    if active_per_camera[camera] >= floor {
                        break;
                    }
                    if !self.tracks[track_index].inactive_pairwise_track {
                        continue;
                    }
                    self.tracks[track_index].inactive_pairwise_track = false;
                    update.pairwise_reactivated += 1;
                    report.pairwise_track_reactivations += 1;
                    for observation in &self.tracks[track_index].template.observations {
                        if observation.fixed_gauge {
                            continue;
                        }
                        active_per_camera[observation.camera] += 1;
                        update.floor_reactivated += 1;
                        report.membership_floor_reactivated_events += 1;
                        if let Some(support) = report.camera_support.get_mut(observation.camera) {
                            support.floor_reactivated_events += 1;
                            support.pairwise_track_reactivations += 1;
                        }
                    }
                }
            }
        }

        self.rebuild_active_tracks(tracks);
        self.refresh_report(report);
        update
    }
}

#[derive(Clone, Copy, Debug)]
struct SeedObservation {
    camera: usize,
    correspondence: AlignmentCorrespondence,
}

#[derive(Clone, Debug)]
struct SeedGroup {
    reference_pixel: Vec2,
    reference_covariance: [[f64; 2]; 2],
    reference_structure: f64,
    targets: Vec<Option<AlignmentCorrespondence>>,
}

#[derive(Clone, Debug)]
struct PoolFeature {
    pixel: Vec2,
    localization_covariance: [[f64; 2]; 2],
    confidence: f64,
    structure: f64,
    /// Only used to choose between duplicate seed observations. Detector-only
    /// corners are inserted after all seed observations and never replace them.
    seed_quality: f64,
    descriptor: Option<[f32; DESCRIPTOR_SAMPLES]>,
    ray_direction: Vec3,
}

#[inline]
fn distance_squared(a: Vec2, b: Vec2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn descriptor_for_sensor_pixel(plane: &Plane, pixel: Vec2) -> Option<[f32; DESCRIPTOR_SAMPLES]> {
    // Half-resolution luminance pixel (i,j) is centred at sensor
    // (2i+0.5, 2j+0.5).
    let cx = ((pixel[0] - 0.5) * 0.5) as f32;
    let cy = ((pixel[1] - 0.5) * 0.5) as f32;
    let centre = (DESCRIPTOR_SIDE as f32 - 1.0) * 0.5;
    let mut values = [0.0f32; DESCRIPTOR_SAMPLES];
    let mut sum = 0.0f64;
    let mut index = 0usize;
    for j in 0..DESCRIPTOR_SIDE {
        for i in 0..DESCRIPTOR_SIDE {
            let x = cx + (i as f32 - centre) * DESCRIPTOR_STEP_LUMA;
            let y = cy + (j as f32 - centre) * DESCRIPTOR_STEP_LUMA;
            let value = plane.sample(x, y)?;
            if !value.is_finite() {
                return None;
            }
            values[index] = value;
            sum += f64::from(value);
            index += 1;
        }
    }
    let mean = (sum / DESCRIPTOR_SAMPLES as f64) as f32;
    let mut norm_sq = 0.0f64;
    for value in &mut values {
        *value -= mean;
        norm_sq += f64::from(*value) * f64::from(*value);
    }
    if !norm_sq.is_finite() || norm_sq <= 1.0e-10 {
        return None;
    }
    let inverse_norm = (1.0 / norm_sq.sqrt()) as f32;
    for value in &mut values {
        *value *= inverse_norm;
    }
    Some(values)
}

#[inline]
fn descriptor_similarity(a: &[f32; DESCRIPTOR_SAMPLES], b: &[f32; DESCRIPTOR_SAMPLES]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>()
        .clamp(-1.0, 1.0)
}

/// Signed target-pixel approximation to the calibrated epipolar-plane error.
/// The *difference* between candidate and initial errors is used below, so a
/// capture-wide factory bearing error does not delete the correct alternative.
fn epipolar_plane_unit_normal(
    reference: &ResolvedCamera,
    target: &ResolvedCamera,
    reference_pixel: Vec2,
) -> Option<Vec3> {
    let reference_ray = reference.pixel_to_ray(reference_pixel);
    let baseline = sub(target.center(), reference_ray.origin);
    let normal = cross(reference_ray.direction, baseline);
    let magnitude = norm(normal);
    if !magnitude.is_finite() || magnitude <= 1.0e-12 {
        return None;
    }
    Some(scale(normal, 1.0 / magnitude))
}

#[inline]
fn epipolar_residual_from_direction(
    unit_normal: Vec3,
    target_focal_px: f64,
    target_direction: Vec3,
) -> f64 {
    dot(unit_normal, target_direction).clamp(-1.0, 1.0).asin() * target_focal_px
}

fn correspondence_quality(correspondence: &AlignmentCorrespondence) -> f64 {
    f64::from(correspondence.confidence) + 2.0 * f64::from(correspondence.peak_margin)
        - 0.1 * f64::from(correspondence.forward_backward_error_px)
}

fn filtered_seed_observations(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    resolved: &[ResolvedCamera],
) -> Vec<SeedObservation> {
    let mut observations = Vec::new();
    for (camera, alignment) in alignments.iter().enumerate() {
        if camera == reference_index
            || !cameras[camera].match_evidence_enabled
            || cameras[camera].calibration.is_none()
        {
            continue;
        }
        if !alignment.report.accepted && alignment.correspondences.len() < 8 {
            continue;
        }
        let mut epipolar_inliers = fallback_epipolar_inliers(
            &alignment.correspondences,
            &resolved[reference_index],
            &resolved[camera],
        );
        if !alignment.report.accepted
            && alignment.correspondences.len() >= FALLBACK_ROTATION_MIN_INLIERS
            && resolved[camera].focal_px >= resolved[reference_index].focal_px * 1.35
        {
            let rotation_inliers = fallback_rotation_inliers(
                &alignment.correspondences,
                &resolved[reference_index],
                &resolved[camera],
            );
            if !rotation_inliers.is_empty() {
                let mut keep = vec![false; alignment.correspondences.len()];
                for index in rotation_inliers {
                    keep[index] = true;
                }
                let intersection = epipolar_inliers
                    .iter()
                    .copied()
                    .filter(|&index| keep[index])
                    .collect::<Vec<_>>();
                if intersection.len() >= FALLBACK_ROTATION_MIN_INLIERS {
                    epipolar_inliers = intersection;
                }
            }
        }
        for index in epipolar_inliers {
            let correspondence = alignment.correspondences[index];
            if resolved[reference_index].contains(correspondence.reference_pixel)
                && resolved[camera].contains(correspondence.target_pixel)
            {
                observations.push(SeedObservation {
                    camera,
                    correspondence,
                });
            }
        }
    }
    observations
}

fn group_seed_observations(
    observations: &[SeedObservation],
    camera_count: usize,
    merge_radius_px: f64,
) -> Vec<SeedGroup> {
    let merge_radius_sq = merge_radius_px * merge_radius_px;
    let mut groups = Vec::<SeedGroup>::new();
    for seed in observations {
        let reference_pixel = seed.correspondence.reference_pixel;
        let nearest = groups
            .iter()
            .enumerate()
            .filter_map(|(index, group)| {
                let d2 = distance_squared(reference_pixel, group.reference_pixel);
                (d2 <= merge_radius_sq).then_some((index, d2))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index);
        let group_index = if let Some(index) = nearest {
            index
        } else {
            groups.push(SeedGroup {
                reference_pixel,
                reference_covariance: seed
                    .correspondence
                    .reference_localization_covariance
                    .map(|row| row.map(f64::from)),
                reference_structure: f64::from(seed.correspondence.structure),
                targets: vec![None; camera_count],
            });
            groups.len() - 1
        };
        let group = &mut groups[group_index];
        group.reference_structure = group
            .reference_structure
            .max(f64::from(seed.correspondence.structure));
        let slot = &mut group.targets[seed.camera];
        let replace = match slot.as_ref() {
            None => true,
            Some(current) => {
                correspondence_quality(&seed.correspondence) > correspondence_quality(current)
            }
        };
        if replace {
            *slot = Some(seed.correspondence);
        }
    }
    groups
}

fn build_pool(
    camera: usize,
    observations: &[SeedObservation],
    luminance: Option<&Plane>,
    resolved: &ResolvedCamera,
    dedup_radius_px: f64,
    maximum_detected_corners: usize,
) -> Vec<PoolFeature> {
    let Some(plane) = luminance else {
        return Vec::new();
    };
    let dedup_sq = dedup_radius_px * dedup_radius_px;
    let mut pool = Vec::<PoolFeature>::new();
    for seed in observations.iter().filter(|seed| seed.camera == camera) {
        let correspondence = seed.correspondence;
        if let Some(existing) = pool.iter_mut().find(|feature| {
            distance_squared(feature.pixel, correspondence.target_pixel) <= dedup_sq
        }) {
            if correspondence_quality(&correspondence) > existing.seed_quality {
                existing.pixel = correspondence.target_pixel;
                existing.localization_covariance = correspondence
                    .target_localization_covariance
                    .map(|row| row.map(f64::from));
                existing.confidence = f64::from(correspondence.confidence);
                existing.structure = f64::from(correspondence.structure);
                existing.seed_quality = correspondence_quality(&correspondence);
                existing.descriptor = descriptor_for_sensor_pixel(plane, existing.pixel);
                existing.ray_direction = resolved.pixel_to_ray(existing.pixel).direction;
            }
            continue;
        }
        pool.push(PoolFeature {
            pixel: correspondence.target_pixel,
            localization_covariance: correspondence
                .target_localization_covariance
                .map(|row| row.map(f64::from)),
            confidence: f64::from(correspondence.confidence),
            structure: f64::from(correspondence.structure),
            seed_quality: correspondence_quality(&correspondence),
            descriptor: descriptor_for_sensor_pixel(plane, correspondence.target_pixel),
            ray_direction: resolved.pixel_to_ray(correspondence.target_pixel).direction,
        });
    }

    // The aligner exposes only its accepted correspondences, but latent
    // assignment specifically needs alternatives that may never have won a
    // cross-camera nearest-neighbour test. Reuse the same Shi-Tomasi detector
    // on the target luminance plane and add those independently localized
    // landmarks to the candidate pool. Matched landmarks above stay preferred
    // by the de-duplication pass because they were inserted first.
    for corner in detect_rig_corners(plane)
        .into_iter()
        .take(maximum_detected_corners)
    {
        let pixel = [
            2.0 * f64::from(corner.subpixel[0]) + 0.5,
            2.0 * f64::from(corner.subpixel[1]) + 0.5,
        ];
        if pool
            .iter()
            .any(|feature| distance_squared(feature.pixel, pixel) <= dedup_sq)
        {
            continue;
        }
        pool.push(PoolFeature {
            pixel,
            localization_covariance: corner.covariance.map(|row| row.map(f64::from)),
            confidence: 1.0,
            structure: f64::from(corner.structure),
            seed_quality: f64::NEG_INFINITY,
            descriptor: descriptor_for_sensor_pixel(plane, pixel),
            ray_direction: resolved.pixel_to_ray(pixel).direction,
        });
    }
    pool
}

fn unique_track_key(reference_pixel: Vec2, used: &mut HashSet<[i32; 2]>) -> [i32; 2] {
    let mut key = [
        (reference_pixel[0] * 16.0).round() as i32,
        (reference_pixel[1] * 16.0).round() as i32,
    ];
    while !used.insert(key) {
        key[0] = key[0].wrapping_add(1);
    }
    key
}

pub(super) fn build_latent_tracks(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    resolved: &[ResolvedCamera],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> LatentBuildOutcome {
    let seed_observations =
        filtered_seed_observations(cameras, reference_index, alignments, resolved);
    let groups = group_seed_observations(
        &seed_observations,
        cameras.len(),
        options.latent_reference_merge_radius_px,
    );
    let pools = (0..cameras.len())
        .map(|camera| {
            if camera == reference_index
                || !cameras[camera].match_evidence_enabled
                || !seed_observations.iter().any(|seed| seed.camera == camera)
            {
                Vec::new()
            } else {
                build_pool(
                    camera,
                    &seed_observations,
                    cameras[camera].luminance,
                    &resolved[camera],
                    options.latent_candidate_dedup_radius_px,
                    options.latent_candidate_pool_max_corners,
                )
            }
        })
        .collect::<Vec<_>>();

    // Reuse the mature anchor-graph matcher only as an image-derived
    // correspondence constraint. Running zero propagation rounds performs
    // overlap edge selection, direct mutual image matching, and initial cycle
    // promotion, but never invokes AnchorGraph's intermediate physical rig.
    let (cycle_graph, cycle_candidate_edges, cycle_active_edges) =
        if options.latent_cycle_graph_enabled && options.latent_cycle_graph_max_edges > 0 {
            let mut cycle_options = options.clone();
            cycle_options.strategy = RigRefinementStrategy::AnchorGraph;
            cycle_options.anchor_max_rounds = 0;
            cycle_options.anchor_max_active_edges = cycle_options
                .anchor_max_active_edges
                .min(options.latent_cycle_graph_max_edges)
                .max(1);
            cycle_options.anchor_initial_active_edges = cycle_options.anchor_max_active_edges;
            let anchor = build_anchor_tracks(
                cameras,
                reference_index,
                alignments,
                resolved,
                intrinsics_mode,
                5,
                &cycle_options,
            );
            let candidate_edges = anchor.report.candidate_edges;
            let active_edges = anchor.report.final_active_edges;
            (
                LatentCycleGraph::from_anchor_tracks(
                    &anchor.tracks,
                    options.latent_cycle_min_pair_anchors,
                    reference_index,
                ),
                candidate_edges,
                active_edges,
            )
        } else {
            (LatentCycleGraph::default(), 0, 0)
        };

    let mut tracks = Vec::new();
    let mut candidate_state = LatentCandidateState {
        cycle_graph,
        ..Default::default()
    };
    let mut used_keys = HashSet::new();
    let camera_support = cameras
        .iter()
        .enumerate()
        .map(|(camera, input)| LatentCameraSupportReport {
            camera: input.name.to_owned(),
            detector_candidates: alignments
                .get(camera)
                .map_or(0, |alignment| alignment.report.rig_feature_candidates),
            image_matches: alignments
                .get(camera)
                .map_or(0, |alignment| alignment.report.rig_feature_matches),
            seed_pairwise_matches: seed_observations
                .iter()
                .filter(|seed| seed.camera == camera)
                .count(),
            candidate_pool_landmarks: pools.get(camera).map_or(0, Vec::len),
            ..Default::default()
        })
        .collect();
    let mut report = LatentMatchReport {
        seed_pairwise_matches: seed_observations.len(),
        candidate_pool_landmarks: pools.iter().map(Vec::len).sum(),
        cycle_graph_candidate_edges: cycle_candidate_edges,
        cycle_graph_active_edges: cycle_active_edges,
        cycle_graph_pairs: candidate_state.cycle_graph.pairs.len(),
        cycle_graph_anchor_tracks: candidate_state.cycle_graph.anchor_tracks,
        cycle_graph_anchor_observations: candidate_state.cycle_graph.anchor_observations,
        camera_support,
        ..Default::default()
    };

    for group in groups {
        let target_count = group.targets.iter().flatten().count();
        if target_count == 0 {
            continue;
        }
        let key = unique_track_key(group.reference_pixel, &mut used_keys);
        let mut track_observations = Vec::with_capacity(target_count + 1);
        track_observations.push(TrackObservation {
            camera: reference_index,
            pixel: group.reference_pixel,
            bootstrap_residual_proposal: [0.0, 0.0],
            localization_covariance: group.reference_covariance,
            fixed_gauge: true,
            confidence: 1.0,
            local_scale: 1.0,
            structure: group.reference_structure,
            depth_reliability: None,
            prepared: Default::default(),
        });

        let mut latent_observations = Vec::new();
        for (camera, correspondence) in group.targets.iter().enumerate() {
            let Some(correspondence) = *correspondence else {
                continue;
            };
            let base = LatentCandidate {
                pixel: correspondence.target_pixel,
                localization_covariance: correspondence
                    .target_localization_covariance
                    .map(|row| row.map(f64::from)),
                confidence: f64::from(correspondence.confidence),
                local_scale: f64::from(correspondence.local_scale),
                structure: f64::from(correspondence.structure),
                appearance_similarity: 1.0,
            };
            let mut candidates = vec![base.clone()];
            let base_descriptor = cameras[camera]
                .luminance
                .and_then(|plane| descriptor_for_sensor_pixel(plane, base.pixel));
            let epipolar_normal = epipolar_plane_unit_normal(
                &resolved[reference_index],
                &resolved[camera],
                group.reference_pixel,
            );
            // Cycle evidence is deliberately not used to generate or rank
            // alternatives here. The validation split is frozen later; using
            // the graph now could leak held-out identities into the fit set.

            if let (Some(base_descriptor), Some(epipolar_normal)) =
                (base_descriptor, epipolar_normal)
            {
                let base_direction = resolved[camera].pixel_to_ray(base.pixel).direction;
                let base_epipolar = epipolar_residual_from_direction(
                    epipolar_normal,
                    resolved[camera].focal_px,
                    base_direction,
                );
                let mut alternatives = Vec::<(f64, f64, &PoolFeature)>::new();
                for feature in &pools[camera] {
                    if distance_squared(feature.pixel, base.pixel)
                        <= options.latent_candidate_min_separation_px.powi(2)
                    {
                        continue;
                    }
                    let candidate_epipolar = epipolar_residual_from_direction(
                        epipolar_normal,
                        resolved[camera].focal_px,
                        feature.ray_direction,
                    );
                    let epipolar_delta = (candidate_epipolar - base_epipolar).abs();
                    if epipolar_delta > options.latent_candidate_epipolar_band_px {
                        continue;
                    }
                    let Some(descriptor) = feature.descriptor.as_ref() else {
                        continue;
                    };
                    let similarity = descriptor_similarity(&base_descriptor, descriptor);
                    if similarity < options.latent_min_appearance_similarity {
                        continue;
                    }
                    alternatives.push((similarity, epipolar_delta, feature));
                }
                alternatives.sort_by(|a, b| {
                    let score_a = a.0 - 0.01 * a.1.min(24.0);
                    let score_b = b.0 - 0.01 * b.1.min(24.0);
                    score_b.total_cmp(&score_a)
                });
                for (similarity, _, feature) in alternatives
                    .into_iter()
                    .take(options.latent_max_candidates.saturating_sub(1))
                {
                    candidates.push(LatentCandidate {
                        pixel: feature.pixel,
                        localization_covariance: feature.localization_covariance,
                        confidence: (base.confidence * similarity).min(feature.confidence),
                        local_scale: base.local_scale,
                        structure: base.structure.min(feature.structure.max(1.0e-6)),
                        appearance_similarity: similarity,
                    });
                }
            }

            report.total_candidates += candidates.len();
            report.max_candidates_per_observation =
                report.max_candidates_per_observation.max(candidates.len());
            if candidates.len() > 1 {
                report.ambiguous_observations += 1;
            }
            let validation_reliable = f64::from(correspondence.confidence)
                >= options.latent_validation_min_confidence
                && f64::from(correspondence.peak_margin)
                    >= options.latent_validation_min_peak_margin
                && f64::from(correspondence.forward_backward_error_px)
                    <= options.latent_validation_max_forward_backward_px;
            latent_observations.push(LatentObservationCandidates {
                camera,
                candidates,
                validation_reliable,
            });
            track_observations.push(TrackObservation {
                camera,
                pixel: base.pixel,
                bootstrap_residual_proposal: [0.0, 0.0],
                localization_covariance: base.localization_covariance,
                fixed_gauge: false,
                confidence: base.confidence,
                local_scale: base.local_scale,
                structure: base.structure,
                depth_reliability: None,
                prepared: Default::default(),
            });
        }
        if track_observations.len() < 2 {
            continue;
        }
        candidate_state.by_track.insert(key, latent_observations);
        tracks.push(Track {
            key,
            observations: track_observations,
            condition: f64::NAN,
            max_ray_angle_degrees: f64::NAN,
        });
    }

    // Validation labels must be trustworthy independently of the physical rig.
    // Pairwise confidence/FB checks above catch ordinary local failures, but a
    // repeated industrial motif can still produce a sharp, high-confidence
    // match at the wrong instance. Use the *initial image correspondences* to
    // fit the same robust local constellation used by latent assignment and
    // mark only grossly inconsistent labels as unsuitable for held-out ground
    // truth. No candidate camera, triangulation, or fitted geometry participates
    // in this pass, so it cannot leak the rig solution into validation.
    if options
        .latent_validation_max_constellation_error_px
        .is_finite()
        && options.latent_validation_max_constellation_error_px > 0.0
    {
        let mut constellation_rejected = 0usize;
        for track_index in 0..tracks.len() {
            let key = tracks[track_index].key;
            let cameras_to_check = candidate_state
                .by_track
                .get(&key)
                .map(|sets| {
                    sets.iter()
                        .filter(|set| set.validation_reliable)
                        .map(|set| set.camera)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut rejected = Vec::<usize>::new();
            for camera in cameras_to_check {
                let Some(predicted) = local_affine_prediction(
                    &tracks,
                    track_index,
                    camera,
                    options.latent_neighbour_count,
                ) else {
                    continue;
                };
                let Some(observation) = tracks[track_index]
                    .observations
                    .iter()
                    .find(|observation| observation.camera == camera)
                else {
                    continue;
                };
                let error = distance_squared(observation.pixel, predicted).sqrt()
                    / observation.local_scale.clamp(0.25, 4.0);
                if !error.is_finite()
                    || error > options.latent_validation_max_constellation_error_px
                {
                    rejected.push(camera);
                }
            }
            if rejected.is_empty() {
                continue;
            }
            if let Some(sets) = candidate_state.by_track.get_mut(&key) {
                for set in sets {
                    if rejected.contains(&set.camera) && set.validation_reliable {
                        set.validation_reliable = false;
                        constellation_rejected += 1;
                    }
                }
            }
        }
        report.validation_constellation_rejected_observations = constellation_rejected;
    }

    report.initial_tracks = tracks.len();
    report.initial_three_plus_tracks = tracks
        .iter()
        .filter(|track| track.observations.len() >= 3)
        .count();

    LatentBuildOutcome {
        pairwise_matches: seed_observations.len(),
        tracks,
        candidates: candidate_state,
        report,
    }
}

fn reference_pixel(track: &Track) -> Option<Vec2> {
    track
        .observations
        .iter()
        .find(|observation| observation.fixed_gauge)
        .map(|observation| observation.pixel)
}

fn weighted_local_affine_fit(
    samples: &[(f64, Vec2, Vec2, f64)],
    reference: Vec2,
    distance_scale: f64,
    robust_weights: &[f64],
) -> Option<([f64; 3], [f64; 3])> {
    let mut normal = [[0.0f64; 3]; 3];
    let mut rhs_x = [0.0f64; 3];
    let mut rhs_y = [0.0f64; 3];
    for ((d2, source, target, confidence), &robust_weight) in samples.iter().zip(robust_weights) {
        let row = [1.0, source[0] - reference[0], source[1] - reference[1]];
        let spatial_weight = 1.0 / (1.0 + d2 / (distance_scale * distance_scale));
        let confidence_weight = (0.20 + 0.80 * confidence.clamp(0.0, 1.0)).powi(2);
        let weight = spatial_weight * confidence_weight * robust_weight.clamp(0.0, 1.0);
        for r in 0..3 {
            rhs_x[r] += weight * row[r] * target[0];
            rhs_y[r] += weight * row[r] * target[1];
            for c in 0..3 {
                normal[r][c] += weight * row[r] * row[c];
            }
        }
    }
    for (axis, row) in normal.iter_mut().enumerate() {
        row[axis] += 1.0e-8;
    }
    let inverse = math::inverse(&normal)?;
    let beta_x = math::mul_vec(&inverse, rhs_x);
    let beta_y = math::mul_vec(&inverse, rhs_y);
    beta_x
        .iter()
        .chain(beta_y.iter())
        .all(|value| value.is_finite())
        .then_some((beta_x, beta_y))
}

fn local_affine_prediction(
    tracks: &[Track],
    current_track_index: usize,
    camera: usize,
    neighbour_count: usize,
) -> Option<Vec2> {
    let reference = reference_pixel(&tracks[current_track_index])?;
    let mut nearest = Vec::<(f64, Vec2, Vec2, f64)>::with_capacity(neighbour_count + 1);
    for (index, track) in tracks.iter().enumerate() {
        if index == current_track_index {
            continue;
        }
        let Some(neighbour_reference) = reference_pixel(track) else {
            continue;
        };
        let Some(observation) = track
            .observations
            .iter()
            .find(|observation| observation.camera == camera)
        else {
            continue;
        };
        let d2 = distance_squared(reference, neighbour_reference);
        if !d2.is_finite() || d2 <= 1.0e-9 {
            continue;
        }
        nearest.push((
            d2,
            neighbour_reference,
            observation.pixel,
            observation.confidence,
        ));
    }
    if nearest.len() < 3 {
        return None;
    }
    nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
    nearest.truncate(neighbour_count.max(3));

    // Local affine target = intercept + J * (reference - current_reference).
    // Repeated structure can leave a wrong latent identity among otherwise
    // coherent neighbours. A small IRLS loop prevents one such neighbour from
    // dragging the whole local constellation onto the wrong repetition.
    let distance_scale = nearest[nearest.len().min(4) - 1].0.sqrt().max(8.0);
    let mut robust_weights = vec![1.0; nearest.len()];
    let mut fitted = None;
    for _ in 0..3 {
        let (beta_x, beta_y) =
            weighted_local_affine_fit(&nearest, reference, distance_scale, &robust_weights)?;
        fitted = Some((beta_x, beta_y));

        let residuals = nearest
            .iter()
            .map(|(_, source, target, _)| {
                let dx = source[0] - reference[0];
                let dy = source[1] - reference[1];
                let predicted = [
                    beta_x[0] + beta_x[1] * dx + beta_x[2] * dy,
                    beta_y[0] + beta_y[1] * dx + beta_y[2] * dy,
                ];
                distance_squared(*target, predicted).sqrt()
            })
            .collect::<Vec<_>>();
        let mut sorted = residuals
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        if sorted.is_empty() {
            return None;
        }
        let median = sorted[sorted.len() / 2];
        let huber_limit = (2.5 * (1.4826 * median).max(0.35)).clamp(0.75, 6.0);
        for (weight, residual) in robust_weights.iter_mut().zip(residuals) {
            *weight = if residual.is_finite() && residual > huber_limit {
                huber_limit / residual
            } else if residual.is_finite() {
                1.0
            } else {
                0.0
            };
        }
    }
    let (beta_x, beta_y) = fitted?;
    let predicted = [beta_x[0], beta_y[0]];
    predicted
        .iter()
        .all(|value| value.is_finite())
        .then_some(predicted)
}

fn leave_one_camera_prediction(
    track: &Track,
    camera: usize,
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Option<Vec2> {
    let other_observations = track
        .observations
        .iter()
        .filter(|observation| observation.camera != camera)
        .cloned()
        .collect::<Vec<_>>();
    if other_observations.len() < 2 {
        return None;
    }
    let triangulated = triangulate(&other_observations, cameras, options)?;
    // A latent identity may receive geometry support only when the predicted
    // 3-D point actually lands on this sensor.  `project_unbounded()` was
    // previously used here, which let a point many frames outside the target
    // image vote for a visually similar in-frame candidate.  That is exactly
    // the wrong behaviour at the narrow C-camera field boundaries.
    cameras[camera]
        .project(triangulated.point)
        .filter(|pixel| pixel.iter().all(|value| value.is_finite()))
        .filter(|&pixel| cameras[camera].contains(pixel))
}

#[derive(Clone, Copy, Debug, Default)]
struct CandidateScore {
    total: f64,
    geometry_error: Option<f64>,
    constellation_error: Option<f64>,
    cycle_error: Option<f64>,
}

fn candidate_prediction_error(candidate: &LatentCandidate, prediction: Vec2) -> f64 {
    let residual = [
        candidate.pixel[0] - prediction[0],
        candidate.pixel[1] - prediction[1],
    ];
    let covariance = candidate.localization_covariance;
    let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
    let anisotropic = if determinant.is_finite() && determinant > 1.0e-9 {
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
        (residual[0] * (inverse[0][0] * residual[0] + inverse[0][1] * residual[1])
            + residual[1] * (inverse[1][0] * residual[0] + inverse[1][1] * residual[1]))
            .max(0.0)
            .sqrt()
    } else {
        distance_squared(candidate.pixel, prediction).sqrt()
    };
    anisotropic / candidate.local_scale.clamp(0.25, 4.0)
}

fn candidate_score(
    candidate: &LatentCandidate,
    geometry_prediction: Option<Vec2>,
    constellation_prediction: Option<Vec2>,
    cycle_predictions: &[Vec2],
    options: &RigRefinementOptions,
) -> CandidateScore {
    let geometry_error =
        geometry_prediction.map(|prediction| candidate_prediction_error(candidate, prediction));
    let constellation_error = constellation_prediction
        .map(|prediction| candidate_prediction_error(candidate, prediction));
    let cycle_error = if cycle_predictions.is_empty() {
        None
    } else {
        let mut errors = cycle_predictions
            .iter()
            .map(|&prediction| candidate_prediction_error(candidate, prediction))
            .filter(|error| error.is_finite())
            .collect::<Vec<_>>();
        errors.sort_by(f64::total_cmp);
        (!errors.is_empty()).then(|| errors[errors.len() / 2])
    };
    let appearance = options.latent_appearance_penalty_px
        * (1.0 - candidate.appearance_similarity.clamp(-1.0, 1.0));
    let mut total = appearance;
    if let Some(error) = geometry_error {
        total += error.min(options.latent_score_error_cap_px);
    }
    if let Some(error) = constellation_error {
        total += options.latent_constellation_weight * error.min(options.latent_score_error_cap_px);
    }
    if let Some(error) = cycle_error {
        total += options.latent_cycle_weight * error.min(options.latent_score_error_cap_px);
    }
    CandidateScore {
        total,
        geometry_error,
        constellation_error,
        cycle_error,
    }
}

fn current_candidate_index(candidates: &[LatentCandidate], pixel: Vec2) -> usize {
    candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            distance_squared(a.pixel, pixel).total_cmp(&distance_squared(b.pixel, pixel))
        })
        .map_or(0, |(index, _)| index)
}

#[derive(Clone, Debug)]
struct AssignmentProposal {
    track_index: usize,
    observation_index: usize,
    camera: usize,
    current_pixel: Vec2,
    candidate: LatentCandidate,
    score_gain: f64,
    geometry_supported: bool,
    constellation_supported: bool,
    cycle_supported: bool,
}

pub(super) fn update_assignments(
    tracks: &mut [Track],
    candidates: &LatentCandidateState,
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
    iteration: usize,
    report: &mut LatentMatchReport,
) -> usize {
    let mut proposals = Vec::<AssignmentProposal>::new();
    let mut evaluated = 0usize;
    let mut score_before_sum = 0.0f64;
    let mut score_after_sum = 0.0f64;
    let mut cycle_rejected_switches = 0usize;
    let mut cycle_predictions_evaluated = 0usize;

    // Build every proposal against one immutable assignment snapshot, then
    // apply them together.  Sequentially mutating a repeated lattice lets the
    // first switch drag all later neighbourhood predictions with it.
    for (track_index, track) in tracks.iter().enumerate() {
        let Some(track_candidates) = candidates.by_track.get(&track.key) else {
            continue;
        };
        for (observation_index, observation) in track.observations.iter().enumerate() {
            if observation.fixed_gauge {
                continue;
            }
            let Some(set) = track_candidates
                .iter()
                .find(|set| set.camera == observation.camera)
            else {
                continue;
            };
            if set.candidates.len() < 2 {
                continue;
            }
            let geometry_prediction =
                leave_one_camera_prediction(track, observation.camera, cameras, options);
            let constellation_prediction = local_affine_prediction(
                tracks,
                track_index,
                observation.camera,
                options.latent_neighbour_count,
            );
            let cycle_predictions = candidates.cycle_graph.predictions_for_track(
                track,
                observation.camera,
                options.latent_cycle_neighbour_count,
            );
            if geometry_prediction.is_none()
                && constellation_prediction.is_none()
                && cycle_predictions.is_empty()
            {
                continue;
            }
            cycle_predictions_evaluated += cycle_predictions.len();
            evaluated += 1;
            let current_index = current_candidate_index(&set.candidates, observation.pixel);
            let current_score = candidate_score(
                &set.candidates[current_index],
                geometry_prediction,
                constellation_prediction,
                &cycle_predictions,
                options,
            );
            let mut best_index = current_index;
            let mut best_score = current_score;
            for (candidate_index, candidate) in set.candidates.iter().enumerate() {
                let score = candidate_score(
                    candidate,
                    geometry_prediction,
                    constellation_prediction,
                    &cycle_predictions,
                    options,
                );
                if score.total < best_score.total {
                    best_index = candidate_index;
                    best_score = score;
                }
            }
            score_before_sum += current_score.total;
            if best_index == current_index
                || best_score.total + options.latent_switch_margin_px >= current_score.total
            {
                score_after_sum += current_score.total;
                continue;
            }
            let geometry_supported = best_score
                .geometry_error
                .is_some_and(|error| error <= options.latent_max_reprojection_px);
            let constellation_supported = best_score
                .constellation_error
                .is_some_and(|error| error <= options.latent_max_constellation_error_px);
            let cycle_supported = best_score
                .cycle_error
                .is_some_and(|error| error <= options.latent_cycle_max_error_px);
            // If leave-one-camera-out geometry is available, a repeated local
            // pattern is not allowed to overrule a candidate that is grossly
            // inconsistent with that 3-D prediction. Constellation-only
            // switches remain legal while the track is not yet triangulatable
            // without this camera.
            let geometry_contradicts = best_score
                .geometry_error
                .is_some_and(|error| error > options.latent_max_reprojection_px);
            let cycle_contradicts = best_score
                .cycle_error
                .is_some_and(|error| error > options.latent_cycle_max_error_px)
                && !geometry_supported;
            if cycle_contradicts {
                cycle_rejected_switches += 1;
            }
            if geometry_contradicts
                || cycle_contradicts
                || (!geometry_supported && !constellation_supported && !cycle_supported)
            {
                score_after_sum += current_score.total;
                continue;
            }
            score_after_sum += best_score.total;
            proposals.push(AssignmentProposal {
                track_index,
                observation_index,
                camera: observation.camera,
                current_pixel: observation.pixel,
                candidate: set.candidates[best_index].clone(),
                score_gain: current_score.total - best_score.total,
                geometry_supported,
                constellation_supported,
                cycle_supported,
            });
        }
    }

    // Enforce the partial-permutation constraint that one localized target
    // landmark can belong to at most one track in a camera.  Independent hard
    // proposals previously allowed two repeated reference structures to
    // collapse onto the same target corner.  Resolve proposals synchronously:
    // current locations of tracks that intend to move are temporarily freed,
    // so genuine swaps/cycles remain possible; competing proposals then win by
    // score improvement.
    let moving = proposals
        .iter()
        .map(|proposal| (proposal.track_index, proposal.observation_index))
        .collect::<HashSet<_>>();
    let mut occupied = vec![Vec::<Vec2>::new(); cameras.len()];
    for (track_index, track) in tracks.iter().enumerate() {
        for (observation_index, observation) in track.observations.iter().enumerate() {
            if !moving.contains(&(track_index, observation_index)) {
                occupied[observation.camera].push(observation.pixel);
            }
        }
    }
    proposals.sort_by(|left, right| right.score_gain.total_cmp(&left.score_gain));
    let collision_radius_sq = options.latent_candidate_dedup_radius_px.powi(2);
    let mut accepted = Vec::with_capacity(proposals.len());
    let mut collision_rejected_switches = 0usize;
    for proposal in proposals {
        let collision = occupied[proposal.camera]
            .iter()
            .any(|&pixel| distance_squared(pixel, proposal.candidate.pixel) <= collision_radius_sq);
        if collision {
            // This proposal stays where it was; restore that location to the
            // occupancy set before evaluating lower-priority proposals. The
            // provisional score-after sum already counted its better candidate,
            // so add the gain back to report the assignment we actually keep.
            occupied[proposal.camera].push(proposal.current_pixel);
            score_after_sum += proposal.score_gain;
            collision_rejected_switches += 1;
            continue;
        }
        occupied[proposal.camera].push(proposal.candidate.pixel);
        accepted.push(proposal);
    }

    let mut geometry_supported_switches = 0usize;
    let mut constellation_supported_switches = 0usize;
    let mut cycle_supported_switches = 0usize;
    for proposal in &accepted {
        let observation =
            &mut tracks[proposal.track_index].observations[proposal.observation_index];
        observation.pixel = proposal.candidate.pixel;
        observation.localization_covariance = proposal.candidate.localization_covariance;
        observation.confidence = proposal.candidate.confidence;
        observation.local_scale = proposal.candidate.local_scale;
        observation.structure = proposal.candidate.structure;
        observation.depth_reliability = None;
        observation.prepared = Default::default();
        tracks[proposal.track_index].condition = f64::NAN;
        tracks[proposal.track_index].max_ray_angle_degrees = f64::NAN;
        if proposal.geometry_supported {
            geometry_supported_switches += 1;
        }
        if proposal.constellation_supported {
            constellation_supported_switches += 1;
        }
        if proposal.cycle_supported {
            cycle_supported_switches += 1;
        }
    }

    let switches = accepted.len();
    if evaluated > 0 {
        report.assignment_iterations += 1;
    }
    report.assignment_switches += switches;
    report.geometry_supported_switches += geometry_supported_switches;
    report.constellation_supported_switches += constellation_supported_switches;
    report.cycle_supported_switches += cycle_supported_switches;
    report.cycle_rejected_switches += cycle_rejected_switches;
    report.cycle_predictions_evaluated += cycle_predictions_evaluated;
    report.collision_rejected_switches += collision_rejected_switches;
    report.rounds.push(LatentMatchRoundReport {
        iteration,
        switches,
        geometry_supported_switches,
        constellation_supported_switches,
        cycle_supported_switches,
        cycle_rejected_switches,
        collision_rejected_switches,
        evaluated_observations: evaluated,
        mean_score_before: if evaluated == 0 {
            f64::NAN
        } else {
            score_before_sum / evaluated as f64
        },
        mean_score_after: if evaluated == 0 {
            f64::NAN
        } else {
            score_after_sum / evaluated as f64
        },
    });
    switches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        calibration::{CameraCalibration, CanonicalPose, IntrinsicsBundle, ModuleState},
        geometry::CameraRefinement,
    };

    fn pairwise_test_camera(name: &str, centre: Vec3) -> (CameraCalibration, ModuleState) {
        let calibration = CameraCalibration {
            name: name.to_owned(),
            intrinsics: vec![IntrinsicsBundle {
                hall_code: Some(0.0),
                focus_distance: 1_000.0,
                k: [
                    [1_000.0, 0.0, 500.0],
                    [0.0, 1_000.0, 400.0],
                    [0.0, 0.0, 1.0],
                ],
            }],
            canonical_pose: Some(CanonicalPose {
                rotation_wc: math::IDENTITY,
                translation_wc: scale(centre, -1.0),
            }),
            ..Default::default()
        };
        let state = ModuleState {
            name: name.to_owned(),
            lens_hall: 0.0,
            mirror_hall: 0.0,
            width: 1_000,
            height: 800,
            gain: 1.0,
            exposure_ns: 1,
            focus: Default::default(),
        };
        (calibration, state)
    }

    #[test]
    fn descriptor_is_gain_and_offset_invariant() {
        let mut a = Plane::new(32, 32);
        let mut b = Plane::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let value = (x * x + 3 * y + x * y) as f32 * 0.01;
                a.data[y * 32 + x] = value;
                b.data[y * 32 + x] = 4.0 * value + 7.0;
            }
        }
        let pixel = [31.0, 31.0];
        let da = descriptor_for_sensor_pixel(&a, pixel).unwrap();
        let db = descriptor_for_sensor_pixel(&b, pixel).unwrap();
        assert!(descriptor_similarity(&da, &db) > 0.9999);
    }

    #[test]
    fn membership_floor_protects_sparse_cameras_without_overpromising_support() {
        let observation = |camera: usize, fixed_gauge: bool, x: f64| TrackObservation {
            camera,
            pixel: [x, 100.0],
            bootstrap_residual_proposal: [0.0, 0.0],
            localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
            fixed_gauge,
            confidence: 1.0,
            local_scale: 1.0,
            structure: 1.0,
            depth_reliability: None,
            prepared: Default::default(),
        };
        let tracks = (0..100)
            .map(|index| {
                let mut observations = vec![
                    observation(0, true, index as f64),
                    observation(1, false, index as f64 + 1.0),
                ];
                if index < 20 {
                    observations.push(observation(2, false, index as f64 + 2.0));
                }
                Track {
                    key: [index, 0],
                    observations,
                    condition: f64::NAN,
                    max_ray_angle_degrees: f64::NAN,
                }
            })
            .collect::<Vec<_>>();
        let options = RigRefinementOptions {
            latent_membership_min_camera_observations: 64,
            latent_membership_min_camera_fraction: 0.35,
            ..Default::default()
        };
        let mut report = LatentMatchReport {
            camera_support: vec![LatentCameraSupportReport::default(); 3],
            ..Default::default()
        };
        let state = LatentMembershipState::new(&tracks, 3, &options, &mut report);
        assert_eq!(state.initial_per_camera, vec![0, 100, 20]);
        assert_eq!(state.floor_per_camera, vec![0, 64, 20]);
        assert_eq!(report.camera_support[1].initial_fit_observations, 100);
        assert_eq!(report.camera_support[1].membership_floor, 64);
        assert_eq!(report.camera_support[2].initial_fit_observations, 20);
        assert_eq!(report.camera_support[2].membership_floor, 20);
    }

    #[test]
    fn intrinsically_pairwise_tracks_demote_and_can_recover() {
        let (calibration0, state0) = pairwise_test_camera("B4", [0.0, 0.0, 0.0]);
        let (calibration1, state1) = pairwise_test_camera("C1", [100.0, 0.0, 0.0]);
        let cameras = vec![
            ResolvedCamera::new(
                &calibration0,
                &state0,
                IntrinsicsMode::Clamp,
                &CameraRefinement::default(),
            )
            .unwrap(),
            ResolvedCamera::new(
                &calibration1,
                &state1,
                IntrinsicsMode::Clamp,
                &CameraRefinement::default(),
            )
            .unwrap(),
        ];
        let point = [40.0, 10.0, 2_000.0];
        let reference_pixel = cameras[0].project(point).unwrap();
        let correct_target_pixel = cameras[1].project(point).unwrap();
        let observation = |camera: usize, pixel: Vec2, fixed_gauge: bool| TrackObservation {
            camera,
            pixel,
            bootstrap_residual_proposal: [0.0, 0.0],
            localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
            fixed_gauge,
            confidence: 1.0,
            local_scale: 1.0,
            structure: 1.0,
            depth_reliability: None,
            prepared: Default::default(),
        };
        let mut tracks = vec![Track {
            key: [1, 2],
            observations: vec![
                observation(0, reference_pixel, true),
                observation(
                    1,
                    [correct_target_pixel[0], correct_target_pixel[1] + 20.0],
                    false,
                ),
            ],
            condition: f64::NAN,
            max_ray_angle_degrees: f64::NAN,
        }];
        let options = RigRefinementOptions {
            strategy: super::super::RigRefinementStrategy::LatentGraph,
            latent_membership_min_camera_observations: 0,
            latent_membership_min_camera_fraction: 0.0,
            latent_pairwise_membership_max_reference_px: 2.5,
            latent_pairwise_membership_recovery_reference_px: 1.5,
            ..Default::default()
        };
        let mut report = LatentMatchReport {
            camera_support: vec![LatentCameraSupportReport::default(); 2],
            ..Default::default()
        };
        let mut membership = LatentMembershipState::new(&tracks, 2, &options, &mut report);
        let first = membership.update(&mut tracks, &cameras, &options, &mut report);
        assert_eq!(first.pairwise_demoted, 1);
        assert!(tracks.is_empty());
        assert_eq!(report.pairwise_track_demotions, 1);

        membership.tracks[0].template.observations[1].pixel = correct_target_pixel;
        let second = membership.update(&mut tracks, &cameras, &options, &mut report);
        assert_eq!(second.pairwise_reactivated, 1);
        assert_eq!(tracks.len(), 1);
        assert_eq!(report.pairwise_track_reactivations, 1);
    }

    #[test]
    fn grouping_merges_the_same_reference_landmark_across_cameras() {
        let correspondence = |reference_pixel: Vec2, target_pixel: Vec2| AlignmentCorrespondence {
            reference_pixel,
            target_pixel,
            confidence: 0.9,
            local_scale: 1.0,
            structure: 0.1,
            reference_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
            target_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
            peak_margin: 0.1,
            forward_backward_error_px: 0.1,
            depth_reliability: None,
        };
        let seeds = vec![
            SeedObservation {
                camera: 1,
                correspondence: correspondence([100.0, 200.0], [80.0, 180.0]),
            },
            SeedObservation {
                camera: 2,
                correspondence: correspondence([100.4, 199.8], [300.0, 140.0]),
            },
        ];
        let groups = group_seed_observations(&seeds, 3, 1.0);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].targets[1].is_some());
        assert!(groups[0].targets[2].is_some());
    }
}
