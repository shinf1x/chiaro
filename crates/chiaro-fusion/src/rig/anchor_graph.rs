//! Classical anchor/constellation correspondence graph used by the
//! `anchor-graph` rig-refinement strategy.
//!
//! Appearance proposes a correspondence but does not establish global identity
//! on its own. Factory geometry first proposes a redundant camera graph from
//! pairs whose overlap covers enough of the smaller field of view. The already
//! available reference-camera matches seed identities, while non-star edges are
//! activated to raise weak-camera degree and close short cycles. Every direct
//! edge is independently verified from image evidence before it can support a
//! physical track. Missing observations and new tracks are then grown in bounded
//! rounds. Every newly observed point must be found in the target image, close
//! under the reverse constellation, and retain an appearance margin over nearby
//! competing corners.
//!
//! From round two onward the graph runs a global structureless bundle solve
//! over every observable camera parameter. Scene landmarks are retriangulated
//! inside the objective (variable projection / Schur-style point elimination),
//! then the improved rig is used only as a proposal: a new observation is still
//! accepted only when local appearance and the independent 2-D constellation
//! agree.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::{
    align::{AlignmentCorrespondence, ModuleAlignment, RigCorner, detect_rig_corners},
    calibration::IntrinsicsMode,
    geometry::ResolvedCamera,
    image::Plane,
    math::{self, Mat3, Vec2, Vec3, mul_vec},
};

use super::{
    ParameterKind, RigCameraInput, RigRefinementOptions, Track, TrackObservation,
    bearing_bootstrap, fallback_epipolar_inliers, filter_observable_parameter_specs,
    is_validation_track, parameter_specs, refinements_from_parameters, remap_parameters,
    resolve_cameras, staged_bundle_optimize_rig, triangulate,
};

const SENSOR_PER_LUMA: f64 = 2.0;
const SENSOR_LUMA_CENTRE: f64 = 0.5;

#[derive(Clone, Debug, Default, Serialize)]
pub struct AnchorGraphRoundReport {
    pub round: usize,
    pub active_edges_before: usize,
    pub activated_edges: usize,
    /// Newly activated edges that were actually searched directly in the
    /// images during this round.
    pub directly_seeded_edges: usize,
    /// High-confidence direct pair correspondences integrated by those seeds.
    pub directly_seeded_matches: usize,
    /// Sparse already-active edges retried under this round's intermediate
    /// geometry rather than the original factory bearing.
    pub geometry_reseeded_edges: usize,
    pub geometry_reseeded_matches: usize,
    pub active_edges_after: usize,
    pub observations_before: usize,
    pub observations_after: usize,
    pub new_observations: usize,
    pub promoted_observations: usize,
    pub new_tracks: usize,
    pub strong_three_plus_before: usize,
    pub strong_three_plus_after: usize,
    pub observation_growth_fraction: f64,
    pub strong_track_growth_fraction: f64,
    pub geometry_projection_proposals: usize,
    pub geometry_projection_consensus: usize,
    /// Scene landmarks used by the per-round global structureless BA.
    pub geometry_fit_tracks: usize,
    /// Bearing-bootstrap + global structureless bundle sweeps run before
    /// this round's 3-D proposals.
    pub geometry_optimizer_iterations: usize,
    /// Number of physical camera parameters jointly present in the global
    /// round solve after observability pruning.  Round one normally keeps only
    /// bearing parameters; later rounds may release observable nuisance DOFs.
    pub geometry_free_parameters: usize,
    /// High-confidence cycle-supported landmarks used as the scene skeleton
    /// for the round solve.
    pub skeleton_landmarks: usize,
    /// Reversible membership diagnostics.  These are observation-level rather
    /// than camera-level decisions.
    pub soft_outlier_observations: usize,
    pub reactivated_observations: usize,
    pub hard_rejected_observations: usize,
    pub split_tracks: usize,
    pub closed_cycle_edges: usize,
    pub stopped_early: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AnchorGraphPairReport {
    pub first_camera: String,
    pub second_camera: String,
    pub anchors: usize,
    pub validated_anchors: usize,
    pub loo_rms_px: f64,
    pub loo_median_px: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AnchorGraphEdgeReport {
    pub first_camera: String,
    pub second_camera: String,
    /// Factory overlap relative to the smaller field of view.
    pub factory_overlap: f64,
    pub factory_baseline: f64,
    pub candidate: bool,
    pub active: bool,
    /// Zero means active in the bootstrap graph; positive values indicate the
    /// propagation round after which the edge was activated.
    pub activated_round: Option<usize>,
    /// Number of physical tracks that currently contain observations from both
    /// cameras, regardless of promotion. This is proposal support only.
    pub shared_tracks: usize,
    /// Number of physical tracks with independently verified image evidence
    /// on this camera edge.
    pub direct_support_tracks: usize,
    /// Whether factory-guided direct image matching was attempted for this
    /// edge, and how many mutually verified seeds it contributed.
    pub direct_seed_attempted: bool,
    pub direct_seed_attempts: usize,
    pub direct_seed_matches: usize,
    pub validated_anchors: usize,
    pub loo_rms_px: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AnchorGraphCameraReport {
    pub camera: String,
    pub active_degree: usize,
    /// Active incident edges that have at least one independently verified
    /// physical track.
    pub verified_degree: usize,
    /// Number of validated incident edges that participate in at least one
    /// validated triangle through this camera.
    pub triangle_edges: usize,
    pub promoted_observations: usize,
    pub occupied_coverage_cells: usize,
    pub spatial_coverage_fraction: f64,
    pub direct_support_tracks: usize,
    pub cycle_supported_tracks: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AnchorGraphReport {
    pub factory_overlap_threshold: f64,
    pub candidate_edges: usize,
    pub initial_active_edges: usize,
    pub final_active_edges: usize,
    pub maximum_active_edges: usize,
    pub target_min_camera_degree: usize,
    pub bootstrap_pairs: usize,
    pub bootstrap_matches_before_constellation: usize,
    pub bootstrap_anchors: usize,
    pub bootstrap_rejected_constellation: usize,
    /// Active edges that required an explicit factory-guided image search
    /// during bootstrap, plus the mutually verified seeds they contributed.
    pub bootstrap_direct_seed_edges: usize,
    pub bootstrap_direct_seed_matches: usize,
    pub initial_tracks: usize,
    pub initial_three_plus_tracks: usize,
    pub initial_cycle_supported_three_plus_tracks: usize,
    pub rounds_run: usize,
    pub stopped_early: bool,
    pub final_tracks: usize,
    pub final_three_plus_tracks: usize,
    pub final_cycle_supported_three_plus_tracks: usize,
    pub final_observations: usize,
    pub propagated_observations: usize,
    pub promoted_observations: usize,
    pub spawned_tracks: usize,
    pub final_soft_outlier_observations: usize,
    pub final_hard_rejected_observations: usize,
    pub split_tracks: usize,
    pub rounds: Vec<AnchorGraphRoundReport>,
    pub pairs: Vec<AnchorGraphPairReport>,
    pub graph_edges: Vec<AnchorGraphEdgeReport>,
    pub graph_cameras: Vec<AnchorGraphCameraReport>,
}

pub(crate) struct AnchorGraphBuild {
    pub tracks: Vec<Track>,
    pub pairwise_matches: usize,
    pub report: AnchorGraphReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObservationMembership {
    /// Directly or cycle-verified and currently trusted by the global solve.
    Inlier,
    /// Useful as an identity proposal but not yet strong enough for bundle
    /// adjustment.
    Provisional,
    /// Temporarily inconsistent with the rest of the scene.  This state is
    /// reversible: a later global solve may reactivate the observation.
    SoftOutlier,
}

#[derive(Clone, Debug)]
struct GraphObservation {
    observation: TrackObservation,
    promoted: bool,
    #[allow(dead_code)]
    round: usize,
    membership: ObservationMembership,
    bad_iterations: usize,
    good_iterations: usize,
    loo_reprojection_px: Option<f64>,
    constellation_loo_px: Option<f64>,
    identity_confidence: f64,
}

fn graph_observation(
    observation: TrackObservation,
    promoted: bool,
    round: usize,
    identity_confidence: f64,
) -> GraphObservation {
    GraphObservation {
        observation,
        promoted,
        round,
        membership: if promoted {
            ObservationMembership::Inlier
        } else {
            ObservationMembership::Provisional
        },
        bad_iterations: 0,
        good_iterations: 0,
        loo_reprojection_px: None,
        constellation_loo_px: None,
        identity_confidence: identity_confidence.clamp(0.0, 1.0),
    }
}

fn confirm_observation(observation: &mut GraphObservation) {
    // Soft outliers are deliberately sticky for one or more global geometry
    // updates. Pair seeding/cycle closure may keep supplying fresh direct
    // evidence, but only the explicit LOO membership update is allowed to
    // erase a bad streak and reactivate the measurement.
    if observation.membership == ObservationMembership::SoftOutlier {
        return;
    }
    observation.promoted = true;
    observation.membership = ObservationMembership::Inlier;
    observation.bad_iterations = 0;
    observation.good_iterations = observation.good_iterations.saturating_add(1);
}

#[derive(Clone, Debug)]
struct GraphTrack {
    key: [i32; 2],
    observations: Vec<GraphObservation>,
    /// Direct image evidence. Transitive membership does not create an edge.
    direct_edges: HashSet<(usize, usize)>,
}

#[derive(Clone, Copy, Debug)]
struct PairAnchor {
    track: usize,
    first: Vec2,
    second: Vec2,
}

#[derive(Clone, Copy, Debug)]
struct Affine2 {
    x: [f64; 3],
    y: [f64; 3],
}

impl Affine2 {
    fn apply(&self, point: Vec2) -> Vec2 {
        [
            self.x[0] * point[0] + self.x[1] * point[1] + self.x[2],
            self.y[0] * point[0] + self.y[1] * point[1] + self.y[2],
        ]
    }

    fn jacobian(&self) -> [[f64; 2]; 2] {
        [[self.x[0], self.x[1]], [self.y[0], self.y[1]]]
    }

    fn local_scale(&self) -> f64 {
        let jacobian = self.jacobian();
        (jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0])
            .abs()
            .sqrt()
            .max(1.0e-6)
    }
}

#[derive(Clone, Debug)]
struct PairModel {
    first: usize,
    second: usize,
    anchors: Vec<PairAnchor>,
    loo_rms: f64,
    loo_median: f64,
}

#[derive(Clone, Copy, Debug)]
struct FactoryEdge {
    first: usize,
    second: usize,
    overlap: f64,
    baseline: f64,
}

#[derive(Clone, Copy, Debug)]
struct CandidateObservation {
    pixel: Vec2,
    covariance: [[f64; 2]; 2],
    structure: f64,
    score: f64,
    margin: f64,
    closure_error: f64,
    /// Error of the *relative* arrangement to independently established
    /// neighbouring anchors.  Unlike one local patch, this is difficult for a
    /// repeated facade/window/railing element to fake.
    constellation_error: f64,
    /// Number of candidates inside the camera-specific search domain whose
    /// appearance is close enough to the winner to remain plausible.
    conditional_competitors: usize,
    /// Combined local identity confidence.  This is deliberately not a
    /// camera-wide reliability value.
    identity_confidence: f64,
    local_scale: f64,
}

fn ordered_pair(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

fn edge_degree(active: &HashSet<(usize, usize)>, camera: usize) -> usize {
    active
        .iter()
        .filter(|&&(first, second)| first == camera || second == camera)
        .count()
}

fn active_neighbours(active: &HashSet<(usize, usize)>, camera: usize) -> HashSet<usize> {
    active
        .iter()
        .filter_map(|&(first, second)| {
            if first == camera {
                Some(second)
            } else if second == camera {
                Some(first)
            } else {
                None
            }
        })
        .collect()
}

fn common_active_neighbours(
    active: &HashSet<(usize, usize)>,
    first: usize,
    second: usize,
) -> usize {
    let first_neighbours = active_neighbours(active, first);
    active_neighbours(active, second)
        .iter()
        .filter(|camera| first_neighbours.contains(camera))
        .count()
}

fn active_connected(active: &HashSet<(usize, usize)>, first: usize, second: usize) -> bool {
    if first == second {
        return true;
    }
    let mut stack = vec![first];
    let mut visited = HashSet::new();
    while let Some(camera) = stack.pop() {
        if !visited.insert(camera) {
            continue;
        }
        for neighbour in active_neighbours(active, camera) {
            if neighbour == second {
                return true;
            }
            if !visited.contains(&neighbour) {
                stack.push(neighbour);
            }
        }
    }
    false
}

fn triangle_edges_for_camera(active: &HashSet<(usize, usize)>, camera: usize) -> usize {
    let neighbours = active_neighbours(active, camera);
    neighbours
        .iter()
        .filter(|&&neighbour| {
            neighbours.iter().any(|&other| {
                other != neighbour && active.contains(&ordered_pair(neighbour, other))
            })
        })
        .count()
}

fn baseline(first: &ResolvedCamera, second: &ResolvedCamera) -> f64 {
    let a = first.center();
    let b = second.center();
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn overlap_depths(options: &RigRefinementOptions) -> [f64; 5] {
    let near = options.physical_match_near_depth.max(1000.0);
    let far = options.physical_match_far_depth.max(near * 4.0);
    let mid1 = (near * far).sqrt();
    let mid0 = (near * mid1).sqrt();
    let mid2 = (mid1 * far).sqrt();
    [near, mid0, mid1, mid2, far]
}

fn direct_seed_depths(options: &RigRefinementOptions) -> Vec<f64> {
    // A denser logarithmic sampling than the cheap overlap test.  With the
    // default 0.5 m .. 10 km range, nine samples include ~20 m and ~70 m,
    // which matters for outdoor L16 captures where parallax at 25-50 m is
    // still tens of sensor pixels.
    const SAMPLES: usize = 9;
    let near = options.physical_match_near_depth.max(500.0);
    let far = options.physical_match_far_depth.max(near * 4.0);
    let ratio = (far / near).powf(1.0 / (SAMPLES - 1) as f64);
    (0..SAMPLES)
        .map(|index| near * ratio.powi(index as i32))
        .collect()
}

fn directional_factory_overlap(
    source: &ResolvedCamera,
    target: &ResolvedCamera,
    options: &RigRefinementOptions,
) -> f64 {
    const GRID_X: usize = 12;
    const GRID_Y: usize = 9;
    let depths = overlap_depths(options);
    let mut visible = 0usize;
    let mut tested = 0usize;
    for gy in 0..GRID_Y {
        for gx in 0..GRID_X {
            let pixel = [
                (gx as f64 + 0.5) * source.width as f64 / GRID_X as f64 - 0.5,
                (gy as f64 + 0.5) * source.height as f64 / GRID_Y as f64 - 0.5,
            ];
            tested += 1;
            if depths.iter().any(|&depth| {
                target
                    .map_from(source, pixel, depth)
                    .is_some_and(|mapped| target.contains(mapped))
            }) {
                visible += 1;
            }
        }
    }
    if tested == 0 {
        0.0
    } else {
        visible as f64 / tested as f64
    }
}

/// Approximate overlap relative to the smaller field of view.  A narrow C
/// camera that lies almost completely inside a B camera should therefore score
/// near one even though the same region occupies only a fraction of B.
fn factory_overlap(
    first: &ResolvedCamera,
    second: &ResolvedCamera,
    options: &RigRefinementOptions,
) -> f64 {
    directional_factory_overlap(first, second, options)
        .max(directional_factory_overlap(second, first, options))
        .clamp(0.0, 1.0)
}

fn factory_edges(
    cameras: &[RigCameraInput<'_>],
    resolved: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Vec<FactoryEdge> {
    let mut edges = Vec::new();
    for first in 0..cameras.len() {
        if !cameras[first].match_evidence_enabled
            || cameras[first].calibration.is_none()
            || cameras[first].luminance.is_none()
        {
            continue;
        }
        for second in first + 1..cameras.len() {
            if !cameras[second].match_evidence_enabled
                || cameras[second].calibration.is_none()
                || cameras[second].luminance.is_none()
            {
                continue;
            }
            let overlap = factory_overlap(&resolved[first], &resolved[second], options);
            edges.push(FactoryEdge {
                first,
                second,
                overlap,
                baseline: baseline(&resolved[first], &resolved[second]),
            });
        }
    }
    edges.sort_by(|a, b| {
        b.overlap
            .total_cmp(&a.overlap)
            .then_with(|| b.baseline.total_cmp(&a.baseline))
    });
    edges
}

fn camera_promoted_support(tracks: &[GraphTrack], camera_count: usize) -> Vec<usize> {
    let mut support = vec![0usize; camera_count];
    for track in tracks {
        for observation in &track.observations {
            if observation.promoted {
                support[observation.observation.camera] += 1;
            }
        }
    }
    support
}

fn camera_spatial_coverage(
    tracks: &[GraphTrack],
    cameras: &[RigCameraInput<'_>],
) -> Vec<(usize, f64)> {
    const CELLS_X: usize = 8;
    const CELLS_Y: usize = 6;
    let mut occupied = vec![HashSet::<usize>::new(); cameras.len()];
    for track in tracks {
        for observation in &track.observations {
            if !observation.promoted {
                continue;
            }
            let camera = observation.observation.camera;
            let Some(plane) = cameras.get(camera).and_then(|camera| camera.luminance) else {
                continue;
            };
            let width = (plane.width * 2).max(1) as f64;
            let height = (plane.height * 2).max(1) as f64;
            let x = ((observation.observation.pixel[0] / width) * CELLS_X as f64)
                .floor()
                .clamp(0.0, (CELLS_X - 1) as f64) as usize;
            let y = ((observation.observation.pixel[1] / height) * CELLS_Y as f64)
                .floor()
                .clamp(0.0, (CELLS_Y - 1) as f64) as usize;
            occupied[camera].insert(y * CELLS_X + x);
        }
    }
    occupied
        .into_iter()
        .map(|cells| {
            let count = cells.len();
            (count, count as f64 / (CELLS_X * CELLS_Y) as f64)
        })
        .collect()
}

fn edge_selection_score(
    edge: FactoryEdge,
    active: &HashSet<(usize, usize)>,
    support: &[usize],
    coverage: &[f64],
    verified_degree: &[usize],
    shared_support: &HashMap<(usize, usize), usize>,
    min_degree: usize,
) -> f64 {
    let degree_first = edge_degree(active, edge.first);
    let degree_second = edge_degree(active, edge.second);
    let degree_deficit =
        min_degree.saturating_sub(degree_first) + min_degree.saturating_sub(degree_second);
    let verified_target = min_degree.saturating_sub(1).max(1);
    let verified_deficit = verified_target
        .saturating_sub(verified_degree.get(edge.first).copied().unwrap_or(0))
        + verified_target.saturating_sub(verified_degree.get(edge.second).copied().unwrap_or(0));
    let support_max = support.iter().copied().max().unwrap_or(1).max(1) as f64;
    let weak_first = 1.0 - support.get(edge.first).copied().unwrap_or(0) as f64 / support_max;
    let weak_second = 1.0 - support.get(edge.second).copied().unwrap_or(0) as f64 / support_max;
    let coverage_deficit = (1.0 - coverage.get(edge.first).copied().unwrap_or(0.0))
        + (1.0 - coverage.get(edge.second).copied().unwrap_or(0.0));
    let first_neighbours = active_neighbours(active, edge.first);
    let second_neighbours = active_neighbours(active, edge.second);
    let common_neighbours = first_neighbours
        .intersection(&second_neighbours)
        .copied()
        .collect::<Vec<_>>();
    let cycle_bonus = common_neighbours.len() as f64;
    let triangle_rescue = if common_neighbours.is_empty() {
        0.0
    } else {
        (if triangle_edges_for_camera(active, edge.first) == 0 {
            1.0
        } else {
            0.0
        }) + (if triangle_edges_for_camera(active, edge.second) == 0 {
            1.0
        } else {
            0.0
        }) + 0.5
            * common_neighbours
                .iter()
                .filter(|&&camera| triangle_edges_for_camera(active, camera) == 0)
                .count() as f64
    };
    let connects_components = !active_connected(active, edge.first, edge.second);
    let hub_penalty = (degree_first + degree_second) as f64;
    let shared = shared_support
        .get(&ordered_pair(edge.first, edge.second))
        .copied()
        .unwrap_or(0);
    // Existing shared observations are only proposal evidence, but they make
    // an edge much cheaper and safer to verify empirically. Prefer them when
    // rescuing a weak camera instead of repeatedly guessing arbitrary factory
    // neighbours.
    let shared_bonus = 30.0 * (shared as f64 + 1.0).ln();
    // First make one connected graph, then satisfy camera degree, then spend
    // the remaining budget on short cycles and weak/spatially uncovered
    // cameras.  This avoids both the old reference-star topology and a set of
    // high-overlap but disconnected cliques.
    let connectivity_bonus = if connects_components { 1000.0 } else { 0.0 };
    connectivity_bonus
        + 180.0 * degree_deficit as f64
        + 220.0 * verified_deficit as f64
        + 25.0 * edge.overlap
        + shared_bonus
        + 35.0 * cycle_bonus
        + 180.0 * triangle_rescue
        + 8.0 * (weak_first + weak_second)
        + 12.0 * coverage_deficit
        + 0.01 * edge.baseline.min(100.0)
        - 1.5 * hub_penalty
}

fn activate_best_edges(
    candidates: &[FactoryEdge],
    active: &mut HashSet<(usize, usize)>,
    support: &[usize],
    coverage: &[f64],
    verified_degree: &[usize],
    shared_support: &HashMap<(usize, usize), usize>,
    target_count: usize,
    min_degree: usize,
    maximum_new: usize,
) -> Vec<(usize, usize)> {
    let mut activated = Vec::new();
    while active.len() < target_count && activated.len() < maximum_new {
        let best = candidates
            .iter()
            .copied()
            .filter(|edge| !active.contains(&ordered_pair(edge.first, edge.second)))
            .max_by(|a, b| {
                edge_selection_score(
                    *a,
                    active,
                    support,
                    coverage,
                    verified_degree,
                    shared_support,
                    min_degree,
                )
                .total_cmp(&edge_selection_score(
                    *b,
                    active,
                    support,
                    coverage,
                    verified_degree,
                    shared_support,
                    min_degree,
                ))
            });
        let Some(edge) = best else {
            break;
        };
        let pair = ordered_pair(edge.first, edge.second);
        active.insert(pair);
        activated.push(pair);
    }
    activated
}

fn direct_support_tracks(tracks: &[GraphTrack], edge: (usize, usize)) -> usize {
    tracks
        .iter()
        .filter(|track| track.direct_edges.contains(&edge))
        .count()
}

fn shared_track_support(tracks: &[GraphTrack], edge: (usize, usize)) -> usize {
    tracks
        .iter()
        .filter(|track| {
            observation_for_camera(track, edge.0).is_some()
                && observation_for_camera(track, edge.1).is_some()
        })
        .count()
}

fn candidate_shared_support(
    tracks: &[GraphTrack],
    candidates: &[FactoryEdge],
) -> HashMap<(usize, usize), usize> {
    candidates
        .iter()
        .map(|edge| {
            let pair = ordered_pair(edge.first, edge.second);
            (pair, shared_track_support(tracks, pair))
        })
        .collect()
}

fn camera_verified_degrees(
    tracks: &[GraphTrack],
    active: &HashSet<(usize, usize)>,
    camera_count: usize,
    options: &RigRefinementOptions,
) -> Vec<usize> {
    let mut degrees = vec![0usize; camera_count];
    for &(first, second) in active {
        // `direct_edges` are not raw transitive hypotheses: an edge enters a
        // track only after independent two-image evidence (bootstrap
        // constellation validation, or free epipolar + photometric
        // verification).  The old definition additionally required a single
        // local-affine field to predict the whole pair.  That works for B/B,
        // but incorrectly labelled real B/C and C/C edges as "unverified"
        // whenever depth/parallax makes one affine field a poor model.  It
        // caused the scheduler to spend almost the entire edge budget trying
        // to "rescue" already well-observed C2/C4 while leaving C1 at degree
        // two.  Count an edge as directly verified once enough independent
        // per-track image evidence exists; pair-field LOO remains a stronger
        // diagnostic used for propagation/spawning, not a topology gate.
        if direct_support_tracks(tracks, ordered_pair(first, second))
            < options.anchor_min_pair_anchors
        {
            continue;
        }
        degrees[first] += 1;
        degrees[second] += 1;
    }
    degrees
}

fn squared_distance(first: Vec2, second: Vec2) -> f64 {
    let dx = first[0] - second[0];
    let dy = first[1] - second[1];
    dx * dx + dy * dy
}

fn distance(first: Vec2, second: Vec2) -> f64 {
    squared_distance(first, second).sqrt()
}

fn sensor_to_luma(pixel: Vec2) -> Vec2 {
    [
        (pixel[0] - SENSOR_LUMA_CENTRE) / SENSOR_PER_LUMA,
        (pixel[1] - SENSOR_LUMA_CENTRE) / SENSOR_PER_LUMA,
    ]
}

fn luma_to_sensor(pixel: [f32; 2]) -> Vec2 {
    [
        SENSOR_PER_LUMA * f64::from(pixel[0]) + SENSOR_LUMA_CENTRE,
        SENSOR_PER_LUMA * f64::from(pixel[1]) + SENSOR_LUMA_CENTRE,
    ]
}

fn corner_sensor(corner: &RigCorner) -> Vec2 {
    luma_to_sensor(corner.subpixel)
}

/// Cheap whole-image distinctiveness signature used only to *prioritize*
/// bootstrap markers.  It is intentionally not an acceptance gate: a feature
/// that repeats globally in a wide B view may still be unique inside a narrow
/// tele-camera search domain.  Final identity therefore remains conditional on
/// constellation + camera-specific competition.
fn corner_signature(plane: &Plane, corner: &RigCorner) -> Option<u16> {
    const OFFSETS: [[f64; 2]; 12] = [
        [-4.0, 0.0],
        [-3.0, -3.0],
        [0.0, -4.0],
        [3.0, -3.0],
        [4.0, 0.0],
        [3.0, 3.0],
        [0.0, 4.0],
        [-3.0, 3.0],
        [-2.0, 0.0],
        [0.0, -2.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ];
    let centre = sensor_to_luma(corner_sensor(corner));
    let mut values = [0.0f64; OFFSETS.len()];
    for (index, offset) in OFFSETS.iter().enumerate() {
        values[index] = sample_plane(plane, [centre[0] + offset[0], centre[1] + offset[1]])?;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut signature = 0u16;
    for (index, value) in values.iter().enumerate() {
        if *value >= mean {
            signature |= 1u16 << index;
        }
    }
    Some(signature)
}

fn corner_global_uniqueness(corners: &[RigCorner], plane: &Plane) -> Vec<f64> {
    let signatures = corners
        .iter()
        .map(|corner| corner_signature(plane, corner))
        .collect::<Vec<_>>();
    let mut counts = HashMap::<u16, usize>::new();
    for signature in signatures.iter().flatten() {
        *counts.entry(*signature).or_insert(0) += 1;
    }
    signatures
        .into_iter()
        .map(|signature| {
            signature
                .and_then(|signature| counts.get(&signature).copied())
                .map(|count| 1.0 / (count as f64).sqrt())
                .unwrap_or(0.0)
        })
        .collect()
}

type CornerGrid = HashMap<(i32, i32), Vec<usize>>;

#[derive(Clone, Copy, Debug)]
struct DirectSeedChoice {
    other: usize,
    score: f64,
    margin: f64,
    prediction_error: f64,
    local_scale: f64,
}

#[derive(Clone, Copy, Debug)]
struct DirectSeedMatch {
    first_corner: usize,
    second_corner: usize,
    confidence: f64,
    local_scale: f64,
}

fn corner_grid(corners: &[RigCorner], cell_size: f64) -> CornerGrid {
    let cell_size = cell_size.max(8.0);
    let mut grid = HashMap::<(i32, i32), Vec<usize>>::new();
    for (index, corner) in corners.iter().enumerate() {
        let pixel = corner_sensor(corner);
        let key = (
            (pixel[0] / cell_size).floor() as i32,
            (pixel[1] / cell_size).floor() as i32,
        );
        grid.entry(key).or_default().push(index);
    }
    grid
}

fn nearby_corner_indices(
    grid: &CornerGrid,
    centre: Vec2,
    radius: f64,
    cell_size: f64,
) -> Vec<usize> {
    let cell_size = cell_size.max(8.0);
    let min_x = ((centre[0] - radius) / cell_size).floor() as i32;
    let max_x = ((centre[0] + radius) / cell_size).floor() as i32;
    let min_y = ((centre[1] - radius) / cell_size).floor() as i32;
    let max_y = ((centre[1] + radius) / cell_size).floor() as i32;
    let mut indices = Vec::new();
    for gy in min_y..=max_y {
        for gx in min_x..=max_x {
            if let Some(cell) = grid.get(&(gx, gy)) {
                indices.extend(cell.iter().copied());
            }
        }
    }
    indices
}

/// Local differential of the exact factory projection at one depth.  Using
/// the physical camera model here gives B<->C patch comparison the correct
/// magnification and local shear without cropping/resizing either image.
fn factory_local_affine(
    source: &ResolvedCamera,
    target: &ResolvedCamera,
    source_pixel: Vec2,
    depth: f64,
) -> Option<Affine2> {
    const STEP: f64 = 12.0;
    let offsets = [
        [0.0, 0.0],
        [STEP, 0.0],
        [-STEP, 0.0],
        [0.0, STEP],
        [0.0, -STEP],
    ];
    let mut samples = Vec::with_capacity(offsets.len());
    for offset in offsets {
        let source_sample = [source_pixel[0] + offset[0], source_pixel[1] + offset[1]];
        if !source.contains(source_sample) {
            continue;
        }
        let Some(target_sample) = target.map_from(source, source_sample, depth) else {
            continue;
        };
        samples.push((source_sample, target_sample, 1.0));
    }
    fit_weighted_affine(&samples)
}

#[allow(clippy::too_many_arguments)]
fn factory_seed_choices(
    source_camera: usize,
    target_camera: usize,
    corners: &[Vec<RigCorner>],
    cameras: &[RigCameraInput<'_>],
    resolved: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Vec<Option<DirectSeedChoice>> {
    let source_corners = &corners[source_camera];
    let target_corners = &corners[target_camera];
    let mut result = vec![None; source_corners.len()];
    let Some(source_plane) = cameras[source_camera].luminance else {
        return result;
    };
    let Some(target_plane) = cameras[target_camera].luminance else {
        return result;
    };
    if source_corners.is_empty() || target_corners.is_empty() {
        return result;
    }

    let radius = options.anchor_direct_seed_search_radius_px.max(8.0);
    let cell_size = radius.max(32.0);
    let target_grid = corner_grid(target_corners, cell_size);
    let depths = direct_seed_depths(options);
    let maximum_sources = options
        .anchor_direct_seed_max_corners
        .min(source_corners.len());
    let source_uniqueness = corner_global_uniqueness(source_corners, source_plane);
    let mut source_order = (0..source_corners.len()).collect::<Vec<_>>();
    source_order.sort_by(|&first, &second| {
        source_uniqueness[second]
            .total_cmp(&source_uniqueness[first])
            .then_with(|| {
                f64::from(source_corners[second].structure)
                    .total_cmp(&f64::from(source_corners[first].structure))
            })
    });
    source_order.truncate(maximum_sources);

    for source_index in source_order {
        let source_corner = &source_corners[source_index];
        let source_pixel = corner_sensor(source_corner);
        // A candidate corner can be reached by more than one depth sample.
        // Keep only the strongest depth-specific patch warp for that identity.
        let mut candidates = HashMap::<usize, (f64, f64, f64, f64)>::new();
        for &depth in &depths {
            let Some(predicted) =
                resolved[target_camera].map_from(&resolved[source_camera], source_pixel, depth)
            else {
                continue;
            };
            if predicted[0] < -radius
                || predicted[1] < -radius
                || predicted[0] > resolved[target_camera].width as f64 - 1.0 + radius
                || predicted[1] > resolved[target_camera].height as f64 - 1.0 + radius
            {
                continue;
            }
            let Some(affine) = factory_local_affine(
                &resolved[source_camera],
                &resolved[target_camera],
                source_pixel,
                depth,
            ) else {
                continue;
            };
            for target_index in nearby_corner_indices(&target_grid, predicted, radius, cell_size) {
                let target_pixel = corner_sensor(&target_corners[target_index]);
                let prediction_error = distance(target_pixel, predicted);
                if prediction_error > radius {
                    continue;
                }
                let Some(score) = warped_zncc(
                    source_plane,
                    target_plane,
                    source_pixel,
                    target_pixel,
                    affine,
                    options.anchor_patch_radius_luma,
                ) else {
                    continue;
                };
                if !score.is_finite() || score < options.anchor_direct_seed_min_zncc - 0.08 {
                    continue;
                }
                let ranked = score - 0.03 * prediction_error / radius;
                let entry = candidates.entry(target_index).or_insert((
                    ranked,
                    score,
                    prediction_error,
                    affine.local_scale(),
                ));
                if ranked > entry.0 {
                    *entry = (ranked, score, prediction_error, affine.local_scale());
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }
        let mut ranked = candidates
            .into_iter()
            .map(
                |(target_index, (ranked, score, prediction_error, local_scale))| {
                    (ranked, score, prediction_error, local_scale, target_index)
                },
            )
            .collect::<Vec<_>>();
        ranked.sort_by(|first, second| second.0.total_cmp(&first.0));
        let best = ranked[0];
        let second_score = ranked.get(1).map_or(-1.0, |candidate| candidate.1);
        let margin = best.1 - second_score;
        if best.1 < options.anchor_direct_seed_min_zncc
            || margin < options.anchor_direct_seed_min_margin
        {
            continue;
        }
        result[source_index] = Some(DirectSeedChoice {
            other: best.4,
            score: best.1,
            margin,
            prediction_error: best.2,
            local_scale: best.3,
        });
    }
    result
}

fn direct_pair_seed_matches(
    first: usize,
    second: usize,
    corners: &[Vec<RigCorner>],
    cameras: &[RigCameraInput<'_>],
    resolved: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Vec<DirectSeedMatch> {
    let forward = factory_seed_choices(first, second, corners, cameras, resolved, options);
    let reverse = factory_seed_choices(second, first, corners, cameras, resolved, options);
    let mut matches = Vec::<DirectSeedMatch>::new();
    for (first_corner, choice) in forward.iter().enumerate() {
        let Some(choice) = choice else {
            continue;
        };
        let Some(reverse_choice) = reverse.get(choice.other).and_then(Option::as_ref) else {
            continue;
        };
        if reverse_choice.other != first_corner {
            continue;
        }
        // Both searches must independently prefer the same physical corners.
        // The weak prediction-error term only breaks equally photometric peaks;
        // it never overrides the mutual image-space identity test.
        let confidence = choice.score.min(reverse_choice.score)
            - 0.002 * (choice.prediction_error + reverse_choice.prediction_error)
                / options.anchor_direct_seed_search_radius_px.max(1.0);
        let margin = choice.margin.min(reverse_choice.margin);
        if confidence < options.anchor_direct_seed_min_zncc
            || margin < options.anchor_direct_seed_min_margin
        {
            continue;
        }
        matches.push(DirectSeedMatch {
            first_corner,
            second_corner: choice.other,
            confidence,
            local_scale: choice.local_scale,
        });
    }
    matches.sort_by(|first, second| second.confidence.total_cmp(&first.confidence));
    matches.truncate(options.anchor_direct_seed_max_matches);

    // Appearance+mutuality gives point identity candidates.  Before those
    // candidates can establish a camera edge, require the same leave-one-out
    // local-constellation consistency used by the rest of the graph.
    let anchors = matches
        .iter()
        .enumerate()
        .map(|(track, matched)| PairAnchor {
            track,
            first: corner_sensor(&corners[first][matched.first_corner]),
            second: corner_sensor(&corners[second][matched.second_corner]),
        })
        .collect::<Vec<_>>();
    let Some(validated) = validate_pair_anchors(
        anchors,
        options.anchor_neighbour_count,
        options.anchor_seed_loo_max_error_px * 1.25,
        options.anchor_min_pair_anchors,
    ) else {
        return Vec::new();
    };
    let keep = validated
        .anchors
        .iter()
        .map(|anchor| anchor.track)
        .collect::<HashSet<_>>();
    matches
        .into_iter()
        .enumerate()
        .filter_map(|(index, matched)| keep.contains(&index).then_some(matched))
        .collect()
}

fn nearest_track_observing(
    tracks: &[GraphTrack],
    camera: usize,
    pixel: Vec2,
    radius: f64,
) -> Option<usize> {
    let radius_sq = radius * radius;
    let mut best = None::<(f64, usize)>;
    let mut second = None::<f64>;
    for (index, track) in tracks.iter().enumerate() {
        let Some(observation) = observation_for_camera(track, camera) else {
            continue;
        };
        let distance_sq = squared_distance(observation.observation.pixel, pixel);
        if distance_sq > radius_sq {
            continue;
        }
        if best.is_none_or(|current| distance_sq < current.0) {
            second = best.map(|current| current.0);
            best = Some((distance_sq, index));
        } else if second.is_none_or(|current| distance_sq < current) {
            second = Some(distance_sq);
        }
    }
    let (best_distance, best_index) = best?;
    // If two existing physical tracks claim effectively the same image point,
    // do not let a new pair edge arbitrarily union them.
    if second.is_some_and(|second_distance| second_distance.sqrt() - best_distance.sqrt() < 0.75) {
        return None;
    }
    Some(best_index)
}

fn tracks_merge_compatible(first: &GraphTrack, second: &GraphTrack, radius: f64) -> bool {
    for observation in &second.observations {
        if let Some(existing) = observation_for_camera(first, observation.observation.camera)
            && distance(existing.observation.pixel, observation.observation.pixel) > radius
        {
            return false;
        }
    }
    true
}

fn merge_graph_tracks(tracks: &mut [GraphTrack], keep: usize, drop: usize, radius: f64) -> bool {
    if keep == drop {
        return true;
    }
    if !tracks_merge_compatible(&tracks[keep], &tracks[drop], radius)
        || !tracks_merge_compatible(&tracks[drop], &tracks[keep], radius)
    {
        return false;
    }
    let observations = tracks[drop].observations.clone();
    let direct_edges = tracks[drop].direct_edges.clone();
    for observation in observations {
        if let Some(existing) =
            observation_for_camera_mut(&mut tracks[keep], observation.observation.camera)
        {
            if observation.promoted {
                confirm_observation(existing);
            }
            existing.identity_confidence = existing
                .identity_confidence
                .max(observation.identity_confidence);
            if observation.observation.confidence > existing.observation.confidence {
                existing.observation = observation.observation;
            }
        } else {
            tracks[keep].observations.push(observation);
        }
    }
    tracks[keep].direct_edges.extend(direct_edges);
    tracks[drop].observations.clear();
    tracks[drop].direct_edges.clear();
    true
}

#[allow(clippy::too_many_arguments)]
fn integrate_direct_seed_matches(
    tracks: &mut Vec<GraphTrack>,
    first: usize,
    second: usize,
    matches: &[DirectSeedMatch],
    corners: &[Vec<RigCorner>],
    reference_index: usize,
    round: usize,
    options: &RigRefinementOptions,
) -> usize {
    let edge = ordered_pair(first, second);
    let merge_radius = options.anchor_collision_radius_px.max(2.0);
    let mut integrated = 0usize;

    for matched in matches {
        let first_corner = &corners[first][matched.first_corner];
        let second_corner = &corners[second][matched.second_corner];
        let first_pixel = corner_sensor(first_corner);
        let second_pixel = corner_sensor(second_corner);
        let first_track = nearest_track_observing(tracks, first, first_pixel, merge_radius);
        let second_track = nearest_track_observing(tracks, second, second_pixel, merge_radius);

        let track_index = match (first_track, second_track) {
            (Some(first_track), Some(second_track)) if first_track == second_track => first_track,
            (Some(first_track), Some(second_track)) => {
                if merge_graph_tracks(tracks, first_track, second_track, merge_radius) {
                    first_track
                } else {
                    continue;
                }
            }
            (Some(track), None) | (None, Some(track)) => track,
            (None, None) => {
                let mut direct_edges = HashSet::new();
                direct_edges.insert(edge);
                tracks.push(GraphTrack {
                    key: track_key(if first == reference_index {
                        first_pixel
                    } else if second == reference_index {
                        second_pixel
                    } else {
                        first_pixel
                    }),
                    observations: Vec::new(),
                    direct_edges,
                });
                tracks.len() - 1
            }
        };

        if observation_for_camera(&tracks[track_index], first).is_some_and(|existing| {
            distance(existing.observation.pixel, first_pixel) > merge_radius
        }) || observation_for_camera(&tracks[track_index], second).is_some_and(|existing| {
            distance(existing.observation.pixel, second_pixel) > merge_radius
        }) {
            continue;
        }

        let first_observation = graph_observation(
            correspondence_observation(
                first,
                first_pixel,
                first == reference_index,
                matched.confidence,
                1.0,
                f64::from(first_corner.structure),
                first_corner.covariance,
            ),
            true,
            round,
            matched.confidence,
        );
        let second_observation = graph_observation(
            correspondence_observation(
                second,
                second_pixel,
                second == reference_index,
                matched.confidence,
                matched.local_scale,
                f64::from(second_corner.structure),
                second_corner.covariance,
            ),
            true,
            round,
            matched.confidence,
        );

        for observation in [first_observation, second_observation] {
            let camera = observation.observation.camera;
            if let Some(existing) = observation_for_camera_mut(&mut tracks[track_index], camera) {
                confirm_observation(existing);
                existing.identity_confidence = existing
                    .identity_confidence
                    .max(observation.identity_confidence);
                if observation.observation.confidence > existing.observation.confidence {
                    existing.observation = observation.observation;
                }
            } else {
                tracks[track_index].observations.push(observation);
            }
        }
        tracks[track_index].direct_edges.insert(edge);
        integrated += 1;
    }

    tracks.retain(|track| !track.observations.is_empty());
    integrated
}

/// Turn transitive/shared-track evidence into an independently verified direct
/// camera edge before falling back to a wide factory search.
///
/// The crucial distinction is that the shared track only *proposes* identity.
/// We refit a free pairwise epipolar model from the two image observations,
/// build a leave-one-out local affine field from the surviving shared tracks,
/// and finally require direct photometric agreement in the two images. Only
/// then is the edge recorded as direct and are its observations promoted.
///
/// This is intentionally the primary B<->C/C<->C bootstrap path: the empirical
/// local field already contains the real magnification, mirror pose, residual
/// calibration error and scene perspective, avoiding the brittle assumption
/// that factory geometry alone lands within one cross-scale patch search.
fn verify_existing_pair_tracks(
    tracks: &mut [GraphTrack],
    first: usize,
    second: usize,
    cameras: &[RigCameraInput<'_>],
    resolved: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> usize {
    let anchors = pair_anchors(tracks, first, second, false);
    if anchors.len() < options.anchor_min_pair_anchors.max(8) {
        return 0;
    }

    let correspondences = anchors
        .iter()
        .filter_map(|anchor| {
            let track = tracks.get(anchor.track)?;
            let first_observation = observation_for_camera(track, first)?;
            let second_observation = observation_for_camera(track, second)?;
            Some(AlignmentCorrespondence {
                reference_pixel: first_observation.observation.pixel,
                target_pixel: second_observation.observation.pixel,
                confidence: first_observation
                    .observation
                    .confidence
                    .min(second_observation.observation.confidence)
                    .clamp(0.0, 1.0) as f32,
                local_scale: second_observation.observation.local_scale.max(1.0e-6) as f32,
                structure: first_observation
                    .observation
                    .structure
                    .min(second_observation.observation.structure)
                    .max(0.0) as f32,
                reference_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                target_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                peak_margin: 1.0,
                forward_backward_error_px: 0.0,
                depth_reliability: None,
            })
        })
        .collect::<Vec<_>>();
    if correspondences.len() != anchors.len() {
        return 0;
    }

    let epipolar_inliers =
        fallback_epipolar_inliers(&correspondences, &resolved[first], &resolved[second]);
    if epipolar_inliers.len() < options.anchor_min_pair_anchors {
        return 0;
    }
    let inlier_anchors = epipolar_inliers
        .iter()
        .map(|&index| anchors[index])
        .collect::<Vec<_>>();

    let (Some(first_plane), Some(second_plane)) =
        (cameras[first].luminance, cameras[second].luminance)
    else {
        return 0;
    };

    // Proposal observations can be several pixels worse than final promoted
    // anchors (especially for C cameras), so this first direct-verification
    // gate is wider than the normal propagated-anchor gate. The subsequent
    // strict pair-model validation below prevents a coherent wrong repetition
    // from being promoted wholesale.
    let maximum_error = (options.anchor_seed_loo_max_error_px * 2.5).max(12.0);
    let minimum_score = (options.anchor_min_zncc - 0.10).max(0.55);
    let mut photometric = Vec::<PairAnchor>::new();
    for anchor in &inlier_anchors {
        let Some(forward) = local_affine(
            &inlier_anchors,
            anchor.first,
            options.anchor_neighbour_count,
            Some(anchor.track),
            false,
        ) else {
            continue;
        };
        let Some(reverse) = local_affine(
            &inlier_anchors,
            anchor.second,
            options.anchor_neighbour_count,
            Some(anchor.track),
            true,
        ) else {
            continue;
        };
        let forward_error = distance(forward.apply(anchor.first), anchor.second);
        let reverse_error = distance(reverse.apply(anchor.second), anchor.first);
        let symmetric =
            ((forward_error * forward_error + reverse_error * reverse_error) * 0.5).sqrt();
        if !symmetric.is_finite() || symmetric > maximum_error {
            continue;
        }
        let Some(score) = warped_zncc(
            first_plane,
            second_plane,
            anchor.first,
            anchor.second,
            forward,
            options.anchor_patch_radius_luma,
        ) else {
            continue;
        };
        if score.is_finite() && score >= minimum_score {
            photometric.push(*anchor);
        }
    }
    if photometric.len() < options.anchor_min_pair_anchors {
        return 0;
    }

    let strict_error = (options.anchor_seed_loo_max_error_px * 1.5).max(7.5);
    let strict = validate_pair_anchors(
        photometric.clone(),
        options.anchor_neighbour_count,
        strict_error,
        options.anchor_min_pair_anchors,
    );

    let edge = ordered_pair(first, second);
    if let Some(validated) = strict {
        for anchor in &validated.anchors {
            let Some(track) = tracks.get_mut(anchor.track) else {
                continue;
            };
            track.direct_edges.insert(edge);
            if let Some(observation) = observation_for_camera_mut(track, first) {
                confirm_observation(observation);
            }
            if let Some(observation) = observation_for_camera_mut(track, second) {
                confirm_observation(observation);
            }
        }
        return validated.anchors.len();
    }

    // Sparse/high-parallax cameras can have a real direct correspondence set
    // that is epipolar- and photometrically consistent while still failing the
    // local-affine pair-field test. Keep those observations as *provisional
    // direct edges* instead of discarding the camera entirely. They are not
    // promoted here, so they cannot enter calibration merely from this one
    // edge. Once a second independently verified edge closes a local cycle,
    // `promote_cycle_core_observations` promotes the cycle-supported cameras.
    // This is the onboarding path intended for modules such as C4.
    for anchor in &photometric {
        if let Some(track) = tracks.get_mut(anchor.track) {
            track.direct_edges.insert(edge);
        }
    }
    promote_cycle_core_observations(tracks);
    photometric.len()
}

#[allow(clippy::too_many_arguments)]
fn seed_active_edges_directly(
    tracks: &mut Vec<GraphTrack>,
    edges: &[(usize, usize)],
    attempted: &mut HashSet<(usize, usize)>,
    attempt_counts: &mut HashMap<(usize, usize), usize>,
    seeded_matches: &mut HashMap<(usize, usize), usize>,
    corners: &[Vec<RigCorner>],
    cameras: &[RigCameraInput<'_>],
    resolved: &[ResolvedCamera],
    reference_index: usize,
    round: usize,
    options: &RigRefinementOptions,
) -> (usize, usize) {
    let mut attempted_edges = 0usize;
    let mut integrated_matches = 0usize;
    for &(first, second) in edges {
        let edge = ordered_pair(first, second);
        if attempted.contains(&edge) {
            continue;
        }
        // Raw direct-edge counts are not enough to skip verification. V4
        // treated a dozen weak/provisional reference correspondences as a
        // "verified" edge even when no usable pair model existed (notably C2
        // and C3). Only skip work when the promoted observations already form
        // a validated local pair field.
        let already_validated = validate_pair_anchors(
            pair_anchors(tracks, first, second, true),
            options.anchor_neighbour_count,
            options.anchor_pair_loo_max_error_px,
            options.anchor_min_pair_anchors,
        )
        .is_some();
        if already_validated
            && direct_support_tracks(tracks, edge) >= options.anchor_min_pair_anchors
        {
            continue;
        }
        attempted.insert(edge);
        *attempt_counts.entry(edge).or_insert(0) += 1;
        attempted_edges += 1;

        // Prefer the already established global identity graph over factory
        // projection for direct-edge verification. Shared tracks give us an
        // empirical local correspondence field and therefore solve the B/C
        // scale/mirror mismatch much more reliably. The shared identity is
        // only a proposal: this routine independently refits pair geometry and
        // rechecks the two images before promoting the edge.
        let verified_existing =
            verify_existing_pair_tracks(tracks, first, second, cameras, resolved, options);
        if verified_existing >= options.anchor_min_pair_anchors {
            seeded_matches.insert(edge, verified_existing);
            integrated_matches += verified_existing;
            continue;
        }

        let mut matched =
            direct_pair_seed_matches(first, second, corners, cameras, resolved, options);
        if matched.len() < options.anchor_min_pair_anchors {
            // Rescue a structurally sparse/poorly factory-aligned camera
            // without paying the wider search cost on every edge.  C modules
            // can be ~80-100 native pixels away from the factory prediction in
            // real captures; an edge with no seeds gets one wider pass while
            // retaining the same mutual, appearance-margin and constellation
            // gates.
            let mut rescue_options = options.clone();
            rescue_options.anchor_direct_seed_search_radius_px =
                (options.anchor_direct_seed_search_radius_px * 1.75).min(192.0);
            matched = direct_pair_seed_matches(
                first,
                second,
                corners,
                cameras,
                resolved,
                &rescue_options,
            );
        }
        let integrated = integrate_direct_seed_matches(
            tracks,
            first,
            second,
            &matched,
            corners,
            reference_index,
            round,
            options,
        );
        seeded_matches.insert(edge, integrated);
        integrated_matches += integrated;
    }
    (attempted_edges, integrated_matches)
}

#[allow(clippy::too_many_arguments)]
fn reseed_under_supported_active_edges(
    tracks: &mut Vec<GraphTrack>,
    active: &HashSet<(usize, usize)>,
    attempted: &mut HashSet<(usize, usize)>,
    attempt_counts: &mut HashMap<(usize, usize), usize>,
    seeded_matches: &mut HashMap<(usize, usize), usize>,
    corners: &[Vec<RigCorner>],
    cameras: &[RigCameraInput<'_>],
    resolved: &[ResolvedCamera],
    reference_index: usize,
    round: usize,
    options: &RigRefinementOptions,
) -> (usize, usize) {
    // The factory rig can be tens of native pixels wrong for an otherwise
    // perfectly useful camera. Once an intermediate bearing solution exists,
    // give sparse active edges exactly one additional direct-image pass under
    // that improved geometry. This deliberately bypasses the normal
    // "already validated" early-out: an edge with 12 anchors is valid, but it
    // can still be far too sparse to recover a narrow-FOV camera robustly.
    // Matching itself is *not* relaxed; mutuality, appearance margin, reverse
    // closure and constellation LOO validation remain unchanged.
    let target_support = options.anchor_min_pair_anchors.saturating_mul(3);
    let mut retry = active
        .iter()
        .copied()
        .filter(|edge| direct_support_tracks(tracks, *edge) < target_support)
        .filter(|edge| attempt_counts.get(edge).copied().unwrap_or(0) < 2)
        .collect::<Vec<_>>();
    retry.sort_unstable();

    let mut attempted_edges = 0usize;
    let mut integrated_matches = 0usize;
    for (first, second) in retry {
        let edge = ordered_pair(first, second);
        attempted.insert(edge);
        *attempt_counts.entry(edge).or_insert(0) += 1;
        attempted_edges += 1;

        // By the time this recovery pass runs, propagation may already have
        // created shared track identities on this pair.  Re-verify those first
        // under the current intermediate geometry; this is substantially more
        // reliable on repeated structures than launching another blind corner
        // search.
        let verified_existing =
            verify_existing_pair_tracks(tracks, first, second, cameras, resolved, options);
        if verified_existing >= options.anchor_min_pair_anchors {
            seeded_matches
                .entry(edge)
                .and_modify(|count| *count = (*count).max(verified_existing))
                .or_insert(verified_existing);
            integrated_matches += verified_existing;
            continue;
        }

        let mut matched =
            direct_pair_seed_matches(first, second, corners, cameras, resolved, options);
        if matched.len() < options.anchor_min_pair_anchors {
            let mut rescue_options = options.clone();
            // Geometry is already capture-refined here, so this wider pass is
            // only a fallback for local-model error/occlusion, not a way to
            // compensate for the original factory bearing offset.
            rescue_options.anchor_direct_seed_search_radius_px =
                (options.anchor_direct_seed_search_radius_px * 1.5).min(160.0);
            matched = direct_pair_seed_matches(
                first,
                second,
                corners,
                cameras,
                resolved,
                &rescue_options,
            );
        }
        let integrated = integrate_direct_seed_matches(
            tracks,
            first,
            second,
            &matched,
            corners,
            reference_index,
            round,
            options,
        );
        seeded_matches
            .entry(edge)
            .and_modify(|count| *count = (*count).max(integrated))
            .or_insert(integrated);
        integrated_matches += integrated;
    }
    (attempted_edges, integrated_matches)
}

fn retire_failed_direct_edges(
    tracks: &[GraphTrack],
    active: &mut HashSet<(usize, usize)>,
    attempted: &HashSet<(usize, usize)>,
    failed: &mut HashSet<(usize, usize)>,
    minimum_support_tracks: usize,
) -> usize {
    let retiring = active
        .iter()
        .copied()
        .filter(|edge| {
            attempted.contains(edge)
                && direct_support_tracks(tracks, *edge) < minimum_support_tracks
        })
        .collect::<Vec<_>>();
    for edge in &retiring {
        active.remove(edge);
        failed.insert(*edge);
    }
    retiring.len()
}

fn observation_for_camera(track: &GraphTrack, camera: usize) -> Option<&GraphObservation> {
    track
        .observations
        .iter()
        .find(|observation| observation.observation.camera == camera)
}

fn observation_for_camera_mut(
    track: &mut GraphTrack,
    camera: usize,
) -> Option<&mut GraphObservation> {
    track
        .observations
        .iter_mut()
        .find(|observation| observation.observation.camera == camera)
}

fn observation_count(tracks: &[GraphTrack]) -> usize {
    tracks.iter().map(|track| track.observations.len()).sum()
}

fn has_cycle(track: &GraphTrack) -> bool {
    if track.direct_edges.len() < 3 || track.observations.len() < 3 {
        return false;
    }
    let mut cameras = track
        .observations
        .iter()
        .map(|observation| observation.observation.camera)
        .collect::<Vec<_>>();
    cameras.sort_unstable();
    cameras.dedup();
    let mut index = HashMap::new();
    for (position, &camera) in cameras.iter().enumerate() {
        index.insert(camera, position);
    }
    let mut parent = (0..cameras.len()).collect::<Vec<_>>();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    for &(first_camera, second_camera) in &track.direct_edges {
        let (Some(&first), Some(&second)) = (index.get(&first_camera), index.get(&second_camera))
        else {
            continue;
        };
        let first_root = root(&mut parent, first);
        let second_root = root(&mut parent, second);
        if first_root == second_root {
            return true;
        }
        parent[first_root] = second_root;
    }
    false
}

fn cycle_core_cameras(track: &GraphTrack) -> HashSet<usize> {
    let mut core = track
        .observations
        .iter()
        .filter(|observation| observation.membership != ObservationMembership::SoftOutlier)
        .map(|observation| observation.observation.camera)
        .collect::<HashSet<_>>();
    loop {
        let removing = core
            .iter()
            .copied()
            .filter(|camera| {
                track
                    .direct_edges
                    .iter()
                    .filter(|&&(first, second)| {
                        (first == *camera && core.contains(&second))
                            || (second == *camera && core.contains(&first))
                    })
                    .count()
                    < 2
            })
            .collect::<Vec<_>>();
        if removing.is_empty() {
            break;
        }
        for camera in removing {
            core.remove(&camera);
        }
    }
    core
}

fn promote_cycle_core_observations(tracks: &mut [GraphTrack]) -> usize {
    let mut promoted = 0usize;
    for track in tracks {
        let core = cycle_core_cameras(track);
        if core.len() < 3 {
            continue;
        }
        for observation in &mut track.observations {
            if core.contains(&observation.observation.camera)
                && !observation.promoted
                && observation.membership != ObservationMembership::SoftOutlier
            {
                confirm_observation(observation);
                promoted += 1;
            }
        }
    }
    promoted
}

fn strong_three_plus_tracks(tracks: &[GraphTrack]) -> usize {
    tracks
        .iter()
        .filter(|track| cycle_core_cameras(track).len() >= 3)
        .count()
}

fn track_key(pixel: Vec2) -> [i32; 2] {
    [
        (pixel[0] * 16.0).round() as i32,
        (pixel[1] * 16.0).round() as i32,
    ]
}

fn correspondence_observation(
    camera: usize,
    pixel: Vec2,
    fixed_gauge: bool,
    confidence: f64,
    local_scale: f64,
    structure: f64,
    covariance: [[f32; 2]; 2],
) -> TrackObservation {
    TrackObservation {
        camera,
        pixel,
        bootstrap_residual_proposal: [0.0, 0.0],
        localization_covariance: covariance.map(|row| row.map(f64::from)),
        fixed_gauge,
        confidence,
        local_scale,
        structure,
        depth_reliability: None,
        prepared: Default::default(),
    }
}

fn anchor_observation_from_correspondence(
    camera: usize,
    correspondence: &AlignmentCorrespondence,
    target: bool,
    reference_index: usize,
    promoted: bool,
) -> GraphObservation {
    let (pixel, covariance, confidence, local_scale) = if target {
        (
            correspondence.target_pixel,
            correspondence.target_localization_covariance,
            f64::from(correspondence.confidence),
            f64::from(correspondence.local_scale),
        )
    } else {
        (
            correspondence.reference_pixel,
            correspondence.reference_localization_covariance,
            1.0,
            1.0,
        )
    };
    GraphObservation {
        observation: correspondence_observation(
            camera,
            pixel,
            camera == reference_index,
            confidence,
            local_scale,
            f64::from(correspondence.structure),
            covariance,
        ),
        promoted,
        round: 0,
        membership: if promoted {
            ObservationMembership::Inlier
        } else {
            ObservationMembership::Provisional
        },
        bad_iterations: 0,
        good_iterations: 0,
        loo_reprojection_px: None,
        constellation_loo_px: None,
        identity_confidence: confidence.clamp(0.0, 1.0),
    }
}

fn fit_weighted_affine(samples: &[(Vec2, Vec2, f64)]) -> Option<Affine2> {
    if samples.len() < 3 {
        return None;
    }
    let mut normal: Mat3 = [[0.0; 3]; 3];
    let mut bx = [0.0; 3];
    let mut by = [0.0; 3];
    for &(source, target, weight) in samples {
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        let vector = [source[0], source[1], 1.0];
        for row in 0..3 {
            for column in 0..3 {
                normal[row][column] += weight * vector[row] * vector[column];
            }
            bx[row] += weight * vector[row] * target[0];
            by[row] += weight * vector[row] * target[1];
        }
    }
    let trace = normal[0][0] + normal[1][1] + normal[2][2];
    let ridge = trace.abs().max(1.0) * 1.0e-10;
    for axis in 0..3 {
        normal[axis][axis] += ridge;
    }
    let inverse = math::inverse(&normal)?;
    let x = mul_vec(&inverse, bx);
    let y = mul_vec(&inverse, by);
    let affine = Affine2 { x, y };
    let scale = affine.local_scale();
    (scale.is_finite() && (0.20..=5.0).contains(&scale)).then_some(affine)
}

fn local_affine(
    anchors: &[PairAnchor],
    query: Vec2,
    neighbour_count: usize,
    exclude_track: Option<usize>,
    reverse: bool,
) -> Option<Affine2> {
    let mut neighbours = anchors
        .iter()
        .filter(|anchor| Some(anchor.track) != exclude_track)
        .map(|anchor| {
            let source = if reverse { anchor.second } else { anchor.first };
            let target = if reverse { anchor.first } else { anchor.second };
            (squared_distance(source, query), source, target)
        })
        .filter(|(distance_sq, _, _)| distance_sq.is_finite())
        .collect::<Vec<_>>();
    let wanted = neighbour_count.max(3).min(neighbours.len());
    if neighbours.len() > wanted {
        neighbours.select_nth_unstable_by(wanted, |first, second| first.0.total_cmp(&second.0));
        neighbours.truncate(wanted);
    }
    if neighbours.len() < 3 {
        return None;
    }
    let neighbourhood_scale = neighbours
        .iter()
        .map(|entry| entry.0.sqrt())
        .filter(|value| value.is_finite())
        .fold(0.0f64, f64::max)
        .max(32.0);
    let samples = neighbours
        .into_iter()
        .map(|(distance_sq, source, target)| {
            let weight = 1.0 / (1.0 + distance_sq / (neighbourhood_scale * neighbourhood_scale));
            (source, target, weight)
        })
        .collect::<Vec<_>>();
    fit_weighted_affine(&samples)
}

fn pair_anchors(
    tracks: &[GraphTrack],
    first: usize,
    second: usize,
    promoted_only: bool,
) -> Vec<PairAnchor> {
    tracks
        .iter()
        .enumerate()
        .filter_map(|(track_index, track)| {
            let first_observation = observation_for_camera(track, first)?;
            let second_observation = observation_for_camera(track, second)?;
            if first_observation.membership == ObservationMembership::SoftOutlier
                || second_observation.membership == ObservationMembership::SoftOutlier
            {
                return None;
            }
            if promoted_only && (!first_observation.promoted || !second_observation.promoted) {
                return None;
            }
            Some(PairAnchor {
                track: track_index,
                first: first_observation.observation.pixel,
                second: second_observation.observation.pixel,
            })
        })
        .collect()
}

fn validate_pair_anchors(
    mut anchors: Vec<PairAnchor>,
    neighbour_count: usize,
    max_error_px: f64,
    minimum: usize,
) -> Option<PairModel> {
    if anchors.len() < minimum.max(4) {
        return None;
    }
    for _ in 0..2 {
        let mut keep = Vec::with_capacity(anchors.len());
        for anchor in &anchors {
            let Some(forward) = local_affine(
                &anchors,
                anchor.first,
                neighbour_count,
                Some(anchor.track),
                false,
            ) else {
                continue;
            };
            let Some(reverse) = local_affine(
                &anchors,
                anchor.second,
                neighbour_count,
                Some(anchor.track),
                true,
            ) else {
                continue;
            };
            let forward_error = distance(forward.apply(anchor.first), anchor.second);
            let reverse_error = distance(reverse.apply(anchor.second), anchor.first);
            let symmetric =
                ((forward_error * forward_error + reverse_error * reverse_error) * 0.5).sqrt();
            if symmetric.is_finite() && symmetric <= max_error_px {
                keep.push(*anchor);
            }
        }
        anchors = keep;
        if anchors.len() < minimum.max(4) {
            return None;
        }
    }

    let mut errors = Vec::with_capacity(anchors.len());
    for anchor in &anchors {
        let forward = local_affine(
            &anchors,
            anchor.first,
            neighbour_count,
            Some(anchor.track),
            false,
        )?;
        let reverse = local_affine(
            &anchors,
            anchor.second,
            neighbour_count,
            Some(anchor.track),
            true,
        )?;
        let forward_error = distance(forward.apply(anchor.first), anchor.second);
        let reverse_error = distance(reverse.apply(anchor.second), anchor.first);
        errors.push(((forward_error * forward_error + reverse_error * reverse_error) * 0.5).sqrt());
    }
    if errors.is_empty() {
        return None;
    }
    let rms = (errors.iter().map(|value| value * value).sum::<f64>() / errors.len() as f64).sqrt();
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    Some(PairModel {
        first: 0,
        second: 0,
        anchors,
        loo_rms: rms,
        loo_median: median,
    })
}

fn pair_models(
    tracks: &[GraphTrack],
    camera_count: usize,
    options: &RigRefinementOptions,
) -> Vec<PairModel> {
    let mut models = Vec::new();
    for first in 0..camera_count {
        for second in first + 1..camera_count {
            let anchors = pair_anchors(tracks, first, second, true);
            let Some(mut model) = validate_pair_anchors(
                anchors,
                options.anchor_neighbour_count,
                options.anchor_pair_loo_max_error_px,
                options.anchor_min_pair_anchors,
            ) else {
                continue;
            };
            model.first = first;
            model.second = second;
            models.push(model);
        }
    }
    models
}

fn oriented_anchors<'a>(
    model: &'a PairModel,
    source: usize,
    target: usize,
) -> Option<(bool, &'a [PairAnchor])> {
    if model.first == source && model.second == target {
        Some((false, &model.anchors))
    } else if model.first == target && model.second == source {
        Some((true, &model.anchors))
    } else {
        None
    }
}

fn model_for_pair(models: &[PairModel], first: usize, second: usize) -> Option<&PairModel> {
    let pair = ordered_pair(first, second);
    models
        .iter()
        .find(|model| (model.first, model.second) == pair)
}

fn predict_with_model(
    model: &PairModel,
    source: usize,
    target: usize,
    source_pixel: Vec2,
    neighbour_count: usize,
) -> Option<(Vec2, Affine2, f64)> {
    let (reverse, anchors) = oriented_anchors(model, source, target)?;
    let affine = local_affine(anchors, source_pixel, neighbour_count, None, reverse)?;
    Some((affine.apply(source_pixel), affine, model.loo_rms))
}

fn predict_with_model_excluding(
    model: &PairModel,
    source: usize,
    target: usize,
    source_pixel: Vec2,
    neighbour_count: usize,
    exclude_track: usize,
) -> Option<(Vec2, Affine2)> {
    let (reverse, anchors) = oriented_anchors(model, source, target)?;
    let affine = local_affine(
        anchors,
        source_pixel,
        neighbour_count,
        Some(exclude_track),
        reverse,
    )?;
    Some((affine.apply(source_pixel), affine))
}

/// Compare a candidate against the *relative* layout of neighbouring,
/// independently established landmarks.  Translation cancels: each neighbour's
/// observed target offset must agree with the source offset transformed by the
/// local affine Jacobian.  A repeated local patch shifted by one window/railing
/// period therefore accumulates a large error even when its ZNCC is excellent.
#[allow(clippy::too_many_arguments)]
fn constellation_relative_error(
    model: &PairModel,
    source: usize,
    target: usize,
    source_pixel: Vec2,
    target_pixel: Vec2,
    affine: Affine2,
    neighbour_count: usize,
    exclude_track: Option<usize>,
) -> Option<f64> {
    let (reverse, anchors) = oriented_anchors(model, source, target)?;
    let jacobian = affine.jacobian();
    let mut neighbours = anchors
        .iter()
        .filter(|anchor| exclude_track != Some(anchor.track))
        .map(|anchor| {
            let (source_anchor, target_anchor) = if reverse {
                (anchor.second, anchor.first)
            } else {
                (anchor.first, anchor.second)
            };
            (
                squared_distance(source_anchor, source_pixel),
                source_anchor,
                target_anchor,
            )
        })
        .collect::<Vec<_>>();
    let wanted = neighbour_count.max(3).min(neighbours.len());
    if neighbours.len() > wanted {
        neighbours.select_nth_unstable_by(wanted, |first, second| first.0.total_cmp(&second.0));
        neighbours.truncate(wanted);
    }
    if neighbours.len() < 3 {
        return None;
    }

    let neighbourhood_scale = neighbours
        .iter()
        .map(|entry| entry.0.sqrt())
        .filter(|value| value.is_finite())
        .fold(1.0f64, f64::max);
    let mut weighted_squared = 0.0;
    let mut weight_sum = 0.0;
    for (distance_sq, source_anchor, target_anchor) in neighbours {
        let source_delta = [
            source_anchor[0] - source_pixel[0],
            source_anchor[1] - source_pixel[1],
        ];
        let expected_delta = [
            jacobian[0][0] * source_delta[0] + jacobian[0][1] * source_delta[1],
            jacobian[1][0] * source_delta[0] + jacobian[1][1] * source_delta[1],
        ];
        let observed_delta = [
            target_anchor[0] - target_pixel[0],
            target_anchor[1] - target_pixel[1],
        ];
        let residual = [
            expected_delta[0] - observed_delta[0],
            expected_delta[1] - observed_delta[1],
        ];
        let error_sq = residual[0] * residual[0] + residual[1] * residual[1];
        let weight = 1.0 / (1.0 + distance_sq / (neighbourhood_scale * neighbourhood_scale));
        weighted_squared += weight * error_sq;
        weight_sum += weight;
    }
    (weight_sum > 0.0).then(|| (weighted_squared / weight_sum).sqrt())
}

fn sample_plane(plane: &Plane, point: Vec2) -> Option<f64> {
    if !point[0].is_finite() || !point[1].is_finite() || point[0] < 0.0 || point[1] < 0.0 {
        return None;
    }
    let x0 = point[0].floor() as usize;
    let y0 = point[1].floor() as usize;
    if x0 + 1 >= plane.width || y0 + 1 >= plane.height {
        return None;
    }
    let tx = point[0] - x0 as f64;
    let ty = point[1] - y0 as f64;
    let a = f64::from(plane.at(x0, y0));
    let b = f64::from(plane.at(x0 + 1, y0));
    let c = f64::from(plane.at(x0, y0 + 1));
    let d = f64::from(plane.at(x0 + 1, y0 + 1));
    if !a.is_finite() || !b.is_finite() || !c.is_finite() || !d.is_finite() {
        return None;
    }
    let top = a * (1.0 - tx) + b * tx;
    let bottom = c * (1.0 - tx) + d * tx;
    Some(top * (1.0 - ty) + bottom * ty)
}

fn warped_zncc(
    source: &Plane,
    target: &Plane,
    source_sensor: Vec2,
    target_sensor: Vec2,
    affine: Affine2,
    radius: usize,
) -> Option<f64> {
    let source_centre = sensor_to_luma(source_sensor);
    let jacobian = affine.jacobian();
    let side = 2 * radius + 1;
    let mut source_values = Vec::with_capacity(side * side);
    let mut target_values = Vec::with_capacity(side * side);
    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            let source_luma = [source_centre[0] + dx as f64, source_centre[1] + dy as f64];
            let source_value = sample_plane(source, source_luma)?;
            let sensor_offset = [SENSOR_PER_LUMA * dx as f64, SENSOR_PER_LUMA * dy as f64];
            let target_offset = [
                jacobian[0][0] * sensor_offset[0] + jacobian[0][1] * sensor_offset[1],
                jacobian[1][0] * sensor_offset[0] + jacobian[1][1] * sensor_offset[1],
            ];
            let target_luma = sensor_to_luma([
                target_sensor[0] + target_offset[0],
                target_sensor[1] + target_offset[1],
            ]);
            let target_value = sample_plane(target, target_luma)?;
            source_values.push(source_value);
            target_values.push(target_value);
        }
    }
    if source_values.len() < 9 {
        return None;
    }
    let count = source_values.len() as f64;
    let source_mean = source_values.iter().sum::<f64>() / count;
    let target_mean = target_values.iter().sum::<f64>() / count;
    let mut source_energy = 0.0;
    let mut target_energy = 0.0;
    let mut cross = 0.0;
    for (&source_value, &target_value) in source_values.iter().zip(&target_values) {
        let source_value = source_value - source_mean;
        let target_value = target_value - target_mean;
        source_energy += source_value * source_value;
        target_energy += target_value * target_value;
        cross += source_value * target_value;
    }
    if source_energy <= 1.0e-12 || target_energy <= 1.0e-12 {
        return None;
    }
    Some(cross / (source_energy * target_energy).sqrt())
}

#[allow(clippy::too_many_arguments)]
fn find_candidate(
    source_camera: usize,
    target_camera: usize,
    source_pixel: Vec2,
    predicted_target: Vec2,
    affine: Affine2,
    reverse_model: &PairModel,
    exclude_track: Option<usize>,
    corners: &[Vec<RigCorner>],
    cameras: &[RigCameraInput<'_>],
    round: usize,
    options: &RigRefinementOptions,
) -> Option<CandidateObservation> {
    let source_plane = cameras.get(source_camera)?.luminance?;
    let target_plane = cameras.get(target_camera)?.luminance?;
    let round_index = round.saturating_sub(1) as f64;
    let search_radius = (options.anchor_search_radius_px / (1.0 + 0.20 * round_index))
        .max(options.anchor_min_search_radius_px);
    let minimum_score = (options.anchor_min_zncc + 0.015 * round_index).min(0.96);
    let minimum_margin = options.anchor_min_appearance_margin + 0.004 * round_index;
    let maximum_closure = (options.anchor_reverse_max_error_px / (1.0 + 0.15 * round_index))
        .max(options.anchor_reverse_max_error_px * 0.55);
    let constellation_limit = (options.anchor_constellation_max_error_px
        + 0.35 * reverse_model.loo_rms.max(0.0))
    .max(2.0);

    // Candidate ranking is camera/FOV conditional.  We intentionally do not
    // ask whether a patch is globally unique in the entire source frame.  A
    // repetitive B-camera feature can still be unique inside the much smaller
    // physically plausible C-camera search region.
    let mut candidates = corners[target_camera]
        .iter()
        .filter_map(|corner| {
            let pixel = corner_sensor(corner);
            let prediction_error = distance(pixel, predicted_target);
            if prediction_error > search_radius {
                return None;
            }
            let reverse = if let Some(track) = exclude_track {
                predict_with_model_excluding(
                    reverse_model,
                    target_camera,
                    source_camera,
                    pixel,
                    options.anchor_neighbour_count,
                    track,
                )
                .map(|(point, affine)| (point, affine, reverse_model.loo_rms))
            } else {
                predict_with_model(
                    reverse_model,
                    target_camera,
                    source_camera,
                    pixel,
                    options.anchor_neighbour_count,
                )
            }?;
            let closure_error = distance(reverse.0, source_pixel);
            if closure_error > maximum_closure {
                return None;
            }
            let constellation_error = constellation_relative_error(
                reverse_model,
                source_camera,
                target_camera,
                source_pixel,
                pixel,
                affine,
                options.anchor_neighbour_count,
                exclude_track,
            )?;
            if !constellation_error.is_finite() || constellation_error > constellation_limit {
                return None;
            }
            let score = warped_zncc(
                source_plane,
                target_plane,
                source_pixel,
                pixel,
                affine,
                options.anchor_patch_radius_luma,
            )?;
            if !score.is_finite() || score < minimum_score {
                return None;
            }

            // Appearance still dominates, but constellation and reverse
            // closure explicitly break ties between repeated local patterns.
            let ranked_score = score
                - 0.020 * prediction_error / search_radius.max(1.0)
                - 0.070 * constellation_error / constellation_limit
                - 0.020 * closure_error / maximum_closure.max(1.0);
            Some((
                ranked_score,
                score,
                closure_error,
                constellation_error,
                corner,
                pixel,
            ))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|first, second| second.0.total_cmp(&first.0));
    let best = candidates.first()?;
    let second_score = candidates.get(1).map_or(-1.0, |candidate| candidate.1);
    let second_rank = candidates.get(1).map_or(-1.0, |candidate| candidate.0);
    let margin = best.1 - second_score;
    let rank_margin = best.0 - second_rank;

    // Count only alternatives that are plausible *inside this camera's
    // geometry-supported domain*. This is the tele-camera-specific uniqueness
    // requested by the scene-landmark design.
    let competitor_band = minimum_margin.max(0.03);
    let conditional_competitors = candidates
        .iter()
        .skip(1)
        .filter(|candidate| candidate.1 >= best.1 - competitor_band && candidate.3 <= best.3 + 1.5)
        .count();

    let appearance_unique = margin >= minimum_margin;
    let constellation_unique = best.3 <= constellation_limit * 0.45
        && best.2 <= maximum_closure * 0.55
        && rank_margin >= 0.012
        && conditional_competitors <= options.anchor_max_conditional_competitors;
    if !appearance_unique && !constellation_unique {
        return None;
    }

    let appearance_confidence =
        ((best.1 - minimum_score) / (1.0 - minimum_score).max(1.0e-6)).clamp(0.0, 1.0);
    let constellation_confidence = (1.0 - best.3 / constellation_limit).clamp(0.0, 1.0);
    let closure_confidence = (1.0 - best.2 / maximum_closure.max(1.0e-6)).clamp(0.0, 1.0);
    let uniqueness_confidence = if appearance_unique {
        (margin / (minimum_margin * 3.0).max(1.0e-6)).clamp(0.0, 1.0)
    } else {
        (rank_margin / 0.05).clamp(0.0, 1.0)
    };
    let identity_confidence = (0.40 * appearance_confidence
        + 0.30 * constellation_confidence
        + 0.20 * closure_confidence
        + 0.10 * uniqueness_confidence)
        .clamp(0.0, 1.0);

    Some(CandidateObservation {
        pixel: best.5,
        covariance: best.4.covariance.map(|row| row.map(f64::from)),
        structure: f64::from(best.4.structure),
        score: best.1,
        margin,
        closure_error: best.2,
        constellation_error: best.3,
        conditional_competitors,
        identity_confidence,
        local_scale: affine.local_scale(),
    })
}

fn occupied(tracks: &[GraphTrack], camera: usize, pixel: Vec2, radius: f64) -> bool {
    let radius_sq = radius * radius;
    tracks.iter().any(|track| {
        observation_for_camera(track, camera).is_some_and(|observation| {
            squared_distance(observation.observation.pixel, pixel) <= radius_sq
        })
    })
}

fn graph_track_to_track(track: &GraphTrack, enforce_cycle: bool) -> Option<Track> {
    if track.observations.len() < 2 {
        return None;
    }
    // Unpromoted observations are allowed to seed/guide direct edge
    // verification, but they are not calibration measurements. This prevents
    // a weak reference-star hypothesis from leaking into the final rig merely
    // because it was useful for proposing a later, independently verified
    // camera-camera connection.
    let mut observations = track
        .observations
        .iter()
        .filter(|observation| observation.promoted)
        .map(|observation| observation.observation.clone())
        .collect::<Vec<_>>();
    if observations.len() < 2 {
        return None;
    }

    if enforce_cycle && observations.len() >= 3 {
        // A global cycle somewhere in the track is not enough. V4 could keep a
        // dangling C-camera observation attached to a perfectly good B-camera
        // triangle; that single weak observation then dominated held-out RMS.
        // Keep only the 2-core of the direct-evidence graph, i.e. cameras that
        // have at least two independent direct neighbours after iterative
        // pruning. Every retained 3+ view observation is therefore locally
        // cycle-supported.
        let core = cycle_core_cameras(track);
        let core_observations = observations
            .iter()
            .filter(|observation| core.contains(&observation.camera))
            .cloned()
            .collect::<Vec<_>>();
        if core_observations.len() >= 3 {
            observations = core_observations;
        } else {
            // No locally cycle-supported multi-view component exists. Preserve
            // at most one strongest independently observed pair rather than
            // leaking dangling observations into bundle adjustment.
            let mut best_pair: Option<(f64, TrackObservation, TrackObservation)> = None;
            for &(first, second) in &track.direct_edges {
                let first_observation = observations
                    .iter()
                    .find(|observation| observation.camera == first);
                let second_observation = observations
                    .iter()
                    .find(|observation| observation.camera == second);
                let (Some(first_observation), Some(second_observation)) =
                    (first_observation, second_observation)
                else {
                    continue;
                };
                let score = first_observation
                    .confidence
                    .min(second_observation.confidence);
                let replace = match best_pair.as_ref() {
                    Some(entry) => score > entry.0,
                    None => true,
                };
                if replace {
                    best_pair =
                        Some((score, first_observation.clone(), second_observation.clone()));
                }
            }
            if let Some((_, first, second)) = best_pair {
                observations = vec![first, second];
            } else {
                observations
                    .sort_by(|first, second| second.confidence.total_cmp(&first.confidence));
                observations.truncate(2);
            }
        }
    }
    Some(Track {
        key: track.key,
        observations,
        condition: f64::NAN,
        max_ray_angle_degrees: f64::NAN,
    })
}

#[allow(clippy::too_many_arguments)]
fn intermediate_geometry(
    tracks: &[GraphTrack],
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    intrinsics_mode: IntrinsicsMode,
    factory: &[ResolvedCamera],
    validation_modulus: u64,
    round: usize,
    options: &RigRefinementOptions,
) -> (Vec<ResolvedCamera>, usize, usize, usize, usize) {
    // Every objective evaluation re-triangulates each scene landmark. This is
    // structureless/variable-projection bundle adjustment: landmark variables
    // are eliminated from the outer camera solve, the same computational idea
    // that makes Schur BA scale well. The round solve is global across all
    // participating cameras rather than using camera-level quality weights.
    let all_owned = tracks
        .iter()
        .filter_map(|track| graph_track_to_track(track, true))
        .filter(|track| {
            !options.held_out_validation || !is_validation_track(track, validation_modulus, options)
        })
        .collect::<Vec<_>>();
    let skeleton_count = scene_skeleton_track_count(tracks);
    if all_owned.len() < options.min_tracks.min(24) {
        return (factory.to_vec(), all_owned.len(), 0, 0, skeleton_count);
    }

    // Bootstrap from the hardest-to-fake landmarks: cycle-supported
    // constellations with at least three active observations. Later rounds use
    // the complete validated population once geometry is already close.
    let skeleton_owned = tracks
        .iter()
        .filter(|track| {
            track
                .observations
                .iter()
                .filter(|observation| {
                    observation.promoted && observation.membership == ObservationMembership::Inlier
                })
                .count()
                >= 3
                && cycle_core_cameras(track).len() >= 3
        })
        .filter_map(|track| graph_track_to_track(track, true))
        .filter(|track| {
            !options.held_out_validation || !is_validation_track(track, validation_modulus, options)
        })
        .collect::<Vec<_>>();
    let skeleton_minimum = options.min_tracks.min(24).max(12);
    let solve_owned = if round <= 1 && skeleton_owned.len() >= skeleton_minimum {
        &skeleton_owned
    } else {
        &all_owned
    };
    let refs = solve_owned.iter().collect::<Vec<_>>();

    let orientation_specs = parameter_specs(cameras, reference_index, &refs, options)
        .into_iter()
        .filter(|spec| matches!(spec.kind, ParameterKind::Orientation(_)))
        .collect::<Vec<_>>();
    if orientation_specs.is_empty() {
        return (factory.to_vec(), refs.len(), 0, 0, skeleton_count);
    }

    // First recover the effective bearing branch. This is much less sensitive
    // to uncertain metric depth than releasing centre/raster parameters from a
    // cold factory start.
    let (bearing_parameters, _, bearing_iterations) = bearing_bootstrap(
        &orientation_specs,
        cameras,
        &refs,
        reference_index,
        intrinsics_mode,
        options,
    );
    let bearing_parameters = if bearing_parameters.len() == orientation_specs.len() {
        bearing_parameters
    } else {
        vec![0.0; orientation_specs.len()]
    };
    let orientation_sweeps = options.max_iterations.clamp(1, 2);
    let (orientation_parameters, _, orientation_iterations) = staged_bundle_optimize_rig(
        bearing_parameters,
        &orientation_specs,
        orientation_sweeps,
        cameras,
        &refs,
        intrinsics_mode,
        options,
    );

    if round <= 1 {
        let refinements =
            refinements_from_parameters(cameras.len(), &orientation_parameters, &orientation_specs);
        let resolved = resolve_cameras(cameras, &refinements, intrinsics_mode)
            .unwrap_or_else(|| factory.to_vec());
        return (
            resolved,
            refs.len(),
            bearing_iterations.saturating_add(orientation_iterations),
            orientation_specs.len(),
            skeleton_count,
        );
    }

    // Once the scene skeleton exists, release every physical direction that
    // the global retriangulated Jacobian can actually observe. All cameras are
    // optimized in the same objective; there is no capture-wide C/B weight.
    let candidate_specs = parameter_specs(cameras, reference_index, &refs, options);
    let candidate_base = remap_parameters(
        &orientation_parameters,
        &orientation_specs,
        &candidate_specs,
    );
    let (global_specs, _) = filter_observable_parameter_specs(
        &candidate_specs,
        &candidate_base,
        cameras,
        &refs,
        intrinsics_mode,
        options,
    );
    if global_specs.is_empty() {
        let refinements =
            refinements_from_parameters(cameras.len(), &orientation_parameters, &orientation_specs);
        let resolved = resolve_cameras(cameras, &refinements, intrinsics_mode)
            .unwrap_or_else(|| factory.to_vec());
        return (
            resolved,
            refs.len(),
            bearing_iterations.saturating_add(orientation_iterations),
            orientation_specs.len(),
            skeleton_count,
        );
    }
    let global_seed = remap_parameters(&orientation_parameters, &orientation_specs, &global_specs);
    let global_sweeps = options.max_iterations.clamp(1, 2);
    let (global_parameters, _, global_iterations) = staged_bundle_optimize_rig(
        global_seed,
        &global_specs,
        global_sweeps,
        cameras,
        &refs,
        intrinsics_mode,
        options,
    );
    let refinements = refinements_from_parameters(cameras.len(), &global_parameters, &global_specs);
    let resolved =
        resolve_cameras(cameras, &refinements, intrinsics_mode).unwrap_or_else(|| factory.to_vec());
    (
        resolved,
        refs.len(),
        bearing_iterations
            .saturating_add(orientation_iterations)
            .saturating_add(global_iterations),
        global_specs.len(),
        skeleton_count,
    )
}

fn triangulated_points(
    tracks: &[GraphTrack],
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Vec<Option<Vec3>> {
    tracks
        .iter()
        .map(|track| {
            let ordinary = graph_track_to_track(track, false)?;
            if ordinary.observations.len() < 2 {
                return None;
            }
            let result = triangulate(&ordinary.observations, cameras, options)?;
            (result.point.iter().all(|value| value.is_finite())
                && result.condition.is_finite()
                && result.condition <= options.max_triangulation_condition)
                .then_some(result.point)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct MembershipUpdate {
    soft_outliers: usize,
    reactivated: usize,
    hard_rejected: usize,
    split_tracks: usize,
}

fn observation_direct_degree(track: &GraphTrack, camera: usize) -> usize {
    track
        .direct_edges
        .iter()
        .filter(|&&(first, second)| first == camera || second == camera)
        .count()
}

/// Predict one observation from the rest of the same physical landmark.  Weak
/// depth geometry is intentionally treated as "unknown", not "wrong": the LOO
/// test is only returned when the remaining rays have enough angular
/// conditioning to make the prediction meaningful.
fn leave_one_out_reprojection_error(
    track: &GraphTrack,
    observation_index: usize,
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Option<f64> {
    let tested = track.observations.get(observation_index)?;
    let observations = track
        .observations
        .iter()
        .enumerate()
        .filter(|(index, observation)| {
            *index != observation_index
                && observation.promoted
                && observation.membership != ObservationMembership::SoftOutlier
        })
        .map(|(_, observation)| observation.observation.clone())
        .collect::<Vec<_>>();
    if observations.len() < 2 {
        return None;
    }
    let triangulated = triangulate(&observations, cameras, options)?;
    if !triangulated.condition.is_finite()
        || triangulated.condition > options.max_triangulation_condition.min(2.0e7)
        || !triangulated.max_ray_angle_degrees.is_finite()
        || triangulated.max_ray_angle_degrees < options.min_ray_angle_degrees.max(0.01)
    {
        return None;
    }
    let projected = cameras
        .get(tested.observation.camera)?
        .project(triangulated.point)?;
    let sensor_error = distance(projected, tested.observation.pixel);
    Some(sensor_error / tested.observation.local_scale.clamp(0.25, 4.0))
}

/// Scene-level leave-one-landmark-out constellation check. The tested track is
/// explicitly excluded when fitting every pair prediction, so the observation
/// cannot make itself look consistent. Provisional sibling observations are
/// allowed to *query* that independently fitted field; this is what lets a
/// newly split repeated-pattern hypothesis prove itself without putting either
/// sibling into bundle adjustment first. This test does not require metric
/// depth to be well conditioned.
fn constellation_leave_one_out_error(
    track_index: usize,
    track: &GraphTrack,
    observation_index: usize,
    models: &[PairModel],
    options: &RigRefinementOptions,
) -> Option<f64> {
    let target = track.observations.get(observation_index)?;
    let mut errors = Vec::<f64>::new();
    for (source_index, source) in track.observations.iter().enumerate() {
        if source_index == observation_index
            || source.membership == ObservationMembership::SoftOutlier
        {
            continue;
        }
        let Some(model) =
            model_for_pair(models, source.observation.camera, target.observation.camera)
        else {
            continue;
        };
        let Some((predicted, _)) = predict_with_model_excluding(
            model,
            source.observation.camera,
            target.observation.camera,
            source.observation.pixel,
            options.anchor_neighbour_count,
            track_index,
        ) else {
            continue;
        };
        let error = distance(predicted, target.observation.pixel)
            / target.observation.local_scale.clamp(0.25, 4.0);
        if error.is_finite() {
            errors.push(error);
        }
    }
    if errors.is_empty() {
        return None;
    }
    errors.sort_by(f64::total_cmp);
    Some(errors[errors.len() / 2])
}

fn graph_track_is_validation(
    track: &GraphTrack,
    validation_modulus: u64,
    options: &RigRefinementOptions,
) -> bool {
    if !options.held_out_validation {
        return false;
    }
    // is_validation_track hashes only Track::key, so an observation-free probe
    // gives exactly the same spatial block assignment without coupling the
    // validation decision to current graph membership.
    let probe = Track {
        key: track.key,
        observations: Vec::new(),
        condition: f64::NAN,
        max_ray_angle_degrees: f64::NAN,
    };
    is_validation_track(&probe, validation_modulus, options)
}

fn split_soft_outlier_components(
    tracks: &mut Vec<GraphTrack>,
    validation_modulus: u64,
    options: &RigRefinementOptions,
) -> MembershipUpdate {
    let mut update = MembershipUpdate::default();
    let mut spawned = Vec::<GraphTrack>::new();

    for track in tracks.iter_mut() {
        if graph_track_is_validation(track, validation_modulus, options) {
            continue;
        }
        let bad_cameras = track
            .observations
            .iter()
            .filter(|observation| {
                observation.membership == ObservationMembership::SoftOutlier
                    && observation.bad_iterations >= 2
            })
            .map(|observation| observation.observation.camera)
            .collect::<HashSet<_>>();
        let good_cameras = track
            .observations
            .iter()
            .filter(|observation| !bad_cameras.contains(&observation.observation.camera))
            .map(|observation| observation.observation.camera)
            .collect::<HashSet<_>>();

        // Two or more mutually directly observed outliers can be a *second
        // coherent repeated-pattern identity*, not merely garbage. Preserve
        // that hypothesis by splitting it into its own provisional landmark.
        let bad_has_direct_edge = track
            .direct_edges
            .iter()
            .any(|&(first, second)| bad_cameras.contains(&first) && bad_cameras.contains(&second));
        if bad_cameras.len() >= 2 && good_cameras.len() >= 2 && bad_has_direct_edge {
            let mut retained = Vec::new();
            let mut split = Vec::new();
            for mut observation in track.observations.drain(..) {
                if bad_cameras.contains(&observation.observation.camera) {
                    observation.promoted = false;
                    observation.membership = ObservationMembership::Provisional;
                    observation.bad_iterations = 0;
                    observation.good_iterations = 0;
                    split.push(observation);
                } else {
                    retained.push(observation);
                }
            }
            let retained_edges = track
                .direct_edges
                .iter()
                .copied()
                .filter(|(first, second)| {
                    good_cameras.contains(first) && good_cameras.contains(second)
                })
                .collect::<HashSet<_>>();
            let split_edges = track
                .direct_edges
                .iter()
                .copied()
                .filter(|(first, second)| {
                    bad_cameras.contains(first) && bad_cameras.contains(second)
                })
                .collect::<HashSet<_>>();
            track.observations = retained;
            track.direct_edges = retained_edges;
            if split.len() >= 2 && !split_edges.is_empty() {
                let key = split
                    .first()
                    .map(|observation| track_key(observation.observation.pixel))
                    .unwrap_or(track.key);
                spawned.push(GraphTrack {
                    key,
                    observations: split,
                    direct_edges: split_edges,
                });
                update.split_tracks += 1;
            }
            continue;
        }

        // A lone observation that stays bad for several independently updated
        // global geometries is finally removed.  Until this point rejection is
        // reversible, preventing an early bad rig from permanently deleting a
        // correct C-camera correspondence.
        let hard_after = options.anchor_observation_bad_iterations.max(2);
        let removable = track
            .observations
            .iter()
            .filter(|observation| {
                observation.membership == ObservationMembership::SoftOutlier
                    && observation.bad_iterations >= hard_after
            })
            .map(|observation| observation.observation.camera)
            .collect::<HashSet<_>>();
        if !removable.is_empty() && track.observations.len().saturating_sub(removable.len()) >= 2 {
            let before = track.observations.len();
            track
                .observations
                .retain(|observation| !removable.contains(&observation.observation.camera));
            track.direct_edges.retain(|(first, second)| {
                !removable.contains(first) && !removable.contains(second)
            });
            update.hard_rejected += before.saturating_sub(track.observations.len());
        }
    }

    tracks.extend(spawned);
    tracks.retain(|track| track.observations.len() >= 2);
    update
}

/// Reversible observation-level membership update.  No camera receives a
/// global quality penalty: every measurement is judged by its own scene
/// identity, LOO reprojection, and constellation evidence.
fn update_graph_observation_membership(
    tracks: &mut Vec<GraphTrack>,
    cameras: &[ResolvedCamera],
    models: &[PairModel],
    validation_modulus: u64,
    options: &RigRefinementOptions,
) -> MembershipUpdate {
    let mut evidence = Vec::<Vec<(Option<f64>, Option<f64>)>>::with_capacity(tracks.len());
    for (track_index, track) in tracks.iter().enumerate() {
        if graph_track_is_validation(track, validation_modulus, options) {
            evidence.push(vec![(None, None); track.observations.len()]);
            continue;
        }
        let per_observation = (0..track.observations.len())
            .map(|observation_index| {
                (
                    leave_one_out_reprojection_error(track, observation_index, cameras, options),
                    constellation_leave_one_out_error(
                        track_index,
                        track,
                        observation_index,
                        models,
                        options,
                    ),
                )
            })
            .collect::<Vec<_>>();
        evidence.push(per_observation);
    }

    let mut update = MembershipUpdate::default();
    for (track_index, track) in tracks.iter_mut().enumerate() {
        if graph_track_is_validation(track, validation_modulus, options) {
            continue;
        }
        let direct_degrees = track
            .observations
            .iter()
            .map(|observation| {
                let camera = observation.observation.camera;
                (camera, observation_direct_degree(track, camera))
            })
            .collect::<HashMap<_, _>>();
        for (observation_index, observation) in track.observations.iter_mut().enumerate() {
            let (geometric, constellation) = evidence[track_index][observation_index];
            observation.loo_reprojection_px = geometric;
            observation.constellation_loo_px = constellation;

            let soft = options.anchor_observation_loo_soft_px.max(0.5);
            let hard = options.anchor_observation_loo_hard_px.max(soft * 1.5);
            let consistent = match (geometric, constellation) {
                (Some(geo), Some(scene)) => geo <= soft && scene <= soft,
                (Some(value), None) | (None, Some(value)) => value <= soft,
                (None, None) => false,
            };
            // One conflicting diagnostic is not enough to delete a point: the
            // other one may be telling us that metric depth is weak while
            // scene identity is still excellent.
            let inconsistent = match (geometric, constellation) {
                (Some(geo), Some(scene)) => {
                    geo > soft && scene > soft && (geo > hard || scene > hard)
                }
                (Some(value), None) | (None, Some(value)) => value > hard,
                (None, None) => false,
            };

            if consistent {
                observation.bad_iterations = 0;
                observation.good_iterations = observation.good_iterations.saturating_add(1);
                if observation.membership == ObservationMembership::SoftOutlier
                    && observation.good_iterations
                        >= options.anchor_observation_recovery_iterations.max(1)
                {
                    observation.membership = ObservationMembership::Inlier;
                    // Reactivation still needs independent scene identity:
                    // either two direct neighbours or strong local identity.
                    if direct_degrees
                        .get(&observation.observation.camera)
                        .copied()
                        .unwrap_or(0)
                        >= 2
                        || observation.identity_confidence >= 0.72
                    {
                        observation.promoted = true;
                        update.reactivated += 1;
                    } else {
                        observation.membership = ObservationMembership::Provisional;
                    }
                } else if observation.membership == ObservationMembership::Provisional
                    && observation.good_iterations
                        >= options.anchor_observation_recovery_iterations.max(1)
                    && (direct_degrees
                        .get(&observation.observation.camera)
                        .copied()
                        .unwrap_or(0)
                        >= 2
                        || observation.identity_confidence >= 0.78)
                {
                    confirm_observation(observation);
                    update.reactivated += 1;
                }
            } else if inconsistent {
                observation.good_iterations = 0;
                observation.bad_iterations = observation.bad_iterations.saturating_add(1);
                if observation.membership != ObservationMembership::SoftOutlier {
                    observation.membership = ObservationMembership::SoftOutlier;
                    observation.promoted = false;
                    update.soft_outliers += 1;
                }
            }
        }
    }

    let split = split_soft_outlier_components(tracks, validation_modulus, options);
    update.hard_rejected += split.hard_rejected;
    update.split_tracks += split.split_tracks;
    update
}

fn scene_skeleton_track_count(tracks: &[GraphTrack]) -> usize {
    tracks
        .iter()
        .filter(|track| {
            let promoted = track
                .observations
                .iter()
                .filter(|observation| {
                    observation.promoted && observation.membership == ObservationMembership::Inlier
                })
                .count();
            promoted >= 3 && cycle_core_cameras(track).len() >= 3
        })
        .count()
}

fn bootstrap_graph(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    resolved: &[ResolvedCamera],
    _active_edges: &HashSet<(usize, usize)>,
    options: &RigRefinementOptions,
    report: &mut AnchorGraphReport,
) -> (Vec<GraphTrack>, usize) {
    let mut tracks = Vec::<GraphTrack>::new();
    let mut by_reference = HashMap::<(i32, i32), usize>::new();
    let mut pairwise_matches = 0usize;

    for (camera, alignment) in alignments.iter().enumerate() {
        if camera == reference_index
            || !cameras[camera].match_evidence_enabled
            || cameras[camera].calibration.is_none()
        {
            continue;
        }
        let inliers = fallback_epipolar_inliers(
            &alignment.correspondences,
            &resolved[reference_index],
            &resolved[camera],
        );
        if inliers.len() < options.anchor_min_pair_anchors {
            continue;
        }
        report.bootstrap_pairs += 1;
        report.bootstrap_matches_before_constellation += inliers.len();
        let candidates = inliers
            .iter()
            .enumerate()
            .map(|(track, &index)| {
                let correspondence = &alignment.correspondences[index];
                PairAnchor {
                    track,
                    first: correspondence.reference_pixel,
                    second: correspondence.target_pixel,
                }
            })
            .collect::<Vec<_>>();

        // Strict constellation validation determines which reference-star
        // observations are calibration-grade immediately.  Crucially, do not
        // throw the remaining robust epipolar inliers away: they are retained
        // as *unpromoted proposal observations*. They can establish a
        // transitive A<->B field and must be independently re-found in those
        // two images before becoming direct evidence.  This is particularly
        // important for narrow/sky-heavy C views where a local affine LOO
        // test can be underconstrained even though the sparse image matches
        // contain useful identity information.
        let validated = validate_pair_anchors(
            candidates,
            options.anchor_neighbour_count,
            options.anchor_seed_loo_max_error_px,
            options.anchor_min_pair_anchors,
        );
        let validated_indices = validated
            .as_ref()
            .map(|model| {
                model
                    .anchors
                    .iter()
                    .map(|anchor| inliers[anchor.track])
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        report.bootstrap_rejected_constellation +=
            inliers.len().saturating_sub(validated_indices.len());

        for &correspondence_index in &inliers {
            let correspondence = &alignment.correspondences[correspondence_index];
            if !resolved[reference_index].contains(correspondence.reference_pixel)
                || !resolved[camera].contains(correspondence.target_pixel)
            {
                continue;
            }
            let promoted = validated_indices.contains(&correspondence_index);
            if promoted {
                pairwise_matches += 1;
                report.bootstrap_anchors += 1;
            }
            let key_tuple = (
                (correspondence.reference_pixel[0] * 8.0).round() as i32,
                (correspondence.reference_pixel[1] * 8.0).round() as i32,
            );
            let track_index = if let Some(&index) = by_reference.get(&key_tuple) {
                index
            } else {
                let index = tracks.len();
                tracks.push(GraphTrack {
                    key: track_key(correspondence.reference_pixel),
                    observations: vec![anchor_observation_from_correspondence(
                        reference_index,
                        correspondence,
                        false,
                        reference_index,
                        promoted,
                    )],
                    direct_edges: HashSet::new(),
                });
                by_reference.insert(key_tuple, index);
                index
            };
            let track = &mut tracks[track_index];

            // If any target independently validates this reference feature,
            // the reference observation itself is calibration-grade.
            if promoted {
                if let Some(reference) = observation_for_camera_mut(track, reference_index) {
                    confirm_observation(reference);
                }
            }

            if let Some(existing) = observation_for_camera_mut(track, camera) {
                if f64::from(correspondence.confidence) > existing.observation.confidence {
                    let keep_promoted = existing.promoted || promoted;
                    *existing = anchor_observation_from_correspondence(
                        camera,
                        correspondence,
                        true,
                        reference_index,
                        keep_promoted,
                    );
                } else if promoted {
                    confirm_observation(existing);
                }
            } else {
                track
                    .observations
                    .push(anchor_observation_from_correspondence(
                        camera,
                        correspondence,
                        true,
                        reference_index,
                        promoted,
                    ));
            }

            // "Direct" means independently observed in these two images; it
            // does not depend on whether the topology scheduler has chosen to
            // spend an active propagation edge on this pair.  Only strict
            // constellation survivors are granted direct evidence here.
            if promoted {
                track
                    .direct_edges
                    .insert(ordered_pair(reference_index, camera));
            }
        }
    }
    (tracks, pairwise_matches)
}

fn close_existing_cycle_edges(
    tracks: &mut [GraphTrack],
    models: &[PairModel],
    cameras: &[RigCameraInput<'_>],
    active_edges: &HashSet<(usize, usize)>,
    round: usize,
    options: &RigRefinementOptions,
) -> usize {
    let mut accepted = Vec::<(usize, usize, usize)>::new();
    let threshold = (options.anchor_pair_loo_max_error_px
        / (1.0 + 0.10 * round.saturating_sub(1) as f64))
        .max(1.5);
    let minimum_score =
        (options.anchor_min_zncc + 0.05 + 0.01 * round.saturating_sub(1) as f64).min(0.98);
    for (track_index, track) in tracks.iter().enumerate() {
        if track.observations.len() < 3 {
            continue;
        }
        for first_index in 0..track.observations.len() {
            for second_index in first_index + 1..track.observations.len() {
                let first = &track.observations[first_index];
                let second = &track.observations[second_index];
                // A soft outlier must earn reactivation through the explicit
                // LOO membership path. Cycle closure may validate provisional
                // observations, but it must not silently erase a bad streak.
                if first.membership == ObservationMembership::SoftOutlier
                    || second.membership == ObservationMembership::SoftOutlier
                {
                    continue;
                }
                let first_camera = first.observation.camera;
                let second_camera = second.observation.camera;
                let edge = ordered_pair(first_camera, second_camera);
                if !active_edges.contains(&edge) || track.direct_edges.contains(&edge) {
                    continue;
                }
                let Some(model) = model_for_pair(models, first_camera, second_camera) else {
                    continue;
                };
                let Some((predicted_second, affine)) = predict_with_model_excluding(
                    model,
                    first_camera,
                    second_camera,
                    first.observation.pixel,
                    options.anchor_neighbour_count,
                    track_index,
                ) else {
                    continue;
                };
                let Some((predicted_first, _)) = predict_with_model_excluding(
                    model,
                    second_camera,
                    first_camera,
                    second.observation.pixel,
                    options.anchor_neighbour_count,
                    track_index,
                ) else {
                    continue;
                };
                let forward_error = distance(predicted_second, second.observation.pixel);
                let reverse_error = distance(predicted_first, first.observation.pixel);
                let symmetric =
                    ((forward_error * forward_error + reverse_error * reverse_error) * 0.5).sqrt();
                if !symmetric.is_finite() || symmetric > threshold {
                    continue;
                }
                let (Some(first_plane), Some(second_plane)) = (
                    cameras[first_camera].luminance,
                    cameras[second_camera].luminance,
                ) else {
                    continue;
                };
                let Some(score) = warped_zncc(
                    first_plane,
                    second_plane,
                    first.observation.pixel,
                    second.observation.pixel,
                    affine,
                    options.anchor_patch_radius_luma,
                ) else {
                    continue;
                };
                if score >= minimum_score {
                    accepted.push((track_index, first_camera, second_camera));
                }
            }
        }
    }
    for (track_index, first, second) in &accepted {
        tracks[*track_index]
            .direct_edges
            .insert(ordered_pair(*first, *second));
        if let Some(observation) = observation_for_camera_mut(&mut tracks[*track_index], *first) {
            confirm_observation(observation);
        }
        if let Some(observation) = observation_for_camera_mut(&mut tracks[*track_index], *second) {
            confirm_observation(observation);
        }
    }
    accepted.len()
}

fn proposal_for_missing_observation(
    track_index: usize,
    target_camera: usize,
    tracks: &[GraphTrack],
    models: &[PairModel],
    active_edges: &HashSet<(usize, usize)>,
    geometry: &[ResolvedCamera],
    triangulated: &[Option<Vec3>],
    round: usize,
    options: &RigRefinementOptions,
) -> Option<(usize, Vec2, Affine2, bool)> {
    let track = &tracks[track_index];
    let mut proposals = Vec::<(usize, Vec2, Affine2, f64)>::new();
    for source in &track.observations {
        if source.membership == ObservationMembership::SoftOutlier {
            continue;
        }
        let source_camera = source.observation.camera;
        if source_camera == target_camera
            || !active_edges.contains(&ordered_pair(source_camera, target_camera))
        {
            continue;
        }
        let Some(model) = model_for_pair(models, source_camera, target_camera) else {
            continue;
        };
        if let Some((predicted, affine, pair_rms)) = predict_with_model(
            model,
            source_camera,
            target_camera,
            source.observation.pixel,
            options.anchor_neighbour_count,
        ) {
            proposals.push((source_camera, predicted, affine, pair_rms));
        }
    }
    if proposals.is_empty() {
        return None;
    }
    proposals.sort_by(|first, second| first.3.total_cmp(&second.3));
    let best = proposals[0];
    let mut predicted = best.1;
    let mut geometry_consensus = false;
    if round >= 2 {
        if let Some(point) = triangulated.get(track_index).and_then(|point| *point) {
            if let Some(projected) = geometry
                .get(target_camera)
                .and_then(|camera| camera.project(point))
            {
                if distance(projected, predicted) <= options.anchor_geometry_consensus_px {
                    predicted = [
                        0.5 * (predicted[0] + projected[0]),
                        0.5 * (predicted[1] + projected[1]),
                    ];
                    geometry_consensus = true;
                }
            }
        }
    }
    Some((best.0, predicted, best.2, geometry_consensus))
}

/// Build the anchor graph and grow it for up to `anchor_max_rounds` rounds.
pub(crate) fn build_anchor_tracks(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    factory_cameras: &[ResolvedCamera],
    intrinsics_mode: IntrinsicsMode,
    validation_modulus: u64,
    options: &RigRefinementOptions,
) -> AnchorGraphBuild {
    let mut report = AnchorGraphReport::default();
    let all_factory_edges = factory_edges(cameras, factory_cameras, options);
    let candidate_edges = all_factory_edges
        .iter()
        .copied()
        .filter(|edge| edge.overlap + 1.0e-9 >= options.anchor_min_factory_overlap)
        .collect::<Vec<_>>();
    let maximum_active_edges = options.anchor_max_active_edges.min(candidate_edges.len());
    // A minimum degree d needs at least ceil(N*d/2) edges in an ordinary
    // graph. Keep one extra edge of slack for greedy/parity traps, and treat
    // the CLI initial-edge value as a lower bound rather than permission to
    // silently miss the requested topology.
    let degree_floor_edges = cameras
        .len()
        .saturating_mul(options.anchor_min_camera_degree)
        .div_ceil(2)
        .saturating_add(if options.anchor_min_camera_degree > 1 {
            1
        } else {
            0
        });
    let initial_target = options
        .anchor_initial_active_edges
        .max(degree_floor_edges)
        .min(maximum_active_edges);
    let mut active_edges = HashSet::<(usize, usize)>::new();
    let empty_support = vec![0usize; cameras.len()];
    let empty_coverage = vec![0.0f64; cameras.len()];
    let empty_verified_degree = vec![0usize; cameras.len()];
    let empty_shared_support = HashMap::<(usize, usize), usize>::new();
    // Do not pre-spend the topology budget on a B4 star. Reference-target
    // correspondences remain free bootstrap observations, but direct graph
    // edges are selected globally so every camera can participate in a
    // connected, cycle-rich graph.
    let initially_activated = activate_best_edges(
        &candidate_edges,
        &mut active_edges,
        &empty_support,
        &empty_coverage,
        &empty_verified_degree,
        &empty_shared_support,
        initial_target,
        options.anchor_min_camera_degree,
        usize::MAX,
    );
    let mut activation_round = HashMap::<(usize, usize), usize>::new();
    for edge in &active_edges {
        activation_round.insert(*edge, 0);
    }
    for edge in initially_activated {
        activation_round.insert(edge, 0);
    }
    report.factory_overlap_threshold = options.anchor_min_factory_overlap;
    report.candidate_edges = candidate_edges.len();
    report.maximum_active_edges = maximum_active_edges;
    report.target_min_camera_degree = options.anchor_min_camera_degree;

    let corners = cameras
        .iter()
        .map(|camera| camera.luminance.map(detect_rig_corners).unwrap_or_default())
        .collect::<Vec<_>>();
    let (mut tracks, bootstrap_matches) = bootstrap_graph(
        cameras,
        reference_index,
        alignments,
        factory_cameras,
        &active_edges,
        options,
        &mut report,
    );
    // The old implementation only *labelled* non-reference edges active; it
    // never matched those image pairs.  Seed every unsupported active edge now
    // using factory geometry only as a wide proposal and mutual image evidence
    // as the actual identity test.
    let mut direct_seed_attempted = HashSet::<(usize, usize)>::new();
    let mut direct_seed_attempt_counts = HashMap::<(usize, usize), usize>::new();
    let mut direct_seed_matches = HashMap::<(usize, usize), usize>::new();
    let mut initial_edges = active_edges.iter().copied().collect::<Vec<_>>();
    initial_edges.sort_unstable();
    let (mut bootstrap_direct_seed_edges, mut bootstrap_direct_seed_matches) =
        seed_active_edges_directly(
            &mut tracks,
            &initial_edges,
            &mut direct_seed_attempted,
            &mut direct_seed_attempt_counts,
            &mut direct_seed_matches,
            &corners,
            cameras,
            factory_cameras,
            reference_index,
            0,
            options,
        );
    promote_cycle_core_observations(&mut tracks);
    // A failed "active" edge is worse than an inactive candidate because it
    // consumes topology budget while providing no usable pair field. Retire it
    // and try another factory-valid edge. This is especially important for a
    // sky-heavy module such as C4: it should get several chances to connect
    // through whichever neighbours actually contain shared structure.
    let mut failed_direct_edges = HashSet::<(usize, usize)>::new();
    retire_failed_direct_edges(
        &tracks,
        &mut active_edges,
        &direct_seed_attempted,
        &mut failed_direct_edges,
        options.anchor_min_pair_anchors,
    );
    while active_edges.len() < initial_target {
        let support = camera_promoted_support(&tracks, cameras.len());
        let coverage = camera_spatial_coverage(&tracks, cameras);
        let coverage_fraction = coverage.iter().map(|entry| entry.1).collect::<Vec<_>>();
        let verified_degree =
            camera_verified_degrees(&tracks, &active_edges, cameras.len(), options);
        let available_edges = candidate_edges
            .iter()
            .copied()
            .filter(|edge| !failed_direct_edges.contains(&ordered_pair(edge.first, edge.second)))
            .collect::<Vec<_>>();
        let shared_support = candidate_shared_support(&tracks, &available_edges);
        let activated = activate_best_edges(
            &available_edges,
            &mut active_edges,
            &support,
            &coverage_fraction,
            &verified_degree,
            &shared_support,
            initial_target,
            options.anchor_min_camera_degree,
            initial_target,
        );
        if activated.is_empty() {
            break;
        }
        for edge in &activated {
            activation_round.entry(*edge).or_insert(0);
        }
        let (seeded_edges, seeded_matches) = seed_active_edges_directly(
            &mut tracks,
            &activated,
            &mut direct_seed_attempted,
            &mut direct_seed_attempt_counts,
            &mut direct_seed_matches,
            &corners,
            cameras,
            factory_cameras,
            reference_index,
            0,
            options,
        );
        promote_cycle_core_observations(&mut tracks);
        bootstrap_direct_seed_edges += seeded_edges;
        bootstrap_direct_seed_matches += seeded_matches;
        let retired = retire_failed_direct_edges(
            &tracks,
            &mut active_edges,
            &direct_seed_attempted,
            &mut failed_direct_edges,
            options.anchor_min_pair_anchors,
        );
        if retired == 0 && active_edges.len() >= initial_target {
            break;
        }
    }
    report.initial_active_edges = active_edges.len();
    report.bootstrap_direct_seed_edges = bootstrap_direct_seed_edges;
    report.bootstrap_direct_seed_matches = bootstrap_direct_seed_matches;
    report.initial_tracks = tracks.len();
    report.initial_three_plus_tracks = tracks
        .iter()
        .filter(|track| track.observations.len() >= 3)
        .count();
    report.initial_cycle_supported_three_plus_tracks = strong_three_plus_tracks(&tracks);
    let mut propagated_observations = 0usize;
    let mut promoted_observations = 0usize;
    let mut spawned_tracks = 0usize;
    let mut stopped_early = false;

    for round in 1..=options.anchor_max_rounds {
        // A failed edge is not permanently impossible. As other camera pairs
        // become connected, the same pair can acquire enough shared physical
        // tracks to be verified empirically even if its factory-only bootstrap
        // failed. Re-arm only edges that now have enough shared observations,
        // avoiding repeated expensive retries on genuinely empty overlaps.
        let retryable = failed_direct_edges
            .iter()
            .copied()
            .filter(|&(first, second)| {
                pair_anchors(&tracks, first, second, false).len() >= options.anchor_min_pair_anchors
            })
            .collect::<Vec<_>>();
        for edge in retryable {
            failed_direct_edges.remove(&edge);
            direct_seed_attempted.remove(&edge);
        }

        let active_edges_before = active_edges.len();
        let observations_before = observation_count(&tracks);
        let strong_before = strong_three_plus_tracks(&tracks);
        let initial_models = pair_models(&tracks, cameras.len(), options);
        if initial_models.is_empty() {
            stopped_early = true;
            break;
        }
        let closed_cycle_edges = close_existing_cycle_edges(
            &mut tracks,
            &initial_models,
            cameras,
            &active_edges,
            round,
            options,
        );
        promote_cycle_core_observations(&mut tracks);
        // Cycle closure can promote observations and authorize additional pair
        // fields.  Rebuild immediately so the same round can benefit from that
        // newly verified context.
        let models = pair_models(&tracks, cameras.len(), options);
        if models.is_empty() {
            stopped_early = true;
            break;
        }
        let (
            mut geometry,
            mut geometry_fit_tracks,
            mut geometry_optimizer_iterations,
            mut geometry_free_parameters,
            mut skeleton_landmarks,
        ) = intermediate_geometry(
            &tracks,
            cameras,
            reference_index,
            intrinsics_mode,
            factory_cameras,
            validation_modulus,
            round,
            options,
        );

        // V6 still seeded every newly activated/sparse direct edge against the
        // factory rig, even after this round had produced a much better
        // bearing solution. That disproportionately stranded narrow-FOV C
        // modules whose factory bearing error can exceed their useful local
        // search field. Re-seed sparse *active* edges once using the current
        // intermediate geometry, then rebuild the geometry if new identities
        // were recovered.
        let observations_before_geometry_reseed = observation_count(&tracks);
        let (geometry_reseed_edges, geometry_reseed_matches) = reseed_under_supported_active_edges(
            &mut tracks,
            &active_edges,
            &mut direct_seed_attempted,
            &mut direct_seed_attempt_counts,
            &mut direct_seed_matches,
            &corners,
            cameras,
            &geometry,
            reference_index,
            round,
            options,
        );
        if geometry_reseed_matches > 0 {
            promote_cycle_core_observations(&mut tracks);
            let rebuilt = intermediate_geometry(
                &tracks,
                cameras,
                reference_index,
                intrinsics_mode,
                factory_cameras,
                validation_modulus,
                round,
                options,
            );
            geometry = rebuilt.0;
            geometry_fit_tracks = rebuilt.1;
            geometry_optimizer_iterations = geometry_optimizer_iterations.saturating_add(rebuilt.2);
            geometry_free_parameters = rebuilt.3;
            skeleton_landmarks = rebuilt.4;
        }
        let geometry_reseed_observations =
            observation_count(&tracks).saturating_sub(observations_before_geometry_reseed);
        // Direct reseeding can materially improve the local pair field. Do not
        // continue propagation with the pre-reseed constellation models.
        let mut models = if geometry_reseed_matches > 0 {
            pair_models(&tracks, cameras.len(), options)
        } else {
            models
        };
        if models.is_empty() {
            stopped_early = true;
            break;
        }

        // Judge *observations*, not cameras. Each measurement is checked by a
        // leave-one-observation-out 3-D prediction when depth is sufficiently
        // conditioned and, independently, by a leave-one-landmark-out scene
        // constellation. Rejection remains reversible across rounds.
        let membership_update = if round >= 2 {
            update_graph_observation_membership(
                &mut tracks,
                &geometry,
                &models,
                validation_modulus,
                options,
            )
        } else {
            MembershipUpdate::default()
        };
        let membership_changed = membership_update.soft_outliers > 0
            || membership_update.reactivated > 0
            || membership_update.hard_rejected > 0
            || membership_update.split_tracks > 0;
        if membership_changed {
            promote_cycle_core_observations(&mut tracks);
            models = pair_models(&tracks, cameras.len(), options);
            if models.is_empty() {
                stopped_early = true;
                break;
            }
            let rebuilt = intermediate_geometry(
                &tracks,
                cameras,
                reference_index,
                intrinsics_mode,
                factory_cameras,
                validation_modulus,
                round,
                options,
            );
            geometry = rebuilt.0;
            geometry_fit_tracks = rebuilt.1;
            geometry_optimizer_iterations = geometry_optimizer_iterations.saturating_add(rebuilt.2);
            geometry_free_parameters = rebuilt.3;
            skeleton_landmarks = rebuilt.4;
        }
        let points = triangulated_points(&tracks, &geometry, options);
        let snapshot_len = tracks.len();
        let mut additions = Vec::<(usize, usize, usize, CandidateObservation, bool)>::new();
        let mut geometry_proposals = 0usize;
        let mut geometry_consensus = 0usize;

        for track_index in 0..snapshot_len {
            for target_camera in 0..cameras.len() {
                if !cameras[target_camera].match_evidence_enabled
                    || observation_for_camera(&tracks[track_index], target_camera).is_some()
                {
                    continue;
                }
                let Some((source_camera, predicted, affine, used_geometry)) =
                    proposal_for_missing_observation(
                        track_index,
                        target_camera,
                        &tracks,
                        &models,
                        &active_edges,
                        &geometry,
                        &points,
                        round,
                        options,
                    )
                else {
                    continue;
                };
                if round >= 2 {
                    geometry_proposals += 1;
                    if used_geometry {
                        geometry_consensus += 1;
                    }
                }
                let Some(source) = observation_for_camera(&tracks[track_index], source_camera)
                else {
                    continue;
                };
                let Some(reverse_model) = model_for_pair(&models, source_camera, target_camera)
                else {
                    continue;
                };
                let Some(candidate) = find_candidate(
                    source_camera,
                    target_camera,
                    source.observation.pixel,
                    predicted,
                    affine,
                    reverse_model,
                    Some(track_index),
                    &corners,
                    cameras,
                    round,
                    options,
                ) else {
                    continue;
                };
                if occupied(
                    &tracks,
                    target_camera,
                    candidate.pixel,
                    options.anchor_collision_radius_px,
                ) {
                    continue;
                }
                let round_index = round.saturating_sub(1) as f64;
                let strict_score = (options.anchor_min_zncc + 0.04 + 0.015 * round_index).min(0.98);
                let strict_margin =
                    options.anchor_min_appearance_margin + 0.012 + 0.004 * round_index;
                let strict_closure = (options.anchor_reverse_max_error_px * 0.55
                    / (1.0 + 0.08 * round_index))
                    .max(0.75);
                let promoted = candidate.score >= strict_score
                    && candidate.identity_confidence >= 0.58
                    && candidate.constellation_error <= options.anchor_constellation_max_error_px
                    && (candidate.margin >= strict_margin
                        || candidate.conditional_competitors == 0)
                    && candidate.closure_error <= strict_closure;
                additions.push((
                    track_index,
                    source_camera,
                    target_camera,
                    candidate,
                    promoted,
                ));
            }
        }

        additions.sort_by(|first, second| {
            first
                .0
                .cmp(&second.0)
                .then(first.2.cmp(&second.2))
                .then_with(|| second.3.score.total_cmp(&first.3.score))
        });
        let mut seen_addition = HashSet::new();
        let mut round_promoted = geometry_reseed_observations;
        let mut round_new_observations = geometry_reseed_observations;
        for (track_index, source_camera, target_camera, candidate, promoted) in additions {
            if !seen_addition.insert((track_index, target_camera)) {
                continue;
            }
            if observation_for_camera(&tracks[track_index], target_camera).is_some() {
                continue;
            }
            if occupied(
                &tracks,
                target_camera,
                candidate.pixel,
                options.anchor_collision_radius_px,
            ) {
                continue;
            }
            tracks[track_index].observations.push(graph_observation(
                TrackObservation {
                    camera: target_camera,
                    pixel: candidate.pixel,
                    bootstrap_residual_proposal: [0.0, 0.0],
                    localization_covariance: candidate.covariance,
                    fixed_gauge: target_camera == reference_index,
                    confidence: candidate.identity_confidence,
                    local_scale: candidate.local_scale,
                    structure: candidate.structure,
                    depth_reliability: None,
                    prepared: Default::default(),
                },
                promoted,
                round,
                candidate.identity_confidence,
            ));
            tracks[track_index]
                .direct_edges
                .insert(ordered_pair(source_camera, target_camera));
            round_new_observations += 1;
            if promoted {
                round_promoted += 1;
            }
        }

        let mut round_new_tracks = 0usize;
        for model in &models {
            if model.anchors.len() < options.anchor_min_pair_anchors
                || model.loo_rms > options.anchor_spawn_pair_max_loo_rms_px
            {
                continue;
            }
            let first = model.first;
            let second = model.second;
            if !active_edges.contains(&ordered_pair(first, second)) {
                continue;
            }
            if !cameras[first].match_evidence_enabled || !cameras[second].match_evidence_enabled {
                continue;
            }
            let mut added_for_pair = 0usize;
            for source_corner in &corners[first] {
                if added_for_pair >= options.anchor_new_tracks_per_pair_per_round {
                    break;
                }
                let source_pixel = corner_sensor(source_corner);
                if occupied(
                    &tracks,
                    first,
                    source_pixel,
                    options.anchor_new_track_spacing_px,
                ) {
                    continue;
                }
                let Some((predicted, affine, _)) = predict_with_model(
                    model,
                    first,
                    second,
                    source_pixel,
                    options.anchor_neighbour_count,
                ) else {
                    continue;
                };
                let Some(candidate) = find_candidate(
                    first,
                    second,
                    source_pixel,
                    predicted,
                    affine,
                    model,
                    None,
                    &corners,
                    cameras,
                    round,
                    options,
                ) else {
                    continue;
                };
                if occupied(
                    &tracks,
                    second,
                    candidate.pixel,
                    options.anchor_new_track_spacing_px,
                ) {
                    continue;
                }
                let round_index = round.saturating_sub(1) as f64;
                let promoted = candidate.score
                    >= (options.anchor_min_zncc + 0.05 + 0.015 * round_index).min(0.98)
                    && candidate.identity_confidence >= 0.62
                    && candidate.constellation_error
                        <= options.anchor_constellation_max_error_px * 0.85
                    && (candidate.margin
                        >= options.anchor_min_appearance_margin + 0.015 + 0.004 * round_index
                        || candidate.conditional_competitors == 0)
                    && candidate.closure_error <= options.anchor_reverse_max_error_px * 0.50;
                if !promoted {
                    continue;
                }
                let first_observation = graph_observation(
                    correspondence_observation(
                        first,
                        source_pixel,
                        first == reference_index,
                        candidate.identity_confidence,
                        1.0,
                        f64::from(source_corner.structure),
                        source_corner.covariance,
                    ),
                    true,
                    round,
                    candidate.identity_confidence,
                );
                let second_observation = graph_observation(
                    TrackObservation {
                        camera: second,
                        pixel: candidate.pixel,
                        bootstrap_residual_proposal: [0.0, 0.0],
                        localization_covariance: candidate.covariance,
                        fixed_gauge: second == reference_index,
                        confidence: candidate.identity_confidence,
                        local_scale: candidate.local_scale,
                        structure: candidate.structure,
                        depth_reliability: None,
                        prepared: Default::default(),
                    },
                    true,
                    round,
                    candidate.identity_confidence,
                );
                let mut direct_edges = HashSet::new();
                direct_edges.insert(ordered_pair(first, second));
                tracks.push(GraphTrack {
                    key: track_key(if first == reference_index {
                        source_pixel
                    } else if second == reference_index {
                        candidate.pixel
                    } else {
                        source_pixel
                    }),
                    observations: vec![first_observation, second_observation],
                    direct_edges,
                });
                added_for_pair += 1;
                round_new_tracks += 1;
                round_new_observations += 2;
                round_promoted += 2;
            }
        }

        // Grow the topology after the current propagation pass.  Crucially,
        // a new edge no longer needs an already-existing transitive pair
        // model: that circular requirement was why C4 could never acquire its
        // first real non-reference connection.  Newly activated edges are
        // matched directly from the images immediately.
        let next_target = active_edges
            .len()
            .saturating_add(options.anchor_edges_per_round)
            .min(maximum_active_edges);
        let observations_before_direct_seed = observation_count(&tracks);
        let mut activated = Vec::<(usize, usize)>::new();
        let mut directly_seeded_edges = 0usize;
        let mut directly_seeded_matches = 0usize;
        // Direct image verification can legitimately reject an edge (sky,
        // occlusion, repetitive texture).  A rejected edge must not leave a
        // permanent hole in the topology.  Refill within the same round, but
        // cap replacement attempts so pathological scenes cannot explode the
        // matching cost.
        let mut activation_attempt_budget = options.anchor_edges_per_round.saturating_mul(3);
        if options.anchor_edges_per_round > 0 {
            while active_edges.len() < next_target && activation_attempt_budget > 0 {
                let support_now = camera_promoted_support(&tracks, cameras.len());
                let coverage_now = camera_spatial_coverage(&tracks, cameras);
                let coverage_fraction_now =
                    coverage_now.iter().map(|entry| entry.1).collect::<Vec<_>>();
                let verified_degree_now =
                    camera_verified_degrees(&tracks, &active_edges, cameras.len(), options);
                let available_edges = candidate_edges
                    .iter()
                    .copied()
                    .filter(|edge| {
                        !failed_direct_edges.contains(&ordered_pair(edge.first, edge.second))
                    })
                    .collect::<Vec<_>>();
                let want = next_target.saturating_sub(active_edges.len());
                let batch_limit = want.min(activation_attempt_budget).max(1);
                let shared_support_now = candidate_shared_support(&tracks, &available_edges);
                let batch = activate_best_edges(
                    &available_edges,
                    &mut active_edges,
                    &support_now,
                    &coverage_fraction_now,
                    &verified_degree_now,
                    &shared_support_now,
                    next_target,
                    options.anchor_min_camera_degree,
                    batch_limit,
                );
                if batch.is_empty() {
                    break;
                }
                activation_attempt_budget = activation_attempt_budget.saturating_sub(batch.len());
                for edge in &batch {
                    activation_round.entry(*edge).or_insert(round);
                }
                let (seeded_edges, seeded_matches) = seed_active_edges_directly(
                    &mut tracks,
                    &batch,
                    &mut direct_seed_attempted,
                    &mut direct_seed_attempt_counts,
                    &mut direct_seed_matches,
                    &corners,
                    cameras,
                    &geometry,
                    reference_index,
                    round,
                    options,
                );
                promote_cycle_core_observations(&mut tracks);
                directly_seeded_edges += seeded_edges;
                directly_seeded_matches += seeded_matches;
                activated.extend(batch);
                retire_failed_direct_edges(
                    &tracks,
                    &mut active_edges,
                    &direct_seed_attempted,
                    &mut failed_direct_edges,
                    options.anchor_min_pair_anchors,
                );
            }
        }
        let direct_seed_observations =
            observation_count(&tracks).saturating_sub(observations_before_direct_seed);
        round_new_observations += direct_seed_observations;
        round_promoted += direct_seed_observations;

        let observations_after = observation_count(&tracks);
        let strong_after = strong_three_plus_tracks(&tracks);
        let observation_growth = observations_after.saturating_sub(observations_before) as f64
            / observations_before.max(1) as f64;
        let strong_growth =
            strong_after.saturating_sub(strong_before) as f64 / strong_before.max(1) as f64;
        let stop = activated.is_empty()
            && observation_growth < options.anchor_min_observation_growth_fraction
            && strong_growth < options.anchor_min_strong_track_growth_fraction;
        report.rounds.push(AnchorGraphRoundReport {
            round,
            active_edges_before,
            activated_edges: activated.len(),
            directly_seeded_edges,
            directly_seeded_matches,
            geometry_reseeded_edges: geometry_reseed_edges,
            geometry_reseeded_matches: geometry_reseed_matches,
            active_edges_after: active_edges.len(),
            observations_before,
            observations_after,
            new_observations: round_new_observations,
            promoted_observations: round_promoted,
            new_tracks: round_new_tracks,
            strong_three_plus_before: strong_before,
            strong_three_plus_after: strong_after,
            observation_growth_fraction: observation_growth,
            strong_track_growth_fraction: strong_growth,
            geometry_projection_proposals: geometry_proposals,
            geometry_projection_consensus: geometry_consensus,
            geometry_fit_tracks,
            geometry_optimizer_iterations,
            geometry_free_parameters,
            skeleton_landmarks,
            soft_outlier_observations: membership_update.soft_outliers,
            reactivated_observations: membership_update.reactivated,
            hard_rejected_observations: membership_update.hard_rejected,
            split_tracks: membership_update.split_tracks,
            closed_cycle_edges,
            stopped_early: stop,
        });
        report.rounds_run = round;
        propagated_observations += round_new_observations;
        promoted_observations += round_promoted;
        spawned_tracks += round_new_tracks;
        if stop {
            stopped_early = true;
            break;
        }
    }

    let final_models = pair_models(&tracks, cameras.len(), options);
    report.pairs = final_models
        .iter()
        .map(|model| AnchorGraphPairReport {
            first_camera: cameras[model.first].name.to_owned(),
            second_camera: cameras[model.second].name.to_owned(),
            anchors: pair_anchors(&tracks, model.first, model.second, true).len(),
            validated_anchors: model.anchors.len(),
            loo_rms_px: model.loo_rms,
            loo_median_px: model.loo_median,
        })
        .collect();
    report.stopped_early = stopped_early;
    report.propagated_observations = propagated_observations;
    report.promoted_observations = promoted_observations;
    report.spawned_tracks = spawned_tracks;
    report.final_soft_outlier_observations = tracks
        .iter()
        .flat_map(|track| &track.observations)
        .filter(|observation| observation.membership == ObservationMembership::SoftOutlier)
        .count();
    report.final_hard_rejected_observations = report
        .rounds
        .iter()
        .map(|round| round.hard_rejected_observations)
        .sum();
    report.split_tracks = report.rounds.iter().map(|round| round.split_tracks).sum();
    report.final_active_edges = active_edges.len();

    report.graph_edges = all_factory_edges
        .iter()
        .map(|edge| {
            let pair = ordered_pair(edge.first, edge.second);
            let model = model_for_pair(&final_models, edge.first, edge.second);
            AnchorGraphEdgeReport {
                first_camera: cameras[edge.first].name.to_owned(),
                second_camera: cameras[edge.second].name.to_owned(),
                factory_overlap: edge.overlap,
                factory_baseline: edge.baseline,
                candidate: edge.overlap + 1.0e-9 >= options.anchor_min_factory_overlap,
                active: active_edges.contains(&pair),
                activated_round: activation_round.get(&pair).copied(),
                shared_tracks: shared_track_support(&tracks, pair),
                direct_support_tracks: direct_support_tracks(&tracks, pair),
                direct_seed_attempted: direct_seed_attempt_counts.get(&pair).copied().unwrap_or(0)
                    > 0,
                direct_seed_attempts: direct_seed_attempt_counts.get(&pair).copied().unwrap_or(0),
                direct_seed_matches: direct_seed_matches.get(&pair).copied().unwrap_or(0),
                validated_anchors: model.map_or(0, |model| model.anchors.len()),
                loo_rms_px: model.map(|model| model.loo_rms),
            }
        })
        .collect();
    let final_coverage = camera_spatial_coverage(&tracks, cameras);
    // A topology edge is "verified" when it has enough independently
    // image-verified direct tracks.  A validated local pair field is stronger
    // evidence and is still reported separately through `validated_anchors`,
    // but it is deliberately not required here: cross-focal/high-parallax
    // B/C edges can be geometrically excellent while a single local-affine
    // field has high LOO error.
    let final_verified_edges = candidate_edges
        .iter()
        .map(|edge| ordered_pair(edge.first, edge.second))
        .filter(|&edge| direct_support_tracks(&tracks, edge) >= options.anchor_min_pair_anchors)
        .collect::<HashSet<_>>();
    report.graph_cameras = (0..cameras.len())
        .map(|camera| AnchorGraphCameraReport {
            camera: cameras[camera].name.to_owned(),
            active_degree: edge_degree(&active_edges, camera),
            verified_degree: edge_degree(&final_verified_edges, camera),
            triangle_edges: triangle_edges_for_camera(&final_verified_edges, camera),
            promoted_observations: tracks
                .iter()
                .filter_map(|track| observation_for_camera(track, camera))
                .filter(|observation| observation.promoted)
                .count(),
            occupied_coverage_cells: final_coverage.get(camera).map_or(0, |entry| entry.0),
            spatial_coverage_fraction: final_coverage.get(camera).map_or(0.0, |entry| entry.1),
            direct_support_tracks: tracks
                .iter()
                .filter(|track| {
                    observation_for_camera(track, camera).is_some()
                        && track
                            .direct_edges
                            .iter()
                            .any(|&(first, second)| first == camera || second == camera)
                })
                .count(),
            cycle_supported_tracks: tracks
                .iter()
                .filter(|track| {
                    observation_for_camera(track, camera).is_some()
                        && cycle_core_cameras(track).contains(&camera)
                })
                .count(),
        })
        .collect();

    // The reference camera fixes the global parameter gauge, but individual
    // physical tracks need not observe it. Keeping non-reference tracks is the
    // whole point of the interconnected graph: C<->C and B<->C evidence can
    // constrain those cameras through the connected camera topology.
    let ordinary = tracks
        .iter()
        .filter_map(|track| graph_track_to_track(track, true))
        .collect::<Vec<_>>();
    report.final_tracks = ordinary.len();
    report.final_three_plus_tracks = ordinary
        .iter()
        .filter(|track| track.observations.len() >= 3)
        .count();
    // Every returned 3+ view track survived the explicit cycle gate above.
    report.final_cycle_supported_three_plus_tracks = report.final_three_plus_tracks;
    report.final_observations = ordinary.iter().map(|track| track.observations.len()).sum();
    let final_matches = ordinary
        .iter()
        .map(|track| track.observations.len().saturating_sub(1))
        .sum::<usize>();
    AnchorGraphBuild {
        tracks: ordinary,
        pairwise_matches: final_matches.max(bootstrap_matches),
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translated_anchors(alias: Option<(usize, f64)>) -> Vec<PairAnchor> {
        (0..12)
            .map(|index| {
                let x = (index % 4) as f64 * 40.0 + 20.0;
                let y = (index / 4) as f64 * 35.0 + 15.0;
                let mut target = [1.08 * x + 0.04 * y + 13.0, -0.02 * x + 0.97 * y - 7.0];
                if let Some((bad, offset)) = alias {
                    if bad == index {
                        target[0] += offset;
                    }
                }
                PairAnchor {
                    track: index,
                    first: [x, y],
                    second: target,
                }
            })
            .collect()
    }

    #[test]
    fn local_affine_predicts_unseen_point() {
        let anchors = translated_anchors(None);
        let query = [72.0, 58.0];
        let affine = local_affine(&anchors, query, 8, None, false).unwrap();
        let predicted = affine.apply(query);
        let expected = [
            1.08 * query[0] + 0.04 * query[1] + 13.0,
            -0.02 * query[0] + 0.97 * query[1] - 7.0,
        ];
        assert!(
            distance(predicted, expected) < 1.0e-6,
            "{predicted:?} != {expected:?}"
        );
    }

    #[test]
    fn leave_one_out_rejects_twenty_pixel_periodic_alias() {
        let anchors = translated_anchors(Some((5, 20.0)));
        let model = validate_pair_anchors(anchors, 8, 3.0, 6).unwrap();
        assert!(model.anchors.iter().all(|anchor| anchor.track != 5));
        assert!(model.loo_rms < 1.0e-5, "{}", model.loo_rms);
    }

    #[test]
    fn star_is_not_a_cycle_but_closed_triangle_is() {
        let observation = |camera| GraphObservation {
            observation: TrackObservation {
                camera,
                pixel: [camera as f64 * 10.0, 20.0],
                bootstrap_residual_proposal: [0.0, 0.0],
                localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                fixed_gauge: camera == 0,
                confidence: 1.0,
                local_scale: 1.0,
                structure: 1.0,
                depth_reliability: None,
                prepared: Default::default(),
            },
            promoted: true,
            round: 0,
            membership: ObservationMembership::Inlier,
            bad_iterations: 0,
            good_iterations: 0,
            loo_reprojection_px: None,
            constellation_loo_px: None,
            identity_confidence: 1.0,
        };
        let mut track = GraphTrack {
            key: [0, 0],
            observations: vec![observation(0), observation(1), observation(2)],
            direct_edges: HashSet::from([(0, 1), (0, 2)]),
        };
        assert!(!has_cycle(&track));
        track.direct_edges.insert((1, 2));
        assert!(has_cycle(&track));
    }

    #[test]
    fn edge_selector_builds_balanced_redundant_topology() {
        let camera_count = 6usize;
        let mut candidates = Vec::new();
        for first in 0..camera_count {
            for second in first + 1..camera_count {
                candidates.push(FactoryEdge {
                    first,
                    second,
                    overlap: 0.80 - 0.01 * (first + second) as f64,
                    baseline: 10.0 + (first + second) as f64,
                });
            }
        }
        let mut active = HashSet::new();
        let support = vec![0usize; camera_count];
        let coverage = vec![0.0f64; camera_count];
        let verified_degree = vec![0usize; camera_count];
        let activated = activate_best_edges(
            &candidates,
            &mut active,
            &support,
            &coverage,
            &verified_degree,
            &HashMap::new(),
            10,
            3,
            usize::MAX,
        );
        assert_eq!(activated.len(), 10);
        assert_eq!(active.len(), 10);
        for camera in 0..camera_count {
            assert!(
                edge_degree(&active, camera) >= 3,
                "camera {camera} should reach the requested redundant degree"
            );
        }
        assert!(
            active
                .iter()
                .filter(|&&(first, second)| common_active_neighbours(&active, first, second) > 0)
                .count()
                >= 3,
            "the selected graph should contain short cycles"
        );
    }

    #[test]
    fn eleven_camera_selector_is_connected_and_cycle_supported() {
        let camera_count = 11usize;
        let mut candidates = Vec::new();
        for first in 0..camera_count {
            for second in first + 1..camera_count {
                candidates.push(FactoryEdge {
                    first,
                    second,
                    overlap: 0.95 - 0.01 * ((first + second) % 9) as f64,
                    baseline: 12.0 + ((7 * first + 11 * second) % 90) as f64,
                });
            }
        }
        let mut active = HashSet::new();
        let support = vec![0usize; camera_count];
        let coverage = vec![0.0f64; camera_count];
        let verified_degree = vec![0usize; camera_count];
        activate_best_edges(
            &candidates,
            &mut active,
            &support,
            &coverage,
            &verified_degree,
            &HashMap::new(),
            18,
            3,
            usize::MAX,
        );
        assert_eq!(active.len(), 18);
        for camera in 0..camera_count {
            assert!(
                edge_degree(&active, camera) >= 3,
                "camera {camera} degree={} should be at least three",
                edge_degree(&active, camera)
            );
            assert!(
                triangle_edges_for_camera(&active, camera) > 0,
                "camera {camera} should participate in a short cycle"
            );
            assert!(active_connected(&active, 0, camera));
        }
    }

    #[test]
    fn cycle_core_drops_dangling_camera_observation() {
        let observation = |camera| GraphObservation {
            observation: TrackObservation {
                camera,
                pixel: [camera as f64 * 11.0 + 3.0, 17.0],
                bootstrap_residual_proposal: [0.0, 0.0],
                localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                fixed_gauge: camera == 0,
                confidence: 1.0,
                local_scale: 1.0,
                structure: 1.0,
                depth_reliability: None,
                prepared: Default::default(),
            },
            promoted: true,
            round: 0,
            membership: ObservationMembership::Inlier,
            bad_iterations: 0,
            good_iterations: 0,
            loo_reprojection_px: None,
            constellation_loo_px: None,
            identity_confidence: 1.0,
        };
        let track = GraphTrack {
            key: [0, 0],
            observations: vec![
                observation(0),
                observation(1),
                observation(2),
                observation(3),
            ],
            direct_edges: HashSet::from([(0, 1), (1, 2), (2, 0), (2, 3)]),
        };
        let core = cycle_core_cameras(&track);
        assert_eq!(core, HashSet::from([0, 1, 2]));
        let ordinary = graph_track_to_track(&track, true).unwrap();
        assert_eq!(ordinary.observations.len(), 3);
        assert!(
            ordinary
                .observations
                .iter()
                .all(|observation| observation.camera != 3)
        );
    }

    #[test]
    fn cycle_core_promotion_requires_two_independent_neighbours() {
        let observation = |camera| GraphObservation {
            observation: TrackObservation {
                camera,
                pixel: [camera as f64 * 13.0, 9.0],
                bootstrap_residual_proposal: [0.0, 0.0],
                localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                fixed_gauge: camera == 0,
                confidence: 1.0,
                local_scale: 1.0,
                structure: 1.0,
                depth_reliability: None,
                prepared: Default::default(),
            },
            promoted: false,
            round: 0,
            membership: ObservationMembership::Provisional,
            bad_iterations: 0,
            good_iterations: 0,
            loo_reprojection_px: None,
            constellation_loo_px: None,
            identity_confidence: 1.0,
        };
        let mut tracks = vec![GraphTrack {
            key: [0, 0],
            observations: vec![observation(0), observation(1), observation(2)],
            direct_edges: HashSet::from([(0, 1), (1, 2)]),
        }];
        assert_eq!(promote_cycle_core_observations(&mut tracks), 0);
        tracks[0].direct_edges.insert((0, 2));
        assert_eq!(promote_cycle_core_observations(&mut tracks), 3);
        assert!(
            tracks[0]
                .observations
                .iter()
                .all(|observation| observation.promoted)
        );
    }

    #[test]
    fn direct_epipolar_evidence_counts_as_verified_topology_without_affine_field() {
        let observation = |camera, index: usize| GraphObservation {
            observation: TrackObservation {
                camera,
                pixel: [
                    index as f64 * 17.0 + camera as f64,
                    20.0 + index as f64 * 3.0,
                ],
                bootstrap_residual_proposal: [0.0, 0.0],
                localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                fixed_gauge: camera == 0,
                confidence: 0.9,
                local_scale: 1.0,
                structure: 1.0,
                depth_reliability: None,
                prepared: Default::default(),
            },
            promoted: true,
            round: 1,
            membership: ObservationMembership::Inlier,
            bad_iterations: 0,
            good_iterations: 0,
            loo_reprojection_px: None,
            constellation_loo_px: None,
            identity_confidence: 1.0,
        };
        let tracks = (0..8)
            .map(|index| GraphTrack {
                key: [index as i32, 0],
                observations: vec![observation(0, index), observation(1, index)],
                direct_edges: HashSet::from([(0, 1)]),
            })
            .collect::<Vec<_>>();
        let active = HashSet::from([(0, 1)]);
        let options = RigRefinementOptions {
            anchor_min_pair_anchors: 6,
            ..Default::default()
        };
        let degrees = camera_verified_degrees(&tracks, &active, 2, &options);
        assert_eq!(degrees, vec![1, 1]);
    }

    #[test]
    fn relative_constellation_rejects_repeated_pattern_shift() {
        let mut model = validate_pair_anchors(translated_anchors(None), 8, 3.0, 6).unwrap();
        model.first = 0;
        model.second = 1;
        let source = [72.0, 58.0];
        let affine = local_affine(&model.anchors, source, 8, None, false).unwrap();
        let correct = affine.apply(source);
        let correct_error =
            constellation_relative_error(&model, 0, 1, source, correct, affine, 8, None).unwrap();
        let shifted_error = constellation_relative_error(
            &model,
            0,
            1,
            source,
            [correct[0] + 20.0, correct[1]],
            affine,
            8,
            None,
        )
        .unwrap();
        assert!(correct_error < 1.0e-5, "{correct_error}");
        assert!(shifted_error > 15.0, "{shifted_error}");
    }

    #[test]
    fn coherent_soft_outlier_component_is_split_not_deleted() {
        let make = |camera: usize| {
            graph_observation(
                TrackObservation {
                    camera,
                    pixel: [20.0 * camera as f64, 10.0],
                    bootstrap_residual_proposal: [0.0, 0.0],
                    localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    fixed_gauge: camera == 0,
                    confidence: 0.9,
                    local_scale: 1.0,
                    structure: 1.0,
                    depth_reliability: None,
                    prepared: Default::default(),
                },
                true,
                1,
                0.9,
            )
        };
        let mut observations = vec![make(0), make(1), make(2), make(3)];
        for observation in &mut observations[2..] {
            observation.promoted = false;
            observation.membership = ObservationMembership::SoftOutlier;
            observation.bad_iterations = 2;
        }
        let mut tracks = vec![GraphTrack {
            key: [0, 0],
            observations,
            direct_edges: HashSet::from([(0, 1), (0, 2), (1, 3), (2, 3)]),
        }];
        let mut options = RigRefinementOptions::default();
        options.held_out_validation = false;
        let update = split_soft_outlier_components(&mut tracks, 5, &options);
        assert_eq!(update.split_tracks, 1);
        assert_eq!(tracks.len(), 2);
        assert!(tracks.iter().all(|track| track.observations.len() == 2));
        assert!(
            tracks[1]
                .observations
                .iter()
                .all(|observation| observation.membership == ObservationMembership::Provisional)
        );
    }

    #[test]
    fn held_out_track_is_not_split_by_membership_update() {
        let make = |camera: usize| {
            graph_observation(
                TrackObservation {
                    camera,
                    pixel: [20.0 * camera as f64, 10.0],
                    bootstrap_residual_proposal: [0.0, 0.0],
                    localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    fixed_gauge: camera == 0,
                    confidence: 0.9,
                    local_scale: 1.0,
                    structure: 1.0,
                    depth_reliability: None,
                    prepared: Default::default(),
                },
                true,
                1,
                0.9,
            )
        };
        let mut options = RigRefinementOptions::default();
        options.held_out_validation = true;
        // Find a deterministic key that belongs to a validation block rather
        // than coupling the test to stable_track_hash implementation details.
        let validation_modulus = 5;
        let mut key = [0, 0];
        loop {
            let probe = GraphTrack {
                key,
                observations: vec![make(0), make(1)],
                direct_edges: HashSet::from([(0, 1)]),
            };
            if graph_track_is_validation(&probe, validation_modulus, &options) {
                break;
            }
            key[0] += 16;
        }
        let mut observations = vec![make(0), make(1), make(2), make(3)];
        for observation in &mut observations[2..] {
            observation.promoted = false;
            observation.membership = ObservationMembership::SoftOutlier;
            observation.bad_iterations = 4;
        }
        let mut tracks = vec![GraphTrack {
            key,
            observations,
            direct_edges: HashSet::from([(0, 1), (0, 2), (1, 3), (2, 3)]),
        }];
        let update = split_soft_outlier_components(&mut tracks, validation_modulus, &options);
        assert_eq!(update.split_tracks, 0);
        assert_eq!(update.hard_rejected, 0);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].observations.len(), 4);
    }

    #[test]
    fn cycle_promotion_does_not_override_soft_outlier_membership() {
        let make = |camera: usize| {
            graph_observation(
                TrackObservation {
                    camera,
                    pixel: [10.0 * camera as f64, 5.0],
                    bootstrap_residual_proposal: [0.0, 0.0],
                    localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    fixed_gauge: camera == 0,
                    confidence: 0.9,
                    local_scale: 1.0,
                    structure: 1.0,
                    depth_reliability: None,
                    prepared: Default::default(),
                },
                false,
                1,
                0.9,
            )
        };
        let mut track = GraphTrack {
            key: [0, 0],
            observations: vec![make(0), make(1), make(2), make(3)],
            direct_edges: HashSet::from([(0, 1), (1, 2), (0, 2), (2, 3)]),
        };
        track.observations[3].membership = ObservationMembership::SoftOutlier;
        let mut tracks = vec![track];
        promote_cycle_core_observations(&mut tracks);
        assert!(tracks[0].observations[0].promoted);
        assert!(tracks[0].observations[1].promoted);
        assert!(tracks[0].observations[2].promoted);
        assert!(!tracks[0].observations[3].promoted);
    }
}
