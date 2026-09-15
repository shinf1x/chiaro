//! Capture-specific refinement of the physical multi-camera rig.
//!
//! Rig fitting prefers a sparse point-feature population generated independently
//! of the dense/grid image warp. Reference observations are genuine Shi-Tomasi
//! corners; target observations must pass 2-D structure, NCC peak-uniqueness and
//! forward/backward closure checks. Pairwise populations are then filtered in
//! distortion-corrected calibrated coordinates by a rank-2 epipolar RANSAC.
//! This is intentionally separate from the semi-dense physical/depth matcher:
//! repeated industrial structure can produce a photometrically excellent but
//! wrong depth match. The physical matcher remains available as a fallback
//! when pairwise evidence is too sparse. A fit-only epipolar pass estimates
//! bearing/mirror corrections before any candidate-dependent triangulation, so
//! a bad factory pose cannot delete the very observations needed to recover it.
//! The resulting tracks are then triangulated and used to fit orientation,
//! virtual-centre translation, sensor-raster
//! offset, and movable-mirror state corrections. The reference camera is the
//! fixed gauge. Debug runs reserve an independently held-out track subset for
//! validation; production runs use every usable track for the final estimate.

mod anchor_graph;
mod latent_graph;

use std::{
    collections::{HashMap, HashSet},
    thread,
};

use serde::Serialize;

use anchor_graph::build_anchor_tracks;
pub use anchor_graph::{
    AnchorGraphCameraReport, AnchorGraphEdgeReport, AnchorGraphPairReport, AnchorGraphReport,
    AnchorGraphRoundReport,
};
pub use latent_graph::{LatentCameraSupportReport, LatentMatchReport, LatentMatchRoundReport};
use latent_graph::{
    LatentCandidateState, LatentMembershipState, build_latent_tracks,
    update_assignments as update_latent_assignments,
};

use crate::{
    align::{AlignmentCorrespondence, ModuleAlignment},
    calibration::{CameraCalibration, IntrinsicsMode, MirrorAngleMode, ModuleState},
    geometry::{CameraRefinement, ResolvedCamera, ResolvedCameraTemplate},
    image::Plane,
    math::{self, Mat3, Vec2, Vec3, add, cross, dot, mul_vec, norm, normalize, scale, sub},
};

// A failed/physically nonsensical reprojection must remain visible to held-out
// validation instead of disappearing from the sample population.  Keep the
// penalty finite so one bad track cannot create 1e14-pixel RMS through Brown
// polynomial extrapolation, while still making any such sample incompatible
// with the ~1 px acceptance target.
const INVALID_REPROJECTION_PENALTY_PX: f64 = 64.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RigRefinementStrategy {
    #[default]
    Physical,
    AnchorGraph,
    /// Keep several visually plausible observations per camera and alternate
    /// correspondence assignment with the physical rig solve.
    LatentGraph,
}

fn select_solved_candidate(strategy: RigRefinementStrategy, validation_passed: bool) -> bool {
    validation_passed || strategy == RigRefinementStrategy::LatentGraph
}

#[derive(Clone, Debug)]
pub struct RigRefinementOptions {
    /// Run capture-specific rig refinement before the residual image warp.
    pub enabled: bool,
    /// Correspondence/rig strategy. `Physical` keeps the V10 path;
    /// `AnchorGraph` grows a classical constellation graph; `LatentGraph`
    /// carries several repeated-structure hypotheses into alternating bundle
    /// and discrete correspondence-assignment passes.
    pub strategy: RigRefinementStrategy,
    /// Maximum anchor/constellation propagation rounds after the bootstrap.
    pub anchor_max_rounds: usize,
    /// Candidate camera pairs whose factory-calibrated overlap covers at least
    /// this fraction of the smaller field are eligible for direct graph edges.
    pub anchor_min_factory_overlap: f64,
    /// Number of direct camera edges activated before the first propagation
    /// round and the hard ceiling after adaptive graph growth.
    pub anchor_initial_active_edges: usize,
    pub anchor_max_active_edges: usize,
    /// Desired minimum degree in the active camera graph when factory overlap
    /// makes it possible. The selector favours short cycles after satisfying
    /// weak-camera degree deficits.
    pub anchor_min_camera_degree: usize,
    /// Maximum number of new direct camera edges activated after one round.
    pub anchor_edges_per_round: usize,
    /// Stop when both newly observed points and cycle-supported 3+ view tracks
    /// grow by less than these fractions in one round.
    pub anchor_min_observation_growth_fraction: f64,
    pub anchor_min_strong_track_growth_fraction: f64,
    /// Number of nearby anchors used for each local affine constellation.
    pub anchor_neighbour_count: usize,
    /// Minimum promoted anchors needed to authorize a camera-pair field.
    pub anchor_min_pair_anchors: usize,
    /// Leave-one-out gates in native sensor pixels.
    pub anchor_seed_loo_max_error_px: f64,
    pub anchor_pair_loo_max_error_px: f64,
    /// Target search radius around the constellation prediction. Later rounds
    /// tighten this automatically down to `anchor_min_search_radius_px`.
    pub anchor_search_radius_px: f64,
    pub anchor_min_search_radius_px: f64,
    /// Factory-guided search radius used only when an active camera edge has
    /// no existing pair field.  This direct seeding step is what makes a
    /// newly activated non-reference edge real image evidence rather than a
    /// merely decorative graph edge.
    pub anchor_direct_seed_search_radius_px: f64,
    /// Direct factory-guided seeds are deliberately stricter than propagated
    /// points because they establish a new camera-pair identity field.
    pub anchor_direct_seed_min_zncc: f64,
    pub anchor_direct_seed_min_margin: f64,
    /// Bound direct-seeding cost per edge. Corners are already response-sorted.
    pub anchor_direct_seed_max_corners: usize,
    pub anchor_direct_seed_max_matches: usize,
    /// Full-res sensor closure error under the reverse local constellation.
    pub anchor_reverse_max_error_px: f64,
    /// ZNCC patch radius in the 2x2-CFA-cell luminance plane. This is /2, not
    /// the earlier /4 neural working raster.
    pub anchor_patch_radius_luma: usize,
    pub anchor_min_zncc: f64,
    pub anchor_min_appearance_margin: f64,
    /// Explicit landmark-constellation gate.  This compares the relative
    /// arrangement of several independently established neighbouring anchors,
    /// not only the candidate patch itself.  It is therefore able to reject a
    /// repeated local texture whose surrounding scene layout is wrong.
    pub anchor_constellation_max_error_px: f64,
    /// Appearance ambiguity may be tolerated when the constellation is very
    /// strong.  This is the maximum number of near-best candidates allowed in
    /// the geometry-supported search domain before a candidate is considered
    /// conditionally non-unique.
    pub anchor_max_conditional_competitors: usize,
    /// Observation membership is deliberately reversible.  A measurement can
    /// be demoted when its leave-one-out reprojection/constellation evidence is
    /// bad, then promoted again after the global rig improves.
    pub anchor_observation_loo_soft_px: f64,
    pub anchor_observation_loo_hard_px: f64,
    pub anchor_observation_bad_iterations: usize,
    pub anchor_observation_recovery_iterations: usize,
    /// A round-2+ 3-D proposal may tighten a search only when it agrees with
    /// the independently inferred 2-D constellation within this distance.
    pub anchor_geometry_consensus_px: f64,
    /// Spatial de-duplication for newly recovered observations/tracks.
    pub anchor_collision_radius_px: f64,
    pub anchor_new_track_spacing_px: f64,
    /// New points are spawned only on pair fields whose LOO RMS is below this
    /// limit, with a deterministic per-pair cap per round.
    pub anchor_spawn_pair_max_loo_rms_px: f64,
    pub anchor_new_tracks_per_pair_per_round: usize,

    /// Merge independently matched reference observations into one latent
    /// landmark when their reference-sensor locations are within this radius.
    pub latent_reference_merge_radius_px: f64,
    /// De-duplicate target-side candidate landmarks in one camera.
    pub latent_candidate_dedup_radius_px: f64,
    /// Do not add a second candidate that is merely another localization of
    /// the currently selected target feature.
    pub latent_candidate_min_separation_px: f64,
    /// Maximum candidates retained per track/camera observation, including the
    /// initially selected correspondence.
    pub latent_max_candidates: usize,
    /// Maximum additional Shi-Tomasi target landmarks admitted to the
    /// per-camera candidate pool.  Already matched sub-pixel landmarks are
    /// always retained even if they exceed this count.
    pub latent_candidate_pool_max_corners: usize,
    /// Minimum normalized target-side self-similarity for an alternative.
    pub latent_min_appearance_similarity: f64,
    /// Alternatives must have approximately the same signed factory epipolar
    /// residual as the initial match.  Comparing residual *differences* keeps
    /// this permissive to capture-wide factory bearing error.
    pub latent_candidate_epipolar_band_px: f64,
    /// Maximum alternating assignment/bundle passes.
    pub latent_max_assignment_iterations: usize,
    /// LatentGraph membership is reversible rather than destructive. Keep at
    /// least this many active fit observations per target camera when enough
    /// observations were available initially.
    pub latent_membership_min_camera_observations: usize,
    /// Also preserve this fraction of each camera's initial fit support. The
    /// larger of the absolute and fractional floors is used, capped by the
    /// camera's initial support.
    pub latent_membership_min_camera_fraction: f64,
    /// An inactive observation is promoted again once leave-one-camera-out
    /// reprojection falls below this reference-equivalent error. This is lower
    /// than the demotion threshold to provide hysteresis.
    pub latent_membership_recovery_reference_px: f64,
    /// If a camera falls below its support floor, the best dormant observations
    /// may be restored up to this (looser) error. This prevents narrow-FOV
    /// cameras from collapsing to a handful of observations while still
    /// refusing grossly inconsistent identities.
    pub latent_membership_floor_max_reference_px: f64,
    /// Intrinsically two-view latent tracks cannot be leave-one-camera-out
    /// tested. Instead they use a symmetric calibrated epipolar residual and
    /// become dormant as a whole when this reference-equivalent error is too
    /// large. They can re-enter after later rig updates below the recovery
    /// threshold.
    pub latent_pairwise_membership_max_reference_px: f64,
    pub latent_pairwise_membership_recovery_reference_px: f64,
    /// Relative bundle-objective authority of intrinsically two-view tracks.
    /// They still provide epipolar information, but cannot steer calibration
    /// as strongly as a 3+-view track with independent leave-one-out support.
    pub latent_pairwise_bundle_weight: f64,
    /// Frozen held-out labels with an alternative this similar are excluded
    /// from validation (and from fit, because the whole spatial hold-out block
    /// remains isolated). They remain fully available in ordinary fit blocks.
    pub latent_validation_max_alternative_similarity: f64,
    /// Additional image-only gates for a latent held-out observation.  These
    /// are evaluated before any candidate rig exists, so pruning an uncertain
    /// label here cannot leak fitted geometry into validation.
    pub latent_validation_min_confidence: f64,
    pub latent_validation_min_peak_margin: f64,
    pub latent_validation_max_forward_backward_px: f64,
    /// Image-only local-constellation disagreement allowed for a frozen
    /// validation label, in reference-equivalent pixels. This is evaluated
    /// before any rig candidate exists and removes high-confidence but
    /// structurally impossible repeated-feature labels from the validation
    /// ground truth.
    pub latent_validation_max_constellation_error_px: f64,
    /// Nearby already-selected landmarks used to fit the local affine
    /// constellation prediction for one candidate observation.
    pub latent_neighbour_count: usize,
    /// Relative weight of the local constellation error in candidate scoring.
    pub latent_constellation_weight: f64,
    /// Build a bootstrap-only multi-camera anchor graph and use its
    /// cycle-supported image correspondences as an additional, image-derived
    /// latent assignment constraint. This does not replace the latent
    /// reference tracks or run a second rig solver: only the graph's direct
    /// image evidence is consumed by candidate scoring.
    pub latent_cycle_graph_enabled: bool,
    /// Maximum direct camera edges used by the bootstrap cycle graph.
    pub latent_cycle_graph_max_edges: usize,
    /// Minimum number of cycle-supported 3+-view anchor tracks required before
    /// one camera pair contributes a local cross-camera prediction.
    pub latent_cycle_min_pair_anchors: usize,
    /// Nearby cycle anchors used for the local affine pair prediction.
    pub latent_cycle_neighbour_count: usize,
    /// Relative weight of multi-camera cycle consistency in candidate scoring.
    pub latent_cycle_weight: f64,
    /// Candidate switches with available cycle evidence must remain within
    /// this native-sensor error unless ordinary leave-one-camera-out geometry
    /// independently supports them.
    pub latent_cycle_max_error_px: f64,
    /// Convert (1-self-similarity) into a small pixel-equivalent prior cost.
    pub latent_appearance_penalty_px: f64,
    /// Clip individual geometric/constellation score terms so one bad current
    /// iterate cannot dominate all appearance evidence.
    pub latent_score_error_cap_px: f64,
    /// Required score improvement before an assignment may switch.
    pub latent_switch_margin_px: f64,
    /// At least one of the geometry or constellation gates below must support a
    /// proposed switch.
    pub latent_max_reprojection_px: f64,
    pub latent_max_constellation_error_px: f64,

    /// Worker threads used by independent physical reference candidates
    /// (`0` = all available cores).
    pub threads: usize,
    /// Reserve spatially separated tracks for independent validation instead
    /// of using them to estimate the capture rig. The production pipeline
    /// keeps this enabled so physical geometry must generalise before it is
    /// allowed to drive dense depth.
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
    /// Dimensionless shared B/C-group scale applied to CRA/Hall focus travel
    /// along each physical camera's optical axis. Zero disables the nested
    /// physical pupil-shift experiment.
    pub max_focus_pupil_scale: f64,
    /// Strict calibration-raster origin correction bound per sensor axis.
    pub max_sensor_offset_px: f64,
    /// LatentGraph common focal correction bound as a fractional change from
    /// factory (0.02 = +/-2%).
    pub max_focal_scale_delta: f64,
    /// Additional C-camera focal anisotropy bound. The resolved model applies
    /// `fx *= 1+s+a`, `fy *= 1+s-a` to first order, where `s` is the common
    /// scale and `a` this aspect term.
    pub max_focal_aspect_delta: f64,
    /// Additive capture-local Brown radial/tangential coefficient bounds. These
    /// are exposed only for LatentGraph C modules with factory distortion and
    /// still have to pass the retriangulated observability/rank test.
    pub max_distortion_k1_delta: f64,
    pub max_distortion_k2_delta: f64,
    pub max_distortion_tangential_delta: f64,
    /// Independent displacement of the factory Brown/OpenCV distortion centre
    /// from the principal-point/raster correction, in native sensor pixels.
    /// LatentGraph exposes this only for C modules with calibrated distortion.
    pub max_distortion_center_offset_px: f64,
    /// Gaussian factory prior scale for each orientation component.
    pub orientation_prior_sigma_degrees: f64,
    /// Gaussian factory prior scale for the movable-mirror angle.
    pub mirror_prior_sigma_degrees: f64,
    /// Gaussian factory prior scale for each optical-centre axis.
    pub center_prior_sigma: f64,
    pub focus_pupil_prior_sigma: f64,
    /// Gaussian factory prior scale for each sensor-raster axis.
    pub sensor_offset_prior_sigma_px: f64,
    /// Gaussian prior sigma for the fractional common focal-scale correction.
    pub focal_scale_prior_sigma: f64,
    /// Gaussian priors for C-camera focal anisotropy and capture-local Brown
    /// coefficient corrections.
    pub focal_aspect_prior_sigma: f64,
    pub distortion_k1_prior_sigma: f64,
    pub distortion_k2_prior_sigma: f64,
    pub distortion_tangential_prior_sigma: f64,
    pub distortion_center_prior_sigma_px: f64,
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
    /// Fit-only membership gate in reference-equivalent pixels.  This is
    /// intentionally tighter than the old 6 px diagnostic p95 line: once
    /// latent identity has had a chance to switch, observations several pixels
    /// away should not keep steering a rig whose target accuracy is ~1 px.
    pub fit_membership_max_reference_px: f64,
    /// Absolute held-out quality gate in native target-sensor pixels.  Rig
    /// geometry is only production-worthy once the independent population is
    /// genuinely around/sub-pixel; a huge relative improvement from a broken
    /// factory projection is not sufficient.
    pub max_validation_rms_px: f64,
    /// Tail guards keep a low mean from hiding repeated-structure failures.
    pub max_validation_p90_px: f64,
    pub max_validation_p95_px: f64,
    /// Per-camera validation prevents a weak/narrow-FOV module from hiding
    /// behind the global population. Every camera with frozen validation evidence
    /// needs at least this many observations and must satisfy the RMS/p90 gate.
    pub min_validation_camera_samples: usize,
    pub max_validation_camera_rms_px: f64,
    pub max_validation_camera_p90_px: f64,
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

    /// Allow the calibrated semi-dense physical/depth matcher as a fallback
    /// when independently epipolar-verified pairwise image correspondences do
    /// not provide a sufficient fit/validation population.
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
    /// Smallest native-sensor refinement step after a physical match has been
    /// selected. The matcher repeatedly halves its 3x3 search step down to
    /// this value, so the final observation is not quantized to the coarse
    /// bootstrap grid.
    pub physical_match_subpixel_step_px: f64,
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
            strategy: RigRefinementStrategy::Physical,
            anchor_max_rounds: 4,
            anchor_min_factory_overlap: 0.20,
            anchor_initial_active_edges: 18,
            anchor_max_active_edges: 24,
            anchor_min_camera_degree: 3,
            anchor_edges_per_round: 3,
            anchor_min_observation_growth_fraction: 0.01,
            anchor_min_strong_track_growth_fraction: 0.01,
            anchor_neighbour_count: 10,
            anchor_min_pair_anchors: 12,
            anchor_seed_loo_max_error_px: 6.0,
            anchor_pair_loo_max_error_px: 4.0,
            anchor_search_radius_px: 16.0,
            anchor_min_search_radius_px: 6.0,
            anchor_direct_seed_search_radius_px: 96.0,
            anchor_direct_seed_min_zncc: 0.72,
            anchor_direct_seed_min_margin: 0.035,
            anchor_direct_seed_max_corners: 1200,
            anchor_direct_seed_max_matches: 700,
            anchor_reverse_max_error_px: 5.0,
            anchor_patch_radius_luma: 5,
            anchor_min_zncc: 0.68,
            anchor_min_appearance_margin: 0.025,
            anchor_constellation_max_error_px: 4.5,
            anchor_max_conditional_competitors: 2,
            anchor_observation_loo_soft_px: 4.0,
            anchor_observation_loo_hard_px: 10.0,
            anchor_observation_bad_iterations: 3,
            anchor_observation_recovery_iterations: 1,
            anchor_geometry_consensus_px: 12.0,
            anchor_collision_radius_px: 3.0,
            anchor_new_track_spacing_px: 6.0,
            anchor_spawn_pair_max_loo_rms_px: 2.5,
            anchor_new_tracks_per_pair_per_round: 160,
            latent_reference_merge_radius_px: 1.5,
            latent_candidate_dedup_radius_px: 2.0,
            latent_candidate_min_separation_px: 6.0,
            latent_max_candidates: 6,
            latent_candidate_pool_max_corners: 6000,
            latent_min_appearance_similarity: 0.72,
            latent_candidate_epipolar_band_px: 12.0,
            latent_max_assignment_iterations: 6,
            latent_membership_min_camera_observations: 64,
            latent_membership_min_camera_fraction: 0.35,
            latent_membership_recovery_reference_px: 1.25,
            latent_membership_floor_max_reference_px: 6.0,
            latent_pairwise_membership_max_reference_px: 2.5,
            latent_pairwise_membership_recovery_reference_px: 1.5,
            latent_pairwise_bundle_weight: 0.25,
            latent_validation_max_alternative_similarity: 0.90,
            latent_validation_min_confidence: 0.70,
            latent_validation_min_peak_margin: 0.035,
            latent_validation_max_forward_backward_px: 0.45,
            latent_validation_max_constellation_error_px: 6.0,
            latent_neighbour_count: 10,
            latent_constellation_weight: 0.35,
            latent_cycle_graph_enabled: true,
            latent_cycle_graph_max_edges: 24,
            latent_cycle_min_pair_anchors: 10,
            latent_cycle_neighbour_count: 10,
            latent_cycle_weight: 0.45,
            latent_cycle_max_error_px: 6.0,
            latent_appearance_penalty_px: 4.0,
            latent_score_error_cap_px: 24.0,
            latent_switch_margin_px: 0.35,
            latent_max_reprojection_px: 12.0,
            latent_max_constellation_error_px: 8.0,
            threads: 0,
            held_out_validation: true,
            min_tracks: 80,
            min_validation_tracks: 20,
            // A C camera contributes up to fifteen physical correction
            // parameters after v9.5, and the subsequent finite-difference observability,
            // robust membership, bounds, and factory priors independently
            // gate them. Requiring 150 observations excluded every narrow-FOV
            // C module before those checks even ran on ordinary captures.
            min_camera_observations: 48,
            validation_fraction: 0.20,
            validation_block_size_px: 256,
            max_iterations: 6,
            max_membership_iterations: 3,
            // Factory geometry is an excellent physical prior, but capture-
            // state mirror/pose error can be several degrees.  The previous
            // +/-0.5 degree box plus a very stiff 0.2 degree prior made such
            // solutions mathematically unreachable even when image evidence
            // was unambiguous.  Held-out validation and per-camera regression
            // checks remain the acceptance gate in diagnostic runs.
            max_orientation_degrees: 4.0,
            // Large bearing errors are observable from image geometry; large
            // centre/raster changes are not, especially for distant scenes.
            // Keep those nuisance parameters near factory so they cannot
            // absorb an orientation correction (or explode on weak parallax).
            max_mirror_degrees: 2.0,
            max_center_offset: 5.0,
            max_focus_pupil_scale: 0.0,
            max_sensor_offset_px: 64.0,
            max_focal_scale_delta: 0.02,
            max_focal_aspect_delta: 0.01,
            max_distortion_k1_delta: 0.03,
            max_distortion_k2_delta: 0.08,
            max_distortion_tangential_delta: 0.008,
            max_distortion_center_offset_px: 12.0,
            orientation_prior_sigma_degrees: 3.0,
            mirror_prior_sigma_degrees: 0.35,
            center_prior_sigma: 1.0,
            focus_pupil_prior_sigma: 0.5,
            sensor_offset_prior_sigma_px: 12.0,
            focal_scale_prior_sigma: 0.004,
            focal_aspect_prior_sigma: 0.002,
            distortion_k1_prior_sigma: 0.006,
            distortion_k2_prior_sigma: 0.015,
            distortion_tangential_prior_sigma: 0.0015,
            distortion_center_prior_sigma_px: 2.0,
            factory_prior_weight: 0.002,
            huber_delta: 2.5,
            max_triangulation_condition: 1.0e9,
            min_ray_angle_degrees: 0.003,
            max_initial_track_rms: 100.0,
            min_validation_improvement: 0.005,
            fit_membership_max_reference_px: 2.0,
            max_validation_rms_px: 1.0,
            max_validation_p90_px: 1.5,
            max_validation_p95_px: 2.5,
            min_validation_camera_samples: 12,
            max_validation_camera_rms_px: 1.25,
            max_validation_camera_p90_px: 2.0,
            min_positive_depth_fraction: 0.80,
            min_image_space_correction_improvement: 0.05,
            max_validation_p95_reference_px: 6.0,
            max_validation_p95_angular_degrees: 0.10,
            physical_matching: true,
            physical_match_stride_px: 32,
            physical_match_max_candidates: 6000,
            physical_match_patch_radius: 4,
            physical_match_planes: 48,
            physical_match_max_projected_step_px: 8.0,
            physical_match_depth_beam_width: 6,
            physical_match_max_depth_refinements: 6,
            physical_match_near_depth: 500.0,
            physical_match_far_depth: 10_000_000.0,
            physical_match_residual_radius_px: 32.0,
            physical_match_subpixel_step_px: 0.25,
            physical_match_min_score: 0.60,
            physical_match_min_structure: 0.008,
            physical_match_min_margin: 0.05,
            // Some 150-mm L16 modules share only a narrow overlap with the
            // 75-mm reference and may have no third camera seeing the same
            // patch. Two-view tracks are still valid epipolar/depth evidence;
            // the candidate ranking already rewards extra independent views
            // whenever they exist.
            physical_match_min_views: 2,
            physical_match_max_reprojection_reference_px: 6.0,
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

/// Capture-time factory-model audit kept beside every rig result. These are
/// inputs and competing model predictions, never optimized measurements.
#[derive(Clone, Debug, Serialize)]
pub struct RigFactoryModelReport {
    pub camera: String,
    pub capture_lens_hall: f64,
    pub calibrated_lens_hall_min: Option<f64>,
    pub calibrated_lens_hall_max: Option<f64>,
    /// Signed counts outside the calibrated interval; zero means in-range.
    pub lens_hall_extrapolation_counts: f64,
    pub resolved_k: Option<Mat3>,
    pub cra_sensor_distance: Option<f64>,
    pub cra_exit_pupil_distance: Option<f64>,
    pub cra_pixel_size: Option<f64>,
    pub cra_lens_hall: Option<f64>,
    pub cra_distance_hall_ratio: Option<f64>,
    /// `(capture_lens_hall + 1) * distance_hall_ratio`.
    pub cra_implied_capture_sensor_distance: Option<f64>,
    pub cra_radial_samples: Vec<Vec2>,
    pub cra_fitted_coefficients: Vec<Vec2>,
    pub cra_fit_cost: Option<f64>,
    /// `[x, y, width, height]` in the calibration sensor raster.
    pub cra_valid_roi: Option<[i32; 4]>,
    pub capture_mirror_hall: f64,
    pub af_mirror_hall: Option<f64>,
    pub quadratic_mirror_angle_degrees: Option<f64>,
    pub inverse_quadratic_mirror_angle_degrees: Option<f64>,
    pub pair_linear_mirror_angle_degrees: Option<f64>,
    pub selected_mirror_angle_degrees: Option<f64>,
    /// Pair-linear minus current-quadratic.
    pub pair_minus_quadratic_degrees: Option<f64>,
    pub inverse_minus_current_degrees: Option<f64>,
    /// First-order reflected-ray displacement, `2*f*abs(delta_angle)`.
    pub approximate_mirror_difference_px: Option<f64>,
    pub approximate_inverse_difference_px: Option<f64>,
    pub quadratic_use_rplus_left: Option<bool>,
    pub quadratic_use_rplus_right: Option<bool>,
    pub quadratic_inflection_value: Option<f64>,
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
    pub strategy: RigRefinementStrategy,
    /// True when the candidate is the geometry selected by the pipeline.
    /// LatentGraph selects every successfully solved candidate so that its
    /// result can be inspected without silently reverting to factory geometry.
    pub accepted: bool,
    /// Result of the independent held-out/physical-quality gate. This remains
    /// diagnostic for LatentGraph even though it no longer controls selection.
    pub validation_passed: bool,
    pub mirror_angle_mode: MirrorAngleMode,
    pub factory_model: Vec<RigFactoryModelReport>,
    /// Detailed matching/propagation diagnostics for the anchor-graph path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_graph: Option<AnchorGraphReport>,
    /// Candidate-generation and assignment-switch diagnostics for the latent
    /// multi-hypothesis strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_match: Option<LatentMatchReport>,
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
    /// Spatial hold-out tracks deliberately omitted because their frozen
    /// latent identity had a near-duplicate image candidate.
    pub validation_excluded_ambiguous_tracks: usize,
    /// Individual target observations removed from otherwise usable held-out
    /// latent tracks by image-only identity-quality gates.
    pub validation_excluded_ambiguous_observations: usize,
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
    /// Held-out RMS over target cameras only.  The fixed reference observation
    /// is useful for triangulation but must not dilute the camera-to-camera
    /// accuracy target by contributing a comparatively easy residual.
    pub held_out_target_rms_before: f64,
    pub held_out_target_rms_after: f64,
    pub held_out_target_relative_improvement: f64,
    pub held_out_target_residuals_before: RigResidualDistributionReport,
    pub held_out_target_residuals_after: RigResidualDistributionReport,
    /// RMS after removing only the worst 1% of held-out residual samples.
    /// The raw RMS above remains authoritative and is never replaced; this
    /// companion metric makes near-parallel/infinite-depth triangulation tails
    /// visible without letting a handful of numerically explosive samples hide
    /// the quality of the other 99% of the frozen validation population.
    pub held_out_p99_trimmed_rms_before: f64,
    pub held_out_p99_trimmed_rms_after: f64,
    /// Number of held-out sensor-space residuals above 50 native pixels.
    pub held_out_over_50px_before: usize,
    pub held_out_over_50px_after: usize,
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
    /// Pearson correlations of candidate held-out residual components with
    /// image position/radius and inverse depth. These are diagnostics only:
    /// spatial correlation points toward intrinsics/distortion, while depth
    /// correlation points toward baseline/centre error.
    pub residual_correlations: Vec<RigResidualCorrelationReport>,
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
    /// Why a selected LatentGraph candidate missed the validation gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_warning: Option<String>,
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
    /// Robust held-out tail metrics in the camera's native sensor pixels.
    /// These are acceptance gates for cameras with frozen validation evidence:
    /// one weak narrow-FOV module must not hide behind easy observations elsewhere.
    pub validation_median_after: f64,
    pub validation_p90_after: f64,
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
    pub p99: f64,
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
pub struct RigResidualCorrelationReport {
    pub camera: String,
    pub samples: usize,
    pub inverse_depth_samples: usize,
    pub dx_vs_x: f64,
    pub dx_vs_y: f64,
    pub dy_vs_x: f64,
    pub dy_vs_y: f64,
    pub dx_vs_r2: f64,
    pub dy_vs_r2: f64,
    pub dx_vs_inverse_depth: f64,
    pub dy_vs_inverse_depth: f64,
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
    pub factory_inverse_depth: f64,
    pub candidate_inverse_depth: f64,
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
    pub focus_pupil_scale: f64,
    pub sensor_offset_px: [f64; 2],
    /// Fractional common focal change from factory (0.001 = +0.1%).
    pub focal_scale_delta: f64,
    /// Additional focal anisotropy; approximately fx += aspect, fy -= aspect.
    pub focal_aspect_delta: f64,
    /// Independent Brown/OpenCV distortion-centre displacement relative to
    /// the raster/principal-point correction, in native sensor pixels.
    pub distortion_center_offset_px: [f64; 2],
    /// Additive `[dk1, dk2, dp1, dp2]` Brown coefficient corrections.
    pub distortion_delta: [f64; 4],
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
    prepared: std::sync::OnceLock<ObservationPrepared>,
}

#[derive(Clone, Copy, Debug)]
struct ObservationPrepared {
    sigma: f64,
    inverse_covariance: [[f64; 2]; 2],
}

#[derive(Clone, Debug)]
struct Track {
    key: [i32; 2],
    observations: Vec<TrackObservation>,
    condition: f64,
    max_ray_angle_degrees: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParameterKind {
    Orientation(usize),
    Mirror,
    Center(usize),
    FocusPupilScale(char),
    Sensor(usize),
    FocalScale,
    FocalAspect,
    DistortionCenter(usize),
    DistortionRadial(usize),
    DistortionTangential(usize),
}

#[derive(Clone, Copy, Debug)]
struct ParameterSpec {
    camera: usize,
    /// Cameras changed together by this parameter. Ordinary camera-local
    /// parameters contain exactly `camera`; shared pupil scales contain the
    /// complete focal group, including the reference camera when applicable.
    affected_cameras: u16,
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
    /// Inverse signed depth along this camera's observed ray, in reciprocal
    /// calibration distance units. NaN means the sample could not be given a
    /// physical leave-one-out depth.
    inverse_depth: f64,
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

/// Fit a bounded capture-specific physical model. The pipeline reserves an
/// independent deterministic spatial split in both diagnostic and production
/// runs; callers may still disable validation explicitly for synthetic tests.
pub fn refine_capture_rig(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    provisional_alignments: &[ModuleAlignment],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> RigRefinementOutcome {
    let mut report = RigRefinementReport {
        enabled: options.enabled,
        strategy: options.strategy,
        mirror_angle_mode: cameras
            .iter()
            .find_map(|camera| {
                camera
                    .calibration
                    .and_then(|calibration| calibration.mirror.as_ref())
                    .map(|mirror| mirror.actuator.angle_mode)
            })
            .unwrap_or_default(),
        factory_model: factory_model_report(cameras, intrinsics_mode),
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
    let validation_modulus = (1.0 / options.validation_fraction.clamp(0.05, 0.5)).round() as u64;
    let (raw_tracks, pairwise_matches, mut latent_candidates): (
        Vec<Track>,
        usize,
        Option<LatentCandidateState>,
    ) = match options.strategy {
        RigRefinementStrategy::AnchorGraph => {
            let graph = build_anchor_tracks(
                cameras,
                reference_index,
                provisional_alignments,
                &factory_cameras,
                intrinsics_mode,
                validation_modulus,
                options,
            );
            report.anchor_graph = Some(graph.report);
            report.physical_match_used = false;
            (graph.tracks, graph.pairwise_matches, None)
        }
        RigRefinementStrategy::LatentGraph => {
            let latent = build_latent_tracks(
                cameras,
                reference_index,
                provisional_alignments,
                &factory_cameras,
                intrinsics_mode,
                options,
            );
            report.latent_match = Some(latent.report);
            report.physical_match_used = false;
            (
                latent.tracks,
                latent.pairwise_matches,
                Some(latent.candidates),
            )
        }
        RigRefinementStrategy::Physical => {
            let (mut legacy_tracks, legacy_matches) = build_tracks(
                cameras,
                reference_index,
                provisional_alignments,
                &factory_cameras,
            );
            let legacy_validation_tracks = legacy_tracks
                .iter()
                .filter(|track| {
                    options.held_out_validation
                        && is_validation_track(track, validation_modulus, options)
                })
                .count();
            let legacy_fit_tracks = legacy_tracks.len() - legacy_validation_tracks;
            let legacy_population_is_sufficient = legacy_fit_tracks >= options.min_tracks
                && (!options.held_out_validation
                    || legacy_validation_tracks >= options.min_validation_tracks);

            let (raw_tracks, pairwise_matches) = if legacy_population_is_sufficient {
                // Prefer the independently rank-2/epipolar-verified pairwise
                // population. Besides being much denser, it avoids the repeated-depth
                // ambiguity of semi-dense physical matching and lets us skip that
                // expensive stage entirely on ordinary captures.
                report.physical_match_used = false;
                (legacy_tracks, legacy_matches)
            } else {
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
                report.physical_match_max_depth_refinement_levels =
                    physical.max_depth_refinement_levels;
                report.physical_match_depth_refinement_histogram =
                    physical.depth_refinement_histogram;
                report.physical_match_observed_max_projected_step_px =
                    physical.observed_max_projected_step_px;
                report.physical_match_per_camera =
                    physical_match_support_reports(cameras, &physical_tracks, reference_index);
                report.physical_match_rejected_no_supported_depth =
                    physical.rejected_no_supported_depth;
                report.physical_match_rejected_ambiguous_depth = physical.rejected_ambiguous_depth;
                report.physical_match_rejected_insufficient_views =
                    physical.rejected_insufficient_views;
                report.rejected_inconsistent_observations = physical.rejected_observations;
                report.rejected_inconsistent_tracks = physical.rejected_tracks;

                let physical_validation_tracks = physical_tracks
                    .iter()
                    .filter(|track| {
                        options.held_out_validation
                            && is_validation_track(track, validation_modulus, options)
                    })
                    .count();
                let physical_fit_tracks = physical_tracks.len() - physical_validation_tracks;
                let physical_population_is_sufficient = physical_fit_tracks >= options.min_tracks
                    && (!options.held_out_validation
                        || physical_validation_tracks >= options.min_validation_tracks);
                if physical_population_is_sufficient {
                    // Physical matching is a last-resort fallback. Supplement cameras
                    // with too little physical support using any verified pairwise
                    // evidence that survived the independent epipolar filter.
                    let mut physical_support = vec![0usize; cameras.len()];
                    for track in &physical_tracks {
                        for observation in &track.observations {
                            if observation.camera != reference_index {
                                physical_support[observation.camera] += 1;
                            }
                        }
                    }
                    let under_supported = physical_support
                        .iter()
                        .map(|&count| count < options.min_camera_observations)
                        .collect::<Vec<_>>();
                    let mut combined = physical_tracks;
                    let mut supplemental_matches = 0usize;
                    legacy_tracks.retain(|track| {
                        let target = track
                            .observations
                            .iter()
                            .find(|observation| observation.camera != reference_index)
                            .map(|observation| observation.camera);
                        let keep = target.map_or(false, |camera| under_supported[camera]);
                        if keep {
                            supplemental_matches += 1;
                        }
                        keep
                    });
                    combined.extend(legacy_tracks);
                    report.physical_match_used = true;
                    (combined, physical_observations + supplemental_matches)
                } else {
                    report.physical_match_used = false;
                    (legacy_tracks, legacy_matches)
                }
            };
            (raw_tracks, pairwise_matches, None)
        }
    };
    report.pairwise_matches = pairwise_matches;
    report.tracks = raw_tracks.len();
    report.tracks_three_plus = raw_tracks
        .iter()
        .filter(|track| track.observations.len() >= 3)
        .count();

    // Correspondence identity must be established independently of the rig
    // candidate. Do not throw image-verified tracks away merely because the
    // factory rig cannot triangulate them: that was happening before the
    // epipolar initializer and made the initializer mathematically unable to
    // recover the cameras with the largest factory error.
    let mut tracks = Vec::new();
    let mut rejected = 0usize;
    for track in raw_tracks {
        let finite = track.observations.iter().all(|observation| {
            observation.pixel[0].is_finite() && observation.pixel[1].is_finite()
        });
        // Anchor-graph tracks are allowed to connect non-reference cameras.
        // The reference camera still fixes the parameter gauge globally, but
        // requiring every physical point to include that camera discards the
        // very C<->C / B<->C support the graph was built to recover.
        if track.observations.len() < 2 || !finite {
            rejected += 1;
            continue;
        }
        tracks.push(track);
    }
    report.rejected_degenerate_tracks = rejected;
    report.rejected_outlier_tracks = 0;
    report.rejected_nonpositive_observations = 0;
    if tracks.len() < options.min_tracks {
        report.fallback_reason = Some(format!(
            "only {} independently verified tracks (need {})",
            tracks.len(),
            options.min_tracks
        ));
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }

    // Freeze the spatial held-out split before *any* candidate-dependent
    // geometry is estimated. The fit side may subsequently be triangulated,
    // pruned and membership-refined; validation correspondence membership is
    // never changed using the fitted rig.
    let latent_validation_selection = if options.held_out_validation
        && options.strategy == RigRefinementStrategy::LatentGraph
    {
        latent_candidates.as_ref().map(|candidates| {
            latent_validation_blocks(&tracks, candidates, cameras.len(), reference_index, options)
        })
    } else {
        None
    };
    let latent_validation_block_size_px = options.validation_block_size_px.div_ceil(2).max(64);

    // The cycle graph is image-only, but held-out validation is intended to be
    // blind even to correspondence identity from those spatial blocks. Remove
    // every graph anchor originating in the selected validation population
    // before the first latent assignment update. Candidate alternatives were
    // generated without cycle evidence, so this closes the remaining leakage
    // path while preserving target<->target constraints on the fit side.
    if let (Some(selected_blocks), Some(candidates)) = (
        latent_validation_selection.as_ref(),
        latent_candidates.as_mut(),
    ) {
        let removed = candidates.exclude_cycle_validation_blocks(
            selected_blocks,
            latent_validation_block_size_px,
            options.latent_cycle_min_pair_anchors,
        );
        if let Some(latent_report) = report.latent_match.as_mut() {
            latent_report.cycle_graph_validation_anchors_excluded = removed;
            latent_report.cycle_graph_pairs = candidates.cycle_graph_pair_count();
        }
    }

    let mut preliminary_validation = Vec::<Track>::new();
    let mut preliminary_fit = Vec::new();
    let mut validation_excluded_ambiguous_tracks = 0usize;
    let mut validation_excluded_ambiguous_observations = 0usize;
    for track in &tracks {
        let held_out_block = options.held_out_validation
            && if let Some(selected_blocks) = latent_validation_selection.as_ref() {
                selected_blocks.contains(&validation_block_key(
                    track,
                    latent_validation_block_size_px,
                ))
            } else {
                is_validation_track(track, validation_modulus, options)
            };
        if !held_out_block {
            preliminary_fit.push(track);
            continue;
        }
        if options.strategy == RigRefinementStrategy::LatentGraph {
            let filtered = latent_candidates.as_ref().and_then(|candidates| {
                candidates
                    .validation_track(track, options.latent_validation_max_alternative_similarity)
            });
            if let Some((validation_track, removed)) = filtered {
                validation_excluded_ambiguous_observations += removed;
                preliminary_validation.push(validation_track);
            } else {
                // Do not leak the same spatial block back into fit merely
                // because its frozen correspondence label is ambiguous.
                validation_excluded_ambiguous_tracks += 1;
            }
        } else {
            preliminary_validation.push(track.clone());
        }
    }
    report.validation_excluded_ambiguous_tracks = validation_excluded_ambiguous_tracks;
    report.validation_excluded_ambiguous_observations = validation_excluded_ambiguous_observations;
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

    // Phase 1: fit only bearing-changing parameters from robust world-bearing
    // alignment, using all independently verified FIT correspondences. For
    // this low-parallax capture an infinity/bearing objective is much better
    // conditioned than essential-matrix cheirality; nearby finite points are
    // tolerated by the robust loss and metric depth is solved only afterward.
    let bootstrap_specs = parameter_specs(cameras, reference_index, &preliminary_fit, options)
        .into_iter()
        .filter(|spec| {
            matches!(
                spec.kind,
                ParameterKind::Orientation(_) | ParameterKind::Mirror
            )
        })
        .collect::<Vec<_>>();
    let bootstrap_zero = vec![0.0; bootstrap_specs.len()];
    let (bootstrap_parameters, _, bootstrap_iterations) = if bootstrap_specs.is_empty() {
        (bootstrap_zero, f64::INFINITY, 0usize)
    } else {
        // Solve the inexpensive low-parallax bearing problem independently
        // per camera. This avoids the mirrored cheirality minima that allowed
        // C3 to jump to a physically implausible multi-degree epipolar basin.
        bearing_bootstrap(
            &bootstrap_specs,
            cameras,
            &preliminary_fit,
            reference_index,
            intrinsics_mode,
            options,
        )
    };
    let bootstrap_refinements =
        refinements_from_parameters(cameras.len(), &bootstrap_parameters, &bootstrap_specs);
    let Some(bootstrap_cameras) = resolve_cameras(cameras, &bootstrap_refinements, intrinsics_mode)
    else {
        report.fallback_reason =
            Some("bearing bootstrap camera model could not be resolved".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    };

    // Candidate-dependent triangulation is allowed on the FIT population. It
    // is used only to decide which nuisance parameters are observable and to
    // seed bundle adjustment; it never changes held-out membership.
    let prepare_fit = |source: &[&Track], geometry_cameras: &[ResolvedCamera], allowed: &[bool]| {
        source
            .iter()
            .filter_map(|track| {
                let mut track = (*track).clone();
                track.observations.retain(|observation| {
                    allowed.get(observation.camera).copied().unwrap_or(false)
                });
                if track.observations.len() < 2 {
                    return None;
                }
                // Do not use the current camera candidate to select the fit
                // population by cheirality. A bad initializer can put the
                // *correct* line intersection behind one camera; deleting that
                // track here lets the optimiser choose a geometry that explains
                // only the surviving positive-depth subset. Held-out tracks are
                // not pruned this way, so that selection bias showed up as the
                // 100% fit / ~44% held-out positive-depth split on the real L16
                // capture. The bundle objective already has a continuous
                // reflected-line reprojection residual plus an explicit
                // behind-camera penalty, so keep finite triangulations alive
                // and let optimisation move them onto the physical branch.
                let triangulated = triangulate(&track.observations, geometry_cameras, options)?;
                let rms = track_rms(&track.observations, geometry_cameras, triangulated.point);
                if !rms.is_finite() || rms > options.max_initial_track_rms {
                    return None;
                }
                track.condition = triangulated.condition;
                track.max_ray_angle_degrees = triangulated.max_ray_angle_degrees;
                Some(track)
            })
            .collect::<Vec<_>>()
    };
    // For the latent strategy, give candidate identity one chance to move
    // before the strict finite-depth preparation gate. Otherwise a badly
    // repeated initial winner can be discarded for high triangulation RMS
    // before the alternative identity is ever evaluated.
    let mut latent_assignment_round = 0usize;
    let mut fit_assignment_tracks = preliminary_fit
        .iter()
        .map(|track| (*track).clone())
        .collect::<Vec<_>>();
    if options.strategy == RigRefinementStrategy::LatentGraph
        && options.latent_max_assignment_iterations > 0
    {
        latent_assignment_round = 1;
        if let (Some(candidates), Some(latent_report)) =
            (latent_candidates.as_ref(), report.latent_match.as_mut())
        {
            update_latent_assignments(
                &mut fit_assignment_tracks,
                candidates,
                &bootstrap_cameras,
                options,
                latent_assignment_round,
                latent_report,
            );
        }
    }
    let fit_assignment_refs = fit_assignment_tracks.iter().collect::<Vec<_>>();
    let all_cameras = vec![true; cameras.len()];
    let bootstrap_fit_tracks = prepare_fit(&fit_assignment_refs, &bootstrap_cameras, &all_cameras);
    if bootstrap_fit_tracks.len() < options.min_tracks {
        report.rejected_degenerate_tracks += preliminary_fit.len() - bootstrap_fit_tracks.len();
        report.fallback_reason = Some(format!(
            "only {} fit tracks triangulate after epipolar bootstrap (need {})",
            bootstrap_fit_tracks.len(),
            options.min_tracks,
        ));
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    }

    // Phase 2: expose finite-depth nuisance parameters only where the
    // bootstrap geometry can actually observe them. Compute observability
    // around the epipolar solution rather than around the known-worse factory
    // seed.
    let bootstrap_fit_refs = bootstrap_fit_tracks.iter().collect::<Vec<_>>();
    let candidate_specs = parameter_specs(cameras, reference_index, &bootstrap_fit_refs, options);
    let candidate_base_parameters =
        remap_parameters(&bootstrap_parameters, &bootstrap_specs, &candidate_specs);
    let (specs, parameter_observability) = filter_observable_parameter_specs(
        &candidate_specs,
        &candidate_base_parameters,
        cameras,
        &bootstrap_fit_refs,
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

    // Re-express the bearing seed in the final observable parameter basis and
    // run one more fit-only bearing pass. If observability removed a nearly
    // degenerate nuisance direction, this lets the retained bearing DOF absorb
    // the image evidence without reopening an essential-matrix branch jump.
    let seeded_parameters = remap_parameters(&bootstrap_parameters, &bootstrap_specs, &specs);
    let observable_camera = (0..cameras.len())
        .map(|camera| camera == reference_index || specs.iter().any(|spec| spec.camera == camera))
        .collect::<Vec<_>>();
    let mut epipolar_seed_tracks = fit_assignment_tracks
        .iter()
        .filter_map(|track| {
            let mut track = track.clone();
            track
                .observations
                .retain(|observation| observable_camera[observation.camera]);
            (track.observations.len() >= 2).then_some(track)
        })
        .collect::<Vec<_>>();
    let epipolar_seed_refs = epipolar_seed_tracks.iter().collect::<Vec<_>>();
    let final_bearing_specs =
        bearing_trust_region_specs(&specs, &epipolar_seed_refs, cameras.len());
    let (initialized, _, final_epipolar_iterations) = coordinate_optimize_rig(
        seeded_parameters,
        &final_bearing_specs,
        options.max_iterations,
        cameras,
        &epipolar_seed_refs,
        intrinsics_mode,
        options,
        IncrementalObjectiveMode::Bearing { reference_index },
    );
    let initialized_refinements = refinements_from_parameters(cameras.len(), &initialized, &specs);
    let Some(initialized_cameras) =
        resolve_cameras(cameras, &initialized_refinements, intrinsics_mode)
    else {
        report.fallback_reason =
            Some("initialized physical model could not be resolved".to_owned());
        return RigRefinementOutcome {
            refinements: zero,
            report,
        };
    };

    // Re-score once more under the final bearing initialization before the
    // finite-depth fit population is frozen. This is still fit-only and uses no
    // held-out assignments.
    if options.strategy == RigRefinementStrategy::LatentGraph
        && latent_assignment_round < options.latent_max_assignment_iterations
    {
        latent_assignment_round += 1;
        if let (Some(candidates), Some(latent_report)) =
            (latent_candidates.as_ref(), report.latent_match.as_mut())
        {
            update_latent_assignments(
                &mut epipolar_seed_tracks,
                candidates,
                &initialized_cameras,
                options,
                latent_assignment_round,
                latent_report,
            );
        }
    }
    let epipolar_seed_track_refs = epipolar_seed_tracks.iter().collect::<Vec<_>>();
    let mut fit_tracks = prepare_fit(
        &epipolar_seed_track_refs,
        &initialized_cameras,
        &observable_camera,
    );
    report.rejected_degenerate_tracks += preliminary_fit.len().saturating_sub(fit_tracks.len());

    // Validation tracks are intentionally *not* triangulation-pruned with the
    // fitted cameras. They retain the correspondence population frozen above.
    // evaluate() will triangulate them independently under factory/candidate
    // geometry and the positive-depth fraction counts failures.
    let validation_tracks = preliminary_validation;
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
    let initialization_iterations = bootstrap_iterations + final_epipolar_iterations;
    let (mut parameters, _, bundle_iterations) = staged_bundle_optimize_rig(
        initialized,
        &specs,
        options.max_iterations,
        cameras,
        &initial_fit,
        intrinsics_mode,
        options,
    );
    drop(initial_fit);
    let mut iterations = initialization_iterations + bundle_iterations;
    let mut membership_iterations = 0usize;
    let mut membership_rejected_observations = 0usize;
    let mut membership_rejected_tracks = 0usize;
    let latent_strategy = options.strategy == RigRefinementStrategy::LatentGraph;
    let mut latent_membership = if latent_strategy {
        report.latent_match.as_mut().map(|latent_report| {
            LatentMembershipState::new(&fit_tracks, cameras.len(), options, latent_report)
        })
    } else {
        None
    };
    let outer_iterations = if latent_strategy {
        options
            .latent_max_assignment_iterations
            // Reversible membership needs several bundle/recovery passes after
            // assignment switching has converged. Do not reduce that recovery
            // budget merely because the legacy destructive path uses three.
            .saturating_add(options.max_membership_iterations.max(6))
            .saturating_add(1)
    } else {
        options.max_membership_iterations
    };
    for _ in 0..outer_iterations {
        let refinements = refinements_from_parameters(cameras.len(), &parameters, &specs);
        let Some(current_cameras) = resolve_cameras(cameras, &refinements, intrinsics_mode) else {
            break;
        };

        // Latent correspondence is a discrete outer variable.  Update every
        // candidate against one frozen rig/assignment snapshot, apply all
        // accepted switches together, then re-run the continuous bundle solve.
        // A round that switches anything deliberately skips membership
        // reassessment; the new identity gets one full bundle pass to settle
        // before it can be demoted. LatentGraph membership is reversible.
        let latent_switches = if latent_strategy
            && latent_assignment_round < options.latent_max_assignment_iterations
        {
            latent_assignment_round += 1;
            match (latent_candidates.as_ref(), report.latent_match.as_mut()) {
                (Some(candidates), Some(latent_report)) => update_latent_assignments(
                    &mut fit_tracks,
                    candidates,
                    &current_cameras,
                    options,
                    latent_assignment_round,
                    latent_report,
                ),
                _ => 0,
            }
        } else {
            0
        };

        let tracks_before_membership = fit_tracks.len();
        let (rejected_observations, rejected_tracks, membership_changed) = if latent_switches > 0 {
            (0, 0, false)
        } else if latent_strategy {
            match (latent_membership.as_mut(), report.latent_match.as_mut()) {
                (Some(state), Some(latent_report)) => {
                    let update =
                        state.update(&mut fit_tracks, &current_cameras, options, latent_report);
                    (
                        update.demoted,
                        tracks_before_membership.saturating_sub(fit_tracks.len()),
                        update.changed(),
                    )
                }
                _ => (0, 0, false),
            }
        } else {
            let (rejected_observations, rejected_tracks) =
                update_fit_track_membership(&mut fit_tracks, &current_cameras, options);
            (
                rejected_observations,
                rejected_tracks,
                rejected_observations > 0 || rejected_tracks > 0,
            )
        };
        if membership_changed {
            membership_iterations += 1;
            membership_rejected_observations += rejected_observations;
            membership_rejected_tracks += rejected_tracks;
        }
        if latent_switches == 0 && !membership_changed {
            break;
        }
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
        let (next_parameters, _, next_iterations) = staged_bundle_optimize_rig(
            parameters,
            &specs,
            options.max_iterations,
            cameras,
            &current_fit,
            intrinsics_mode,
            options,
        );
        parameters = next_parameters;
        iterations += next_iterations;
    }
    report.membership_iterations = membership_iterations;
    report.fit_membership_rejected_observations = membership_rejected_observations;
    report.fit_membership_rejected_tracks = membership_rejected_tracks;
    report.fit_tracks = fit_tracks.len();
    if let (Some(state), Some(latent_report)) =
        (latent_membership.as_ref(), report.latent_match.as_mut())
    {
        state.refresh_report(latent_report);
    }

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
    let validation_before =
        evaluate_validation_leave_one_out(&validation, &factory_cameras, options, true);
    let validation_after =
        evaluate_validation_leave_one_out(&validation, &candidate_cameras, options, true);
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
    // The reference observation participates in triangulation, but counting it
    // in the headline acceptance RMS makes a multi-view track look better simply
    // because every track contains one fixed-gauge sample.  Build a target-only
    // frozen population and use that for the production sub-pixel gate.
    let held_out_target_before = options
        .held_out_validation
        .then(|| {
            validation_before
                .residuals
                .iter()
                .filter(|sample| sample.camera != reference_index)
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let held_out_target_after = options
        .held_out_validation
        .then(|| {
            validation_after
                .residuals
                .iter()
                .filter(|sample| sample.camera != reference_index)
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let held_out_target_rms_before = residual_sample_rms(&held_out_target_before);
    let held_out_target_rms_after = residual_sample_rms(&held_out_target_after);
    let held_out_target_residuals_before = residual_distribution(&held_out_target_before);
    let held_out_target_residuals_after = residual_distribution(&held_out_target_after);
    let held_out_factory_p95_reference_px =
        held_out_residuals_before.reference_equivalent_pixels.p95;
    let held_out_p95_reference_px = held_out_residuals_after.reference_equivalent_pixels.p95;
    let held_out_improvement = if options.held_out_validation {
        (validation_before.rms() - validation_after.rms()) / validation_before.rms().max(1.0e-12)
    } else {
        0.0
    };
    let held_out_target_improvement = if options.held_out_validation
        && held_out_target_rms_before.is_finite()
        && held_out_target_rms_after.is_finite()
    {
        (held_out_target_rms_before - held_out_target_rms_after)
            / held_out_target_rms_before.max(1.0e-12)
    } else {
        f64::NAN
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
        // B4 is the fixed gauge. Its leave-one-out residual measures how well
        // the *other* cameras reconstruct a point that lands back in B4; it is
        // useful diagnostically but B4 has no free parameter that could repair
        // such a failure. Target-camera gates below already catch the offending
        // geometry, so do not double-count the reference as a rejection cause.
        let affects_acceptance = camera_index != reference_index
            && (camera.validation_samples > 0
                || specs.iter().any(|spec| spec.camera == camera_index));
        affects_acceptance
            && camera.validation_samples >= options.min_validation_camera_samples
            && camera.validation_rms_after > camera.validation_rms_before * 1.05 + 0.02
    });
    let camera_absolute_failure = options.held_out_validation
        && per_camera.iter().enumerate().any(|(camera_index, camera)| {
            if camera_index == reference_index {
                return false;
            }
            // Gate every target camera for which the frozen validation split
            // has image evidence, not only cameras that happened to receive
            // free parameters. An unchanged factory target with a 10 px
            // residual is still unsafe for PhysicalRig dense depth.
            let has_validation_evidence = camera.validation_samples > 0
                || specs.iter().any(|spec| spec.camera == camera_index);
            has_validation_evidence
                && (camera.validation_samples < options.min_validation_camera_samples
                    || !camera.validation_rms_after.is_finite()
                    || camera.validation_rms_after > options.max_validation_camera_rms_px
                    || !camera.validation_p90_after.is_finite()
                    || camera.validation_p90_after > options.max_validation_camera_p90_px)
        });
    let training_improved = fit_after.rms() < fit_before.rms();
    let validation_improved = !options.held_out_validation
        || (held_out_target_improvement.is_finite()
            && held_out_target_improvement >= options.min_validation_improvement);
    let validation_absolute_quality = !options.held_out_validation
        || (held_out_target_rms_after.is_finite()
            && held_out_target_rms_after <= options.max_validation_rms_px
            && held_out_target_residuals_after
                .sensor_pixels
                .p90
                .is_finite()
            && held_out_target_residuals_after.sensor_pixels.p90 <= options.max_validation_p90_px
            && held_out_target_residuals_after
                .sensor_pixels
                .p95
                .is_finite()
            && held_out_target_residuals_after.sensor_pixels.p95 <= options.max_validation_p95_px);
    let fit_positive_depth = fit_after.positive_depth_fraction();
    let validation_positive_depth = options
        .held_out_validation
        .then(|| validation_after.positive_depth_fraction())
        .unwrap_or(0.0);
    let physical_depth_valid = fit_positive_depth >= options.min_positive_depth_fraction
        && (!options.held_out_validation
            || validation_positive_depth >= options.min_positive_depth_fraction);
    // Relative improvement alone is unsafe when the factory projection is
    // numerically broken: 1e14 -> 100 px is an enormous percentage gain but
    // still unusable geometry. Require the independent population to reach
    // the configured around/sub-pixel target as well.
    let validation_passed = training_improved
        && validation_improved
        && validation_absolute_quality
        && physical_depth_valid
        && !reached_bound
        && !camera_regression
        && !camera_absolute_failure;

    let mut conditions = fit_tracks
        .iter()
        .map(|track| track.condition)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    conditions.sort_by(f64::total_cmp);
    let mut ray_angles = fit_tracks
        .iter()
        .map(|track| track.max_ray_angle_degrees)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    ray_angles.sort_by(f64::total_cmp);
    let validation_failure_reason = (!validation_passed).then(|| {
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
        if camera_absolute_failure {
            let failures = per_camera
                .iter()
                .enumerate()
                .filter_map(|(camera_index, camera)| {
                    if camera_index == reference_index {
                        return None;
                    }
                    let has_validation_evidence = camera.validation_samples > 0
                        || specs.iter().any(|spec| spec.camera == camera_index);
                    if !has_validation_evidence {
                        return None;
                    }
                    if camera.validation_samples < options.min_validation_camera_samples {
                        return Some(format!(
                            "{} has only {} held-out observations (need {})",
                            camera.camera,
                            camera.validation_samples,
                            options.min_validation_camera_samples,
                        ));
                    }
                    if !camera.validation_rms_after.is_finite()
                        || camera.validation_rms_after > options.max_validation_camera_rms_px
                        || !camera.validation_p90_after.is_finite()
                        || camera.validation_p90_after > options.max_validation_camera_p90_px
                    {
                        return Some(format!(
                            "{} held-out RMS/p90 {:.3}/{:.3} px exceeds {:.3}/{:.3}",
                            camera.camera,
                            camera.validation_rms_after,
                            camera.validation_p90_after,
                            options.max_validation_camera_rms_px,
                            options.max_validation_camera_p90_px,
                        ));
                    }
                    None
                })
                .collect::<Vec<_>>();
            reasons.push(format!("per-camera held-out gate failed: {}", failures.join(", ")));
        }
        if !training_improved {
            reasons.push("training reprojection RMS did not improve".to_owned());
        }
        if options.held_out_validation && !validation_improved {
            reasons.push(format!(
                "target-camera held-out improvement {:+.3}% is below required {:+.3}%",
                held_out_target_improvement * 100.0,
                options.min_validation_improvement * 100.0
            ));
        }
        if options.held_out_validation && !validation_absolute_quality {
            reasons.push(format!(
                "target-camera held-out geometry misses absolute target: RMS {:.3} px (<= {:.3}), p90 {:.3} px (<= {:.3}), p95 {:.3} px (<= {:.3})",
                held_out_target_rms_after,
                options.max_validation_rms_px,
                held_out_target_residuals_after.sensor_pixels.p90,
                options.max_validation_p90_px,
                held_out_target_residuals_after.sensor_pixels.p95,
                options.max_validation_p95_px,
            ));
        }
        reasons.join("; ")
    });

    // LatentGraph is an explicit, single-path optimizer. Once it has reached
    // a resolved candidate, select that candidate unconditionally and expose
    // any held-out failure as a warning. Earlier optimizer failures return
    // without `accepted`; the pipeline turns those into an error instead of a
    // factory-rig fallback.
    let select_latent_candidate = options.strategy == RigRefinementStrategy::LatentGraph;
    let accepted = select_solved_candidate(options.strategy, validation_passed);
    report.accepted = accepted;
    report.validation_passed = validation_passed;
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
    report.held_out_target_rms_before = options
        .held_out_validation
        .then_some(held_out_target_rms_before)
        .unwrap_or(0.0);
    report.held_out_target_rms_after = options
        .held_out_validation
        .then_some(held_out_target_rms_after)
        .unwrap_or(0.0);
    report.held_out_target_relative_improvement = options
        .held_out_validation
        .then_some(held_out_target_improvement)
        .unwrap_or(0.0);
    report.held_out_target_residuals_before = held_out_target_residuals_before;
    report.held_out_target_residuals_after = held_out_target_residuals_after;
    report.held_out_p99_trimmed_rms_before = options
        .held_out_validation
        .then(|| trimmed_residual_rms(&validation_before.residuals, 0.99))
        .unwrap_or(0.0);
    report.held_out_p99_trimmed_rms_after = options
        .held_out_validation
        .then(|| trimmed_residual_rms(&validation_after.residuals, 0.99))
        .unwrap_or(0.0);
    report.held_out_over_50px_before = options
        .held_out_validation
        .then(|| residuals_over_px(&validation_before.residuals, 50.0))
        .unwrap_or(0);
    report.held_out_over_50px_after = options
        .held_out_validation
        .then(|| residuals_over_px(&validation_after.residuals, 50.0))
        .unwrap_or(0);
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
    report.residual_correlations =
        residual_correlation_reports(cameras, &validation_after.residuals);
    report.held_out_observations = held_out_observation_reports(
        cameras,
        &validation_before.residuals,
        &validation_after.residuals,
        held_out_factory_p95_reference_px,
        held_out_p95_reference_px,
    );
    report.per_camera = per_camera;
    if select_latent_candidate {
        report.validation_warning = validation_failure_reason;
        report.fallback_reason = None;
    } else {
        report.fallback_reason = validation_failure_reason;
    }

    RigRefinementOutcome {
        refinements: if accepted {
            candidate_refinements
        } else {
            zero
        },
        report,
    }
}

fn factory_model_report(
    cameras: &[RigCameraInput<'_>],
    intrinsics_mode: IntrinsicsMode,
) -> Vec<RigFactoryModelReport> {
    cameras
        .iter()
        .filter_map(|input| {
            let calibration = input.calibration?;
            let state = input.state?;
            let mut halls = calibration
                .intrinsics
                .iter()
                .filter_map(|bundle| bundle.hall_code)
                .filter(|hall| hall.is_finite())
                .collect::<Vec<_>>();
            halls.sort_by(f64::total_cmp);
            let hall_min = halls.first().copied();
            let hall_max = halls.last().copied();
            let extrapolation = match (hall_min, hall_max) {
                (Some(minimum), _) if state.lens_hall < minimum => state.lens_hall - minimum,
                (_, Some(maximum)) if state.lens_hall > maximum => state.lens_hall - maximum,
                _ => 0.0,
            };
            let resolved_k = calibration
                .k_for_hall(state.lens_hall, intrinsics_mode)
                .ok();
            let (quadratic, inverse, pair_linear, selected, use_left, use_right, inflection) =
                calibration.mirror.as_ref().map_or_else(
                    || (None, None, None, None, None, None, None),
                    |mirror| {
                        (
                            mirror
                                .actuator
                                .quadratic_angle_for_hall(state.mirror_hall)
                                .ok(),
                            mirror
                                .actuator
                                .inverse_quadratic_angle_for_hall(state.mirror_hall)
                                .ok(),
                            mirror.actuator.interpolated_angle(state.mirror_hall).ok(),
                            mirror.actuator.angle_for_hall(state.mirror_hall).ok(),
                            mirror.actuator.use_rplus_for_left_segment,
                            mirror.actuator.use_rplus_for_right_segment,
                            mirror.actuator.inflection_value,
                        )
                    },
                );
            let difference = pair_linear.zip(quadratic).map(|(pairs, quad)| pairs - quad);
            let inverse_difference = inverse
                .zip(quadratic)
                .map(|(inverse, current)| inverse - current);
            let focal = resolved_k.map(|k| 0.5 * (k[0][0].abs() + k[1][1].abs()));
            let approximate_px = difference
                .zip(focal)
                .map(|(angle, focal)| 2.0 * focal * angle.abs().to_radians());
            let approximate_inverse_px = inverse_difference
                .zip(focal)
                .map(|(angle, focal)| 2.0 * focal * angle.abs().to_radians());
            let cra = calibration.cra.as_ref();
            let ratio = cra.and_then(|cra| cra.distance_hall_ratio);
            Some(RigFactoryModelReport {
                camera: input.name.to_owned(),
                capture_lens_hall: state.lens_hall,
                calibrated_lens_hall_min: hall_min,
                calibrated_lens_hall_max: hall_max,
                lens_hall_extrapolation_counts: extrapolation,
                resolved_k,
                cra_sensor_distance: cra.and_then(|cra| cra.sensor_distance),
                cra_exit_pupil_distance: cra.and_then(|cra| cra.exit_pupil_distance),
                cra_pixel_size: cra.and_then(|cra| cra.pixel_size),
                cra_lens_hall: cra.and_then(|cra| cra.lens_hall_code),
                cra_distance_hall_ratio: ratio,
                cra_implied_capture_sensor_distance: ratio
                    .map(|ratio| (state.lens_hall + 1.0) * ratio),
                cra_radial_samples: cra
                    .map(|cra| cra.radial_samples.clone())
                    .unwrap_or_default(),
                cra_fitted_coefficients: cra
                    .map(|cra| cra.fitted_coefficients.clone())
                    .unwrap_or_default(),
                cra_fit_cost: cra.and_then(|cra| cra.fit_cost),
                cra_valid_roi: cra.and_then(|cra| cra.valid_roi),
                capture_mirror_hall: state.mirror_hall,
                af_mirror_hall: state.focus.af_mirror_hall,
                quadratic_mirror_angle_degrees: quadratic,
                inverse_quadratic_mirror_angle_degrees: inverse,
                pair_linear_mirror_angle_degrees: pair_linear,
                selected_mirror_angle_degrees: selected,
                pair_minus_quadratic_degrees: difference,
                inverse_minus_current_degrees: inverse_difference,
                approximate_mirror_difference_px: approximate_px,
                approximate_inverse_difference_px: approximate_inverse_px,
                quadratic_use_rplus_left: use_left,
                quadratic_use_rplus_right: use_right,
                quadratic_inflection_value: inflection,
            })
        })
        .collect()
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
    epipolar_tangent: Option<Vec2>,
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

#[derive(Clone, Copy, Debug)]
struct ReferencePatchRays {
    centre: crate::geometry::Ray,
    left: crate::geometry::Ray,
    right: crate::geometry::Ray,
    above: crate::geometry::Ray,
    below: crate::geometry::Ray,
}

#[derive(Clone, Copy, Debug)]
struct ReferenceZnccSample {
    point: Vec2,
    value: f32,
    weight: f32,
}

#[derive(Clone, Debug)]
struct ReferenceZnccPatch {
    samples: Vec<ReferenceZnccSample>,
    weight_sum: f32,
    sum_reference: f32,
    sum_reference_sq: f32,
}

#[derive(Debug)]
struct ResidualGrid {
    values: [Vec2; 25],
    len: usize,
}

impl ResidualGrid {
    fn new() -> Self {
        Self {
            values: [[0.0; 2]; 25],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, value: Vec2) {
        self.values[self.len] = value;
        self.len += 1;
    }

    #[inline]
    fn as_slice(&self) -> &[Vec2] {
        &self.values[..self.len]
    }
}

#[derive(Debug, Default)]
struct PhysicalMatchScratch {
    sum_target: Vec<f32>,
    sum_target_sq: Vec<f32>,
    sum_product: Vec<f32>,
    valid: Vec<bool>,
}

impl PhysicalMatchScratch {
    fn prepare(&mut self, count: usize) {
        self.sum_target.resize(count, 0.0);
        self.sum_target_sq.resize(count, 0.0);
        self.sum_product.resize(count, 0.0);
        self.valid.resize(count, true);
        self.sum_target.fill(0.0);
        self.sum_target_sq.fill(0.0);
        self.sum_product.fill(0.0);
        self.valid.fill(true);
    }
}

fn reference_patch_rays(reference_camera: &ResolvedCamera, centre: Vec2) -> ReferencePatchRays {
    const DERIVATIVE_STEP: f64 = 2.0;
    ReferencePatchRays {
        centre: reference_camera.pixel_to_ray(centre),
        left: reference_camera.pixel_to_ray([centre[0] - DERIVATIVE_STEP, centre[1]]),
        right: reference_camera.pixel_to_ray([centre[0] + DERIVATIVE_STEP, centre[1]]),
        above: reference_camera.pixel_to_ray([centre[0], centre[1] - DERIVATIVE_STEP]),
        below: reference_camera.pixel_to_ray([centre[0], centre[1] + DERIVATIVE_STEP]),
    }
}

fn prepare_reference_zncc_patch(
    reference: &Plane,
    centre: Vec2,
    radius: usize,
    pyramid_level: usize,
) -> Option<ReferenceZnccPatch> {
    let centre_reference = reference.sample(
        sensor_to_luminance_coordinate(centre[0], pyramid_level),
        sensor_to_luminance_coordinate(centre[1], pyramid_level),
    )?;
    let sensor_step = 1.0 / luminance_level_scale(pyramid_level);
    let sigma = (radius as f32 * 0.75).max(1.0);
    let mut samples = Vec::with_capacity((radius * 2 + 1).pow(2));
    let mut weight_sum = 0.0f32;
    let mut sum_reference = 0.0f32;
    let mut sum_reference_sq = 0.0f32;
    for dy in -(radius as isize)..=radius as isize {
        for dx in -(radius as isize)..=radius as isize {
            let point = [
                centre[0] + dx as f64 * sensor_step,
                centre[1] + dy as f64 * sensor_step,
            ];
            let value = reference.sample(
                sensor_to_luminance_coordinate(point[0], pyramid_level),
                sensor_to_luminance_coordinate(point[1], pyramid_level),
            )?;
            let distance_sq = (dx * dx + dy * dy) as f32;
            let spatial = (-distance_sq / (2.0 * sigma * sigma)).exp();
            let range = (-1.2 * (value - centre_reference).abs()).exp();
            let weight = spatial * range;
            weight_sum += weight;
            sum_reference += weight * value;
            sum_reference_sq += weight * value * value;
            samples.push(ReferenceZnccSample {
                point,
                value,
                weight,
            });
        }
    }
    (weight_sum > 1.0e-6).then_some(ReferenceZnccPatch {
        samples,
        weight_sum,
        sum_reference,
        sum_reference_sq,
    })
}

#[inline]
fn project_ray_to_surface(
    ray: crate::geometry::Ray,
    surface_point: Vec3,
    surface_normal: Vec3,
    target_camera: &ResolvedCamera,
) -> Option<Vec2> {
    let denominator = dot(ray.direction, surface_normal);
    if denominator.abs() <= 1.0e-9 {
        return None;
    }
    let distance = dot(sub(surface_point, ray.origin), surface_normal) / denominator;
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    target_camera.project(add(ray.origin, scale(ray.direction, distance)))
}

#[inline]
fn project_reference_centre_at_depth(
    centre_ray: crate::geometry::Ray,
    target_camera: &ResolvedCamera,
    depth: f64,
) -> Option<Vec2> {
    target_camera.project(add(centre_ray.origin, scale(centre_ray.direction, depth)))
}

/// A candidate depth is the correspondence model.  The source patch is lifted
/// onto the local scene plane implied by that depth, then projected through the
/// complete target camera model.  Unlike the legacy aligner, no homography or
/// pre-existing image warp defines the match locus.
fn physical_patch_projection(
    rays: &ReferencePatchRays,
    centre: Vec2,
    target_camera: &ResolvedCamera,
    depth: f64,
) -> Option<PhysicalPatchProjection> {
    const DERIVATIVE_STEP: f64 = 2.0;
    let surface_point = add(rays.centre.origin, scale(rays.centre.direction, depth));
    let surface_normal = rays.centre.direction;
    let target_centre = project_reference_centre_at_depth(rays.centre, target_camera, depth)?;
    let left = project_ray_to_surface(rays.left, surface_point, surface_normal, target_camera)?;
    let right = project_ray_to_surface(rays.right, surface_point, surface_normal, target_camera)?;
    let above = project_ray_to_surface(rays.above, surface_point, surface_normal, target_camera)?;
    let below = project_ray_to_surface(rays.below, surface_point, surface_normal, target_camera)?;
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
    target: &Plane,
    projection: PhysicalPatchProjection,
    reference_patch: &ReferenceZnccPatch,
    residual: Vec2,
    pyramid_level: usize,
) -> Option<f32> {
    let mut sum_target = 0.0f32;
    let mut sum_target_sq = 0.0f32;
    let mut sum_product = 0.0f32;
    for sample in &reference_patch.samples {
        let mapped = projection.map(sample.point);
        let target_value = target.sample(
            sensor_to_luminance_coordinate(mapped[0] + residual[0], pyramid_level),
            sensor_to_luminance_coordinate(mapped[1] + residual[1], pyramid_level),
        )?;
        sum_target += sample.weight * target_value;
        sum_target_sq += sample.weight * target_value * target_value;
        sum_product += sample.weight * sample.value * target_value;
    }
    let covariance =
        sum_product - reference_patch.sum_reference * sum_target / reference_patch.weight_sum;
    let reference_variance = (reference_patch.sum_reference_sq
        - reference_patch.sum_reference * reference_patch.sum_reference
            / reference_patch.weight_sum)
        .max(0.0);
    let target_variance =
        (sum_target_sq - sum_target * sum_target / reference_patch.weight_sum).max(0.0);
    let denominator = (reference_variance * target_variance).sqrt();
    (denominator > 1.0e-8).then_some((covariance / denominator).clamp(-1.0, 1.0))
}

fn search_physical_patch_zncc(
    target: &Plane,
    projection: PhysicalPatchProjection,
    reference_patch: &ReferenceZnccPatch,
    proposal: Vec2,
    offsets: &[Vec2],
    pyramid_level: usize,
    residual_radius: f64,
    scratch: &mut PhysicalMatchScratch,
) -> Option<(f32, f32, Vec2)> {
    scratch.prepare(offsets.len());
    for sample in &reference_patch.samples {
        let mapped = projection.map(sample.point);
        for (index, offset) in offsets.iter().enumerate() {
            if !scratch.valid[index] {
                continue;
            }
            let residual = [proposal[0] + offset[0], proposal[1] + offset[1]];
            let Some(target_value) = target.sample(
                sensor_to_luminance_coordinate(mapped[0] + residual[0], pyramid_level),
                sensor_to_luminance_coordinate(mapped[1] + residual[1], pyramid_level),
            ) else {
                scratch.valid[index] = false;
                continue;
            };
            scratch.sum_target[index] += sample.weight * target_value;
            scratch.sum_target_sq[index] += sample.weight * target_value * target_value;
            scratch.sum_product[index] += sample.weight * sample.value * target_value;
        }
    }
    let reference_variance = (reference_patch.sum_reference_sq
        - reference_patch.sum_reference * reference_patch.sum_reference
            / reference_patch.weight_sum)
        .max(0.0);
    let mut best: Option<(f32, f32, Vec2)> = None;
    for (index, offset) in offsets.iter().enumerate() {
        if !scratch.valid[index] {
            continue;
        }
        let sum_target = scratch.sum_target[index];
        let covariance = scratch.sum_product[index]
            - reference_patch.sum_reference * sum_target / reference_patch.weight_sum;
        let target_variance = (scratch.sum_target_sq[index]
            - sum_target * sum_target / reference_patch.weight_sum)
            .max(0.0);
        let denominator = (reference_variance * target_variance).sqrt();
        if denominator <= 1.0e-8 {
            continue;
        }
        let score = (covariance / denominator).clamp(-1.0, 1.0);
        let penalty = ((offset[0] * offset[0] + offset[1] * offset[1]).sqrt()
            / residual_radius.max(1.0)) as f32
            * 0.001;
        let objective = score - penalty;
        let residual = [proposal[0] + offset[0], proposal[1] + offset[1]];
        if best.is_none_or(|(best_objective, _, _)| objective > best_objective) {
            best = Some((objective, score, residual));
        }
    }
    best
}

fn physical_epipolar_tangent(
    centre_ray: crate::geometry::Ray,
    target_camera: &ResolvedCamera,
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
    let a = project_reference_centre_at_depth(centre_ray, target_camera, 1.0 / a_inverse)?;
    let b = project_reference_centre_at_depth(centre_ray, target_camera, 1.0 / b_inverse)?;
    let delta = [b[0] - a[0], b[1] - a[1]];
    let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    (length > 1.0e-6).then_some([delta[0] / length, delta[1] / length])
}

fn physical_localization_covariance(tangent: Option<Vec2>) -> [[f64; 2]; 2] {
    // The initial matcher deliberately has more freedom along the physical
    // depth locus than perpendicular to it. Preserve that anisotropy in the
    // persistent observation so BA does not treat both axes as equally known.
    let Some(tangent) = tangent else {
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

fn epipolar_residual_grid(tangent: Option<Vec2>, radius: f64) -> ResidualGrid {
    let mut offsets = ResidualGrid::new();
    if !radius.is_finite() || radius <= 0.0 {
        offsets.push([0.0, 0.0]);
        return offsets;
    }
    let Some(tangent) = tangent else {
        let half = radius * 0.5;
        let coordinates = [-radius, -half, 0.0, half, radius];
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
    projection: PhysicalPatchProjection,
    centre: Vec2,
    tangent: Option<Vec2>,
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
    target: &Plane,
    reference_rays: &ReferencePatchRays,
    reference_patch: &ReferenceZnccPatch,
    target_camera: &ResolvedCamera,
    measured_alignment: &ModuleAlignment,
    target_index: usize,
    centre: Vec2,
    depth: f64,
    pyramid_level: usize,
    options: &RigRefinementOptions,
    scratch: &mut PhysicalMatchScratch,
) -> Option<PhysicalViewMatch> {
    let projection = physical_patch_projection(reference_rays, centre, target_camera, depth)?;
    let tangent = physical_epipolar_tangent(reference_rays.centre, target_camera, depth, options);
    let measured_proposal = if measured_alignment.report.accepted {
        measured_epipolar_residual_proposal(measured_alignment, projection, centre, tangent)
    } else {
        // A rejected global alignment is diagnostic evidence that its measured
        // warp is not a trustworthy proposal. Do not let that failed model
        // steer the physical matcher into a repeated-texture basin.
        [0.0, 0.0]
    };
    let offsets = epipolar_residual_grid(tangent, options.physical_match_residual_radius_px);
    let mut search = |proposal: Vec2| {
        search_physical_patch_zncc(
            target,
            projection,
            reference_patch,
            proposal,
            offsets.as_slice(),
            pyramid_level,
            options.physical_match_residual_radius_px,
            scratch,
        )
    };

    // Always evaluate both plausible bootstrap centres.  Previously a merely
    // acceptable factory-centred peak (score >= min_score) prevented the
    // measured proposal from being searched at all.  Repeated L16 scene
    // structure (railings, windows, beams) makes that failure mode common: a
    // wrong 0.55 peak could hide a 0.95 peak at the measured proposal.
    let factory = search([0.0, 0.0]).map(|candidate| (candidate, [0.0, 0.0]));
    let measured = (dot2(measured_proposal, measured_proposal) > 1.0)
        .then(|| search(measured_proposal))
        .flatten()
        .map(|candidate| (candidate, measured_proposal));
    let ((_, score, residual), residual_proposal) = match (factory, measured) {
        (Some(factory), Some(measured)) => {
            // Compare the same search objective (ZNCC with the tiny local
            // offset penalty), not just whether one branch crossed a threshold.
            if measured.0.0 > factory.0.0 {
                measured
            } else {
                factory
            }
        }
        (Some(factory), None) => factory,
        (None, Some(measured)) => measured,
        (None, None) => return None,
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
        epipolar_tangent: tangent,
        local_scale: projection.local_scale(),
    })
}

fn refine_physical_view_match(
    target_input: &RigCameraInput<'_>,
    reference_rays: &ReferencePatchRays,
    reference_patch: &ReferenceZnccPatch,
    target_camera: &ResolvedCamera,
    centre: Vec2,
    depth: f64,
    mut matched: PhysicalViewMatch,
    options: &RigRefinementOptions,
) -> PhysicalViewMatch {
    let (Some(target), Some(projection)) = (
        target_input.luminance,
        physical_patch_projection(reference_rays, centre, target_camera, depth),
    ) else {
        return matched;
    };
    // Coarse matching only establishes correspondence identity.  Refine that
    // observation on the native luminance plane with successively smaller
    // fractional-pixel steps.  The old code performed a single +/-4 px 3x3
    // search, leaving observations quantized at multi-pixel precision.
    let minimum_step = options.physical_match_subpixel_step_px.clamp(0.0625, 1.0);
    let mut step = (options.physical_match_residual_radius_px / 8.0).clamp(1.0, 4.0);
    while step + 1.0e-12 >= minimum_step {
        let base = matched.residual;
        let mut best_score = matched.score;
        let mut best_residual = matched.residual;
        for dy in [-step, 0.0, step] {
            for dx in [-step, 0.0, step] {
                if dx == 0.0 && dy == 0.0 {
                    continue;
                }
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
                let Some(score) =
                    physical_patch_zncc(target, projection, reference_patch, residual, 0)
                else {
                    continue;
                };
                if score > best_score {
                    best_score = score;
                    best_residual = residual;
                }
            }
        }
        matched.score = best_score;
        matched.residual = best_residual;
        matched.target_pixel = [
            projection.target_centre[0] + best_residual[0],
            projection.target_centre[1] + best_residual[1],
        ];
        step *= 0.5;
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
    reference_rays: &ReferencePatchRays,
    reference_patch: &ReferenceZnccPatch,
    centre: Vec2,
    label: usize,
    inverse_depth: f64,
    pyramid_level: usize,
    options: &RigRefinementOptions,
    scratch: &mut PhysicalMatchScratch,
) -> Option<PhysicalDepthCandidate> {
    if !inverse_depth.is_finite() || inverse_depth <= 0.0 {
        return None;
    }
    let depth = 1.0 / inverse_depth;
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
            target,
            reference_rays,
            reference_patch,
            &resolved[target_index],
            &measured_alignments[target_index],
            target_index,
            centre,
            depth,
            pyramid_level,
            options,
            scratch,
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
    let mut rays = Vec::with_capacity(track.observations.len());
    fill_observation_rays(&track.observations, cameras, &mut rays);
    let mut terms = track
        .observations
        .iter()
        .zip(&rays)
        .map(|(observation, &ray)| triangulation_normal_term(observation, ray))
        .collect::<Vec<_>>();
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for ((observation, &term), _) in track.observations.iter().zip(&terms).zip(&rays) {
        add_triangulation_term(
            &mut normal,
            &mut rhs,
            term,
            observation_balance(observation, track.observations.len()),
        );
    }
    loop {
        let triangulated = triangulate_normal_system(&rays, normal, rhs, options)?;
        if !triangulation_has_positive_depth_with_rays(&rays, triangulated.point) {
            return None;
        }
        let mut all_consistent = true;
        let mut worst_target: Option<(usize, f64)> = None;
        for (index, (observation, &ray)) in track.observations.iter().zip(&rays).enumerate() {
            let projected = project_observation_with_ray(
                &cameras[observation.camera],
                ray,
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

        // Target observations always carry unit balance. Removing one also
        // reduces every fixed-gauge observation's balance by exactly one, so
        // update the accumulated 3x3 triangulation system instead of rebuilding
        // it from all remaining observations.
        add_triangulation_term(&mut normal, &mut rhs, terms[worst_index], -1.0);
        for (index, observation) in track.observations.iter().enumerate() {
            if index != worst_index && observation.fixed_gauge {
                add_triangulation_term(&mut normal, &mut rhs, terms[index], -1.0);
            }
        }
        track.observations.remove(worst_index);
        rays.remove(worst_index);
        terms.remove(worst_index);
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
    for mut track in tracks.drain(..) {
        let original_observations = track.observations.len();

        // A two-view track has no redundant observation that can be removed.
        // More importantly, rejecting it solely because the current iterate
        // triangulates behind one camera recreates the same candidate-dependent
        // selection bias that the initial fit preparation avoids above. Keep a
        // pairwise track when its *line* reprojection is still geometrically
        // consistent; the bundle objective's cheirality penalty can then move
        // the camera onto the physical branch in a later sweep. Multi-view
        // physical tracks still use the stricter positive-depth membership
        // pruning because they contain redundant observations.
        if track.observations.len() == 2 {
            let mut rays = Vec::with_capacity(2);
            if let Some(triangulated) =
                triangulate_with_rays(&track.observations, cameras, options, &mut rays)
            {
                let mut consistent = true;
                for (observation, &ray) in track.observations.iter().zip(&rays) {
                    let Some(projected) = project_observation_with_ray(
                        &cameras[observation.camera],
                        ray,
                        triangulated.point,
                    ) else {
                        consistent = false;
                        break;
                    };
                    let residual = [
                        projected[0] - observation.pixel[0],
                        projected[1] - observation.pixel[1],
                    ];
                    let sensor_error = dot2(residual, residual).sqrt();
                    let reference_error = sensor_error / observation.local_scale.clamp(0.25, 4.0);
                    if !reference_error.is_finite()
                        || reference_error > options.fit_membership_max_reference_px
                    {
                        consistent = false;
                        break;
                    }
                }
                if consistent {
                    track.condition = triangulated.condition;
                    track.max_ray_angle_degrees = triangulated.max_ray_angle_degrees;
                    retained.push(track);
                    continue;
                }
            }
            rejected_observations += original_observations;
            continue;
        }

        if let Some((track, removed)) = prune_track_observations(
            track,
            cameras,
            options,
            options.physical_match_min_views.max(2),
            options.fit_membership_max_reference_px,
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
#[cfg(test)]
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
    let maximum = available_levels.min(options.physical_match_max_depth_refinements);
    let coarse_intervals = options.physical_match_planes.saturating_sub(1);
    if coarse_intervals == 0 {
        return (0, f64::INFINITY);
    }
    let reference_ray = resolved[reference_index].pixel_to_ray(centre);
    let active_targets = (0..cameras.len())
        .filter(|&target_index| {
            target_index != reference_index
                && cameras[target_index].match_evidence_enabled
                && cameras[target_index].luminance.is_some()
                && cameras[target_index].calibration.is_some()
                && cameras[target_index].state.is_some()
        })
        .collect::<Vec<_>>();

    let project = |target_index: usize, t: f64| {
        let inverse_depth = min_inverse + t * (max_inverse - min_inverse);
        let point = add(
            reference_ray.origin,
            scale(reference_ray.direction, 1.0 / inverse_depth),
        );
        resolved[target_index]
            .project(point)
            .filter(|pixel| resolved[target_index].contains(*pixel))
    };

    // Start with the coarse grid, then insert only the new midpoint samples at
    // each refinement. Previously projected depth hypotheses are retained.
    let mut target_samples = active_targets
        .iter()
        .map(|&target_index| {
            (0..=coarse_intervals)
                .map(|index| project(target_index, index as f64 / coarse_intervals as f64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut intervals = coarse_intervals;
    for level in 0..=maximum {
        let mut maximum_step = 0.0f64;
        for samples in &target_samples {
            let mut previous: Option<Vec2> = None;
            for &sample in samples {
                if let (Some(a), Some(b)) = (previous, sample) {
                    let delta = [b[0] - a[0], b[1] - a[1]];
                    maximum_step = maximum_step.max(dot2(delta, delta).sqrt());
                }
                previous = sample;
            }
        }
        if maximum_step <= options.physical_match_max_projected_step_px || level >= maximum {
            return (level, maximum_step);
        }

        let next_intervals = intervals.saturating_mul(2);
        for (target_slot, &target_index) in active_targets.iter().enumerate() {
            let old = std::mem::take(&mut target_samples[target_slot]);
            let mut refined = Vec::with_capacity(next_intervals + 1);
            for (index, sample) in old.into_iter().enumerate() {
                refined.push(sample);
                if index < intervals {
                    let midpoint_index = index * 2 + 1;
                    refined.push(project(
                        target_index,
                        midpoint_index as f64 / next_intervals as f64,
                    ));
                }
            }
            target_samples[target_slot] = refined;
        }
        intervals = next_intervals;
    }
    unreachable!("refinement loop always returns at maximum level")
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
    reference_rays: &ReferencePatchRays,
    reference_patches: &[Option<ReferenceZnccPatch>],
    centre: Vec2,
    available_levels: usize,
    options: &RigRefinementOptions,
    scratch: &mut PhysicalMatchScratch,
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
    let Some(reference_patch) = reference_patches[refinement_levels].as_ref() else {
        return (
            Vec::new(),
            evaluated,
            refinement_levels,
            final_projected_step,
        );
    };
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
            reference_rays,
            reference_patch,
            centre,
            label,
            inverse_depth,
            refinement_levels,
            options,
            scratch,
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
        let mut labels = Vec::<usize>::with_capacity(beam.len() * 5);
        for candidate in beam {
            let centre_label = candidate.label.saturating_mul(2);
            let first = centre_label.saturating_sub(2);
            let last = centre_label.saturating_add(2).min(intervals);
            labels.extend(first..=last);
        }
        labels.sort_unstable();
        labels.dedup();
        candidates = Vec::with_capacity(labels.len());
        let pyramid_level = refinement_levels - refinement;
        let Some(reference_patch) = reference_patches[pyramid_level].as_ref() else {
            return (
                Vec::new(),
                evaluated,
                refinement_levels,
                final_projected_step,
            );
        };
        for label in labels {
            let t = label as f64 / intervals as f64;
            let inverse_depth = min_inverse + t * (max_inverse - min_inverse);
            evaluated += 1;
            if let Some(candidate) = evaluate_physical_depth_candidate(
                cameras,
                pyramids,
                reference_index,
                resolved,
                measured_alignments,
                reference_rays,
                reference_patch,
                centre,
                label,
                inverse_depth,
                pyramid_level,
                options,
                scratch,
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
    let mut match_scratch = PhysicalMatchScratch::default();

    for &(centre, structure) in candidates {
        let reference_rays = reference_patch_rays(reference_camera, centre);
        let reference_patches = (0..=available_pyramid_levels)
            .map(|level| {
                pyramids[reference_index]
                    .level(cameras[reference_index].luminance, level)
                    .and_then(|reference| {
                        prepare_reference_zncc_patch(
                            reference,
                            centre,
                            options.physical_match_patch_radius,
                            level,
                        )
                    })
            })
            .collect::<Vec<_>>();
        if reference_patches.first().and_then(Option::as_ref).is_none() {
            rejected_no_supported_depth += 1;
            continue;
        }
        let (mut depth_candidates, evaluated, refinement_levels, projected_step) =
            hierarchical_physical_depth_candidates(
                cameras,
                pyramids,
                reference_index,
                resolved,
                measured_alignments,
                &reference_rays,
                &reference_patches,
                centre,
                available_pyramid_levels,
                options,
                &mut match_scratch,
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
            prepared: Default::default(),
        });
        for matched in best.matches {
            let matched = refine_physical_view_match(
                &cameras[matched.camera],
                &reference_rays,
                reference_patches[0]
                    .as_ref()
                    .expect("native reference patch prepared"),
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
                localization_covariance: physical_localization_covariance(matched.epipolar_tangent),
                fixed_gauge: false,
                confidence: f64::from(matched.score),
                local_scale: matched.local_scale,
                structure: f64::from(structure),
                depth_reliability: Some(f64::from(depth_reliability)),
                prepared: Default::default(),
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

fn merge_physical_track_build(combined: &mut PhysicalTrackBuild, mut chunk: PhysicalTrackBuild) {
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
    let dynamic_chunk = 32usize.min(candidates.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let pyramids = &pyramids;
    let chunks = thread::scope(|scope| {
        (0..workers)
            .map(|_| {
                let next = &next;
                let candidates = &candidates;
                scope.spawn(move || {
                    let mut local = PhysicalTrackBuild::default();
                    loop {
                        let start =
                            next.fetch_add(dynamic_chunk, std::sync::atomic::Ordering::Relaxed);
                        if start >= candidates.len() {
                            break;
                        }
                        let end = (start + dynamic_chunk).min(candidates.len());
                        let chunk = build_physical_track_chunk(
                            &candidates[start..end],
                            cameras,
                            pyramids,
                            reference_index,
                            resolved,
                            measured_alignments,
                            available_pyramid_levels,
                            options,
                        );
                        merge_physical_track_build(&mut local, chunk);
                    }
                    local
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("physical matcher worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut combined = PhysicalTrackBuild::default();
    for chunk in chunks {
        merge_physical_track_build(&mut combined, chunk);
    }
    // Dynamic scheduling changes completion order only. Restore the original
    // spatial candidate order before any deterministic fit/validation split or
    // floating-point accumulation consumes the tracks.
    combined
        .tracks
        .sort_by_key(|track| (track.key[1], track.key[0]));
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

const FALLBACK_EPIPOLAR_RANSAC_THRESHOLD_REFERENCE_PX: f64 = 3.0;
const FALLBACK_EPIPOLAR_MIN_INLIER_RATIO: f64 = 0.45;
const FALLBACK_EPIPOLAR_MIN_INLIERS: usize = 24;
// Scoring every one of 10k+ correspondences for every minimal RANSAC model is
// needlessly expensive. Use a deterministic representative subset to rank
// hypotheses, then classify and refit on the complete population.
const FALLBACK_EPIPOLAR_RANSAC_SCORE_CAP: usize = 2_500;

fn normalize_epipolar_points(points: &[Vec2]) -> Option<(Vec<Vec2>, Mat3)> {
    if points.is_empty() {
        return None;
    }
    let count = points.len() as f64;
    let mean = [
        points.iter().map(|point| point[0]).sum::<f64>() / count,
        points.iter().map(|point| point[1]).sum::<f64>() / count,
    ];
    let mean_distance = points
        .iter()
        .map(|point| {
            let delta = [point[0] - mean[0], point[1] - mean[1]];
            dot2(delta, delta).sqrt()
        })
        .sum::<f64>()
        / count;
    if !mean_distance.is_finite() || mean_distance <= 1.0e-9 {
        return None;
    }
    let normalization_scale = std::f64::consts::SQRT_2 / mean_distance;
    let transform = [
        [normalization_scale, 0.0, -normalization_scale * mean[0]],
        [0.0, normalization_scale, -normalization_scale * mean[1]],
        [0.0, 0.0, 1.0],
    ];
    let normalized = points
        .iter()
        .map(|point| {
            [
                (point[0] - mean[0]) * normalization_scale,
                (point[1] - mean[1]) * normalization_scale,
            ]
        })
        .collect();
    Some((normalized, transform))
}

/// Smallest eigenvector of a real symmetric 9x9 matrix by Jacobi rotations.
/// The fallback epipolar verifier only runs when the preferred physical matcher
/// could not establish enough tracks, so this deliberately avoids pulling a
/// heavyweight linear-algebra dependency into the fusion crate.
fn smallest_symmetric_eigenvector_9(mut matrix: [[f64; 9]; 9]) -> Option<[f64; 9]> {
    let mut eigenvectors = [[0.0; 9]; 9];
    for (index, row) in eigenvectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    for _ in 0..192 {
        let mut pivot = (0usize, 1usize);
        let mut largest = matrix[0][1].abs();
        for row in 0..9 {
            for column in row + 1..9 {
                let magnitude = matrix[row][column].abs();
                if magnitude > largest {
                    largest = magnitude;
                    pivot = (row, column);
                }
            }
        }
        if !largest.is_finite() {
            return None;
        }
        if largest <= 1.0e-12 {
            break;
        }

        let (p, q) = pivot;
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let cosine = angle.cos();
        let sine = angle.sin();

        for index in 0..9 {
            if index == p || index == q {
                continue;
            }
            let aip = matrix[index][p];
            let aiq = matrix[index][q];
            let rotated_p = cosine * aip - sine * aiq;
            let rotated_q = sine * aip + cosine * aiq;
            matrix[index][p] = rotated_p;
            matrix[p][index] = rotated_p;
            matrix[index][q] = rotated_q;
            matrix[q][index] = rotated_q;
        }
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for row in &mut eigenvectors {
            let vip = row[p];
            let viq = row[q];
            row[p] = cosine * vip - sine * viq;
            row[q] = sine * vip + cosine * viq;
        }
    }

    let smallest =
        (0..9).min_by(|&first, &second| matrix[first][first].total_cmp(&matrix[second][second]))?;
    let mut vector: [f64; 9] = std::array::from_fn(|row| eigenvectors[row][smallest]);
    let length = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !length.is_finite() || length <= 1.0e-12 {
        return None;
    }
    for value in &mut vector {
        *value /= length;
    }
    Some(vector)
}

fn smallest_symmetric_eigenvector_3(mut matrix: Mat3) -> Option<Vec3> {
    let mut eigenvectors = math::IDENTITY;

    for _ in 0..48 {
        let mut pivot = (0usize, 1usize);
        let mut largest = matrix[0][1].abs();
        for row in 0..3 {
            for column in row + 1..3 {
                let magnitude = matrix[row][column].abs();
                if magnitude > largest {
                    largest = magnitude;
                    pivot = (row, column);
                }
            }
        }
        if !largest.is_finite() {
            return None;
        }
        if largest <= 1.0e-15 {
            break;
        }

        let (p, q) = pivot;
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let cosine = angle.cos();
        let sine = angle.sin();

        for index in 0..3 {
            if index == p || index == q {
                continue;
            }
            let aip = matrix[index][p];
            let aiq = matrix[index][q];
            let rotated_p = cosine * aip - sine * aiq;
            let rotated_q = sine * aip + cosine * aiq;
            matrix[index][p] = rotated_p;
            matrix[p][index] = rotated_p;
            matrix[index][q] = rotated_q;
            matrix[q][index] = rotated_q;
        }
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for row in &mut eigenvectors {
            let vip = row[p];
            let viq = row[q];
            row[p] = cosine * vip - sine * viq;
            row[q] = sine * vip + cosine * viq;
        }
    }

    let smallest =
        (0..3).min_by(|&first, &second| matrix[first][first].total_cmp(&matrix[second][second]))?;
    let mut vector: Vec3 = std::array::from_fn(|row| eigenvectors[row][smallest]);
    let length = norm(vector);
    if !length.is_finite() || length <= 1.0e-14 {
        return None;
    }
    vector = math::scale(vector, 1.0 / length);
    Some(vector)
}

/// Project a 3x3 matrix onto the closest rank-2 matrix in Frobenius norm.
/// A true fundamental matrix is rank 2; allowing the unconstrained 8-point
/// fit to remain full-rank gives RANSAC an extra degree of freedom that can
/// absorb exactly the repeated-structure outliers this verifier exists to
/// reject.  If v is the right singular vector for the smallest singular
/// value, F - (F v) v^T is the rank-2 SVD projection without requiring a
/// general-purpose SVD dependency.
fn enforce_rank_two(matrix: Mat3) -> Option<Mat3> {
    let transpose = math::transpose(&matrix);
    let normal = math::mul(&transpose, &matrix);
    let smallest_right = smallest_symmetric_eigenvector_3(normal)?;
    let smallest_component = mul_vec(&matrix, smallest_right);
    let rank_two: Mat3 = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            matrix[row][column] - smallest_component[row] * smallest_right[column]
        })
    });
    let magnitude = rank_two
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !magnitude.is_finite() || magnitude <= 1.0e-14 {
        return None;
    }
    Some(rank_two.map(|row| row.map(|value| value / magnitude)))
}

fn fit_fallback_fundamental(
    correspondences: &[AlignmentCorrespondence],
    indices: &[usize],
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
) -> Option<Mat3> {
    if indices.len() < 8 {
        return None;
    }
    // Work in distortion-corrected normalized camera coordinates. The old
    // fallback fitted F directly to distorted raster pixels, which lets lens
    // distortion (especially on the narrow-FOV C modules near the edge) leak
    // into the epipolar model and weaken the RANSAC identity test.
    let reference = indices
        .iter()
        .map(|&index| {
            reference_camera.pixel_to_normalized_camera(correspondences[index].reference_pixel)
        })
        .collect::<Option<Vec<_>>>()?;
    let target = indices
        .iter()
        .map(|&index| target_camera.pixel_to_normalized_camera(correspondences[index].target_pixel))
        .collect::<Option<Vec<_>>>()?;
    let (reference, reference_transform) = normalize_epipolar_points(&reference)?;
    let (target, target_transform) = normalize_epipolar_points(&target)?;

    // Homogeneous least squares: the smallest eigenvector of A^T A minimises
    // x2^T E x1 in calibrated coordinates. Enforce the mandatory rank-2
    // epipolar-matrix constraint
    // before denormalisation; otherwise the ninth degree of freedom can make
    // a minimal sample spuriously explain repeated-structure outliers.
    let mut normal = [[0.0f64; 9]; 9];
    for (reference, target) in reference.iter().zip(&target) {
        let row = [
            target[0] * reference[0],
            target[0] * reference[1],
            target[0],
            target[1] * reference[0],
            target[1] * reference[1],
            target[1],
            reference[0],
            reference[1],
            1.0,
        ];
        for first in 0..9 {
            for second in first..9 {
                normal[first][second] += row[first] * row[second];
            }
        }
    }
    for row in 0..9 {
        for column in 0..row {
            normal[row][column] = normal[column][row];
        }
    }
    let vector = smallest_symmetric_eigenvector_9(normal)?;
    let normalized = enforce_rank_two([
        [vector[0], vector[1], vector[2]],
        [vector[3], vector[4], vector[5]],
        [vector[6], vector[7], vector[8]],
    ])?;
    let denormalized = math::mul(
        &math::transpose(&target_transform),
        &math::mul(&normalized, &reference_transform),
    );
    let magnitude = denormalized
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !magnitude.is_finite() || magnitude <= 1.0e-14 {
        return None;
    }
    Some(denormalized.map(|row| row.map(|value| value / magnitude)))
}

fn fallback_epipolar_error(
    epipolar: &Mat3,
    correspondence: &AlignmentCorrespondence,
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
) -> f64 {
    let Some(reference_xy) =
        reference_camera.pixel_to_normalized_camera(correspondence.reference_pixel)
    else {
        return f64::INFINITY;
    };
    let Some(target_xy) = target_camera.pixel_to_normalized_camera(correspondence.target_pixel)
    else {
        return f64::INFINITY;
    };
    let reference = [reference_xy[0], reference_xy[1], 1.0];
    let target = [target_xy[0], target_xy[1], 1.0];
    let target_line = mul_vec(epipolar, reference);
    let reference_line = mul_vec(&math::transpose(epipolar), target);
    let numerator = dot(target, target_line).abs();
    let target_denominator =
        (target_line[0] * target_line[0] + target_line[1] * target_line[1]).sqrt();
    let reference_denominator =
        (reference_line[0] * reference_line[0] + reference_line[1] * reference_line[1]).sqrt();
    if !target_denominator.is_finite()
        || !reference_denominator.is_finite()
        || target_denominator <= 1.0e-12
        || reference_denominator <= 1.0e-12
    {
        return f64::INFINITY;
    }
    // Symmetric point-to-epipolar-line distance, converted from normalized
    // camera coordinates back to an approximate sensor-pixel scale separately
    // in each view. This keeps B<->C pairs on the same acceptance threshold.
    let target_pixels = numerator / target_denominator * target_camera.focal_scale_px();
    let reference_pixels = numerator / reference_denominator * reference_camera.focal_scale_px();
    ((target_pixels * target_pixels + reference_pixels * reference_pixels) * 0.5).sqrt()
}

// Additional pure-bearing consensus used only when the legacy 2-D alignment
// itself was rejected.  In a distant L16 capture, correct B4<->target feature
// pairs agree with one relative camera rotation to a few hundredths of a
// degree, while a repeated railing can satisfy an epipolar line at the wrong
// position along that line.  This filter uses camera-frame bearings only, so
// it does not bake the factory extrinsic pose into the verification.
const FALLBACK_ROTATION_RANSAC_THRESHOLD_DEGREES: f64 = 0.12;
const FALLBACK_ROTATION_RANSAC_TRIALS: usize = 2048;
const FALLBACK_ROTATION_MIN_INLIERS: usize = 24;
const FALLBACK_ROTATION_MIN_RATIO: f64 = 0.35;

fn basis_from_two_directions(first: Vec3, second: Vec3) -> Option<Mat3> {
    let e0 = normalize(first);
    let orthogonal = sub(second, math::scale(e0, dot(second, e0)));
    let length = norm(orthogonal);
    if !length.is_finite() || length <= 1.0e-5 {
        return None;
    }
    let e1 = math::scale(orthogonal, 1.0 / length);
    let e2 = normalize(cross(e0, e1));
    Some([
        [e0[0], e1[0], e2[0]],
        [e0[1], e1[1], e2[1]],
        [e0[2], e1[2], e2[2]],
    ])
}

fn relative_rotation_from_two_matches(
    first: &AlignmentCorrespondence,
    second: &AlignmentCorrespondence,
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
) -> Option<Mat3> {
    let target_basis = basis_from_two_directions(
        target_camera.pixel_to_camera_direction(first.target_pixel),
        target_camera.pixel_to_camera_direction(second.target_pixel),
    )?;
    let reference_basis = basis_from_two_directions(
        reference_camera.pixel_to_camera_direction(first.reference_pixel),
        reference_camera.pixel_to_camera_direction(second.reference_pixel),
    )?;
    Some(math::mul(&reference_basis, &math::transpose(&target_basis)))
}

fn fallback_rotation_error_degrees(
    rotation: &Mat3,
    correspondence: &AlignmentCorrespondence,
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
) -> f64 {
    let target = target_camera.pixel_to_camera_direction(correspondence.target_pixel);
    let reference = reference_camera.pixel_to_camera_direction(correspondence.reference_pixel);
    let predicted = normalize(mul_vec(rotation, target));
    let sine = norm(cross(predicted, reference));
    let cosine = dot(predicted, reference).clamp(-1.0, 1.0);
    sine.atan2(cosine).to_degrees().abs()
}

fn fallback_rotation_inliers(
    correspondences: &[AlignmentCorrespondence],
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
) -> Vec<usize> {
    if correspondences.len() < FALLBACK_ROTATION_MIN_INLIERS {
        return Vec::new();
    }
    let mut state = 0xA076_1D64_78BD_642Fu64 ^ correspondences.len() as u64;
    let mut next = |limit: usize| {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(2685821657736338717) as usize) % limit
    };
    let mut best = Vec::new();
    let mut best_error = f64::INFINITY;
    for _ in 0..FALLBACK_ROTATION_RANSAC_TRIALS {
        let first = next(correspondences.len());
        let mut second = next(correspondences.len());
        if second == first {
            second = (second + 1) % correspondences.len();
        }
        let Some(rotation) = relative_rotation_from_two_matches(
            &correspondences[first],
            &correspondences[second],
            reference_camera,
            target_camera,
        ) else {
            continue;
        };
        let mut inliers = Vec::new();
        let mut error_sum = 0.0;
        for (index, correspondence) in correspondences.iter().enumerate() {
            let error = fallback_rotation_error_degrees(
                &rotation,
                correspondence,
                reference_camera,
                target_camera,
            );
            if error <= FALLBACK_ROTATION_RANSAC_THRESHOLD_DEGREES {
                inliers.push(index);
                error_sum += error;
            }
        }
        if inliers.len() > best.len() || (inliers.len() == best.len() && error_sum < best_error) {
            best = inliers;
            best_error = error_sum;
        }
    }
    if best.len() < FALLBACK_ROTATION_MIN_INLIERS
        || (best.len() as f64 / correspondences.len() as f64) < FALLBACK_ROTATION_MIN_RATIO
    {
        Vec::new()
    } else {
        best
    }
}

/// Robust, image-only verification of sparse reference/target feature
/// correspondences in distortion-corrected calibrated coordinates. The
/// physical matcher already has its own multi-view depth consistency and does
/// not pass through this function.
fn fallback_epipolar_inliers(
    correspondences: &[AlignmentCorrespondence],
    reference_camera: &ResolvedCamera,
    target_camera: &ResolvedCamera,
) -> Vec<usize> {
    // Never silently trust a tiny sparse population.  V7 bypassed RANSAC for
    // fewer than 24 matches, which meant the hardest C cameras (C2/C3) were
    // precisely the ones whose correspondences received *no* independent
    // geometric verification.  Eight points are enough to fit the normalized
    // rank-2 model; for small populations require a stronger consensus ratio
    // instead of skipping verification.
    if correspondences.len() < 8 {
        return Vec::new();
    }
    let minimum_inliers = if correspondences.len() < FALLBACK_EPIPOLAR_MIN_INLIERS {
        ((correspondences.len() as f64 * 0.60).ceil() as usize).max(8)
    } else {
        FALLBACK_EPIPOLAR_MIN_INLIERS
    };
    let minimum_ratio = if correspondences.len() < FALLBACK_EPIPOLAR_MIN_INLIERS {
        0.60
    } else {
        FALLBACK_EPIPOLAR_MIN_INLIER_RATIO
    };

    let threshold = FALLBACK_EPIPOLAR_RANSAC_THRESHOLD_REFERENCE_PX;
    let mut state = 0xD1B5_4A32_D192_ED03u64 ^ correspondences.len() as u64;
    let mut next = |limit: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as usize) % limit
    };

    let score_count = correspondences
        .len()
        .min(FALLBACK_EPIPOLAR_RANSAC_SCORE_CAP);
    let score_indices = (0..score_count)
        .map(|slot| slot * correspondences.len() / score_count)
        .collect::<Vec<_>>();
    let mut best_fundamental = None::<Mat3>;
    let mut best_count = 0usize;
    let mut best_error = f64::INFINITY;
    for _ in 0..4096 {
        let mut sample = [0usize; 8];
        for slot in 0..sample.len() {
            loop {
                let candidate = next(correspondences.len());
                if !sample[..slot].contains(&candidate) {
                    sample[slot] = candidate;
                    break;
                }
            }
        }
        let Some(fundamental) =
            fit_fallback_fundamental(correspondences, &sample, reference_camera, target_camera)
        else {
            continue;
        };
        let mut inlier_count = 0usize;
        let mut error_sum = 0.0;
        for &index in &score_indices {
            let correspondence = &correspondences[index];
            let error = fallback_epipolar_error(
                &fundamental,
                correspondence,
                reference_camera,
                target_camera,
            );
            if error <= threshold {
                inlier_count += 1;
                error_sum += error;
            }
        }
        if inlier_count > best_count || (inlier_count == best_count && error_sum < best_error) {
            best_fundamental = Some(fundamental);
            best_count = inlier_count;
            best_error = error_sum;
        }
    }

    let Some(best_fundamental) = best_fundamental else {
        return Vec::new();
    };
    // Rank hypotheses on the capped subset, but decide membership on every
    // correspondence so the extra 15k-feature search actually reaches BA.
    let best = correspondences
        .iter()
        .enumerate()
        .filter(|(_, correspondence)| {
            fallback_epipolar_error(
                &best_fundamental,
                correspondence,
                reference_camera,
                target_camera,
            ) <= threshold
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    if best.len() < minimum_inliers
        || (best.len() as f64 / correspondences.len() as f64) < minimum_ratio
    {
        return Vec::new();
    }

    // One all-inlier refit removes the minimal-sample noise, followed by a
    // final deterministic inlier classification.
    let Some(fundamental) =
        fit_fallback_fundamental(correspondences, &best, reference_camera, target_camera)
    else {
        return best;
    };
    let refined = correspondences
        .iter()
        .enumerate()
        .filter(|(_, correspondence)| {
            fallback_epipolar_error(
                &fundamental,
                correspondence,
                reference_camera,
                target_camera,
            ) <= threshold
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if refined.len() >= minimum_inliers
        && (refined.len() as f64 / correspondences.len() as f64) >= minimum_ratio
    {
        refined
    } else {
        best
    }
}

fn build_tracks(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    alignments: &[ModuleAlignment],
    resolved: &[ResolvedCamera],
) -> (Vec<Track>, usize) {
    let mut tracks = Vec::<Track>::new();
    let mut pairwise_matches = 0;
    for (camera, alignment) in alignments.iter().enumerate() {
        if camera == reference_index
            || !cameras[camera].match_evidence_enabled
            || cameras[camera].calibration.is_none()
        {
            continue;
        }

        // The alignment homography remains a 2-D warp diagnostic only. Sparse
        // rig correspondences are genuine point features with their own
        // mutual/uniqueness checks, then independently filtered here in
        // distortion-corrected calibrated epipolar coordinates.
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
            // The extra rotation consensus is a low-parallax narrow-FOV aid,
            // not a replacement for epipolar geometry. If no dominant
            // infinity-like component exists (e.g. a genuinely close 3-D
            // scene), preserve the ordinary epipolar population. When it does
            // exist, intersect the two independent tests; this is what rejects
            // C3's along-epipolar repeated-structure aliases.
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
        for &correspondence_index in &epipolar_inliers {
            let correspondence = &alignment.correspondences[correspondence_index];
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

            // Keep independently verified B4<->target observations pairwise.
            // Matching the same B4 corner in two target cameras does *not*
            // prove that both targets selected the same repeated structure
            // along the reference epipolar ray. V5 re-merged those matches and
            // a single bad target observation could make an otherwise good
            // 3+-view validation track triangulate nearly at infinity, blowing
            // up every camera's residual. Multi-view tracks should only be
            // formed after an additional target<->target consistency check.
            tracks.push(Track {
                key,
                observations: vec![
                    TrackObservation {
                        camera: reference_index,
                        pixel: correspondence.reference_pixel,
                        bootstrap_residual_proposal: [0.0, 0.0],
                        localization_covariance: correspondence
                            .reference_localization_covariance
                            .map(|row| row.map(f64::from)),
                        fixed_gauge: true,
                        confidence: 1.0,
                        local_scale: 1.0,
                        structure: f64::from(correspondence.structure),
                        depth_reliability: None,
                        prepared: Default::default(),
                    },
                    TrackObservation {
                        camera,
                        pixel: correspondence.target_pixel,
                        bootstrap_residual_proposal: [0.0, 0.0],
                        localization_covariance: correspondence
                            .target_localization_covariance
                            .map(|row| row.map(f64::from)),
                        fixed_gauge: false,
                        confidence: f64::from(correspondence.confidence),
                        local_scale: f64::from(correspondence.local_scale),
                        structure: f64::from(correspondence.structure),
                        depth_reliability: correspondence.depth_reliability.map(f64::from),
                        prepared: Default::default(),
                    },
                ],
                condition: f64::NAN,
                max_ray_angle_degrees: f64::NAN,
            });
        }
    }
    (tracks, pairwise_matches)
}

/// Deterministic camera-local bearing bootstrap for low-parallax L16 captures.
///
/// The previous V8/V9 initializer minimized an essential/epipolar objective
/// and selected among cheirality branches. On the real distant capture that
/// objective could fit C3 with a ~10-degree effective bearing change despite
/// hundreds of image matches, because translation is too weak to distinguish
/// the branches reliably. World-bearing alignment has a single nearby basin
/// for this regime: finite scene parallax is handled by the robust loss and
/// finite-depth BA is left to the next phase.
fn bearing_bootstrap(
    specs: &[ParameterSpec],
    cameras: &[RigCameraInput<'_>],
    tracks: &[&Track],
    reference_index: usize,
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> (Vec<f64>, f64, usize) {
    let mut assembled = vec![0.0; specs.len()];
    let mut total_iterations = 0usize;
    let mut total_objective = 0.0f64;

    for camera in 0..cameras.len() {
        if camera == reference_index {
            continue;
        }
        let global_indices = specs
            .iter()
            .enumerate()
            .filter(|(_, spec)| spec.camera == camera)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if global_indices.is_empty() {
            continue;
        }
        let local_specs = global_indices
            .iter()
            .map(|&index| specs[index])
            .collect::<Vec<_>>();
        let local_tracks = tracks
            .iter()
            .copied()
            .filter(|track| {
                // The bearing bootstrap is explicitly reference-relative, so
                // only tracks that contain both endpoints carry information in
                // this phase. Non-reference B<->C/C<->C tracks remain in the
                // subsequent finite-depth bundle objective, where they do
                // constrain the connected camera graph.
                let has_camera = track
                    .observations
                    .iter()
                    .any(|observation| observation.camera == camera);
                let has_reference = track
                    .observations
                    .iter()
                    .any(|observation| observation.camera == reference_index);
                has_camera && has_reference
            })
            .collect::<Vec<_>>();
        if local_tracks.is_empty() {
            continue;
        }

        let (best_parameters, best_objective, best_iterations) = coordinate_optimize_rig(
            vec![0.0; local_specs.len()],
            &local_specs,
            options.max_iterations,
            cameras,
            &local_tracks,
            intrinsics_mode,
            options,
            IncrementalObjectiveMode::Bearing { reference_index },
        );
        for (local_index, &global_index) in global_indices.iter().enumerate() {
            assembled[global_index] = best_parameters[local_index];
        }
        total_iterations += best_iterations;
        if best_objective.is_finite() {
            total_objective += best_objective;
        }
    }

    (assembled, total_objective, total_iterations)
}

fn parameter_specs(
    cameras: &[RigCameraInput<'_>],
    reference_index: usize,
    tracks: &[&Track],
    options: &RigRefinementOptions,
) -> Vec<ParameterSpec> {
    let mut observations = vec![0usize; cameras.len()];
    let mut parallax = vec![Vec::<f64>::new(); cameras.len()];
    for track in tracks {
        for observation in &track.observations {
            observations[observation.camera] += 1;
            if track.max_ray_angle_degrees.is_finite() {
                parallax[observation.camera].push(track.max_ray_angle_degrees);
            }
        }
    }
    for values in &mut parallax {
        values.sort_by(f64::total_cmp);
    }
    let mut global_parallax = tracks
        .iter()
        .map(|track| track.max_ray_angle_degrees)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    global_parallax.sort_by(f64::total_cmp);

    // Bearing parameters need less finite-depth support than translation, but
    // the algebraic eight-point minimum is nowhere near enough to authorize a
    // multi-degree physical camera correction on a repeated-structure scene.
    // V8 let C3 optimize from only five fit tracks after the held-out split and
    // it found a perfectly cheiral but physically absurd ~10-degree branch.
    // The sparse matcher now performs a high-recall retry for narrow-FOV views;
    // if that still cannot supply roughly half of the normal per-camera support
    // budget, keep the factory camera and let the image residual warp handle it
    // rather than poisoning the physical rig/depth solve.
    let bearing_min_observations = (options.min_camera_observations / 2).max(16);
    const MIN_CENTER_PARALLAX_DEGREES: f64 = 0.10;
    let global_median_parallax = if global_parallax.is_empty() {
        f64::NAN
    } else {
        global_parallax[global_parallax.len() / 2]
    };
    let globally_depth_observable =
        global_median_parallax.is_finite() && global_median_parallax >= MIN_CENTER_PARALLAX_DEGREES;

    let mut specs = Vec::new();
    for (camera, input) in cameras.iter().enumerate() {
        if camera == reference_index
            || input.calibration.is_none()
            || input.state.is_none()
            || observations[camera] < bearing_min_observations
        {
            continue;
        }

        let high_magnification_c = input.name.starts_with('C');

        if options.max_orientation_degrees > 0.0 {
            for axis in 0..3 {
                specs.push(ParameterSpec {
                    camera,
                    affected_cameras: 1u16 << camera,
                    kind: ParameterKind::Orientation(axis),
                    bound: options.max_orientation_degrees,
                    prior_sigma: options.orientation_prior_sigma_degrees,
                    difference_step: 0.02,
                    maximum_update: (options.max_orientation_degrees / 6.0).clamp(0.25, 2.0),
                });
            }
        }

        // Translation is intrinsically weak in a distant single-capture scene.
        // Do not decide this from one camera in isolation: residual orientation
        // error can make that camera's rays cross at an apparently healthy
        // angle even when the capture as a whole has almost no metric-depth
        // leverage. V6 did exactly that for B1/B3/C4, releasing centre axes in
        // a globally ~0.05-degree scene and letting training depth absorb a
        // bearing error. Require both a robust capture-wide parallax signal and
        // per-camera support before moving any optical centre. Otherwise the
        // factory baselines remain the metric gauge.
        let median_parallax = if parallax[camera].is_empty() {
            f64::NAN
        } else {
            parallax[camera][parallax[camera].len() / 2]
        };
        let center_policy_allows = match options.strategy {
            RigRefinementStrategy::AnchorGraph => false,
            // V9.5's held-out residual/depth diagnostics show a strong 1/Z
            // component specifically on the 150-mm C modules. Requiring a
            // capture-wide 0.10-degree median parallax prevents those centre
            // terms from even reaching the stronger retriangulated
            // observability/rank test below. Release only C-camera centres in
            // LatentGraph and keep a tight factory prior; B cameras retain
            // their factory baseline gauge.
            RigRefinementStrategy::LatentGraph => {
                high_magnification_c && observations[camera] >= options.min_camera_observations
            }
            RigRefinementStrategy::Physical => {
                globally_depth_observable
                    && observations[camera] >= options.min_camera_observations
                    && median_parallax.is_finite()
                    && median_parallax >= MIN_CENTER_PARALLAX_DEGREES
            }
        };
        if center_policy_allows
            && options.max_center_offset > 0.0
            && options.max_focus_pupil_scale <= 0.0
        {
            for axis in 0..3 {
                specs.push(ParameterSpec {
                    camera,
                    affected_cameras: 1u16 << camera,
                    kind: ParameterKind::Center(axis),
                    bound: options.max_center_offset,
                    prior_sigma: options.center_prior_sigma,
                    difference_step: 0.05,
                    maximum_update: if options.strategy == RigRefinementStrategy::LatentGraph {
                        0.25
                    } else {
                        0.5
                    },
                });
            }
        }

        // Raster offset can be distinguished from rotation by field-dependent
        // evidence. LatentGraph can safely release it at the same support level
        // as bearing parameters because repeated-feature identity is jointly
        // resolved and the observability Jacobian below still rejects a raster
        // shift that is degenerate with rotation. This matters for narrow-FOV
        // cameras such as C1: the first real run retained only ~30 fit samples,
        // enough to estimate a coherent crop/principal-point shift but below the
        // generic 48-observation threshold, leaving a ~10 px systematic residual.
        let sensor_min_observations = if options.strategy == RigRefinementStrategy::LatentGraph {
            bearing_min_observations
        } else {
            options.min_camera_observations
        };
        if observations[camera] >= sensor_min_observations && options.max_sensor_offset_px > 0.0 {
            for axis in 0..2 {
                specs.push(ParameterSpec {
                    camera,
                    affected_cameras: 1u16 << camera,
                    kind: ParameterKind::Sensor(axis),
                    bound: options.max_sensor_offset_px,
                    prior_sigma: options.sensor_offset_prior_sigma_px,
                    difference_step: 0.25,
                    maximum_update: 4.0,
                });
            }
        }

        // The real latent-graph capture exposed a coherent field-dependent
        // residual on narrow C modules after orientation had converged.  A
        // small common focal scale is the minimal physical intrinsics DOF that
        // can explain that radial pattern without giving the optimizer enough
        // freedom to invent arbitrary calibration.  Keep this latent-only and
        // let the finite-difference rank/correlation test below decide whether
        // this capture actually observes it independently of rotation/raster.
        if options.strategy == RigRefinementStrategy::LatentGraph
            && observations[camera] >= bearing_min_observations
            && options.max_focal_scale_delta > 0.0
        {
            specs.push(ParameterSpec {
                camera,
                affected_cameras: 1u16 << camera,
                kind: ParameterKind::FocalScale,
                bound: options.max_focal_scale_delta,
                prior_sigma: options.focal_scale_prior_sigma,
                difference_step: 0.0005,
                maximum_update: 0.0025,
            });
        }

        // The v9.4 held-out field is already ~1 px for the 75-mm B cameras,
        // while every 150-mm C module retains a coherent 2-6 px spatial field.
        // Give only those high-magnification modules a minimal extra intrinsic
        // basis: one focal-aspect DOF plus small Brown k1/k2/p1/p2 deltas.
        // These are merely candidates here; the retriangulated Jacobian/rank
        // test below must independently observe each direction before it is
        // released into bundle adjustment.
        if options.strategy == RigRefinementStrategy::LatentGraph
            && high_magnification_c
            && observations[camera] >= bearing_min_observations
            && options.max_focal_aspect_delta > 0.0
        {
            specs.push(ParameterSpec {
                camera,
                affected_cameras: 1u16 << camera,
                kind: ParameterKind::FocalAspect,
                bound: options.max_focal_aspect_delta,
                prior_sigma: options.focal_aspect_prior_sigma,
                difference_step: 0.00025,
                maximum_update: 0.0015,
            });
        }
        if options.strategy == RigRefinementStrategy::LatentGraph
            && high_magnification_c
            && observations[camera] >= bearing_min_observations
            && input
                .calibration
                .is_some_and(|calibration| calibration.distortion.is_some())
        {
            if options.max_distortion_center_offset_px > 0.0 {
                for axis in 0..2 {
                    specs.push(ParameterSpec {
                        camera,
                        affected_cameras: 1u16 << camera,
                        kind: ParameterKind::DistortionCenter(axis),
                        bound: options.max_distortion_center_offset_px,
                        prior_sigma: options.distortion_center_prior_sigma_px,
                        difference_step: 0.25,
                        maximum_update: 1.0,
                    });
                }
            }
            if options.max_distortion_k1_delta > 0.0 {
                specs.push(ParameterSpec {
                    camera,
                    affected_cameras: 1u16 << camera,
                    kind: ParameterKind::DistortionRadial(0),
                    bound: options.max_distortion_k1_delta,
                    prior_sigma: options.distortion_k1_prior_sigma,
                    difference_step: 0.001,
                    maximum_update: 0.004,
                });
            }
            if options.max_distortion_k2_delta > 0.0 {
                specs.push(ParameterSpec {
                    camera,
                    affected_cameras: 1u16 << camera,
                    kind: ParameterKind::DistortionRadial(1),
                    bound: options.max_distortion_k2_delta,
                    prior_sigma: options.distortion_k2_prior_sigma,
                    difference_step: 0.0025,
                    maximum_update: 0.010,
                });
            }
            if options.max_distortion_tangential_delta > 0.0 {
                for axis in 0..2 {
                    specs.push(ParameterSpec {
                        camera,
                        affected_cameras: 1u16 << camera,
                        kind: ParameterKind::DistortionTangential(axis),
                        bound: options.max_distortion_tangential_delta,
                        prior_sigma: options.distortion_tangential_prior_sigma,
                        difference_step: 0.00025,
                        maximum_update: 0.001,
                    });
                }
            }
        }

        if options.strategy != RigRefinementStrategy::AnchorGraph
            && options.max_mirror_degrees > 0.0
            && input
                .calibration
                .is_some_and(|calibration| calibration.mirror.is_some())
        {
            specs.push(ParameterSpec {
                camera,
                affected_cameras: 1u16 << camera,
                kind: ParameterKind::Mirror,
                bound: options.max_mirror_degrees,
                prior_sigma: options.mirror_prior_sigma_degrees,
                difference_step: 0.01,
                maximum_update: (options.max_mirror_degrees / 6.0).clamp(0.10, 0.75),
            });
        }
    }
    if options.strategy == RigRefinementStrategy::LatentGraph && options.max_focus_pupil_scale > 0.0
    {
        for group in ['B', 'C'] {
            let affected_cameras = cameras
                .iter()
                .enumerate()
                .filter(|(camera, input)| {
                    input.name.starts_with(group)
                        && input.state.is_some()
                        && input.calibration.is_some_and(|calibration| {
                            calibration.cra.as_ref().is_some_and(|cra| {
                                cra.sensor_distance.is_some() && cra.distance_hall_ratio.is_some()
                            })
                        })
                        && (*camera == reference_index
                            || observations[*camera] >= bearing_min_observations)
                })
                .fold(0u16, |mask, (camera, _)| mask | (1u16 << camera));
            if affected_cameras.count_ones() < 2 {
                continue;
            }
            let camera = affected_cameras.trailing_zeros() as usize;
            specs.push(ParameterSpec {
                camera,
                affected_cameras,
                kind: ParameterKind::FocusPupilScale(group),
                bound: options.max_focus_pupil_scale,
                prior_sigma: options.focus_pupil_prior_sigma,
                difference_step: 0.025,
                maximum_update: 0.15,
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
        ParameterKind::FocusPupilScale(group) => format!("focus_pupil_scale_{group}"),
        ParameterKind::Sensor(0) => "sensor_x".to_owned(),
        ParameterKind::Sensor(1) => "sensor_y".to_owned(),
        ParameterKind::Sensor(axis) => format!("sensor_{axis}"),
        ParameterKind::FocalScale => "focal_scale".to_owned(),
        ParameterKind::FocalAspect => "focal_aspect".to_owned(),
        ParameterKind::DistortionCenter(0) => "distortion_center_x".to_owned(),
        ParameterKind::DistortionCenter(1) => "distortion_center_y".to_owned(),
        ParameterKind::DistortionCenter(axis) => format!("distortion_center_{axis}"),
        ParameterKind::DistortionRadial(0) => "distortion_k1".to_owned(),
        ParameterKind::DistortionRadial(1) => "distortion_k2".to_owned(),
        ParameterKind::DistortionRadial(axis) => format!("distortion_radial_{axis}"),
        ParameterKind::DistortionTangential(0) => "distortion_p1".to_owned(),
        ParameterKind::DistortionTangential(1) => "distortion_p2".to_owned(),
        ParameterKind::DistortionTangential(axis) => format!("distortion_tangential_{axis}"),
    }
}

/// Retriangulated normalized reprojection Jacobian used only to determine
/// whether one candidate physical parameter is independently observable. The
/// nuisance 3-D point is re-estimated on both sides of the finite difference,
/// so a parameter is observable only when its residual cannot be absorbed by
/// moving the track point. Unstable gated tracks get a zero derivative while
/// retaining fixed row correspondence between all parameter columns.
fn observability_jacobian_column(
    parameter_index: usize,
    specs: &[ParameterSpec],
    base_parameters: &[f64],
    parameter: &ParameterSpec,
    difference_step: f64,
    inputs: &[RigCameraInput<'_>],
    tracks: &[&Track],
    _base_cameras: &[ResolvedCamera],
    base_templates: &[ResolvedCameraTemplate],
    intrinsics_mode: IntrinsicsMode,
    options: &RigRefinementOptions,
) -> Option<Vec<f64>> {
    let centre = *base_parameters.get(parameter_index)?;
    let minus_value = (centre - difference_step).max(-parameter.bound);
    let plus_value = (centre + difference_step).min(parameter.bound);
    let finite_difference_span = plus_value - minus_value;
    if !finite_difference_span.is_finite() || finite_difference_span.abs() <= 1.0e-12 {
        return None;
    }
    let mut minus_parameters = base_parameters.to_vec();
    minus_parameters[parameter_index] = minus_value;
    let mut plus_parameters = base_parameters.to_vec();
    plus_parameters[parameter_index] = plus_value;
    let minus_refinements = refinements_from_parameters(inputs.len(), &minus_parameters, specs);
    let plus_refinements = refinements_from_parameters(inputs.len(), &plus_parameters, specs);
    let _ = intrinsics_mode;
    let minus_cameras = base_templates
        .iter()
        .zip(&minus_refinements)
        .map(|(template, refinement)| template.resolve(refinement).ok())
        .collect::<Option<Vec<_>>>()?;
    let plus_cameras = base_templates
        .iter()
        .zip(&plus_refinements)
        .map(|(template, refinement)| template.resolve(refinement).ok())
        .collect::<Option<Vec<_>>>()?;

    // Preserve fixed row correspondence for correlation testing, but only
    // evaluate tracks that actually observe the perturbed camera. All other
    // rows are mathematically zero for this column.
    let total_observations = tracks
        .iter()
        .map(|track| track.observations.len())
        .sum::<usize>();
    let mut minus_projected = Vec::<Option<Vec2>>::with_capacity(total_observations);
    let mut cameras = minus_cameras;
    let mut rays = Vec::new();
    for track in tracks {
        if !track
            .observations
            .iter()
            .any(|observation| parameter.affected_cameras & (1u16 << observation.camera) != 0)
        {
            minus_projected.extend(std::iter::repeat_n(None, track.observations.len()));
            continue;
        }
        let Some(triangulated) =
            triangulate_with_rays(&track.observations, &cameras, options, &mut rays)
        else {
            minus_projected.extend(std::iter::repeat_n(None, track.observations.len()));
            continue;
        };
        for (observation, &ray) in track.observations.iter().zip(&rays) {
            minus_projected.push(project_observation_with_ray(
                &cameras[observation.camera],
                ray,
                triangulated.point,
            ));
        }
    }

    cameras = plus_cameras;
    let mut column = Vec::with_capacity(total_observations * 2);
    let mut minus_index = 0usize;
    for track in tracks {
        if !track
            .observations
            .iter()
            .any(|observation| parameter.affected_cameras & (1u16 << observation.camera) != 0)
        {
            column.extend(std::iter::repeat_n(0.0, track.observations.len() * 2));
            minus_index += track.observations.len();
            continue;
        }
        let Some(triangulated) =
            triangulate_with_rays(&track.observations, &cameras, options, &mut rays)
        else {
            column.extend(std::iter::repeat_n(0.0, track.observations.len() * 2));
            minus_index += track.observations.len();
            continue;
        };
        for (local_index, (observation, &ray)) in track.observations.iter().zip(&rays).enumerate() {
            let minus = minus_projected[minus_index + local_index];
            let plus =
                project_observation_with_ray(&cameras[observation.camera], ray, triangulated.point);
            let sigma = observation_sigma(observation).max(1.0e-6);
            if let (Some(minus), Some(plus)) = (minus, plus) {
                let scale = parameter.prior_sigma / (finite_difference_span * sigma);
                column.push((plus[0] - minus[0]) * scale);
                column.push((plus[1] - minus[1]) * scale);
            } else {
                column.extend([0.0, 0.0]);
            }
        }
        minus_index += track.observations.len();
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

/// Fraction of a Jacobian column that remains after projection onto an
/// existing orthonormal basis. The observability columns already retriangulate
/// every affected track for each finite difference, so this is a cheap
/// Schur-like rank test on camera parameters after point motion has been
/// absorbed rather than a raw fixed-point camera Jacobian.
fn orthogonal_column_novelty(column: &[f64], basis: &[Vec<f64>]) -> (f64, Option<Vec<f64>>) {
    let norm0 = column.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm0.is_finite() || norm0 <= 1.0e-12 {
        return (0.0, None);
    }
    let mut residual = column.iter().map(|value| value / norm0).collect::<Vec<_>>();
    // Modified Gram-Schmidt with a second pass is materially more stable for
    // the almost-parallel bearing/raster directions seen in low-parallax L16
    // captures than a single pairwise correlation threshold.
    for _ in 0..2 {
        for direction in basis {
            if direction.len() != residual.len() {
                continue;
            }
            let projection = residual
                .iter()
                .zip(direction)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            for (value, &component) in residual.iter_mut().zip(direction) {
                *value -= projection * component;
            }
        }
    }
    let novelty = residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !novelty.is_finite() || novelty <= 1.0e-12 {
        return (0.0, None);
    }
    for value in &mut residual {
        *value /= novelty;
    }
    (novelty.clamp(0.0, 1.0), Some(residual))
}

/// Release physical parameters only when the data can actually see them.
/// Observation count is still used as the cheap first gate; this second gate
/// measures finite-difference sensitivity and removes same-camera nuisance
/// directions that are numerically indistinguishable. Generic orientation and
/// mirror parameters are deliberately retained even when strongly correlated:
/// they span the effective bearing model and are regularized by their factory
/// priors. V6 sometimes deleted two orientation axes from that span (B3 was
/// left with only z + mirror), producing sub-pixel fit residuals but a large
/// spatial held-out tail. Correlated centre/raster nuisance directions still
/// lose to the bearing basis.
fn filter_observable_parameter_specs(
    candidates: &[ParameterSpec],
    base_parameters: &[f64],
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
    if base_parameters.len() != candidates.len() {
        return (Vec::new(), Vec::new());
    }
    let base_refinements = refinements_from_parameters(inputs.len(), base_parameters, candidates);
    let base_templates = inputs
        .iter()
        .map(|input| {
            ResolvedCameraTemplate::new(input.calibration?, input.state?, intrinsics_mode).ok()
        })
        .collect::<Option<Vec<_>>>();
    let base_cameras = base_templates.as_ref().and_then(|templates| {
        templates
            .iter()
            .zip(&base_refinements)
            .map(|(template, refinement)| template.resolve(refinement).ok())
            .collect::<Option<Vec<_>>>()
    });
    let (Some(base_templates), Some(base_cameras)) = (base_templates, base_cameras) else {
        let reports = candidates
            .iter()
            .map(|spec| RigParameterObservabilityReport {
                camera: inputs[spec.camera].name.to_owned(),
                parameter: parameter_name(spec.kind),
                sensitivity_rms: 0.0,
                max_correlation: 0.0,
                optimized: false,
                rejection_reason: Some("factory camera model could not be resolved".to_owned()),
            })
            .collect();
        return (Vec::new(), reports);
    };
    let automatic = thread::available_parallelism().map_or(1, usize::from);
    let workers = if options.threads == 0 {
        automatic
    } else {
        options.threads.min(automatic)
    }
    .clamp(1, candidates.len().max(1));
    let parameters_per_worker = candidates.len().div_ceil(workers);
    let mut column_results = thread::scope(|scope| {
        let base_cameras = &base_cameras;
        let base_templates = &base_templates;
        let handles = candidates
            .chunks(parameters_per_worker.max(1))
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let first_index = chunk_index * parameters_per_worker.max(1);
                scope.spawn(move || {
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(local_index, spec)| {
                            let index = first_index + local_index;
                            let step = spec.difference_step.max(1.0e-6);
                            let column = observability_jacobian_column(
                                index,
                                candidates,
                                base_parameters,
                                spec,
                                step,
                                inputs,
                                tracks,
                                base_cameras,
                                base_templates,
                                intrinsics_mode,
                                options,
                            )
                            .unwrap_or_default();
                            let sensitivity = if column.is_empty() {
                                0.0
                            } else {
                                (column.iter().map(|value| value * value).sum::<f64>()
                                    / column.len() as f64)
                                    .sqrt()
                            };
                            (index, column, sensitivity)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("rig observability worker panicked"))
            .collect::<Vec<_>>()
    });
    column_results.sort_by_key(|(index, _, _)| *index);
    let mut columns = Vec::with_capacity(candidates.len());
    let mut sensitivities = Vec::with_capacity(candidates.len());
    for (_, column, sensitivity) in column_results {
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

    // A raster shift is only distinguishable from a small bearing change when
    // the current scene skeleton samples a substantial fraction of the image
    // in both axes.  Observation count alone is not enough: V8 released C4
    // sensor-y from a sparse/clustered narrow-FOV population and the nuisance
    // parameter absorbed scene/correspondence error while degrading held-out
    // geometry.  This is a parameter-observability test, not a camera quality
    // weight: orientation remains jointly optimised even when raster DOFs are
    // not independently identifiable.
    let mut camera_spread = vec![(0usize, 0.0f64, 0.0f64, 0usize); inputs.len()];
    for camera in 0..inputs.len() {
        let width = base_cameras[camera].width.max(1) as f64;
        let height = base_cameras[camera].height.max(1) as f64;
        let mut count = 0usize;
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut cells = std::collections::HashSet::<(usize, usize)>::new();
        for track in tracks {
            if let Some(observation) = track
                .observations
                .iter()
                .find(|observation| observation.camera == camera)
            {
                let [x, y] = observation.pixel;
                if !x.is_finite() || !y.is_finite() {
                    continue;
                }
                count += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                let cell_x = ((x / width).clamp(0.0, 0.999_999) * 4.0) as usize;
                let cell_y = ((y / height).clamp(0.0, 0.999_999) * 4.0) as usize;
                cells.insert((cell_x, cell_y));
            }
        }
        let span_x = if count >= 2 {
            ((max_x - min_x) / width).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let span_y = if count >= 2 {
            ((max_y - min_y) / height).clamp(0.0, 1.0)
        } else {
            0.0
        };
        camera_spread[camera] = (count, span_x, span_y, cells.len());
    }
    for (index, spec) in candidates.iter().enumerate() {
        if !keep[index]
            || !matches!(
                spec.kind,
                ParameterKind::Sensor(_)
                    | ParameterKind::FocalScale
                    | ParameterKind::FocalAspect
                    | ParameterKind::DistortionCenter(_)
                    | ParameterKind::DistortionRadial(_)
                    | ParameterKind::DistortionTangential(_)
            )
        {
            continue;
        }
        let (count, span_x, span_y, cells) = camera_spread[spec.camera];
        // LatentGraph has already paid for explicit correspondence ambiguity
        // handling and keeps a held-out geometry gate.  Requiring 64 samples
        // over 45% of both axes made the narrow C cameras mathematically unable
        // to correct a coherent principal-point/focal error even when the
        // Jacobian showed it was independently observable.  Use a smaller but
        // still spatially distributed support floor here and let the actual
        // retriangulated column rank test below reject degeneracy.
        let spatially_supported = if options.strategy == RigRefinementStrategy::LatentGraph {
            // The 150-mm C modules often overlap the 75-mm reference as a
            // narrow strip. Requiring >=20% span on *both* sensor axes rejected
            // C1's sensor/focal terms even though its retriangulated Jacobian
            // was strong and the held-out residual field showed a coherent
            // image-plane trend. Permit a narrow strip when it still spans a
            // useful major axis and at least two 4x4 cells; the rank/novelty
            // tests below remain the final guard against a nuisance DOF that is
            // actually degenerate with bearing.
            let minimum_count = (options.min_camera_observations / 2).max(16);
            let major_span = span_x.max(span_y);
            let minor_span = span_x.min(span_y);
            count >= minimum_count && major_span >= 0.20 && minor_span >= 0.08 && cells >= 2
        } else {
            count >= options.min_camera_observations.max(64)
                && span_x >= 0.45
                && span_y >= 0.45
                && cells >= 6
        };
        if !spatially_supported {
            keep[index] = false;
            reasons[index] = Some(format!(
                "insufficient spatial leverage for {} (n={count}, span={span_x:.2}x{span_y:.2}, cells={cells}/16)",
                parameter_name(spec.kind)
            ));
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
            let first_is_focal = matches!(
                candidates[first].kind,
                ParameterKind::FocalScale | ParameterKind::FocalAspect
            );
            let second_is_focal = matches!(
                candidates[second].kind,
                ParameterKind::FocalScale | ParameterKind::FocalAspect
            );
            let first_is_distortion_center =
                matches!(candidates[first].kind, ParameterKind::DistortionCenter(_));
            let second_is_distortion_center =
                matches!(candidates[second].kind, ParameterKind::DistortionCenter(_));
            let first_is_distortion = matches!(
                candidates[first].kind,
                ParameterKind::DistortionRadial(_) | ParameterKind::DistortionTangential(_)
            );
            let second_is_distortion = matches!(
                candidates[second].kind,
                ParameterKind::DistortionRadial(_) | ParameterKind::DistortionTangential(_)
            );
            let first_is_center = matches!(
                candidates[first].kind,
                ParameterKind::Center(_) | ParameterKind::FocusPupilScale(_)
            );
            let second_is_center = matches!(
                candidates[second].kind,
                ParameterKind::Center(_) | ParameterKind::FocusPupilScale(_)
            );
            let first_is_orientation =
                matches!(candidates[first].kind, ParameterKind::Orientation(_));
            let second_is_orientation =
                matches!(candidates[second].kind, ParameterKind::Orientation(_));
            let first_is_bearing = first_is_orientation || first_is_mirror;
            let second_is_bearing = second_is_orientation || second_is_mirror;

            // Do not collapse the bearing model merely because this particular
            // capture makes two bearing columns almost parallel. The factory
            // priors resolve their parameter decomposition; deleting one here
            // changes the *effective camera transform* after bootstrap. This
            // was the dominant V6 B3 failure: x/y orientation were discarded
            // in favour of a correlated mirror direction.
            if first_is_bearing && second_is_bearing {
                continue;
            }

            let first_is_nuisance = first_is_sensor
                || first_is_center
                || first_is_focal
                || first_is_distortion_center
                || first_is_distortion;
            let second_is_nuisance = second_is_sensor
                || second_is_center
                || second_is_focal
                || second_is_distortion_center
                || second_is_distortion;
            let first_is_image_nuisance = first_is_sensor
                || first_is_focal
                || first_is_distortion_center
                || first_is_distortion;
            let second_is_image_nuisance = second_is_sensor
                || second_is_focal
                || second_is_distortion_center
                || second_is_distortion;
            let loser = match (
                first_is_bearing,
                second_is_bearing,
                first_is_nuisance,
                second_is_nuisance,
            ) {
                // Raster and centre shifts are nuisance parameters. If they are
                // indistinguishable from an effective bearing correction, keep
                // the bearing basis and let the factory prior hold the nuisance
                // parameter fixed. Focal scale follows the same rule.
                (true, false, _, true) => second,
                (false, true, true, _) => first,
                // If metric centre and an image-plane nuisance are nearly the
                // same Jacobian direction, prefer the image-plane explanation.
                // Centre is intentionally released last and should survive only
                // when depth gives it genuinely new leverage.
                _ if first_is_center && second_is_image_nuisance => first,
                _ if second_is_center && first_is_image_nuisance => second,
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

    // Pairwise correlation misses multi-parameter null directions such as
    // 0.4*p1 - 0.7*p2 + 0.3*p3 ~= 0. Build a global pivoted orthogonal basis
    // over the retained retriangulated Jacobian columns. Bearing parameters are
    // deliberately seeded first and never deleted here: together with their
    // factory priors they define the effective camera transform. Nuisance
    // centre/raster directions must contribute genuinely new image evidence.
    const MIN_GLOBAL_NUISANCE_NOVELTY: f64 = 0.075;
    let mut order = (0..candidates.len())
        .filter(|&index| keep[index])
        .collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let priority = |kind: ParameterKind| match kind {
            ParameterKind::Orientation(_) | ParameterKind::Mirror => 3u8,
            ParameterKind::Center(_) | ParameterKind::FocusPupilScale(_) => 1u8,
            _ => 2u8,
        };
        priority(candidates[right].kind)
            .cmp(&priority(candidates[left].kind))
            .then_with(|| sensitivities[right].total_cmp(&sensitivities[left]))
    });
    let mut global_basis = Vec::<Vec<f64>>::new();
    for index in order {
        if columns[index].is_empty() || !keep[index] {
            continue;
        }
        let is_bearing = matches!(
            candidates[index].kind,
            ParameterKind::Orientation(_) | ParameterKind::Mirror
        );
        let (novelty, direction) = orthogonal_column_novelty(&columns[index], &global_basis);
        if !is_bearing && novelty < MIN_GLOBAL_NUISANCE_NOVELTY {
            keep[index] = false;
            reasons[index] = Some(format!(
                "globally rank-deficient after retriangulation (independent Jacobian fraction={novelty:.4})"
            ));
            continue;
        }
        // A nearly duplicate bearing direction is retained as a parameter but
        // need not be inserted into the numerical basis twice.
        if novelty >= 1.0e-4
            && let Some(direction) = direction
        {
            global_basis.push(direction);
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

fn remap_parameters(
    source_parameters: &[f64],
    source_specs: &[ParameterSpec],
    target_specs: &[ParameterSpec],
) -> Vec<f64> {
    target_specs
        .iter()
        .map(|target| {
            source_specs
                .iter()
                .enumerate()
                .find(|(_, source)| source.camera == target.camera && source.kind == target.kind)
                .map_or(0.0, |(index, _)| source_parameters[index])
        })
        .collect()
}

fn refinements_from_parameters(
    camera_count: usize,
    parameters: &[f64],
    specs: &[ParameterSpec],
) -> Vec<CameraRefinement> {
    let mut orientations = vec![[0.0; 3]; camera_count];
    let mut mirrors = vec![0.0; camera_count];
    let mut centers = vec![[0.0; 3]; camera_count];
    let mut focus_pupil_scales = vec![0.0; camera_count];
    let mut sensors = vec![[0.0; 2]; camera_count];
    let mut focal_scales = vec![0.0; camera_count];
    let mut focal_aspects = vec![0.0; camera_count];
    let mut distortion_centers = vec![[0.0; 2]; camera_count];
    let mut distortion_deltas = vec![[0.0; 4]; camera_count];
    for (&value, spec) in parameters.iter().zip(specs) {
        match spec.kind {
            ParameterKind::Orientation(axis) => orientations[spec.camera][axis] = value,
            ParameterKind::Mirror => mirrors[spec.camera] = value,
            ParameterKind::Center(axis) => centers[spec.camera][axis] = value,
            ParameterKind::FocusPupilScale(_) => {
                for (camera, scale) in focus_pupil_scales.iter_mut().enumerate() {
                    if spec.affected_cameras & (1u16 << camera) != 0 {
                        *scale = value;
                    }
                }
            }
            ParameterKind::Sensor(axis) => sensors[spec.camera][axis] = value,
            ParameterKind::FocalScale => focal_scales[spec.camera] = value,
            ParameterKind::FocalAspect => focal_aspects[spec.camera] = value,
            ParameterKind::DistortionCenter(axis) if axis < 2 => {
                distortion_centers[spec.camera][axis] = value
            }
            ParameterKind::DistortionRadial(axis) if axis < 2 => {
                distortion_deltas[spec.camera][axis] = value
            }
            ParameterKind::DistortionTangential(axis) if axis < 2 => {
                distortion_deltas[spec.camera][axis + 2] = value
            }
            ParameterKind::DistortionCenter(_)
            | ParameterKind::DistortionRadial(_)
            | ParameterKind::DistortionTangential(_) => {}
        }
    }
    (0..camera_count)
        .map(|camera| CameraRefinement {
            mirror_angle_offset_degrees: mirrors[camera],
            orientation_offset_degrees: (orientations[camera] != [0.0; 3])
                .then_some(orientations[camera]),
            center_offset_world: (centers[camera] != [0.0; 3]).then_some(centers[camera]),
            focus_pupil_scale: (focus_pupil_scales[camera] != 0.0)
                .then_some(focus_pupil_scales[camera]),
            sensor_offset_px: (sensors[camera] != [0.0; 2]).then_some(sensors[camera]),
            focal_scale_delta: (focal_scales[camera] != 0.0).then_some(focal_scales[camera]),
            focal_aspect_delta: (focal_aspects[camera] != 0.0).then_some(focal_aspects[camera]),
            distortion_center_offset_px: (distortion_centers[camera] != [0.0; 2])
                .then_some(distortion_centers[camera]),
            distortion_delta: (distortion_deltas[camera] != [0.0; 4])
                .then_some(distortion_deltas[camera]),
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum IncrementalObjectiveMode {
    /// Low-parallax initializer: align world-space bearing directions while
    /// deliberately ignoring translation/cheirality. This is much better
    /// conditioned than essential geometry when the scene is tens of metres
    /// away and the L16 baseline produces only a tiny parallax angle.
    Bearing {
        reference_index: usize,
    },
    Epipolar {
        reference_index: usize,
    },
    Bundle,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrackObjectiveContribution {
    cost: f64,
    samples: usize,
}

fn cheirality_penalty(depth: f64, baseline_scale: f64) -> f64 {
    if depth >= 0.0 {
        return 0.0;
    }
    let normalized = (-depth / baseline_scale.max(1.0e-6)).clamp(0.0, 1.0e6);
    // Smooth at the physical boundary and still strongly penalize a point that
    // runs far behind a camera. The old constant +10 had essentially no local
    // signal until a finite-difference step happened to cross depth=0.
    4.0 * normalized.ln_1p().powi(2)
}

fn ray_baseline_scale(rays: &[crate::geometry::Ray]) -> f64 {
    let mut scale = 0.0f64;
    for first in 0..rays.len() {
        for second in first + 1..rays.len() {
            scale = scale.max(norm(sub(rays[first].origin, rays[second].origin)));
        }
    }
    scale.max(1.0e-6)
}

fn track_authority_weight(track: &Track, options: &RigRefinementOptions) -> f64 {
    if options.strategy == RigRefinementStrategy::LatentGraph && track.observations.len() == 2 {
        options.latent_pairwise_bundle_weight.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn invalid_bundle_observation_cost(
    observation: &TrackObservation,
    track_size: usize,
    options: &RigRefinementOptions,
) -> f64 {
    // Use the same finite invalid-projection scale as held-out evaluation, but
    // pass it through the robust objective in normalized measurement units.
    // A flat +25 cost made it cheaper for bundle adjustment to throw a hard
    // observation outside the calibrated sensor than to explain a moderately
    // large in-frame residual, which could manufacture exactly the boundary
    // failures seen in the first latent-graph run.
    let normalized = INVALID_REPROJECTION_PENALTY_PX / observation_sigma(observation).max(1.0e-6);
    observation_balance(observation, track_size) * huber(normalized, options.huber_delta)
}

fn bundle_track_contribution(
    track: &Track,
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
    rays: &mut Vec<crate::geometry::Ray>,
) -> TrackObjectiveContribution {
    let Some(triangulated) = triangulate_with_rays(&track.observations, cameras, options, rays)
    else {
        return TrackObjectiveContribution {
            cost: track_authority_weight(track, options)
                * track
                    .observations
                    .iter()
                    .map(|observation| {
                        invalid_bundle_observation_cost(
                            observation,
                            track.observations.len(),
                            options,
                        )
                    })
                    .sum::<f64>(),
            samples: track.observations.len(),
        };
    };
    let baseline_scale = ray_baseline_scale(rays);
    let mut result = TrackObjectiveContribution::default();
    for (observation, &ray) in track.observations.iter().zip(rays.iter()) {
        let signed_depth = dot(sub(triangulated.point, ray.origin), ray.direction);
        result.cost += cheirality_penalty(signed_depth, baseline_scale);
        let Some(projected) =
            project_observation_with_ray(&cameras[observation.camera], ray, triangulated.point)
        else {
            result.cost +=
                invalid_bundle_observation_cost(observation, track.observations.len(), options);
            result.samples += 1;
            continue;
        };
        let residual = [
            projected[0] - observation.pixel[0],
            projected[1] - observation.pixel[1],
        ];
        let normalized = normalized_observation_residual(observation, residual);
        result.cost += observation_balance(observation, track.observations.len())
            * huber(normalized, options.huber_delta);
        result.samples += 1;
    }
    result.cost *= track_authority_weight(track, options);
    result
}

fn bearing_track_contribution(
    track: &Track,
    cameras: &[ResolvedCamera],
    reference_index: usize,
    options: &RigRefinementOptions,
) -> TrackObjectiveContribution {
    let Some(reference) = track
        .observations
        .iter()
        .find(|observation| observation.camera == reference_index)
    else {
        return TrackObjectiveContribution::default();
    };
    let reference_ray = cameras[reference_index].pixel_to_ray(reference.pixel);
    let mut result = TrackObjectiveContribution::default();
    for observation in track
        .observations
        .iter()
        .filter(|observation| observation.camera != reference_index)
    {
        let target_camera = &cameras[observation.camera];
        let target_ray = target_camera.pixel_to_ray(observation.pixel);
        // For a distant scene, corresponding world rays should be almost
        // parallel. Finite parallax remains as a bounded robust residual, but
        // unlike the essential/cheirality objective there are no mirrored
        // positive-depth branches for the optimizer to jump between.
        let angular_sine = norm(cross(reference_ray.direction, target_ray.direction));
        let focal = (cameras[reference_index].focal_px * target_camera.focal_px)
            .abs()
            .sqrt();
        let pair_sigma =
            (observation_sigma(reference).powi(2) + observation_sigma(observation).powi(2)).sqrt();
        let normalized = angular_sine * focal / pair_sigma.max(1.0e-6);
        // The bearing model intentionally treats genuine finite-scene
        // parallax as a robust residual. A slightly wider Huber transition
        // avoids letting nearby foreground rails steer the infinity rotation.
        result.cost += huber(normalized, options.huber_delta * 2.0);
        result.samples += 1;
    }
    result.cost *= track_authority_weight(track, options);
    result
}

fn epipolar_track_contribution(
    track: &Track,
    cameras: &[ResolvedCamera],
    reference_index: usize,
    options: &RigRefinementOptions,
) -> TrackObjectiveContribution {
    let Some(reference) = track
        .observations
        .iter()
        .find(|observation| observation.camera == reference_index)
    else {
        return TrackObjectiveContribution::default();
    };
    let reference_ray = cameras[reference_index].pixel_to_ray(reference.pixel);
    let mut result = TrackObjectiveContribution::default();
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
        let normalised_baseline = math::scale(baseline, 1.0 / baseline_length);
        let angular_error = dot(
            normalised_baseline,
            cross(reference_ray.direction, target_ray.direction),
        )
        .abs();
        let focal = (cameras[reference_index].focal_px * target_camera.focal_px)
            .abs()
            .sqrt();
        let pair_sigma =
            (observation_sigma(reference).powi(2) + observation_sigma(observation).powi(2)).sqrt();
        let normalized = angular_error * focal / pair_sigma.max(1.0e-6);
        result.cost += huber(normalized, options.huber_delta);
        if let Some((reference_depth, target_depth)) = pair_signed_depths(reference_ray, target_ray)
        {
            result.cost += cheirality_penalty(reference_depth, baseline_length);
            result.cost += cheirality_penalty(target_depth, baseline_length);
        } else {
            result.cost += 25.0;
        }
        result.samples += 1;
    }
    result.cost *= track_authority_weight(track, options);
    result
}

struct RigTrialEvaluation {
    parameter: usize,
    value: f64,
    cameras: Vec<(usize, ResolvedCamera)>,
    changed_tracks: Vec<(usize, TrackObjectiveContribution)>,
    total_cost: f64,
    total_samples: usize,
    prior_sum: f64,
    objective: f64,
}

struct IncrementalRigObjective<'a> {
    parameters: Vec<f64>,
    specs: &'a [ParameterSpec],
    tracks: &'a [&'a Track],
    options: &'a RigRefinementOptions,
    mode: IncrementalObjectiveMode,
    templates: Vec<ResolvedCameraTemplate>,
    cameras: Vec<ResolvedCamera>,
    tracks_by_camera: Vec<Vec<usize>>,
    contributions: Vec<TrackObjectiveContribution>,
    total_cost: f64,
    total_samples: usize,
    prior_sum: f64,
}

impl<'a> IncrementalRigObjective<'a> {
    fn new(
        parameters: Vec<f64>,
        specs: &'a [ParameterSpec],
        inputs: &'a [RigCameraInput<'a>],
        tracks: &'a [&'a Track],
        intrinsics_mode: IntrinsicsMode,
        options: &'a RigRefinementOptions,
        mode: IncrementalObjectiveMode,
    ) -> Option<Self> {
        let refinements = refinements_from_parameters(inputs.len(), &parameters, specs);
        let templates = inputs
            .iter()
            .map(|input| {
                ResolvedCameraTemplate::new(input.calibration?, input.state?, intrinsics_mode).ok()
            })
            .collect::<Option<Vec<_>>>()?;
        let cameras = templates
            .iter()
            .zip(&refinements)
            .map(|(template, refinement)| template.resolve(refinement).ok())
            .collect::<Option<Vec<_>>>()?;
        let mut tracks_by_camera = vec![Vec::new(); inputs.len()];
        for (track_index, track) in tracks.iter().enumerate() {
            for observation in &track.observations {
                let entries = &mut tracks_by_camera[observation.camera];
                if entries.last().copied() != Some(track_index) && !entries.contains(&track_index) {
                    entries.push(track_index);
                }
            }
        }
        let mut ray_scratch = Vec::new();
        let mut contributions = Vec::with_capacity(tracks.len());
        let mut total_cost = 0.0;
        let mut total_samples = 0usize;
        for track in tracks {
            let contribution = match mode {
                IncrementalObjectiveMode::Bundle => {
                    bundle_track_contribution(track, &cameras, options, &mut ray_scratch)
                }
                IncrementalObjectiveMode::Bearing { reference_index } => {
                    bearing_track_contribution(track, &cameras, reference_index, options)
                }
                IncrementalObjectiveMode::Epipolar { reference_index } => {
                    epipolar_track_contribution(track, &cameras, reference_index, options)
                }
            };
            total_cost += contribution.cost;
            total_samples += contribution.samples;
            contributions.push(contribution);
        }
        let prior_sum = parameters
            .iter()
            .zip(specs)
            .map(|(&value, spec)| (value / spec.prior_sigma).powi(2))
            .sum();
        Some(Self {
            parameters,
            specs,
            tracks,
            options,
            mode,
            templates,
            cameras,
            tracks_by_camera,
            contributions,
            total_cost,
            total_samples,
            prior_sum,
        })
    }

    #[inline]
    fn objective_from_parts(&self, cost: f64, samples: usize, prior_sum: f64) -> f64 {
        cost / samples.max(1) as f64 + self.options.factory_prior_weight * prior_sum
    }

    fn current_objective(&self) -> f64 {
        self.objective_from_parts(self.total_cost, self.total_samples, self.prior_sum)
    }

    fn evaluate_trial(&mut self, parameter: usize, value: f64) -> RigTrialEvaluation {
        self.evaluate_trial_readonly(parameter, value)
    }

    fn evaluate_trial_readonly(&self, parameter: usize, value: f64) -> RigTrialEvaluation {
        let spec = self.specs[parameter];
        let old_value = self.parameters[parameter];
        let old_prior = (old_value / spec.prior_sigma).powi(2);
        let new_prior = (value / spec.prior_sigma).powi(2);
        let prior_sum = self.prior_sum - old_prior + new_prior;
        if value == old_value {
            return RigTrialEvaluation {
                parameter,
                value,
                cameras: Vec::new(),
                changed_tracks: Vec::new(),
                total_cost: self.total_cost,
                total_samples: self.total_samples,
                prior_sum,
                objective: self.objective_from_parts(
                    self.total_cost,
                    self.total_samples,
                    prior_sum,
                ),
            };
        }
        let mut trial_parameters = self.parameters.clone();
        trial_parameters[parameter] = value;
        let refinements =
            refinements_from_parameters(self.cameras.len(), &trial_parameters, self.specs);
        let mut cameras = self.cameras.clone();
        let mut cameras_for_commit = Vec::new();
        for camera in 0..cameras.len() {
            if spec.affected_cameras & (1u16 << camera) == 0 {
                continue;
            }
            let Ok(trial_camera) = self.templates[camera].resolve(&refinements[camera]) else {
                return self.invalid_trial(parameter, value, prior_sum);
            };
            cameras[camera] = trial_camera.clone();
            cameras_for_commit.push((camera, trial_camera));
        }
        let mut total_cost = self.total_cost;
        let mut total_samples = self.total_samples;
        let mut affected = Vec::new();
        for camera in 0..self.cameras.len() {
            if spec.affected_cameras & (1u16 << camera) != 0 {
                affected.extend(self.tracks_by_camera[camera].iter().copied());
            }
        }
        affected.sort_unstable();
        affected.dedup();
        let mut changed_tracks = Vec::with_capacity(affected.len());
        let mut ray_scratch = Vec::new();
        for track_index in affected {
            let old = self.contributions[track_index];
            let track = self.tracks[track_index];
            let new = match self.mode {
                IncrementalObjectiveMode::Bundle => {
                    bundle_track_contribution(track, &cameras, self.options, &mut ray_scratch)
                }
                IncrementalObjectiveMode::Bearing { reference_index } => {
                    bearing_track_contribution(track, &cameras, reference_index, self.options)
                }
                IncrementalObjectiveMode::Epipolar { reference_index } => {
                    epipolar_track_contribution(track, &cameras, reference_index, self.options)
                }
            };
            total_cost += new.cost - old.cost;
            total_samples = total_samples - old.samples + new.samples;
            changed_tracks.push((track_index, new));
        }
        let objective = self.objective_from_parts(total_cost, total_samples, prior_sum);
        RigTrialEvaluation {
            parameter,
            value,
            cameras: cameras_for_commit,
            changed_tracks,
            total_cost,
            total_samples,
            prior_sum,
            objective,
        }
    }

    fn invalid_trial(&self, parameter: usize, value: f64, prior_sum: f64) -> RigTrialEvaluation {
        RigTrialEvaluation {
            parameter,
            value,
            cameras: Vec::new(),
            changed_tracks: Vec::new(),
            total_cost: f64::INFINITY,
            total_samples: self.total_samples,
            prior_sum,
            objective: f64::INFINITY,
        }
    }

    fn commit(&mut self, trial: RigTrialEvaluation) {
        if trial.value == self.parameters[trial.parameter] {
            return;
        }
        if trial.cameras.is_empty() {
            return;
        }
        self.parameters[trial.parameter] = trial.value;
        for (camera_index, camera) in trial.cameras {
            self.cameras[camera_index] = camera;
        }
        for (track_index, contribution) in trial.changed_tracks {
            self.contributions[track_index] = contribution;
        }
        self.total_cost = trial.total_cost;
        self.total_samples = trial.total_samples;
        self.prior_sum = trial.prior_sum;
    }
}

fn camera_track_support(tracks: &[&Track], camera_count: usize) -> Vec<usize> {
    let mut support = vec![0usize; camera_count];
    for track in tracks {
        let mut seen = Vec::<usize>::new();
        for observation in &track.observations {
            if !seen.contains(&observation.camera) {
                seen.push(observation.camera);
                support[observation.camera] += 1;
            }
        }
    }
    support
}

fn bearing_trust_region_specs(
    specs: &[ParameterSpec],
    tracks: &[&Track],
    camera_count: usize,
) -> Vec<ParameterSpec> {
    let support = camera_track_support(tracks, camera_count);
    let mut trusted = specs.to_vec();
    for spec in &mut trusted {
        if !matches!(
            spec.kind,
            ParameterKind::Orientation(_) | ParameterKind::Mirror
        ) {
            continue;
        }
        // The infinity/bearing bootstrap carries most of the observable signal
        // in this capture.  Finite-depth passes may polish it, but the trust
        // region shrinks when only a few independent scene landmarks observe
        // the camera. This is parameter-level regularisation, not a camera
        // measurement weight.
        let trust = if support[spec.camera] < 48 {
            0.05
        } else if support[spec.camera] < 96 {
            0.10
        } else {
            0.20
        };
        spec.maximum_update = spec.maximum_update.min(trust);
    }
    trusted
}

/// Block-coordinate bundle refinement used after the low-parallax bearing
/// initializer.  Bearing and nuisance DOFs are deliberately not allowed to
/// chase each other in the same sweep.  Weakly supported cameras still
/// participate in the joint solve, but finite-depth bundle adjustment is only
/// allowed to *polish* their already-estimated bearing; it cannot replace it
/// with a different sparse-depth branch.
fn staged_bundle_optimize_rig<'a>(
    parameters: Vec<f64>,
    specs: &'a [ParameterSpec],
    max_iterations: usize,
    inputs: &'a [RigCameraInput<'a>],
    tracks: &'a [&'a Track],
    intrinsics_mode: IntrinsicsMode,
    options: &'a RigRefinementOptions,
) -> (Vec<f64>, f64, usize) {
    if specs.is_empty() {
        return (parameters, f64::INFINITY, 0);
    }
    // Stage A: preserve the robust infinity/bearing solution. Bundle depth may
    // polish it, but sparse C-camera geometry gets a much smaller trust region.
    let mut bearing_specs = bearing_trust_region_specs(specs, tracks, inputs.len());
    for spec in &mut bearing_specs {
        if !matches!(
            spec.kind,
            ParameterKind::Orientation(_) | ParameterKind::Mirror
        ) {
            spec.maximum_update = 0.0;
        }
    }
    let sweeps = max_iterations.clamp(1, 2);
    let (parameters, _, bearing_iterations) = coordinate_optimize_rig(
        parameters,
        &bearing_specs,
        sweeps,
        inputs,
        tracks,
        intrinsics_mode,
        options,
        IncrementalObjectiveMode::Bundle,
    );

    // Stage B1: solve image-plane nuisance directions first while holding both
    // bearing and metric centre fixed. Releasing centre at the same time would
    // let weak depth leverage trade baseline against lens shape.
    let mut intrinsic_specs = specs.to_vec();
    for spec in &mut intrinsic_specs {
        if matches!(
            spec.kind,
            ParameterKind::Orientation(_)
                | ParameterKind::Mirror
                | ParameterKind::Center(_)
                | ParameterKind::FocusPupilScale(_)
        ) {
            spec.maximum_update = 0.0;
        }
    }
    let (parameters, _, intrinsic_iterations) = coordinate_optimize_rig(
        parameters,
        &intrinsic_specs,
        sweeps,
        inputs,
        tracks,
        intrinsics_mode,
        options,
        IncrementalObjectiveMode::Bundle,
    );

    // Stage B2: expose only independently observable optical-centre axes.
    // Strong factory priors and small coordinate steps make this a baseline
    // polish rather than a second unconstrained structure-from-motion solve.
    let mut center_specs = specs.to_vec();
    for spec in &mut center_specs {
        if matches!(
            spec.kind,
            ParameterKind::Center(_) | ParameterKind::FocusPupilScale(_)
        ) {
            spec.maximum_update = spec.maximum_update.min(0.20);
        } else {
            spec.maximum_update = 0.0;
        }
    }
    let (parameters, _, center_iterations) = coordinate_optimize_rig(
        parameters,
        &center_specs,
        sweeps,
        inputs,
        tracks,
        intrinsics_mode,
        options,
        IncrementalObjectiveMode::Bundle,
    );

    // Stage B3: one small joint nuisance polish after centre refinement.
    let mut nuisance_polish_specs = specs.to_vec();
    for spec in &mut nuisance_polish_specs {
        if matches!(
            spec.kind,
            ParameterKind::Orientation(_) | ParameterKind::Mirror
        ) {
            spec.maximum_update = 0.0;
        } else if matches!(
            spec.kind,
            ParameterKind::Center(_) | ParameterKind::FocusPupilScale(_)
        ) {
            spec.maximum_update = spec.maximum_update.min(0.10);
        } else if matches!(spec.kind, ParameterKind::DistortionCenter(_)) {
            spec.maximum_update = spec.maximum_update.min(0.35);
        } else {
            spec.maximum_update *= 0.5;
        }
    }
    let (parameters, _, nuisance_polish_iterations) = coordinate_optimize_rig(
        parameters,
        &nuisance_polish_specs,
        1,
        inputs,
        tracks,
        intrinsics_mode,
        options,
        IncrementalObjectiveMode::Bundle,
    );

    // Stage C: one tiny bearing polish with nuisance parameters frozen. This
    // lets the common scene skeleton absorb the consequence of the nuisance
    // update without reopening a different low-parallax solution branch.
    let mut polish_specs = specs.to_vec();
    for spec in &mut polish_specs {
        if matches!(
            spec.kind,
            ParameterKind::Orientation(_) | ParameterKind::Mirror
        ) {
            spec.maximum_update = spec.maximum_update.min(0.04);
        } else {
            spec.maximum_update = 0.0;
        }
    }
    let (parameters, objective, polish_iterations) = coordinate_optimize_rig(
        parameters,
        &polish_specs,
        1,
        inputs,
        tracks,
        intrinsics_mode,
        options,
        IncrementalObjectiveMode::Bundle,
    );
    (
        parameters,
        objective,
        bearing_iterations
            .saturating_add(intrinsic_iterations)
            .saturating_add(center_iterations)
            .saturating_add(nuisance_polish_iterations)
            .saturating_add(polish_iterations),
    )
}

fn coordinate_optimize_rig<'a>(
    parameters: Vec<f64>,
    specs: &'a [ParameterSpec],
    max_iterations: usize,
    inputs: &'a [RigCameraInput<'a>],
    tracks: &'a [&'a Track],
    intrinsics_mode: IntrinsicsMode,
    options: &'a RigRefinementOptions,
    mode: IncrementalObjectiveMode,
) -> (Vec<f64>, f64, usize) {
    let Some(mut objective) = IncrementalRigObjective::new(
        parameters.clone(),
        specs,
        inputs,
        tracks,
        intrinsics_mode,
        options,
        mode,
    ) else {
        return (parameters, f64::INFINITY, 0);
    };
    let mut current_objective = objective.current_objective();
    let mut iterations = 0;
    for iteration in 0..max_iterations {
        let sweep_before = current_objective;
        for parameter in 0..objective.parameters.len() {
            let spec = specs[parameter];
            // The epipolar bootstrap exists to recover a potentially large
            // bearing error before finite-depth bundle adjustment. Centre and
            // raster shifts are weakly/degenerately observed in distant L16
            // scenes and previously consumed the rotation signal (the same
            // failure was visible in the synthetic small-rotation test).
            if matches!(
                mode,
                IncrementalObjectiveMode::Bearing { .. }
                    | IncrementalObjectiveMode::Epipolar { .. }
            ) && !matches!(
                spec.kind,
                ParameterKind::Orientation(_) | ParameterKind::Mirror
            ) {
                continue;
            }
            let centre = objective.parameters[parameter];
            let step = spec.difference_step;
            let minus_value = (centre - step).max(-spec.bound);
            let plus_value = (centre + step).min(spec.bound);
            let (minus, plus) = if objective.tracks_by_camera[spec.camera].len() >= 64 {
                let shared = &objective;
                thread::scope(|scope| {
                    let minus_handle =
                        scope.spawn(|| shared.evaluate_trial_readonly(parameter, minus_value));
                    let plus = shared.evaluate_trial_readonly(parameter, plus_value);
                    let minus = minus_handle
                        .join()
                        .expect("rig minus finite-difference worker panicked");
                    (minus, plus)
                })
            } else {
                (
                    objective.evaluate_trial(parameter, minus_value),
                    objective.evaluate_trial(parameter, plus_value),
                )
            };
            let f_minus = minus.objective;
            let f_plus = plus.objective;
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
            let candidate = objective.evaluate_trial(parameter, candidate_value);
            if candidate.objective < current_objective {
                current_objective = candidate.objective;
                objective.commit(candidate);
            } else if f_minus < current_objective || f_plus < current_objective {
                if f_minus <= f_plus {
                    current_objective = f_minus;
                    objective.commit(minus);
                } else {
                    current_objective = f_plus;
                    objective.commit(plus);
                }
            }
        }
        iterations = iteration + 1;
        if sweep_before - current_objective < 1.0e-6 {
            break;
        }
    }
    (objective.parameters, current_objective, iterations)
}

fn pair_signed_depths(
    first: crate::geometry::Ray,
    second: crate::geometry::Ray,
) -> Option<(f64, f64)> {
    let cosine = dot(first.direction, second.direction);
    let denominator = 1.0 - cosine * cosine;
    if denominator <= 1.0e-14 {
        return None;
    }
    let origins = sub(first.origin, second.origin);
    let first_origin = dot(first.direction, origins);
    let second_origin = dot(second.direction, origins);
    let first_depth = (cosine * second_origin - first_origin) / denominator;
    let second_depth = (second_origin - cosine * first_origin) / denominator;
    Some((first_depth, second_depth))
}

fn pair_has_positive_depth(first: crate::geometry::Ray, second: crate::geometry::Ray) -> bool {
    pair_signed_depths(first, second)
        .is_some_and(|(first_depth, second_depth)| first_depth > 0.0 && second_depth > 0.0)
}

fn triangulation_has_positive_depth_with_rays(rays: &[crate::geometry::Ray], point: Vec3) -> bool {
    rays.iter()
        .all(|ray| dot(sub(point, ray.origin), ray.direction) > 0.0)
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
    let mut rays = Vec::new();
    for track in tracks {
        let contribution = bundle_track_contribution(track, &cameras, options, &mut rays);
        cost += contribution.cost;
        samples += contribution.samples;
    }
    let data = cost / samples.max(1) as f64;
    let prior = parameters
        .iter()
        .zip(specs)
        .map(|(&value, spec)| (value / spec.prior_sigma).powi(2))
        .sum::<f64>();
    data + options.factory_prior_weight * prior
}

fn push_validation_failure(
    evaluation: &mut Evaluation,
    observation: &TrackObservation,
    retain_residuals: bool,
) {
    let residual = [INVALID_REPROJECTION_PENALTY_PX, 0.0];
    evaluation.sum_squared += dot2(residual, residual);
    evaluation.samples += 1;
    if retain_residuals {
        evaluation.residuals.push(ResidualSample {
            camera: observation.camera,
            pixel: observation.pixel,
            residual,
            reference_equivalent_pixels: INVALID_REPROJECTION_PENALTY_PX
                / observation.local_scale.clamp(0.25, 4.0),
            angular_degrees: 90.0,
            inverse_depth: f64::NAN,
        });
    }
}

/// Independent camera-model validation. Each observation is predicted from a
/// 3-D point triangulated *without that observation*. The old evaluator fitted
/// the validation point to every camera and then scored those same cameras; a
/// single bad correspondence therefore pulled the common point and created
/// large residuals on several otherwise-correct cameras. Leave-one-out
/// prediction localises the failure to the camera/label that disagrees with the
/// rest of the frozen track and is a much closer test of the production use
/// case: predict where an independently reconstructed scene point lands.
fn evaluate_validation_leave_one_out(
    tracks: &[&Track],
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
    retain_residuals: bool,
) -> Evaluation {
    let mut evaluation = Evaluation::default();
    let mut all_rays = Vec::new();

    for track in tracks {
        evaluation.tracks += 1;

        // Keep the existing positive-depth statistic track-based so it remains
        // directly comparable with fit evaluation. It is deliberately
        // independent of whether one leave-one-out prediction later fails.
        if let Some(triangulated) =
            triangulate_with_rays(&track.observations, cameras, options, &mut all_rays)
        {
            if triangulation_has_positive_depth_with_rays(&all_rays, triangulated.point) {
                evaluation.positive_depth_tracks += 1;
            }
        }

        for held_out_index in 0..track.observations.len() {
            let observation = &track.observations[held_out_index];
            let other_observations = track
                .observations
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != held_out_index)
                .map(|(_, observation)| observation.clone())
                .collect::<Vec<_>>();
            if other_observations.len() < 2 {
                push_validation_failure(&mut evaluation, observation, retain_residuals);
                continue;
            }
            let Some(triangulated) = triangulate(&other_observations, cameras, options) else {
                push_validation_failure(&mut evaluation, observation, retain_residuals);
                continue;
            };
            let Some(projected) = cameras[observation.camera]
                .project(triangulated.point)
                .filter(|pixel| pixel.iter().all(|value| value.is_finite()))
                .filter(|&pixel| cameras[observation.camera].contains(pixel))
            else {
                push_validation_failure(&mut evaluation, observation, retain_residuals);
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
                    inverse_depth: {
                        let ray = cameras[observation.camera].pixel_to_ray(observation.pixel);
                        let depth = dot(sub(triangulated.point, ray.origin), ray.direction);
                        if depth.is_finite() && depth > 1.0e-9 {
                            1.0 / depth
                        } else {
                            f64::NAN
                        }
                    },
                });
            }
        }
    }
    evaluation
}

fn evaluate(
    tracks: &[&Track],
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
    retain_residuals: bool,
) -> Evaluation {
    let mut evaluation = Evaluation::default();
    let mut rays = Vec::new();
    for track in tracks {
        // Count the complete frozen population. A track that cannot be
        // triangulated under a candidate is a failed positive-depth case, not
        // something validation is allowed to silently remove.
        evaluation.tracks += 1;
        let Some(triangulated) =
            triangulate_with_rays(&track.observations, cameras, options, &mut rays)
        else {
            // Keep the validation sample population fixed even when the
            // candidate rig cannot triangulate a frozen track. Otherwise a
            // solver can improve its reported RMS simply by making difficult
            // tracks disappear. Each missing observation is an explicit
            // finite failure, capped so it cannot numerically dominate the
            // entire independent population.
            for observation in &track.observations {
                let residual = [INVALID_REPROJECTION_PENALTY_PX, 0.0];
                evaluation.sum_squared += dot2(residual, residual);
                evaluation.samples += 1;
                if retain_residuals {
                    evaluation.residuals.push(ResidualSample {
                        camera: observation.camera,
                        pixel: observation.pixel,
                        residual,
                        reference_equivalent_pixels: INVALID_REPROJECTION_PENALTY_PX
                            / observation.local_scale.clamp(0.25, 4.0),
                        angular_degrees: 90.0,
                        inverse_depth: f64::NAN,
                    });
                }
            }
            continue;
        };
        if triangulation_has_positive_depth_with_rays(&rays, triangulated.point) {
            evaluation.positive_depth_tracks += 1;
        }
        for (observation, &ray) in track.observations.iter().zip(&rays) {
            let Some(projected) =
                project_observation_with_ray(&cameras[observation.camera], ray, triangulated.point)
            else {
                // Keep physically invalid projections in the frozen validation
                // denominator as an explicit finite failure.  Silently dropping
                // them biases RMS downward and hides field-of-view mistakes.
                let residual = [INVALID_REPROJECTION_PENALTY_PX, 0.0];
                evaluation.sum_squared += dot2(residual, residual);
                evaluation.samples += 1;
                if retain_residuals {
                    evaluation.residuals.push(ResidualSample {
                        camera: observation.camera,
                        pixel: observation.pixel,
                        residual,
                        reference_equivalent_pixels: INVALID_REPROJECTION_PENALTY_PX
                            / observation.local_scale.clamp(0.25, 4.0),
                        angular_degrees: 90.0,
                        inverse_depth: f64::NAN,
                    });
                }
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
                let observed_direction = ray.direction;
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
                    inverse_depth: {
                        let ray = cameras[observation.camera].pixel_to_ray(observation.pixel);
                        let depth = dot(sub(triangulated.point, ray.origin), ray.direction);
                        if depth.is_finite() && depth > 1.0e-9 {
                            1.0 / depth
                        } else {
                            f64::NAN
                        }
                    },
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

#[derive(Clone, Copy)]
struct TriangulationNormalTerm {
    normal: Mat3,
    rhs: Vec3,
}

fn triangulation_normal_term(
    observation: &TrackObservation,
    ray: crate::geometry::Ray,
) -> TriangulationNormalTerm {
    let sigma = observation_sigma(observation);
    let inverse_variance = 1.0 / (sigma * sigma);
    let projector: Mat3 = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let identity = f64::from(row == column);
            identity - ray.direction[row] * ray.direction[column]
        })
    });
    TriangulationNormalTerm {
        normal: projector.map(|row| row.map(|value| value * inverse_variance)),
        rhs: projector.map(|row| dot(row, ray.origin) * inverse_variance),
    }
}

#[inline]
fn add_triangulation_term(
    normal: &mut Mat3,
    rhs: &mut Vec3,
    term: TriangulationNormalTerm,
    scale: f64,
) {
    for row in 0..3 {
        rhs[row] += term.rhs[row] * scale;
        for column in 0..3 {
            normal[row][column] += term.normal[row][column] * scale;
        }
    }
}

fn triangulate_normal_system(
    rays: &[crate::geometry::Ray],
    normal: Mat3,
    rhs: Vec3,
    options: &RigRefinementOptions,
) -> Option<Triangulated> {
    if rays.len() < 2 {
        return None;
    }
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

fn fill_observation_rays(
    observations: &[TrackObservation],
    cameras: &[ResolvedCamera],
    rays: &mut Vec<crate::geometry::Ray>,
) {
    rays.clear();
    if rays.capacity() < observations.len() {
        rays.reserve(observations.len() - rays.capacity());
    }
    rays.extend(
        observations
            .iter()
            .map(|observation| cameras[observation.camera].pixel_to_ray(observation.pixel)),
    );
}

fn triangulate_with_rays(
    observations: &[TrackObservation],
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
    rays: &mut Vec<crate::geometry::Ray>,
) -> Option<Triangulated> {
    fill_observation_rays(observations, cameras, rays);
    triangulate_precomputed(observations, rays, options)
}

fn triangulate_precomputed(
    observations: &[TrackObservation],
    rays: &[crate::geometry::Ray],
    options: &RigRefinementOptions,
) -> Option<Triangulated> {
    if observations.len() < 2 || rays.len() != observations.len() {
        return None;
    }
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for (observation, &ray) in observations.iter().zip(rays) {
        let term = triangulation_normal_term(observation, ray);
        add_triangulation_term(
            &mut normal,
            &mut rhs,
            term,
            observation_balance(observation, observations.len()),
        );
    }
    triangulate_normal_system(rays, normal, rhs, options)
}

fn triangulate(
    observations: &[TrackObservation],
    cameras: &[ResolvedCamera],
    options: &RigRefinementOptions,
) -> Option<Triangulated> {
    let mut rays = Vec::with_capacity(observations.len());
    triangulate_with_rays(observations, cameras, options, &mut rays)
}

/// Reproject the triangulated *line* intersection along the observed ray's
/// forward half-line. Factory angular errors can put the least-squares line
/// intersection behind a camera before refinement; flipping that camera's
/// line direction supplies a continuous calibration residual without treating
/// the non-physical point as valid scene depth.
fn project_observation_with_ray(
    camera: &ResolvedCamera,
    ray: crate::geometry::Ray,
    point: Vec3,
) -> Option<Vec2> {
    let displacement = sub(point, ray.origin);
    let forward_point = if dot(displacement, ray.direction) >= 0.0 {
        point
    } else {
        sub(ray.origin, displacement)
    };
    // Never evaluate the Brown model arbitrarily far outside its calibrated
    // field.  The old unbounded projection made a single bad line intersection
    // explode to 1e12-1e15 pixel residuals and, paradoxically, made almost any
    // candidate look like a 100% validation improvement.
    camera
        .project(forward_point)
        .filter(|pixel| pixel.iter().all(|value| value.is_finite()))
        .filter(|&pixel| camera.contains(pixel))
}

fn project_observation(
    camera: &ResolvedCamera,
    observation: &TrackObservation,
    point: Vec3,
) -> Option<Vec2> {
    let ray = camera.pixel_to_ray(observation.pixel);
    project_observation_with_ray(camera, ray, point)
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

fn observation_prepared(observation: &TrackObservation) -> &ObservationPrepared {
    observation.prepared.get_or_init(|| {
        let score = observation.confidence.clamp(0.0, 1.0);
        let structure_support = (observation.structure / 0.08).clamp(0.25, 1.0);
        let depth_support = observation
            .depth_reliability
            .unwrap_or(1.0)
            .clamp(0.25, 1.0);
        let scale = observation.local_scale.clamp(0.5, 3.0);
        let sigma = ((0.35 + 1.65 * (1.0 - score)) / (structure_support * depth_support).sqrt()
            * scale.sqrt())
        .clamp(0.30, 3.0);
        let covariance = observation.localization_covariance;
        let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
        let inverse_covariance = if determinant.is_finite() && determinant > 1.0e-9 {
            [
                [
                    covariance[1][1] / determinant,
                    -covariance[0][1] / determinant,
                ],
                [
                    -covariance[1][0] / determinant,
                    covariance[0][0] / determinant,
                ],
            ]
        } else {
            [[1.0, 0.0], [0.0, 1.0]]
        };
        ObservationPrepared {
            sigma,
            inverse_covariance,
        }
    })
}

fn observation_sigma(observation: &TrackObservation) -> f64 {
    observation_prepared(observation).sigma
}

fn normalized_observation_residual(observation: &TrackObservation, residual: Vec2) -> f64 {
    let prepared = observation_prepared(observation);
    let inverse = prepared.inverse_covariance;
    let squared = residual[0] * (inverse[0][0] * residual[0] + inverse[0][1] * residual[1])
        + residual[1] * (inverse[1][0] * residual[0] + inverse[1][1] * residual[1]);
    squared.max(0.0).sqrt() / prepared.sigma.max(1.0e-6)
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

fn symmetric_eigenvalues(matrix: Mat3) -> [f64; 3] {
    // Closed-form eigenvalues for a real symmetric 3x3 matrix (Kopp/Smith
    // formulation). Triangulation only needs the spectrum, not eigenvectors.
    let p1 =
        matrix[0][1] * matrix[0][1] + matrix[0][2] * matrix[0][2] + matrix[1][2] * matrix[1][2];
    if p1 <= 1.0e-28 {
        return [matrix[0][0], matrix[1][1], matrix[2][2]];
    }
    let q = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / 3.0;
    let a00 = matrix[0][0] - q;
    let a11 = matrix[1][1] - q;
    let a22 = matrix[2][2] - q;
    let p2 = a00 * a00 + a11 * a11 + a22 * a22 + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if !p.is_finite() || p <= 1.0e-18 {
        return [q, q, q];
    }
    let inv_p = 1.0 / p;
    let b = [
        [a00 * inv_p, matrix[0][1] * inv_p, matrix[0][2] * inv_p],
        [matrix[1][0] * inv_p, a11 * inv_p, matrix[1][2] * inv_p],
        [matrix[2][0] * inv_p, matrix[2][1] * inv_p, a22 * inv_p],
    ];
    let determinant = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let r = (determinant * 0.5).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let two_p = 2.0 * p;
    let largest = q + two_p * phi.cos();
    let smallest = q + two_p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    let middle = 3.0 * q - largest - smallest;
    [largest, middle, smallest]
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
            let mut validation_after_magnitudes = validation_after
                .residuals
                .iter()
                .filter(|sample| sample.camera == camera)
                .map(|sample| dot2(sample.residual, sample.residual).sqrt())
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            validation_after_magnitudes.sort_by(f64::total_cmp);
            RigCameraResidualReport {
                camera: input.name.to_owned(),
                fit_samples,
                fit_rms_before: rms(fit_before_sum, fit_samples),
                fit_rms_after: rms(fit_after_sum, fit_samples),
                validation_samples,
                validation_rms_before: rms(validation_before_sum, validation_samples),
                validation_rms_after: rms(validation_after_sum, validation_samples),
                validation_median_after: percentile(&validation_after_magnitudes, 0.50),
                validation_p90_after: percentile(&validation_after_magnitudes, 0.90),
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

fn pearson_correlation(pairs: &[(f64, f64)]) -> f64 {
    let pairs = pairs
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    if pairs.len() < 3 {
        return f64::NAN;
    }
    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    let denominator = (variance_x * variance_y).sqrt();
    if !denominator.is_finite() || denominator <= 1.0e-18 {
        f64::NAN
    } else {
        covariance / denominator
    }
}

fn residual_correlation_reports(
    cameras: &[RigCameraInput<'_>],
    samples: &[ResidualSample],
) -> Vec<RigResidualCorrelationReport> {
    cameras
        .iter()
        .enumerate()
        .filter_map(|(camera, input)| {
            let state = input.state?;
            let width = state.width.max(2) as f64 - 1.0;
            let height = state.height.max(2) as f64 - 1.0;
            let mut dx_x = Vec::new();
            let mut dx_y = Vec::new();
            let mut dy_x = Vec::new();
            let mut dy_y = Vec::new();
            let mut dx_r2 = Vec::new();
            let mut dy_r2 = Vec::new();
            let mut dx_inverse_depth = Vec::new();
            let mut dy_inverse_depth = Vec::new();
            let mut count = 0usize;
            for sample in samples.iter().filter(|sample| sample.camera == camera) {
                if !sample.pixel.iter().all(|value| value.is_finite())
                    || !sample.residual.iter().all(|value| value.is_finite())
                {
                    continue;
                }
                // Normalise position to roughly [-1,+1] so the diagnostic is
                // comparable between the 75 mm and 150 mm sensor footprints.
                let x = 2.0 * sample.pixel[0] / width - 1.0;
                let y = 2.0 * sample.pixel[1] / height - 1.0;
                let r2 = x * x + y * y;
                let dx = sample.residual[0];
                let dy = sample.residual[1];
                dx_x.push((dx, x));
                dx_y.push((dx, y));
                dy_x.push((dy, x));
                dy_y.push((dy, y));
                dx_r2.push((dx, r2));
                dy_r2.push((dy, r2));
                count += 1;
                if sample.inverse_depth.is_finite() && sample.inverse_depth > 0.0 {
                    dx_inverse_depth.push((dx, sample.inverse_depth));
                    dy_inverse_depth.push((dy, sample.inverse_depth));
                }
            }
            (count > 0).then(|| RigResidualCorrelationReport {
                camera: input.name.to_owned(),
                samples: count,
                inverse_depth_samples: dx_inverse_depth.len(),
                dx_vs_x: pearson_correlation(&dx_x),
                dx_vs_y: pearson_correlation(&dx_y),
                dy_vs_x: pearson_correlation(&dy_x),
                dy_vs_y: pearson_correlation(&dy_y),
                dx_vs_r2: pearson_correlation(&dx_r2),
                dy_vs_r2: pearson_correlation(&dy_r2),
                dx_vs_inverse_depth: pearson_correlation(&dx_inverse_depth),
                dy_vs_inverse_depth: pearson_correlation(&dy_inverse_depth),
            })
        })
        .collect()
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
                factory_inverse_depth: factory.inverse_depth,
                candidate_inverse_depth: candidate.inverse_depth,
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
            optimized: specs
                .iter()
                .any(|spec| spec.affected_cameras & (1u16 << camera) != 0),
            orientation_offset_degrees: refinement.orientation_offset_degrees.unwrap_or([0.0; 3]),
            mirror_angle_offset_degrees: refinement.mirror_angle_offset_degrees,
            center_offset_world: refinement.center_offset_world.unwrap_or([0.0; 3]),
            focus_pupil_scale: refinement.focus_pupil_scale.unwrap_or(0.0),
            sensor_offset_px: refinement.sensor_offset_px.unwrap_or([0.0; 2]),
            focal_scale_delta: refinement.focal_scale_delta.unwrap_or(0.0),
            focal_aspect_delta: refinement.focal_aspect_delta.unwrap_or(0.0),
            distortion_center_offset_px: refinement.distortion_center_offset_px.unwrap_or([0.0; 2]),
            distortion_delta: refinement.distortion_delta.unwrap_or([0.0; 4]),
            reached_bound: parameters.iter().zip(specs).any(|(&value, spec)| {
                spec.affected_cameras & (1u16 << camera) != 0 && value.abs() >= spec.bound * 0.98
            }),
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

fn validation_block_key(track: &Track, block_size_px: usize) -> [i32; 2] {
    // Track keys use sixteenth-pixel reference coordinates. Keeping this in a
    // helper makes the ordinary hash split and LatentGraph's balanced split
    // use exactly the same spatial semantics.
    let block_units = block_size_px.max(1).saturating_mul(16) as i32;
    [
        track.key[0].div_euclid(block_units),
        track.key[1].div_euclid(block_units),
    ]
}

fn is_validation_track(
    track: &Track,
    validation_modulus: u64,
    options: &RigRefinementOptions,
) -> bool {
    // Hash the spatial block rather than the individual track so correlated
    // samples from the same edge/facade cannot appear in both fit and
    // validation populations.
    let block = validation_block_key(track, options.validation_block_size_px);
    stable_track_hash(block).is_multiple_of(validation_modulus.max(2))
}

#[derive(Clone, Debug)]
struct LatentValidationBlock {
    key: [i32; 2],
    eligible_tracks: usize,
    camera_observations: Vec<usize>,
    hash: u64,
}

/// Choose LatentGraph validation blocks with approximately the requested
/// fraction *per camera*, rather than relying on one global reference-space
/// hash. Narrow-overlap modules otherwise get pathological splits (the real
/// capture put ~72% of C1's usable observations in validation while C3 got
/// ~3%), leaving the optimizer almost no evidence for one camera and almost no
/// independent evidence for another. The selection is still image-only and
/// frozen before any candidate rig is fitted.
fn latent_validation_blocks(
    tracks: &[Track],
    candidates: &LatentCandidateState,
    camera_count: usize,
    reference_index: usize,
    options: &RigRefinementOptions,
) -> HashSet<[i32; 2]> {
    if tracks.is_empty() || camera_count == 0 {
        return HashSet::new();
    }

    // Latent validation needs a little more granularity than the generic
    // 256-px block split because the 150-mm C cameras see only a narrow part of
    // the 75-mm reference. 128 px is still much larger than one feature patch
    // and therefore keeps local correlated evidence together.
    let block_size_px = options.validation_block_size_px.div_ceil(2).max(64);
    let mut by_block = HashMap::<[i32; 2], LatentValidationBlock>::new();
    let mut total_camera_observations = vec![0usize; camera_count];
    let mut total_eligible_tracks = 0usize;

    for track in tracks {
        let Some((validation_track, _)) = candidates
            .validation_track(track, options.latent_validation_max_alternative_similarity)
        else {
            continue;
        };
        let key = validation_block_key(track, block_size_px);
        let block = by_block
            .entry(key)
            .or_insert_with(|| LatentValidationBlock {
                key,
                eligible_tracks: 0,
                camera_observations: vec![0; camera_count],
                hash: stable_track_hash(key),
            });
        block.eligible_tracks += 1;
        total_eligible_tracks += 1;
        for observation in &validation_track.observations {
            if observation.camera == reference_index || observation.camera >= camera_count {
                continue;
            }
            block.camera_observations[observation.camera] += 1;
            total_camera_observations[observation.camera] += 1;
        }
    }

    if by_block.is_empty() {
        return HashSet::new();
    }

    let fraction = options.validation_fraction.clamp(0.05, 0.5);
    let minimum_fit_camera = (options.min_camera_observations / 2).max(16);
    let mut camera_targets = vec![0usize; camera_count];
    for camera in 0..camera_count {
        if camera == reference_index {
            continue;
        }
        let total = total_camera_observations[camera];
        if total == 0 {
            continue;
        }
        let desired = ((total as f64 * fraction).round() as usize)
            .max(options.min_validation_camera_samples.min(total));
        let maximum = total.saturating_sub(minimum_fit_camera);
        camera_targets[camera] = desired.min(maximum);
    }
    let desired_tracks = ((total_eligible_tracks as f64 * fraction).round() as usize)
        .max(options.min_validation_tracks.min(total_eligible_tracks))
        .min(total_eligible_tracks.saturating_sub(options.min_tracks));

    let mut blocks = by_block.into_values().collect::<Vec<_>>();
    blocks.sort_by_key(|block| block.hash);
    let mut selected = HashSet::<[i32; 2]>::new();
    let mut selected_camera = vec![0usize; camera_count];
    let mut selected_tracks = 0usize;

    // Squared normalized deficit is intentionally dominated by rare cameras:
    // adding ten C1 samples matters much more than adding ten of thousands of
    // B-camera samples. A small track-count term keeps the global fraction near
    // the requested target after camera quotas are satisfied.
    let selection_objective = |camera_counts: &[usize], track_count: usize| -> f64 {
        let mut objective = 0.0;
        for camera in 0..camera_count {
            let target = camera_targets[camera];
            if target == 0 {
                continue;
            }
            let error = camera_counts[camera] as f64 - target as f64;
            objective += (error / target.max(1) as f64).powi(2);
        }
        if desired_tracks > 0 {
            let error = track_count as f64 - desired_tracks as f64;
            objective += 0.35 * (error / desired_tracks as f64).powi(2);
        }
        objective
    };

    loop {
        let current_objective = selection_objective(&selected_camera, selected_tracks);
        let mut best: Option<(usize, f64)> = None;
        for (index, block) in blocks.iter().enumerate() {
            if selected.contains(&block.key) {
                continue;
            }
            let mut trial_camera = selected_camera.clone();
            for camera in 0..camera_count {
                trial_camera[camera] += block.camera_observations[camera];
            }
            let trial_tracks = selected_tracks + block.eligible_tracks;
            let trial_objective = selection_objective(&trial_camera, trial_tracks);
            let gain = current_objective - trial_objective;
            if gain > 1.0e-12
                && best.map_or(true, |(best_index, best_gain)| {
                    gain > best_gain + 1.0e-12
                        || ((gain - best_gain).abs() <= 1.0e-12
                            && block.hash < blocks[best_index].hash)
                })
            {
                best = Some((index, gain));
            }
        }
        let Some((index, _)) = best else {
            break;
        };
        let block = &blocks[index];
        selected.insert(block.key);
        selected_tracks += block.eligible_tracks;
        for camera in 0..camera_count {
            selected_camera[camera] += block.camera_observations[camera];
        }
    }

    // The least-squares objective can stop just short of a hard minimum when
    // one block overshoots another camera's soft target. Fill any remaining
    // per-camera/track deficits deterministically, choosing the block with the
    // largest normalized deficit coverage.
    loop {
        let camera_deficit = (0..camera_count).any(|camera| {
            camera_targets[camera] > 0 && selected_camera[camera] < camera_targets[camera]
        });
        let track_deficit = desired_tracks > 0 && selected_tracks < desired_tracks;
        if !camera_deficit && !track_deficit {
            break;
        }
        let mut best: Option<(usize, f64)> = None;
        for (index, block) in blocks.iter().enumerate() {
            if selected.contains(&block.key) {
                continue;
            }
            let mut score = 0.0;
            for camera in 0..camera_count {
                let target = camera_targets[camera];
                if target == 0 || selected_camera[camera] >= target {
                    continue;
                }
                let deficit = target - selected_camera[camera];
                score +=
                    block.camera_observations[camera].min(deficit) as f64 / target.max(1) as f64;
            }
            if track_deficit {
                score += 0.10 * block.eligible_tracks.min(desired_tracks - selected_tracks) as f64
                    / desired_tracks.max(1) as f64;
            }
            if score > 0.0
                && best.map_or(true, |(best_index, best_score)| {
                    score > best_score + 1.0e-12
                        || ((score - best_score).abs() <= 1.0e-12
                            && block.hash < blocks[best_index].hash)
                })
            {
                best = Some((index, score));
            }
        }
        let Some((index, _)) = best else {
            break;
        };
        let block = &blocks[index];
        selected.insert(block.key);
        selected_tracks += block.eligible_tracks;
        for camera in 0..camera_count {
            selected_camera[camera] += block.camera_observations[camera];
        }
    }

    selected
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
        p99: percentile(&values, 0.99),
    }
}

fn residual_sample_rms(samples: &[ResidualSample]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    (samples
        .iter()
        .map(|sample| dot2(sample.residual, sample.residual))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

fn trimmed_residual_rms(samples: &[ResidualSample], keep_fraction: f64) -> f64 {
    let mut squared = samples
        .iter()
        .map(|sample| dot2(sample.residual, sample.residual))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if squared.is_empty() {
        return f64::NAN;
    }
    squared.sort_by(f64::total_cmp);
    let keep = ((squared.len() as f64 * keep_fraction.clamp(0.0, 1.0)).floor() as usize)
        .clamp(1, squared.len());
    (squared[..keep].iter().sum::<f64>() / keep as f64).sqrt()
}

fn residuals_over_px(samples: &[ResidualSample], threshold: f64) -> usize {
    let squared_threshold = threshold * threshold;
    samples
        .iter()
        .filter(|sample| dot2(sample.residual, sample.residual) > squared_threshold)
        .count()
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

    #[test]
    fn latent_graph_selects_a_solved_candidate_without_factory_fallback() {
        assert!(select_solved_candidate(
            RigRefinementStrategy::LatentGraph,
            false,
        ));
        assert!(!select_solved_candidate(
            RigRefinementStrategy::Physical,
            false,
        ));
        assert!(!select_solved_candidate(
            RigRefinementStrategy::AnchorGraph,
            false,
        ));
    }

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
                            reference_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                            target_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                            peak_margin: 0.5,
                            forward_backward_error_px: 0.0,
                            depth_reliability: None,
                        });
                }
            }
        }
        alignments
    }

    #[test]
    fn global_observability_detects_multi_parameter_rank_deficiency() {
        let first = vec![1.0, 0.0, 1.0, 0.0];
        let second = vec![0.0, 1.0, 0.0, 1.0];
        let dependent = vec![1.0, 1.0, 1.0, 1.0];
        let (first_novelty, first_direction) = orthogonal_column_novelty(&first, &[]);
        assert!(first_novelty > 0.99);
        let first_direction = first_direction.expect("first basis direction");
        let (second_novelty, second_direction) =
            orthogonal_column_novelty(&second, std::slice::from_ref(&first_direction));
        assert!(second_novelty > 0.99);
        let second_direction = second_direction.expect("second basis direction");
        let (dependent_novelty, _) =
            orthogonal_column_novelty(&dependent, &[first_direction, second_direction]);
        assert!(
            dependent_novelty < 1.0e-9,
            "linear combination should add no independent camera information: {dependent_novelty}"
        );
    }

    #[test]
    fn legacy_fallback_salvages_rejected_homography_with_epipolar_consensus() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
        ];
        let states = [state("B1"), state("B2"), state("B3")];
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
        let mut alignments = synthetic_alignments(&resolved);
        // A single homography is the wrong model for a finite-depth scene.
        // Simulate the stage-2 image-space gate rejecting this camera while
        // preserving a perfectly coherent epipolar correspondence population.
        alignments[1].report.accepted = false;
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

        let (tracks, matches) = build_tracks(&inputs, 0, &alignments, &resolved);
        assert_eq!(matches, 240);
        assert_eq!(
            tracks
                .iter()
                .filter(|track| track
                    .observations
                    .iter()
                    .any(|observation| observation.camera == 1))
                .count(),
            120
        );
    }

    #[test]
    fn sparse_feature_fallback_keeps_cross_camera_matches_pairwise_until_cycle_verified() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
        ];
        let states = [state("B1"), state("B2"), state("B3")];
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
        let alignments = synthetic_alignments(&resolved);
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

        let (tracks, matches) = build_tracks(&inputs, 0, &alignments, &resolved);
        assert_eq!(matches, 240);
        assert_eq!(tracks.len(), 240);
        assert!(tracks.iter().all(|track| track.observations.len() == 2));
        assert_eq!(
            tracks
                .iter()
                .filter(|track| track.observations[1].camera == 1)
                .count(),
            120
        );
        assert_eq!(
            tracks
                .iter()
                .filter(|track| track.observations[1].camera == 2)
                .count(),
            120
        );
    }

    #[test]
    fn rejected_narrow_view_rotation_consensus_rejects_large_along_line_aliases() {
        let mut calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("C1", [80.0, 0.0, 0.0]),
        ];
        // Model a narrow-FOV target while keeping the scene sufficiently far
        // that true rays are close to a single relative rotation.
        calibrations[1].intrinsics[0].k[0][0] = 2200.0;
        calibrations[1].intrinsics[0].k[1][1] = 2200.0;
        let states = [state("B1"), state("C1")];
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
        let mut correspondences = Vec::new();
        for row in 0..10 {
            for column in 0..12 {
                let point = [
                    (column as f64 - 5.5) * 800.0,
                    (row as f64 - 4.5) * 700.0,
                    100_000.0 + ((row * 17 + column * 31) % 11) as f64 * 2_000.0,
                ];
                let reference_pixel = resolved[0].project(point).unwrap();
                let target_pixel = resolved[1].project(point).unwrap();
                correspondences.push(AlignmentCorrespondence {
                    reference_pixel,
                    target_pixel,
                    confidence: 0.95,
                    local_scale: 2.2,
                    structure: 0.1,
                    reference_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    target_localization_covariance: [[1.0, 0.0], [0.0, 1.0]],
                    peak_margin: 0.5,
                    forward_backward_error_px: 0.0,
                    depth_reliability: None,
                });
            }
        }
        let true_count = correspondences.len();
        for index in 0..40 {
            let mut alias = correspondences[index].clone();
            alias.target_pixel[0] += 120.0 + (index % 5) as f64 * 8.0;
            correspondences.push(alias);
        }
        let inliers = fallback_rotation_inliers(&correspondences, &resolved[0], &resolved[1]);
        assert!(
            inliers.len() >= 100,
            "retained only {} true-bearing candidates",
            inliers.len()
        );
        let injected = inliers.iter().filter(|&&index| index >= true_count).count();
        assert!(
            injected <= 2,
            "rotation consensus retained {injected} injected aliases"
        );
    }

    #[test]
    fn fallback_epipolar_filter_rejects_repeated_structure_outliers() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
            calibration("B3", [0.0, 70.0, 0.0]),
        ];
        let states = [state("B1"), state("B2"), state("B3")];
        let truth = calibrations
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
        let mut alignments = synthetic_alignments(&truth);
        let correspondences = &mut alignments[1].correspondences;
        let genuine = correspondences.len();

        // Duplicate valid reference locations but send them to a different,
        // repeated-looking target location. A photometric/homography fallback
        // can admit these; one epipolar model cannot.
        let outliers = correspondences
            .iter()
            .take(30)
            .copied()
            .enumerate()
            .map(|(index, mut correspondence)| {
                correspondence.target_pixel[0] += 90.0 + (index % 5) as f64 * 13.0;
                correspondence.target_pixel[1] -= 70.0 + (index % 7) as f64 * 11.0;
                correspondence
            })
            .collect::<Vec<_>>();
        correspondences.extend(outliers);

        let inliers = fallback_epipolar_inliers(correspondences, &truth[0], &truth[1]);
        assert!(
            inliers.len() >= genuine * 9 / 10,
            "too many genuine correspondences rejected: {}/{}",
            inliers.len(),
            genuine
        );
        let retained_injected = inliers.iter().filter(|&&index| index >= genuine).count();
        assert!(
            retained_injected <= 3,
            "epipolar fallback retained {retained_injected} injected outliers"
        );
    }

    #[test]
    fn fallback_epipolar_filter_verifies_small_sparse_populations() {
        let calibrations = [
            calibration("B1", [0.0, 0.0, 0.0]),
            calibration("B2", [80.0, 0.0, 0.0]),
        ];
        let states = [state("B1"), state("B2")];
        let truth = calibrations
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
        let mut alignments = synthetic_alignments(&truth);
        // Keep this small-set test spatially well conditioned.  The previous
        // `truncate(18)` selected the first 18 points emitted by
        // `synthetic_alignments`: all 12 points from row 0 plus only the first
        // six from row 1.  That near-two-scanline configuration is a critical
        // / nearly-degenerate sample for an 8-point epipolar fit, so an
        // alternative rank-2 model can legitimately pass through all 12 true
        // correspondences *and* several injected points.  The test was then
        // asserting that RANSAC must distinguish geometry that the synthetic
        // sample itself does not uniquely distinguish.
        //
        // Sample the same 18/120 population across the full image instead.
        // This keeps the test focused on what it is named for: verifying that
        // the small-population RANSAC path actually runs and rejects gross
        // outliers when the epipolar model is observable.
        let all_correspondences = std::mem::take(&mut alignments[1].correspondences);
        alignments[1].correspondences = all_correspondences
            .into_iter()
            .step_by(7)
            .take(18)
            .collect();
        let correspondences = &mut alignments[1].correspondences;
        let genuine = 12usize;
        for (index, correspondence) in correspondences.iter_mut().enumerate().skip(genuine) {
            correspondence.target_pixel[0] += 120.0 + (index % 3) as f64 * 17.0;
            correspondence.target_pixel[1] -= 90.0 + (index % 4) as f64 * 19.0;
        }
        let inliers = fallback_epipolar_inliers(correspondences, &truth[0], &truth[1]);
        assert!(
            inliers.len() >= 10 && inliers.len() <= 14,
            "small-set epipolar verification retained {} of 18 matches",
            inliers.len(),
        );
        let retained_injected = inliers.iter().filter(|&&index| index >= genuine).count();
        assert!(
            retained_injected <= 2,
            "small-set epipolar verification retained {retained_injected} injected outliers",
        );
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
            // This fixture contains pure orientation perturbations. Keeping
            // nuisance translations/raster shifts disabled makes the test
            // identify the parameter it claims to test instead of relying on
            // arbitrary default regularisation to break a finite-depth gauge.
            max_center_offset: 0.0,
            max_sensor_offset_px: 0.0,
            max_mirror_degrees: 0.0,
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
            // Some 150-mm L16 modules share only a narrow overlap with the
            // 75-mm reference and may have no third camera seeing the same
            // patch. Two-view tracks are still valid epipolar/depth evidence;
            // the candidate ranking already rewards extra independent views
            // whenever they exist.
            physical_match_min_views: 2,
            ..Default::default()
        };
        let view = |camera: usize, score: f32| PhysicalViewMatch {
            camera,
            score,
            target_pixel: [100.0 + camera as f64, 80.0],
            residual: [0.0, 0.0],
            residual_proposal: [0.0, 0.0],
            epipolar_tangent: None,
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
                prepared: Default::default(),
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
            // Some 150-mm L16 modules share only a narrow overlap with the
            // 75-mm reference and may have no third camera seeing the same
            // patch. Two-view tracks are still valid epipolar/depth evidence;
            // the candidate ranking already rewards extra independent views
            // whenever they exist.
            physical_match_min_views: 2,
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
                    prepared: Default::default(),
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
            // Some 150-mm L16 modules share only a narrow overlap with the
            // 75-mm reference and may have no third camera seeing the same
            // patch. Two-view tracks are still valid epipolar/depth evidence;
            // the candidate ranking already rewards extra independent views
            // whenever they exist.
            physical_match_min_views: 2,
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
            prepared: Default::default(),
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

    #[test]
    fn focus_pupil_parameter_is_shared_only_with_its_affected_group() {
        let specs = [ParameterSpec {
            camera: 0,
            affected_cameras: 0b0101,
            kind: ParameterKind::FocusPupilScale('B'),
            bound: 1.0,
            prior_sigma: 0.5,
            difference_step: 0.025,
            maximum_update: 0.15,
        }];
        let refinements = refinements_from_parameters(3, &[0.25], &specs);
        assert_eq!(refinements[0].focus_pupil_scale, Some(0.25));
        assert_eq!(refinements[1].focus_pupil_scale, None);
        assert_eq!(refinements[2].focus_pupil_scale, Some(0.25));
    }
}
