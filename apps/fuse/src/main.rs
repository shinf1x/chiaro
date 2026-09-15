use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};

use chiaro_fusion::array_color::ColorProfileMode;
use chiaro_fusion::calibration::{IntrinsicsMode, MirrorAngleMode};
use chiaro_fusion::crosstalk::CrosstalkMode;
use chiaro_fusion::pipeline::{FusionOptions, HotpixelStage, fuse};
use chiaro_fusion::resolution::ResolutionReconstruction;
use chiaro_fusion::rig::RigRefinementStrategy;
use chiaro_fusion::synth::{CanvasMode, CropWindow, OutputColor};
use chiaro_hotpixel_core::demosaic::DemosaicMethod;
use chiaro_hotpixel_core::highlight::HighlightRecovery;
use chiaro_hotpixel_core::scan::mmap_file;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Color {
    Display,
    Linear,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FactoryProfile {
    CctOnly,
    ArrayAware,
    A,
    F11,
    D65,
}

impl From<FactoryProfile> for ColorProfileMode {
    fn from(value: FactoryProfile) -> Self {
        match value {
            FactoryProfile::ArrayAware => Self::ArrayAware,
            FactoryProfile::CctOnly => Self::CctOnly,
            FactoryProfile::A => Self::ForceA,
            FactoryProfile::F11 => Self::ForceF11,
            FactoryProfile::D65 => Self::ForceD65,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Intrinsics {
    LinearHall,
    Clamp,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MirrorAngleModel {
    CurrentQuadratic,
    CalibrationQuadraticInverse,
    CalibrationPairsLinear,
}

impl From<MirrorAngleModel> for MirrorAngleMode {
    fn from(value: MirrorAngleModel) -> Self {
        match value {
            MirrorAngleModel::CurrentQuadratic => Self::CurrentQuadratic,
            MirrorAngleModel::CalibrationQuadraticInverse => Self::CalibrationQuadraticInverse,
            MirrorAngleModel::CalibrationPairsLinear => Self::CalibrationPairsLinear,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RigStrategy {
    Physical,
    AnchorGraph,
    LatentGraph,
}

impl From<RigStrategy> for RigRefinementStrategy {
    fn from(value: RigStrategy) -> Self {
        match value {
            RigStrategy::Physical => Self::Physical,
            RigStrategy::AnchorGraph => Self::AnchorGraph,
            RigStrategy::LatentGraph => Self::LatentGraph,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Demosaic {
    Simple,
    Amaze,
    Rcd,
    Lmmse,
    Igv,
}

impl From<Demosaic> for DemosaicMethod {
    fn from(value: Demosaic) -> Self {
        match value {
            Demosaic::Simple => Self::Simple,
            Demosaic::Amaze => Self::Amaze,
            Demosaic::Rcd => Self::Rcd,
            Demosaic::Lmmse => Self::Lmmse,
            Demosaic::Igv => Self::Igv,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RawHighlightRecovery {
    None,
    LocalBayer,
    MultiscaleBayer,
    MultiCamera,
}

impl From<RawHighlightRecovery> for HighlightRecovery {
    fn from(value: RawHighlightRecovery) -> Self {
        match value {
            RawHighlightRecovery::None => Self::None,
            RawHighlightRecovery::LocalBayer => Self::LocalBayer,
            RawHighlightRecovery::MultiscaleBayer => Self::MultiscaleBayer,
            RawHighlightRecovery::MultiCamera => Self::MultiCamera,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "chiaro-fuse",
    version,
    about = "Align the cameras of a Light L16 capture and synthesise one high-resolution frame"
)]
struct Cli {
    /// Capture to fuse.
    input: PathBuf,

    /// Output PNG (16-bit RGB). A `.fusion.json` report is written beside it.
    #[arg(long, short)]
    output: PathBuf,

    /// Factory hotpixel.rec; enables the hot-pixel stage.
    #[arg(long)]
    hotpixel_rec: Option<PathBuf>,

    /// Camera-specific learned defect/line profile. Requires the exact
    /// hotpixel.rec against which the profile was trained.
    #[arg(long, value_name = "CAMERA.chiaro-cleanup")]
    cleanup_profile: Option<PathBuf>,

    /// Device calibration overlays (calibration.lri, zoom_calib_v0.lri). Repeatable.
    #[arg(long = "calibration", value_name = "FILE")]
    overlays: Vec<PathBuf>,

    /// Reference module (default: the capture's reference camera).
    #[arg(long)]
    reference: Option<String>,

    /// Use only these modules; repeatable.
    #[arg(long)]
    camera: Vec<String>,

    /// Exclude a module from reconstruction but retain it for experimental
    /// held-out physical-CFA validation; repeatable.
    #[arg(long)]
    cfa_held_out: Vec<String>,

    /// Canvas size: `native` (13 MP), `max` (as the finest covering module
    /// allows, capped by --max-megapixels), or a number of canvas pixels per
    /// reference pixel.
    #[arg(long, default_value = "max")]
    canvas: String,

    /// Cap for `--canvas max`.
    #[arg(long, default_value_t = 82.0)]
    max_megapixels: f32,

    /// Render the full wide frame instead of cropping to the framed focal length.
    #[arg(long)]
    no_crop: bool,

    /// Explicit reference-raster crop as x,y,width,height. Intended for
    /// reproducible matched diagnostic crops.
    #[arg(long, value_name = "X,Y,W,H", conflicts_with = "no_crop")]
    crop: Option<String>,

    #[arg(long, value_enum, default_value = "display")]
    color: Color,

    /// Use the original D65 factory profile, or select an experimental colour
    /// profile mode.
    #[arg(long, value_enum, default_value = "d65")]
    factory_profile: FactoryProfile,

    /// Bayer reconstruction method.
    #[arg(long, value_enum, default_value = "amaze")]
    demosaic: Demosaic,

    /// Reconstruct clipped Bayer samples before crosstalk and demosaicing.
    #[arg(long, value_enum, default_value = "multi-camera")]
    highlight_recovery: RawHighlightRecovery,

    /// Apply no crosstalk, the factory mesh, or a capture-adaptive residual.
    #[arg(long, default_value = "adaptive")]
    crosstalk: CrosstalkMode,

    /// Resample cameras independently or reconstruct from their physical samples.
    #[arg(long, default_value = "joint-cfa")]
    resolution_reconstruction: ResolutionReconstruction,

    /// Diagnose flat-region Joint-CFA support by running solves whose output
    /// is guaranteed to retain the baseline.
    #[arg(long)]
    joint_cfa_solve_flat: bool,

    /// Leave monochrome modules out of the synthesis (they contribute luminance).
    #[arg(long)]
    exclude_mono: bool,

    /// Disable smooth clipped-highlight reconstruction for downstream processing.
    #[arg(long)]
    no_highlight_correction: bool,

    /// Keep the factory geometry without correlation refinement (diagnostics).
    #[arg(long)]
    no_refine: bool,

    /// Disable capture-specific physical rig refinement while retaining the
    /// existing residual image-space alignment.
    #[arg(long)]
    no_rig_refine: bool,

    /// Rig correspondence strategy. `physical` keeps the V10 matcher;
    /// `anchor-graph` grows constellation-validated tracks before fitting;
    /// `latent-graph` keeps several repeated-structure hypotheses and lets
    /// them switch between bundle passes.
    #[arg(long, value_enum, default_value = "physical")]
    rig_strategy: RigStrategy,

    /// Maximum anchor-graph propagation rounds after bootstrap.
    #[arg(long)]
    rig_anchor_rounds: Option<usize>,

    /// Minimum factory overlap, relative to the smaller field, for a camera
    /// pair to be eligible as a direct anchor-graph edge.
    #[arg(long)]
    rig_anchor_min_overlap: Option<f64>,

    /// Direct camera edges activated before propagation starts.
    #[arg(long)]
    rig_anchor_initial_edges: Option<usize>,

    /// Hard ceiling on direct camera edges after adaptive graph growth.
    #[arg(long)]
    rig_anchor_max_edges: Option<usize>,

    /// Desired minimum degree per camera in the active graph.
    #[arg(long)]
    rig_anchor_min_degree: Option<usize>,

    /// Maximum camera edges activated after any one propagation round.
    #[arg(long)]
    rig_anchor_edges_per_round: Option<usize>,

    /// Early-stop threshold for new validated observations per round.
    #[arg(long)]
    rig_anchor_min_observation_growth: Option<f64>,

    /// Early-stop threshold for new cycle-supported 3+ view tracks per round.
    #[arg(long)]
    rig_anchor_min_track_growth: Option<f64>,

    /// Native-sensor search radius around an anchor/constellation prediction.
    #[arg(long)]
    rig_anchor_search_radius_px: Option<f64>,

    /// Wide native-sensor radius used only to establish a newly activated
    /// camera edge directly from factory geometry.
    #[arg(long)]
    rig_anchor_direct_seed_radius_px: Option<f64>,

    /// Maximum response-sorted corners considered per camera when directly
    /// seeding a newly activated edge.
    #[arg(long)]
    rig_anchor_direct_seed_max_corners: Option<usize>,

    /// Maximum candidates retained for one latent track/camera observation.
    #[arg(long)]
    rig_latent_candidates: Option<usize>,

    /// Maximum latent correspondence-assignment rounds.
    #[arg(long)]
    rig_latent_rounds: Option<usize>,

    /// Minimum target-side self-similarity for a latent alternative.
    #[arg(long)]
    rig_latent_min_similarity: Option<f64>,

    /// Maximum difference from the initial candidate's signed factory
    /// epipolar residual, in target sensor pixels.
    #[arg(long)]
    rig_latent_epipolar_band_px: Option<f64>,

    /// Local constellation neighbours used when scoring candidate switches.
    #[arg(long)]
    rig_latent_neighbours: Option<usize>,

    /// Near-duplicate latent alternatives above this similarity are not used
    /// as frozen held-out labels.
    #[arg(long)]
    rig_latent_validation_max_similarity: Option<f64>,

    /// Minimum image-only seed confidence for a latent held-out observation.
    #[arg(long)]
    rig_latent_validation_min_confidence: Option<f64>,

    /// Minimum best-vs-runner-up image-match margin for a latent held-out label.
    #[arg(long)]
    rig_latent_validation_min_peak_margin: Option<f64>,

    /// Maximum forward/backward localization disagreement for a latent
    /// held-out label, in native target-sensor pixels.
    #[arg(long)]
    rig_latent_validation_max_forward_backward_px: Option<f64>,

    /// Maximum image-only local-constellation disagreement allowed for a
    /// latent held-out label, in reference-equivalent pixels.
    #[arg(long)]
    rig_latent_validation_max_constellation_error_px: Option<f64>,

    /// Minimum active fit observations protected per target camera by
    /// reversible latent membership.
    #[arg(long)]
    rig_latent_membership_min_camera_observations: Option<usize>,

    /// Minimum fraction of each camera's initial fit support protected by
    /// reversible latent membership.
    #[arg(long)]
    rig_latent_membership_min_camera_fraction: Option<f64>,

    /// Dormant observation reactivation threshold in reference-equivalent px.
    #[arg(long)]
    rig_latent_membership_recovery_reference_px: Option<f64>,

    /// Looser LOO gate used only when restoring a starved camera to its
    /// protected fit-support floor.
    #[arg(long)]
    rig_latent_membership_floor_max_reference_px: Option<f64>,

    /// Symmetric calibrated epipolar cutoff for intrinsically two-view latent
    /// tracks. Tracks above it become dormant as a whole.
    #[arg(long)]
    rig_latent_pairwise_max_reference_px: Option<f64>,

    /// Hysteretic recovery cutoff for a dormant two-view latent track.
    #[arg(long)]
    rig_latent_pairwise_recovery_reference_px: Option<f64>,

    /// Relative bundle-objective weight of intrinsically two-view tracks.
    #[arg(long)]
    rig_latent_pairwise_bundle_weight: Option<f64>,

    /// Disable bootstrap multi-camera cycle constraints inside LatentGraph.
    #[arg(long)]
    rig_latent_no_cycle_graph: bool,

    /// Maximum direct camera edges used by LatentGraph's image-only cycle graph.
    #[arg(long)]
    rig_latent_cycle_max_edges: Option<usize>,

    /// Minimum cycle-supported anchors required for a cross-camera pair field.
    #[arg(long)]
    rig_latent_cycle_min_anchors: Option<usize>,

    /// Weight of multi-camera cycle consistency in latent candidate scoring.
    #[arg(long)]
    rig_latent_cycle_weight: Option<f64>,

    /// Maximum cycle-prediction disagreement allowed for an unsupported switch.
    #[arg(long)]
    rig_latent_cycle_max_error_px: Option<f64>,

    /// Fit-only robust-membership cutoff in reference-equivalent pixels.
    #[arg(long)]
    rig_fit_membership_max_reference_px: Option<f64>,

    /// Absolute held-out rig RMS target in native target-sensor pixels.
    #[arg(long)]
    rig_validation_max_rms_px: Option<f64>,

    /// Maximum held-out p90 residual accepted for a physical rig.
    #[arg(long)]
    rig_validation_max_p90_px: Option<f64>,

    /// Maximum held-out p95 residual accepted for a physical rig.
    #[arg(long)]
    rig_validation_max_p95_px: Option<f64>,

    /// Minimum frozen held-out observations required for every optimized camera.
    #[arg(long)]
    rig_validation_min_camera_samples: Option<usize>,

    /// Maximum camera-local held-out RMS accepted for an optimized camera.
    #[arg(long)]
    rig_validation_max_camera_rms_px: Option<f64>,

    /// Maximum camera-local held-out p90 accepted for an optimized camera.
    #[arg(long)]
    rig_validation_max_camera_p90_px: Option<f64>,

    /// Maximum per-axis capture-specific camera orientation correction.
    #[arg(long)]
    rig_max_orientation_degrees: Option<f64>,

    /// Maximum additive movable-mirror angle correction.
    #[arg(long)]
    rig_max_mirror_degrees: Option<f64>,

    /// Maximum optical-centre correction per world axis, in calibration units.
    #[arg(long)]
    rig_max_center_offset: Option<f64>,

    /// Maximum shared B/C-group scale of CRA/Hall-implied focus travel along
    /// each physical camera optical axis. Zero (the default) disables it.
    #[arg(long)]
    rig_max_focus_pupil_scale: Option<f64>,

    /// Maximum calibration-raster origin correction per sensor axis.
    #[arg(long)]
    rig_max_sensor_offset_px: Option<f64>,

    /// Maximum LatentGraph isotropic focal correction, as percent from factory.
    #[arg(long)]
    rig_max_focal_scale_percent: Option<f64>,

    /// Maximum LatentGraph C-camera focal anisotropy, as percent. Positive
    /// values expand fx while contracting fy around the common focal scale.
    #[arg(long)]
    rig_max_focal_aspect_percent: Option<f64>,

    /// Maximum additive C-camera Brown k1 correction.
    #[arg(long)]
    rig_max_distortion_k1_delta: Option<f64>,

    /// Maximum additive C-camera Brown k2 correction.
    #[arg(long)]
    rig_max_distortion_k2_delta: Option<f64>,

    /// Maximum absolute additive C-camera Brown p1/p2 correction.
    #[arg(long)]
    rig_max_distortion_tangential_delta: Option<f64>,

    /// Maximum independent C-camera distortion-centre offset per sensor axis.
    #[arg(long)]
    rig_max_distortion_center_offset_px: Option<f64>,

    /// One-sigma scale of the factory prior for orientation corrections.
    #[arg(long)]
    rig_orientation_prior_sigma_degrees: Option<f64>,

    /// One-sigma scale of the factory prior for movable-mirror corrections.
    #[arg(long)]
    rig_mirror_prior_sigma_degrees: Option<f64>,

    /// One-sigma scale of the factory prior for optical-centre corrections.
    #[arg(long)]
    rig_center_prior_sigma: Option<f64>,

    /// One-sigma prior for the dimensionless shared focus-pupil scale.
    #[arg(long)]
    rig_focus_pupil_prior_sigma: Option<f64>,

    /// One-sigma scale of the factory prior for sensor-raster offsets.
    #[arg(long)]
    rig_sensor_prior_sigma_px: Option<f64>,

    /// One-sigma LatentGraph focal-scale prior, as percent from factory.
    #[arg(long)]
    rig_focal_scale_prior_percent: Option<f64>,

    /// One-sigma LatentGraph C-camera focal-anisotropy prior, as percent.
    #[arg(long)]
    rig_focal_aspect_prior_percent: Option<f64>,

    /// One-sigma prior for additive C-camera Brown k1 correction.
    #[arg(long)]
    rig_distortion_k1_prior: Option<f64>,

    /// One-sigma prior for additive C-camera Brown k2 correction.
    #[arg(long)]
    rig_distortion_k2_prior: Option<f64>,

    /// One-sigma prior for additive C-camera Brown p1/p2 correction.
    #[arg(long)]
    rig_distortion_tangential_prior: Option<f64>,

    /// One-sigma prior for C-camera distortion-centre offset, in sensor pixels.
    #[arg(long)]
    rig_distortion_center_prior_px: Option<f64>,

    /// Weight of the normalized quadratic factory-rig prior.
    #[arg(long)]
    rig_factory_prior_weight: Option<f64>,

    /// Finest native-sensor step used by physical-match localization.
    #[arg(long)]
    rig_match_subpixel_step_px: Option<f64>,

    /// Half-resolution ZNCC patch radius for physical matching.
    #[arg(long)]
    rig_match_patch_radius: Option<usize>,

    /// Minimum per-camera ZNCC score accepted by physical matching.
    #[arg(long)]
    rig_match_min_score: Option<f32>,

    /// Minimum relative separation between competing physical depth modes.
    #[arg(long)]
    rig_match_min_margin: Option<f32>,

    /// Pre-solve proposal-normalized reprojection gate, in reference pixels.
    #[arg(long)]
    rig_match_pre_solve_reprojection_px: Option<f64>,

    /// Disable calibrated local inverse-depth refinement and keep one global
    /// homography per module.
    #[arg(long)]
    no_depth: bool,

    /// Nearest depth considered by local parallax refinement, in millimetres.
    #[arg(long, default_value_t = 500.0)]
    depth_near: f64,

    /// Farthest finite depth considered by local parallax refinement, in millimetres.
    #[arg(long, default_value_t = 10_000_000.0)]
    depth_far: f64,

    /// Focus calibration outside the measured Hall range.
    #[arg(long, value_enum, default_value = "linear-hall")]
    intrinsics: Intrinsics,

    /// Hall-code to movable-mirror angle model. The default implements the
    /// factory quadratic's stored inverse-root semantics; the other modes are
    /// retained for controlled A/B comparisons and rollback.
    #[arg(long, value_enum, default_value = "calibration-quadratic-inverse")]
    mirror_angle_model: MirrorAngleModel,

    /// Disable the factory mirror-angle optical-center mapping candidate for
    /// A1->B and B4->C image alignment.
    #[arg(long)]
    no_angle_optical_center_prior: bool,

    /// Skip the factory flat-field (vignetting) correction.
    #[arg(long)]
    no_flat_field: bool,

    /// Write a visual trace of physical, measured, rig-candidate and final
    /// depth warps, plus confidence/visibility/ownership diagnostics.
    #[arg(long, value_name = "DIRECTORY")]
    debug_dir: Option<PathBuf>,

    /// Disable the bundled universal hot-pixel model.
    #[arg(long)]
    no_universal_hotpixel_model: bool,

    /// Disable the bundled corner-glow correction.
    #[arg(long)]
    no_glow_correction: bool,

    /// Worker threads (0 = all cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// PNG deflate level 0-9.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(0..=9))]
    png_level: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_cleanup_pair(&cli.cleanup_profile, &cli.hotpixel_rec)?;
    let lri = mmap_file(&cli.input)?;
    let mut options = FusionOptions {
        reference: cli.reference.clone(),
        overlays: cli.overlays.clone(),
        intrinsics_mode: match cli.intrinsics {
            Intrinsics::LinearHall => IntrinsicsMode::LinearHall,
            Intrinsics::Clamp => IntrinsicsMode::Clamp,
        },
        mirror_angle_mode: cli.mirror_angle_model.into(),
        angle_optical_center_prior: !cli.no_angle_optical_center_prior,
        hotpixel: cli.hotpixel_rec.clone().map(|rec| HotpixelStage {
            rec,
            universal_model: !cli.no_universal_hotpixel_model,
            glow_correction: !cli.no_glow_correction,
            cleanup_profile: cli.cleanup_profile.clone(),
        }),
        cameras: cli.camera.clone(),
        cfa_held_out: cli.cfa_held_out.clone(),
        threads: cli.threads,
        flat_field: !cli.no_flat_field,
        debug_dir: cli.debug_dir.clone(),
        ..FusionOptions::default()
    };
    options.align.refine = !cli.no_refine;
    options.rig_refinement.enabled = !cli.no_refine && !cli.no_rig_refine;
    options.rig_refinement.strategy = cli.rig_strategy.into();
    if let Some(value) = cli.rig_anchor_rounds {
        options.rig_refinement.anchor_max_rounds = value.min(12);
    }
    if let Some(value) = cli.rig_anchor_min_overlap {
        options.rig_refinement.anchor_min_factory_overlap = value.clamp(0.0, 1.0);
    }
    if let Some(value) = cli.rig_anchor_initial_edges {
        options.rig_refinement.anchor_initial_active_edges = value.max(1);
    }
    if let Some(value) = cli.rig_anchor_max_edges {
        options.rig_refinement.anchor_max_active_edges = value.max(1);
    }
    if let Some(value) = cli.rig_anchor_min_degree {
        options.rig_refinement.anchor_min_camera_degree = value;
    }
    if let Some(value) = cli.rig_anchor_edges_per_round {
        options.rig_refinement.anchor_edges_per_round = value;
    }
    if let Some(value) = cli.rig_anchor_min_observation_growth {
        options
            .rig_refinement
            .anchor_min_observation_growth_fraction = value.clamp(0.0, 1.0);
    }
    if let Some(value) = cli.rig_anchor_min_track_growth {
        options
            .rig_refinement
            .anchor_min_strong_track_growth_fraction = value.clamp(0.0, 1.0);
    }
    if let Some(value) = cli.rig_anchor_search_radius_px {
        options.rig_refinement.anchor_search_radius_px = value.max(1.0);
    }
    if let Some(value) = cli.rig_anchor_direct_seed_radius_px {
        options.rig_refinement.anchor_direct_seed_search_radius_px = value.max(4.0);
    }
    if let Some(value) = cli.rig_anchor_direct_seed_max_corners {
        options.rig_refinement.anchor_direct_seed_max_corners = value.max(64);
    }
    if let Some(value) = cli.rig_latent_candidates {
        options.rig_refinement.latent_max_candidates = value.clamp(1, 16);
    }
    if let Some(value) = cli.rig_latent_rounds {
        options.rig_refinement.latent_max_assignment_iterations = value.min(16);
    }
    if let Some(value) = cli.rig_latent_min_similarity {
        options.rig_refinement.latent_min_appearance_similarity = value.clamp(-1.0, 1.0);
    }
    if let Some(value) = cli.rig_latent_epipolar_band_px {
        options.rig_refinement.latent_candidate_epipolar_band_px = value.max(0.0);
    }
    if let Some(value) = cli.rig_latent_neighbours {
        let neighbours = value.max(3);
        options.rig_refinement.latent_neighbour_count = neighbours;
        options.rig_refinement.latent_cycle_neighbour_count = neighbours;
    }
    if let Some(value) = cli.rig_latent_validation_max_similarity {
        options
            .rig_refinement
            .latent_validation_max_alternative_similarity = value.clamp(-1.0, 1.0);
    }
    if let Some(value) = cli.rig_latent_validation_min_confidence {
        options.rig_refinement.latent_validation_min_confidence = value.clamp(0.0, 1.0);
    }
    if let Some(value) = cli.rig_latent_validation_min_peak_margin {
        options.rig_refinement.latent_validation_min_peak_margin = value.max(0.0);
    }
    if let Some(value) = cli.rig_latent_validation_max_forward_backward_px {
        options
            .rig_refinement
            .latent_validation_max_forward_backward_px = value.max(0.0);
    }
    if let Some(value) = cli.rig_latent_validation_max_constellation_error_px {
        options
            .rig_refinement
            .latent_validation_max_constellation_error_px = value.max(0.0);
    }
    if let Some(value) = cli.rig_latent_membership_min_camera_observations {
        options
            .rig_refinement
            .latent_membership_min_camera_observations = value.max(2);
    }
    if let Some(value) = cli.rig_latent_membership_min_camera_fraction {
        options.rig_refinement.latent_membership_min_camera_fraction = value.clamp(0.0, 1.0);
    }
    if let Some(value) = cli.rig_latent_membership_recovery_reference_px {
        options
            .rig_refinement
            .latent_membership_recovery_reference_px = value.max(0.1);
    }
    if let Some(value) = cli.rig_latent_membership_floor_max_reference_px {
        options
            .rig_refinement
            .latent_membership_floor_max_reference_px = value.max(0.1);
    }
    if let Some(value) = cli.rig_latent_pairwise_max_reference_px {
        options
            .rig_refinement
            .latent_pairwise_membership_max_reference_px = value.max(0.1);
    }
    if let Some(value) = cli.rig_latent_pairwise_recovery_reference_px {
        options
            .rig_refinement
            .latent_pairwise_membership_recovery_reference_px = value.max(0.05);
    }
    if let Some(value) = cli.rig_latent_pairwise_bundle_weight {
        options.rig_refinement.latent_pairwise_bundle_weight = value.clamp(0.0, 1.0);
    }
    if cli.rig_latent_no_cycle_graph {
        options.rig_refinement.latent_cycle_graph_enabled = false;
    }
    if let Some(value) = cli.rig_latent_cycle_max_edges {
        options.rig_refinement.latent_cycle_graph_max_edges = value;
    }
    if let Some(value) = cli.rig_latent_cycle_min_anchors {
        options.rig_refinement.latent_cycle_min_pair_anchors = value.max(3);
    }
    if let Some(value) = cli.rig_latent_cycle_weight {
        options.rig_refinement.latent_cycle_weight = value.clamp(0.0, 4.0);
    }
    if let Some(value) = cli.rig_latent_cycle_max_error_px {
        options.rig_refinement.latent_cycle_max_error_px = value.max(0.1);
    }
    if let Some(value) = cli.rig_fit_membership_max_reference_px {
        options.rig_refinement.fit_membership_max_reference_px = value.max(0.25);
    }
    if let Some(value) = cli.rig_validation_max_rms_px {
        options.rig_refinement.max_validation_rms_px = value.max(0.05);
    }
    if let Some(value) = cli.rig_validation_max_p90_px {
        options.rig_refinement.max_validation_p90_px = value.max(0.05);
    }
    if let Some(value) = cli.rig_validation_max_p95_px {
        options.rig_refinement.max_validation_p95_px = value.max(0.05);
    }
    if let Some(value) = cli.rig_validation_min_camera_samples {
        options.rig_refinement.min_validation_camera_samples = value.max(3);
    }
    if let Some(value) = cli.rig_validation_max_camera_rms_px {
        options.rig_refinement.max_validation_camera_rms_px = value.max(0.05);
    }
    if let Some(value) = cli.rig_validation_max_camera_p90_px {
        options.rig_refinement.max_validation_camera_p90_px = value.max(0.05);
    }
    // Keep RigRefinementOptions::default() as the single source of truth.
    // CLI values override it only when the user explicitly supplies them;
    // duplicating defaults here previously left stale bounds/priors active
    // even after the library defaults were tightened.
    if let Some(value) = cli.rig_max_orientation_degrees {
        options.rig_refinement.max_orientation_degrees = value.max(0.0);
    }
    if let Some(value) = cli.rig_max_mirror_degrees {
        options.rig_refinement.max_mirror_degrees = value.max(0.0);
    }
    if let Some(value) = cli.rig_max_center_offset {
        options.rig_refinement.max_center_offset = value.max(0.0);
    }
    if let Some(value) = cli.rig_max_focus_pupil_scale {
        options.rig_refinement.max_focus_pupil_scale = value.clamp(0.0, 4.0);
    }
    if let Some(value) = cli.rig_max_sensor_offset_px {
        options.rig_refinement.max_sensor_offset_px = value.max(0.0);
    }
    if let Some(value) = cli.rig_max_focal_scale_percent {
        options.rig_refinement.max_focal_scale_delta = (value.max(0.0) / 100.0).min(0.10);
    }
    if let Some(value) = cli.rig_max_focal_aspect_percent {
        options.rig_refinement.max_focal_aspect_delta = (value.max(0.0) / 100.0).min(0.05);
    }
    if let Some(value) = cli.rig_max_distortion_k1_delta {
        options.rig_refinement.max_distortion_k1_delta = value.max(0.0).min(0.25);
    }
    if let Some(value) = cli.rig_max_distortion_k2_delta {
        options.rig_refinement.max_distortion_k2_delta = value.max(0.0).min(0.50);
    }
    if let Some(value) = cli.rig_max_distortion_tangential_delta {
        options.rig_refinement.max_distortion_tangential_delta = value.max(0.0).min(0.05);
    }
    if let Some(value) = cli.rig_max_distortion_center_offset_px {
        options.rig_refinement.max_distortion_center_offset_px = value.clamp(0.0, 128.0);
    }
    if let Some(value) = cli.rig_orientation_prior_sigma_degrees {
        options.rig_refinement.orientation_prior_sigma_degrees = value.max(1.0e-6);
    }
    if let Some(value) = cli.rig_mirror_prior_sigma_degrees {
        options.rig_refinement.mirror_prior_sigma_degrees = value.max(1.0e-6);
    }
    if let Some(value) = cli.rig_center_prior_sigma {
        options.rig_refinement.center_prior_sigma = value.max(1.0e-6);
    }
    if let Some(value) = cli.rig_focus_pupil_prior_sigma {
        options.rig_refinement.focus_pupil_prior_sigma = value.max(1.0e-6);
    }
    if let Some(value) = cli.rig_sensor_prior_sigma_px {
        options.rig_refinement.sensor_offset_prior_sigma_px = value.max(1.0e-6);
    }
    if let Some(value) = cli.rig_focal_scale_prior_percent {
        options.rig_refinement.focal_scale_prior_sigma = (value.max(1.0e-6) / 100.0).min(0.10);
    }
    if let Some(value) = cli.rig_focal_aspect_prior_percent {
        options.rig_refinement.focal_aspect_prior_sigma = (value.max(1.0e-6) / 100.0).min(0.05);
    }
    if let Some(value) = cli.rig_distortion_k1_prior {
        options.rig_refinement.distortion_k1_prior_sigma = value.max(1.0e-8).min(0.25);
    }
    if let Some(value) = cli.rig_distortion_k2_prior {
        options.rig_refinement.distortion_k2_prior_sigma = value.max(1.0e-8).min(0.50);
    }
    if let Some(value) = cli.rig_distortion_tangential_prior {
        options.rig_refinement.distortion_tangential_prior_sigma = value.max(1.0e-8).min(0.05);
    }
    if let Some(value) = cli.rig_distortion_center_prior_px {
        options.rig_refinement.distortion_center_prior_sigma_px = value.clamp(1.0e-6, 64.0);
    }
    if let Some(value) = cli.rig_factory_prior_weight {
        options.rig_refinement.factory_prior_weight = value.max(0.0);
    }
    if let Some(value) = cli.rig_match_subpixel_step_px {
        options.rig_refinement.physical_match_subpixel_step_px = value.clamp(0.0625, 1.0);
    }
    if let Some(value) = cli.rig_match_patch_radius {
        options.rig_refinement.physical_match_patch_radius = value.max(1);
    }
    if let Some(value) = cli.rig_match_min_score {
        options.rig_refinement.physical_match_min_score = value.clamp(-1.0, 1.0);
    }
    if let Some(value) = cli.rig_match_min_margin {
        options.rig_refinement.physical_match_min_margin = value.max(0.0);
    }
    if let Some(value) = cli.rig_match_pre_solve_reprojection_px {
        options
            .rig_refinement
            .physical_match_max_reprojection_reference_px = value.max(0.1);
    }
    options.align.depth.enabled = !cli.no_depth;
    options.align.depth.near_depth = cli.depth_near;
    options.align.depth.far_depth = cli.depth_far;
    options.crop_to_framing = !cli.no_crop;
    options.crop = cli.crop.as_deref().map(parse_crop).transpose()?;
    options.synth.canvas =
        match cli.canvas.to_ascii_lowercase().as_str() {
            "native" => CanvasMode::Native,
            "max" | "maximum" => CanvasMode::Maximum {
                max_megapixels: cli.max_megapixels,
            },
            other => CanvasMode::Scale(other.parse::<f32>().with_context(|| {
                format!("--canvas must be native, max, or a number, not {other}")
            })?),
        };
    options.synth.include_mono = !cli.exclude_mono;
    options.synth.demosaic = cli.demosaic.into();
    options.synth.highlight_recovery = cli.highlight_recovery.into();
    options.crosstalk = cli.crosstalk;
    options.color_profile = cli.factory_profile.into();
    options.synth.resolution_reconstruction = cli.resolution_reconstruction;
    options.synth.joint_cfa_solve_flat = cli.joint_cfa_solve_flat;
    options.synth.highlight_correction = !cli.no_highlight_correction;
    options.synth.threads = cli.threads;
    options.synth.png_level = cli.png_level;
    options.synth.color = match cli.color {
        Color::Display => OutputColor::Display,
        Color::Linear => OutputColor::Linear,
    };
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let report = fuse(&lri, &options, &cli.output, &mut |progress| {
        println!(
            "[{:>3.0}%] {}: {}",
            progress.fraction * 100.0,
            progress.stage,
            progress.detail
        );
    })?;
    println!(
        "reference {} - {} calibrated modules - framed at {} - crop {:.0}x{:.0} @ {:.2}x -> canvas {}x{} ({:.1}% covered)",
        report.reference,
        report.calibration_modules,
        report
            .framed_focal_length_mm
            .map_or("unknown focal length".to_owned(), |f| format!("{f} mm")),
        report.synthesis.crop[2],
        report.synthesis.crop[3],
        report.synthesis.scale,
        report.synthesis.canvas_width,
        report.synthesis.canvas_height,
        report.synthesis.covered * 100.0,
    );
    let rig = &report.rig_refinement;
    let rig_strategy = match rig.strategy {
        RigRefinementStrategy::Physical => "physical",
        RigRefinementStrategy::AnchorGraph => "anchor-graph",
        RigRefinementStrategy::LatentGraph => "latent-graph",
    };
    if rig.validation_evaluated {
        println!(
            "rig ({rig_strategy}): {} - {} tracks ({} with 3+ cameras; {} fit/{} held out; {} ambiguous hold-out excluded), RMS {:.3}->{:.3} px, held-out {:.3}->{:.3} px ({:+.2}%){}",
            if rig.accepted && rig.validation_passed {
                "selected; validation passed"
            } else if rig.accepted {
                "selected; validation warning"
            } else {
                "factory retained"
            },
            rig.tracks,
            rig.tracks_three_plus,
            rig.fit_tracks,
            rig.validation_tracks,
            rig.validation_excluded_ambiguous_tracks,
            rig.reprojection_rms_before,
            rig.reprojection_rms_after,
            rig.held_out_rms_before,
            rig.held_out_rms_after,
            rig.held_out_relative_improvement * 100.0,
            rig.validation_warning
                .as_ref()
                .or(rig.fallback_reason.as_ref())
                .map_or(String::new(), |reason| format!("; {reason}")),
        );
    } else if rig.enabled && rig.accepted {
        println!(
            "rig ({rig_strategy}): accepted - {} all-track production fit tracks, RMS {:.3}->{:.3} px{}",
            rig.fit_tracks,
            rig.reprojection_rms_before,
            rig.reprojection_rms_after,
            rig.fallback_reason
                .as_ref()
                .map_or(String::new(), |reason| format!("; {reason}")),
        );
    } else if rig.enabled {
        println!(
            "rig ({rig_strategy}): factory retained - {} all-track production fit tracks{}",
            rig.fit_tracks,
            rig.fallback_reason
                .as_ref()
                .map_or(String::new(), |reason| format!("; {reason}")),
        );
    } else {
        println!("rig: not run (--no-rig-refine)");
    }
    if let Some(anchor) = &rig.anchor_graph {
        println!(
            "  anchor graph: {} bootstrap anchors, {} -> {} tracks, {} -> {} cycle-supported 3+ tracks; {} propagation rounds{}",
            anchor.bootstrap_anchors,
            anchor.initial_tracks,
            anchor.final_tracks,
            anchor.initial_cycle_supported_three_plus_tracks,
            anchor.final_cycle_supported_three_plus_tracks,
            anchor.rounds_run,
            if anchor.stopped_early {
                " (early stop)"
            } else {
                ""
            },
        );
        println!(
            "  anchor growth: {} new observations, {} promoted anchors, {} spawned tracks",
            anchor.propagated_observations, anchor.promoted_observations, anchor.spawned_tracks,
        );
        println!(
            "  camera graph: {} factory-overlap candidates (>= {:.0}%), {} -> {} active edges (max {}, target degree {})",
            anchor.candidate_edges,
            100.0 * anchor.factory_overlap_threshold,
            anchor.initial_active_edges,
            anchor.final_active_edges,
            anchor.maximum_active_edges,
            anchor.target_min_camera_degree,
        );
        for round in &anchor.rounds {
            println!(
                "    round {}: edges {} -> {} (+{}), +{} obs ({:.2}%), +{} promoted, +{} tracks, +{} cycle edges; strong 3+ {} -> {} ({:.2}%){}",
                round.round,
                round.active_edges_before,
                round.active_edges_after,
                round.activated_edges,
                round.new_observations,
                round.observation_growth_fraction * 100.0,
                round.promoted_observations,
                round.new_tracks,
                round.closed_cycle_edges,
                round.strong_three_plus_before,
                round.strong_three_plus_after,
                round.strong_track_growth_fraction * 100.0,
                if round.stopped_early { "; stop" } else { "" },
            );
        }
        if !anchor.pairs.is_empty() {
            let mut pair_rms = anchor
                .pairs
                .iter()
                .map(|pair| pair.loo_rms_px)
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            pair_rms.sort_by(f64::total_cmp);
            if let Some(median) = pair_rms.get(pair_rms.len() / 2) {
                println!(
                    "  anchor pair local-LOO RMS: median {:.3} native px across {} validated pairs",
                    median,
                    pair_rms.len(),
                );
            }
        }
    }
    if let Some(latent) = &rig.latent_match {
        println!(
            "  latent graph: {} pairwise seeds -> {} tracks ({} with 3+ cameras); {} target landmarks searched; {} ambiguous observations, {} total candidates (max {})",
            latent.seed_pairwise_matches,
            latent.initial_tracks,
            latent.initial_three_plus_tracks,
            latent.candidate_pool_landmarks,
            latent.ambiguous_observations,
            latent.total_candidates,
            latent.max_candidates_per_observation,
        );
        println!(
            "  latent assignments: {} rounds, {} switches ({} geometry-supported, {} constellation-supported, {} cycle-supported; {} cycle / {} collision proposals rejected)",
            latent.assignment_iterations,
            latent.assignment_switches,
            latent.geometry_supported_switches,
            latent.constellation_supported_switches,
            latent.cycle_supported_switches,
            latent.cycle_rejected_switches,
            latent.collision_rejected_switches,
        );
        if latent.cycle_graph_candidate_edges > 0 {
            println!(
                "  latent cycle graph: {} / {} active/candidate edges, {} fit-only validated pair fields from {} bootstrap cycle-supported tracks ({} observations); {} held-out pair anchors excluded; {} pair predictions evaluated",
                latent.cycle_graph_active_edges,
                latent.cycle_graph_candidate_edges,
                latent.cycle_graph_pairs,
                latent.cycle_graph_anchor_tracks,
                latent.cycle_graph_anchor_observations,
                latent.cycle_graph_validation_anchors_excluded,
                latent.cycle_predictions_evaluated,
            );
        }
        for round in &latent.rounds {
            println!(
                "    round {}: {} switches over {} ambiguous observations; score {:.3}->{:.3}; {} geometry / {} constellation / {} cycle supported; {} cycle / {} collision proposals rejected",
                round.iteration,
                round.switches,
                round.evaluated_observations,
                round.mean_score_before,
                round.mean_score_after,
                round.geometry_supported_switches,
                round.constellation_supported_switches,
                round.cycle_supported_switches,
                round.cycle_rejected_switches,
                round.collision_rejected_switches,
            );
        }
    }
    if rig.image_space_evaluated_cameras > 0 {
        println!(
            "  downstream residual correction: median {:.2}->{:.2} px across {} fitted cameras ({:+.2}%)",
            rig.image_space_median_correction_before_px,
            rig.image_space_median_correction_after_px,
            rig.image_space_evaluated_cameras,
            rig.image_space_relative_improvement * 100.0,
        );
    }
    if let Some(warning) = &rig.image_space_warning {
        println!("  downstream residual diagnostic warning: {warning}");
    }
    if rig.validation_evaluated {
        println!(
            "  rig optimizer: {} coordinate sweeps, {} robust membership passes",
            rig.optimizer_iterations, rig.membership_iterations,
        );
        println!(
            "  positive-depth tracks: fit {:.1}->{:.1}%, held-out {:.1}->{:.1}%",
            rig.fit_positive_depth_fraction_before * 100.0,
            rig.fit_positive_depth_fraction_after * 100.0,
            rig.held_out_positive_depth_fraction_before * 100.0,
            rig.held_out_positive_depth_fraction_after * 100.0,
        );
        println!(
            "  held-out residual percentiles: sensor median/p75/p90/p95 {:.3}/{:.3}/{:.3}/{:.3} px; reference-equivalent {:.3}/{:.3}/{:.3}/{:.3} px; angular {:.5}/{:.5}/{:.5}/{:.5} deg",
            rig.held_out_residuals_after.sensor_pixels.median,
            rig.held_out_residuals_after.sensor_pixels.p75,
            rig.held_out_residuals_after.sensor_pixels.p90,
            rig.held_out_residuals_after.sensor_pixels.p95,
            rig.held_out_residuals_after
                .reference_equivalent_pixels
                .median,
            rig.held_out_residuals_after.reference_equivalent_pixels.p75,
            rig.held_out_residuals_after.reference_equivalent_pixels.p90,
            rig.held_out_residuals_after.reference_equivalent_pixels.p95,
            rig.held_out_residuals_after.angular_degrees.median,
            rig.held_out_residuals_after.angular_degrees.p75,
            rig.held_out_residuals_after.angular_degrees.p90,
            rig.held_out_residuals_after.angular_degrees.p95,
        );
    }
    if rig.physical_match_candidates > 0 {
        println!(
            "  physical matcher: {} candidates, {} tracks/{} target observations; pruned {} inconsistent observations and rejected {} inconsistent tracks; used {}",
            rig.physical_match_candidates,
            rig.physical_match_tracks,
            rig.physical_match_observations,
            rig.rejected_inconsistent_observations,
            rig.rejected_inconsistent_tracks,
            rig.physical_match_used,
        );
        println!(
            "  candidate funnel: {} no supported depth, {} ambiguous depth, {} insufficient native-resolution views, {} failed physical consistency",
            rig.physical_match_rejected_no_supported_depth,
            rig.physical_match_rejected_ambiguous_depth,
            rig.physical_match_rejected_insufficient_views,
            rig.rejected_inconsistent_tracks,
        );
        println!(
            "  depth hierarchy: {:.1} hypotheses/candidate, up to {} refinement levels, {:.2} px worst final projected step ({:.1} px limit)",
            rig.physical_match_depth_hypotheses as f64
                / rig.physical_match_candidates.max(1) as f64,
            rig.physical_match_max_depth_refinement_levels,
            rig.physical_match_observed_max_projected_step_px,
            rig.physical_match_max_projected_step_px,
        );
        println!(
            "  depth levels (level: candidates): {}",
            rig.physical_match_depth_refinement_histogram
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .map(|(level, count)| format!("{level}:{count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if rig.membership_iterations > 0 {
            println!(
                "  persistent-track solve: {} membership passes, {} fit observations/{} tracks rejected without rematching",
                rig.membership_iterations,
                rig.fit_membership_rejected_observations,
                rig.fit_membership_rejected_tracks,
            );
        }
    }
    for correction in rig.corrections.iter().filter(|correction| {
        correction.orientation_offset_degrees != [0.0; 3]
            || correction.mirror_angle_offset_degrees != 0.0
            || correction.center_offset_world != [0.0; 3]
            || correction.focus_pupil_scale != 0.0
            || correction.sensor_offset_px != [0.0; 2]
            || correction.focal_scale_delta != 0.0
            || correction.focal_aspect_delta != 0.0
            || correction.distortion_center_offset_px != [0.0; 2]
            || correction.distortion_delta != [0.0; 4]
    }) {
        println!(
            "  {} {}{rig_strategy} correction: orientation {:+.4},{:+.4},{:+.4} deg, centre {:+.3},{:+.3},{:+.3}, focus-pupil {:+.4}, sensor {:+.2},{:+.2} px, focal(s,a) {:+.3},{:+.3}%, dist-centre {:+.2},{:+.2} px, d[k1,k2,p1,p2]=[{:+.5},{:+.5},{:+.5},{:+.5}], mirror {:+.4} deg{}",
            correction.camera,
            if rig.accepted { "" } else { "candidate " },
            correction.orientation_offset_degrees[0],
            correction.orientation_offset_degrees[1],
            correction.orientation_offset_degrees[2],
            correction.center_offset_world[0],
            correction.center_offset_world[1],
            correction.center_offset_world[2],
            correction.focus_pupil_scale,
            correction.sensor_offset_px[0],
            correction.sensor_offset_px[1],
            correction.focal_scale_delta * 100.0,
            correction.focal_aspect_delta * 100.0,
            correction.distortion_center_offset_px[0],
            correction.distortion_center_offset_px[1],
            correction.distortion_delta[0],
            correction.distortion_delta[1],
            correction.distortion_delta[2],
            correction.distortion_delta[3],
            correction.mirror_angle_offset_degrees,
            if correction.reached_bound {
                " (at bound)"
            } else {
                ""
            },
        );
    }
    if let Some(audit) = &report.dense_depth_audit {
        println!(
            "dense-depth A/B audit (selected: {}; common anchor: {}):",
            audit.selected_path, audit.common_anchor,
        );
        for (name, branch) in [("factory", &audit.factory), ("candidate", &audit.candidate)] {
            println!(
                "  {name:<9} {}/{} measured ({:.2}%), {} regularized, {} accepted target views, available {}; funnel {} selected -> {} neighbour -> {} component, {} supported fallback",
                branch.measured_nodes,
                branch.tested_nodes,
                branch.reconstructed_fraction * 100.0,
                branch.regularized_nodes,
                branch.accepted_views,
                branch.depth_available,
                branch.direct_selected_nodes,
                branch.neighbour_consistent_nodes,
                branch.component_consistent_nodes,
                branch.far_supported_nodes,
            );
        }
        println!(
            "  candidate-factory: {:+} measured nodes, {:+.2} percentage points",
            audit.candidate.measured_nodes as i64 - audit.factory.measured_nodes as i64,
            (audit.candidate.reconstructed_fraction - audit.factory.reconstructed_fraction) * 100.0,
        );
    }
    for module in &report.modules {
        let depth_support = module.depth.as_ref().map(|depth| {
            let unknown_fraction = if depth.tested_nodes == 0 {
                0.0
            } else {
                depth.unknown_nodes as f32 / depth.tested_nodes as f32
            };
            format!(
                ", warp defined {:.1}%/direct {:.1}%/unknown {:.1}%",
                depth.defined_fraction * 100.0,
                depth.directly_supported_fraction * 100.0,
                unknown_fraction * 100.0,
            )
        });
        println!(
            "  {:<3} {:<14} overlap {:>5.1}%{}  inliers {:>4}/{:<4} residual median {:>5.2} px p90 {:>5.2} px  correction {:+.1},{:+.1} px  {}",
            module.camera,
            module.initialised_from,
            module.coverage * 100.0,
            depth_support.as_deref().unwrap_or(""),
            module.inliers,
            module.patches,
            module.residual_median_px,
            module.residual_p90_px,
            module.correction_median_px[0],
            module.correction_median_px[1],
            module.status
        );
        if module.rig_feature_candidates > 0 {
            println!(
                "      rig features: {}/{} accepted; {} ambiguous, {} forward/back rejected",
                module.rig_feature_matches,
                module.rig_feature_candidates,
                module.rig_feature_rejected_ambiguous,
                module.rig_feature_rejected_forward_backward,
            );
        }
        if let (Some(reference_pixel), Some(shift), Some(prior_quality)) = (
            module.angle_optical_center_prior_reference_px,
            module.angle_optical_center_prior_shift_target_px,
            module.angle_optical_center_prior_quality,
        ) {
            println!(
                "      angle optical-center mapping: applied at reference {:.1},{:.1} px; target shift {:+.1},{:+.1} px; measured quality {:.2}",
                reference_pixel[0], reference_pixel[1], shift[0], shift[1], prior_quality,
            );
        }
    }
    for (camera, cleanup) in &report.cleanup {
        if cleanup.profile_supplied {
            println!(
                "  {camera:<3} cleanup {}: temperature {:?}->{:?} C{}, defects {}, rows {}, columns {}, mean/max correction {:.3}/{:.3} RAW",
                if cleanup.profile_available {
                    "available"
                } else {
                    "not calibrated"
                },
                cleanup.correction.requested_temperature_c,
                cleanup.correction.applied_temperature_c,
                if cleanup.correction.temperature_clamped {
                    " (clamped)"
                } else {
                    ""
                },
                cleanup.active_learned_defects,
                cleanup.correction.active_rows,
                cleanup.correction.active_columns,
                cleanup.correction.mean_absolute_change,
                cleanup.correction.maximum_absolute_change,
            );
        }
    }
    let array_color = &report.array_color;
    println!(
        "factory colour: {} selected {:?}; prior {:?}; samples {}, modules {}, spatial {:.1}%, confidence {:.3}{}",
        array_color.mode,
        array_color.selected_weights,
        array_color.prior_weights,
        array_color.sample_count,
        array_color.target_modules,
        array_color.spatial_coverage * 100.0,
        array_color.confidence,
        array_color
            .fallback_reason
            .as_ref()
            .map_or(String::new(), |reason| format!("; fallback: {reason}")),
    );
    if let Some(best) = &array_color.best_candidate {
        println!(
            "  best {:?}: array {:.6}, prior {:.6}, total {:.6}",
            best.weights, best.array_disagreement, best.cct_prior_penalty, best.total_score,
        );
    }
    if let Some(second) = &array_color.second_best_candidate {
        println!(
            "  second {:?}: array {:.6}, prior {:.6}, total {:.6}; gap {:.8}",
            second.weights,
            second.array_disagreement,
            second.cct_prior_penalty,
            second.total_score,
            array_color.score_difference.unwrap_or(0.0),
        );
    }
    println!(
        "robust detail/edge rejection: {:.2}% of compared non-reference samples",
        report.synthesis.edge_rejected_fraction * 100.0
    );
    let resolution = &report.synthesis.resolution_reconstruction;
    if resolution.mode == chiaro_fusion::resolution::ResolutionReconstruction::Resample {
        println!("resolution reconstruction: disabled (resample mode)");
    } else {
        println!(
            "resolution reconstruction: {} - {:.2}% candidates, {:.2}% sampling-supported, {:.2}% reconstructed, {:.2} cameras, {:.3} px phase spread, {:.3} confidence",
            resolution.mode,
            resolution.candidate_fraction * 100.0,
            resolution.phase_supported_fraction * 100.0,
            resolution.reconstructed_fraction * 100.0,
            resolution.mean_cameras,
            resolution.mean_phase_spread,
            resolution.mean_confidence,
        );
    }
    if let Some(joint) = &report.synthesis.joint_cfa {
        println!(
            "joint CFA: {:.2}% of {} candidate points reconstructed (stride {}), solver ran at {:.2}% and skipped {:.2}% by structure gate, {:.1} observations from {:.2} cameras/pixel, {:.3} px phase spread, {:.1}% applied, {:.1} iterations, residual {:.6}; in-sample affine fit {:+.2}% (diagnostic only)",
            joint.reconstructed_fraction * 100.0,
            joint.attempted_pixels,
            joint.sampling_stride,
            joint.solver_attempted_fraction * 100.0,
            joint.structure_skipped_fraction * 100.0,
            joint.mean_observations_per_pixel,
            joint.mean_cameras_per_pixel,
            joint.mean_phase_spread,
            joint.mean_application_weight * 100.0,
            joint.mean_solver_iterations,
            joint.mean_weighted_residual,
            joint.in_sample_relative_fit * 100.0,
        );
        println!(
            "  Joint-CFA rejection funnel (of solver attempts): geometry {:.2}%, footprint samples {:.2}%, conditioning {:.2}%",
            joint.insufficient_geometry_fraction * 100.0,
            joint.insufficient_samples_fraction * 100.0,
            joint.solver_rejected_fraction * 100.0,
        );
    }
    for held_out in &report.synthesis.held_out_cfa {
        println!(
            "held-out {}: baseline {:.4}, emitted Joint CFA/fallback {:.4}, improvement {:+.2}% over {} fixed physical CFA samples; solver supported {} ({:.1}%)",
            held_out.camera,
            held_out.overall.baseline,
            held_out.overall.joint_cfa,
            held_out.overall.relative_improvement * 100.0,
            held_out.overall.samples,
            held_out.solver_supported_samples,
            held_out.solver_supported_fraction * 100.0,
        );
    }
    for source in &report.synthesis.source_contributions {
        if let Some(local) = &source.resolution_alignment {
            println!(
                "  {:<3} {:>4.2}x resolution alignment {:>5.1}% verified/{:>5.1}% supported, median correction {:.2} px, confidence {:.2}; candidate {:>5.1}%, accepted {:>5.1}%",
                source.camera,
                source.magnification,
                local.verified_fraction * 100.0,
                local.supported_fraction * 100.0,
                local.median_correction_px,
                local.mean_confidence,
                source.resolution_candidate_fraction * 100.0,
                source.resolution_contributor_fraction * 100.0,
            );
        }
    }
    println!("blend contribution (actual normalized weights; owner is only the per-pixel argmax):");
    for source in &report.synthesis.source_contributions {
        println!(
            "  {:<3} luminance {:>6.2}% (owner {:>6.2}%), colour {:>6.2}% (owner {:>6.2}%)",
            source.camera,
            source.luminance_weight_fraction * 100.0,
            source.luminance_owner_fraction * 100.0,
            source.color_weight_fraction * 100.0,
            source.color_owner_fraction * 100.0,
        );
    }
    println!(
        "timings: load {:.1}s, hotpixel {:.1}s, align {:.1}s, synthesize {:.1}s",
        report.seconds.load,
        report.seconds.hotpixel,
        report.seconds.align,
        report.seconds.synthesize
    );
    println!(
        "resources: {:.2} MP, {:.2}s/MP total, {:.2}s/MP synthesis{}",
        report.resources.output_megapixels,
        report.resources.total_seconds_per_megapixel,
        report.resources.synthesis_seconds_per_megapixel,
        report.resources.peak_resident_bytes.map_or_else(
            || String::from(", peak RSS unavailable"),
            |bytes| format!(", peak RSS {:.1} MiB", bytes as f64 / 1_048_576.0),
        ),
    );
    if let Some(debug_dir) = &cli.debug_dir {
        println!(
            "visual pipeline trace: {}",
            debug_dir.join("index.html").display()
        );
    }
    Ok(())
}

fn validate_cleanup_pair(cleanup: &Option<PathBuf>, hotpixel: &Option<PathBuf>) -> Result<()> {
    if cleanup.is_some() && hotpixel.is_none() {
        bail!("--cleanup-profile requires the corresponding --hotpixel-rec");
    }
    Ok(())
}

fn parse_crop(value: &str) -> Result<CropWindow> {
    let values = value
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("--crop expects four numbers x,y,width,height, not {value}"))?;
    if values.len() != 4 {
        bail!("--crop expects four numbers x,y,width,height, not {value}");
    }
    if values.iter().any(|component| !component.is_finite()) {
        bail!("--crop components must all be finite, not {value}");
    }
    Ok(CropWindow {
        x: values[0],
        y: values[1],
        width: values[2],
        height: values[3],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_cfa_is_the_cli_default() {
        let cli =
            Cli::try_parse_from(["chiaro-fuse", "capture.lri", "--output", "output.png"]).unwrap();
        assert_eq!(
            cli.resolution_reconstruction,
            ResolutionReconstruction::JointCfa
        );
        assert!(!cli.joint_cfa_solve_flat);
        assert!(!cli.no_rig_refine);
    }

    #[test]
    fn flat_joint_cfa_solves_are_an_explicit_diagnostic_mode() {
        let cli = Cli::try_parse_from([
            "chiaro-fuse",
            "capture.lri",
            "--output",
            "output.png",
            "--joint-cfa-solve-flat",
        ])
        .unwrap();
        assert!(cli.joint_cfa_solve_flat);
    }

    #[test]
    fn cleanup_profile_requires_hotpixel_rec() {
        let cleanup = Some(PathBuf::from("camera.chiaro-cleanup"));
        assert!(
            validate_cleanup_pair(&cleanup, &None)
                .unwrap_err()
                .to_string()
                .contains("--hotpixel-rec")
        );
        assert!(validate_cleanup_pair(&cleanup, &Some(PathBuf::from("hotpixel.rec"))).is_ok());
    }

    #[test]
    fn explicit_crop_parses_reference_coordinates() {
        let crop = parse_crop("12.5, 20, 640,480").unwrap();
        assert_eq!(crop.x, 12.5);
        assert_eq!(crop.y, 20.0);
        assert_eq!(crop.width, 640.0);
        assert_eq!(crop.height, 480.0);
        assert!(parse_crop("1,2,3").is_err());
        assert!(parse_crop("1,2,no,4").is_err());
        assert!(parse_crop("NaN,2,3,4").is_err());
        assert!(parse_crop("1,2,inf,4").is_err());
    }
}
