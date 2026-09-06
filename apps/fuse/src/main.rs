use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};

use chiaro_fusion::array_color::ColorProfileMode;
use chiaro_fusion::calibration::IntrinsicsMode;
use chiaro_fusion::crosstalk::CrosstalkMode;
use chiaro_fusion::pipeline::{FusionOptions, HotpixelStage, fuse};
use chiaro_fusion::resolution::ResolutionReconstruction;
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

    /// Skip the factory flat-field (vignetting) correction.
    #[arg(long)]
    no_flat_field: bool,

    /// Write per-module alignment checkerboards into this folder.
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
    for module in &report.modules {
        println!(
            "  {:<3} {:<14} coverage {:>5.1}%  inliers {:>4}/{:<4} residual median {:>5.2} px p90 {:>5.2} px  correction {:+.1},{:+.1} px  {}",
            module.camera,
            module.initialised_from,
            module.coverage * 100.0,
            module.inliers,
            module.patches,
            module.residual_median_px,
            module.residual_p90_px,
            module.correction_median_px[0],
            module.correction_median_px[1],
            module.status
        );
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
