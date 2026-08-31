use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use chiaro_fusion::calibration::IntrinsicsMode;
use chiaro_fusion::pipeline::{FusionOptions, HotpixelStage, fuse};
use chiaro_fusion::synth::{CanvasMode, OutputColor};
use chiaro_hotpixel_core::demosaic::DemosaicMethod;
use chiaro_hotpixel_core::scan::mmap_file;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Color {
    Display,
    Linear,
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

    /// Device calibration overlays (calibration.lri, zoom_calib_v0.lri). Repeatable.
    #[arg(long = "calibration", value_name = "FILE")]
    overlays: Vec<PathBuf>,

    /// Reference module (default: the capture's reference camera).
    #[arg(long)]
    reference: Option<String>,

    /// Use only these modules; repeatable.
    #[arg(long)]
    camera: Vec<String>,

    /// Canvas size: `native` (13 MP), `max` (as the finest covering module
    /// allows, capped by --max-megapixels), or a number of canvas pixels per
    /// reference pixel.
    #[arg(long, default_value = "native")]
    canvas: String,

    /// Cap for `--canvas max`.
    #[arg(long, default_value_t = 64.0)]
    max_megapixels: f32,

    /// Render the full wide frame instead of cropping to the framed focal length.
    #[arg(long)]
    no_crop: bool,

    #[arg(long, value_enum, default_value = "display")]
    color: Color,

    /// Bayer reconstruction method.
    #[arg(long, value_enum, default_value = "amaze")]
    demosaic: Demosaic,

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
        }),
        cameras: cli.camera.clone(),
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
    println!(
        "robust detail/edge rejection: {:.2}% of compared non-reference samples",
        report.synthesis.edge_rejected_fraction * 100.0
    );
    println!(
        "timings: load {:.1}s, hotpixel {:.1}s, align {:.1}s, synthesize {:.1}s",
        report.seconds.load,
        report.seconds.hotpixel,
        report.seconds.align,
        report.seconds.synthesize
    );
    Ok(())
}
