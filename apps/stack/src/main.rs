use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chiaro_hotpixel_core::{
    demosaic::DemosaicMethod,
    hotpixel::HotpixelRec,
    png16::{write_gray16_native_atomic, write_rgb16_native_atomic},
    scan::mmap_file,
};
use chiaro_stack::{
    StackOptions,
    fusion::{NightFusionOptions, fuse_night, set_output_color},
    stack_burst,
};
use clap::{Parser, ValueEnum};

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
    name = "chiaro-stack",
    version,
    about = "Align and motion-safely denoise repeated frames in a Light L16 night capture"
)]
struct Cli {
    /// Night-mode LRI capture.
    input: PathBuf,

    /// Output 16-bit RGB PNG.
    #[arg(long, short)]
    output: PathBuf,

    /// Stack only one physical camera (for example A1 or B3). Omit this to
    /// stack and fuse every temporal module in the capture.
    #[arg(long)]
    camera: Option<String>,

    /// Factory hotpixel.rec. Its map for the selected physical camera is
    /// applied independently to every temporal frame.
    #[arg(long)]
    hotpixel_rec: Option<PathBuf>,

    /// Device calibration overlays (calibration.lri, zoom_calib_v0.lri).
    #[arg(long = "calibration", value_name = "FILE")]
    overlays: Vec<PathBuf>,

    /// Reference physical module for all-module fusion.
    #[arg(long)]
    reference: Option<String>,

    /// Motion rejection cutoff in predicted standard deviations. Lower is
    /// safer around movement; higher retains more samples.
    #[arg(long, default_value_t = 4.0)]
    motion_sigma: f32,

    /// Skip correlation refinement (diagnostic only).
    #[arg(long)]
    no_refine: bool,

    /// Disable the row-indexed gyroscope rotation seed.
    #[arg(long)]
    no_gyro_seed: bool,

    /// Preserve linear camera RGB. By default a simple white balance and sRGB
    /// transfer are applied for convenient visual comparison.
    #[arg(long)]
    linear: bool,

    /// Bayer reconstruction method.
    #[arg(long, value_enum, default_value = "lmmse")]
    demosaic: Demosaic,

    /// Worker threads (0 = all cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Write the chosen reference and effective-frame-count images beside the output.
    #[arg(long)]
    diagnostics: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let lri = mmap_file(&cli.input)?;
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    if cli.camera.is_none() {
        let mut options = NightFusionOptions {
            reference: cli.reference.clone(),
            overlays: cli.overlays.clone(),
            hotpixel_rec: cli.hotpixel_rec.clone(),
            motion_sigma: cli.motion_sigma,
            gyro_seed: !cli.no_gyro_seed,
            threads: cli.threads,
            ..NightFusionOptions::default()
        };
        options.temporal_align.refine = !cli.no_refine;
        options.module_align.refine = !cli.no_refine;
        options.synth.threads = cli.threads;
        options.synth.demosaic = cli.demosaic.into();
        set_output_color(&mut options, cli.linear);
        let report = fuse_night(&lri, &options, &cli.output, &mut |detail| {
            eprintln!("{detail}…");
        })?;
        eprintln!(
            "wrote {} from {} stacked modules at common temporal frame {}",
            cli.output.display(),
            report.temporal.len(),
            report.reference_frame
        );
        eprintln!(
            "report: {}",
            cli.output.with_extension("night-fusion.json").display()
        );
        return Ok(());
    }
    let camera = cli.camera.as_deref().expect("single-camera branch");
    let mut options = StackOptions {
        camera: camera.to_ascii_uppercase(),
        motion_sigma: cli.motion_sigma,
        gyro_seed: !cli.no_gyro_seed,
        demosaic: cli.demosaic.into(),
        threads: cli.threads,
        ..StackOptions::default()
    };
    options.align.refine = !cli.no_refine;
    if let Some(path) = &cli.hotpixel_rec {
        let camera_id = camera_index(&options.camera)
            .with_context(|| format!("unknown L16 camera {}", options.camera))?;
        let rec = HotpixelRec::open(path)?;
        let frame = chiaro::lri::parse_frame_layout(&lri, &Default::default())?
            .frames
            .into_iter()
            .find(|frame| frame.camera.id == camera_id)
            .with_context(|| format!("capture has no {} frame", options.camera))?;
        options.severity_map =
            Some(rec.load_rotated_map(camera_id, frame.camera.width, frame.camera.height)?);
    }
    eprintln!("stacking {} temporal frames…", options.camera);
    let result = stack_burst(&lri, &options)?;
    let (output, reference) = if cli.linear {
        (result.rgb16.clone(), result.reference_rgb16.clone())
    } else {
        let treatment = DisplayTreatment::estimate(&result.rgb16);
        (
            treatment.apply(&result.rgb16),
            treatment.apply(&result.reference_rgb16),
        )
    };
    write_rgb16_native_atomic(&cli.output, result.width, result.height, &output)?;
    let report_path = sibling(&cli.output, "night-stack", "json");
    std::fs::write(&report_path, serde_json::to_vec_pretty(&result.report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    if cli.diagnostics {
        let reference_path = sibling(&cli.output, "reference", "png");
        let count_path = sibling(&cli.output, "effective-frames", "png");
        write_rgb16_native_atomic(&reference_path, result.width, result.height, &reference)?;
        write_gray16_native_atomic(
            &count_path,
            result.width,
            result.height,
            &result.effective_count,
        )?;
        eprintln!("reference: {}", reference_path.display());
        eprintln!("contributions: {}", count_path.display());
    }
    eprintln!(
        "wrote {} (reference frame {}, {:.2} effective frames average, {:.1}% reference-only)",
        cli.output.display(),
        result.report.reference_frame,
        result.report.mean_effective_frames,
        result.report.fallback_fraction * 100.0
    );
    eprintln!("report: {}", report_path.display());
    Ok(())
}

fn camera_index(name: &str) -> Option<usize> {
    const NAMES: [&str; 16] = [
        "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5",
        "C6",
    ];
    NAMES.iter().position(|candidate| *candidate == name)
}

fn sibling(path: &Path, label: &str, extension: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("stack");
    path.with_file_name(format!("{stem}.{label}.{extension}"))
}

#[derive(Clone, Copy)]
struct DisplayTreatment {
    gains: [f32; 3],
    exposure: f32,
}

impl DisplayTreatment {
    fn estimate(rgb: &[u16]) -> Self {
        let pixels = rgb.len() / 3;
        let step = (pixels / 200_000).max(1);
        let mut sums = [0.0f64; 3];
        let mut count = 0usize;
        let mut luminances = Vec::with_capacity(pixels.div_ceil(step));
        for index in (0..pixels).step_by(step) {
            let sample = [
                rgb[index * 3] as f32 / 65535.0,
                rgb[index * 3 + 1] as f32 / 65535.0,
                rgb[index * 3 + 2] as f32 / 65535.0,
            ];
            if sample.iter().all(|value| *value > 0.002 && *value < 0.95) {
                for channel in 0..3 {
                    sums[channel] += f64::from(sample[channel]);
                }
                count += 1;
            }
        }
        let means = sums.map(|sum| (sum / count.max(1) as f64) as f32);
        let green = means[1].max(1e-4);
        let gains = [
            (green / means[0].max(1e-4)).clamp(0.25, 4.0),
            1.0,
            (green / means[2].max(1e-4)).clamp(0.25, 4.0),
        ];
        for index in (0..pixels).step_by(step) {
            let r = rgb[index * 3] as f32 / 65535.0 * gains[0];
            let g = rgb[index * 3 + 1] as f32 / 65535.0;
            let b = rgb[index * 3 + 2] as f32 / 65535.0 * gains[2];
            luminances.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
        }
        luminances.sort_by(f32::total_cmp);
        let high = luminances
            .get(luminances.len().saturating_mul(995) / 1000)
            .copied()
            .unwrap_or(1.0)
            .max(1e-4);
        Self {
            gains,
            exposure: (0.95 / high).clamp(0.25, 16.0),
        }
    }

    fn apply(self, rgb: &[u16]) -> Vec<u16> {
        rgb.chunks_exact(3)
            .flat_map(|pixel| {
                [0, 1, 2].map(|channel| {
                    let linear =
                        pixel[channel] as f32 / 65535.0 * self.gains[channel] * self.exposure;
                    let mapped = linear / (1.0 + linear);
                    let srgb = if mapped <= 0.0031308 {
                        12.92 * mapped
                    } else {
                        1.055 * mapped.powf(1.0 / 2.4) - 0.055
                    };
                    (srgb.clamp(0.0, 1.0) * 65535.0).round() as u16
                })
            })
            .collect()
    }
}
