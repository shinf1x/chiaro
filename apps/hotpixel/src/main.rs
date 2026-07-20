use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use memmap2::Mmap;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

use chiaro::lri::{RawCamera, SensorPattern, parse_raw_layout};
use chiaro_hotpixel::correct::{
    CorrectionConfig, CorrectionMode, CorrectionStats, correct_hot_pixels, demosaic_bilinear,
};
use chiaro_hotpixel::hotpixel::HotpixelRec;
use chiaro_hotpixel::png16::{write_gray16_atomic, write_rgb16_atomic};
use chiaro_hotpixel::raw10::unpack_l16_10bit;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum OutputMode {
    /// Bayer cameras become linear 16-bit RGB PNGs through simple bilinear demosaicing.
    /// Monochrome cameras remain 16-bit grayscale.
    Rgb,
    /// Preserve Bayer mosaics as linear 16-bit grayscale PNGs.
    Mosaic,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliCorrectionMode {
    Adaptive,
    Replace,
}

impl From<CliCorrectionMode> for CorrectionMode {
    fn from(value: CliCorrectionMode) -> Self {
        match value {
            CliCorrectionMode::Adaptive => CorrectionMode::Adaptive,
            CliCorrectionMode::Replace => CorrectionMode::Replace,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "chiaro-hotpixel",
    version,
    about = "Convert a folder of Light L16 LRIs into per-camera hot-pixel-corrected 16-bit PNG stacks"
)]
struct Args {
    /// Folder containing .lri captures.
    input: PathBuf,

    /// Output root. Each physical camera receives its own PNG-only subfolder.
    output: PathBuf,

    /// Factory hotpixel.rec belonging to this Light L16.
    #[arg(long)]
    hotpixel_rec: PathBuf,

    /// Search subdirectories under INPUT.
    #[arg(long)]
    recursive: bool,

    /// Process only named cameras; repeat, for example: --camera B1 --camera C3.
    #[arg(long)]
    camera: Vec<String>,

    /// Override missing or incorrect sensor metadata, for example A2=MONO or B1=RGGB.
    #[arg(long, value_name = "CAMERA=PATTERN")]
    pattern: Vec<String>,

    /// RGB is stacker-friendly. Mosaic preserves the corrected Bayer mosaic.
    #[arg(long, value_enum, default_value = "rgb")]
    mode: OutputMode,

    /// Factory severity at which a coordinate becomes a correction candidate.
    #[arg(long, default_value_t = 16)]
    severity_threshold: u8,

    /// Required local outlier strength in robust-deviation units.
    #[arg(long, default_value_t = 6.0)]
    sigma_threshold: f64,

    /// Minimum deviation in original 10-bit RAW codes.
    #[arg(long, default_value_t = 4)]
    absolute_threshold: i32,

    /// Same-color median neighborhood size.
    #[arg(long, value_parser = clap::value_parser!(usize), default_value_t = 5)]
    kernel: usize,

    /// Adaptive verifies that the calibrated pixel is also a local outlier.
    #[arg(long, value_enum, default_value = "adaptive")]
    correction_mode: CliCorrectionMode,

    /// Delete OUTPUT first and regenerate every frame.
    #[arg(long, conflicts_with = "resume")]
    overwrite: bool,

    /// Keep existing PNGs and process only missing frames.
    #[arg(long, conflicts_with = "overwrite")]
    resume: bool,

    /// Continue processing other frames after an error; exits nonzero at the end.
    #[arg(long)]
    continue_on_error: bool,
}

#[derive(Clone, Debug)]
struct Job {
    source: PathBuf,
    source_relative: String,
    output_name: String,
    camera: RawCamera,
}

type JobsByCamera = BTreeMap<String, Vec<Job>>;
type JobScan = (JobsByCamera, Vec<FailureReport>);

#[derive(Debug, Serialize)]
struct HotpixelManifest {
    path: String,
    sha256: String,
    record_count: usize,
    camera_record_rule: &'static str,
    orientation: &'static str,
}

#[derive(Debug, Serialize)]
struct SettingsManifest {
    output_mode: OutputMode,
    severity_threshold: u8,
    sigma_threshold: f64,
    absolute_threshold: i32,
    kernel: usize,
    correction_mode: &'static str,
    png_scaling: &'static str,
    color_processing: &'static str,
}

#[derive(Debug, Serialize)]
struct FrameReport {
    source: String,
    camera: String,
    output: String,
    pattern: String,
    width: usize,
    height: usize,
    png_color_type: &'static str,
    status: &'static str,
    candidates: usize,
    corrected: usize,
    positive_corrected: usize,
    negative_corrected: usize,
    corrected_fraction: f64,
    mean_absolute_change: f64,
    maximum_absolute_change: u16,
    elapsed_seconds: f64,
}

#[derive(Debug, Serialize)]
struct FailureReport {
    source: String,
    camera: Option<String>,
    error: String,
}

#[derive(Debug, Serialize)]
struct RunManifest {
    tool: &'static str,
    version: &'static str,
    input: String,
    output: String,
    hotpixel: HotpixelManifest,
    settings: SettingsManifest,
    frames: Vec<FrameReport>,
    failures: Vec<FailureReport>,
}

fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: the mapping is read-only and the File remains valid during creation.
    unsafe { Mmap::map(&file) }.with_context(|| format!("memory-map {}", path.display()))
}

fn parse_pattern_overrides(values: &[String]) -> Result<HashMap<String, SensorPattern>> {
    let mut result = HashMap::new();
    for value in values {
        let (camera, pattern) = value
            .split_once('=')
            .with_context(|| format!("pattern override must be CAMERA=PATTERN: {value}"))?;
        result.insert(camera.to_ascii_uppercase(), SensorPattern::parse(pattern)?);
    }
    Ok(result)
}

fn discover_lri_files(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!("input is not a directory: {}", root.display());
    }
    let mut files = Vec::new();
    let walker = if recursive {
        WalkDir::new(root)
    } else {
        WalkDir::new(root).max_depth(1)
    };
    for entry in walker.follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let is_lri = entry
            .path()
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case("lri"))
            .unwrap_or(false);
        if is_lri {
            files.push(entry.into_path());
        }
    }
    files.sort();
    if files.is_empty() {
        bail!("no .lri files found under {}", root.display());
    }
    Ok(files)
}

fn sanitize_component(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
            output.push(character);
        } else {
            output.push('_');
        }
    }
    if output.is_empty() {
        "capture".to_owned()
    } else {
        output
    }
}

fn capture_key(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let mut parts = Vec::new();
    for component in relative.components() {
        if let Component::Normal(value) = component {
            parts.push(value.to_string_lossy().to_string());
        }
    }
    if let Some(last) = parts.last_mut()
        && let Some(stem) = Path::new(last).file_stem().and_then(|value| value.to_str())
    {
        *last = stem.to_owned();
    }
    parts
        .iter()
        .map(|part| sanitize_component(part))
        .collect::<Vec<_>>()
        .join("__")
}

fn prepare_output(args: &Args) -> Result<()> {
    if args.output.exists() {
        if args.overwrite {
            fs::remove_dir_all(&args.output)
                .with_context(|| format!("remove {}", args.output.display()))?;
        } else if !args.resume {
            let mut entries = fs::read_dir(&args.output)?;
            if entries.next().is_some() {
                bail!(
                    "output is not empty: {}; pass --overwrite or --resume",
                    args.output.display()
                );
            }
        }
    }
    fs::create_dir_all(&args.output)?;
    Ok(())
}

fn build_jobs(
    args: &Args,
    files: &[PathBuf],
    pattern_overrides: &HashMap<String, SensorPattern>,
) -> Result<JobScan> {
    let selected: HashSet<String> = args
        .camera
        .iter()
        .map(|camera| camera.to_ascii_uppercase())
        .collect();
    let mut grouped = BTreeMap::<String, Vec<Job>>::new();
    let mut failures = Vec::new();

    for path in files {
        let source_relative = path
            .strip_prefix(&args.input)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let result = (|| -> Result<()> {
            let mmap = mmap_file(path)?;
            let layout = parse_raw_layout(&mmap, pattern_overrides)?;
            let key = capture_key(&args.input, path);
            for camera in layout.cameras {
                if !selected.is_empty() && !selected.contains(&camera.name) {
                    continue;
                }
                grouped.entry(camera.name.clone()).or_default().push(Job {
                    source: path.clone(),
                    source_relative: source_relative.clone(),
                    output_name: format!("{key}.png"),
                    camera,
                });
            }
            Ok(())
        })();

        if let Err(error) = result {
            if !args.continue_on_error {
                return Err(error).with_context(|| format!("scan {}", path.display()));
            }
            failures.push(FailureReport {
                source: source_relative,
                camera: None,
                error: format!("{error:#}"),
            });
        }
    }

    if grouped.is_empty() {
        bail!("none of the selected cameras were found in the input LRIs");
    }

    for (camera, jobs) in &grouped {
        let mut names = HashSet::new();
        for job in jobs {
            if !names.insert(job.output_name.clone()) {
                bail!(
                    "two input LRIs produce the same output name in {camera}/: {}",
                    job.output_name
                );
            }
        }
    }
    Ok((grouped, failures))
}

fn process_frame(
    args: &Args,
    job: &Job,
    severity_map: &[u8],
    config: &CorrectionConfig,
) -> Result<FrameReport> {
    let camera_dir = args.output.join(&job.camera.name);
    fs::create_dir_all(&camera_dir)?;
    let output_path = camera_dir.join(&job.output_name);
    let output_relative = output_path
        .strip_prefix(&args.output)
        .unwrap_or(&output_path)
        .to_string_lossy()
        .to_string();

    if output_path.exists() {
        if args.resume {
            return Ok(FrameReport {
                source: job.source_relative.clone(),
                camera: job.camera.name.clone(),
                output: output_relative,
                pattern: job.camera.pattern.as_str().to_owned(),
                width: job.camera.width,
                height: job.camera.height,
                png_color_type: if matches!(args.mode, OutputMode::Rgb)
                    && job.camera.pattern != SensorPattern::Mono
                {
                    "RGB16"
                } else {
                    "GRAY16"
                },
                status: "skipped-existing",
                candidates: 0,
                corrected: 0,
                positive_corrected: 0,
                negative_corrected: 0,
                corrected_fraction: 0.0,
                mean_absolute_change: 0.0,
                maximum_absolute_change: 0,
                elapsed_seconds: 0.0,
            });
        }
        fs::remove_file(&output_path)?;
    }

    let started = Instant::now();
    let mmap = mmap_file(&job.source)?;
    let start = job.camera.absolute_offset;
    let end = start + job.camera.byte_len;
    if end > mmap.len() {
        bail!("RAW span for {} lies outside the LRI", job.camera.name);
    }
    let expected_packed = job
        .camera
        .width
        .checked_mul(job.camera.height)
        .and_then(|samples| samples.checked_mul(5))
        .map(|bytes| bytes / 4)
        .context("RAW dimensions overflow")?;
    if job.camera.byte_len != expected_packed {
        bail!(
            "{} uses a padded RAW stride ({} bytes, expected {}); this version supports tightly packed RAW10",
            job.camera.name,
            job.camera.byte_len,
            expected_packed
        );
    }

    let mut raw = unpack_l16_10bit(&mmap[start..end], job.camera.width * job.camera.height)?;
    let stats: CorrectionStats = correct_hot_pixels(
        &mut raw,
        job.camera.width,
        job.camera.height,
        job.camera.pattern,
        severity_map,
        config,
    )?;

    let png_color_type = match args.mode {
        OutputMode::Rgb if job.camera.pattern != SensorPattern::Mono => {
            let rgb = demosaic_bilinear(
                &raw,
                job.camera.width,
                job.camera.height,
                job.camera.pattern,
            )?;
            write_rgb16_atomic(&output_path, job.camera.width, job.camera.height, &rgb)?;
            "RGB16"
        }
        _ => {
            write_gray16_atomic(&output_path, job.camera.width, job.camera.height, &raw)?;
            "GRAY16"
        }
    };

    Ok(FrameReport {
        source: job.source_relative.clone(),
        camera: job.camera.name.clone(),
        output: output_relative,
        pattern: job.camera.pattern.as_str().to_owned(),
        width: job.camera.width,
        height: job.camera.height,
        png_color_type,
        status: "written",
        candidates: stats.candidates,
        corrected: stats.corrected,
        positive_corrected: stats.positive_corrected,
        negative_corrected: stats.negative_corrected,
        corrected_fraction: stats.corrected as f64 / raw.len() as f64,
        mean_absolute_change: stats.mean_absolute_change,
        maximum_absolute_change: stats.maximum_absolute_change,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn write_root_notes(args: &Args) -> Result<()> {
    let text = format!(
        "Light L16 astrophotography stacks\n\n\
Each camera subdirectory contains PNG frames only and can be selected directly in stacking software.\n\
The frames are linear 16-bit images. The only sensor correction is factory-guided hot/dead-pixel interpolation.\n\
No black-level subtraction, white balance, color matrix, exposure normalization, gamma, stretch, sharpening,\n\
flat-field correction, dark subtraction, denoising, or alignment has been applied.\n\
Bayer output mode: {:?}. Bayer RGB output uses simple linear bilinear demosaicing.\n",
        args.mode
    );
    fs::write(args.output.join("README.txt"), text)?;
    Ok(())
}

fn run(args: Args) -> Result<()> {
    if !matches!(args.kernel, 3 | 5 | 7) {
        bail!("--kernel must be 3, 5, or 7");
    }
    if args.sigma_threshold < 0.0 || args.absolute_threshold < 0 {
        bail!("correction thresholds must be non-negative");
    }
    if !args.hotpixel_rec.is_file() {
        bail!(
            "hotpixel record does not exist: {}",
            args.hotpixel_rec.display()
        );
    }

    prepare_output(&args)?;
    let files = discover_lri_files(&args.input, args.recursive)?;
    let pattern_overrides = parse_pattern_overrides(&args.pattern)?;
    let hotpixel = HotpixelRec::open(&args.hotpixel_rec)?;
    if hotpixel.records.len() != 16 {
        eprintln!(
            "warning: expected 16 factory records, found {}",
            hotpixel.records.len()
        );
    }

    println!(
        "Found {} LRI files; factory calibration has {} maps ({})",
        files.len(),
        hotpixel.records.len(),
        hotpixel.sha256
    );

    let (grouped, mut failures) = build_jobs(&args, &files, &pattern_overrides)?;
    let config = CorrectionConfig {
        mode: args.correction_mode.into(),
        severity_threshold: args.severity_threshold,
        sigma_threshold: args.sigma_threshold,
        absolute_threshold: args.absolute_threshold,
        kernel: args.kernel,
    };
    let mut reports = Vec::new();

    for (camera_name, jobs) in grouped {
        let first = jobs.first().context("empty camera job group")?;
        let map =
            hotpixel.load_rotated_map(first.camera.id, first.camera.width, first.camera.height)?;
        println!("{camera_name}: {} frames", jobs.len());

        for (index, job) in jobs.iter().enumerate() {
            if job.camera.width != first.camera.width || job.camera.height != first.camera.height {
                let error = anyhow::anyhow!(
                    "camera {} changes dimensions between captures",
                    job.camera.name
                );
                if !args.continue_on_error {
                    return Err(error);
                }
                failures.push(FailureReport {
                    source: job.source_relative.clone(),
                    camera: Some(job.camera.name.clone()),
                    error: format!("{error:#}"),
                });
                continue;
            }

            match process_frame(&args, job, &map, &config) {
                Ok(report) => {
                    println!(
                        "  [{:>3}/{}] {}: {} pixels corrected ({:.3}s)",
                        index + 1,
                        jobs.len(),
                        job.output_name,
                        report.corrected,
                        report.elapsed_seconds
                    );
                    reports.push(report);
                }
                Err(error) => {
                    if !args.continue_on_error {
                        return Err(error).with_context(|| {
                            format!("process {} {}", job.source.display(), job.camera.name)
                        });
                    }
                    eprintln!(
                        "  ERROR {} {}: {error:#}",
                        job.source.display(),
                        job.camera.name
                    );
                    failures.push(FailureReport {
                        source: job.source_relative.clone(),
                        camera: Some(job.camera.name.clone()),
                        error: format!("{error:#}"),
                    });
                }
            }
        }
    }

    reports.sort_by(|left, right| {
        left.camera
            .cmp(&right.camera)
            .then_with(|| left.source.cmp(&right.source))
    });

    let manifest = RunManifest {
        tool: "chiaro-hotpixel",
        version: env!("CARGO_PKG_VERSION"),
        input: args.input.to_string_lossy().to_string(),
        output: args.output.to_string_lossy().to_string(),
        hotpixel: HotpixelManifest {
            path: args.hotpixel_rec.to_string_lossy().to_string(),
            sha256: hotpixel.sha256.clone(),
            record_count: hotpixel.records.len(),
            camera_record_rule: "A1..A5=0..4; B1..B5=5..9; C1..C6=10..15",
            orientation: "rotate180",
        },
        settings: SettingsManifest {
            output_mode: args.mode,
            severity_threshold: args.severity_threshold,
            sigma_threshold: args.sigma_threshold,
            absolute_threshold: args.absolute_threshold,
            kernel: args.kernel,
            correction_mode: match args.correction_mode {
                CliCorrectionMode::Adaptive => "adaptive",
                CliCorrectionMode::Replace => "replace",
            },
            png_scaling: "10-bit RAW code left-shifted by 6 into linear 16-bit PNG",
            color_processing: match args.mode {
                OutputMode::Rgb => "Bayer cameras: bilinear demosaic only; mono: grayscale",
                OutputMode::Mosaic => "corrected RAW mosaic/grayscale; no demosaic",
            },
        },
        frames: reports,
        failures,
    };

    fs::write(
        args.output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    write_root_notes(&args)?;

    println!("Wrote camera stacks under {}", args.output.display());
    if !manifest.failures.is_empty() {
        bail!(
            "{} frames/captures failed; see manifest.json",
            manifest.failures.len()
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
