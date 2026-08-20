use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use memmap2::Mmap;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

use chiaro::lri::{RawCamera, SensorPattern, parse_raw_layout};
use chiaro_hotpixel::cleanup::{
    BuildCleanupProfileOptions, CleanupCameraProfile, CleanupCorrectionStats, CleanupProfile,
    build_cleanup_profile,
};
use chiaro_hotpixel::correct::{
    CorrectionConfig, CorrectionMode, CorrectionStats, correct_hot_pixels_with_forced_map,
    demosaic_bilinear,
};
use chiaro_hotpixel::hotpixel::HotpixelRec;
use chiaro_hotpixel::png16::{write_gray16_native_atomic, write_rgb16_native_atomic};
use chiaro_hotpixel::raw10::unpack_l16_10bit;
use chiaro_hotpixel::thermal::{ThermalCorrectionStats, ThermalProfile};
use chiaro_hotpixel::universal_hotpixel::{UniversalHotpixelProfile, UniversalHotpixelStats};

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
    about = "Correct Light L16 captures or fit one portable cleanup file from dark LRIs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Correct LRIs and write per-camera PNG stacks.
    Extract(ExtractArgs),
    /// Fit one portable cleanup file from dark LRIs.
    Calibrate(CalibrateArgs),
}

#[derive(Debug, clap::Args)]
struct ExtractArgs {
    /// Folder containing .lri captures.
    #[arg(long, value_name = "DIRECTORY")]
    input: PathBuf,

    /// Output root containing one PNG-only directory per physical camera.
    #[arg(long, value_name = "DIRECTORY")]
    output: PathBuf,

    /// Factory hotpixel.rec belonging to this Light L16.
    #[arg(long)]
    hotpixel_rec: PathBuf,

    /// Universal temperature-conditioned corner-glow profile.
    #[arg(
        long = "glow-profile",
        alias = "thermal-profile",
        value_name = "DIRECTORY",
        conflicts_with = "no_glow_correction"
    )]
    glow_profile: Option<PathBuf>,

    /// Disable the bundled universal corner-glow correction.
    #[arg(long)]
    no_glow_correction: bool,

    /// Disable the bundled A/B/C temperature/exposure/gain hot-pixel prior.
    #[arg(long)]
    no_universal_hotpixel_model: bool,

    /// Override the bundled coordinate-free factory-hotpixel response model.
    #[arg(
        long,
        value_name = "FILE",
        conflicts_with = "no_universal_hotpixel_model"
    )]
    universal_hotpixel_profile: Option<PathBuf>,

    /// Optional camera-specific temperature hot-pixel and row/column cleanup file.
    #[arg(long, value_name = "FILE")]
    cleanup_profile: Option<PathBuf>,

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

    /// Replace OUTPUT before extraction.
    #[arg(long, conflicts_with = "resume")]
    overwrite: bool,

    /// Keep existing PNGs and process only missing frames.
    #[arg(long, conflicts_with = "overwrite")]
    resume: bool,

    /// Continue processing other frames after an error; exits nonzero at the end.
    #[arg(long)]
    continue_on_error: bool,
}

#[derive(Debug, clap::Args)]
struct CalibrateArgs {
    /// Folder containing dark .lri captures from one exposure and ISO.
    #[arg(long, value_name = "DIRECTORY")]
    input: PathBuf,

    /// One portable .chiaro-cleanup output file.
    #[arg(long, value_name = "FILE")]
    output: PathBuf,

    /// Factory hotpixel.rec belonging to this Light L16.
    #[arg(long)]
    hotpixel_rec: PathBuf,

    /// Search subdirectories under INPUT.
    #[arg(long)]
    recursive: bool,

    /// Calibrate only named cameras; repeat, for example: --camera B1 --camera C3.
    #[arg(long)]
    camera: Vec<String>,

    /// Override missing or incorrect sensor metadata, for example A2=MONO.
    #[arg(long, value_name = "CAMERA=PATTERN")]
    pattern: Vec<String>,

    /// Minimum factory severity included in the learned defect model.
    #[arg(long, default_value_t = 16)]
    severity_threshold: u8,

    /// Same-parity neighbors used to isolate lines and wider bands.
    #[arg(long, default_value_t = 32)]
    line_neighborhood_radius: usize,

    /// Use at most this many temperature-stratified frames per camera.
    #[arg(long)]
    max_frames_per_camera: Option<usize>,

    /// Replace an existing cleanup file.
    #[arg(long)]
    overwrite: bool,
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
    glow_profile: Option<String>,
    universal_hotpixel_profile: Option<String>,
    cleanup_profile: Option<String>,
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
    temperature_forced_corrected: usize,
    universal_hotpixel_applied: bool,
    universal_hotpixel_reason: Option<String>,
    universal_hotpixel_temperature_c: Option<f32>,
    universal_hotpixel_temperature_clamped: bool,
    universal_hotpixel_exposure_scale: Option<f32>,
    universal_hotpixel_analog_gain_scale: Option<f32>,
    universal_hotpixel_digital_gain_scale: Option<f32>,
    universal_hotpixel_active_pixels: usize,
    corrected_fraction: f64,
    mean_absolute_change: f64,
    maximum_absolute_change: u16,
    thermal_applied: bool,
    thermal_reason: Option<String>,
    sensor_temperature_c: Option<i32>,
    applied_temperature_c: Option<f32>,
    temperature_clamped: bool,
    exposure_scale: Option<f32>,
    mean_absolute_dark_change: f64,
    maximum_absolute_dark_change: u16,
    cleanup_applied: bool,
    cleanup_reason: Option<String>,
    cleanup_temperature_c: Option<f32>,
    cleanup_temperature_clamped: bool,
    cleanup_exposure_scale: Option<f32>,
    cleanup_analog_gain_scale: Option<f32>,
    cleanup_digital_gain_scale: Option<f32>,
    temperature_active_hot_pixels: usize,
    active_cleanup_rows: usize,
    active_cleanup_columns: usize,
    mean_absolute_cleanup_change: f64,
    maximum_absolute_cleanup_change: f64,
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
        if !entry.file_type().is_file() && !entry.path().is_file() {
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

fn prepare_output(args: &ExtractArgs) -> Result<()> {
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
    args: &ExtractArgs,
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
    args: &ExtractArgs,
    job: &Job,
    severity_map: &[u8],
    config: &CorrectionConfig,
    cleanup: Option<&CleanupCameraProfile>,
    universal_hotpixel: Option<&UniversalHotpixelProfile>,
    thermal: Option<&ThermalProfile>,
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
                temperature_forced_corrected: 0,
                universal_hotpixel_applied: false,
                universal_hotpixel_reason: Some("skipped existing output".to_owned()),
                universal_hotpixel_temperature_c: None,
                universal_hotpixel_temperature_clamped: false,
                universal_hotpixel_exposure_scale: None,
                universal_hotpixel_analog_gain_scale: None,
                universal_hotpixel_digital_gain_scale: None,
                universal_hotpixel_active_pixels: 0,
                corrected_fraction: 0.0,
                mean_absolute_change: 0.0,
                maximum_absolute_change: 0,
                thermal_applied: false,
                thermal_reason: Some("skipped existing output".to_owned()),
                sensor_temperature_c: job.camera.sensor_temperature_c,
                applied_temperature_c: None,
                temperature_clamped: false,
                exposure_scale: None,
                mean_absolute_dark_change: 0.0,
                maximum_absolute_dark_change: 0,
                cleanup_applied: false,
                cleanup_reason: Some("skipped existing output".to_owned()),
                cleanup_temperature_c: None,
                cleanup_temperature_clamped: false,
                cleanup_exposure_scale: None,
                cleanup_analog_gain_scale: None,
                cleanup_digital_gain_scale: None,
                temperature_active_hot_pixels: 0,
                active_cleanup_rows: 0,
                active_cleanup_columns: 0,
                mean_absolute_cleanup_change: 0.0,
                maximum_absolute_cleanup_change: 0.0,
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
    let mut cleanup_stats = CleanupCorrectionStats::default();
    let (mut forced_map, universal_hotpixel_stats) = if let Some(profile) = universal_hotpixel {
        let (active, stats) =
            profile.active_map(&job.camera, severity_map, config.absolute_threshold as f32);
        (stats.applied.then_some(active), stats)
    } else {
        (
            None,
            UniversalHotpixelStats {
                reason: Some("bundled universal hotpixel model disabled".to_owned()),
                requested_temperature_c: job.camera.sensor_temperature_c,
                ..UniversalHotpixelStats::default()
            },
        )
    };
    let (stats, mut output_samples) = if let Some(profile) = cleanup {
        let mut output_samples = raw.iter().map(|sample| sample << 6).collect::<Vec<_>>();
        cleanup_stats = profile.correct_q6(
            &job.camera,
            &mut output_samples,
            config.absolute_threshold as f32,
        );
        if let Ok(personal_map) =
            profile.temperature_active_map(&job.camera, config.absolute_threshold as f32)
        {
            let combined = forced_map.get_or_insert_with(|| vec![false; personal_map.len()]);
            for (active, personal) in combined.iter_mut().zip(personal_map) {
                *active |= personal;
            }
        }
        let mut q6_config = config.clone();
        q6_config.absolute_threshold = config.absolute_threshold.saturating_mul(64);
        let mut stats = correct_hot_pixels_with_forced_map(
            &mut output_samples,
            job.camera.width,
            job.camera.height,
            job.camera.pattern,
            severity_map,
            forced_map.as_deref(),
            &q6_config,
        )?;
        stats.mean_absolute_change /= 64.0;
        stats.maximum_absolute_change =
            ((u32::from(stats.maximum_absolute_change) + 32) / 64) as u16;
        (stats, output_samples)
    } else {
        let stats: CorrectionStats = correct_hot_pixels_with_forced_map(
            &mut raw,
            job.camera.width,
            job.camera.height,
            job.camera.pattern,
            severity_map,
            forced_map.as_deref(),
            config,
        )?;
        // Promote RAW10 to Q6 before glow subtraction. The original samples remain
        // exact multiples of 64, while the smooth model can retain sub-code detail.
        let output_samples = raw.iter().map(|sample| sample << 6).collect::<Vec<_>>();
        (stats, output_samples)
    };
    if cleanup.is_none() && args.cleanup_profile.is_some() {
        cleanup_stats.reason = Some("cleanup profile has no entry for this camera".to_owned());
    }
    let mut thermal_stats = ThermalCorrectionStats::default();
    if let Some(profile) = thermal {
        output_samples.reverse();
        thermal_stats = profile.correct_calibrated_plane_q6(&job.camera, &mut output_samples)?;
        output_samples.reverse();
    }

    let png_color_type = match args.mode {
        OutputMode::Rgb if job.camera.pattern != SensorPattern::Mono => {
            let rgb = demosaic_bilinear(
                &output_samples,
                job.camera.width,
                job.camera.height,
                job.camera.pattern,
            )?;
            write_rgb16_native_atomic(&output_path, job.camera.width, job.camera.height, &rgb)?;
            "RGB16"
        }
        _ => {
            write_gray16_native_atomic(
                &output_path,
                job.camera.width,
                job.camera.height,
                &output_samples,
            )?;
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
        temperature_forced_corrected: stats.forced_corrected,
        universal_hotpixel_applied: universal_hotpixel_stats.applied,
        universal_hotpixel_reason: universal_hotpixel_stats.reason,
        universal_hotpixel_temperature_c: universal_hotpixel_stats.applied_temperature_c,
        universal_hotpixel_temperature_clamped: universal_hotpixel_stats.temperature_clamped,
        universal_hotpixel_exposure_scale: universal_hotpixel_stats.exposure_scale,
        universal_hotpixel_analog_gain_scale: universal_hotpixel_stats.analog_gain_scale,
        universal_hotpixel_digital_gain_scale: universal_hotpixel_stats.digital_gain_scale,
        universal_hotpixel_active_pixels: universal_hotpixel_stats.active_pixels,
        corrected_fraction: stats.corrected as f64 / raw.len() as f64,
        mean_absolute_change: stats.mean_absolute_change,
        maximum_absolute_change: stats.maximum_absolute_change,
        thermal_applied: thermal_stats.applied,
        thermal_reason: thermal_stats.reason,
        sensor_temperature_c: job.camera.sensor_temperature_c,
        applied_temperature_c: thermal_stats.applied_temperature_c,
        temperature_clamped: thermal_stats.temperature_clamped,
        exposure_scale: thermal_stats.exposure_scale,
        mean_absolute_dark_change: thermal_stats.mean_absolute_dark_change,
        maximum_absolute_dark_change: thermal_stats.maximum_absolute_dark_change,
        cleanup_applied: cleanup_stats.applied,
        cleanup_reason: cleanup_stats.reason,
        cleanup_temperature_c: cleanup_stats.applied_temperature_c,
        cleanup_temperature_clamped: cleanup_stats.temperature_clamped,
        cleanup_exposure_scale: cleanup_stats.exposure_scale,
        cleanup_analog_gain_scale: cleanup_stats.analog_gain_scale,
        cleanup_digital_gain_scale: cleanup_stats.digital_gain_scale,
        temperature_active_hot_pixels: cleanup_stats.temperature_active_hot_pixels,
        active_cleanup_rows: cleanup_stats.active_rows,
        active_cleanup_columns: cleanup_stats.active_columns,
        mean_absolute_cleanup_change: cleanup_stats.mean_absolute_change,
        maximum_absolute_cleanup_change: cleanup_stats.maximum_absolute_change,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn write_root_notes(args: &ExtractArgs) -> Result<()> {
    let text = format!(
        "Light L16 astrophotography stacks\n\n\
Each camera subdirectory contains PNG frames only and can be selected directly in stacking software.\n\
The frames are linear 16-bit images. Device-specific factory hot/dead-pixel interpolation is applied.\n\
Universal A/B/C temperature/exposure/gain hot-pixel prior: {}.\n\
Universal temperature-conditioned sensor-glow subtraction: {}.\n\
Optional camera-specific temperature defect/line cleanup: {}.\n\
No white balance, color matrix, exposure normalization, gamma, stretch, sharpening,\n\
flat-field correction, spatial denoising, or alignment has been applied.\n\
Bayer output mode: {:?}. Bayer RGB output uses simple linear bilinear demosaicing.\n",
        if args.no_universal_hotpixel_model {
            "disabled".to_owned()
        } else {
            args.universal_hotpixel_profile.as_ref().map_or_else(
                || "bundled l16-universal-hotpixel-v1".to_owned(),
                |path| path.display().to_string(),
            )
        },
        if args.no_glow_correction {
            "disabled".to_owned()
        } else {
            args.glow_profile.as_ref().map_or_else(
                || "bundled universal model".to_owned(),
                |path| path.display().to_string(),
            )
        },
        args.cleanup_profile
            .as_ref()
            .map_or_else(|| "disabled".to_owned(), |path| path.display().to_string()),
        args.mode
    );
    fs::write(args.output.join("README.txt"), text)?;
    Ok(())
}

fn run_calibration(args: CalibrateArgs) -> Result<()> {
    if args.line_neighborhood_radius < 1 {
        bail!("--line-neighborhood-radius must be at least one");
    }
    let hotpixel = HotpixelRec::open(&args.hotpixel_rec)?;
    let options = BuildCleanupProfileOptions {
        input: args.input.clone(),
        output: args.output.clone(),
        recursive: args.recursive,
        selected_cameras: args
            .camera
            .iter()
            .map(|camera| camera.to_ascii_uppercase())
            .collect::<HashSet<_>>(),
        pattern_overrides: parse_pattern_overrides(&args.pattern)?,
        overwrite: args.overwrite,
        severity_threshold: args.severity_threshold,
        line_neighborhood_radius: args.line_neighborhood_radius,
        max_frames_per_camera: args.max_frames_per_camera,
    };
    let manifest = build_cleanup_profile(&options, &hotpixel, |message| println!("{message}"))?;
    println!(
        "Wrote one cleanup file with {} physical cameras to {}",
        manifest.cameras.len(),
        args.output.display()
    );
    Ok(())
}

fn run_extract(args: ExtractArgs) -> Result<()> {
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

    let universal_hotpixel = if args.no_universal_hotpixel_model {
        None
    } else if let Some(path) = args.universal_hotpixel_profile.as_ref() {
        Some(UniversalHotpixelProfile::open(path)?)
    } else {
        Some(UniversalHotpixelProfile::bundled()?)
    };
    if universal_hotpixel.is_some() {
        println!(
            "Universal factory-hotpixel response: {}",
            args.universal_hotpixel_profile.as_ref().map_or_else(
                || "bundled A/B/C model".to_owned(),
                |path| path.display().to_string()
            )
        );
    }

    let thermal = if args.no_glow_correction {
        None
    } else if let Some(path) = args.glow_profile.as_ref() {
        Some(ThermalProfile::open(path)?)
    } else {
        Some(ThermalProfile::bundled()?)
    };
    if let Some(profile) = thermal.as_ref() {
        println!(
            "Universal glow profile: {} contributing sensors from {}",
            profile.manifest.contributing_cameras.len(),
            profile.root.display()
        );
    }

    let cleanup = args
        .cleanup_profile
        .as_ref()
        .map(|path| CleanupProfile::open(path, &hotpixel))
        .transpose()?;
    if let Some(profile) = cleanup.as_ref() {
        println!(
            "Optional defect/line cleanup profile: {} cameras from {}",
            profile.manifest.cameras.len(),
            profile.root.display()
        );
    }

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
        let cleanup_camera = cleanup
            .as_ref()
            .map(|profile| profile.load_camera(&first.camera))
            .transpose()?
            .flatten();
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

            match process_frame(
                &args,
                job,
                &map,
                &config,
                cleanup_camera.as_ref(),
                universal_hotpixel.as_ref(),
                thermal.as_ref(),
            ) {
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
            png_scaling: "linear Q6 RAW codes; unmodified RAW10 samples are exact value << 6",
            color_processing: match args.mode {
                OutputMode::Rgb => "Bayer cameras: bilinear demosaic only; mono: grayscale",
                OutputMode::Mosaic => "corrected RAW mosaic/grayscale; no demosaic",
            },
            glow_profile: if args.no_glow_correction {
                None
            } else {
                Some(args.glow_profile.as_ref().map_or_else(
                    || "bundled:l16-glow-v2".to_owned(),
                    |path| path.to_string_lossy().to_string(),
                ))
            },
            universal_hotpixel_profile: if args.no_universal_hotpixel_model {
                None
            } else {
                Some(args.universal_hotpixel_profile.as_ref().map_or_else(
                    || "bundled:l16-universal-hotpixel-v1".to_owned(),
                    |path| path.to_string_lossy().to_string(),
                ))
            },
            cleanup_profile: args
                .cleanup_profile
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
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
    let result = match Cli::parse().command {
        Command::Extract(args) => run_extract(args),
        Command::Calibrate(args) => run_calibration(args),
    };
    if let Err(error) = result {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
