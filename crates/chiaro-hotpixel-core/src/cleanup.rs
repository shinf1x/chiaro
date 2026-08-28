use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

use chiaro::lri::{RawCamera, SensorPattern, parse_raw_layout};

use crate::hotpixel::HotpixelRec;
use crate::raw10::unpack_l16_10bit;
use crate::scan::{discover_lri_files, mmap_file};

const PROFILE_FORMAT: &str = "chiaro-defect-cleanup-profile";
const PROFILE_VERSION: u32 = 2;
const MANIFEST_NAME: &str = "manifest.json";
const DEFECT_ENTRY_BYTES: usize = 11;
const LINE_ENTRY_BYTES: usize = 6;
const DEFECT_REFERENCE_SCALE: f32 = 16.0;
const DEFECT_SLOPE_SCALE: f32 = 256.0;
const DEFECT_CURVATURE_SCALE: f32 = 4096.0;
const LINE_REFERENCE_SCALE: f32 = 256.0;
const LINE_SLOPE_SCALE: f32 = 4096.0;
const LINE_CURVATURE_SCALE: f32 = 4096.0;
const OUTPUT_Q6_SCALE: f32 = 64.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DefectClass {
    HotOnly = 1,
    HotOrDead = 2,
}

impl DefectClass {
    fn from_byte(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::HotOnly),
            2 => Ok(Self::HotOrDead),
            _ => bail!("unsupported learned defect class {value}"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CleanupProfileManifest {
    pub format: String,
    pub version: u32,
    pub source: String,
    pub hotpixel_sha256: String,
    pub severity_threshold: u8,
    pub line_neighborhood_radius: usize,
    pub cameras: Vec<CleanupCameraManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CleanupCameraManifest {
    pub camera: String,
    pub camera_index: usize,
    pub width: usize,
    pub height: usize,
    pub pattern: String,
    pub frame_count: usize,
    pub temperature_min_c: i32,
    pub temperature_max_c: i32,
    pub reference_temperature_c: f32,
    #[serde(default = "default_temperature_model")]
    pub temperature_model: String,
    pub exposure_ns: u64,
    pub analog_gain: f32,
    pub digital_gain: f32,
    pub defects: String,
    pub lines: String,
    pub defect_count: usize,
    pub hot_only_count: usize,
    pub hot_or_dead_count: usize,
    pub defect_layout: String,
    pub line_layout: String,
}

fn default_temperature_model() -> String {
    "quadratic".to_owned()
}

#[derive(Clone, Debug)]
struct DefectCoefficient {
    index: usize,
    reference: f32,
    slope: f32,
    curvature: f32,
    class: DefectClass,
}

#[derive(Clone, Debug)]
pub struct CleanupCameraProfile {
    pub manifest: CleanupCameraManifest,
    defects: Vec<DefectCoefficient>,
    row_reference: Vec<f32>,
    row_slope: Vec<f32>,
    row_curvature: Vec<f32>,
    column_reference: Vec<f32>,
    column_slope: Vec<f32>,
    column_curvature: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct CleanupProfile {
    pub root: PathBuf,
    pub manifest: CleanupProfileManifest,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CleanupCorrectionStats {
    pub applied: bool,
    pub reason: Option<String>,
    pub requested_temperature_c: Option<i32>,
    pub applied_temperature_c: Option<f32>,
    pub temperature_clamped: bool,
    pub exposure_scale: Option<f32>,
    pub analog_gain_scale: Option<f32>,
    pub digital_gain_scale: Option<f32>,
    pub temperature_active_hot_pixels: usize,
    pub active_rows: usize,
    pub active_columns: usize,
    pub mean_absolute_change: f64,
    pub maximum_absolute_change: f64,
}

#[derive(Clone, Copy, Debug)]
struct CleanupState {
    temperature: f32,
    temperature_clamped: bool,
    exposure_scale: f32,
    analog_gain_scale: f32,
    digital_gain_scale: f32,
    total_scale: f32,
}

#[derive(Clone, Debug)]
pub struct BuildCleanupProfileOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub recursive: bool,
    pub selected_cameras: HashSet<String>,
    pub pattern_overrides: HashMap<String, SensorPattern>,
    pub overwrite: bool,
    pub severity_threshold: u8,
    pub line_neighborhood_radius: usize,
    pub max_frames_per_camera: Option<usize>,
}

#[derive(Clone, Debug)]
struct CalibrationJob {
    source: PathBuf,
    camera: RawCamera,
}

impl CleanupProfile {
    pub fn open(path: impl AsRef<Path>, hotpixel: &HotpixelRec) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        let manifest_bytes = read_profile_entry(&root, MANIFEST_NAME)?;
        let manifest: CleanupProfileManifest = serde_json::from_slice(&manifest_bytes)
            .with_context(|| format!("parse cleanup manifest in {}", root.display()))?;
        if manifest.format != PROFILE_FORMAT || manifest.version != PROFILE_VERSION {
            bail!(
                "unsupported cleanup profile format/version: {}/{}",
                manifest.format,
                manifest.version
            );
        }
        if manifest.hotpixel_sha256 != hotpixel.sha256 {
            bail!(
                "cleanup profile was trained with factory map {}, but supplied hotpixel.rec is {}",
                manifest.hotpixel_sha256,
                hotpixel.sha256
            );
        }
        Ok(Self { root, manifest })
    }

    pub fn load_camera(&self, camera: &RawCamera) -> Result<Option<CleanupCameraProfile>> {
        let Some(manifest) = self
            .manifest
            .cameras
            .iter()
            .find(|entry| entry.camera.eq_ignore_ascii_case(&camera.name))
            .cloned()
        else {
            return Ok(None);
        };
        if manifest.camera_index != camera.id
            || manifest.width != camera.width
            || manifest.height != camera.height
            || manifest.pattern != camera.pattern.as_str()
        {
            bail!(
                "cleanup profile metadata does not match camera {}",
                camera.name
            );
        }

        let defect_bytes = read_profile_entry(&self.root, &manifest.defects)
            .with_context(|| format!("read cleanup defects for {}", camera.name))?;
        if defect_bytes.len() != manifest.defect_count * DEFECT_ENTRY_BYTES {
            bail!(
                "cleanup defect file has the wrong length for {}",
                camera.name
            );
        }
        let mut defects = Vec::with_capacity(manifest.defect_count);
        for chunk in defect_bytes.chunks_exact(DEFECT_ENTRY_BYTES) {
            let index = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as usize;
            if index >= camera.width * camera.height {
                bail!("cleanup defect index lies outside camera {}", camera.name);
            }
            defects.push(DefectCoefficient {
                index,
                reference: i16::from_le_bytes(chunk[4..6].try_into().unwrap()) as f32
                    / DEFECT_REFERENCE_SCALE,
                slope: i16::from_le_bytes(chunk[6..8].try_into().unwrap()) as f32
                    / DEFECT_SLOPE_SCALE,
                curvature: i16::from_le_bytes(chunk[8..10].try_into().unwrap()) as f32
                    / DEFECT_CURVATURE_SCALE,
                class: DefectClass::from_byte(chunk[10])?,
            });
        }

        let line_bytes = read_profile_entry(&self.root, &manifest.lines)
            .with_context(|| format!("read cleanup lines for {}", camera.name))?;
        let line_count = camera.height + camera.width;
        if line_bytes.len() != line_count * LINE_ENTRY_BYTES {
            bail!("cleanup line file has the wrong length for {}", camera.name);
        }
        let mut references = Vec::with_capacity(line_count);
        let mut slopes = Vec::with_capacity(line_count);
        let mut curvatures = Vec::with_capacity(line_count);
        for chunk in line_bytes.chunks_exact(LINE_ENTRY_BYTES) {
            references.push(
                i16::from_le_bytes(chunk[0..2].try_into().unwrap()) as f32 / LINE_REFERENCE_SCALE,
            );
            slopes.push(
                i16::from_le_bytes(chunk[2..4].try_into().unwrap()) as f32 / LINE_SLOPE_SCALE,
            );
            curvatures.push(
                i16::from_le_bytes(chunk[4..6].try_into().unwrap()) as f32 / LINE_CURVATURE_SCALE,
            );
        }
        let row_reference = references[..camera.height].to_vec();
        let column_reference = references[camera.height..].to_vec();
        let row_slope = slopes[..camera.height].to_vec();
        let column_slope = slopes[camera.height..].to_vec();
        let row_curvature = curvatures[..camera.height].to_vec();
        let column_curvature = curvatures[camera.height..].to_vec();
        Ok(Some(CleanupCameraProfile {
            manifest,
            defects,
            row_reference,
            row_slope,
            row_curvature,
            column_reference,
            column_slope,
            column_curvature,
        }))
    }
}

impl CleanupCameraProfile {
    fn state(&self, camera: &RawCamera) -> Result<CleanupState> {
        if camera.width != self.manifest.width || camera.height != self.manifest.height {
            bail!("capture dimensions do not match the cleanup profile");
        }
        if camera.exposure_ns == 0
            || camera.analog_gain <= 0.0
            || camera.digital_gain <= 0.0
            || self.manifest.exposure_ns == 0
            || self.manifest.analog_gain <= 0.0
            || self.manifest.digital_gain <= 0.0
        {
            bail!("capture or cleanup profile has incomplete exposure/gain metadata");
        }
        let requested = camera
            .sensor_temperature_c
            .context("capture has no sensor temperature for cleanup profile")?;
        let applied = (requested as f32).clamp(
            self.manifest.temperature_min_c as f32,
            self.manifest.temperature_max_c as f32,
        );
        let exposure_scale = camera.exposure_ns as f32 / self.manifest.exposure_ns as f32;
        let analog_gain_scale = camera.analog_gain / self.manifest.analog_gain;
        let digital_gain_scale = camera.digital_gain / self.manifest.digital_gain;
        Ok(CleanupState {
            temperature: applied,
            temperature_clamped: applied != requested as f32,
            exposure_scale,
            analog_gain_scale,
            digital_gain_scale,
            total_scale: exposure_scale * analog_gain_scale * digital_gain_scale,
        })
    }

    /// Build an opt-in activity map for factory-listed hot-only coordinates.
    /// An active entry forces local interpolation; it never supplies the
    /// replacement value itself.
    pub fn temperature_active_map(
        &self,
        camera: &RawCamera,
        activation_threshold: f32,
    ) -> Result<Vec<bool>> {
        let state = self.state(camera)?;
        let delta_temperature = state.temperature - self.manifest.reference_temperature_c;
        let mut active = vec![false; camera.width * camera.height];
        for defect in &self.defects {
            if defect.class != DefectClass::HotOnly {
                continue;
            }
            let predicted = (defect.reference
                + defect.slope * delta_temperature
                + defect.curvature * delta_temperature * delta_temperature)
                * state.total_scale;
            active[defect.index] = predicted >= activation_threshold;
        }
        Ok(active)
    }

    /// Apply learned row/column fixed-pattern correction to Q6 RAW samples.
    /// Hot pixels are repaired afterward, so interpolation happens in the
    /// corrected field and remains authoritative over the temperature model.
    pub fn correct_q6(
        &self,
        camera: &RawCamera,
        raw_q6: &mut [u16],
        hot_activation_threshold: f32,
    ) -> CleanupCorrectionStats {
        let requested_temperature = camera.sensor_temperature_c;
        let state = match self.state(camera) {
            Ok(state) => state,
            Err(error) => {
                return CleanupCorrectionStats {
                    reason: Some(error.to_string()),
                    requested_temperature_c: requested_temperature,
                    ..CleanupCorrectionStats::default()
                };
            }
        };
        if raw_q6.len() != camera.width * camera.height {
            return CleanupCorrectionStats {
                reason: Some("cleanup profile dimensions do not match RAW samples".to_owned()),
                requested_temperature_c: requested_temperature,
                ..CleanupCorrectionStats::default()
            };
        }

        let delta_temperature = state.temperature - self.manifest.reference_temperature_c;
        let mut total_change_q6 = 0u64;
        let mut maximum_change_q6 = 0u16;
        let temperature_active_hot_pixels = self
            .defects
            .iter()
            .filter(|defect| {
                defect.class == DefectClass::HotOnly
                    && (defect.reference
                        + defect.slope * delta_temperature
                        + defect.curvature * delta_temperature * delta_temperature)
                        * state.total_scale
                        >= hot_activation_threshold
            })
            .count();

        let row_values = self
            .row_reference
            .iter()
            .zip(&self.row_slope)
            .zip(&self.row_curvature)
            .map(|((&reference, &slope), &curvature)| {
                (reference
                    + slope * delta_temperature
                    + curvature * delta_temperature * delta_temperature)
                    * state.total_scale
            })
            .collect::<Vec<_>>();
        let column_values = self
            .column_reference
            .iter()
            .zip(&self.column_slope)
            .zip(&self.column_curvature)
            .map(|((&reference, &slope), &curvature)| {
                (reference
                    + slope * delta_temperature
                    + curvature * delta_temperature * delta_temperature)
                    * state.total_scale
            })
            .collect::<Vec<_>>();
        let active_rows = row_values
            .iter()
            .filter(|value| value.abs() >= 1.0 / 64.0)
            .count();
        let active_columns = column_values
            .iter()
            .filter(|value| value.abs() >= 1.0 / 64.0)
            .count();
        for (index, sample) in raw_q6.iter_mut().enumerate() {
            let y = index / camera.width;
            let x = index % camera.width;
            let correction_q6 = ((row_values[y] + column_values[x]) * OUTPUT_Q6_SCALE).round();
            let before = *sample;
            *sample = (f32::from(before) - correction_q6)
                .round()
                .clamp(0.0, 65535.0) as u16;
            let change = before.abs_diff(*sample);
            total_change_q6 += u64::from(change);
            maximum_change_q6 = maximum_change_q6.max(change);
        }

        CleanupCorrectionStats {
            applied: true,
            reason: None,
            requested_temperature_c: requested_temperature,
            applied_temperature_c: Some(state.temperature),
            temperature_clamped: state.temperature_clamped,
            exposure_scale: Some(state.exposure_scale),
            analog_gain_scale: Some(state.analog_gain_scale),
            digital_gain_scale: Some(state.digital_gain_scale),
            temperature_active_hot_pixels,
            active_rows,
            active_columns,
            mean_absolute_change: total_change_q6 as f64
                / raw_q6.len().max(1) as f64
                / f64::from(OUTPUT_Q6_SCALE),
            maximum_absolute_change: f64::from(maximum_change_q6) / f64::from(OUTPUT_Q6_SCALE),
        }
    }
}

pub fn build_cleanup_profile(
    options: &BuildCleanupProfileOptions,
    hotpixel: &HotpixelRec,
    mut on_progress: impl FnMut(&str),
) -> Result<CleanupProfileManifest> {
    if options.line_neighborhood_radius < 1 {
        bail!("line neighborhood radius must be at least one");
    }
    if options
        .max_frames_per_camera
        .is_some_and(|maximum| maximum < 3)
    {
        bail!("maximum frames per camera must be at least three");
    }
    prepare_output_file(&options.output, options.overwrite)?;
    let files = discover_lri_files(&options.input, options.recursive)?;
    let mut jobs = BTreeMap::<String, Vec<CalibrationJob>>::new();
    for path in files {
        let mmap = mmap_file(&path)?;
        let layout = parse_raw_layout(&mmap, &options.pattern_overrides)
            .with_context(|| format!("parse RAW layout in {}", path.display()))?;
        for camera in layout.cameras {
            if !options.selected_cameras.is_empty()
                && !options
                    .selected_cameras
                    .contains(&camera.name.to_ascii_uppercase())
            {
                continue;
            }
            jobs.entry(camera.name.clone())
                .or_default()
                .push(CalibrationJob {
                    source: path.clone(),
                    camera,
                });
        }
    }
    if jobs.is_empty() {
        bail!("no selected camera frames found for cleanup training");
    }
    if let Some(maximum) = options.max_frames_per_camera {
        for camera_jobs in jobs.values_mut() {
            *camera_jobs = temperature_stratified_subset(camera_jobs, maximum);
        }
    }

    let mut cameras = Vec::new();
    let mut coefficient_files = Vec::new();
    for (camera_name, camera_jobs) in jobs {
        on_progress(&format!(
            "{camera_name}: fitting defect and line cleanup from {} dark frames",
            camera_jobs.len()
        ));
        let (camera, defects, lines) =
            fit_camera(&camera_jobs, options, hotpixel, &mut on_progress)?;
        coefficient_files.push((camera.defects.clone(), defects));
        coefficient_files.push((camera.lines.clone(), lines));
        cameras.push(camera);
    }
    let manifest = CleanupProfileManifest {
        format: PROFILE_FORMAT.to_owned(),
        version: PROFILE_VERSION,
        source: options.input.to_string_lossy().to_string(),
        hotpixel_sha256: hotpixel.sha256.clone(),
        severity_threshold: options.severity_threshold,
        line_neighborhood_radius: options.line_neighborhood_radius,
        cameras,
    };
    write_profile_archive(&options.output, &manifest, &coefficient_files)?;
    Ok(manifest)
}

fn temperature_stratified_subset(jobs: &[CalibrationJob], maximum: usize) -> Vec<CalibrationJob> {
    if jobs.len() <= maximum {
        return jobs.to_vec();
    }
    let mut sorted = jobs.to_vec();
    sorted.sort_by(|left, right| {
        left.camera
            .sensor_temperature_c
            .cmp(&right.camera.sensor_temperature_c)
            .then_with(|| left.source.cmp(&right.source))
    });
    (0..maximum)
        .map(|index| {
            let source_index = index * (sorted.len() - 1) / (maximum - 1);
            sorted[source_index].clone()
        })
        .collect()
}

fn fit_camera(
    jobs: &[CalibrationJob],
    options: &BuildCleanupProfileOptions,
    hotpixel: &HotpixelRec,
    on_progress: &mut impl FnMut(&str),
) -> Result<(CleanupCameraManifest, Vec<u8>, Vec<u8>)> {
    let first = &jobs.first().context("empty cleanup camera group")?.camera;
    let count = first.width * first.height;
    let temperatures = jobs
        .iter()
        .map(|job| {
            validate_camera(first, &job.camera)?;
            job.camera
                .sensor_temperature_c
                .context("cleanup training frame has no sensor temperature")
        })
        .collect::<Result<Vec<_>>>()?;
    let temperature_min_c = *temperatures.iter().min().unwrap();
    let temperature_max_c = *temperatures.iter().max().unwrap();
    if temperature_min_c == temperature_max_c {
        bail!("cleanup training needs at least two sensor temperatures");
    }
    let reference_temperature_c = median_f32(
        temperatures
            .iter()
            .map(|temperature| *temperature as f32)
            .collect(),
    );
    let centered_temperatures = temperatures
        .iter()
        .map(|temperature| *temperature as f32 - reference_temperature_c)
        .collect::<Vec<_>>();
    let sum_t = centered_temperatures.iter().sum::<f32>();
    let sum_t2 = centered_temperatures
        .iter()
        .map(|temperature| temperature * temperature)
        .sum::<f32>();
    let sum_t3 = centered_temperatures
        .iter()
        .map(|temperature| temperature.powi(3))
        .sum::<f32>();
    let sum_t4 = centered_temperatures
        .iter()
        .map(|temperature| temperature.powi(4))
        .sum::<f32>();
    let unique_temperature_count = temperatures.iter().collect::<HashSet<_>>().len();
    let quadratic_inverse = if unique_temperature_count >= 3 {
        Some(
            invert_3x3([
                jobs.len() as f32,
                sum_t,
                sum_t2,
                sum_t,
                sum_t2,
                sum_t3,
                sum_t2,
                sum_t3,
                sum_t4,
            ])
            .context("cleanup quadratic temperature regression is singular")?,
        )
    } else {
        None
    };
    let linear_determinant = jobs.len() as f32 * sum_t2 - sum_t * sum_t;

    let mut reference = vec![0f32; count];
    let mut slope = vec![0f32; count];
    let mut curvature = vec![0f32; count];
    for (frame_index, (job, &temperature)) in jobs.iter().zip(&centered_temperatures).enumerate() {
        let mmap = mmap_file(&job.source)?;
        let start = job.camera.absolute_offset;
        let end = start + job.camera.byte_len;
        if end > mmap.len() {
            bail!("RAW span lies outside {}", job.source.display());
        }
        let raw = unpack_l16_10bit(&mmap[start..end], count)?;
        for (((sum, sum_ty), sum_t2y), value) in reference
            .iter_mut()
            .zip(&mut slope)
            .zip(&mut curvature)
            .zip(raw)
        {
            *sum += f32::from(value);
            *sum_ty += temperature * f32::from(value);
            *sum_t2y += temperature * temperature * f32::from(value);
        }
        if frame_index == 0 || (frame_index + 1) % 5 == 0 || frame_index + 1 == jobs.len() {
            on_progress(&format!(
                "{}: {}/{} frames accumulated",
                first.name,
                frame_index + 1,
                jobs.len()
            ));
        }
    }
    for ((sum, sum_ty), sum_t2y) in reference.iter_mut().zip(&mut slope).zip(&mut curvature) {
        let sum_y = *sum;
        let sum_ty_observed = *sum_ty;
        if let Some(inverse) = quadratic_inverse {
            let observations = [sum_y, sum_ty_observed, *sum_t2y];
            *sum = inverse[0] * observations[0]
                + inverse[1] * observations[1]
                + inverse[2] * observations[2];
            *sum_ty = inverse[3] * observations[0]
                + inverse[4] * observations[1]
                + inverse[5] * observations[2];
            *sum_t2y = inverse[6] * observations[0]
                + inverse[7] * observations[1]
                + inverse[8] * observations[2];
        } else {
            *sum = (sum_t2 * sum_y - sum_t * sum_ty_observed) / linear_determinant;
            *sum_ty = (-sum_t * sum_y + jobs.len() as f32 * sum_ty_observed) / linear_determinant;
            *sum_t2y = 0.0;
        }
    }

    let factory_map = hotpixel.load_rotated_map(first.id, first.width, first.height)?;
    let excluded = factory_map
        .iter()
        .map(|severity| *severity >= options.severity_threshold || *severity == 255)
        .collect::<Vec<_>>();
    let (row_reference, column_reference) = fit_lines(
        &reference,
        &excluded,
        first.width,
        first.height,
        first.pattern,
        options.line_neighborhood_radius,
    );
    let (row_slope, column_slope) = fit_lines(
        &slope,
        &excluded,
        first.width,
        first.height,
        first.pattern,
        options.line_neighborhood_radius,
    );
    let (row_curvature, column_curvature) = fit_lines(
        &curvature,
        &excluded,
        first.width,
        first.height,
        first.pattern,
        options.line_neighborhood_radius,
    );

    let mut defects = Vec::new();
    let mut hot_only_count = 0usize;
    let mut hot_or_dead_count = 0usize;
    for (index, &severity) in factory_map.iter().enumerate() {
        if severity < options.severity_threshold && severity != 255 {
            continue;
        }
        let x = index % first.width;
        let y = index / first.width;
        let local_reference = local_prediction(
            &reference,
            &row_reference,
            &column_reference,
            first.width,
            first.height,
            x,
            y,
            first.pattern,
        );
        let local_slope = local_prediction(
            &slope,
            &row_slope,
            &column_slope,
            first.width,
            first.height,
            x,
            y,
            first.pattern,
        );
        let local_curvature = local_prediction(
            &curvature,
            &row_curvature,
            &column_curvature,
            first.width,
            first.height,
            x,
            y,
            first.pattern,
        );
        let fitted_reference = reference[index] - row_reference[y] - column_reference[x];
        let fitted_slope = slope[index] - row_slope[y] - column_slope[x];
        let class = if severity == 255 {
            hot_or_dead_count += 1;
            DefectClass::HotOrDead
        } else {
            hot_only_count += 1;
            DefectClass::HotOnly
        };
        defects.push(DefectCoefficient {
            index,
            reference: fitted_reference - local_reference,
            slope: fitted_slope - local_slope,
            curvature: curvature[index] - row_curvature[y] - column_curvature[x] - local_curvature,
            class,
        });
    }

    let defect_name = format!("{}.defects", first.name);
    let line_name = format!("{}.lines", first.name);
    let defect_bytes = encode_defects(&defects)?;
    let line_bytes = encode_lines(
        &row_reference,
        &row_slope,
        &row_curvature,
        &column_reference,
        &column_slope,
        &column_curvature,
    )?;

    let manifest = CleanupCameraManifest {
        camera: first.name.clone(),
        camera_index: first.id,
        width: first.width,
        height: first.height,
        pattern: first.pattern.as_str().to_owned(),
        frame_count: jobs.len(),
        temperature_min_c,
        temperature_max_c,
        reference_temperature_c,
        temperature_model: if unique_temperature_count >= 3 {
            "quadratic".to_owned()
        } else {
            "linear".to_owned()
        },
        exposure_ns: first.exposure_ns,
        analog_gain: first.analog_gain,
        digital_gain: first.digital_gain,
        defects: defect_name,
        lines: line_name,
        defect_count: defects.len(),
        hot_only_count,
        hot_or_dead_count,
        defect_layout:
            "sorted entries: u32 decoded-RAW index, i16 excess@reference Q4, i16 slope-per-C Q8, i16 curvature-per-C2 Q12, u8 class; little-endian; factory-gated"
                .to_owned(),
        line_layout:
            "all rows then all columns: i16 additive offset@reference Q8, i16 slope-per-C Q12, i16 curvature-per-C2 Q12; little-endian"
                .to_owned(),
    };
    Ok((manifest, defect_bytes, line_bytes))
}

fn fit_lines(
    field: &[f32],
    excluded: &[bool],
    width: usize,
    height: usize,
    pattern: SensorPattern,
    radius: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut row_means = vec![0f32; height];
    for (y, row_mean) in row_means.iter_mut().enumerate() {
        let mut sum = 0f64;
        let mut count = 0usize;
        for x in 0..width {
            let index = y * width + x;
            if !excluded[index] {
                sum += f64::from(field[index]);
                count += 1;
            }
        }
        *row_mean = (sum / count.max(1) as f64) as f32;
    }
    let row_offsets = high_pass_lines(&row_means, pattern, radius);

    let mut column_means = vec![0f32; width];
    for (x, column_mean) in column_means.iter_mut().enumerate() {
        let mut sum = 0f64;
        let mut count = 0usize;
        for (y, &row_offset) in row_offsets.iter().enumerate() {
            let index = y * width + x;
            if !excluded[index] {
                sum += f64::from(field[index] - row_offset);
                count += 1;
            }
        }
        *column_mean = (sum / count.max(1) as f64) as f32;
    }
    let column_offsets = high_pass_lines(&column_means, pattern, radius);
    (row_offsets, column_offsets)
}

fn high_pass_lines(values: &[f32], pattern: SensorPattern, radius: usize) -> Vec<f32> {
    let step = if pattern == SensorPattern::Mono { 1 } else { 2 };
    let mut output = vec![0f32; values.len()];
    for index in 0..values.len() {
        let mut neighbors = Vec::with_capacity(radius * 2);
        for distance in 1..=radius {
            neighbors.push(
                values[reflect_index(index as isize - (distance * step) as isize, values.len())],
            );
            neighbors.push(
                values[reflect_index(index as isize + (distance * step) as isize, values.len())],
            );
        }
        output[index] = values[index] - median_f32(neighbors);
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn local_prediction(
    field: &[f32],
    row_offsets: &[f32],
    column_offsets: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    pattern: SensorPattern,
) -> f32 {
    let step = if pattern == SensorPattern::Mono { 1 } else { 2 };
    let mut values = Vec::with_capacity(24);
    for dy in -2isize..=2 {
        for dx in -2isize..=2 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let source_x = reflect_index(x as isize + dx * step as isize, width);
            let source_y = reflect_index(y as isize + dy * step as isize, height);
            values.push(
                field[source_y * width + source_x]
                    - row_offsets[source_y]
                    - column_offsets[source_x],
            );
        }
    }
    median_f32(values)
}

fn encode_defects(defects: &[DefectCoefficient]) -> Result<Vec<u8>> {
    let mut writer = Vec::with_capacity(defects.len() * DEFECT_ENTRY_BYTES);
    for defect in defects {
        writer.write_all(&(defect.index as u32).to_le_bytes())?;
        writer.write_all(&quantize_i16(defect.reference, DEFECT_REFERENCE_SCALE).to_le_bytes())?;
        writer.write_all(&quantize_i16(defect.slope, DEFECT_SLOPE_SCALE).to_le_bytes())?;
        writer.write_all(&quantize_i16(defect.curvature, DEFECT_CURVATURE_SCALE).to_le_bytes())?;
        writer.write_all(&[defect.class as u8])?;
    }
    Ok(writer)
}

fn encode_lines(
    row_reference: &[f32],
    row_slope: &[f32],
    row_curvature: &[f32],
    column_reference: &[f32],
    column_slope: &[f32],
    column_curvature: &[f32],
) -> Result<Vec<u8>> {
    let mut writer =
        Vec::with_capacity((row_reference.len() + column_reference.len()) * LINE_ENTRY_BYTES);
    for ((&reference, &slope), &curvature) in row_reference
        .iter()
        .zip(row_slope)
        .zip(row_curvature)
        .chain(
            column_reference
                .iter()
                .zip(column_slope)
                .zip(column_curvature),
        )
    {
        writer.write_all(&quantize_i16(reference, LINE_REFERENCE_SCALE).to_le_bytes())?;
        writer.write_all(&quantize_i16(slope, LINE_SLOPE_SCALE).to_le_bytes())?;
        writer.write_all(&quantize_i16(curvature, LINE_CURVATURE_SCALE).to_le_bytes())?;
    }
    Ok(writer)
}

fn quantize_i16(value: f32, scale: f32) -> i16 {
    (value * scale)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn invert_3x3(matrix: [f32; 9]) -> Option<[f32; 9]> {
    let [a, b, c, d, e, f, g, h, i] = matrix;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if determinant.abs() < 1e-6 {
        return None;
    }
    let scale = 1.0 / determinant;
    Some([
        (e * i - f * h) * scale,
        (c * h - b * i) * scale,
        (b * f - c * e) * scale,
        (f * g - d * i) * scale,
        (a * i - c * g) * scale,
        (c * d - a * f) * scale,
        (d * h - e * g) * scale,
        (b * g - a * h) * scale,
        (a * e - b * d) * scale,
    ])
}

fn validate_camera(expected: &RawCamera, actual: &RawCamera) -> Result<()> {
    if expected.id != actual.id
        || expected.name != actual.name
        || expected.width != actual.width
        || expected.height != actual.height
        || expected.pattern != actual.pattern
        || expected.exposure_ns != actual.exposure_ns
        || !approximately_equal(expected.analog_gain, actual.analog_gain)
        || !approximately_equal(expected.digital_gain, actual.digital_gain)
    {
        bail!("camera settings change inside cleanup training set");
    }
    Ok(())
}

fn approximately_equal(left: f32, right: f32) -> bool {
    (left - right).abs() <= 0.001 * left.abs().max(right.abs()).max(1.0)
}

fn reflect_index(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let length = length as isize;
    while index < 0 || index >= length {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * length - 2 - index;
        }
    }
    index as usize
}

fn median_f32(mut values: Vec<f32>) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn prepare_output_file(path: &Path, overwrite: bool) -> Result<()> {
    if path.exists() {
        if path.is_dir() {
            bail!(
                "cleanup output must be a file, but {} is a directory",
                path.display()
            );
        }
        if overwrite {
            fs::remove_file(path).with_context(|| format!("remove existing {}", path.display()))?;
        } else {
            bail!(
                "cleanup profile output exists: {}; pass --overwrite",
                path.display()
            );
        }
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    Ok(())
}

fn read_profile_entry(path: &Path, name: &str) -> Result<Vec<u8>> {
    if path.is_dir() {
        return fs::read(path.join(name))
            .with_context(|| format!("read {} from {}", name, path.display()));
    }
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut archive = ZipArchive::new(file)
        .with_context(|| format!("open cleanup profile {}", path.display()))?;
    let mut entry = archive
        .by_name(name)
        .with_context(|| format!("read {name} from {}", path.display()))?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn write_profile_archive(
    path: &Path,
    manifest: &CleanupProfileManifest,
    files: &[(String, Vec<u8>)],
) -> Result<()> {
    let file_name = path
        .file_name()
        .context("cleanup profile output has no filename")?
        .to_string_lossy();
    let temporary = path.with_file_name(format!("{file_name}.tmp"));
    let file =
        File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    let mut archive = ZipWriter::new(BufWriter::new(file));
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o644);
    archive.start_file(MANIFEST_NAME, options)?;
    archive.write_all(&serde_json::to_vec_pretty(manifest)?)?;
    for (name, contents) in files {
        archive.start_file(name, options)?;
        archive.write_all(contents)?;
    }
    archive.finish()?.flush()?;
    fs::rename(&temporary, path)
        .with_context(|| format!("rename {} to {}", temporary.display(), path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_state_scales_exposure_and_both_gains() {
        let profile = CleanupCameraProfile {
            manifest: CleanupCameraManifest {
                camera: "A3".to_owned(),
                camera_index: 2,
                width: 4,
                height: 2,
                pattern: "GRBG".to_owned(),
                frame_count: 3,
                temperature_min_c: 20,
                temperature_max_c: 50,
                reference_temperature_c: 35.0,
                temperature_model: "quadratic".to_owned(),
                exposure_ns: 15_000_000_000,
                analog_gain: 6.25,
                digital_gain: 1.015625,
                defects: "A3.defects".to_owned(),
                lines: "A3.lines".to_owned(),
                defect_count: 0,
                hot_only_count: 0,
                hot_or_dead_count: 0,
                defect_layout: String::new(),
                line_layout: String::new(),
            },
            defects: Vec::new(),
            row_reference: vec![0.0; 2],
            row_slope: vec![0.0; 2],
            row_curvature: vec![0.0; 2],
            column_reference: vec![0.0; 4],
            column_slope: vec![0.0; 4],
            column_curvature: vec![0.0; 4],
        };
        let camera = RawCamera {
            id: 2,
            name: "A3".to_owned(),
            width: 4,
            height: 2,
            row_stride: 0,
            absolute_offset: 0,
            byte_len: 0,
            pattern: SensorPattern::Grbg,
            sensor_temperature_c: Some(40),
            analog_gain: 3.125,
            digital_gain: 2.03125,
            exposure_ns: 10_000_000_000,
            black_level: 0.0,
            white_level: 1023.0,
        };
        let state = profile.state(&camera).unwrap();
        assert!((state.exposure_scale - 2.0 / 3.0).abs() < 1e-6);
        assert!((state.analog_gain_scale - 0.5).abs() < 1e-6);
        assert!((state.digital_gain_scale - 2.0).abs() < 1e-6);
        assert!((state.total_scale - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn single_file_profile_archive_round_trips_entries() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("camera.chiaro-cleanup");
        let manifest = CleanupProfileManifest {
            format: PROFILE_FORMAT.to_owned(),
            version: PROFILE_VERSION,
            source: "test".to_owned(),
            hotpixel_sha256: "factory".to_owned(),
            severity_threshold: 16,
            line_neighborhood_radius: 32,
            cameras: Vec::new(),
        };
        let files = vec![("A1.defects".to_owned(), vec![1, 2, 3, 4])];
        write_profile_archive(&path, &manifest, &files).unwrap();

        let decoded: CleanupProfileManifest =
            serde_json::from_slice(&read_profile_entry(&path, MANIFEST_NAME).unwrap()).unwrap();
        assert_eq!(decoded.format, PROFILE_FORMAT);
        assert_eq!(
            read_profile_entry(&path, "A1.defects").unwrap(),
            [1, 2, 3, 4]
        );
    }

    #[test]
    fn calibration_never_overwrites_a_directory_as_a_file() {
        let directory = tempfile::tempdir().unwrap();
        let error = prepare_output_file(directory.path(), true).unwrap_err();
        assert!(error.to_string().contains("is a directory"));
    }

    #[test]
    fn line_high_pass_preserves_parity_and_finds_one_bad_row() {
        let mut values = (0..12).map(|index| index as f32 * 0.1).collect::<Vec<_>>();
        values[6] += 3.0;
        let offsets = high_pass_lines(&values, SensorPattern::Rggb, 2);
        assert!(offsets[6] > 2.5);
        assert!(offsets[5].abs() < 0.5);
        assert!(offsets[7].abs() < 0.5);
    }

    #[test]
    fn quantization_round_trips_cleanup_precision() {
        let value = 1.25;
        let encoded = quantize_i16(value, LINE_REFERENCE_SCALE);
        assert_eq!(encoded as f32 / LINE_REFERENCE_SCALE, value);
    }

    #[test]
    fn quadratic_regression_inverse_recovers_known_coefficients() {
        let temperatures = [-3.0f32, -1.0, 0.0, 2.0, 4.0];
        let matrix = [
            temperatures.len() as f32,
            temperatures.iter().sum(),
            temperatures.iter().map(|value| value.powi(2)).sum(),
            temperatures.iter().sum(),
            temperatures.iter().map(|value| value.powi(2)).sum(),
            temperatures.iter().map(|value| value.powi(3)).sum(),
            temperatures.iter().map(|value| value.powi(2)).sum(),
            temperatures.iter().map(|value| value.powi(3)).sum(),
            temperatures.iter().map(|value| value.powi(4)).sum(),
        ];
        let inverse = invert_3x3(matrix).unwrap();
        let observations = [
            temperatures
                .iter()
                .map(|temperature| 7.0 + 1.5 * temperature - 0.25 * temperature.powi(2))
                .sum::<f32>(),
            temperatures
                .iter()
                .map(|temperature| {
                    temperature * (7.0 + 1.5 * temperature - 0.25 * temperature.powi(2))
                })
                .sum::<f32>(),
            temperatures
                .iter()
                .map(|temperature| {
                    temperature.powi(2) * (7.0 + 1.5 * temperature - 0.25 * temperature.powi(2))
                })
                .sum::<f32>(),
        ];
        let coefficients = [
            inverse[0] * observations[0]
                + inverse[1] * observations[1]
                + inverse[2] * observations[2],
            inverse[3] * observations[0]
                + inverse[4] * observations[1]
                + inverse[5] * observations[2],
            inverse[6] * observations[0]
                + inverse[7] * observations[1]
                + inverse[8] * observations[2],
        ];
        assert!((coefficients[0] - 7.0).abs() < 1e-4);
        assert!((coefficients[1] - 1.5).abs() < 1e-4);
        assert!((coefficients[2] + 0.25).abs() < 1e-4);
    }
}
