//! Hot-pixel corrected 16-bit PNG stacks, powered by `chiaro-hotpixel-core`.
//!
//! The options dialog, size estimate, and job runner live here; the actual
//! correction is the same `FramePipeline` the `chiaro-hotpixel` CLI uses, so
//! both front ends produce identical frames.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    thread::{self, JoinHandle},
};

use chiaro::lri::{RawCamera, parse_raw_layout};
use chiaro_hotpixel_core::{
    cleanup::CleanupProfile,
    correct::CorrectionConfig,
    hotpixel::HotpixelRec,
    pipeline::{CleanupStage, FramePipeline, OutputMode},
    scan::mmap_file,
    thermal::ThermalProfile,
    universal_hotpixel::UniversalHotpixelProfile,
};
use eframe::egui::{self, Color32, RichText};

use super::{
    CalibrationPath, ExportEstimate, ExportMonitor, ExportPipeline, ExportTarget, ExportUiServices,
    PathRow, PickField, PickKind, calibration_status, path_row,
};
use crate::source::{CaptureData, CaptureLocator, read_capture_with_updates};

const FIELD_OUTPUT: PickField = 1;
const FIELD_HOTPIXEL_REC: PickField = 2;
const FIELD_CLEANUP: PickField = 3;

/// Typical Light L16 capture used when a remote capture cannot be probed.
const ASSUMED_FRAMES: u64 = 10;
const ASSUMED_WIDTH: u64 = 4160;
const ASSUMED_HEIGHT: u64 = 3120;
/// Deflate level for the "faster, larger files" option. Measured on real
/// frames: ~20% faster end to end, files ~50% larger than the default level.
const FAST_DEFLATE_LEVEL: u32 = 1;

#[derive(Clone, Debug)]
pub struct HotpixelExport {
    output_dir: String,
    hotpixel_rec: CalibrationPath,
    cleanup_profile: String,
    output_mode: OutputMode,
    universal_model: bool,
    glow_correction: bool,
    skip_existing: bool,
    /// Trade file size for speed with a lighter deflate level.
    fast_compression: bool,
}

impl Default for HotpixelExport {
    fn default() -> Self {
        let output_dir = std::env::home_dir()
            .map(|home| home.join("chiaro-frames"))
            .map_or_else(String::new, |path| path.display().to_string());
        Self {
            output_dir,
            hotpixel_rec: CalibrationPath::default(),
            cleanup_profile: String::new(),
            output_mode: OutputMode::Rgb,
            universal_model: true,
            glow_correction: true,
            skip_existing: false,
            fast_compression: false,
        }
    }
}

impl HotpixelExport {
    fn bytes_per_sample(&self, mono: bool) -> u64 {
        if self.output_mode == OutputMode::Rgb && !mono {
            6
        } else {
            2
        }
    }

    fn deflate_level(&self) -> u32 {
        if self.fast_compression {
            FAST_DEFLATE_LEVEL
        } else {
            chiaro_hotpixel_core::png16::DEFAULT_DEFLATE_LEVEL
        }
    }
}

impl ExportPipeline for HotpixelExport {
    fn clone_box(&self) -> Box<dyn ExportPipeline> {
        Box::new(self.clone())
    }

    fn label(&self) -> &'static str {
        "Hot-pixel corrected frames"
    }

    fn description(&self) -> &'static str {
        "One folder of linear 16-bit PNGs per camera, ready for stacking"
    }

    fn options_ui(&mut self, ui: &mut egui::Ui, services: &mut ExportUiServices<'_>) {
        self.hotpixel_rec
            .adopt(services.device_calibration, "hotpixel.rec");
        let muted = Color32::from_gray(150);

        ui.label(RichText::new("Destination").strong());
        path_row(
            ui,
            services,
            &mut self.output_dir,
            PathRow {
                field: FIELD_OUTPUT,
                hint: "Folder that will receive one sub-folder per camera",
                title: "Choose the export folder",
                kind: PickKind::Folder,
                clearable: false,
            },
        );
        ui.add_space(8.0);

        ui.label(RichText::new("Factory calibration").strong());
        ui.label(
            RichText::new(
                "hotpixel.rec from this Light L16. It maps the defects of these exact sensors.",
            )
            .color(muted)
            .size(12.0),
        );
        path_row(
            ui,
            services,
            &mut self.hotpixel_rec.value,
            PathRow {
                field: FIELD_HOTPIXEL_REC,
                hint: "Path to hotpixel.rec",
                title: "Choose the camera's hotpixel.rec",
                kind: PickKind::File {
                    filter: Some(("Light hotpixel record", &["rec"])),
                },
                clearable: false,
            },
        );
        calibration_status(ui, services, &self.hotpixel_rec, "hotpixel.rec");
        ui.add_space(8.0);

        ui.label(RichText::new("Output format").strong());
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut self.output_mode,
                OutputMode::Rgb,
                "RGB (demosaiced Bayer, stacker-friendly)",
            );
            ui.radio_value(
                &mut self.output_mode,
                OutputMode::Mosaic,
                "Bayer mosaic / grayscale (no demosaic)",
            );
        });
        ui.add_space(8.0);

        ui.label(RichText::new("Corrections").strong());
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.universal_model, "Universal hot-pixel model")
                .on_hover_text(
                    "Bundled temperature/exposure/gain prior fitted from A/B/C sensors. \
                     Factory-listed pixels predicted to be active are always repaired.",
                );
            ui.checkbox(&mut self.glow_correction, "Corner-glow correction")
                .on_hover_text(
                    "Subtract the bundled low-frequency sensor-glow model when the exposure \
                     and temperature metadata fall inside its validated range.",
                );
        });
        ui.add_space(4.0);
        ui.label(
            RichText::new("Optional camera-specific cleanup profile (.chiaro-cleanup)")
                .color(muted)
                .size(12.0),
        );
        path_row(
            ui,
            services,
            &mut self.cleanup_profile,
            PathRow {
                field: FIELD_CLEANUP,
                hint: "Leave empty unless you trained one with chiaro-hotpixel calibrate",
                title: "Choose a cleanup profile",
                kind: PickKind::File {
                    filter: Some(("Chiaro cleanup profile", &["chiaro-cleanup"])),
                },
                clearable: true,
            },
        );
        ui.add_space(6.0);
        ui.checkbox(
            &mut self.skip_existing,
            "Skip frames whose PNG already exists in the destination",
        );
        ui.checkbox(
            &mut self.fast_compression,
            "Faster export with larger files (lighter PNG compression)",
        )
        .on_hover_text(
            "Default compression is the measured sweet spot (about 33 MB per RGB frame).              This option is roughly 20% faster and writes files about 50% larger.",
        );
    }

    fn apply_pick(&mut self, field: PickField, path: PathBuf) {
        let value = path.display().to_string();
        match field {
            FIELD_OUTPUT => self.output_dir = value,
            FIELD_HOTPIXEL_REC => self.hotpixel_rec.set_manual(path),
            FIELD_CLEANUP => self.cleanup_profile = value,
            _ => {}
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.output_dir.trim().is_empty() {
            return Err("Choose a destination folder".to_owned());
        }
        let output = Path::new(self.output_dir.trim());
        if output.is_file() {
            return Err("The destination is an existing file".to_owned());
        }
        let Some(rec) = self.hotpixel_rec.path() else {
            return Err("Choose the camera's hotpixel.rec".to_owned());
        };
        if !rec.is_file() {
            return Err("hotpixel.rec was not found at that path".to_owned());
        }
        if !self.cleanup_profile.trim().is_empty()
            && !Path::new(self.cleanup_profile.trim()).exists()
        {
            return Err("The cleanup profile was not found at that path".to_owned());
        }
        Ok(())
    }

    fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.output_dir.trim())
    }

    fn estimate(&self, targets: &[ExportTarget]) -> ExportEstimate {
        let mut bytes = 0u64;
        let mut frames = 0u64;
        let mut approximate = false;
        for target in targets {
            match &target.frames {
                Some(known) => {
                    for frame in known {
                        bytes +=
                            (frame.width * frame.height) as u64 * self.bytes_per_sample(frame.mono);
                        frames += 1;
                    }
                }
                None => {
                    approximate = true;
                    bytes += ASSUMED_FRAMES
                        * ASSUMED_WIDTH
                        * ASSUMED_HEIGHT
                        * self.bytes_per_sample(false);
                    frames += ASSUMED_FRAMES;
                }
            }
        }
        let detail = if approximate {
            format!(
                "about {frames} frames; camera captures are assumed to hold {ASSUMED_FRAMES} \
                 full-size modules until downloaded"
            )
        } else {
            format!("{frames} frames, uncompressed upper bound; PNG compression usually needs less")
        };
        ExportEstimate {
            bytes,
            approximate,
            detail,
        }
    }

    fn start(&self, targets: Vec<ExportTarget>, monitor: ExportMonitor) -> JoinHandle<()> {
        let options = self.clone();
        thread::Builder::new()
            .name("chiaro-export-hotpixel".to_owned())
            .spawn(move || {
                let outcome = run_job(&options, &targets, &monitor);
                monitor.hold_transport(false);
                monitor.update(|progress| {
                    if let Err(error) = outcome {
                        progress.fatal = Some(error);
                    }
                    progress.finished = true;
                    progress.current.clear();
                });
            })
            .expect("failed to start hot-pixel export")
    }
}

struct LoadedModels {
    rec: HotpixelRec,
    universal: Option<UniversalHotpixelProfile>,
    thermal: Option<ThermalProfile>,
    cleanup: Option<CleanupProfile>,
    config: CorrectionConfig,
}

fn run_job(
    options: &HotpixelExport,
    targets: &[ExportTarget],
    monitor: &ExportMonitor,
) -> Result<(), String> {
    let output_root = options.output_dir();
    fs::create_dir_all(&output_root)
        .map_err(|error| format!("create {}: {error}", output_root.display()))?;
    monitor.update(|progress| progress.current = "Loading calibration models".to_owned());
    let rec = HotpixelRec::open(options.hotpixel_rec.value.trim())
        .map_err(|error| format!("hotpixel.rec: {error:#}"))?;
    let models = LoadedModels {
        universal: options
            .universal_model
            .then(UniversalHotpixelProfile::bundled)
            .transpose()
            .map_err(|error| format!("{error:#}"))?,
        thermal: options
            .glow_correction
            .then(ThermalProfile::bundled)
            .transpose()
            .map_err(|error| format!("{error:#}"))?,
        cleanup: (!options.cleanup_profile.trim().is_empty())
            .then(|| CleanupProfile::open(options.cleanup_profile.trim(), &rec))
            .transpose()
            .map_err(|error| format!("cleanup profile: {error:#}"))?,
        rec,
        config: CorrectionConfig::default(),
    };

    let mut log = Vec::<String>::new();
    // Camera captures take 10-15 s to copy over USB; fetch the next one while
    // the current one is processed. At most one capture is in flight ahead,
    // so memory stays at two captures plus one frame.
    let mut prefetch = Prefetch::start(targets.first(), monitor);
    for (index, target) in targets.iter().enumerate() {
        if monitor.cancelled() {
            break;
        }
        monitor.update(|progress| {
            progress.current = target.name.clone();
            progress.current_fraction = 0.0;
        });
        let data = prefetch.take(target, monitor);
        prefetch = Prefetch::start(targets.get(index + 1), monitor);
        let result = data.and_then(|data| {
            export_capture(options, &models, target, &data, &output_root, monitor)
        });
        match result {
            Ok(written) => {
                log.push(format!("{}: {written} frames written", target.name));
                monitor.update(|progress| {
                    progress.outputs_written += written;
                    progress.completed += 1;
                    if let Some(identity) = &target.identity {
                        progress.succeeded_hashes.push(identity.hash.clone());
                    }
                });
            }
            Err(error) => {
                log.push(format!("{}: FAILED: {error}", target.name));
                monitor.update(|progress| {
                    progress.failures.push((target.name.clone(), error));
                    progress.completed += 1;
                });
            }
        }
    }
    let cancelled = monitor.cancelled();
    let notes = format!(
        "Chiaro Gallery hot-pixel export{}\n\
         hotpixel.rec: {} ({})\n\
         output: {:?}; universal model: {}; glow correction: {}; cleanup profile: {}\n\n{}\n",
        if cancelled { " (cancelled)" } else { "" },
        options.hotpixel_rec.value.trim(),
        models.rec.sha256,
        options.output_mode,
        options.universal_model,
        options.glow_correction,
        if options.cleanup_profile.trim().is_empty() {
            "none"
        } else {
            options.cleanup_profile.trim()
        },
        log.join("\n")
    );
    let _ = fs::write(output_root.join("export-log.txt"), notes);
    Ok(())
}

/// A capture being read ahead of its turn.
enum Prefetch {
    None,
    Running(std::thread::JoinHandle<Result<CaptureData, String>>),
}

impl Prefetch {
    fn start(target: Option<&ExportTarget>, monitor: &ExportMonitor) -> Self {
        let Some(target) = target else {
            return Self::None;
        };
        let locator = target.capture.clone();
        let name = target.name.clone();
        let monitor = monitor.clone();
        let handle = thread::Builder::new()
            .name("chiaro-export-prefetch".to_owned())
            .spawn(move || read_capture(&locator, &name, &monitor))
            .expect("failed to start capture prefetch");
        Self::Running(handle)
    }

    /// Wait for the read to finish. Progress of a still-running camera
    /// transfer is shown as the current capture's first half.
    fn take(self, target: &ExportTarget, monitor: &ExportMonitor) -> Result<CaptureData, String> {
        match self {
            Self::None => read_capture(&target.capture, &target.name, monitor),
            Self::Running(handle) => handle
                .join()
                .map_err(|_| "capture prefetch panicked".to_owned())?,
        }
    }
}

/// Read one capture: a local path is used in place, a camera object is
/// downloaded fully while the gallery's preview transport is paused.
fn read_capture(
    locator: &CaptureLocator,
    name: &str,
    monitor: &ExportMonitor,
) -> Result<CaptureData, String> {
    let remote = matches!(locator, CaptureLocator::Device(_));
    if remote {
        monitor.hold_transport(true);
        monitor.update(|progress| progress.transfer = Some((name.to_owned(), 0.0)));
    }
    let data = read_capture_with_updates(
        locator,
        |transferred, total| {
            if total > 0 {
                monitor.update(|progress| {
                    progress.transfer = Some((name.to_owned(), transferred as f32 / total as f32));
                });
            }
        },
        |_| {},
        || !monitor.cancelled(),
    );
    if remote {
        monitor.hold_transport(false);
        monitor.update(|progress| progress.transfer = None);
    }
    if monitor.cancelled() {
        return Err("cancelled".to_owned());
    }
    data
}

fn export_capture(
    options: &HotpixelExport,
    models: &LoadedModels,
    target: &ExportTarget,
    data: &CaptureData,
    output_root: &Path,
    monitor: &ExportMonitor,
) -> Result<usize, String> {
    let mapped;
    let bytes: &[u8] = match data {
        CaptureData::Local(path) => {
            mapped = mmap_file(path).map_err(|error| format!("{error:#}"))?;
            &mapped
        }
        CaptureData::Memory(bytes) => bytes,
    };
    let layout = parse_raw_layout(bytes, &HashMap::new()).map_err(|error| error.to_string())?;
    if layout.cameras.is_empty() {
        return Err("capture has no packed RAW10 cameras".to_owned());
    }

    let cleanup_cameras = match &models.cleanup {
        Some(profile) => layout
            .cameras
            .iter()
            .map(|camera| {
                profile
                    .load_camera(camera)
                    .map_err(|error| format!("{}: {error:#}", camera.name))
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => vec![None; layout.cameras.len()],
    };

    // Frames run one at a time; each one already spreads its rows over every
    // core, so this keeps peak memory at a single frame (about 60 MB for RAW
    // plus Q6 samples, with the PNG encoder streaming in small bands). The
    // factory map is re-read per frame (~20 ms) rather than cached for all
    // cameras, which would hold ~130 MB for a ten-camera capture.
    let stem = target.stem();
    let total = layout.cameras.len();
    let mut written = 0usize;
    let mut failures = Vec::new();
    for (finished, (camera, cleanup)) in layout.cameras.iter().zip(&cleanup_cameras).enumerate() {
        if monitor.cancelled() {
            failures.push(format!("{}: cancelled", camera.name));
            break;
        }
        monitor.update(|progress| {
            progress.current = format!("{} - {}", target.name, camera.name);
            progress.current_fraction = finished as f32 / total as f32;
        });
        let output = output_root.join(&camera.name).join(format!("{stem}.png"));
        let result = models
            .rec
            .load_rotated_map(camera.id, camera.width, camera.height)
            .map_err(|error| format!("{error:#}"))
            .and_then(|map| {
                export_frame(
                    options,
                    models,
                    &map,
                    cleanup.as_ref(),
                    bytes,
                    camera,
                    &output,
                )
            });
        match result {
            Ok(true) => written += 1,
            Ok(false) => {}
            Err(error) => failures.push(format!("{}: {error}", camera.name)),
        }
    }
    if !failures.is_empty() {
        return Err(failures.join("; "));
    }
    Ok(written)
}

/// Returns `Ok(true)` when a PNG was written, `Ok(false)` when skipped.
fn export_frame(
    options: &HotpixelExport,
    models: &LoadedModels,
    map: &[u8],
    cleanup: Option<&chiaro_hotpixel_core::cleanup::CleanupCameraProfile>,
    lri: &[u8],
    camera: &RawCamera,
    output: &Path,
) -> Result<bool, String> {
    if options.skip_existing && output.is_file() {
        return Ok(false);
    }
    let pipeline = FramePipeline {
        config: models.config.clone(),
        universal_hotpixel: models.universal.as_ref(),
        thermal: models.thermal.as_ref(),
        cleanup: CleanupStage::from_loaded(models.cleanup.is_some(), cleanup),
        threads: 0,
    };
    let frame = pipeline
        .correct_lri(lri, camera, map)
        .map_err(|error| format!("{error:#}"))?;
    frame
        .write_png_with_options(output, options.output_mode, 0, options.deflate_level())
        .map_err(|error| format!("{error:#}"))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{DeviceCalibration, FrameInfo};

    fn target(frames: Option<Vec<FrameInfo>>) -> ExportTarget {
        ExportTarget {
            name: "x.lri".to_owned(),
            capture: CaptureLocator::Local(PathBuf::from("/x.lri")),
            identity: None,
            device_id: None,
            frames,
        }
    }

    #[test]
    fn estimate_uses_six_bytes_for_demosaiced_bayer_and_two_for_mono() {
        let export = HotpixelExport::default();
        let frames = vec![
            FrameInfo {
                camera: "A1".into(),
                width: 100,
                height: 10,
                mono: false,
            },
            FrameInfo {
                camera: "A2".into(),
                width: 100,
                height: 10,
                mono: true,
            },
        ];
        let estimate = export.estimate(&[target(Some(frames.clone()))]);
        assert_eq!(estimate.bytes, 1000 * 6 + 1000 * 2);
        assert!(!estimate.approximate);

        let mosaic = HotpixelExport {
            output_mode: OutputMode::Mosaic,
            ..HotpixelExport::default()
        };
        assert_eq!(mosaic.estimate(&[target(Some(frames))]).bytes, 1000 * 4);
    }

    #[test]
    fn unprobed_targets_fall_back_to_a_typical_capture() {
        let export = HotpixelExport::default();
        let estimate = export.estimate(&[target(None)]);
        assert!(estimate.approximate);
        assert_eq!(estimate.bytes, 10 * 4160 * 3120 * 6);
    }

    #[test]
    fn validation_requires_destination_and_existing_record() {
        let mut export = HotpixelExport {
            output_dir: String::new(),
            ..HotpixelExport::default()
        };
        assert!(export.validate().is_err());
        let dir = tempfile::tempdir().unwrap();
        export.output_dir = dir.path().join("out").display().to_string();
        assert!(export.validate().unwrap_err().contains("hotpixel.rec"));
        export.hotpixel_rec.value = dir.path().join("missing.rec").display().to_string();
        assert!(export.validate().unwrap_err().contains("not found"));
        let rec = dir.path().join("hotpixel.rec");
        fs::write(&rec, b"stub").unwrap();
        export.hotpixel_rec.value = rec.display().to_string();
        assert!(export.validate().is_ok());
    }

    #[test]
    fn job_exports_every_camera_of_mock_captures_and_can_skip_existing() {
        use chiaro::mock::{MockCamera, MockCapture};
        use chiaro_hotpixel_core::hotpixel::write_hotpixel_rec;

        let dir = tempfile::tempdir().unwrap();
        let (width, height) = (64usize, 48usize);
        // Plant a hot pixel in A1 at RAW (10, 7) and list it in the factory map.
        // Factory maps are stored rotated 180 degrees relative to decoded RAW.
        let raw_index = 7 * width + 10;
        let mut a1_map = vec![0u8; width * height];
        a1_map[width * height - 1 - raw_index] = 200;
        let maps = (0..16)
            .map(|record| {
                let map = if record == 0 {
                    a1_map.clone()
                } else {
                    vec![0u8; width * height]
                };
                (width, height, map)
            })
            .collect::<Vec<_>>();
        let rec = dir.path().join("hotpixel.rec");
        write_hotpixel_rec(&rec, &maps).unwrap();

        let mut targets = Vec::new();
        for number in 1..=2 {
            let mut capture = MockCapture::small();
            capture.cameras[0] = MockCamera::gradient(
                "A1",
                width,
                height,
                chiaro::lri::SensorPattern::Bggr,
                64,
                700,
            )
            .with_defects(&[(10, 7, 1000)]);
            let path = dir.path().join(format!("L16_0000{number}.lri"));
            fs::write(&path, capture.encode().unwrap()).unwrap();
            targets.push(ExportTarget::new(
                format!("L16_0000{number}.lri"),
                CaptureLocator::Local(path),
            ));
        }

        let output = dir.path().join("frames");
        let options = HotpixelExport {
            output_dir: output.display().to_string(),
            hotpixel_rec: CalibrationPath {
                value: rec.display().to_string(),
                ..CalibrationPath::default()
            },
            ..HotpixelExport::default()
        };
        let monitor = ExportMonitor::new(
            super::super::ExportProgress {
                total: targets.len(),
                ..Default::default()
            },
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            egui::Context::default(),
        );
        run_job(&options, &targets, &monitor).unwrap();

        let progress = monitor.progress.lock().unwrap().clone();
        assert_eq!(progress.completed, 2);
        assert_eq!(progress.outputs_written, 6);
        assert!(progress.failures.is_empty(), "{:?}", progress.failures);
        assert_eq!(progress.succeeded_hashes.len(), 2);
        for camera in ["A1", "A2", "B1"] {
            for number in 1..=2 {
                let png = output.join(camera).join(format!("L16_0000{number}.png"));
                assert!(png.is_file(), "{}", png.display());
                assert!(fs::metadata(&png).unwrap().len() > 0);
            }
        }
        let log = fs::read_to_string(output.join("export-log.txt")).unwrap();
        assert!(log.contains("L16_00001.lri: 3 frames written"));

        // The planted defect is repaired: re-run the same pipeline in memory
        // and compare against the untouched neighbourhood.
        let lri = fs::read(dir.path().join("L16_00001.lri")).unwrap();
        let layout = parse_raw_layout(&lri, &HashMap::new()).unwrap();
        let a1 = layout.cameras.iter().find(|c| c.name == "A1").unwrap();
        let rec = HotpixelRec::open(&rec).unwrap();
        let map = rec.load_rotated_map(0, width, height).unwrap();
        let frame = FramePipeline::default()
            .correct_lri(&lri, a1, &map)
            .unwrap();
        assert_eq!(frame.hotpixel.corrected, 1);
        assert!(frame.samples_q6[raw_index] < 1000 << 6);

        // Second run with skip_existing writes nothing new.
        let skipping = HotpixelExport {
            skip_existing: true,
            ..options
        };
        monitor.update(|progress| *progress = Default::default());
        run_job(&skipping, &targets, &monitor).unwrap();
        assert_eq!(monitor.progress.lock().unwrap().outputs_written, 0);
    }

    #[test]
    fn device_calibration_fills_only_untouched_fields() {
        let mut field = CalibrationPath::default();
        let camera_file = PathBuf::from("/cache/hotpixel.rec");
        let ready = DeviceCalibration::Ready(
            [("hotpixel.rec".to_owned(), camera_file.clone())]
                .into_iter()
                .collect(),
        );
        field.adopt(Some(&ready), "hotpixel.rec");
        assert_eq!(field.value, camera_file.display().to_string());
        assert!(field.is_from_camera());

        field.set_manual(PathBuf::from("/my/own.rec"));
        field.adopt(Some(&ready), "hotpixel.rec");
        assert_eq!(field.value, "/my/own.rec");
        assert!(!field.is_from_camera());
        field.adopt(Some(&DeviceCalibration::Downloading), "calibration.lri");
        assert_eq!(field.value, "/my/own.rec");

        let mut stale_automatic = CalibrationPath::default();
        stale_automatic.adopt(Some(&ready), "hotpixel.rec");
        stale_automatic.adopt(None, "hotpixel.rec");
        assert!(stale_automatic.value.is_empty());
        assert!(!stale_automatic.is_from_camera());
    }
}
