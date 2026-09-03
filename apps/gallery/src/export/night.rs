//! Motion-aware temporal and multi-camera fusion for night-mode captures.

use std::{
    fs,
    path::{Path, PathBuf},
    thread::{self, JoinHandle},
};

use chiaro_fusion::{
    crosstalk::CrosstalkMode,
    resolution::ResolutionReconstruction,
    synth::{CanvasMode, OutputColor},
};
use chiaro_hotpixel_core::{
    demosaic::DemosaicMethod, highlight::HighlightRecovery, scan::mmap_file,
};
use chiaro_stack::fusion::{NightFusionOptions, fuse_night};
use eframe::egui::{self, Color32, RichText};

use super::{
    CalibrationPath, CaptureRequirement, ExportEstimate, ExportMonitor, ExportPipeline,
    ExportTarget, ExportUiServices, PathRow, PickField, PickKind, calibration_status, path_row,
};
use crate::source::{CaptureData, CaptureLocator, read_capture_with_updates};

const FIELD_OUTPUT: PickField = 1;
const FIELD_HOTPIXEL_REC: PickField = 2;
const FIELD_CALIBRATION: PickField = 3;
const FIELD_ZOOM_CALIBRATION: PickField = 4;

const NATIVE_PIXELS: u64 = 4160 * 3120;
// Lumen's full-resolution 28 mm output is 10432 x 7824 (81.6 MP). Keep the
// cap just above that tier so calibrated A/B magnification is not truncated.
const MAX_MEGAPIXELS: f32 = 82.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CanvasChoice {
    Native,
    Maximum,
}

#[derive(Clone, Debug)]
pub struct NightStackExport {
    output_dir: String,
    hotpixel_rec: CalibrationPath,
    calibration: CalibrationPath,
    zoom_calibration: CalibrationPath,
    hotpixel_enabled: bool,
    motion_sigma: f32,
    gyro_seed: bool,
    refine_alignment: bool,
    flat_field: bool,
    local_photometric: bool,
    canvas: CanvasChoice,
    crop_to_framing: bool,
    color: OutputColor,
    demosaic: DemosaicMethod,
    highlight_recovery: HighlightRecovery,
    crosstalk: CrosstalkMode,
    resolution_reconstruction: ResolutionReconstruction,
    include_mono: bool,
    highlight_correction: bool,
    fast_compression: bool,
    skip_existing: bool,
}

impl Default for NightStackExport {
    fn default() -> Self {
        let output_dir = std::env::home_dir()
            .map(|home| home.join("chiaro-night-stacks"))
            .map_or_else(String::new, |path| path.display().to_string());
        Self {
            output_dir,
            hotpixel_rec: CalibrationPath::default(),
            calibration: CalibrationPath::default(),
            zoom_calibration: CalibrationPath::default(),
            hotpixel_enabled: true,
            motion_sigma: 4.0,
            gyro_seed: true,
            refine_alignment: true,
            flat_field: true,
            local_photometric: true,
            canvas: CanvasChoice::Native,
            crop_to_framing: true,
            color: OutputColor::Display,
            demosaic: DemosaicMethod::Lmmse,
            highlight_recovery: HighlightRecovery::MultiCamera,
            crosstalk: CrosstalkMode::default(),
            resolution_reconstruction: ResolutionReconstruction::default(),
            include_mono: true,
            highlight_correction: true,
            fast_compression: false,
            skip_existing: false,
        }
    }
}

impl NightStackExport {
    fn options(&self) -> NightFusionOptions {
        let mut options = NightFusionOptions {
            overlays: [&self.calibration, &self.zoom_calibration]
                .iter()
                .filter_map(|field| field.path())
                .collect(),
            hotpixel_rec: self
                .hotpixel_enabled
                .then(|| self.hotpixel_rec.path())
                .flatten(),
            motion_sigma: self.motion_sigma,
            gyro_seed: self.gyro_seed,
            flat_field: self.flat_field,
            local_photometric: self.local_photometric,
            crop_to_framing: self.crop_to_framing,
            ..NightFusionOptions::default()
        };
        options.temporal_align.refine = self.refine_alignment;
        options.module_align.refine = self.refine_alignment;
        options.synth.canvas = match self.canvas {
            CanvasChoice::Native => CanvasMode::Native,
            CanvasChoice::Maximum => CanvasMode::Maximum {
                max_megapixels: MAX_MEGAPIXELS,
            },
        };
        options.synth.color = self.color;
        options.synth.demosaic = self.demosaic;
        options.synth.highlight_recovery = self.highlight_recovery;
        options.crosstalk = self.crosstalk;
        options.synth.resolution_reconstruction = self.resolution_reconstruction;
        options.synth.include_mono = self.include_mono;
        options.synth.highlight_correction = self.highlight_correction;
        options.synth.png_level = if self.fast_compression {
            1
        } else {
            chiaro_hotpixel_core::png16::DEFAULT_DEFLATE_LEVEL
        };
        options
    }
}

impl ExportPipeline for NightStackExport {
    fn clone_box(&self) -> Box<dyn ExportPipeline> {
        Box::new(self.clone())
    }

    fn label(&self) -> &'static str {
        "Night stack"
    }

    fn description(&self) -> &'static str {
        "Motion-aware temporal denoising and calibrated multi-camera fusion"
    }

    fn capture_requirement(&self) -> CaptureRequirement {
        CaptureRequirement::Night
    }

    fn options_ui(&mut self, ui: &mut egui::Ui, services: &mut ExportUiServices<'_>) {
        self.hotpixel_rec
            .adopt(services.device_calibration, "hotpixel.rec");
        self.calibration
            .adopt(services.device_calibration, "calibration.lri");
        self.zoom_calibration
            .adopt(services.device_calibration, "zoom_calib_v0.lri");
        let muted = Color32::from_gray(150);

        ui.label(RichText::new("Destination").strong());
        path_row(
            ui,
            services,
            &mut self.output_dir,
            PathRow {
                field: FIELD_OUTPUT,
                hint: "Folder for one stacked PNG and report per night capture",
                title: "Choose the night-stack export folder",
                kind: PickKind::Folder,
                clearable: false,
            },
        );
        ui.add_space(8.0);

        ui.label(RichText::new("Camera calibration").strong());
        ui.checkbox(
            &mut self.hotpixel_enabled,
            "Remove factory-listed hot pixels from every temporal frame",
        );
        if self.hotpixel_enabled {
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
        }
        path_row(
            ui,
            services,
            &mut self.calibration.value,
            PathRow {
                field: FIELD_CALIBRATION,
                hint: "calibration.lri (strongly recommended)",
                title: "Choose calibration.lri",
                kind: PickKind::File {
                    filter: Some(("Light calibration", &["lri"])),
                },
                clearable: true,
            },
        );
        calibration_status(ui, services, &self.calibration, "calibration.lri");
        path_row(
            ui,
            services,
            &mut self.zoom_calibration.value,
            PathRow {
                field: FIELD_ZOOM_CALIBRATION,
                hint: "zoom_calib_v0.lri (strongly recommended)",
                title: "Choose zoom_calib_v0.lri",
                kind: PickKind::File {
                    filter: Some(("Light calibration", &["lri"])),
                },
                clearable: true,
            },
        );
        calibration_status(ui, services, &self.zoom_calibration, "zoom_calib_v0.lri");
        ui.add_space(8.0);

        ui.label(RichText::new("Motion and alignment").strong());
        ui.horizontal(|ui| {
            ui.label("Motion rejection");
            ui.add(
                egui::Slider::new(&mut self.motion_sigma, 1.0..=8.0)
                    .fixed_decimals(1)
                    .suffix(" σ"),
            )
            .on_hover_text(
                "Lower values reject more moving or misaligned samples; higher values retain \
                 more frames for denoising.",
            );
        });
        ui.checkbox(&mut self.gyro_seed, "Seed alignment from gyroscope data");
        ui.checkbox(
            &mut self.refine_alignment,
            "Refine temporal and cross-camera alignment from image content",
        );
        ui.checkbox(
            &mut self.local_photometric,
            "Match local brightness between camera modules",
        );
        ui.add_space(8.0);

        ui.label(RichText::new("Output").strong());
        ui.checkbox(
            &mut self.crop_to_framing,
            "Crop to the field of view framed on the camera",
        );
        ui.horizontal(|ui| {
            ui.label("Resolution");
            ui.radio_value(&mut self.canvas, CanvasChoice::Native, "Native (13 MP)");
            ui.radio_value(
                &mut self.canvas,
                CanvasChoice::Maximum,
                format!("Maximum detail (up to {MAX_MEGAPIXELS:.0} MP)"),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Demosaicing");
            egui::ComboBox::from_id_salt("night-stack-demosaic")
                .selected_text(self.demosaic.label())
                .show_ui(ui, |ui| {
                    for method in DemosaicMethod::ALL {
                        ui.selectable_value(
                            &mut self.demosaic,
                            method,
                            format!("{} — {}", method.label(), method.recommendation()),
                        );
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("RAW highlight recovery");
            egui::ComboBox::from_id_salt("night-highlight-recovery")
                .selected_text(self.highlight_recovery.label())
                .show_ui(ui, |ui| {
                    for mode in HighlightRecovery::ALL {
                        ui.selectable_value(&mut self.highlight_recovery, mode, mode.label());
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("RAW crosstalk");
            egui::ComboBox::from_id_salt("night-crosstalk")
                .selected_text(self.crosstalk.label())
                .show_ui(ui, |ui| {
                    for mode in CrosstalkMode::ALL {
                        ui.selectable_value(&mut self.crosstalk, mode, mode.label());
                    }
                });
        })
        .response
        .on_hover_text(
            "Adaptive preserves the factory mesh and fits a constrained capture-specific \
             residual from smooth aligned overlap between camera modules.",
        );
        ui.horizontal(|ui| {
            ui.label("Resolution reconstruction");
            egui::ComboBox::from_id_salt("night-resolution-reconstruction")
                .selected_text(self.resolution_reconstruction.label())
                .show_ui(ui, |ui| {
                    for mode in ResolutionReconstruction::ALL {
                        ui.selectable_value(
                            &mut self.resolution_reconstruction,
                            mode,
                            mode.label(),
                        );
                    }
                });
        })
        .response
        .on_hover_text(
            "Multi-camera reconstructs luminance detail from independently projected physical \
             sensor samples while retaining the robust colour fusion path.",
        );
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut self.color,
                OutputColor::Display,
                "sRGB (display-ready)",
            );
            ui.radio_value(&mut self.color, OutputColor::Linear, "Linear camera RGB");
        });
        ui.checkbox(
            &mut self.include_mono,
            "Use monochrome modules for luminance detail",
        );
        ui.checkbox(&mut self.flat_field, "Apply factory flat-field correction");
        ui.checkbox(
            &mut self.highlight_correction,
            "Neutralize false colour in clipped highlights",
        );
        ui.checkbox(
            &mut self.skip_existing,
            "Skip captures whose night-stack PNG already exists",
        );
        ui.checkbox(
            &mut self.fast_compression,
            "Faster export with larger files (lighter PNG compression)",
        );
        ui.label(
            RichText::new(
                "Only captures marked as night mode are included. Each output combines temporal \
                 frames first, then aligns the denoised physical camera modules. A \
                 .night-fusion.json diagnostic report is written beside the PNG.",
            )
            .color(muted)
            .size(12.0),
        );
    }

    fn apply_pick(&mut self, field: PickField, path: PathBuf) {
        match field {
            FIELD_OUTPUT => self.output_dir = path.display().to_string(),
            FIELD_HOTPIXEL_REC => self.hotpixel_rec.set_manual(path),
            FIELD_CALIBRATION => self.calibration.set_manual(path),
            FIELD_ZOOM_CALIBRATION => self.zoom_calibration.set_manual(path),
            _ => {}
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.output_dir.trim().is_empty() {
            return Err("Choose a destination folder".to_owned());
        }
        if Path::new(self.output_dir.trim()).is_file() {
            return Err("The destination is an existing file".to_owned());
        }
        if !self.motion_sigma.is_finite() || !(1.0..=8.0).contains(&self.motion_sigma) {
            return Err("Motion rejection must be between 1 and 8 sigma".to_owned());
        }
        if self.hotpixel_enabled {
            match self.hotpixel_rec.path() {
                None => {
                    return Err(
                        "Choose the camera's hotpixel.rec or disable hot-pixel removal".to_owned(),
                    );
                }
                Some(rec) if !rec.is_file() => {
                    return Err("hotpixel.rec was not found at that path".to_owned());
                }
                _ => {}
            }
        }
        for (field, name) in [
            (&self.calibration, "calibration.lri"),
            (&self.zoom_calibration, "zoom_calib_v0.lri"),
        ] {
            if let Some(path) = field.path()
                && !path.is_file()
            {
                return Err(format!("{name} was not found at that path"));
            }
        }
        Ok(())
    }

    fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.output_dir.trim())
    }

    fn estimate(&self, targets: &[ExportTarget]) -> ExportEstimate {
        let pixels = match self.canvas {
            CanvasChoice::Native => NATIVE_PIXELS,
            CanvasChoice::Maximum => (MAX_MEGAPIXELS * 1.0e6) as u64,
        };
        ExportEstimate {
            bytes: pixels * 6 * targets.len() as u64,
            approximate: self.canvas == CanvasChoice::Maximum,
            detail: format!(
                "{} night-stack outputs of up to {:.0} MP 16-bit RGB, plus JSON reports",
                targets.len(),
                pixels as f64 / 1.0e6
            ),
        }
    }

    fn start(&self, targets: Vec<ExportTarget>, monitor: ExportMonitor) -> JoinHandle<()> {
        let export = self.clone();
        thread::Builder::new()
            .name("chiaro-export-night-stack".to_owned())
            .spawn(move || {
                let outcome = run_job(&export, &targets, &monitor);
                monitor.hold_transport(false);
                monitor.update(|progress| {
                    if let Err(error) = outcome {
                        progress.fatal = Some(error);
                    }
                    progress.finished = true;
                    progress.current.clear();
                });
            })
            .expect("failed to start night-stack export")
    }
}

fn run_job(
    export: &NightStackExport,
    targets: &[ExportTarget],
    monitor: &ExportMonitor,
) -> Result<(), String> {
    if targets.iter().any(|target| !target.night_mode) {
        return Err("night-stack export received a standard capture".to_owned());
    }
    let output_root = export.output_dir();
    fs::create_dir_all(&output_root)
        .map_err(|error| format!("create {}: {error}", output_root.display()))?;
    let options = export.options();
    let mut log = Vec::new();
    for target in targets {
        if monitor.cancelled() {
            break;
        }
        monitor.update(|progress| {
            progress.current = target.name.clone();
            progress.current_fraction = 0.0;
        });
        let output = output_root.join(format!("{}_night.png", target.stem()));
        let report = output.with_extension("night-fusion.json");
        let result = if export.skip_existing && output.is_file() && report.is_file() {
            Ok(false)
        } else {
            stack_capture(&options, target, &output, monitor).map(|()| true)
        };
        match result {
            Ok(written) => {
                log.push(format!(
                    "{}: {}",
                    target.name,
                    if written {
                        "stacked"
                    } else {
                        "skipped (exists)"
                    }
                ));
                monitor.update(|progress| {
                    progress.outputs_written += usize::from(written);
                    progress.completed += 1;
                    progress.current_fraction = 0.0;
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
                    progress.current_fraction = 0.0;
                });
            }
        }
    }
    let _ = fs::write(
        output_root.join("night-export-log.txt"),
        format!(
            "Chiaro Gallery night-stack export\nhotpixel.rec: {}\ncalibration: {} / {}\n\
             motion rejection: {:.1} sigma, gyro seed: {}, refinement: {}\n\
             canvas: {:?}, crop to framing: {}, demosaicing: {}, RAW highlight recovery: {}, RAW crosstalk: {}, output color: {:?}\n\n{}\n",
            export.hotpixel_rec.value.trim(),
            export.calibration.value.trim(),
            export.zoom_calibration.value.trim(),
            export.motion_sigma,
            export.gyro_seed,
            export.refine_alignment,
            export.canvas,
            export.crop_to_framing,
            export.demosaic,
            export.highlight_recovery,
            export.crosstalk,
            export.color,
            log.join("\n")
        ),
    );
    Ok(())
}

fn stack_capture(
    options: &NightFusionOptions,
    target: &ExportTarget,
    output: &Path,
    monitor: &ExportMonitor,
) -> Result<(), String> {
    let remote = matches!(target.capture, CaptureLocator::Device(_));
    if remote {
        monitor.hold_transport(true);
    }
    let data = read_capture_with_updates(
        &target.capture,
        |transferred, total| {
            if total > 0 {
                monitor.update(|progress| {
                    progress.transfer =
                        Some((target.name.clone(), transferred as f32 / total as f32));
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
    let data = data?;
    if monitor.cancelled() {
        return Err("cancelled".to_owned());
    }
    let mapped;
    let bytes: &[u8] = match &data {
        CaptureData::Local(path) => {
            mapped = mmap_file(path).map_err(|error| format!("{error:#}"))?;
            &mapped
        }
        CaptureData::Memory(bytes) => bytes,
    };
    let name = target.name.clone();
    let mut temporal_modules = 0usize;
    fuse_night(bytes, options, output, &mut |detail| {
        let fraction = if detail.starts_with("temporal stack ") {
            temporal_modules += 1;
            (0.08 + temporal_modules as f32 * 0.04).min(0.60)
        } else if detail == "align denoised modules" {
            0.68
        } else if detail == "synthesise all-module night image" {
            0.88
        } else {
            0.05
        };
        monitor.update(|progress| {
            progress.current = format!("{name} - {detail}");
            progress.current_fraction = fraction;
        });
    })
    .map(|_| ())
    .map_err(|error| format!("{error:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(night_mode: bool) -> ExportTarget {
        ExportTarget::new(
            "L16_00001.lri".to_owned(),
            CaptureLocator::Local(PathBuf::from("/x/L16_00001.lri")),
        )
        .with_night_mode(night_mode)
    }

    #[test]
    fn pipeline_accepts_only_night_captures() {
        let export = NightStackExport::default();
        assert_eq!(export.demosaic, DemosaicMethod::Lmmse);
        assert_eq!(export.capture_requirement(), CaptureRequirement::Night);
        assert!(
            export
                .capture_requirement()
                .accepts(target(true).night_mode)
        );
        assert!(
            !export
                .capture_requirement()
                .accepts(target(false).night_mode)
        );
    }

    #[test]
    fn estimate_scales_with_canvas_size() {
        let export = NightStackExport::default();
        assert_eq!(export.estimate(&[target(true)]).bytes, NATIVE_PIXELS * 6);
        let maximum = NightStackExport {
            canvas: CanvasChoice::Maximum,
            ..NightStackExport::default()
        };
        assert_eq!(maximum.estimate(&[target(true)]).bytes, 82_000_000 * 6);
        assert!(maximum.estimate(&[target(true)]).approximate);
    }

    #[test]
    fn validation_checks_motion_and_calibration_paths() {
        let dir = tempfile::tempdir().unwrap();
        let mut export = NightStackExport {
            output_dir: dir.path().join("out").display().to_string(),
            hotpixel_enabled: false,
            ..NightStackExport::default()
        };
        assert!(export.validate().is_ok());
        export.motion_sigma = 0.0;
        assert!(export.validate().unwrap_err().contains("Motion rejection"));
        export.motion_sigma = 4.0;
        export.calibration.value = dir.path().join("missing.lri").display().to_string();
        assert!(export.validate().unwrap_err().contains("calibration.lri"));
    }

    #[test]
    fn settings_are_forwarded_to_the_stack_pipeline() {
        let export = NightStackExport {
            hotpixel_enabled: false,
            motion_sigma: 2.5,
            gyro_seed: false,
            refine_alignment: false,
            flat_field: false,
            local_photometric: false,
            canvas: CanvasChoice::Maximum,
            crop_to_framing: false,
            color: OutputColor::Linear,
            demosaic: DemosaicMethod::Igv,
            highlight_recovery: HighlightRecovery::LocalBayer,
            crosstalk: CrosstalkMode::Factory,
            resolution_reconstruction: ResolutionReconstruction::Resample,
            include_mono: false,
            highlight_correction: false,
            fast_compression: true,
            ..NightStackExport::default()
        };
        let options = export.options();
        assert_eq!(options.motion_sigma, 2.5);
        assert!(!options.gyro_seed);
        assert!(!options.temporal_align.refine);
        assert!(!options.module_align.refine);
        assert!(!options.flat_field);
        assert!(!options.local_photometric);
        assert!(!options.crop_to_framing);
        assert_eq!(
            options.synth.canvas,
            CanvasMode::Maximum {
                max_megapixels: MAX_MEGAPIXELS
            }
        );
        assert_eq!(options.synth.color, OutputColor::Linear);
        assert_eq!(options.synth.demosaic, DemosaicMethod::Igv);
        assert_eq!(
            options.synth.highlight_recovery,
            HighlightRecovery::LocalBayer
        );
        assert_eq!(options.crosstalk, CrosstalkMode::Factory);
        assert_eq!(
            options.synth.resolution_reconstruction,
            ResolutionReconstruction::Resample
        );
        assert!(!options.synth.include_mono);
        assert!(!options.synth.highlight_correction);
        assert_eq!(options.synth.png_level, 1);
    }

    #[test]
    fn runner_rejects_standard_captures_defensively() {
        let export = NightStackExport {
            output_dir: tempfile::tempdir()
                .unwrap()
                .path()
                .join("out")
                .display()
                .to_string(),
            hotpixel_enabled: false,
            ..NightStackExport::default()
        };
        let monitor = ExportMonitor::new(
            super::super::ExportProgress {
                total: 1,
                ..Default::default()
            },
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            egui::Context::default(),
        );
        assert!(
            run_job(&export, &[target(false)], &monitor)
                .unwrap_err()
                .contains("standard capture")
        );
    }
}
