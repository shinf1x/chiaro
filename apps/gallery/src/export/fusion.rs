//! Fused high-resolution frame per capture, powered by `chiaro-fusion`:
//! hot-pixel removal -> alignment -> synthesis.

use std::{
    fs,
    path::{Path, PathBuf},
    thread::{self, JoinHandle},
};

use chiaro_fusion::{
    pipeline::{FusionOptions, HotpixelStage, fuse},
    synth::{CanvasMode, OutputColor},
};
use chiaro_hotpixel_core::{demosaic::DemosaicMethod, scan::mmap_file};
use eframe::egui::{self, Color32, RichText};

use super::{
    CalibrationPath, ExportEstimate, ExportMonitor, ExportPipeline, ExportTarget, ExportUiServices,
    PathRow, PickField, PickKind, calibration_status, path_row,
};
use crate::source::{CaptureData, CaptureLocator, read_capture_with_updates};

const FIELD_OUTPUT: PickField = 1;
const FIELD_HOTPIXEL_REC: PickField = 2;
const FIELD_CALIBRATION: PickField = 3;
const FIELD_ZOOM_CALIBRATION: PickField = 4;

/// Native sensor resolution, the size of a "native" canvas.
const NATIVE_PIXELS: u64 = 4160 * 3120;
/// Cap for the "maximum detail" canvas.
const MAX_MEGAPIXELS: f32 = 64.0;

/// Canvas choices offered in the dialog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CanvasChoice {
    Native,
    Maximum,
}

#[derive(Clone, Debug)]
pub struct FusionExport {
    output_dir: String,
    hotpixel_rec: CalibrationPath,
    calibration: CalibrationPath,
    zoom_calibration: CalibrationPath,
    hotpixel_enabled: bool,
    canvas: CanvasChoice,
    crop_to_framing: bool,
    color: OutputColor,
    demosaic: DemosaicMethod,
    include_mono: bool,
    highlight_correction: bool,
    fast_compression: bool,
    skip_existing: bool,
}

impl Default for FusionExport {
    fn default() -> Self {
        let output_dir = std::env::home_dir()
            .map(|home| home.join("chiaro-fused"))
            .map_or_else(String::new, |path| path.display().to_string());
        Self {
            output_dir,
            hotpixel_rec: CalibrationPath::default(),
            calibration: CalibrationPath::default(),
            zoom_calibration: CalibrationPath::default(),
            hotpixel_enabled: true,
            canvas: CanvasChoice::Native,
            crop_to_framing: true,
            color: OutputColor::Display,
            demosaic: DemosaicMethod::default(),
            include_mono: true,
            highlight_correction: true,
            fast_compression: false,
            skip_existing: false,
        }
    }
}

impl FusionExport {
    fn options(&self) -> FusionOptions {
        let mut options = FusionOptions {
            overlays: [&self.calibration, &self.zoom_calibration]
                .iter()
                .filter_map(|field| field.path())
                .collect(),
            hotpixel: (self.hotpixel_enabled)
                .then(|| self.hotpixel_rec.path())
                .flatten()
                .map(|rec| HotpixelStage {
                    rec,
                    universal_model: true,
                    glow_correction: true,
                }),
            ..FusionOptions::default()
        };
        options.crop_to_framing = self.crop_to_framing;
        options.synth.canvas = match self.canvas {
            CanvasChoice::Native => CanvasMode::Native,
            CanvasChoice::Maximum => CanvasMode::Maximum {
                max_megapixels: MAX_MEGAPIXELS,
            },
        };
        options.synth.color = self.color;
        options.synth.demosaic = self.demosaic;
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

impl ExportPipeline for FusionExport {
    fn clone_box(&self) -> Box<dyn ExportPipeline> {
        Box::new(self.clone())
    }

    fn label(&self) -> &'static str {
        "Fused high-resolution frame"
    }

    fn description(&self) -> &'static str {
        "Align every camera and synthesise one frame per capture (far scenes)"
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
                hint: "Folder for one fused PNG (and report) per capture",
                title: "Choose the export folder",
                kind: PickKind::Folder,
                clearable: false,
            },
        );
        ui.add_space(8.0);

        ui.label(RichText::new("Hot-pixel removal").strong());
        ui.checkbox(
            &mut self.hotpixel_enabled,
            "Remove factory-listed hot pixels first",
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
        ui.add_space(8.0);

        ui.label(RichText::new("Geometric calibration").strong());
        ui.label(
            RichText::new(
                "Captures embed only part of the camera's calibration. The device files supply \
                 mirror aiming data and complete the geometry; without them, cross-module \
                 alignment is likely to be poor.",
            )
            .color(muted)
            .size(12.0),
        );
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

        ui.label(RichText::new("Output").strong());
        ui.checkbox(
            &mut self.crop_to_framing,
            "Crop to the framed field of view (the focal length set on the camera)",
        )
        .on_hover_text(
            "The crop is relative to the capture's reference group: 28 mm for A, 70 mm for B, \
             or 150 mm for C. Unchecked renders the whole reference frame.",
        );
        ui.horizontal(|ui| {
            ui.label("Resolution");
            ui.radio_value(&mut self.canvas, CanvasChoice::Native, "Native (13 MP)");
            ui.radio_value(
                &mut self.canvas,
                CanvasChoice::Maximum,
                format!("Maximum detail (up to {MAX_MEGAPIXELS:.0} MP)"),
            )
            .on_hover_text(
                "As many pixels as the finest module covering the view justifies: about \
                 2.5x the native resolution where the B modules cover, 5.5x for C.",
            );
        });
        ui.horizontal(|ui| {
            ui.label("Demosaicing");
            egui::ComboBox::from_id_salt("fusion-demosaic")
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
        )
        .on_hover_text(
            "Panchromatic modules (A2, C6) sharpen luminance; colour comes from the Bayer modules.",
        );
        ui.checkbox(
            &mut self.highlight_correction,
            "Neutralize false colour in clipped highlights",
        )
        .on_hover_text(
            "Enabled by default for display-ready output. Disable to preserve unequal clipped \
             raw-channel colour for processing elsewhere.",
        );
        ui.checkbox(
            &mut self.skip_existing,
            "Skip captures whose fused PNG already exists",
        );
        ui.checkbox(
            &mut self.fast_compression,
            "Faster export with larger files (lighter PNG compression)",
        );
        ui.label(
            RichText::new(
                "Alignment fits one homography per module, exact for distant scenes; near \
                 objects may ghost. A .fusion.json report with per-module residuals is \
                 written beside each frame.",
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
                "{} frames of up to {:.0} MP 16-bit RGB, uncompressed upper bound; PNG compression usually needs half",
                targets.len(),
                pixels as f64 / 1.0e6
            ),
        }
    }

    fn start(&self, targets: Vec<ExportTarget>, monitor: ExportMonitor) -> JoinHandle<()> {
        let export = self.clone();
        thread::Builder::new()
            .name("chiaro-export-fusion".to_owned())
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
            .expect("failed to start fusion export")
    }
}

fn run_job(
    export: &FusionExport,
    targets: &[ExportTarget],
    monitor: &ExportMonitor,
) -> Result<(), String> {
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
        let output = output_root.join(format!("{}_fused.png", target.stem()));
        let result = if export.skip_existing && output.is_file() {
            Ok(false)
        } else {
            fuse_capture(&options, target, &output, monitor).map(|()| true)
        };
        match result {
            Ok(written) => {
                log.push(format!(
                    "{}: {}",
                    target.name,
                    if written { "fused" } else { "skipped (exists)" }
                ));
                monitor.update(|progress| {
                    progress.outputs_written += usize::from(written);
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
    let _ = fs::write(
        output_root.join("export-log.txt"),
        format!(
            "Chiaro Gallery fusion export\nhotpixel.rec: {}\ncalibration: {} / {}\ncanvas: {:?}, crop to framing: {}, demosaicing: {}, highlight correction: {}\n\n{}\n",
            export.hotpixel_rec.value.trim(),
            export.calibration.value.trim(),
            export.zoom_calibration.value.trim(),
            export.canvas,
            export.crop_to_framing,
            export.demosaic,
            export.highlight_correction,
            log.join("\n")
        ),
    );
    Ok(())
}

fn fuse_capture(
    options: &FusionOptions,
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
    let mapped;
    let bytes: &[u8] = match &data {
        CaptureData::Local(path) => {
            mapped = mmap_file(path).map_err(|error| format!("{error:#}"))?;
            &mapped
        }
        CaptureData::Memory(bytes) => bytes,
    };
    let name = target.name.clone();
    fuse(bytes, options, output, &mut |progress| {
        monitor.update(|state| {
            state.current = format!("{name} - {}: {}", progress.stage, progress.detail);
            state.current_fraction = progress.fraction;
        });
    })
    .map(|_| ())
    .map_err(|error| format!("{error:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_scales_with_canvas_size() {
        let export = FusionExport::default();
        let target = ExportTarget {
            name: "x.lri".to_owned(),
            capture: CaptureLocator::Local(PathBuf::from("/x.lri")),
            identity: None,
            device_id: None,
            frames: None,
        };
        assert_eq!(
            export.estimate(std::slice::from_ref(&target)).bytes,
            4160 * 3120 * 6
        );
        let big = FusionExport {
            canvas: CanvasChoice::Maximum,
            ..FusionExport::default()
        };
        assert_eq!(
            big.estimate(&[target.clone(), target]).bytes,
            2 * 64_000_000 * 6
        );
    }

    #[test]
    fn validation_allows_skipping_hot_pixels_and_checks_overlays() {
        let dir = tempfile::tempdir().unwrap();
        let mut export = FusionExport {
            output_dir: dir.path().join("out").display().to_string(),
            ..FusionExport::default()
        };
        assert!(export.validate().unwrap_err().contains("hotpixel.rec"));
        export.hotpixel_enabled = false;
        assert!(export.validate().is_ok());
        export.calibration.value = dir.path().join("missing.lri").display().to_string();
        assert!(export.validate().unwrap_err().contains("calibration.lri"));
    }

    #[test]
    fn job_fuses_mock_captures() {
        use chiaro::mock::MockCapture;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("L16_00001.lri");
        fs::write(&path, MockCapture::small().encode().unwrap()).unwrap();
        let export = FusionExport {
            output_dir: dir.path().join("fused").display().to_string(),
            hotpixel_enabled: false,
            ..FusionExport::default()
        };
        let targets = vec![ExportTarget::new(
            "L16_00001.lri".to_owned(),
            CaptureLocator::Local(path),
        )];
        let monitor = ExportMonitor::new(
            super::super::ExportProgress {
                total: 1,
                ..Default::default()
            },
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            egui::Context::default(),
        );
        run_job(&export, &targets, &monitor).unwrap();
        let progress = monitor.progress.lock().unwrap().clone();
        assert!(progress.failures.is_empty(), "{:?}", progress.failures);
        assert_eq!(progress.succeeded_hashes.len(), 1);
        assert!(dir.path().join("fused/L16_00001_fused.png").is_file());
        assert!(
            dir.path()
                .join("fused/L16_00001_fused.fusion.json")
                .is_file()
        );
    }
}
