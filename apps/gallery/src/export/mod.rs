//! Export pipelines: batch processing of selected captures.
//!
//! This module is deliberately independent from preview loading. Previews are
//! the gallery's display path; exports are user-initiated batch jobs that read
//! the selected captures again (locally by memory map, from a camera by a full
//! download) and hand them to a processing library.
//!
//! Each kind of export implements [`ExportPipeline`]. The gallery only knows
//! how to list pipelines, draw the options UI a pipeline provides, check disk
//! space against the pipeline's estimate, and run the job it starts. Adding a
//! new export type means adding one implementation to [`ExportRegistry::new`];
//! nothing in the card grid or status bar needs to change.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use chiaro::lri::{SensorPattern, parse_raw_layout};
use eframe::egui;

use crate::source::CaptureLocator;

pub mod disk;
mod fusion;
mod hotpixel;

pub use fusion::FusionExport;
pub use hotpixel::HotpixelExport;

/// One capture chosen for export.
#[derive(Clone, Debug)]
pub struct ExportTarget {
    /// Display name, normally the LRI file name.
    pub name: String,
    pub capture: CaptureLocator,
    /// RAW frames in the capture when they could be probed cheaply. `None`
    /// for remote captures that would require a full download to inspect.
    pub frames: Option<Vec<FrameInfo>>,
}

impl ExportTarget {
    /// Build a target from a gallery item, probing local files for their
    /// RAW layout so estimates can be exact.
    pub fn new(name: String, capture: CaptureLocator) -> Self {
        let frames = match &capture {
            CaptureLocator::Local(path) => probe_frames(path),
            CaptureLocator::Device(_) => None,
        };
        Self {
            name,
            capture,
            frames,
        }
    }

    /// File stem used for per-capture outputs.
    pub fn stem(&self) -> String {
        let stem = Path::new(&self.name)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or(&self.name);
        let mut output = String::with_capacity(stem.len());
        for character in stem.chars() {
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
}

/// Shape of one RAW frame, enough for output-size estimates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameInfo {
    pub camera: String,
    pub width: usize,
    pub height: usize,
    pub mono: bool,
}

fn probe_frames(path: &Path) -> Option<Vec<FrameInfo>> {
    let mmap = chiaro_hotpixel_core::scan::mmap_file(path).ok()?;
    let layout = parse_raw_layout(&mmap, &Default::default()).ok()?;
    Some(
        layout
            .cameras
            .iter()
            .map(|camera| FrameInfo {
                camera: camera.name.clone(),
                width: camera.width,
                height: camera.height,
                mono: camera.pattern == SensorPattern::Mono,
            })
            .collect(),
    )
}

/// Predicted output size of a job.
#[derive(Clone, Debug)]
pub struct ExportEstimate {
    /// Upper bound of bytes the job may write.
    pub bytes: u64,
    /// Some targets could not be probed and were estimated from typical
    /// Light L16 captures.
    pub approximate: bool,
    /// Short explanation shown beside the number.
    pub detail: String,
}

/// Live progress of a running job, shared between the job thread and the UI.
#[derive(Clone, Debug, Default)]
pub struct ExportProgress {
    pub completed: usize,
    pub total: usize,
    /// What the job is doing right now (capture name, stage).
    pub current: String,
    /// Progress within the current capture, 0..=1.
    pub current_fraction: f32,
    /// A camera transfer in flight: capture name and fraction copied. It may
    /// belong to the next capture when the job reads ahead.
    pub transfer: Option<(String, f32)>,
    pub outputs_written: usize,
    pub failures: Vec<(String, String)>,
    pub finished: bool,
    /// Fatal error that stopped the job before completion.
    pub fatal: Option<String>,
}

impl ExportProgress {
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        ((self.completed as f32 + self.current_fraction.clamp(0.0, 1.0)) / self.total as f32)
            .clamp(0.0, 1.0)
    }
}

/// Handles a job thread uses to report progress and observe cancellation.
#[derive(Clone)]
pub struct ExportMonitor {
    pub progress: Arc<Mutex<ExportProgress>>,
    pub cancel: Arc<AtomicBool>,
    /// Raised while the job transfers from a connected camera so preview
    /// loading yields the USB transport.
    pub transport_hold: Arc<AtomicBool>,
    pub ctx: egui::Context,
    /// When the UI was last asked to repaint, so frequent progress updates
    /// do not turn into a repaint storm that competes with the job for CPU.
    last_repaint: Arc<Mutex<Option<Instant>>>,
}

/// Minimum spacing between repaint requests caused by progress updates.
const REPAINT_INTERVAL: Duration = Duration::from_millis(80);

impl ExportMonitor {
    pub fn new(
        progress: ExportProgress,
        transport_hold: Arc<AtomicBool>,
        ctx: egui::Context,
    ) -> Self {
        Self {
            progress: Arc::new(Mutex::new(progress)),
            cancel: Arc::new(AtomicBool::new(false)),
            transport_hold,
            ctx,
            last_repaint: Arc::new(Mutex::new(None)),
        }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    /// Apply a progress change. The UI is repainted immediately when the job
    /// finishes or fails and at most every [`REPAINT_INTERVAL`] otherwise.
    pub fn update(&self, apply: impl FnOnce(&mut ExportProgress)) {
        let finished = {
            let mut progress = self.progress.lock().expect("export progress poisoned");
            apply(&mut progress);
            progress.finished || progress.fatal.is_some()
        };
        let mut last = self.last_repaint.lock().expect("repaint clock poisoned");
        let due = last.is_none_or(|at| at.elapsed() >= REPAINT_INTERVAL);
        if finished || due {
            *last = Some(Instant::now());
            self.ctx.request_repaint();
        } else if let Some(at) = *last {
            // Make sure the latest state is shown even if no further update
            // arrives before the interval elapses.
            self.ctx
                .request_repaint_after(REPAINT_INTERVAL.saturating_sub(at.elapsed()));
        }
    }

    pub fn hold_transport(&self, held: bool) {
        self.transport_hold.store(held, Ordering::Release);
    }
}

/// A running or finished export.
pub struct ExportJob {
    pub pipeline_label: &'static str,
    pub monitor: ExportMonitor,
    handle: Option<JoinHandle<()>>,
}

impl ExportJob {
    pub fn progress(&self) -> ExportProgress {
        self.monitor
            .progress
            .lock()
            .expect("export progress poisoned")
            .clone()
    }

    pub fn cancel(&self) {
        self.monitor.cancel.store(true, Ordering::Release);
    }

    pub fn is_finished(&self) -> bool {
        self.handle
            .as_ref()
            .is_none_or(|handle| handle.is_finished())
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.monitor.hold_transport(false);
    }
}

impl Drop for ExportJob {
    fn drop(&mut self) {
        self.cancel();
        self.monitor.hold_transport(false);
    }
}

/// Identifies which option of a pipeline a file dialog result belongs to.
pub type PickField = u32;

/// What a pipeline wants the user to choose.
#[derive(Clone, Debug)]
pub enum PickKind {
    Folder,
    File {
        /// Optional `(description, extensions)` filter.
        filter: Option<(&'static str, &'static [&'static str])>,
    },
}

/// Runs native file dialogs off the UI thread and routes results back to the
/// pipeline field that asked for them.
pub struct FilePicker {
    dialog: rfd::FileDialog,
    pending: HashSet<PickField>,
    results: Option<Receiver<(PickField, Option<PathBuf>)>>,
    sender: mpsc::Sender<(PickField, Option<PathBuf>)>,
}

impl FilePicker {
    pub fn new(dialog: rfd::FileDialog) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            dialog,
            pending: HashSet::new(),
            results: Some(receiver),
            sender,
        }
    }

    pub fn is_pending(&self, field: PickField) -> bool {
        self.pending.contains(&field)
    }

    pub fn request(
        &mut self,
        ctx: &egui::Context,
        field: PickField,
        title: &str,
        kind: PickKind,
        start_dir: Option<PathBuf>,
    ) {
        if !self.pending.insert(field) {
            return;
        }
        let mut dialog = self.dialog.clone().set_title(title);
        if let Some(dir) = start_dir.filter(|dir| dir.is_dir()) {
            dialog = dialog.set_directory(dir);
        }
        let sender = self.sender.clone();
        let repaint = ctx.clone();
        thread::Builder::new()
            .name("chiaro-export-picker".to_owned())
            .spawn(move || {
                let picked = match kind {
                    PickKind::Folder => dialog.pick_folder(),
                    PickKind::File { filter } => {
                        if let Some((name, extensions)) = filter {
                            dialog = dialog.add_filter(name, extensions);
                        }
                        dialog.pick_file()
                    }
                };
                let _ = sender.send((field, picked));
                repaint.request_repaint();
            })
            .expect("failed to start export file picker");
    }

    /// Collect finished dialogs.
    pub fn poll(&mut self) -> Vec<(PickField, PathBuf)> {
        let mut picked = Vec::new();
        if let Some(receiver) = &self.results {
            for (field, path) in receiver.try_iter() {
                self.pending.remove(&field);
                if let Some(path) = path {
                    picked.push((field, path));
                }
            }
        }
        picked
    }
}

/// Factory calibration files found on a connected camera and mirrored to a
/// local cache, keyed by lower-case file name (`hotpixel.rec`,
/// `calibration.lri`, `zoom_calib_v0.lri`).
#[derive(Clone, Debug)]
pub enum DeviceCalibration {
    Downloading,
    Ready(std::collections::HashMap<String, PathBuf>),
    Failed(String),
}

impl DeviceCalibration {
    /// Local copy of a camera file once downloaded.
    pub fn file(&self, name: &str) -> Option<&PathBuf> {
        match self {
            Self::Ready(files) => files.get(&name.to_ascii_lowercase()),
            _ => None,
        }
    }
}

/// Where the export's captures come from, which decides calibration defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportSource {
    /// Captures on the connected camera: its calibration is authoritative.
    Camera,
    /// Captures in a folder; a connected camera's files are a good default.
    Folder,
}

/// Host services a pipeline may use while drawing its options.
pub struct ExportUiServices<'a> {
    pub ctx: &'a egui::Context,
    pub picker: &'a mut FilePicker,
    pub source: ExportSource,
    /// Calibration files of the connected camera, when one is connected and
    /// the files were located on it.
    pub device_calibration: Option<&'a DeviceCalibration>,
}

/// A path option that defaults to a file from the connected camera.
///
/// The camera's copy is adopted whenever the field is empty or still holds a
/// previously adopted camera path, so a manual choice is never overwritten.
#[derive(Clone, Debug, Default)]
pub struct CalibrationPath {
    pub value: String,
    adopted: Option<PathBuf>,
}

impl CalibrationPath {
    pub fn adopt(&mut self, calibration: Option<&DeviceCalibration>, file_name: &str) {
        let Some(path) = calibration.and_then(|c| c.file(file_name)) else {
            return;
        };
        let untouched = self.value.trim().is_empty()
            || self
                .adopted
                .as_ref()
                .is_some_and(|previous| previous.display().to_string() == self.value);
        if untouched {
            self.value = path.display().to_string();
            self.adopted = Some(path.clone());
        }
    }

    /// Set from a file dialog: counts as a manual choice.
    pub fn set_manual(&mut self, path: PathBuf) {
        self.value = path.display().to_string();
        self.adopted = None;
    }

    /// `true` while the field shows the camera's own copy.
    pub fn is_from_camera(&self) -> bool {
        self.adopted
            .as_ref()
            .is_some_and(|p| p.display().to_string() == self.value)
    }

    pub fn path(&self) -> Option<PathBuf> {
        let trimmed = self.value.trim();
        (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
    }
}

/// One kind of export.
pub trait ExportPipeline {
    /// Snapshot the configured pipeline for a queued job. Later edits in a
    /// second export dialog must not change work that is already queued.
    fn clone_box(&self) -> Box<dyn ExportPipeline>;

    /// Menu and dialog title.
    fn label(&self) -> &'static str;
    /// One-line description shown in the pipeline menu.
    fn description(&self) -> &'static str;

    /// Draw the pipeline's options. Called every frame while the dialog is open.
    fn options_ui(&mut self, ui: &mut egui::Ui, services: &mut ExportUiServices<'_>);
    /// Receive the path chosen by a file dialog requested from `options_ui`.
    fn apply_pick(&mut self, field: PickField, path: PathBuf);

    /// Check the options before a job may start.
    fn validate(&self) -> Result<(), String>;
    /// Directory the job writes into; used for the free-space check.
    fn output_dir(&self) -> PathBuf;
    /// Upper-bound size of everything the job would write.
    fn estimate(&self, targets: &[ExportTarget]) -> ExportEstimate;

    /// Start the job on a background thread.
    fn start(&self, targets: Vec<ExportTarget>, monitor: ExportMonitor) -> JoinHandle<()>;
}

/// A validated export request waiting for its turn. The pipeline is cloned at
/// submission time so every queued request retains its own destination and
/// processing options.
pub struct PendingExport {
    pipeline: Box<dyn ExportPipeline>,
    targets: Vec<ExportTarget>,
}

impl PendingExport {
    pub fn label(&self) -> &'static str {
        self.pipeline.label()
    }

    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    pub fn start(self, ctx: &egui::Context, transport_hold: Arc<AtomicBool>) -> ExportJob {
        let pipeline_label = self.pipeline.label();
        let monitor = ExportMonitor::new(
            ExportProgress {
                total: self.targets.len(),
                ..ExportProgress::default()
            },
            transport_hold,
            ctx.clone(),
        );
        let handle = self.pipeline.start(self.targets, monitor.clone());
        ExportJob {
            pipeline_label,
            monitor,
            handle: Some(handle),
        }
    }
}

/// Every available pipeline plus the one used most recently.
pub struct ExportRegistry {
    pipelines: Vec<Box<dyn ExportPipeline>>,
    last_used: usize,
}

impl ExportRegistry {
    pub fn new() -> Self {
        Self {
            pipelines: vec![
                Box::new(HotpixelExport::default()),
                Box::new(FusionExport::default()),
            ],
            last_used: 0,
        }
    }

    pub fn labels(&self) -> impl Iterator<Item = (usize, &'static str, &'static str)> + '_ {
        self.pipelines
            .iter()
            .enumerate()
            .map(|(index, pipeline)| (index, pipeline.label(), pipeline.description()))
    }

    pub fn last_used(&self) -> usize {
        self.last_used
    }

    pub fn set_last_used(&mut self, index: usize) {
        if index < self.pipelines.len() {
            self.last_used = index;
        }
    }

    pub fn get(&self, index: usize) -> Option<&dyn ExportPipeline> {
        self.pipelines.get(index).map(|pipeline| pipeline.as_ref())
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut (dyn ExportPipeline + 'static)> {
        self.pipelines
            .get_mut(index)
            .map(|pipeline| pipeline.as_mut())
    }

    /// Snapshot `index` and its current options for immediate or queued use.
    pub fn prepare(&mut self, index: usize, targets: Vec<ExportTarget>) -> Option<PendingExport> {
        let pipeline = self.pipelines.get(index)?;
        self.last_used = index;
        Some(PendingExport {
            pipeline: pipeline.clone_box(),
            targets,
        })
    }
}

/// Presentation of one path option: text field plus dialog buttons.
pub struct PathRow {
    pub field: PickField,
    pub hint: &'static str,
    pub title: &'static str,
    pub kind: PickKind,
    pub clearable: bool,
}

pub fn path_row(
    ui: &mut egui::Ui,
    services: &mut ExportUiServices<'_>,
    value: &mut String,
    row: PathRow,
) {
    let PathRow {
        field,
        hint,
        title,
        kind,
        clearable,
    } = row;
    ui.horizontal(|ui| {
        let buttons = if clearable { 150.0 } else { 92.0 };
        let width = (ui.available_width() - buttons).max(120.0);
        ui.add_sized(
            [width, 26.0],
            egui::TextEdit::singleline(value).hint_text(hint),
        );
        let pending = services.picker.is_pending(field);
        if ui
            .add_enabled(!pending, egui::Button::new("Browse..."))
            .clicked()
        {
            let start = Path::new(value.trim());
            let start_dir = if start.is_dir() {
                Some(start.to_path_buf())
            } else {
                start.parent().map(Path::to_path_buf)
            };
            services
                .picker
                .request(services.ctx, field, title, kind, start_dir);
        }
        if clearable
            && ui
                .add_enabled(!value.is_empty(), egui::Button::new("Clear"))
                .clicked()
        {
            value.clear();
        }
    });
}

/// Status line under a calibration path: where the file came from and the
/// state of the camera download.
pub fn calibration_status(
    ui: &mut egui::Ui,
    services: &ExportUiServices<'_>,
    field: &CalibrationPath,
    file_name: &str,
) {
    use eframe::egui::{Color32, RichText};
    let muted = Color32::from_gray(150);
    match services.device_calibration {
        Some(DeviceCalibration::Downloading) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new(format!("Copying {file_name} from the connected camera..."))
                        .color(muted)
                        .size(12.0),
                );
            });
        }
        Some(ready @ DeviceCalibration::Ready(_)) => match ready.file(file_name) {
            Some(path) if field.is_from_camera() => {
                ui.label(
                    RichText::new(match services.source {
                        ExportSource::Camera => format!("Using the camera's own {file_name}."),
                        ExportSource::Folder => {
                            format!("Using {file_name} from the connected camera.")
                        }
                    })
                    .color(Color32::from_rgb(105, 205, 135))
                    .size(12.0),
                );
                let _ = path;
            }
            Some(_) => {
                ui.label(
                    RichText::new(format!(
                        "A manual path is set; clear it to use the camera's {file_name}."
                    ))
                    .color(muted)
                    .size(12.0),
                );
            }
            None => {
                ui.label(
                    RichText::new(format!(
                        "{file_name} was not found on the connected camera."
                    ))
                    .color(muted)
                    .size(12.0),
                );
            }
        },
        Some(DeviceCalibration::Failed(error)) => {
            ui.label(
                RichText::new(format!(
                    "Could not copy calibration from the camera: {error}"
                ))
                .color(Color32::from_rgb(225, 125, 125))
                .size(12.0),
            );
        }
        None if services.source == ExportSource::Camera => {
            ui.label(
                RichText::new(format!(
                    "The connected camera's {file_name} was not found in \
                     DCIM/Camera/lightcal."
                ))
                .color(Color32::from_rgb(225, 170, 105))
                .size(12.0),
            );
        }
        None => {}
    }
}

/// Human-readable byte count.
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if value >= 100.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_target_stems_are_filesystem_safe() {
        let target = ExportTarget {
            name: "L16_04480.lri".to_owned(),
            capture: CaptureLocator::Local(PathBuf::from("/x/L16_04480.lri")),
            frames: None,
        };
        assert_eq!(target.stem(), "L16_04480");
        let odd = ExportTarget {
            name: "night sky (1).LRI".to_owned(),
            capture: CaptureLocator::Local(PathBuf::from("/x/y")),
            frames: None,
        };
        assert_eq!(odd.stem(), "night_sky__1_");
    }

    #[test]
    fn local_targets_are_probed_for_their_raw_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("L16_00001.lri");
        std::fs::write(&path, chiaro::mock::MockCapture::small().encode().unwrap()).unwrap();
        let target = ExportTarget::new("L16_00001.lri".to_owned(), CaptureLocator::Local(path));
        let frames = target.frames.expect("local capture probes");
        assert_eq!(frames.len(), 3);
        assert!(
            frames
                .iter()
                .any(|frame| frame.camera == "A2" && frame.mono)
        );
        assert!(
            frames
                .iter()
                .all(|frame| (frame.width, frame.height) == (64, 48))
        );
    }

    #[test]
    fn progress_fraction_blends_completed_and_current() {
        let progress = ExportProgress {
            completed: 1,
            total: 4,
            current_fraction: 0.5,
            ..ExportProgress::default()
        };
        assert!((progress.fraction() - 0.375).abs() < 1e-6);
        assert_eq!(ExportProgress::default().fraction(), 0.0);
    }

    #[test]
    fn prepared_exports_snapshot_pipeline_options() {
        let mut registry = ExportRegistry::new();
        let first = PathBuf::from("/tmp/chiaro-first-export");
        let second = PathBuf::from("/tmp/chiaro-second-export");
        registry.get_mut(0).unwrap().apply_pick(1, first.clone());
        let pending = registry
            .prepare(
                0,
                vec![ExportTarget::new(
                    "L16_00001.lri".to_owned(),
                    CaptureLocator::Local(PathBuf::from("/tmp/L16_00001.lri")),
                )],
            )
            .unwrap();
        registry.get_mut(0).unwrap().apply_pick(1, second);

        assert_eq!(pending.pipeline.output_dir(), first);
        assert_eq!(pending.target_count(), 1);
        assert_eq!(pending.label(), "Hot-pixel corrected frames");
    }

    #[test]
    fn sizes_format_with_sensible_units() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1_500_000), "1.5 MB");
        assert_eq!(format_size(78_000_000_000), "78.0 GB");
        assert_eq!(format_size(512_000_000_000), "512 GB");
    }
}
