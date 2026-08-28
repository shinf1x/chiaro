//! Export entry points: the status-bar button, the pipeline menu, the options
//! dialog with its disk-space check, and running-job progress.
//!
//! Everything pipeline-specific is behind `crate::export::ExportPipeline`;
//! this file only hosts whichever pipeline the user picked.

use std::{path::PathBuf, thread};

use super::*;
use crate::{
    export::{ExportEstimate, ExportSource, ExportTarget, ExportUiServices, disk, format_size},
    gallery::calibration_cache,
    source::{ObjectLocator, read_preview},
};

/// Reject exports whose estimate leaves less than this much room on the disk.
const FREE_SPACE_MARGIN: u64 = 256 * 1_000_000;

pub(super) struct ExportDialog {
    pipeline: usize,
    targets: Vec<ExportTarget>,
    /// Output directory the free-space figure was last computed for.
    free_space_dir: Option<PathBuf>,
    free_space: Option<Result<u64, String>>,
    /// USB location of the camera whose calibration files apply, if any.
    device_location: Option<u64>,
    /// Exact device-matched persistent calibration for local captures.
    cached_calibration: Option<DeviceCalibration>,
    source: ExportSource,
    /// Selected night-mode captures left out of the job.
    skipped_night: Vec<String>,
    error: Option<String>,
}

impl GalleryApp {
    /// Right-hand side of the status bar: selection count, Export, pipeline menu.
    pub(super) fn export_controls(&mut self, ui: &mut egui::Ui) {
        let selected = self.selected_count();
        let job_running = self
            .export_job
            .as_ref()
            .is_some_and(|job| !job.progress().finished);
        let can_export = selected > 0 && self.export_dialog.is_none();
        let accent = ui.visuals().selection.bg_fill;

        // One split button: the main segment runs the last-used pipeline, the
        // narrow arrow segment opens the pipeline menu. Laid out right-to-left,
        // so it is the rightmost element of the status bar.
        let pipeline_label = self
            .exports
            .get(self.exports.last_used())
            .map_or("the last used pipeline", |pipeline| pipeline.label());
        let split = split_button(ui, "Export", can_export, accent);
        let mut chosen = None;
        egui::Popup::menu(&split.arrow)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClick)
            .show(|ui| {
                ui.set_min_width(280.0);
                let last = self.exports.last_used();
                for (index, label, description) in self.exports.labels() {
                    let text = if index == last {
                        RichText::new(label).strong()
                    } else {
                        RichText::new(label)
                    };
                    if ui.button(text).on_hover_text(description).clicked() {
                        chosen = Some(index);
                    }
                }
            });
        let hover = if selected == 0 {
            "Select captures to export (click a card outside its photo)".to_owned()
        } else if job_running || !self.export_queue.is_empty() {
            format!(
                "Add {selected} capture{} to the export queue with {pipeline_label}",
                if selected == 1 { "" } else { "s" }
            )
        } else {
            format!(
                "Export {selected} capture{} with {pipeline_label}",
                if selected == 1 { "" } else { "s" }
            )
        };
        if split.main.clicked() {
            chosen = Some(self.exports.last_used());
        }
        split.main.on_hover_text(hover.clone());
        split.arrow.on_hover_text(if can_export {
            "Choose an export pipeline".to_owned()
        } else {
            hover
        });

        if selected > 0 {
            if ui
                .add(egui::Button::new(RichText::new("Clear").size(12.0)).small())
                .on_hover_text("Deselect all")
                .clicked()
            {
                self.clear_selection();
            }
            ui.label(RichText::new(format!("{selected} selected")).color(Color32::from_gray(200)));
        }

        if let Some(index) = chosen
            && can_export
        {
            self.open_export_dialog(index, ui.ctx());
        }
    }

    fn open_export_dialog(&mut self, pipeline: usize, ctx: &egui::Context) {
        if self.exports.get(pipeline).is_none() {
            return;
        }
        self.exports.set_last_used(pipeline);
        // Night-mode captures are stacks of 40+ short frames that need their
        // own processing; the pipelines skip them rather than mangle them.
        let mut skipped_night = Vec::new();
        let targets = self
            .current_view
            .items
            .iter()
            .filter(|item| self.current_view.selected.contains(&item.id))
            .filter(|item| {
                let night =
                    matches!(&item.state, ItemState::Ready { metadata, .. } if metadata.night_mode);
                if night {
                    skipped_night.push(item.source.name.clone());
                }
                !night
            })
            .map(|item| ExportTarget::new(item.source.name.clone(), item.source.capture.clone()))
            .collect::<Vec<_>>();
        if targets.is_empty() && skipped_night.is_empty() {
            return;
        }
        let source = match &self.active_tab {
            TabKey::Device { .. } => ExportSource::Camera,
            _ => ExportSource::Folder,
        };
        // Remote captures use the camera they live on. Local captures use only
        // a persistent calibration whose physical device id matches every
        // selected LRI; an arbitrary connected camera is never substituted.
        let device_location = (source == ExportSource::Camera)
            .then(|| self.connected_calibration_source())
            .flatten();
        let target_device_id = targets.first().and_then(|first| {
            let id = first.device_id?;
            targets
                .iter()
                .all(|target| target.device_id == Some(id))
                .then_some(id)
        });
        let cached_calibration = target_device_id
            .and_then(|id| calibration_cache::load_for_device_id(id).ok().flatten())
            .map(|cached| DeviceCalibration::Ready(cached.files));
        if let Some((location, objects)) = device_location.clone() {
            let should_download = !matches!(
                self.device_calibrations.get(&location),
                Some(DeviceCalibration::Downloading | DeviceCalibration::Ready(_))
            );
            if should_download {
                self.download_device_calibration(location, objects, ctx);
            }
        }
        self.export_dialog = Some(ExportDialog {
            pipeline,
            targets,
            free_space_dir: None,
            free_space: None,
            device_location: device_location.map(|(location, _)| location),
            cached_calibration,
            source,
            skipped_night,
            error: None,
        });
    }

    /// The connected camera whose calibration files were located, preferring
    /// the camera of the active tab.
    fn connected_calibration_source(&self) -> Option<(u64, Vec<RemoteObject>)> {
        let connected = |location: u64| self.devices.iter().any(|d| d.location_id == location);
        let from_view = |key: &TabKey, view: &TabViewState| match key {
            TabKey::Device { location_id, .. }
                if connected(*location_id) && !view.device_calibration.is_empty() =>
            {
                Some((*location_id, view.device_calibration.clone()))
            }
            _ => None,
        };
        from_view(&self.active_tab, &self.current_view).or_else(|| {
            self.saved_views
                .iter()
                .find_map(|(key, view)| from_view(key, view))
        })
    }

    /// Copy the camera's calibration files into the local cache in the background.
    pub(super) fn download_device_calibration(
        &mut self,
        location: u64,
        objects: Vec<RemoteObject>,
        ctx: &egui::Context,
    ) {
        self.device_calibrations
            .insert(location, DeviceCalibration::Downloading);
        let sender = self.calibration_sender.clone();
        let hold = self.loader.transport_hold();
        let repaint = ctx.clone();
        let label = self
            .devices
            .iter()
            .find(|device| device.location_id == location)
            .and_then(|device| device.serial_number.clone())
            .unwrap_or_else(|| format!("usb-{location}"));
        if let Ok(Some(cached)) = calibration_cache::load_for_label(&label) {
            self.device_calibrations
                .insert(location, DeviceCalibration::Ready(cached.files));
            return;
        }
        hold.store(true, std::sync::atomic::Ordering::Release);
        thread::Builder::new()
            .name("chiaro-calibration-download".to_owned())
            .spawn(move || {
                let result = (|| {
                    let mut files = HashMap::new();
                    for object in objects {
                        let name = object.name.to_ascii_lowercase();
                        let bytes = read_preview(&ObjectLocator::Device(object))?;
                        files.insert(name, bytes);
                    }
                    calibration_cache::store_device_files(&label, files).map(|cached| cached.files)
                })();
                hold.store(false, std::sync::atomic::Ordering::Release);
                let _ = sender.send((location, result));
                repaint.request_repaint();
            })
            .expect("failed to start calibration download");
    }

    /// Per-frame housekeeping: dialog file pickers, calibration downloads, and
    /// job completion.
    pub(super) fn poll_exports(&mut self, ctx: &egui::Context) {
        for (field, path) in self.export_picker.poll() {
            if let Some(dialog) = &self.export_dialog
                && let Some(pipeline) = self.exports.get_mut(dialog.pipeline)
            {
                pipeline.apply_pick(field, path);
            }
        }
        if let Some(receiver) = &self.calibration_downloads {
            for (location, result) in receiver.try_iter() {
                let state = match result {
                    Ok(files) => DeviceCalibration::Ready(files),
                    Err(error) => DeviceCalibration::Failed(error),
                };
                self.device_calibrations.insert(location, state);
            }
        }
        let finished = self.export_job.as_ref().is_some_and(ExportJob::is_finished);
        if finished {
            let mut job = self
                .export_job
                .take()
                .expect("finished export job disappeared");
            job.join();
            let progress = job.progress();
            let label = job.pipeline_label;
            let succeeded = progress
                .succeeded_hashes
                .iter()
                .cloned()
                .collect::<HashSet<_>>();
            let exported = job
                .history_targets
                .iter()
                .filter(|identity| succeeded.contains(&identity.hash))
                .cloned()
                .collect::<Vec<_>>();
            let database = self.loader.database().cloned();
            let mut catalog_error = None;
            if let Some(database) = database {
                for identity in &exported {
                    if let Err(error) = database.record_export(identity, label, &job.output_dir) {
                        catalog_error = Some(error.to_string());
                        break;
                    }
                }
            }
            let exported_hashes = exported
                .iter()
                .map(|identity| identity.hash.clone())
                .collect::<HashSet<_>>();
            self.loader.mark_exported(exported_hashes.iter().cloned());
            self.mark_exported_cards(&exported_hashes);

            let mut summary = if let Some(error) = &progress.fatal {
                format!("{label}: export failed - {error}")
            } else if job.monitor.cancelled() {
                format!(
                    "{label}: export cancelled after {} of {} captures",
                    progress.completed, progress.total
                )
            } else if progress.failures.is_empty() {
                format!(
                    "{label}: exported {} frames from {} captures",
                    progress.outputs_written, progress.total
                )
            } else {
                format!(
                    "{label}: exported {} frames; {} captures failed (see export-log.txt)",
                    progress.outputs_written,
                    progress.failures.len()
                )
            };
            if let Some(error) = catalog_error {
                summary.push_str(&format!("; could not update gallery.sqlite3: {error}"));
            }
            self.current_view.status = Some(summary);
            ctx.request_repaint();
        }
        if self.export_job.is_none()
            && let Some(pending) = self.export_queue.pop_front()
        {
            let hold = self.loader.transport_hold();
            self.export_job = Some(pending.start(ctx, hold));
            self.current_view.status = None;
            ctx.request_repaint();
        }
    }

    fn mark_exported_cards(&mut self, hashes: &HashSet<String>) {
        let mark = |view: &mut TabViewState| {
            for item in &mut view.items {
                if item
                    .capture_hash
                    .as_ref()
                    .is_some_and(|hash| hashes.contains(hash))
                {
                    item.exported = true;
                }
            }
        };
        mark(&mut self.current_view);
        for view in self.saved_views.values_mut() {
            mark(view);
        }
    }

    /// Status-bar text and progress while a job runs. Returns `true` when it
    /// drew something, so the caller can skip the idle message.
    pub(super) fn export_status(&mut self, ui: &mut egui::Ui, pending_previews: usize) -> bool {
        let Some(job) = &self.export_job else {
            return false;
        };
        let progress = job.progress();
        if progress.finished {
            return false;
        }
        // No spinner here: it would request a repaint every frame for the
        // whole export, and the progress bar already shows activity.
        ui.label(format!(
            "Exporting {} / {}: {}",
            (progress.completed + 1).min(progress.total.max(1)),
            progress.total,
            progress.current
        ));
        if let Some((name, fraction)) = &progress.transfer {
            ui.label(
                RichText::new(format!(
                    "copying {name} from camera {:.0}%",
                    fraction * 100.0
                ))
                .color(Color32::from_gray(150)),
            );
        }
        if pending_previews > 0 {
            let message = if self.loader.export_transport_held() {
                format!("{pending_previews} previews waiting for camera copy")
            } else {
                format!("{pending_previews} previews loading")
            };
            ui.label(RichText::new(message).color(Color32::from_gray(150)));
        }
        if let Some(next) = self.export_queue.front() {
            ui.label(
                RichText::new(format!(
                    "{} queued; next: {} ({} capture{})",
                    self.export_queue.len(),
                    next.label(),
                    next.target_count(),
                    if next.target_count() == 1 { "" } else { "s" }
                ))
                .color(Color32::from_gray(150)),
            );
        }
        let width = (ui.available_width() - 90.0).clamp(60.0, 320.0);
        ui.add(egui::ProgressBar::new(progress.fraction()).desired_width(width));
        let cancel_label = if self.export_queue.is_empty() {
            "Cancel"
        } else {
            "Cancel current"
        };
        if ui.small_button(cancel_label).clicked() {
            job.cancel();
        }
        true
    }

    pub(super) fn show_export_dialog(&mut self, ctx: &egui::Context, status_bar_top: f32) {
        let Some(dialog) = self.export_dialog.as_mut() else {
            return;
        };
        let Some(pipeline) = self.exports.get_mut(dialog.pipeline) else {
            self.export_dialog = None;
            return;
        };

        // Refresh the free-space figure whenever the destination changes.
        let output_dir = pipeline.output_dir();
        if dialog.free_space_dir.as_ref() != Some(&output_dir) {
            dialog.free_space_dir = Some(output_dir.clone());
            dialog.free_space = (!output_dir.as_os_str().is_empty())
                .then(|| disk::available_space(&output_dir).map_err(|error| error.to_string()));
        }
        let estimate = pipeline.estimate(&dialog.targets);
        let validation = pipeline.validate();
        let free = dialog
            .free_space
            .as_ref()
            .and_then(|result| result.as_ref().ok().copied());
        let insufficient = free.is_some_and(|free| estimate.bytes + FREE_SPACE_MARGIN > free);
        if dialog.cached_calibration.is_none()
            && let Some(device_id) = dialog.targets.first().and_then(|target| target.device_id)
            && dialog
                .targets
                .iter()
                .all(|target| target.device_id == Some(device_id))
            && let Ok(Some(cached)) = calibration_cache::load_for_device_id(device_id)
        {
            dialog.cached_calibration = Some(DeviceCalibration::Ready(cached.files));
        }
        let device_calibration = dialog
            .device_location
            .and_then(|location| self.device_calibrations.get(&location))
            .or(dialog.cached_calibration.as_ref());
        let calibration_pending =
            matches!(device_calibration, Some(DeviceCalibration::Downloading));
        let will_queue = self.export_job.is_some() || !self.export_queue.is_empty();

        let viewport = ctx.content_rect();
        let mut usable = viewport;
        usable.max.y = usable.max.y.min(status_bar_top);
        let width = (usable.width() * 0.9).min(640.0);
        let modal_id = egui::Id::new("export-dialog");
        let mut start = false;
        let mut cancel = false;
        let response = egui::Modal::new(modal_id)
            .area(egui::Modal::default_area(modal_id).anchor(
                egui::Align2::CENTER_CENTER,
                usable.center() - viewport.center(),
            ))
            .backdrop_color(Color32::from_black_alpha(170))
            .frame(
                egui::Frame::popup(&ctx.style())
                    .fill(Color32::from_rgb(24, 27, 32))
                    .corner_radius(CornerRadius::same(10))
                    .inner_margin(Margin::same(18)),
            )
            .show(ctx, |ui| {
                ui.set_width(width);
                ui.horizontal(|ui| {
                    ui.heading(format!("Export - {}", pipeline.label()));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Close").clicked() {
                            cancel = true;
                        }
                    });
                });
                ui.label(
                    RichText::new(format!(
                        "{} capture{} selected - {}",
                        dialog.targets.len(),
                        if dialog.targets.len() == 1 { "" } else { "s" },
                        pipeline.description()
                    ))
                    .color(Color32::from_gray(155)),
                );
                if !dialog.skipped_night.is_empty() {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(58, 48, 30))
                        .corner_radius(CornerRadius::same(6))
                        .inner_margin(Margin::symmetric(10, 6))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{} night-mode capture{} will be skipped: {}. Night mode \
                                     stacks dozens of short frames and is not supported by the \
                                     export pipelines yet.",
                                    dialog.skipped_night.len(),
                                    if dialog.skipped_night.len() == 1 {
                                        ""
                                    } else {
                                        "s"
                                    },
                                    dialog.skipped_night.join(", ")
                                ))
                                .color(Color32::from_rgb(240, 205, 140)),
                            );
                        });
                }
                ui.separator();

                let mut services = ExportUiServices {
                    ctx,
                    picker: &mut self.export_picker,
                    source: dialog.source,
                    device_calibration,
                };
                pipeline.options_ui(ui, &mut services);

                ui.separator();
                disk_summary(ui, &estimate, dialog.free_space.as_ref(), insufficient);
                if let Some(error) = &dialog.error {
                    ui.colored_label(Color32::from_rgb(225, 125, 125), error);
                }

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let ready = validation.is_ok()
                        && !insufficient
                        && !calibration_pending
                        && !dialog.targets.is_empty();
                    let action = if will_queue {
                        "Add to queue"
                    } else {
                        "Start export"
                    };
                    let button = egui::Button::new(RichText::new(action).strong())
                        .min_size(Vec2::new(120.0, 30.0))
                        .fill(if ready {
                            ui.visuals().selection.bg_fill
                        } else {
                            Color32::from_rgb(43, 46, 54)
                        });
                    let mut response = ui.add_enabled(ready, button);
                    if dialog.targets.is_empty() {
                        response = response
                            .on_disabled_hover_text("Every selected capture is a night-mode stack");
                    } else if let Err(reason) = &validation {
                        response = response.on_disabled_hover_text(reason.clone());
                    } else if insufficient {
                        response = response
                            .on_disabled_hover_text("Not enough free space at the destination");
                    } else if calibration_pending {
                        response = response
                            .on_disabled_hover_text("Waiting for the camera's hotpixel.rec");
                    }
                    if response.clicked() {
                        start = true;
                    }
                    if ui
                        .add(egui::Button::new("Cancel").min_size(Vec2::new(120.0, 30.0)))
                        .clicked()
                    {
                        cancel = true;
                    }
                    if let Err(reason) = &validation {
                        ui.label(
                            RichText::new(reason)
                                .color(Color32::from_gray(150))
                                .size(12.0),
                        );
                    }
                });
            });

        if response.should_close() || cancel {
            self.export_dialog = None;
            return;
        }
        if start {
            let targets = dialog.targets.clone();
            let pipeline_index = dialog.pipeline;
            match self.exports.prepare(pipeline_index, targets) {
                Some(pending) => {
                    if self.export_job.is_some() || !self.export_queue.is_empty() {
                        self.export_queue.push_back(pending);
                    } else {
                        let hold = self.loader.transport_hold();
                        self.export_job = Some(pending.start(ctx, hold));
                    }
                    self.export_dialog = None;
                    self.current_view.status = None;
                    self.clear_selection();
                }
                None => {
                    if let Some(dialog) = self.export_dialog.as_mut() {
                        dialog.error = Some("The selected pipeline is unavailable".to_owned());
                    }
                }
            }
        }
    }
}

/// Responses of the two segments of a [`split_button`].
struct SplitButton {
    main: egui::Response,
    arrow: egui::Response,
}

/// A primary action with an attached dropdown arrow, drawn as one rounded
/// button with a thin divider, as in most desktop toolkits.
fn split_button(ui: &mut egui::Ui, label: &str, enabled: bool, accent: Color32) -> SplitButton {
    const HEIGHT: f32 = 26.0;
    const MAIN_WIDTH: f32 = 74.0;
    const ARROW_WIDTH: f32 = 22.0;
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(MAIN_WIDTH + ARROW_WIDTH, HEIGHT), Sense::hover());
    let main_rect = egui::Rect::from_min_size(rect.min, Vec2::new(MAIN_WIDTH, HEIGHT));
    let arrow_rect = egui::Rect::from_min_max(egui::pos2(main_rect.max.x, rect.min.y), rect.max);
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let id = ui.id().with(label);
    let main = ui.interact(main_rect, id.with("main"), sense);
    let arrow = ui.interact(arrow_rect, id.with("arrow"), sense);

    let fill = if enabled {
        accent
    } else {
        Color32::from_rgb(43, 46, 54)
    };
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(4), fill);
    // Highlight only the hovered segment so the two targets read as distinct.
    for (segment, response, left) in [(main_rect, &main, true), (arrow_rect, &arrow, false)] {
        if enabled && response.hovered() {
            let radius = if left {
                CornerRadius {
                    nw: 4,
                    sw: 4,
                    ne: 0,
                    se: 0,
                }
            } else {
                CornerRadius {
                    nw: 0,
                    sw: 0,
                    ne: 4,
                    se: 4,
                }
            };
            painter.rect_filled(segment, radius, Color32::from_white_alpha(22));
        }
    }
    let divider_x = main_rect.max.x;
    painter.line_segment(
        [
            egui::pos2(divider_x, rect.min.y + 5.0),
            egui::pos2(divider_x, rect.max.y - 5.0),
        ],
        Stroke::new(1.0_f32, Color32::from_black_alpha(110)),
    );
    let text_color = if enabled {
        Color32::WHITE
    } else {
        Color32::from_gray(150)
    };
    painter.text(
        main_rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        FontId::proportional(14.0),
        text_color,
    );
    paint_down_triangle(ui, arrow_rect, text_color);
    SplitButton { main, arrow }
}

/// The default fonts have no reliable glyph for a dropdown caret; draw one.
fn paint_down_triangle(ui: &egui::Ui, rect: egui::Rect, color: Color32) {
    let center = rect.center();
    let half = 4.5;
    ui.painter().add(egui::Shape::convex_polygon(
        vec![
            center + Vec2::new(-half, -half * 0.55),
            center + Vec2::new(half, -half * 0.55),
            center + Vec2::new(0.0, half * 0.6),
        ],
        color,
        Stroke::NONE,
    ));
}

fn disk_summary(
    ui: &mut egui::Ui,
    estimate: &ExportEstimate,
    free: Option<&Result<u64, String>>,
    insufficient: bool,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new("Disk usage:").strong());
        ui.label(format!(
            "{}{} needed",
            if estimate.approximate { "~" } else { "up to " },
            format_size(estimate.bytes)
        ));
        match free {
            Some(Ok(free)) => {
                ui.label("-");
                ui.label(
                    RichText::new(format!("{} free at the destination", format_size(*free))).color(
                        if insufficient {
                            Color32::from_rgb(225, 125, 125)
                        } else {
                            Color32::from_rgb(105, 205, 135)
                        },
                    ),
                );
            }
            Some(Err(error)) => {
                ui.label("-");
                ui.label(
                    RichText::new(format!("free space unknown ({error})"))
                        .color(Color32::from_gray(150)),
                );
            }
            None => {}
        }
    });
    ui.label(
        RichText::new(&estimate.detail)
            .color(Color32::from_gray(140))
            .size(12.0),
    );
    if insufficient {
        ui.colored_label(
            Color32::from_rgb(225, 125, 125),
            "Not enough free space: choose another destination or export fewer captures.",
        );
    }
}
