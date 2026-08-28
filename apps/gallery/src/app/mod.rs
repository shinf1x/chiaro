use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use chiaro::lri::{CaptureDateTime, CaptureMetadata, PreviewImage};
use eframe::egui::{
    self, Align, Color32, CornerRadius, FontId, Layout, Margin, RichText, Sense, Stroke,
    StrokeKind, Vec2,
};

use crate::{
    export::{DeviceCalibration, ExportJob, ExportRegistry, FilePicker, PendingExport},
    gallery::{
        GalleryItem, ItemState, LoadedEvent, PreviewKey, PreviewLoader, database::GalleryDatabase,
        overlay::CaptureOverlayGeometry,
    },
    source::{
        CaptureData, DeviceMode, DeviceMonitor, LightDevice, RemoteObject, SourceItem, local_items,
    },
};

mod branding;
mod cards;
mod events;
mod export_ui;
mod modal;
mod settings;
mod tabs;
mod visuals;

use export_ui::ExportDialog;
use modal::{modal_state, upload_preview};
use settings::Settings;
use visuals::*;

use branding::BrandTextures;

const CARD_GAP: f32 = 4.0;
const CARD_EXTRA_HEIGHT: f32 = 88.0;
const MIN_PREVIEW_SIZE: f32 = 172.5;
const DEVICE_DISCONNECT_GRACE: Duration = Duration::from_secs(5);
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TabKey {
    Device { location_id: u64, mode: DeviceMode },
    FolderInput,
    Folder(u64),
    Settings,
}

impl TabKey {
    fn for_device(device: &LightDevice) -> Self {
        Self::Device {
            location_id: device.location_id,
            mode: device.mode,
        }
    }
}

/// Result of mirroring a camera's calibration files: USB location and the
/// local copies by lower-case file name.
type CalibrationDownload = (u64, Result<HashMap<String, PathBuf>, String>);

struct FolderTab {
    id: u64,
    path: PathBuf,
}

#[derive(Default)]
struct TabViewState {
    loaded: bool,
    items: Vec<GalleryItem>,
    status: Option<String>,
    busy: Option<String>,
    listing_progress: Option<(usize, usize)>,
    /// Ids of items marked for export.
    selected: HashSet<u64>,
    /// Item that anchors the next shift-click range.
    selection_anchor: Option<u64>,
    /// Factory calibration files located on the connected camera.
    device_calibration: Vec<RemoteObject>,
}

pub struct GalleryApp {
    brand: BrandTextures,
    active_tab: TabKey,
    current_view: TabViewState,
    saved_views: HashMap<TabKey, TabViewState>,
    indexing_tab: Option<TabKey>,
    folder_prompt_open: bool,
    devices: Vec<LightDevice>,
    device_monitor: DeviceMonitor,
    folder_input: String,
    folder_tabs: Vec<FolderTab>,
    next_folder_id: u64,
    folder_dialog: rfd::FileDialog,
    folder_picker: Option<Receiver<Option<PathBuf>>>,
    loader: PreviewLoader,
    generation: u64,
    next_item_id: u64,
    preview_size: f32,
    device_retry_at: Option<Instant>,
    device_missing_since: Option<Instant>,
    contact_sheet: Option<ContactSheet>,
    next_modal_id: u64,
    exports: ExportRegistry,
    export_dialog: Option<ExportDialog>,
    export_job: Option<ExportJob>,
    export_queue: VecDeque<PendingExport>,
    export_picker: FilePicker,
    /// Camera `hotpixel.rec` copies keyed by USB location id.
    device_calibrations: HashMap<u64, DeviceCalibration>,
    calibration_downloads: Option<Receiver<CalibrationDownload>>,
    calibration_sender: mpsc::Sender<CalibrationDownload>,
    settings: Settings,
}

struct ContactSheet {
    id: u64,
    name: String,
    state: ContactState,
    full: Option<FullPreview>,
}

enum ContactState {
    Loading {
        transferred: u64,
        total: u64,
        reference_camera: Option<String>,
        cameras: Vec<CameraPreview>,
    },
    Ready {
        reference_camera: String,
        cameras: Vec<CameraPreview>,
        data: CaptureData,
        metadata: CaptureMetadata,
        overlay_geometry: Option<CaptureOverlayGeometry>,
    },
    Failed(String),
}

struct CameraPreview {
    camera: String,
    frame_index: u64,
    state: ModalImageState,
}

struct FullPreview {
    camera: String,
    frame_index: u64,
    show_frame: bool,
    state: ModalImageState,
    zoom: f32,
    pan: Vec2,
    auto_fit: bool,
}

enum ModalImageState {
    Pending,
    Ready {
        texture: egui::TextureHandle,
        dimensions: [usize; 2],
        color_calibrated: bool,
        orientation: u64,
    },
    Failed(String),
}

impl GalleryApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let database = settings::database_path().and_then(|path| {
            GalleryDatabase::open(&path)
                .map(Arc::new)
                .map_err(|error| eprintln!("gallery database {}: {error}", path.display()))
                .ok()
        });
        let loader = PreviewLoader::new(cc.egui_ctx.clone(), database);
        let generation = loader.begin_source_load();
        let device_monitor = DeviceMonitor::new(cc.egui_ctx.clone());
        let folder_dialog = rfd::FileDialog::new().set_parent(cc);
        let export_picker = FilePicker::new(folder_dialog.clone());
        let (calibration_sender, calibration_downloads) = mpsc::channel();
        let app = Self {
            brand: BrandTextures::new(&cc.egui_ctx),
            active_tab: TabKey::FolderInput,
            current_view: TabViewState::default(),
            saved_views: HashMap::new(),
            indexing_tab: None,
            folder_prompt_open: false,
            devices: Vec::new(),
            device_monitor,
            folder_input: String::new(),
            folder_tabs: Vec::new(),
            next_folder_id: 1,
            folder_dialog,
            folder_picker: None,
            loader,
            generation,
            next_item_id: 0,
            preview_size: 210.0,
            device_retry_at: None,
            device_missing_since: None,
            contact_sheet: None,
            next_modal_id: 1,
            exports: ExportRegistry::new(),
            export_dialog: None,
            export_job: None,
            export_queue: VecDeque::new(),
            export_picker,
            device_calibrations: HashMap::new(),
            calibration_downloads: Some(calibration_downloads),
            calibration_sender,
            settings: Settings::load(),
        };
        app.apply_settings();
        app
    }

    fn open_folder_input(&mut self) {
        let input = self.folder_input.trim();
        if !input.is_empty() {
            self.add_folder_tab(PathBuf::from(input));
        }
    }

    fn add_folder_tab(&mut self, folder: PathBuf) {
        let key = if let Some(tab) = self.folder_tabs.iter().find(|tab| tab.path == folder) {
            TabKey::Folder(tab.id)
        } else {
            let id = self.next_folder_id;
            self.next_folder_id += 1;
            self.folder_tabs.push(FolderTab {
                id,
                path: folder.clone(),
            });
            TabKey::Folder(id)
        };
        self.folder_input = folder.display().to_string();
        self.select_tab(key, true);
    }

    fn select_tab(&mut self, key: TabKey, force_reload: bool) {
        if !force_reload && self.active_tab == key && self.current_view.loaded {
            return;
        }
        if let Some(sheet) = self.contact_sheet.take() {
            self.loader.cancel_modal(sheet.id);
        }
        if self.current_view.loaded {
            self.saved_views.insert(
                self.active_tab.clone(),
                std::mem::take(&mut self.current_view),
            );
        } else {
            self.current_view = TabViewState::default();
        }
        self.active_tab = key;
        if self.active_tab != TabKey::FolderInput {
            self.folder_prompt_open = false;
        }
        if force_reload {
            self.saved_views.remove(&self.active_tab);
        }
        if !force_reload && let Some(view) = self.saved_views.remove(&self.active_tab) {
            self.current_view = view;
            return;
        }
        self.load_active_tab();
    }

    fn load_active_tab(&mut self) {
        self.current_view = TabViewState::default();
        match self.active_tab.clone() {
            TabKey::FolderInput | TabKey::Settings => {}
            TabKey::Folder(id) => {
                let Some(folder) = self
                    .folder_tabs
                    .iter()
                    .find(|tab| tab.id == id)
                    .map(|tab| tab.path.clone())
                else {
                    self.active_tab = TabKey::FolderInput;
                    return;
                };
                match local_items(&folder) {
                    Ok(items) => self.install_items(items, folder.display().to_string()),
                    Err(error) => {
                        self.current_view.loaded = true;
                        self.current_view.status = Some(format!("Could not read folder: {error}"));
                    }
                }
            }
            TabKey::Device { location_id, mode } => {
                let Some(device) = self
                    .devices
                    .iter()
                    .find(|device| device.location_id == location_id && device.mode == mode)
                    .cloned()
                else {
                    self.current_view.loaded = true;
                    self.current_view.status = Some("The L16 is no longer connected".to_owned());
                    return;
                };
                self.current_view.loaded = true;
                self.current_view.busy = Some(format!(
                    "Preparing Light L16 {} connection...",
                    device.mode.label()
                ));
                self.indexing_tab = Some(self.active_tab.clone());
                self.loader.list_device(self.generation, device);
            }
        }
    }

    fn tab_view_mut(&mut self, key: &TabKey) -> Option<&mut TabViewState> {
        if &self.active_tab == key {
            Some(&mut self.current_view)
        } else {
            self.saved_views.get_mut(key)
        }
    }

    fn gallery_item_mut(&mut self, id: u64, revision: u64) -> Option<&mut GalleryItem> {
        if let Some(item) = self
            .current_view
            .items
            .iter_mut()
            .find(|item| item.id == id && item.preview_revision == revision)
        {
            return Some(item);
        }
        self.saved_views.values_mut().find_map(|view| {
            view.items
                .iter_mut()
                .find(|item| item.id == id && item.preview_revision == revision)
        })
    }

    fn open_folder_picker(&mut self, ctx: &egui::Context) {
        if self.folder_picker.is_some() {
            return;
        }
        let (sender, receiver) = mpsc::channel();
        let repaint = ctx.clone();
        let dialog = self
            .folder_dialog
            .clone()
            .set_title("Choose a folder containing LRI captures");
        thread::Builder::new()
            .name("chiaro-folder-picker".to_owned())
            .spawn(move || {
                let picked = dialog.pick_folder();
                let _ = sender.send(picked);
                repaint.request_repaint();
            })
            .expect("failed to start folder picker");
        self.folder_picker = Some(receiver);
    }

    fn receive_folder_picker(&mut self) {
        let Some(receiver) = &self.folder_picker else {
            return;
        };
        match receiver.try_recv() {
            Ok(Some(folder)) => {
                self.folder_picker = None;
                self.add_folder_tab(folder);
            }
            Ok(None) | Err(mpsc::TryRecvError::Disconnected) => self.folder_picker = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn receive_device_snapshot(&mut self) {
        let Some(snapshot) = self.device_monitor.latest() else {
            return;
        };
        let previous = self.devices.clone();
        match snapshot {
            Err(error) => self.current_view.status = Some(error),
            Ok(mut devices) => {
                let active_removed = match &self.active_tab {
                    TabKey::Device { location_id, mode } => !devices
                        .iter()
                        .any(|device| device.location_id == *location_id && device.mode == *mode),
                    _ => false,
                };
                let newly_connected = devices
                    .iter()
                    .find(|device| !previous.contains(device))
                    .cloned();
                let disconnected = previous.iter().any(|old| !devices.contains(old));
                if let Some(device) = newly_connected {
                    // A different camera can appear at the same USB location;
                    // never reuse calibration copied for the previous device.
                    self.device_calibrations.remove(&device.location_id);
                    self.device_missing_since = None;
                    self.devices = devices;
                    if self.current_view.loaded {
                        self.saved_views.insert(
                            self.active_tab.clone(),
                            std::mem::take(&mut self.current_view),
                        );
                    }
                    self.active_tab = TabKey::for_device(&device);
                    self.current_view = TabViewState {
                        loaded: true,
                        ..Default::default()
                    };
                    self.folder_prompt_open = false;
                    self.current_view.busy = Some(format!(
                        "Waiting for Light L16 {} to become ready...",
                        device.mode.label()
                    ));
                    self.device_retry_at = Some(Instant::now() + Duration::from_millis(900));
                } else if active_removed {
                    self.device_missing_since.get_or_insert_with(Instant::now);
                    if let TabKey::Device { location_id, mode } = self.active_tab
                        && let Some(active) = previous
                            .iter()
                            .find(|device| device.location_id == location_id && device.mode == mode)
                    {
                        devices.push(active.clone());
                    }
                    self.devices = devices;
                    self.current_view.status = Some(
                        "Light L16 connection interrupted; keeping loaded previews...".to_owned(),
                    );
                } else if disconnected {
                    self.device_missing_since = None;
                    self.devices = devices;
                    self.current_view.status = Some("A Light L16 was disconnected".to_owned());
                } else {
                    self.device_missing_since = None;
                    self.devices = devices;
                    if self.current_view.status.as_deref()
                        == Some("Light L16 connection interrupted; keeping loaded previews...")
                    {
                        self.current_view.status = None;
                    }
                }
            }
        }
    }

    fn expire_missing_device(&mut self, ctx: &egui::Context) {
        let Some(since) = self.device_missing_since else {
            return;
        };
        let elapsed = since.elapsed();
        if elapsed < DEVICE_DISCONNECT_GRACE {
            ctx.request_repaint_after(DEVICE_DISCONNECT_GRACE - elapsed);
            return;
        }
        self.device_missing_since = None;
        let TabKey::Device { location_id, mode } = self.active_tab else {
            return;
        };
        self.devices
            .retain(|device| device.location_id != location_id || device.mode != mode);
        self.device_calibrations.remove(&location_id);
        self.saved_views
            .remove(&TabKey::Device { location_id, mode });
        self.active_tab = TabKey::FolderInput;
        self.current_view = TabViewState::default();
        self.folder_prompt_open = false;
        self.current_view.status = Some("Light L16 disconnected".to_owned());
    }

    fn retry_device_if_due(&mut self, ctx: &egui::Context) {
        let Some(deadline) = self.device_retry_at else {
            return;
        };
        let now = Instant::now();
        if now < deadline {
            ctx.request_repaint_after(deadline - now);
            return;
        }
        self.device_retry_at = None;
        let TabKey::Device { location_id, mode } = self.active_tab else {
            return;
        };
        if self
            .devices
            .iter()
            .any(|device| device.location_id == location_id && device.mode == mode)
        {
            self.load_active_tab();
        }
    }

    fn install_items(&mut self, items: Vec<SourceItem>, label: String) {
        let count = items.len();
        let first_id = self.next_item_id;
        self.current_view.items = self.loader.load_items(items, first_id);
        for item in &mut self.current_view.items {
            self.loader.load_gallery_preview(
                self.generation,
                item.id,
                item.preview_revision,
                item.source.preview.clone(),
                item.source.capture.clone(),
            );
            item.state = ItemState::Pending {
                transferred: 0,
                total: 0,
            };
        }
        self.next_item_id += count as u64;
        let _ = label;
        self.current_view.loaded = true;
        self.current_view.busy = None;
        self.current_view.listing_progress = None;
        self.current_view.status =
            (count == 0).then(|| "No .lri files found in this source".to_owned());
    }
}

impl eframe::App for GalleryApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive_device_snapshot();
        self.receive_folder_picker();
        let dropped_folder = ctx.input(|input| {
            input.raw.dropped_files.iter().find_map(|file| {
                let path = file.path.as_ref()?;
                if path.is_dir() {
                    Some(path.clone())
                } else if path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("lri"))
                {
                    path.parent().map(ToOwned::to_owned)
                } else {
                    None
                }
            })
        });
        if let Some(folder) = dropped_folder {
            self.add_folder_tab(folder);
        }
        self.receive_events(ctx);
        self.retry_device_if_due(ctx);
        self.expire_missing_device(ctx);
        self.poll_exports(ctx);

        egui::TopBottomPanel::top("toolbar")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(24, 26, 31))
                    .inner_margin(BrandTextures::toolbar_margin(ctx))
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgb(48, 51, 59))),
            )
            .show(ctx, |ui| self.toolbar(ui));

        let status_panel = egui::TopBottomPanel::bottom("status-bar")
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(24, 26, 31))
                    .inner_margin(Margin::symmetric(18, 8))
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgb(48, 51, 59))),
            )
            .show(ctx, |ui| self.status_bar(ui));

        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(18, 20, 24))
                    .inner_margin(Margin::same(18)),
            )
            .show(ctx, |ui| {
                if self.active_tab == TabKey::Settings {
                    self.settings_view(ui);
                } else if self.current_view.items.is_empty() {
                    if self.current_view.busy.is_none() && self.current_view.status.is_none() {
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                RichText::new(
                                    "Connect a Light L16, drop a folder, or choose + Folder...",
                                )
                                .size(17.0)
                                .color(Color32::from_gray(165)),
                            );
                        });
                    }
                } else {
                    self.gallery(ui);
                }
            });

        self.show_contact_sheet(ctx, status_panel.response.rect.top());
        self.show_export_dialog(ctx, status_panel.response.rect.top());
    }
}
