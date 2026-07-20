use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
};

use chiaro::lri::{
    CaptureMetadata, CaptureSummary, PreviewImage, decode_camera_frame_preview,
    decode_camera_frame_preview_bytes, inspect_capture, inspect_capture_bytes,
};
use eframe::egui;

use crate::source::{
    CaptureData, CaptureLocator, CapturePreviewUpdate, DeviceItemEvent, LightDevice,
    PreviewLocator, SourceItem, apply_preview_orientation, decode_jpeg_preview,
    decode_reference_source, device_items, read_capture_metadata, read_capture_with_updates,
    read_preview,
};

mod worker;

use worker::{preview_is_current, worker};

const DECODE_EDGE: usize = 720;

pub struct GalleryItem {
    pub id: u64,
    pub preview_revision: u64,
    pub source: SourceItem,
    pub state: ItemState,
}

pub enum ItemState {
    Idle,
    Pending {
        transferred: u64,
        total: u64,
    },
    Ready {
        texture: egui::TextureHandle,
        camera: String,
        dimensions: [usize; 2],
        metadata: CaptureMetadata,
    },
    Failed(String),
}

enum Task {
    Preview {
        generation: u64,
        key: PreviewKey,
        source: PreviewLocator,
        capture: CaptureLocator,
        max_edge: usize,
        promoted: bool,
    },
    Camera {
        generation: u64,
        key: PreviewKey,
        data: CaptureData,
        camera: String,
        frame_index: u64,
        max_edge: usize,
    },
    OpenCapture {
        generation: u64,
        modal: u64,
        source: CaptureLocator,
    },
    ListDevice {
        generation: u64,
        device: LightDevice,
    },
}

enum WorkerResult {
    Preview {
        generation: u64,
        key: PreviewKey,
        result: Result<PreviewImage, String>,
    },
    PreviewProgress {
        generation: u64,
        key: PreviewKey,
        transferred: u64,
        total: u64,
    },
    PreviewSkipped {
        generation: u64,
        key: PreviewKey,
    },
    Capture {
        generation: u64,
        modal: u64,
        result: Result<LoadedCapture, String>,
    },
    CaptureProgress {
        generation: u64,
        modal: u64,
        transferred: u64,
        total: u64,
    },
    CapturePreview {
        generation: u64,
        modal: u64,
        reference_camera: String,
        frame_index: u64,
        preview: PreviewImage,
    },
    DeviceDone {
        generation: u64,
        result: Result<(), String>,
    },
    DeviceItem {
        generation: u64,
        item: SourceItem,
    },
    DevicePreview {
        generation: u64,
        capture_name: String,
        preview: PreviewLocator,
    },
    DeviceProgress {
        generation: u64,
        fetched: usize,
        total: usize,
    },
    DeviceStatus {
        generation: u64,
        message: String,
    },
}

impl WorkerResult {
    fn generation(&self) -> u64 {
        match self {
            Self::Preview { generation, .. }
            | Self::PreviewProgress { generation, .. }
            | Self::PreviewSkipped { generation, .. }
            | Self::Capture { generation, .. }
            | Self::CaptureProgress { generation, .. }
            | Self::CapturePreview { generation, .. }
            | Self::DeviceDone { generation, .. }
            | Self::DeviceItem { generation, .. }
            | Self::DevicePreview { generation, .. }
            | Self::DeviceProgress { generation, .. }
            | Self::DeviceStatus { generation, .. } => *generation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewKey {
    Gallery {
        id: u64,
        revision: u64,
    },
    Contact {
        modal: u64,
        camera: String,
        frame_index: u64,
    },
    Full {
        modal: u64,
        camera: String,
        frame_index: u64,
    },
}

pub struct LoadedCapture {
    pub summary: CaptureSummary,
    pub data: CaptureData,
}

pub enum LoadedEvent {
    Preview {
        key: PreviewKey,
        result: Result<PreviewImage, String>,
    },
    PreviewProgress {
        key: PreviewKey,
        transferred: u64,
        total: u64,
    },
    PreviewSkipped {
        key: PreviewKey,
    },
    Capture {
        modal: u64,
        result: Result<LoadedCapture, String>,
    },
    CaptureProgress {
        modal: u64,
        transferred: u64,
        total: u64,
    },
    CapturePreview {
        modal: u64,
        reference_camera: String,
        frame_index: u64,
        preview: PreviewImage,
    },
    DeviceDone(Result<(), String>),
    DeviceItem(SourceItem),
    DevicePreview {
        capture_name: String,
        preview: PreviewLocator,
    },
    DeviceProgress {
        fetched: usize,
        total: usize,
    },
    DeviceStatus(String),
}

#[derive(Default)]
struct GalleryQueueState {
    visible: Mutex<HashSet<u64>>,
    revisions: Mutex<HashMap<u64, u64>>,
    claimed: Mutex<HashSet<(u64, u64, u64)>>,
    device_indexing: AtomicBool,
    modal_loading: AtomicBool,
}

pub struct PreviewLoader {
    tasks: Sender<Task>,
    priority_tasks: Sender<Task>,
    modal_tasks: Sender<Task>,
    results: Receiver<WorkerResult>,
    generation: Arc<AtomicU64>,
    gallery_queue: Arc<GalleryQueueState>,
    promoted_gallery: Mutex<HashSet<(u64, u64, u64)>>,
    active_modals: Arc<Mutex<HashSet<u64>>>,
}

#[derive(Clone)]
struct WorkerQueues {
    tasks: Arc<Mutex<Receiver<Task>>>,
    priority: Arc<Mutex<Receiver<Task>>>,
    modal: Arc<Mutex<Receiver<Task>>>,
}

impl PreviewLoader {
    pub fn new(ctx: egui::Context) -> Self {
        let (task_tx, task_rx) = mpsc::channel::<Task>();
        let (priority_tx, priority_rx) = mpsc::channel::<Task>();
        let (modal_tx, modal_rx) = mpsc::channel::<Task>();
        let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();
        let task_rx = Arc::new(Mutex::new(task_rx));
        let priority_rx = Arc::new(Mutex::new(priority_rx));
        let modal_rx = Arc::new(Mutex::new(modal_rx));
        let generation = Arc::new(AtomicU64::new(0));
        let gallery_queue = Arc::new(GalleryQueueState::default());
        let active_modals = Arc::new(Mutex::new(HashSet::new()));
        let queues = WorkerQueues {
            tasks: task_rx,
            priority: priority_rx,
            modal: modal_rx,
        };
        for thread_index in 0..2 {
            let queues = queues.clone();
            let results = result_tx.clone();
            let generation = Arc::clone(&generation);
            let gallery_queue = Arc::clone(&gallery_queue);
            let active_modals = Arc::clone(&active_modals);
            let ctx = ctx.clone();
            thread::Builder::new()
                .name(format!("lri-preview-{thread_index}"))
                .spawn(move || {
                    worker(
                        queues,
                        results,
                        generation,
                        gallery_queue,
                        active_modals,
                        ctx,
                    )
                })
                .expect("failed to start preview worker");
        }

        Self {
            tasks: task_tx,
            priority_tasks: priority_tx,
            modal_tasks: modal_tx,
            results: result_rx,
            generation,
            gallery_queue,
            promoted_gallery: Mutex::new(HashSet::new()),
            active_modals,
        }
    }

    pub fn begin_source_load(&self) -> u64 {
        self.gallery_queue
            .visible
            .lock()
            .expect("visible preview set poisoned")
            .clear();
        self.gallery_queue
            .revisions
            .lock()
            .expect("gallery revisions poisoned")
            .clear();
        self.gallery_queue
            .claimed
            .lock()
            .expect("claimed gallery set poisoned")
            .clear();
        self.gallery_queue
            .device_indexing
            .store(false, Ordering::Release);
        self.gallery_queue
            .modal_loading
            .store(false, Ordering::Release);
        self.promoted_gallery
            .lock()
            .expect("promoted gallery set poisoned")
            .clear();
        self.active_modals
            .lock()
            .expect("active modal set poisoned")
            .clear();
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub fn set_visible_gallery(&self, ids: impl IntoIterator<Item = u64>) {
        let new_visible = ids.into_iter().collect::<HashSet<_>>();
        *self
            .gallery_queue
            .visible
            .lock()
            .expect("visible preview set poisoned") = new_visible.clone();
        self.promoted_gallery
            .lock()
            .expect("promoted gallery set poisoned")
            .retain(|(_, id, _)| new_visible.contains(id));
    }

    pub fn load_items(&self, sources: Vec<SourceItem>, first_id: u64) -> Vec<GalleryItem> {
        let mut items = Vec::with_capacity(sources.len());
        for (offset, source) in sources.into_iter().enumerate() {
            let state = ItemState::Idle;
            items.push(GalleryItem {
                id: first_id + offset as u64,
                preview_revision: 0,
                source,
                state,
            });
        }
        items
    }

    pub fn list_device(&self, generation: u64, device: LightDevice) {
        let _ = self
            .priority_tasks
            .send(Task::ListDevice { generation, device });
    }

    pub fn load_gallery_preview(
        &self,
        generation: u64,
        id: u64,
        revision: u64,
        source: PreviewLocator,
        capture: CaptureLocator,
    ) {
        self.gallery_queue
            .revisions
            .lock()
            .expect("gallery revisions poisoned")
            .insert(id, revision);
        self.promoted_gallery
            .lock()
            .expect("promoted gallery set poisoned")
            .retain(|(_, promoted_id, _)| *promoted_id != id);
        let _ = self.tasks.send(Task::Preview {
            generation,
            key: PreviewKey::Gallery { id, revision },
            source,
            capture,
            max_edge: DECODE_EDGE,
            promoted: false,
        });
    }

    pub fn prioritize_gallery_preview(
        &self,
        generation: u64,
        id: u64,
        revision: u64,
        source: PreviewLocator,
        capture: CaptureLocator,
    ) {
        if !preview_is_current(&self.gallery_queue.revisions, id, revision) {
            return;
        }
        let promotion = (generation, id, revision);
        if !self
            .promoted_gallery
            .lock()
            .expect("promoted gallery set poisoned")
            .insert(promotion)
        {
            return;
        }
        let _ = self.priority_tasks.send(Task::Preview {
            generation,
            key: PreviewKey::Gallery { id, revision },
            source,
            capture,
            max_edge: DECODE_EDGE,
            promoted: true,
        });
    }

    pub fn open_capture(&self, generation: u64, modal: u64, source: CaptureLocator) {
        self.gallery_queue
            .modal_loading
            .store(true, Ordering::Release);
        self.active_modals
            .lock()
            .expect("active modal set poisoned")
            .insert(modal);
        let _ = self.modal_tasks.send(Task::OpenCapture {
            generation,
            modal,
            source,
        });
    }

    pub fn cancel_modal(&self, modal: u64) {
        self.active_modals
            .lock()
            .expect("active modal set poisoned")
            .remove(&modal);
        self.gallery_queue
            .modal_loading
            .store(false, Ordering::Release);
    }

    pub fn set_modal_loading(&self, loading: bool) {
        self.gallery_queue
            .modal_loading
            .store(loading, Ordering::Release);
    }

    pub fn retry_gallery_preview(&self, generation: u64, id: u64, revision: u64) {
        self.promoted_gallery
            .lock()
            .expect("promoted gallery set poisoned")
            .remove(&(generation, id, revision));
        self.gallery_queue
            .claimed
            .lock()
            .expect("claimed gallery set poisoned")
            .remove(&(generation, id, revision));
    }

    pub fn load_camera(
        &self,
        generation: u64,
        key: PreviewKey,
        data: CaptureData,
        camera: String,
        frame_index: u64,
        max_edge: usize,
    ) {
        self.gallery_queue
            .modal_loading
            .store(true, Ordering::Release);
        let _ = self.modal_tasks.send(Task::Camera {
            generation,
            key,
            data,
            camera,
            frame_index,
            max_edge,
        });
    }

    pub fn drain(&self, active_generation: u64) -> Vec<LoadedEvent> {
        self.results
            .try_iter()
            .filter(|result| result.generation() == active_generation)
            .map(|result| match result {
                WorkerResult::Preview { key, result, .. } => LoadedEvent::Preview { key, result },
                WorkerResult::PreviewProgress {
                    key,
                    transferred,
                    total,
                    ..
                } => LoadedEvent::PreviewProgress {
                    key,
                    transferred,
                    total,
                },
                WorkerResult::PreviewSkipped { key, .. } => LoadedEvent::PreviewSkipped { key },
                WorkerResult::Capture { modal, result, .. } => {
                    LoadedEvent::Capture { modal, result }
                }
                WorkerResult::CaptureProgress {
                    modal,
                    transferred,
                    total,
                    ..
                } => LoadedEvent::CaptureProgress {
                    modal,
                    transferred,
                    total,
                },
                WorkerResult::CapturePreview {
                    modal,
                    reference_camera,
                    frame_index,
                    preview,
                    ..
                } => LoadedEvent::CapturePreview {
                    modal,
                    reference_camera,
                    frame_index,
                    preview,
                },
                WorkerResult::DeviceDone { result, .. } => LoadedEvent::DeviceDone(result),
                WorkerResult::DeviceItem { item, .. } => LoadedEvent::DeviceItem(item),
                WorkerResult::DevicePreview {
                    capture_name,
                    preview,
                    ..
                } => LoadedEvent::DevicePreview {
                    capture_name,
                    preview,
                },
                WorkerResult::DeviceProgress { fetched, total, .. } => {
                    LoadedEvent::DeviceProgress { fetched, total }
                }
                WorkerResult::DeviceStatus { message, .. } => LoadedEvent::DeviceStatus(message),
            })
            .collect()
    }
}
