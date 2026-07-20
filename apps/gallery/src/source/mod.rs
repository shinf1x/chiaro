//! Capture discovery and byte transport, separate from LRI decoding.
//!
//! Local files are memory-mapped by the decoder. Connected cameras are read
//! directly with the pure-Rust `mtp-rs` PTP/MTP stack. Remote files are kept in
//! memory and never materialized as temporary previews.

use std::{
    collections::HashMap,
    fmt, fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::Duration,
};

use chiaro::lri::{
    CaptureMetadata, PreviewImage, ReferencePayloadRead, camera_frame_preview_color_ready,
    complete_lelr_prefix_len, decode_reference_preview, decode_reference_preview_bytes,
    inspect_capture, inspect_capture_metadata_blocks, inspect_capture_prefix,
    inspect_lelr_block_header, reference_camera_payload_read, reference_preview_color_ready,
    try_decode_camera_frame_preview_prefix, try_decode_reference_preview_prefix,
};
use eframe::egui;
use futures::executor::block_on;
use mtp_rs::mtp::{
    ByteRange, DEFAULT_CANCEL_TIMEOUT, MtpDevice, ObjectHandle, ObjectInfo, Storage,
};
use mtp_rs::ptp::PtpDevice;
use mtp_rs::transport::NusbTransport;
use nusb::MaybeFuture;

const LIGHT_VENDOR_ID: u16 = 0x2de7;
const LIGHT_MTP_PRODUCT_ID: u16 = 0x0005;
const LIGHT_PTP_PRODUCT_ID: u16 = 0x0007;
const LIGHT_USB_IDS: [(u16, u16); 2] = [
    (LIGHT_VENDOR_ID, LIGHT_MTP_PRODUCT_ID),
    (LIGHT_VENDOR_ID, LIGHT_PTP_PRODUCT_ID),
];
const CANCELLED_LOAD: &str = "camera load cancelled";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceMode {
    Mtp,
    Ptp,
}

impl DeviceMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Mtp => "MTP",
            Self::Ptp => "PTP",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LightDevice {
    pub location_id: u64,
    pub serial_number: Option<String>,
    pub mode: DeviceMode,
}

impl LightDevice {
    pub fn tab_label(&self) -> String {
        format!("L16 - {}", self.mode.label())
    }
}

#[derive(Clone)]
pub struct RemoteObject {
    storage: Arc<Storage>,
    handle: ObjectHandle,
    pub name: String,
    pub size: u64,
    device: LightDevice,
}

impl fmt::Debug for RemoteObject {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteObject")
            .field("handle", &self.handle)
            .field("name", &self.name)
            .field("size", &self.size)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub enum ObjectLocator {
    Device(RemoteObject),
}

#[derive(Clone, Debug)]
pub enum CaptureLocator {
    Local(PathBuf),
    Device(RemoteObject),
}

#[derive(Clone, Debug)]
pub enum PreviewLocator {
    Lri(CaptureLocator),
    Jpeg(ObjectLocator),
}

#[derive(Clone, Debug)]
pub struct SourceItem {
    pub name: String,
    pub capture: CaptureLocator,
    pub preview: PreviewLocator,
    pub location_label: String,
}

pub enum DeviceItemEvent {
    Add(SourceItem),
    Preview {
        capture_name: String,
        preview: PreviewLocator,
    },
}

pub struct CapturePreviewUpdate {
    pub reference_camera: String,
    pub frame_index: u64,
    pub preview: PreviewImage,
}

#[derive(Clone)]
pub enum CaptureData {
    Local(PathBuf),
    Memory(Arc<Vec<u8>>),
}

pub struct DeviceMonitor {
    snapshots: Receiver<Result<Vec<LightDevice>, String>>,
    stop: Arc<AtomicBool>,
}

impl DeviceMonitor {
    pub fn new(ctx: egui::Context) -> Self {
        let (sender, snapshots) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        thread::Builder::new()
            .name("lri-device-monitor".to_owned())
            .spawn(move || {
                let mut previous = None;
                while !thread_stop.load(Ordering::Acquire) {
                    let snapshot = discover_light_devices();
                    let changed = match (&previous, &snapshot) {
                        (Some(Ok(old)), Ok(new)) => old != new,
                        (Some(Err(old)), Err(new)) => old != new,
                        (None, _) | (Some(_), _) => true,
                    };
                    if changed {
                        previous = Some(snapshot.clone());
                        if sender.send(snapshot).is_err() {
                            return;
                        }
                        ctx.request_repaint();
                    }
                    thread::sleep(Duration::from_secs(1));
                }
            })
            .expect("failed to start device monitor");
        Self { snapshots, stop }
    }

    pub fn latest(&self) -> Option<Result<Vec<LightDevice>, String>> {
        self.snapshots.try_iter().last()
    }
}

impl Drop for DeviceMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

mod discovery;
mod jpeg;
mod transfer;

pub use discovery::{device_items, discover_light_devices, local_items};
pub use jpeg::decode_jpeg_preview;
pub use transfer::{
    apply_preview_orientation, decode_reference_source, read_capture_metadata,
    read_capture_with_updates, read_preview,
};

use jpeg::{file_name, file_stem_lower, has_extension, lri_paths};
