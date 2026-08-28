use super::*;

pub(super) fn worker(
    queues: WorkerQueues,
    results: Sender<WorkerResult>,
    active_generation: Arc<AtomicU64>,
    gallery_queue: Arc<GalleryQueueState>,
    active_modals: Arc<Mutex<HashSet<u64>>>,
    ctx: egui::Context,
) {
    loop {
        let modal = {
            let receiver = queues.modal.lock().expect("modal task queue poisoned");
            receiver.try_recv().ok()
        };
        if modal.is_none() && gallery_queue.preview_transport_busy() {
            thread::sleep(std::time::Duration::from_millis(20));
            continue;
        }
        let priority = {
            let receiver = queues
                .priority
                .lock()
                .expect("priority task queue poisoned");
            receiver.try_recv().ok()
        };
        let task = modal.or(priority).or_else(|| {
            let receiver = queues.tasks.lock().expect("preview task queue poisoned");
            receiver
                .recv_timeout(std::time::Duration::from_millis(100))
                .ok()
        });
        let Some(task) = task else { continue };
        let task_generation = match &task {
            Task::Preview { generation, .. }
            | Task::OpenCapture { generation, .. }
            | Task::ListDevice { generation, .. } => *generation,
        };
        if active_generation.load(Ordering::Acquire) != task_generation {
            continue;
        }
        let result = match task {
            Task::Preview {
                generation,
                key,
                source,
                capture,
                max_edge,
                promoted,
            } => {
                if let PreviewKey::Gallery { id, revision } = &key
                    && ((promoted && !gallery_is_visible(&gallery_queue.visible, *id))
                        || !preview_is_current(&gallery_queue.revisions, *id, *revision)
                        || !gallery_queue
                            .claimed
                            .lock()
                            .expect("claimed gallery set poisoned")
                            .insert((generation, *id, *revision)))
                {
                    continue;
                }
                let progress_results = results.clone();
                let progress_ctx = ctx.clone();
                let progress_key = key.clone();
                let decoded = decode_preview_source(
                    gallery_queue.thumbnails.as_deref(),
                    &source,
                    &capture,
                    max_edge,
                    |transferred, total| {
                        let _ = progress_results.send(WorkerResult::PreviewProgress {
                            generation,
                            key: progress_key.clone(),
                            transferred,
                            total,
                        });
                        progress_ctx.request_repaint();
                    },
                    || {
                        generation == task_generation
                            && generation == active_generation.load(Ordering::Acquire)
                            && match &progress_key {
                                PreviewKey::Gallery { id, revision } => {
                                    preview_is_current(&gallery_queue.revisions, *id, *revision)
                                }
                                _ => true,
                            }
                    },
                );
                match decoded {
                    Ok(Some(preview)) => WorkerResult::Preview {
                        generation,
                        key,
                        result: Ok(preview),
                    },
                    Ok(None) => WorkerResult::PreviewSkipped { generation, key },
                    Err(error) => WorkerResult::Preview {
                        generation,
                        key,
                        result: Err(error),
                    },
                }
            }
            Task::OpenCapture {
                generation,
                modal,
                source,
            } => {
                let progress_results = results.clone();
                let progress_ctx = ctx.clone();
                let result = load_capture(
                    source,
                    |transferred, total| {
                        let _ = progress_results.send(WorkerResult::CaptureProgress {
                            generation,
                            modal,
                            transferred,
                            total,
                        });
                        progress_ctx.request_repaint();
                    },
                    |update| {
                        let _ = progress_results.send(WorkerResult::CapturePreview {
                            generation,
                            modal,
                            reference_camera: update.reference_camera,
                            frame_index: update.frame_index,
                            preview: update.preview,
                        });
                        progress_ctx.request_repaint();
                    },
                    || modal_is_active(&active_modals, modal),
                );
                WorkerResult::Capture {
                    generation,
                    modal,
                    result,
                }
            }
            Task::ListDevice {
                generation: list_generation,
                device,
            } => {
                gallery_queue.device_indexing.store(true, Ordering::Release);
                let progress_results = results.clone();
                let progress_ctx = ctx.clone();
                let result = device_items(
                    &device,
                    |fetched, total| {
                        let _ = progress_results.send(WorkerResult::DeviceProgress {
                            generation: list_generation,
                            fetched,
                            total,
                        });
                        progress_ctx.request_repaint();
                    },
                    |event| match event {
                        DeviceItemEvent::Add(item) => {
                            let _ = progress_results.send(WorkerResult::DeviceItem {
                                generation: list_generation,
                                item,
                            });
                        }
                        DeviceItemEvent::Preview {
                            capture_name,
                            preview,
                        } => {
                            let _ = progress_results.send(WorkerResult::DevicePreview {
                                generation: list_generation,
                                capture_name,
                                preview,
                            });
                        }
                        DeviceItemEvent::Calibration(object) => {
                            let _ = progress_results.send(WorkerResult::DeviceCalibration {
                                generation: list_generation,
                                object,
                            });
                        }
                    },
                    |message| {
                        let _ = progress_results.send(WorkerResult::DeviceStatus {
                            generation: list_generation,
                            message,
                        });
                        progress_ctx.request_repaint();
                    },
                    || active_generation.load(Ordering::Acquire) == task_generation,
                    || gallery_queue.device_indexing_should_yield(),
                );
                gallery_queue
                    .device_indexing
                    .store(false, Ordering::Release);
                WorkerResult::DeviceDone {
                    generation: list_generation,
                    result,
                }
            }
        };

        if active_generation.load(Ordering::Acquire) == task_generation {
            let _ = results.send(result);
            ctx.request_repaint();
        }
    }
}

pub(super) fn camera_worker(
    queues: CameraWorkerQueues,
    results: Sender<WorkerResult>,
    active_generation: Arc<AtomicU64>,
    active_modals: Arc<Mutex<HashSet<u64>>>,
    ctx: egui::Context,
) {
    loop {
        let priority = {
            let receiver = queues
                .priority
                .lock()
                .expect("priority camera task queue poisoned");
            receiver.try_recv().ok()
        };
        let task = priority.or_else(|| {
            let receiver = queues.tasks.lock().expect("camera task queue poisoned");
            receiver
                .recv_timeout(std::time::Duration::from_millis(100))
                .ok()
        });
        let Some(task) = task else { continue };
        let Some(modal) = preview_modal(&task.key) else {
            continue;
        };
        if active_generation.load(Ordering::Acquire) != task.generation
            || !modal_is_active(&active_modals, modal)
        {
            continue;
        }

        let result = decode_camera(&task.data, &task.camera, task.frame_index, task.max_edge);
        if active_generation.load(Ordering::Acquire) == task.generation
            && modal_is_active(&active_modals, modal)
        {
            let _ = results.send(WorkerResult::Preview {
                generation: task.generation,
                key: task.key,
                result,
            });
            ctx.request_repaint();
        }
    }
}

fn preview_modal(key: &PreviewKey) -> Option<u64> {
    match key {
        PreviewKey::Contact { modal, .. } | PreviewKey::Full { modal, .. } => Some(*modal),
        PreviewKey::Gallery { .. } => None,
    }
}

/// Decode a gallery preview, serving it from the thumbnail cache when the
/// same capture was decoded before and storing fresh decodes for next time.
fn decode_preview_source(
    thumbnails: Option<&cache::ThumbnailCache>,
    source: &PreviewLocator,
    capture: &CaptureLocator,
    max_edge: usize,
    mut on_progress: impl FnMut(u64, u64),
    mut should_continue: impl FnMut() -> bool,
) -> Result<Option<PreviewImage>, String> {
    let key = thumbnails.and_then(|_| cache::ThumbnailKey::for_preview(source, capture, max_edge));
    if let (Some(cache), Some(key)) = (thumbnails, key.as_ref())
        && let Some(preview) = cache.load(key)
    {
        return Ok(Some(preview));
    }
    let decoded = match source {
        PreviewLocator::Lri(locator) => {
            let mut decoded =
                decode_reference_source(locator, max_edge, &mut on_progress, &mut should_continue)?;
            // The reference module is the wide view; show what was framed. A
            // local file is cheap to decode again at the edge the crop needs so
            // the thumbnail keeps its resolution; a camera download is not.
            if let (Some(preview), CaptureLocator::Local(_)) = (&decoded, locator)
                && let Some(edge) = cache::framed_decode_edge(preview, max_edge)
            {
                decoded =
                    decode_reference_source(locator, edge, &mut on_progress, &mut should_continue)?;
            }
            decoded.map(|mut preview| {
                cache::crop_to_framing(&mut preview);
                preview
            })
        }
        PreviewLocator::Jpeg(locator) => {
            let bytes = read_preview(locator)?;
            let mut preview = decode_jpeg_preview(&bytes, max_edge)?;
            let metadata = match read_capture_metadata(capture, should_continue) {
                Ok(metadata) => metadata,
                Err(error) if error == "camera load cancelled" => return Ok(None),
                Err(error) => return Err(error),
            };
            let orientation = metadata.orientation;
            preview.metadata = metadata;
            apply_preview_orientation(&mut preview, orientation);
            Some(preview)
        }
    };
    if let (Some(cache), Some(key), Some(preview)) = (thumbnails, key.as_ref(), decoded.as_ref())
        && let Err(error) = cache.store(key, preview)
    {
        eprintln!("thumbnail cache: could not store {}: {error}", key.as_str());
    }
    Ok(decoded)
}

fn load_capture(
    source: CaptureLocator,
    on_progress: impl FnMut(u64, u64),
    on_preview: impl FnMut(CapturePreviewUpdate) + Send,
    should_continue: impl FnMut() -> bool,
) -> Result<LoadedCapture, String> {
    let data = read_capture_with_updates(&source, on_progress, on_preview, should_continue)?;
    let summary = match &data {
        CaptureData::Local(path) => inspect_capture(path),
        CaptureData::Memory(bytes) => inspect_capture_bytes(bytes),
    }
    .map_err(|error| error.to_string())?;
    Ok(LoadedCapture { summary, data })
}

pub(super) fn preview_is_current(
    revisions: &Mutex<HashMap<u64, u64>>,
    id: u64,
    revision: u64,
) -> bool {
    revisions
        .lock()
        .expect("gallery revisions poisoned")
        .get(&id)
        .is_some_and(|current| *current == revision)
}

fn gallery_is_visible(visible: &Mutex<HashSet<u64>>, id: u64) -> bool {
    visible
        .lock()
        .expect("visible preview set poisoned")
        .contains(&id)
}

fn modal_is_active(active: &Mutex<HashSet<u64>>, modal: u64) -> bool {
    active
        .lock()
        .expect("active modal set poisoned")
        .contains(&modal)
}

fn decode_camera(
    data: &CaptureData,
    camera: &str,
    frame_index: u64,
    max_edge: usize,
) -> Result<PreviewImage, String> {
    match data {
        CaptureData::Local(path) => {
            decode_camera_frame_preview(path, camera, frame_index, max_edge)
        }
        CaptureData::Memory(bytes) => {
            decode_camera_frame_preview_bytes(bytes, camera, frame_index, max_edge)
        }
    }
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_processing_does_not_pause_preview_workers_between_camera_transfers() {
        let queue = GalleryQueueState::default();
        assert!(!queue.preview_transport_busy());

        queue.transport_hold.store(true, Ordering::Release);
        assert!(queue.preview_transport_busy());

        queue.transport_hold.store(false, Ordering::Release);
        assert!(!queue.preview_transport_busy());
    }
}
