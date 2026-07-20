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
        if modal.is_none() && gallery_queue.device_indexing.load(Ordering::Acquire) {
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
            | Task::Camera { generation, .. }
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
            Task::Camera {
                generation,
                key,
                data,
                camera,
                max_edge,
            } => WorkerResult::Preview {
                generation,
                key,
                result: decode_camera(&data, &camera, max_edge),
            },
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
                    },
                    |message| {
                        let _ = progress_results.send(WorkerResult::DeviceStatus {
                            generation: list_generation,
                            message,
                        });
                        progress_ctx.request_repaint();
                    },
                    || active_generation.load(Ordering::Acquire) == task_generation,
                    || gallery_queue.modal_loading.load(Ordering::Acquire),
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

fn decode_preview_source(
    source: &PreviewLocator,
    capture: &CaptureLocator,
    max_edge: usize,
    on_progress: impl FnMut(u64, u64),
    should_continue: impl FnMut() -> bool,
) -> Result<Option<PreviewImage>, String> {
    match source {
        PreviewLocator::Lri(locator) => {
            decode_reference_source(locator, max_edge, on_progress, should_continue)
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
            Ok(Some(preview))
        }
    }
}

fn load_capture(
    source: CaptureLocator,
    on_progress: impl FnMut(u64, u64),
    on_preview: impl FnMut(CapturePreviewUpdate),
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
    max_edge: usize,
) -> Result<PreviewImage, String> {
    match data {
        CaptureData::Local(path) => decode_camera_preview(path, camera, max_edge),
        CaptureData::Memory(bytes) => decode_camera_preview_bytes(bytes, camera, max_edge),
    }
    .map_err(|error| error.to_string())
}
