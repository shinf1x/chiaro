use super::*;
use crate::parallel;

const LELR_HEADER_BYTES: usize = 32;
const CONTACT_TRANSFER_WINDOW: u32 = 16 * 1024 * 1024;

pub fn read_capture_with_updates(
    locator: &CaptureLocator,
    on_progress: impl FnMut(u64, u64),
    on_preview: impl FnMut(CapturePreviewUpdate) + Send,
    should_continue: impl FnMut() -> bool,
) -> Result<CaptureData, String> {
    match locator {
        CaptureLocator::Local(path) => Ok(CaptureData::Local(path.clone())),
        CaptureLocator::Device(object) => {
            download_capture_with_updates(object, on_progress, on_preview, should_continue)
                .map(|data| CaptureData::Memory(Arc::new(data)))
        }
    }
}

pub fn read_preview(locator: &ObjectLocator) -> Result<Vec<u8>, String> {
    match locator {
        ObjectLocator::Device(object) => download_object(object),
    }
}

pub fn read_capture_metadata(
    locator: &CaptureLocator,
    should_continue: impl FnMut() -> bool,
) -> Result<CaptureMetadata, String> {
    match locator {
        CaptureLocator::Local(path) => inspect_capture(path)
            .map(|summary| summary.metadata)
            .map_err(|error| error.to_string()),
        CaptureLocator::Device(object) => download_capture_metadata_sparse(object, should_continue),
    }
}

pub fn decode_reference_source(
    locator: &CaptureLocator,
    max_edge: usize,
    on_progress: impl FnMut(u64, u64),
    should_continue: impl FnMut() -> bool,
) -> Result<Option<PreviewImage>, String> {
    match locator {
        CaptureLocator::Local(path) => decode_reference_preview(path, max_edge)
            .map(Some)
            .map_err(|error| error.to_string()),
        CaptureLocator::Device(object) => {
            download_reference_preview(object, max_edge, on_progress, should_continue)
        }
    }
}

fn download_object(object: &RemoteObject) -> Result<Vec<u8>, String> {
    download_object_with_progress(object, |_, _| {})
}

fn download_object_with_progress(
    object: &RemoteObject,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<Vec<u8>, String> {
    block_on(async {
        on_progress(0, object.size);
        let mut download = object
            .storage
            .download(object.handle, ByteRange::Full)
            .await
            .map_err(|error| format!("Could not start download of {}: {error}", object.name))?;
        let capacity = usize::try_from(download.size()).unwrap_or(0);
        let mut bytes = Vec::with_capacity(capacity);
        while let Some(chunk) = download.next_chunk().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    let _ = download.cancel(DEFAULT_CANCEL_TIMEOUT).await;
                    return Err(format!("Could not download {}: {error}", object.name));
                }
            };
            bytes.extend_from_slice(&chunk);
            on_progress(bytes.len() as u64, object.size);
        }
        Ok(bytes)
    })
}

fn download_capture_with_updates(
    object: &RemoteObject,
    mut on_progress: impl FnMut(u64, u64),
    mut on_preview: impl FnMut(CapturePreviewUpdate) + Send,
    mut should_continue: impl FnMut() -> bool,
) -> Result<Vec<u8>, String> {
    let decoding = AtomicBool::new(true);
    thread::scope(|scope| {
        let decoder_active = &decoding;
        let (chunk_sender, chunk_receiver) = mpsc::channel::<Vec<u8>>();
        let capacity = usize::try_from(object.size).unwrap_or(0);
        let decoder = scope.spawn(move || {
            decode_capture_windows(chunk_receiver, decoder_active, capacity, &mut on_preview)
        });

        let transfer = block_on(async {
            on_progress(0, object.size);
            let mut download = object
                .storage
                .download_windowed(object.handle, ByteRange::Full, CONTACT_TRANSFER_WINDOW)
                .await
                .map_err(|error| format!("Could not start download of {}: {error}", object.name))?;
            let mut transferred = 0u64;
            while should_continue() {
                let Some(chunk) = download.next_window().await else {
                    return Ok(());
                };
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        return Err(format!("Could not download {}: {error}", object.name));
                    }
                };
                transferred += chunk.len() as u64;
                on_progress(transferred, object.size);
                // The completed window moves into the decoder queue without an
                // extra transfer-thread copy. The USB loop immediately requests
                // the next window and never waits for parsing or demosaicing. A
                // 16 MiB window is the measured L16 throughput knee while keeping
                // transaction restarts infrequent.
                chunk_sender
                    .send(chunk)
                    .map_err(|_| "contact preview decoder stopped".to_owned())?;
            }
            Err(CANCELLED_LOAD.to_owned())
        });

        if transfer.is_err() {
            decoding.store(false, Ordering::Release);
        }
        drop(chunk_sender);
        let decoded = decoder
            .join()
            .map_err(|_| "contact preview decoder panicked".to_owned())?;
        match (transfer, decoded) {
            (_, Err(error)) => Err(error),
            (Err(error), _) => Err(error),
            (Ok(()), Ok(bytes)) => Ok(bytes),
        }
    })
}

fn decode_capture_windows(
    chunk_receiver: mpsc::Receiver<Vec<u8>>,
    active: &AtomicBool,
    capacity: usize,
    on_preview: &mut impl FnMut(CapturePreviewUpdate),
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(capacity);
    // RAW payloads commonly precede calibration blocks. Track whether each
    // early card has already been replaced by its color-corrected version.
    let mut emitted = HashMap::new();
    let mut complete_prefix = 0usize;
    while active.load(Ordering::Acquire) {
        let Ok(chunk) = chunk_receiver.recv() else {
            break;
        };
        bytes.extend_from_slice(&chunk);
        let previous_prefix = complete_prefix;
        loop {
            let Some(header_end) = complete_prefix.checked_add(LELR_HEADER_BYTES) else {
                return Err("LRI block offset overflow".to_owned());
            };
            if bytes.len() < header_end {
                break;
            }
            let descriptor =
                inspect_lelr_block_header(&bytes[complete_prefix..header_end], complete_prefix)
                    .map_err(|error| error.to_string())?;
            let block_end = descriptor.block_range().end;
            if bytes.len() < block_end {
                break;
            }
            complete_prefix = block_end;
        }
        if complete_prefix == previous_prefix {
            continue;
        }
        emit_ready_camera_previews(&bytes[..complete_prefix], &mut emitted, on_preview)?;
    }
    Ok(bytes)
}

struct CameraPreviewRequest {
    camera: String,
    frame_index: u64,
    color_ready: bool,
}

fn emit_ready_camera_previews(
    prefix: &[u8],
    emitted: &mut HashMap<(String, u64), bool>,
    on_preview: &mut impl FnMut(CapturePreviewUpdate),
) -> Result<(), String> {
    let Some(summary) = inspect_capture_prefix(prefix).map_err(|error| error.to_string())? else {
        return Ok(());
    };
    let mut requests = Vec::new();
    for camera in summary.cameras {
        let identity = (camera.camera.clone(), camera.frame_index);
        if emitted.get(&identity) == Some(&true) {
            continue;
        }
        let color_ready =
            camera_frame_preview_color_ready(prefix, &camera.camera, camera.frame_index)
                .map_err(|error| error.to_string())?
                .unwrap_or(false);
        if emitted.get(&identity) == Some(&false) && !color_ready {
            continue;
        }
        requests.push(CameraPreviewRequest {
            camera: camera.camera,
            frame_index: camera.frame_index,
            color_ready,
        });
    }

    let reference_camera = summary.reference_camera;
    let mut first_error = None;
    parallel::for_each(
        &requests,
        parallel::available_workers(),
        |request| {
            try_decode_camera_frame_preview_prefix(
                prefix,
                &request.camera,
                request.frame_index,
                520,
            )
            .map_err(|error| error.to_string())
        },
        |index, result| match result {
            Ok(Some(preview)) => {
                let request = &requests[index];
                emitted.insert(
                    (request.camera.clone(), request.frame_index),
                    request.color_ready,
                );
                on_preview(CapturePreviewUpdate {
                    reference_camera: reference_camera.clone(),
                    frame_index: request.frame_index,
                    preview,
                });
            }
            Ok(None) => {}
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        },
    );
    first_error.map_or(Ok(()), Err)
}

fn download_reference_preview(
    object: &RemoteObject,
    max_edge: usize,
    mut on_progress: impl FnMut(u64, u64),
    mut should_continue: impl FnMut() -> bool,
) -> Result<Option<PreviewImage>, String> {
    match download_reference_preview_sparse(
        object,
        max_edge,
        &mut on_progress,
        &mut should_continue,
    ) {
        Ok(preview) => Ok(preview),
        Err(sparse_error) if should_continue() => {
            match download_reference_preview_continuous(
                object,
                max_edge,
                on_progress,
                should_continue,
            ) {
                Ok(preview) => Ok(preview),
                Err(stream_error) => Err(format!(
                    "Selective preview failed ({sparse_error}); continuous fallback failed ({stream_error})"
                )),
            }
        }
        Err(_) => Ok(None),
    }
}

fn download_reference_preview_sparse(
    object: &RemoteObject,
    max_edge: usize,
    on_progress: &mut impl FnMut(u64, u64),
    should_continue: &mut impl FnMut() -> bool,
) -> Result<Option<PreviewImage>, String> {
    block_on(async {
        const LELR_HEADER_BYTES: u32 = 32;
        on_progress(0, object.size);
        let file_size = usize::try_from(object.size)
            .map_err(|_| format!("{} is too large for this platform", object.name))?;
        let mut sparse = Vec::new();
        let mut block_offset = 0usize;
        let mut transferred = 0u64;
        let mut payload_loaded = false;
        let mut preferences_loaded = false;
        while block_offset < file_size {
            if !should_continue() {
                return Ok(None);
            }
            let header = object
                .storage
                .read_range(object.handle, block_offset as u64, LELR_HEADER_BYTES)
                .await
                .map_err(|error| format!("Could not read {} LELR header: {error}", object.name))?;
            if header.len() != LELR_HEADER_BYTES as usize {
                return Err(format!("Short LELR header read from {}", object.name));
            }
            transferred += header.len() as u64;
            let descriptor = inspect_lelr_block_header(&header, block_offset)
                .map_err(|error| error.to_string())?;
            let block_range = descriptor.block_range();
            if block_range.end > file_size {
                return Err(format!("{} has a block past its object size", object.name));
            }
            sparse.resize(block_range.end, 0);
            sparse[block_offset..block_offset + header.len()].copy_from_slice(&header);

            let message_range = descriptor.message_range();
            if !message_range.is_empty() {
                let message_length = u32::try_from(message_range.len())
                    .map_err(|_| "LELR metadata message is too large".to_owned())?;
                let message = object
                    .storage
                    .read_range(object.handle, message_range.start as u64, message_length)
                    .await
                    .map_err(|error| {
                        format!("Could not read {} LELR metadata: {error}", object.name)
                    })?;
                if message.len() != message_range.len() {
                    return Err(format!("Short LELR metadata read from {}", object.name));
                }
                transferred += message.len() as u64;
                sparse[message_range].copy_from_slice(&message);
            }
            on_progress(transferred, object.size);

            if !payload_loaded
                && let Some(mut payload_read) =
                    reference_camera_payload_read(&sparse).map_err(|error| error.to_string())?
            {
                if let ReferencePayloadRead::BayerJpegHeader(header_range) = payload_read {
                    let header_length = u32::try_from(header_range.len())
                        .map_err(|_| "Bayer-JPEG header is too large".to_owned())?;
                    let bayer_header = object
                        .storage
                        .read_range(object.handle, header_range.start as u64, header_length)
                        .await
                        .map_err(|error| {
                            format!("Could not read {} Bayer-JPEG header: {error}", object.name)
                        })?;
                    if bayer_header.len() != header_range.len() {
                        return Err(format!("Short Bayer-JPEG header read from {}", object.name));
                    }
                    transferred += bayer_header.len() as u64;
                    sparse[header_range].copy_from_slice(&bayer_header);
                    payload_read = reference_camera_payload_read(&sparse)
                        .map_err(|error| error.to_string())?
                        .ok_or_else(|| {
                            "Reference camera disappeared after reading its header".to_owned()
                        })?;
                }
                let ReferencePayloadRead::Payload(payload_range) = payload_read else {
                    return Err(format!("Invalid Bayer-JPEG header in {}", object.name));
                };
                let payload_length = u32::try_from(payload_range.len())
                    .map_err(|_| "Reference camera payload is too large".to_owned())?;
                let payload = object
                    .storage
                    .read_range(object.handle, payload_range.start as u64, payload_length)
                    .await
                    .map_err(|error| {
                        format!("Could not read {} reference camera: {error}", object.name)
                    })?;
                if payload.len() != payload_range.len() {
                    return Err(format!("Short reference camera read from {}", object.name));
                }
                transferred += payload.len() as u64;
                sparse[payload_range].copy_from_slice(&payload);
                on_progress(transferred, object.size);
                payload_loaded = true;
            }
            if descriptor.message_type == 1 {
                preferences_loaded = true;
            }
            if payload_loaded
                && preferences_loaded
                && reference_preview_color_ready(&sparse)
                    .map_err(|error| error.to_string())?
                    .unwrap_or(false)
            {
                return decode_reference_preview_bytes(&sparse, max_edge)
                    .map(Some)
                    .map_err(|error| error.to_string());
            }
            block_offset = block_range.end;
        }
        if payload_loaded {
            decode_reference_preview_bytes(&sparse, max_edge)
                .map(Some)
                .map_err(|error| error.to_string())
        } else {
            Err(format!(
                "No reference camera payload found in {}",
                object.name
            ))
        }
    })
}

fn download_capture_metadata_sparse(
    object: &RemoteObject,
    mut should_continue: impl FnMut() -> bool,
) -> Result<CaptureMetadata, String> {
    block_on(async {
        const LELR_HEADER_BYTES: u32 = 32;
        let file_size = usize::try_from(object.size)
            .map_err(|_| format!("{} is too large for this platform", object.name))?;
        let mut messages = Vec::<(u8, usize, usize, Vec<u8>)>::new();
        let mut block_offset = 0usize;
        while block_offset < file_size && should_continue() {
            let header = object
                .storage
                .read_range(object.handle, block_offset as u64, LELR_HEADER_BYTES)
                .await
                .map_err(|error| {
                    format!("Could not read {} metadata header: {error}", object.name)
                })?;
            if header.len() != LELR_HEADER_BYTES as usize {
                return Err(format!("Short metadata header read from {}", object.name));
            }
            let descriptor = inspect_lelr_block_header(&header, block_offset)
                .map_err(|error| error.to_string())?;
            let block_range = descriptor.block_range();
            if block_range.end > file_size {
                return Err(format!("{} has a block past its object size", object.name));
            }
            let message_range = descriptor.message_range();
            let message = if message_range.is_empty() {
                Vec::new()
            } else {
                let length = u32::try_from(message_range.len())
                    .map_err(|_| "LRI metadata message is too large".to_owned())?;
                let message = object
                    .storage
                    .read_range(object.handle, message_range.start as u64, length)
                    .await
                    .map_err(|error| format!("Could not read {} metadata: {error}", object.name))?;
                if message.len() != message_range.len() {
                    return Err(format!("Short metadata read from {}", object.name));
                }
                message
            };
            messages.push((
                descriptor.message_type,
                block_offset,
                block_range.end,
                message,
            ));
            block_offset = block_range.end;
            if descriptor.message_type == 1 {
                let blocks = messages
                    .iter()
                    .map(|(kind, start, end, message)| (*kind, *start, *end, message.as_slice()))
                    .collect::<Vec<_>>();
                return inspect_capture_metadata_blocks(&blocks).map_err(|error| error.to_string());
            }
        }
        if !should_continue() {
            return Err(CANCELLED_LOAD.to_owned());
        }
        let blocks = messages
            .iter()
            .map(|(kind, start, end, message)| (*kind, *start, *end, message.as_slice()))
            .collect::<Vec<_>>();
        inspect_capture_metadata_blocks(&blocks).map_err(|error| error.to_string())
    })
}

pub fn apply_preview_orientation(preview: &mut PreviewImage, orientation: u64) {
    if orientation == 0 || preview.size[0] == 0 || preview.size[1] == 0 {
        return;
    }
    let [width, height] = preview.size;
    let swapped = matches!(orientation, 1..=4);
    let [out_width, out_height] = if swapped {
        [height, width]
    } else {
        [width, height]
    };
    let mut oriented = vec![0; preview.rgb.len()];
    for y in 0..out_height {
        for x in 0..out_width {
            let (source_x, source_y) = match orientation {
                1 => (y, height - 1 - x),
                2 => (width - 1 - y, x),
                3 => (width - 1 - y, height - 1 - x),
                4 => (y, x),
                5 => (x, height - 1 - y),
                6 => (width - 1 - x, y),
                7 => (width - 1 - x, height - 1 - y),
                _ => (x, y),
            };
            let source = (source_y * width + source_x) * 3;
            let target = (y * out_width + x) * 3;
            oriented[target..target + 3].copy_from_slice(&preview.rgb[source..source + 3]);
        }
    }
    preview.size = [out_width, out_height];
    preview.rgb = oriented;
}

fn download_reference_preview_continuous(
    object: &RemoteObject,
    max_edge: usize,
    mut on_progress: impl FnMut(u64, u64),
    mut should_continue: impl FnMut() -> bool,
) -> Result<Option<PreviewImage>, String> {
    block_on(async {
        on_progress(0, object.size);
        let mut download = object
            .storage
            .download(object.handle, ByteRange::Full)
            .await
            .map_err(|error| format!("Could not start download of {}: {error}", object.name))?;
        let mut bytes = Vec::new();
        let mut inspected_prefix = 0usize;
        let mut preview_ready_at = None;
        let mut blocks_after_preview = 0usize;
        while should_continue() {
            let Some(chunk) = download.next_chunk().await else {
                return decode_reference_preview_bytes(&bytes, max_edge)
                    .map(Some)
                    .map_err(|error| error.to_string());
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    let _ = download.cancel(DEFAULT_CANCEL_TIMEOUT).await;
                    return Err(format!("Could not download {}: {error}", object.name));
                }
            };
            bytes.extend_from_slice(&chunk);
            on_progress(bytes.len() as u64, object.size);
            let complete_prefix = match complete_lelr_prefix_len(&bytes) {
                Ok(complete_prefix) => complete_prefix,
                Err(error) => {
                    let _ = download.cancel(DEFAULT_CANCEL_TIMEOUT).await;
                    return Err(error.to_string());
                }
            };
            if complete_prefix == inspected_prefix {
                continue;
            }
            inspected_prefix = complete_prefix;
            match try_decode_reference_preview_prefix(&bytes[..complete_prefix], max_edge) {
                Ok(Some(preview)) => {
                    let has_preferences = prefix_has_message_type(&bytes[..complete_prefix], 1)?;
                    let color_ready = reference_preview_color_ready(&bytes[..complete_prefix])
                        .map_err(|error| error.to_string())?
                        .unwrap_or(false);
                    if color_ready && (has_preferences || blocks_after_preview >= 2) {
                        let _ = download.cancel(DEFAULT_CANCEL_TIMEOUT).await;
                        return Ok(Some(preview));
                    }
                    if preview_ready_at.is_some() {
                        blocks_after_preview += 1;
                    } else {
                        preview_ready_at = Some(complete_prefix);
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = download.cancel(DEFAULT_CANCEL_TIMEOUT).await;
                    return Err(error.to_string());
                }
            }
        }
        let _ = download.cancel(DEFAULT_CANCEL_TIMEOUT).await;
        Ok(None)
    })
}

fn prefix_has_message_type(data: &[u8], expected: u8) -> Result<bool, String> {
    let mut offset = 0usize;
    while offset < data.len() {
        let descriptor = inspect_lelr_block_header(&data[offset..], offset)
            .map_err(|error| error.to_string())?;
        if descriptor.message_type == expected {
            return Ok(true);
        }
        offset = descriptor.block_range().end;
    }
    Ok(false)
}
