use super::*;

pub fn discover_light_devices() -> Result<Vec<LightDevice>, String> {
    let candidates = MtpDevice::list_devices_with_known(&LIGHT_USB_IDS)
        .map_err(|error| format!("USB camera discovery failed: {error}"))?;
    let mut devices = candidates
        .into_iter()
        .filter(|device| device.vendor_id == LIGHT_VENDOR_ID)
        .filter_map(|device| {
            let mode = match device.product_id {
                LIGHT_MTP_PRODUCT_ID => DeviceMode::Mtp,
                LIGHT_PTP_PRODUCT_ID => DeviceMode::Ptp,
                _ => return None,
            };
            Some(LightDevice {
                location_id: device.location_id,
                serial_number: device.serial_number,
                mode,
            })
        })
        .collect::<Vec<_>>();
    devices.sort_by_key(|device| (device.mode.label(), device.location_id));
    Ok(devices)
}

pub fn local_items(folder: &Path) -> std::io::Result<Vec<SourceItem>> {
    lri_paths(folder).map(|paths| {
        paths
            .into_iter()
            .map(|path| SourceItem {
                name: file_name(&path),
                location_label: path.display().to_string(),
                preview: PreviewLocator::Lri(CaptureLocator::Local(path.clone())),
                capture: CaptureLocator::Local(path),
            })
            .collect()
    })
}

pub fn device_items(
    device: &LightDevice,
    mut on_progress: impl FnMut(usize, usize),
    mut on_item: impl FnMut(DeviceItemEvent),
    mut on_status: impl FnMut(String),
    mut should_continue: impl FnMut() -> bool,
    mut should_pause: impl FnMut() -> bool,
) -> Result<(), String> {
    block_on(device_items_async(
        device,
        &mut on_progress,
        &mut on_item,
        &mut on_status,
        &mut should_continue,
        &mut should_pause,
    ))
}

async fn device_items_async<F, I, S, C, P>(
    descriptor: &LightDevice,
    on_progress: &mut F,
    on_item: &mut I,
    on_status: &mut S,
    should_continue: &mut C,
    should_pause: &mut P,
) -> Result<(), String>
where
    F: FnMut(usize, usize),
    I: FnMut(DeviceItemEvent),
    S: FnMut(String),
    C: FnMut() -> bool,
    P: FnMut() -> bool,
{
    if !should_continue() {
        return Ok(());
    }
    let device = open_light_device(descriptor, on_status).await?;
    if !should_continue() {
        return Ok(());
    }
    let identity = device.device_info();
    if !identity.manufacturer.eq_ignore_ascii_case("Light")
        || !identity.model.eq_ignore_ascii_case("L16")
    {
        return Err(format!(
            "USB device identified itself as {} {}, not Light L16",
            identity.manufacturer, identity.model
        ));
    }
    on_progress(0, 0);

    for storage in device
        .storages()
        .await
        .map_err(|error| format!("Could not list L16 storage: {error}"))?
    {
        if !should_continue() {
            return Ok(());
        }
        let root = storage
            .list_objects(None)
            .await
            .map_err(|error| format!("Could not list the L16 storage root: {error}"))?;
        let Some(dcim_folder) = named_folder(&root, "DCIM") else {
            continue;
        };
        let dcim = storage
            .list_objects(Some(dcim_folder))
            .await
            .map_err(|error| format!("Could not list the L16 DCIM folder: {error}"))?;
        let Some(camera_folder) = named_folder(&dcim, "Camera") else {
            continue;
        };
        let storage = Arc::new(storage);
        let mut listing = storage
            .list_objects_stream(Some(camera_folder))
            .await
            .map_err(|error| format!("Could not list L16 DCIM/Camera: {error}"))?;
        let total = listing.total();
        on_progress(0, total);
        let mut camera_objects = Vec::with_capacity(total);
        let mut fetched = 0;
        while should_continue() {
            while should_pause() && should_continue() {
                thread::sleep(Duration::from_millis(15));
            }
            if !should_continue() {
                return Ok(());
            }
            let Some(object) = listing.next().await else {
                break;
            };
            let object = object.map_err(|error| format!("Could not list L16 files: {error}"))?;
            fetched += 1;
            camera_objects.push(object);
            if fetched % 16 == 0 || fetched == total {
                on_progress(fetched, total);
            }
        }
        // ObjectListing must be consumed or dropped before another storage
        // request. Discover the fixed Camera/lightcal directory from that one
        // pass, then fetch its three known files before thumbnail work begins.
        drop(listing);
        let calibration_folder = named_folder(&camera_objects, DEVICE_CALIBRATION_FOLDER);
        match find_calibration_files(&storage, calibration_folder, &DEVICE_CALIBRATION_FILES).await
        {
            Ok(calibrations) => {
                for calibration in calibrations {
                    on_item(DeviceItemEvent::Calibration(remote_object(
                        Arc::clone(&storage),
                        descriptor,
                        &calibration,
                    )));
                }
            }
            Err(error) => on_status(format!(
                "Could not read the L16 DCIM/Camera/lightcal folder: {error}"
            )),
        }

        let mut jpegs = HashMap::<String, ObjectInfo>::new();
        let mut captures = HashMap::<String, String>::new();
        for object in camera_objects {
            let path = Path::new(&object.filename);
            if has_extension(path, "jpg") || has_extension(path, "jpeg") {
                if let Some(stem) = file_stem_lower(path) {
                    if let Some(capture_name) = captures.get(&stem) {
                        on_item(DeviceItemEvent::Preview {
                            capture_name: capture_name.clone(),
                            preview: PreviewLocator::Jpeg(ObjectLocator::Device(remote_object(
                                Arc::clone(&storage),
                                descriptor,
                                &object,
                            ))),
                        });
                    }
                    jpegs.insert(stem, object);
                }
            } else if has_extension(path, "lri") {
                let stem = file_stem_lower(path);
                if let Some(stem) = &stem {
                    captures.insert(stem.clone(), object.filename.clone());
                }
                let jpeg = stem.as_ref().and_then(|stem| jpegs.get(stem));
                on_item(DeviceItemEvent::Add(device_source_item(
                    Arc::clone(&storage),
                    descriptor,
                    &object,
                    jpeg,
                )));
            }
        }
        return Ok(());
    }
    Err("The L16 has no DCIM/Camera folder".to_owned())
}

async fn open_light_device(
    descriptor: &LightDevice,
    on_status: &mut impl FnMut(String),
) -> Result<MtpDevice, String> {
    let mut last_error = None;
    for attempt in 0..2 {
        on_status(if attempt == 0 {
            format!("Opening Light L16 over {}...", descriptor.mode.label())
        } else {
            format!("Retrying Light L16 over {}...", descriptor.mode.label())
        });
        match MtpDevice::builder()
            .known_devices(&LIGHT_USB_IDS)
            .timeout(Duration::from_secs(5))
            .open_by_location(descriptor.location_id)
            .await
        {
            Ok(device) => return Ok(device),
            Err(error) => {
                let message = error.to_string();
                if should_reset_transport(&error) {
                    on_status(format!(
                        "Light L16 did not respond ({message}); resetting USB transport..."
                    ));
                    reset_light_transport(descriptor.location_id).await;
                }
                last_error = Some(message);
                if attempt == 1 {
                    break;
                }
                thread::sleep(Duration::from_millis(350));
            }
        }
    }
    Err(format!(
        "Could not open L16 over {}: {}",
        descriptor.mode.label(),
        last_error.unwrap_or_else(|| "unknown camera error".to_owned())
    ))
}

fn should_reset_transport(error: &mtp_rs::mtp::Error) -> bool {
    matches!(
        error,
        mtp_rs::mtp::Error::InvalidData { .. }
            | mtp_rs::mtp::Error::Disconnected
            | mtp_rs::mtp::Error::DeviceReset
            | mtp_rs::mtp::Error::Timeout
    )
}

async fn reset_light_transport(location_id: u64) {
    if let Ok(device) =
        PtpDevice::open_by_location_with_timeout(location_id, Duration::from_secs(2)).await
        && device.reset_device().await.is_ok()
    {
        thread::sleep(Duration::from_secs(3));
        return;
    }

    // If PTP cannot be opened at all, force a USB re-enumeration. This is the
    // software equivalent of the physical reconnect that clears the L16 state.
    if let Ok(candidates) = NusbTransport::list_mtp_devices_with_known(&LIGHT_USB_IDS)
        && let Some(candidate) = candidates
            .into_iter()
            .find(|candidate| candidate.location_id == location_id)
        && let Ok(device) = candidate.open()
    {
        // nusb exposes reset as a MaybeFuture backed by a blocking ioctl. It
        // must be waited synchronously because this app intentionally uses no
        // Tokio/smol runtime; awaiting it would panic on real hardware.
        let _ = device.reset().wait();
    }
    thread::sleep(Duration::from_secs(3));
}

/// Read the known factory calibration files from `DCIM/Camera/lightcal`.
///
/// The caller resolves the fixed folder handle; this function does not recurse.
async fn find_calibration_files(
    storage: &Storage,
    folder: Option<ObjectHandle>,
    names: &[&str],
) -> Result<Vec<ObjectInfo>, mtp_rs::Error> {
    let Some(folder) = folder else {
        return Ok(Vec::new());
    };
    Ok(storage
        .list_objects(Some(folder))
        .await?
        .into_iter()
        .filter(|object| {
            !object.is_folder()
                && names
                    .iter()
                    .any(|name| object.filename.eq_ignore_ascii_case(name))
        })
        .collect())
}

fn named_folder(objects: &[ObjectInfo], name: &str) -> Option<ObjectHandle> {
    objects
        .iter()
        .find(|object| object.is_folder() && object.filename.eq_ignore_ascii_case(name))
        .map(|object| object.handle)
}

fn device_source_item(
    storage: Arc<Storage>,
    device: &LightDevice,
    file: &ObjectInfo,
    jpeg: Option<&ObjectInfo>,
) -> SourceItem {
    let capture = remote_object(Arc::clone(&storage), device, file);
    let preview = jpeg
        .map(|jpeg| {
            PreviewLocator::Jpeg(ObjectLocator::Device(remote_object(
                Arc::clone(&storage),
                device,
                jpeg,
            )))
        })
        .unwrap_or_else(|| PreviewLocator::Lri(CaptureLocator::Device(capture.clone())));
    SourceItem {
        name: file.filename.clone(),
        location_label: format!(
            "L16 {} - object {} - {} MB",
            device.mode.label(),
            file.handle.0,
            file.size / 1_000_000
        ),
        preview,
        capture: CaptureLocator::Device(capture),
    }
}

fn remote_object(storage: Arc<Storage>, device: &LightDevice, file: &ObjectInfo) -> RemoteObject {
    RemoteObject {
        storage,
        handle: file.handle,
        name: file.filename.clone(),
        size: file.size,
        device: device.clone(),
    }
}
