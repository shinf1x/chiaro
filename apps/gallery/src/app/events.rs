use super::*;

impl GalleryApp {
    pub(super) fn receive_events(&mut self, ctx: &egui::Context) {
        for event in self.loader.drain(self.generation) {
            match event {
                LoadedEvent::Preview { key, result } => self.receive_preview(ctx, key, result),
                LoadedEvent::PreviewProgress {
                    key: PreviewKey::Gallery { id, revision },
                    transferred,
                    total,
                } => {
                    if let Some(item) = self.gallery_item_mut(id, revision) {
                        item.state = ItemState::Pending { transferred, total };
                    }
                }
                LoadedEvent::PreviewProgress { .. } => {}
                LoadedEvent::PreviewSkipped {
                    key: PreviewKey::Gallery { id, revision },
                } => {
                    self.loader
                        .retry_gallery_preview(self.generation, id, revision);
                    if let Some(item) = self.gallery_item_mut(id, revision)
                        && matches!(item.state, ItemState::Pending { .. })
                    {
                        item.state = ItemState::Pending {
                            transferred: 0,
                            total: 0,
                        };
                    }
                }
                LoadedEvent::PreviewSkipped { .. } => {}
                LoadedEvent::Capture { modal, result } => {
                    let Some(sheet) = self
                        .contact_sheet
                        .as_mut()
                        .filter(|sheet| sheet.id == modal)
                    else {
                        continue;
                    };
                    match result {
                        Ok(capture) => {
                            let mut progressive = match &mut sheet.state {
                                ContactState::Loading { cameras, .. } => std::mem::take(cameras),
                                _ => Vec::new(),
                            };
                            let cameras = capture
                                .summary
                                .cameras
                                .into_iter()
                                .map(|camera| {
                                    if let Some(position) =
                                        progressive.iter().position(|existing| {
                                            existing.camera == camera.camera
                                                && existing.frame_index == camera.frame_index
                                        })
                                    {
                                        progressive.remove(position)
                                    } else {
                                        CameraPreview {
                                            camera: camera.camera,
                                            frame_index: camera.frame_index,
                                            state: ModalImageState::Pending,
                                        }
                                    }
                                })
                                .collect::<Vec<_>>();
                            for camera in &cameras {
                                if !matches!(camera.state, ModalImageState::Pending) {
                                    continue;
                                }
                                self.loader.load_camera(
                                    self.generation,
                                    PreviewKey::Contact {
                                        modal,
                                        camera: camera.camera.clone(),
                                        frame_index: camera.frame_index,
                                    },
                                    capture.data.clone(),
                                    camera.camera.clone(),
                                    camera.frame_index,
                                    520,
                                );
                            }
                            sheet.state = ContactState::Ready {
                                reference_camera: capture.summary.reference_camera,
                                cameras,
                                data: capture.data,
                                metadata: capture.summary.metadata,
                            };
                            if matches!(&sheet.state, ContactState::Ready { cameras, .. } if cameras
                                .iter()
                                .all(|camera| !matches!(camera.state, ModalImageState::Pending)))
                            {
                                self.loader.set_modal_loading(false);
                            }
                        }
                        Err(error) => {
                            sheet.state = ContactState::Failed(error);
                            self.loader.set_modal_loading(false);
                        }
                    }
                }
                LoadedEvent::CaptureProgress {
                    modal,
                    transferred,
                    total,
                } => {
                    let Some(ContactState::Loading {
                        transferred: current,
                        total: expected,
                        ..
                    }) = self
                        .contact_sheet
                        .as_mut()
                        .filter(|sheet| sheet.id == modal)
                        .map(|sheet| &mut sheet.state)
                    else {
                        continue;
                    };
                    *current = transferred;
                    *expected = total;
                }
                LoadedEvent::CapturePreview {
                    modal,
                    reference_camera,
                    frame_index,
                    preview,
                } => {
                    let Some(ContactState::Loading {
                        reference_camera: current_reference,
                        cameras,
                        ..
                    }) = self
                        .contact_sheet
                        .as_mut()
                        .filter(|sheet| sheet.id == modal)
                        .map(|sheet| &mut sheet.state)
                    else {
                        continue;
                    };
                    *current_reference = Some(reference_camera);
                    let camera = preview.camera.clone();
                    let state = modal_state(
                        ctx,
                        format!("contact-stream-{modal}-{camera}-{frame_index}"),
                        Ok(preview),
                    );
                    if let Some(existing) = cameras.iter_mut().find(|existing| {
                        existing.camera == camera && existing.frame_index == frame_index
                    }) {
                        // A later calibration block can upgrade a progressively
                        // decoded RAW card without waiting for the full capture.
                        existing.state = state;
                        continue;
                    }
                    cameras.push(CameraPreview {
                        camera: camera.clone(),
                        frame_index,
                        state,
                    });
                    cameras.sort_by(|left, right| {
                        left.camera
                            .cmp(&right.camera)
                            .then(left.frame_index.cmp(&right.frame_index))
                    });
                }
                LoadedEvent::DeviceDone(result) => {
                    let Some(key) = self.indexing_tab.clone() else {
                        continue;
                    };
                    let connected = match key {
                        TabKey::Device { location_id, mode } => self
                            .devices
                            .iter()
                            .any(|device| device.location_id == location_id && device.mode == mode),
                        _ => false,
                    };
                    match result {
                        Ok(()) => {
                            self.device_retry_at = None;
                            if let Some(view) = self.tab_view_mut(&key) {
                                view.busy = None;
                                view.listing_progress = None;
                                view.status = view
                                    .items
                                    .is_empty()
                                    .then(|| "No .lri files found on this camera".to_owned());
                            }
                        }
                        Err(error) => {
                            if let Some(view) = self.tab_view_mut(&key) {
                                view.busy = None;
                                view.status = Some(if connected {
                                    format!(
                                        "{error}. The camera is still connected; retrying automatically..."
                                    )
                                } else {
                                    error
                                });
                            }
                            if connected && self.active_tab == key {
                                self.device_retry_at =
                                    Some(Instant::now() + Duration::from_millis(1200));
                            }
                        }
                    }
                    self.indexing_tab = None;
                }
                LoadedEvent::DeviceItem(source) => {
                    let Some(key) = self.indexing_tab.clone() else {
                        continue;
                    };
                    let id = self.next_item_id;
                    self.next_item_id += 1;
                    let mut added = self.loader.load_items(vec![source], id);
                    if let Some(item) = added.first_mut() {
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
                    if let Some(view) = self.tab_view_mut(&key) {
                        view.items.extend(added);
                        view.items.sort_by(|left, right| {
                            right
                                .source
                                .name
                                .to_lowercase()
                                .cmp(&left.source.name.to_lowercase())
                        });
                        view.status = None;
                    }
                }
                LoadedEvent::DevicePreview {
                    capture_name,
                    preview,
                } => {
                    let Some(key) = self.indexing_tab.clone() else {
                        continue;
                    };
                    let queued = self.tab_view_mut(&key).and_then(|view| {
                        let item = view
                            .items
                            .iter_mut()
                            .find(|item| item.source.name == capture_name)?;
                        item.source.preview = preview.clone();
                        item.preview_revision = item.preview_revision.wrapping_add(1);
                        item.state = ItemState::Pending {
                            transferred: 0,
                            total: 0,
                        };
                        Some((item.id, item.preview_revision, item.source.capture.clone()))
                    });
                    let Some((id, revision, capture)) = queued else {
                        continue;
                    };
                    self.loader.load_gallery_preview(
                        self.generation,
                        id,
                        revision,
                        preview,
                        capture,
                    );
                }
                LoadedEvent::DeviceCalibration(object) => {
                    if let Some(key) = self.indexing_tab.clone()
                        && let Some(view) = self.tab_view_mut(&key)
                        && !view
                            .device_calibration
                            .iter()
                            .any(|existing| existing.name.eq_ignore_ascii_case(&object.name))
                    {
                        view.device_calibration.push(object);
                    }
                }
                LoadedEvent::DeviceProgress { fetched, total } => {
                    if let Some(key) = self.indexing_tab.clone()
                        && let Some(view) = self.tab_view_mut(&key)
                    {
                        view.listing_progress = Some((fetched, total));
                        view.busy = Some(if total == 0 {
                            "Connected to Light L16; locating DCIM/Camera...".to_owned()
                        } else {
                            "Reading Light L16 camera index...".to_owned()
                        });
                    }
                }
                LoadedEvent::DeviceStatus(message) => {
                    if let Some(key) = self.indexing_tab.clone()
                        && let Some(view) = self.tab_view_mut(&key)
                    {
                        view.busy = Some(message);
                    }
                }
            }
        }
    }

    pub(super) fn receive_preview(
        &mut self,
        ctx: &egui::Context,
        key: PreviewKey,
        result: Result<PreviewImage, String>,
    ) {
        match key {
            PreviewKey::Gallery { id, revision } => {
                let generation = self.generation;
                let Some(item) = self.gallery_item_mut(id, revision) else {
                    return;
                };
                item.state = match result {
                    Ok(preview) => {
                        let camera = preview.camera.clone();
                        let metadata = preview.metadata.clone();
                        let (texture, dimensions) =
                            upload_preview(ctx, format!("lri-preview-{generation}-{id}"), preview);
                        ItemState::Ready {
                            texture,
                            camera,
                            dimensions,
                            metadata,
                        }
                    }
                    Err(error) => ItemState::Failed(error),
                };
            }
            PreviewKey::Contact {
                modal,
                camera,
                frame_index,
            } => {
                let all_finished = {
                    let Some(sheet) = self
                        .contact_sheet
                        .as_mut()
                        .filter(|sheet| sheet.id == modal)
                    else {
                        return;
                    };
                    let ContactState::Ready { cameras, .. } = &mut sheet.state else {
                        return;
                    };
                    let Some(card) = cameras
                        .iter_mut()
                        .find(|card| card.camera == camera && card.frame_index == frame_index)
                    else {
                        return;
                    };
                    card.state = modal_state(
                        ctx,
                        format!("contact-{modal}-{camera}-{frame_index}"),
                        result,
                    );
                    cameras
                        .iter()
                        .all(|camera| !matches!(camera.state, ModalImageState::Pending))
                };
                if all_finished {
                    self.loader.set_modal_loading(false);
                }
            }
            PreviewKey::Full {
                modal,
                camera,
                frame_index,
            } => {
                let Some(full) = self
                    .contact_sheet
                    .as_mut()
                    .filter(|sheet| sheet.id == modal)
                    .and_then(|sheet| sheet.full.as_mut())
                    .filter(|full| full.camera == camera && full.frame_index == frame_index)
                else {
                    return;
                };
                full.state =
                    modal_state(ctx, format!("full-{modal}-{camera}-{frame_index}"), result);
                self.loader.set_modal_loading(false);
            }
        }
    }
}
