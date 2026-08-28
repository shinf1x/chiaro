use super::*;

impl GalleryApp {
    pub(super) fn open_contact_sheet(&mut self, index: usize) {
        let Some(item) = self.current_view.items.get(index) else {
            return;
        };
        let id = self.next_modal_id;
        self.next_modal_id += 1;
        self.contact_sheet = Some(ContactSheet {
            id,
            name: item.source.name.clone(),
            state: ContactState::Loading {
                transferred: 0,
                total: 0,
                reference_camera: None,
                cameras: Vec::new(),
            },
            full: None,
        });
        self.loader
            .open_capture(self.generation, id, item.source.capture.clone());
    }

    pub(super) fn show_contact_sheet(&mut self, ctx: &egui::Context, status_bar_top: f32) {
        let stacked_capture_preview = self.settings.stacked_capture_preview;
        let Some(sheet) = self.contact_sheet.as_mut() else {
            return;
        };
        let mut open_camera = None;
        let viewport_rect = ctx.content_rect();
        let mut usable_rect = viewport_rect;
        usable_rect.max.y = usable_rect.max.y.min(status_bar_top);
        let max_size = usable_rect.size() * 0.92;
        let modal_size = Vec2::new(max_size.x.min(1180.0), max_size.y.min(840.0));
        let backdrop_rect = usable_rect;
        egui::Area::new(egui::Id::new(("contact-backdrop", sheet.id)))
            .order(egui::Order::Foreground)
            .fixed_pos(backdrop_rect.min)
            .interactable(false)
            .show(ctx, |ui| {
                ui.set_min_size(backdrop_rect.size());
                ui.painter().rect_filled(
                    backdrop_rect,
                    CornerRadius::ZERO,
                    Color32::from_black_alpha(190),
                );
            });
        let modal_id = egui::Id::new(("contact-sheet", sheet.id));
        let center_offset = usable_rect.center() - viewport_rect.center();
        let response = egui::Modal::new(modal_id)
            .area(
                egui::Modal::default_area(modal_id)
                    .anchor(egui::Align2::CENTER_CENTER, center_offset),
            )
            .backdrop_color(Color32::TRANSPARENT)
            .frame(
                egui::Frame::popup(&ctx.style())
                    .fill(Color32::from_rgb(24, 27, 32))
                    .corner_radius(CornerRadius::same(10))
                    .inner_margin(Margin::same(16)),
            )
            .show(ctx, |ui| {
                // Re-apply both bounds every frame so leaving fullscreen also shrinks the modal.
                ui.set_min_size(modal_size);
                ui.set_max_size(modal_size);
                if let Some(full) = &mut sheet.full {
                    full_view(ui, ctx, &sheet.name, full);
                    return;
                }

                ui.horizontal(|ui| {
                    ui.heading(&sheet.name);
                    if let ContactState::Ready { metadata, .. } = &sheet.state {
                        show_capture_icons(ui, metadata, Color32::from_rgb(24, 27, 32));
                    }
                    let reference_camera = match &sheet.state {
                        ContactState::Loading {
                            reference_camera, ..
                        } => reference_camera.as_ref(),
                        ContactState::Ready {
                            reference_camera, ..
                        } => Some(reference_camera),
                        ContactState::Failed(_) => None,
                    };
                    if let Some(reference_camera) = reference_camera {
                        ui.label(
                            RichText::new(format!("Reference: {reference_camera}"))
                                .color(Color32::from_gray(155)),
                        );
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Close").clicked() {
                            ui.close();
                        }
                    });
                });
                ui.separator();

                match &mut sheet.state {
                    ContactState::Loading {
                        reference_camera,
                        cameras,
                        ..
                    } => {
                        ui.label(
                            RichText::new(
                                "Camera previews appear here as the LRI streams into memory",
                            )
                            .color(Color32::from_gray(145)),
                        );
                        if !cameras.is_empty() {
                            ui.separator();
                            ui.label(
                                RichText::new(format!(
                                    "{} camera previews available; full views unlock when loading completes",
                                    cameras.len()
                                ))
                                .color(Color32::from_gray(145)),
                            );
                            if stacked_capture_preview {
                                camera_stack_view(
                                    ui,
                                    sheet.id,
                                    cameras,
                                    reference_camera.as_deref(),
                                    None,
                                    None,
                                    None,
                                );
                            } else {
                                camera_contact_grid(ui, sheet.id, cameras, None);
                            }
                        }
                    }
                    ContactState::Failed(error) => {
                        ui.centered_and_justified(|ui| {
                            ui.colored_label(Color32::from_rgb(225, 125, 125), error);
                        });
                    }
                    ContactState::Ready {
                        reference_camera,
                        cameras,
                        metadata,
                        overlay_geometry,
                        ..
                    } => {
                        ui.label(
                            RichText::new(metadata_summary(metadata))
                                .color(Color32::from_gray(155)),
                        );
                        ui.label(
                            RichText::new("Select a camera to load its full native preview")
                                .color(Color32::from_gray(145)),
                        );
                        if stacked_capture_preview {
                            camera_stack_view(
                                ui,
                                sheet.id,
                                cameras,
                                Some(reference_camera),
                                overlay_geometry.as_ref(),
                                metadata.focal_length_mm,
                                Some(&mut open_camera),
                            );
                        } else {
                            camera_contact_grid(ui, sheet.id, cameras, Some(&mut open_camera));
                        }
                    }
                }
            });

        if response.should_close() {
            if sheet.full.is_some() {
                sheet.full = None;
            } else {
                self.loader.cancel_modal(sheet.id);
                self.contact_sheet = None;
                return;
            }
        }

        if let Some((camera, frame_index, show_frame)) = open_camera {
            let ContactState::Ready { data, .. } = &sheet.state else {
                return;
            };
            let id = sheet.id;
            sheet.full = Some(FullPreview {
                camera: camera.clone(),
                frame_index,
                show_frame,
                state: ModalImageState::Pending,
                zoom: 1.0,
                pan: Vec2::ZERO,
                auto_fit: true,
            });
            self.loader.load_camera(
                self.generation,
                PreviewKey::Full {
                    modal: id,
                    camera: camera.clone(),
                    frame_index,
                },
                data.clone(),
                camera,
                frame_index,
                usize::MAX,
            );
        }
    }
}

fn full_view(ui: &mut egui::Ui, ctx: &egui::Context, name: &str, full: &mut FullPreview) {
    ui.horizontal(|ui| {
        ui.heading(format!(
            "{} - camera {}",
            name,
            camera_frame_label(&full.camera, full.frame_index, full.show_frame)
        ));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.button("Back to contact sheet").clicked() {
                ui.close();
            }
            ui.separator();
            let zoom_response = ui.add(
                egui::Slider::new(&mut full.zoom, 0.01..=4.0)
                    .custom_formatter(|value, _| format!("{}%", (value * 100.0).round())),
            );
            if zoom_response.changed() {
                full.auto_fit = false;
            }
            ui.label("Zoom");
            if ui.button("Fit").clicked() {
                full.auto_fit = true;
                full.pan = Vec2::ZERO;
            }
        });
    });
    ui.separator();
    let available = ui.available_size().max(Vec2::splat(1.0));
    let (viewer_rect, response) = ui.allocate_exact_size(available, Sense::click_and_drag());
    ui.painter().rect_filled(
        viewer_rect,
        CornerRadius::same(4),
        Color32::from_rgb(12, 14, 17),
    );

    match &full.state {
        ModalImageState::Pending => {
            ui.painter().text(
                viewer_rect.center(),
                egui::Align2::CENTER_CENTER,
                "Loading...",
                FontId::proportional(15.0),
                Color32::from_gray(150),
            );
        }
        ModalImageState::Failed(error) => {
            ui.painter().text(
                viewer_rect.center(),
                egui::Align2::CENTER_CENTER,
                error,
                FontId::proportional(15.0),
                Color32::from_rgb(225, 125, 125),
            );
        }
        ModalImageState::Ready {
            texture,
            dimensions,
            ..
        } => {
            let source = Vec2::new(dimensions[0] as f32, dimensions[1] as f32);
            let fit = (viewer_rect.width() / source.x)
                .min(viewer_rect.height() / source.y)
                .clamp(0.01, 4.0);
            if full.auto_fit {
                full.zoom = fit;
                full.pan = Vec2::ZERO;
            }

            if response.hovered() {
                let wheel = ctx.input(|input| {
                    if input.raw_scroll_delta.y.abs() >= input.raw_scroll_delta.x.abs() {
                        input.raw_scroll_delta.y
                    } else {
                        input.raw_scroll_delta.x
                    }
                });
                if wheel != 0.0 {
                    let pointer = ctx
                        .input(|input| input.pointer.hover_pos())
                        .unwrap_or(viewer_rect.center());
                    let old_zoom = full.zoom;
                    let old_size = source * old_zoom;
                    let old_origin = viewer_rect.center() + full.pan - old_size * 0.5;
                    let image_point = (pointer - old_origin) / old_zoom;
                    let minimum = (fit * 0.25).clamp(0.01, 4.0);
                    let new_zoom = (old_zoom * (wheel * 0.003).exp()).clamp(minimum, 4.0);
                    let new_size = source * new_zoom;
                    let new_origin = pointer - image_point * new_zoom;
                    full.zoom = new_zoom;
                    full.pan = new_origin + new_size * 0.5 - viewer_rect.center();
                    full.auto_fit = false;
                }
            }

            if response.dragged() {
                full.pan += ctx.input(|input| input.pointer.delta());
                full.auto_fit = false;
            }

            let scaled = source * full.zoom;
            let excess = ((scaled - viewer_rect.size()) * 0.5).max(Vec2::ZERO);
            full.pan.x = full.pan.x.clamp(-excess.x, excess.x);
            full.pan.y = full.pan.y.clamp(-excess.y, excess.y);
            let image_rect = egui::Rect::from_center_size(viewer_rect.center() + full.pan, scaled);
            ui.painter().with_clip_rect(viewer_rect).image(
                texture.id(),
                image_rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }
}

pub(super) fn upload_preview(
    ctx: &egui::Context,
    name: String,
    preview: PreviewImage,
) -> (egui::TextureHandle, [usize; 2]) {
    let dimensions = preview.size;
    let image = egui::ColorImage::from_rgb(preview.size, &preview.rgb);
    let texture = ctx.load_texture(name, image, egui::TextureOptions::LINEAR);
    (texture, dimensions)
}

pub(super) fn modal_state(
    ctx: &egui::Context,
    name: String,
    result: Result<PreviewImage, String>,
) -> ModalImageState {
    match result {
        Ok(preview) => {
            let color_calibrated = preview.color_calibrated;
            let orientation = preview.metadata.orientation;
            let (texture, dimensions) = upload_preview(ctx, name, preview);
            ModalImageState::Ready {
                texture,
                dimensions,
                color_calibrated,
                orientation,
            }
        }
        Err(error) => ModalImageState::Failed(error),
    }
}

/// The default contact-sheet presentation: every calibrated frame is blended
/// on one canvas, while the list controls which layer is drawn last.
fn camera_stack_view(
    ui: &mut egui::Ui,
    modal: u64,
    cameras: &[CameraPreview],
    reference_camera: Option<&str>,
    overlay_geometry: Option<&CaptureOverlayGeometry>,
    framed_focal_length_mm: Option<i32>,
    open_camera: Option<&mut Option<(String, u64, bool)>>,
) {
    let available = ui.available_size().max(Vec2::splat(1.0));
    let (whole_rect, _) = ui.allocate_exact_size(available, Sense::hover());
    let gap = 14.0;
    let list_width = (whole_rect.width() * 0.27)
        .clamp(190.0, 280.0)
        .min((whole_rect.width() * 0.46).max(1.0));
    let list_rect = egui::Rect::from_min_max(
        egui::pos2(whole_rect.right() - list_width, whole_rect.top()),
        whole_rect.max,
    );
    let main_rect = egui::Rect::from_min_max(
        whole_rect.min,
        egui::pos2(
            (list_rect.left() - gap).max(whole_rect.left()),
            whole_rect.bottom(),
        ),
    );

    let mut hovered = None;
    let mut clicked = None;
    let mut list_ui = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(("camera-stack-list", modal))
            .max_rect(list_rect),
    );
    list_ui.label(RichText::new("Frames").strong());
    list_ui.label(
        RichText::new("Point to bring a frame to the top; click for full preview")
            .color(Color32::from_gray(145))
            .size(11.5),
    );
    list_ui.add_space(6.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .id_salt(("camera-stack-scroll", modal))
        .show(&mut list_ui, |ui| {
            // A continuous hit region prevents the composite from briefly
            // reverting to the reference layer between adjacent rows.
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.set_min_width(ui.available_width());
            for camera in cameras {
                let show_frame = camera_has_multiple_frames(cameras, &camera.camera);
                let is_reference = reference_camera == Some(camera.camera.as_str());
                let response = camera_frame_list_item(ui, camera, show_frame, is_reference);
                let pointer_over_row = ui
                    .ctx()
                    .pointer_hover_pos()
                    .is_some_and(|pointer| response.rect.expand(1.0).contains(pointer));
                if response.hovered() || pointer_over_row {
                    hovered = Some((camera.camera.as_str(), camera.frame_index));
                }
                if response.clicked() && matches!(camera.state, ModalImageState::Ready { .. }) {
                    clicked = Some((camera.camera.clone(), camera.frame_index, show_frame));
                }
            }
        });

    let active_index = stack_camera_index(cameras, reference_camera, hovered);
    let main_response = ui.interact(
        main_rect,
        egui::Id::new(("camera-stack-main", modal)),
        Sense::click(),
    );
    paint_camera_stack(
        ui,
        main_rect,
        cameras,
        active_index,
        reference_camera,
        overlay_geometry,
        framed_focal_length_mm,
    );
    if main_response.clicked()
        && let Some(camera) = active_index.and_then(|index| cameras.get(index))
        && matches!(camera.state, ModalImageState::Ready { .. })
    {
        clicked = Some((
            camera.camera.clone(),
            camera.frame_index,
            camera_has_multiple_frames(cameras, &camera.camera),
        ));
    }
    main_response.on_hover_cursor(egui::CursorIcon::PointingHand);

    if let (Some(target), Some(clicked)) = (open_camera, clicked) {
        *target = Some(clicked);
    }
}

fn camera_frame_list_item(
    ui: &mut egui::Ui,
    camera: &CameraPreview,
    show_frame: bool,
    is_reference: bool,
) -> egui::Response {
    let width = ui.available_width().max(1.0);
    // The response occupies the full slot so hover remains continuous, while
    // an inset visual rectangle restores breathing room between previews.
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 68.0), Sense::click());
    let visual_rect = rect.shrink2(Vec2::new(0.0, 3.0));
    let fill = if response.hovered() {
        Color32::from_rgb(48, 54, 63)
    } else if is_reference {
        Color32::from_rgb(31, 41, 41)
    } else {
        Color32::from_rgb(31, 34, 40)
    };
    ui.painter()
        .rect_filled(visual_rect, CornerRadius::same(6), fill);
    let thumbnail =
        egui::Rect::from_min_size(visual_rect.min + Vec2::splat(5.0), Vec2::splat(52.0));
    ui.painter().rect_filled(
        thumbnail,
        CornerRadius::same(4),
        Color32::from_rgb(18, 20, 24),
    );
    match &camera.state {
        ModalImageState::Pending => {
            placeholder(ui, thumbnail, "…", Color32::from_gray(145));
        }
        ModalImageState::Failed(_) => {
            placeholder(ui, thumbnail, "!", Color32::from_rgb(225, 125, 125));
        }
        ModalImageState::Ready {
            texture,
            dimensions,
            ..
        } => draw_texture(ui, thumbnail, texture, *dimensions),
    }
    let label_pos = egui::pos2(thumbnail.right() + 9.0, visual_rect.top() + 12.0);
    ui.painter().text(
        label_pos,
        egui::Align2::LEFT_TOP,
        camera_frame_label(&camera.camera, camera.frame_index, show_frame),
        FontId::monospace(13.0),
        Color32::from_gray(225),
    );
    if is_reference {
        ui.painter().text(
            label_pos + Vec2::new(0.0, 21.0),
            egui::Align2::LEFT_TOP,
            "Reference",
            FontId::proportional(11.5),
            Color32::from_rgb(104, 218, 195),
        );
    }
    if let ModalImageState::Failed(error) = &camera.state {
        response.clone().on_hover_text(error);
    }
    response
}

fn paint_camera_stack(
    ui: &egui::Ui,
    rect: egui::Rect,
    cameras: &[CameraPreview],
    active_index: Option<usize>,
    reference_camera: Option<&str>,
    overlay_geometry: Option<&CaptureOverlayGeometry>,
    framed_focal_length_mm: Option<i32>,
) {
    ui.painter()
        .rect_filled(rect, CornerRadius::same(8), Color32::from_rgb(15, 17, 21));
    let caption_height = 48.0_f32.min(rect.height() * 0.2);
    let image_region = egui::Rect::from_min_max(
        rect.min,
        egui::pos2(
            rect.right(),
            (rect.bottom() - caption_height).max(rect.top()),
        ),
    );
    let canvas = image_region.shrink2(Vec2::new(12.0, 12.0));
    ui.painter().rect(
        canvas,
        CornerRadius::same(5),
        Color32::from_rgb(10, 12, 15),
        Stroke::new(1.0_f32, Color32::from_rgb(48, 53, 62)),
        StrokeKind::Inside,
    );
    paint_overlay_grid(ui, canvas);

    let reference_index = reference_camera
        .and_then(|name| cameras.iter().position(|camera| camera.camera == name))
        .or((!cameras.is_empty()).then_some(0));
    let ready = cameras
        .iter()
        .enumerate()
        .filter(|(_, camera)| matches!(camera.state, ModalImageState::Ready { .. }))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let base_alpha = match ready.len() {
        0 | 1 => 255,
        2 => 150,
        3..=5 => 105,
        _ => 78,
    };
    for index in ready
        .iter()
        .copied()
        .filter(|index| Some(*index) != active_index)
    {
        paint_overlay_layer(
            ui,
            canvas,
            &cameras[index],
            reference_index.and_then(|index| cameras.get(index)),
            overlay_geometry,
            base_alpha,
            false,
        );
    }

    let Some(active) = active_index.and_then(|index| cameras.get(index)) else {
        placeholder(
            ui,
            image_region,
            "Waiting for frames…",
            Color32::from_gray(145),
        );
        return;
    };
    if matches!(active.state, ModalImageState::Ready { .. }) {
        paint_overlay_layer(
            ui,
            canvas,
            active,
            reference_index.and_then(|index| cameras.get(index)),
            overlay_geometry,
            if ready.len() <= 1 { 255 } else { 170 },
            true,
        );
    } else {
        match &active.state {
            ModalImageState::Pending => {
                placeholder(ui, canvas, "Loading…", Color32::from_gray(145));
            }
            ModalImageState::Failed(_) => {
                placeholder(ui, canvas, "Unavailable", Color32::from_rgb(225, 125, 125))
            }
            ModalImageState::Ready { .. } => {}
        }
    }
    paint_framing_crop(
        ui,
        canvas,
        reference_index.and_then(|index| cameras.get(index)),
        reference_camera,
        framed_focal_length_mm,
    );

    let show_frame = camera_has_multiple_frames(cameras, &active.camera);
    let label = camera_frame_label(&active.camera, active.frame_index, show_frame);
    ui.painter().text(
        egui::pos2(rect.left() + 12.0, rect.bottom() - caption_height + 8.0),
        egui::Align2::LEFT_TOP,
        label,
        FontId::monospace(14.0),
        Color32::from_gray(230),
    );
    let calibration = match &active.state {
        ModalImageState::Ready {
            color_calibrated: true,
            ..
        } => (
            "embedded colour calibration applied",
            Color32::from_rgb(104, 218, 195),
        ),
        ModalImageState::Ready { .. } => {
            ("no embedded colour matrix applied", Color32::from_gray(145))
        }
        ModalImageState::Pending => ("loading corrected preview…", Color32::from_gray(145)),
        ModalImageState::Failed(_) => ("preview unavailable", Color32::from_rgb(225, 125, 125)),
    };
    let geometry_mode = reference_index
        .and_then(|index| cameras.get(index))
        .filter(|reference| {
            overlay_geometry.is_some_and(|geometry| {
                geometry.cameras.contains_key(&reference.camera)
                    && geometry.cameras.contains_key(&active.camera)
            })
        })
        .map_or("focal-scale fallback", |_| "embedded geometry");
    ui.painter().text(
        egui::pos2(rect.left() + 12.0, rect.bottom() - caption_height + 27.0),
        egui::Align2::LEFT_TOP,
        format!(
            "{} frames overlaid · {geometry_mode} · {}",
            ready.len(),
            calibration.0
        ),
        FontId::proportional(11.5),
        calibration.1,
    );
}

fn paint_overlay_layer(
    ui: &egui::Ui,
    canvas: egui::Rect,
    camera: &CameraPreview,
    reference: Option<&CameraPreview>,
    overlay_geometry: Option<&CaptureOverlayGeometry>,
    alpha: u8,
    foreground: bool,
) {
    match &camera.state {
        ModalImageState::Ready {
            texture,
            dimensions,
            orientation,
            ..
        } => {
            let tint = Color32::from_white_alpha(alpha);
            if !draw_calibrated_overlay(
                ui,
                canvas,
                camera,
                reference,
                overlay_geometry,
                texture,
                *dimensions,
                *orientation,
                tint,
                foreground,
            ) {
                let scale = reference.map_or(1.0, |reference| {
                    overlay_scale(&reference.camera, &camera.camera)
                });
                let layer_rect =
                    egui::Rect::from_center_size(canvas.center(), canvas.size() * scale);
                draw_texture_tinted(ui, canvas, layer_rect, texture, *dimensions, tint);
                if foreground {
                    paint_overlay_outline(ui, canvas, fitted_image_rect(layer_rect, *dimensions));
                }
            }
        }
        ModalImageState::Pending | ModalImageState::Failed(_) => {}
    }
}

fn paint_overlay_outline(ui: &egui::Ui, canvas: egui::Rect, image_rect: egui::Rect) {
    ui.painter().with_clip_rect(canvas).rect_stroke(
        image_rect,
        CornerRadius::ZERO,
        Stroke::new(0.9_f32, Color32::from_rgb(104, 218, 195)),
        StrokeKind::Inside,
    );
}

fn paint_framing_crop(
    ui: &egui::Ui,
    canvas: egui::Rect,
    reference: Option<&CameraPreview>,
    reference_camera: Option<&str>,
    framed_focal_length_mm: Option<i32>,
) {
    let Some(focal_length) = framed_focal_length_mm.filter(|value| *value > 0) else {
        return;
    };
    let reference_name = reference_camera
        .or_else(|| reference.map(|camera| camera.camera.as_str()))
        .unwrap_or("A1");
    let scale = framing_crop_scale(reference_name, focal_length as f32);
    let reference_rect = reference
        .and_then(|camera| match &camera.state {
            ModalImageState::Ready { dimensions, .. } => {
                Some(fitted_image_rect(canvas, *dimensions))
            }
            ModalImageState::Pending | ModalImageState::Failed(_) => None,
        })
        .unwrap_or(canvas);
    let crop = egui::Rect::from_center_size(reference_rect.center(), reference_rect.size() * scale);
    let color = Color32::from_rgb(238, 190, 84);
    let painter = ui.painter().with_clip_rect(canvas);
    painter.rect_stroke(
        crop,
        CornerRadius::same(2),
        Stroke::new(2.0_f32, color),
        StrokeKind::Inside,
    );
    let label_rect = egui::Rect::from_min_size(
        crop.left_top() + Vec2::new(5.0, 5.0),
        Vec2::new(132.0, 20.0),
    );
    painter.rect_filled(
        label_rect,
        CornerRadius::same(3),
        Color32::from_black_alpha(185),
    );
    painter.text(
        label_rect.left_center() + Vec2::new(6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        format!("Final crop · {focal_length} mm"),
        FontId::proportional(11.5),
        color,
    );
}

fn framing_crop_scale(reference_camera: &str, framed_focal_length_mm: f32) -> f32 {
    (focal_group_mm(reference_camera) / framed_focal_length_mm.max(0.01)).clamp(0.01, 1.0)
}

fn fitted_image_rect(container: egui::Rect, dimensions: [usize; 2]) -> egui::Rect {
    let source = Vec2::new(dimensions[0].max(1) as f32, dimensions[1].max(1) as f32);
    let scale = (container.width() / source.x).min(container.height() / source.y);
    egui::Rect::from_center_size(container.center(), source * scale)
}

#[allow(clippy::too_many_arguments)]
fn draw_calibrated_overlay(
    ui: &egui::Ui,
    canvas: egui::Rect,
    camera: &CameraPreview,
    reference: Option<&CameraPreview>,
    geometry: Option<&CaptureOverlayGeometry>,
    texture: &egui::TextureHandle,
    dimensions: [usize; 2],
    orientation: u64,
    tint: Color32,
    foreground: bool,
) -> bool {
    const COLUMNS: usize = 18;
    const ROWS: usize = 14;
    let (Some(reference), Some(geometry)) = (reference, geometry) else {
        return false;
    };
    let (Some(source_model), Some(reference_model)) = (
        geometry.cameras.get(&camera.camera),
        geometry.cameras.get(&reference.camera),
    ) else {
        return false;
    };
    let ModalImageState::Ready {
        dimensions: reference_dimensions,
        orientation: reference_orientation,
        ..
    } = &reference.state
    else {
        return false;
    };
    let reference_rect = fitted_image_rect(canvas, *reference_dimensions);
    let mut mesh = egui::Mesh::with_texture(texture.id());
    let mut vertices = vec![None; (COLUMNS + 1) * (ROWS + 1)];
    let mut positions = vec![None; (COLUMNS + 1) * (ROWS + 1)];
    for row in 0..=ROWS {
        for column in 0..=COLUMNS {
            let uv = egui::pos2(column as f32 / COLUMNS as f32, row as f32 / ROWS as f32);
            let calibration = display_to_calibration(uv, orientation);
            let source_pixel = [
                f64::from(calibration.x) * (source_model.width - 1) as f64,
                f64::from(calibration.y) * (source_model.height - 1) as f64,
            ];
            let Some(reference_pixel) = reference_model.map_from(source_model, source_pixel, 1.0e9)
            else {
                continue;
            };
            let normalized = egui::pos2(
                (reference_pixel[0] / (reference_model.width - 1) as f64) as f32,
                (reference_pixel[1] / (reference_model.height - 1) as f64) as f32,
            );
            let displayed = calibration_to_display(normalized, *reference_orientation);
            let position = egui::pos2(
                reference_rect.left() + displayed.x * reference_rect.width(),
                reference_rect.top() + displayed.y * reference_rect.height(),
            );
            let index = mesh.vertices.len() as u32;
            mesh.vertices.push(egui::epaint::Vertex {
                pos: position,
                uv,
                color: tint,
            });
            let grid_index = row * (COLUMNS + 1) + column;
            vertices[grid_index] = Some(index);
            positions[grid_index] = Some(position);
        }
    }
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let top_left = vertices[row * (COLUMNS + 1) + column];
            let top_right = vertices[row * (COLUMNS + 1) + column + 1];
            let bottom_left = vertices[(row + 1) * (COLUMNS + 1) + column];
            let bottom_right = vertices[(row + 1) * (COLUMNS + 1) + column + 1];
            if let (Some(a), Some(b), Some(c)) = (top_left, top_right, bottom_left) {
                mesh.add_triangle(a, b, c);
            }
            if let (Some(a), Some(b), Some(c)) = (top_right, bottom_right, bottom_left) {
                mesh.add_triangle(a, b, c);
            }
        }
    }
    if mesh.indices.is_empty() || dimensions.contains(&0) {
        return false;
    }
    ui.painter()
        .with_clip_rect(canvas)
        .add(egui::Shape::mesh(mesh));
    if foreground {
        paint_calibrated_outline(ui, canvas, &positions, COLUMNS, ROWS);
    }
    true
}

fn paint_calibrated_outline(
    ui: &egui::Ui,
    canvas: egui::Rect,
    positions: &[Option<egui::Pos2>],
    columns: usize,
    rows: usize,
) {
    let painter = ui.painter().with_clip_rect(canvas);
    let stroke = Stroke::new(0.9_f32, Color32::from_rgb(104, 218, 195));
    let stride = columns + 1;

    for column in 0..columns {
        paint_outline_segment(&painter, positions[column], positions[column + 1], stroke);
        let bottom = rows * stride;
        paint_outline_segment(
            &painter,
            positions[bottom + column],
            positions[bottom + column + 1],
            stroke,
        );
    }
    for row in 0..rows {
        let current = row * stride;
        let next = (row + 1) * stride;
        paint_outline_segment(&painter, positions[current], positions[next], stroke);
        paint_outline_segment(
            &painter,
            positions[current + columns],
            positions[next + columns],
            stroke,
        );
    }
}

fn paint_outline_segment(
    painter: &egui::Painter,
    start: Option<egui::Pos2>,
    end: Option<egui::Pos2>,
    stroke: Stroke,
) {
    if let (Some(start), Some(end)) = (start, end) {
        painter.line_segment([start, end], stroke);
    }
}

fn display_to_calibration(point: egui::Pos2, orientation: u64) -> egui::Pos2 {
    match orientation {
        1 => egui::pos2(point.y, 1.0 - point.x),
        2 => egui::pos2(1.0 - point.y, point.x),
        3 => egui::pos2(1.0 - point.y, 1.0 - point.x),
        4 => egui::pos2(point.y, point.x),
        5 => egui::pos2(point.x, 1.0 - point.y),
        6 => egui::pos2(1.0 - point.x, point.y),
        7 => egui::pos2(1.0 - point.x, 1.0 - point.y),
        _ => point,
    }
}

fn calibration_to_display(point: egui::Pos2, orientation: u64) -> egui::Pos2 {
    match orientation {
        1 => egui::pos2(1.0 - point.y, point.x),
        2 => egui::pos2(point.y, 1.0 - point.x),
        3 => egui::pos2(1.0 - point.y, 1.0 - point.x),
        4 => egui::pos2(point.y, point.x),
        5 => egui::pos2(point.x, 1.0 - point.y),
        6 => egui::pos2(1.0 - point.x, point.y),
        7 => egui::pos2(1.0 - point.x, 1.0 - point.y),
        _ => point,
    }
}

fn draw_texture_tinted(
    ui: &egui::Ui,
    clip_rect: egui::Rect,
    rect: egui::Rect,
    texture: &egui::TextureHandle,
    dimensions: [usize; 2],
    tint: Color32,
) {
    let source = Vec2::new(dimensions[0] as f32, dimensions[1] as f32);
    let scale = (rect.width() / source.x).min(rect.height() / source.y);
    let draw_rect = egui::Rect::from_center_size(rect.center(), source * scale);
    ui.painter().with_clip_rect(clip_rect).image(
        texture.id(),
        draw_rect,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        tint,
    );
}

fn paint_overlay_grid(ui: &egui::Ui, rect: egui::Rect) {
    let painter = ui.painter().with_clip_rect(rect);
    let stroke = Stroke::new(0.6_f32, Color32::from_gray(32));
    let step = 48.0;
    let mut x = rect.left() + step;
    while x < rect.right() {
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            stroke,
        );
        x += step;
    }
    let mut y = rect.top() + step;
    while y < rect.bottom() {
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            stroke,
        );
        y += step;
    }
}

fn overlay_scale(reference: &str, camera: &str) -> f32 {
    focal_group_mm(reference) / focal_group_mm(camera)
}

fn focal_group_mm(camera: &str) -> f32 {
    match camera
        .chars()
        .next()
        .map(|value| value.to_ascii_uppercase())
    {
        Some('B') => 70.0,
        Some('C') => 150.0,
        _ => 28.0,
    }
}

fn stack_camera_index(
    cameras: &[CameraPreview],
    reference_camera: Option<&str>,
    hovered: Option<(&str, u64)>,
) -> Option<usize> {
    hovered
        .and_then(|(name, frame)| {
            cameras
                .iter()
                .position(|camera| camera.camera == name && camera.frame_index == frame)
        })
        .or_else(|| {
            reference_camera
                .and_then(|name| cameras.iter().position(|camera| camera.camera == name))
        })
        .or((!cameras.is_empty()).then_some(0))
}

fn camera_has_multiple_frames(cameras: &[CameraPreview], camera: &str) -> bool {
    cameras
        .iter()
        .filter(|candidate| candidate.camera == camera)
        .count()
        > 1
}

fn camera_contact_grid(
    ui: &mut egui::Ui,
    modal: u64,
    cameras: &[CameraPreview],
    mut open_camera: Option<&mut Option<(String, u64, bool)>>,
) {
    let scroll_style = egui::style::ScrollStyle::solid();
    let scrollbar_width = scroll_style.allocated_width();
    ui.style_mut().spacing.scroll = scroll_style;
    let available_width = (ui.available_width() - scrollbar_width).max(1.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
        .show(ui, |ui| {
            ui.set_min_width(available_width);
            let card_width = 205.0;
            let columns = ((ui.available_width() + CARD_GAP) / (card_width + CARD_GAP))
                .floor()
                .max(1.0) as usize;
            egui::Grid::new(("camera-contact-grid", modal))
                .num_columns(columns)
                .spacing(Vec2::splat(CARD_GAP))
                .show(ui, |ui| {
                    for (index, camera) in cameras.iter().enumerate() {
                        let show_frame = camera_has_multiple_frames(cameras, &camera.camera);
                        if modal_camera_card(ui, camera, show_frame, card_width)
                            && let Some(target) = open_camera.as_deref_mut()
                        {
                            *target = Some((camera.camera.clone(), camera.frame_index, show_frame));
                        }
                        if (index + 1) % columns == 0 {
                            ui.end_row();
                        }
                    }
                });
        });
}

fn modal_camera_card(
    ui: &mut egui::Ui,
    camera: &CameraPreview,
    show_frame: bool,
    width: f32,
) -> bool {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, width + 34.0), Sense::click());
    ui.painter().rect(
        rect,
        CornerRadius::same(8),
        if response.hovered() {
            Color32::from_rgb(42, 45, 53)
        } else {
            Color32::from_rgb(31, 34, 40)
        },
        Stroke::new(1.0_f32, Color32::from_rgb(60, 64, 74)),
        StrokeKind::Inside,
    );
    let image_rect =
        egui::Rect::from_min_size(rect.min + Vec2::splat(8.0), Vec2::splat(width - 16.0));
    ui.painter().rect_filled(
        image_rect,
        CornerRadius::same(5),
        Color32::from_rgb(18, 20, 24),
    );
    match &camera.state {
        ModalImageState::Pending => {
            placeholder(ui, image_rect, "Loading...", Color32::from_gray(145));
        }
        ModalImageState::Failed(error) => {
            placeholder(
                ui,
                image_rect,
                "Unavailable",
                Color32::from_rgb(225, 125, 125),
            );
            response.clone().on_hover_text(error);
        }
        ModalImageState::Ready {
            texture,
            dimensions,
            ..
        } => draw_texture(ui, image_rect, texture, *dimensions),
    }
    ui.painter().text(
        egui::pos2(rect.left() + 9.0, image_rect.bottom() + 8.0),
        egui::Align2::LEFT_TOP,
        camera_frame_label(&camera.camera, camera.frame_index, show_frame),
        FontId::monospace(14.0),
        Color32::from_gray(225),
    );
    response.clicked() && matches!(camera.state, ModalImageState::Ready { .. })
}

fn camera_frame_label(camera: &str, frame_index: u64, show_frame: bool) -> String {
    if show_frame {
        format!("{camera} - frame {}", frame_index + 1)
    } else {
        camera.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(camera: &str, frame_index: u64) -> CameraPreview {
        CameraPreview {
            camera: camera.to_owned(),
            frame_index,
            state: ModalImageState::Pending,
        }
    }

    #[test]
    fn stack_defaults_to_reference_and_hover_temporarily_promotes_a_frame() {
        let cameras = vec![pending("A1", 0), pending("B2", 0), pending("B2", 1)];
        assert_eq!(stack_camera_index(&cameras, Some("B2"), None), Some(1));
        assert_eq!(
            stack_camera_index(&cameras, Some("B2"), Some(("A1", 0))),
            Some(0)
        );
        assert_eq!(
            stack_camera_index(&cameras, Some("B2"), Some(("B2", 1))),
            Some(2)
        );
        assert!(camera_has_multiple_frames(&cameras, "B2"));
        assert!(!camera_has_multiple_frames(&cameras, "A1"));
    }

    #[test]
    fn stack_falls_back_to_first_frame_when_reference_is_absent() {
        let cameras = vec![pending("C1", 0), pending("C2", 0)];
        assert_eq!(stack_camera_index(&cameras, Some("B2"), None), Some(0));
        assert_eq!(stack_camera_index(&[], Some("B2"), None), None);
        assert_eq!(camera_frame_label("C1", 2, true), "C1 - frame 3");
        assert_eq!(camera_frame_label("C1", 2, false), "C1");
    }

    #[test]
    fn overlay_scales_each_focal_group_into_the_reference_field_of_view() {
        assert_eq!(overlay_scale("A1", "A4"), 1.0);
        assert!((overlay_scale("A1", "B2") - 0.4).abs() < f32::EPSILON);
        assert!((overlay_scale("B2", "A1") - 2.5).abs() < f32::EPSILON);
        assert!((overlay_scale("A1", "C3") - 28.0 / 150.0).abs() < f32::EPSILON);
        assert!((framing_crop_scale("A1", 100.0) - 0.28).abs() < f32::EPSILON);
        assert!((framing_crop_scale("B2", 100.0) - 0.7).abs() < f32::EPSILON);
        assert_eq!(framing_crop_scale("C3", 150.0), 1.0);
        assert_eq!(framing_crop_scale("B2", 28.0), 1.0);
    }

    #[test]
    fn display_and_calibration_orientations_are_inverses() {
        let point = egui::pos2(0.23, 0.71);
        for orientation in 0..=7 {
            let calibration = display_to_calibration(point, orientation);
            let restored = calibration_to_display(calibration, orientation);
            assert!((restored.x - point.x).abs() < f32::EPSILON);
            assert!((restored.y - point.y).abs() < f32::EPSILON);
        }
    }
}
