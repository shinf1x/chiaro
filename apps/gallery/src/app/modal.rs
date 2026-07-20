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
                            camera_contact_grid(ui, sheet.id, cameras, None);
                        }
                    }
                    ContactState::Failed(error) => {
                        ui.centered_and_justified(|ui| {
                            ui.colored_label(Color32::from_rgb(225, 125, 125), error);
                        });
                    }
                    ContactState::Ready { cameras, metadata, .. } => {
                        ui.label(
                            RichText::new(metadata_summary(metadata))
                                .color(Color32::from_gray(155)),
                        );
                        ui.label(
                            RichText::new("Select a camera to load its full native preview")
                                .color(Color32::from_gray(145)),
                        );
                        camera_contact_grid(ui, sheet.id, cameras, Some(&mut open_camera));
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
            let (texture, dimensions) = upload_preview(ctx, name, preview);
            ModalImageState::Ready {
                texture,
                dimensions,
            }
        }
        Err(error) => ModalImageState::Failed(error),
    }
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
                        let show_frame = cameras
                            .iter()
                            .filter(|candidate| candidate.camera == camera.camera)
                            .count()
                            > 1;
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
