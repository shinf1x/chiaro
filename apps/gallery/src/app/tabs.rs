use super::*;

impl GalleryApp {
    pub(super) fn toolbar(&mut self, ui: &mut egui::Ui) {
        let mut selected = None;
        let mut new_folder = false;
        let mut close_folder = None;
        let tab_height = ui.spacing().interact_size.y.max(20.0) + 2.0;
        let header_height = BrandTextures::height(ui.ctx()).max(tab_height);
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), header_height),
            Layout::left_to_right(Align::Center),
            |ui| {
                self.brand.show(ui);
                header_divider(ui, tab_height);
                let settings_width = 90.0;
                egui::ScrollArea::horizontal()
                    .id_salt("source-tabs")
                    .max_width((ui.available_width() - settings_width).max(40.0))
                    .max_height(header_height)
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .show(ui, |ui| {
                        ui.allocate_ui_with_layout(
                            Vec2::new(0.0, header_height),
                            Layout::left_to_right(Align::Center),
                            |ui| {
                                for device in &self.devices {
                                    let key = TabKey::for_device(device);
                                    let label = device.tab_label();
                                    Self::source_tab_frame(ui, self.active_tab == key, |ui| {
                                        if ui
                                            .add(egui::Button::new(label).frame(false))
                                            .on_hover_text("Connected Light L16")
                                            .clicked()
                                        {
                                            selected = Some(key);
                                        }
                                    });
                                }
                                for folder in &self.folder_tabs {
                                    let key = TabKey::Folder(folder.id);
                                    let active = self.active_tab == key;
                                    Self::source_tab_frame(ui, active, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.spacing_mut().item_spacing.x = 7.0;
                                            if ui
                                                .add(
                                                    egui::Button::new(folder_tab_name(
                                                        &folder.path,
                                                    ))
                                                    .frame(false),
                                                )
                                                .on_hover_text(folder.path.display().to_string())
                                                .clicked()
                                            {
                                                selected = Some(key);
                                            }
                                            if ui
                                                .add(
                                                    egui::Button::new(
                                                        RichText::new("X")
                                                            .size(12.0)
                                                            .strong()
                                                            .color(Color32::from_gray(210)),
                                                    )
                                                    .small()
                                                    .min_size(Vec2::splat(20.0))
                                                    .fill(Color32::from_rgb(43, 46, 54))
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        Color32::from_rgb(78, 83, 94),
                                                    )),
                                                )
                                                .on_hover_text("Close tab")
                                                .clicked()
                                            {
                                                close_folder = Some(folder.id);
                                            }
                                        });
                                    });
                                }
                                if ui.button("+ Folder...").clicked() {
                                    new_folder = true;
                                }
                            },
                        );
                    });
                // The Settings tab sits at the far right, outside the
                // scrolling strip of sources.
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    Self::source_tab_frame(ui, self.active_tab == TabKey::Settings, |ui| {
                        if ui.add(egui::Button::new("Settings").frame(false)).clicked() {
                            selected = Some(TabKey::Settings);
                        }
                    });
                });
            },
        );

        if let Some(key) = selected {
            self.select_tab(key, false);
        }
        if let Some(id) = close_folder {
            let key = TabKey::Folder(id);
            self.folder_tabs.retain(|tab| tab.id != id);
            self.saved_views.remove(&key);
            if self.active_tab == key {
                if let Some(sheet) = self.contact_sheet.take() {
                    self.loader.cancel_modal(sheet.id);
                }
                self.active_tab = TabKey::FolderInput;
                self.current_view = TabViewState::default();
                self.folder_prompt_open = false;
            }
        }
        if new_folder {
            self.select_tab(TabKey::FolderInput, false);
            self.folder_prompt_open = true;
        }

        ui.separator();
        match self.active_tab.clone() {
            TabKey::FolderInput if self.folder_prompt_open => self.folder_controls(ui),
            TabKey::FolderInput | TabKey::Settings => {}
            TabKey::Folder(id) => self.open_folder_controls(ui, id),
            TabKey::Device { location_id, mode } => self.device_controls(ui, location_id, mode),
        }

        if self.active_tab == TabKey::Settings {
            return;
        }
        ui.add_space(3.0);
        let compact_preview_controls = ui.available_width() < 340.0;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Preview size").color(Color32::from_gray(155)));
            let has_captures = !self.current_view.items.is_empty();
            let trailing_width = if has_captures && !compact_preview_controls {
                112.0
            } else {
                0.0
            };
            ui.spacing_mut().slider_width = (ui.available_width() - trailing_width)
                .clamp(1.0, 240.0)
                .min(ui.available_width());
            ui.add(
                egui::Slider::new(&mut self.preview_size, MIN_PREVIEW_SIZE..=440.0)
                    .show_value(false),
            );
            if has_captures && !compact_preview_controls {
                ui.separator();
                let label = ui.label(format!("{} captures", self.current_view.items.len()));
                if let Some(cache) = self.loader.thumbnail_cache() {
                    label.on_hover_text(format!(
                        "Thumbnails are cached in {}",
                        cache.root().display()
                    ));
                }
            }
        });
        if compact_preview_controls && !self.current_view.items.is_empty() {
            ui.label(
                RichText::new(format!("{} captures", self.current_view.items.len()))
                    .size(12.0)
                    .color(Color32::from_gray(155)),
            );
        }
    }

    pub(super) fn source_tab_frame(
        ui: &mut egui::Ui,
        active: bool,
        add_contents: impl FnOnce(&mut egui::Ui),
    ) {
        egui::Frame::new()
            .fill(if active {
                ui.visuals().selection.bg_fill
            } else {
                Color32::TRANSPARENT
            })
            .stroke(Stroke::new(
                1.0_f32,
                if active {
                    ui.visuals().selection.stroke.color
                } else {
                    Color32::from_gray(58)
                },
            ))
            .corner_radius(CornerRadius::same(5))
            .inner_margin(Margin::symmetric(5, 1))
            .show(ui, |ui| {
                ui.set_min_height(ui.spacing().interact_size.y);
                add_contents(ui);
            });
    }

    pub(super) fn status_bar(&mut self, ui: &mut egui::Ui) {
        let contact_loading = self
            .contact_sheet
            .as_ref()
            .and_then(|sheet| match &sheet.state {
                ContactState::Loading {
                    transferred,
                    total,
                    cameras,
                    ..
                } => {
                    let transfer = if *total > 0 {
                        format!("{} / {}", format_bytes(*transferred), format_bytes(*total))
                    } else {
                        "starting".to_owned()
                    };
                    Some((
                        format!(
                            "Loading contact sheet for {}: {transfer} ({} cameras ready)",
                            sheet.name,
                            cameras.len()
                        ),
                        (*total > 0).then(|| (*transferred as f32 / *total as f32).clamp(0.0, 1.0)),
                    ))
                }
                ContactState::Ready { cameras, .. } => {
                    let pending = cameras
                        .iter()
                        .filter(|camera| matches!(camera.state, ModalImageState::Pending))
                        .count();
                    (pending > 0).then(|| {
                        (
                            format!(
                                "Decoding camera previews for {}: {} / {} ready",
                                sheet.name,
                                cameras.len() - pending,
                                cameras.len()
                            ),
                            None,
                        )
                    })
                }
                ContactState::Failed(_) => None,
            });
        let pending = self
            .current_view
            .items
            .iter()
            .filter(|item| matches!(item.state, ItemState::Pending { .. }))
            .count();
        ui.horizontal(|ui| {
            // Export controls anchor the right edge; the loading message and
            // progress fill whatever remains on the left.
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                self.export_controls(ui);
                if !self.current_view.items.is_empty() {
                    ui.separator();
                }
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    if self.export_status(ui, pending) {
                        return;
                    }
                    if contact_loading.is_some() || self.current_view.busy.is_some() || pending > 0
                    {
                        ui.spinner();
                    }
                    if let Some((message, _)) = &contact_loading {
                        ui.label(message);
                    } else if let Some(busy) = &self.current_view.busy {
                        ui.label(busy);
                    } else if pending > 0 {
                        ui.label(format!("Loading previews ({pending} remaining)"));
                    } else if let Some(status) = &self.current_view.status {
                        ui.label(RichText::new(status).color(Color32::from_gray(165)));
                    } else if !self.current_view.items.is_empty() {
                        ui.label("Ready");
                    }
                    if pending > 0 && self.current_view.busy.is_some() {
                        ui.separator();
                        ui.label(format!("{pending} previews queued"));
                    }
                });
            });
        });
        if let Some((_, Some(fraction))) = contact_loading {
            ui.add(egui::ProgressBar::new(fraction));
        } else if let Some((fetched, total)) = self.current_view.listing_progress
            && total > 0
        {
            let fraction = (fetched as f32 / total as f32).clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(fraction)
                    .text(format!("Reading camera index: {fetched} / {total}")),
            );
        }
    }

    pub(super) fn folder_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let text_width = (ui.available_width() - 190.0).clamp(170.0, 560.0);
            let submitted = ui
                .add_sized(
                    [text_width, 28.0],
                    egui::TextEdit::singleline(&mut self.folder_input)
                        .hint_text("Folder containing .lri files"),
                )
                .lost_focus()
                && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if ui.button("Open path").clicked() || submitted {
                self.open_folder_input();
            }
            if ui.button("Browse...").clicked() {
                self.open_folder_picker(ui.ctx());
            }
            ui.label(
                RichText::new("or drop a folder anywhere in the window")
                    .color(Color32::from_gray(135)),
            );
        });
    }

    pub(super) fn open_folder_controls(&mut self, ui: &mut egui::Ui, id: u64) {
        let Some(folder) = self
            .folder_tabs
            .iter()
            .find(|folder| folder.id == id)
            .map(|folder| folder.path.clone())
        else {
            return;
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(folder.display().to_string()).monospace());
            if ui.button("Reload").clicked() {
                self.select_tab(TabKey::Folder(id), true);
            }
            if ui.button("Choose another...").clicked() {
                self.open_folder_picker(ui.ctx());
            }
        });
    }

    pub(super) fn device_controls(
        &mut self,
        ui: &mut egui::Ui,
        location_id: u64,
        mode: DeviceMode,
    ) {
        let temporarily_missing = self.device_missing_since.is_some()
            && self.active_tab == TabKey::Device { location_id, mode };
        let connected = !temporarily_missing
            && self
                .devices
                .iter()
                .any(|device| device.location_id == location_id && device.mode == mode);
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                if connected {
                    Color32::from_rgb(105, 205, 135)
                } else {
                    Color32::from_rgb(225, 125, 125)
                },
                if connected {
                    format!("Connected: Light L16 over {}", mode.label())
                } else if temporarily_missing {
                    format!("Reconnecting: Light L16 over {}", mode.label())
                } else {
                    "Disconnected: Light L16".to_owned()
                },
            );
            if ui
                .add_enabled(
                    connected && self.current_view.busy.is_none(),
                    egui::Button::new("Reload"),
                )
                .clicked()
            {
                self.select_tab(TabKey::Device { location_id, mode }, true);
            }
        });
    }
}

fn header_divider(ui: &mut egui::Ui, height: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, height), Sense::hover());
    ui.painter().line_segment(
        [rect.center_top(), rect.center_bottom()],
        ui.visuals().widgets.noninteractive.bg_stroke,
    );
}
