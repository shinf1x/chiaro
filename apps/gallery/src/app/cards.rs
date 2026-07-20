use super::*;

impl GalleryApp {
    pub(super) fn gallery(&mut self, ui: &mut egui::Ui) {
        let scroll_style = egui::style::ScrollStyle::solid();
        let scrollbar_width = scroll_style.allocated_width();
        ui.style_mut().spacing.scroll = scroll_style;
        let available = (ui.available_width() - scrollbar_width).max(1.0);
        let (columns, card_width) = gallery_layout(available, self.preview_size);

        let rows = self.current_view.items.len().div_ceil(columns);
        let row_height = card_width + CARD_EXTRA_HEIGHT + CARD_GAP;
        let mut clicked = None;
        let mut visible_ids = Vec::new();
        let mut promotions = Vec::new();
        let items = &mut self.current_view.items;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
            .show_rows(ui, row_height, rows, |ui, visible_rows| {
                ui.set_min_width(available);
                ui.spacing_mut().item_spacing.y = 0.0;
                for row in visible_rows {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = CARD_GAP;
                        for column in 0..columns {
                            let index = row * columns + column;
                            let Some(item) = items.get_mut(index) else {
                                break;
                            };
                            visible_ids.push(item.id);
                            if matches!(item.state, ItemState::Pending { .. }) {
                                promotions.push((
                                    item.id,
                                    item.preview_revision,
                                    item.source.preview.clone(),
                                    item.source.capture.clone(),
                                ));
                            }
                            if Self::card(ui, item, card_width) {
                                clicked = Some(index);
                            }
                        }
                    });
                    ui.add_space(CARD_GAP);
                }
            });
        self.loader.set_visible_gallery(visible_ids.iter().copied());
        for (id, revision, source, capture) in promotions {
            self.loader
                .prioritize_gallery_preview(self.generation, id, revision, source, capture);
        }
        if let Some(index) = clicked {
            self.open_contact_sheet(index);
        }
    }

    pub(super) fn card(ui: &mut egui::Ui, item: &GalleryItem, width: f32) -> bool {
        let height = width + CARD_EXTRA_HEIGHT;
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click());
        let fill = if response.hovered() {
            Color32::from_rgb(37, 40, 47)
        } else {
            Color32::from_rgb(29, 32, 38)
        };
        ui.painter().rect(
            rect,
            CornerRadius::same(10),
            fill,
            Stroke::new(1.0_f32, Color32::from_rgb(55, 59, 68)),
            StrokeKind::Inside,
        );

        let content = rect.shrink2(Vec2::new(9.0, 9.0));
        let image_rect = egui::Rect::from_min_size(content.min, Vec2::splat(width - 18.0));
        ui.painter().rect_filled(
            image_rect,
            CornerRadius::same(6),
            Color32::from_rgb(20, 22, 27),
        );

        match &item.state {
            ItemState::Idle => placeholder(ui, image_rect, "Waiting...", Color32::from_gray(125)),
            ItemState::Pending { transferred, total } => {
                let text = if *total > 0 {
                    format!(
                        "Loading preview\n{:.0}%",
                        (*transferred as f64 / *total as f64 * 100.0).clamp(0.0, 100.0)
                    )
                } else {
                    "Loading...".to_owned()
                };
                placeholder(ui, image_rect, &text, Color32::from_gray(145))
            }
            ItemState::Ready {
                texture,
                camera,
                dimensions,
                metadata: _,
            } => {
                draw_texture(ui, image_rect, texture, *dimensions);
                ui.painter().text(
                    image_rect.right_bottom() - egui::vec2(7.0, 7.0),
                    egui::Align2::RIGHT_BOTTOM,
                    camera,
                    FontId::monospace(12.0),
                    Color32::from_white_alpha(210),
                );
            }
            ItemState::Failed(error) => {
                placeholder(
                    ui,
                    image_rect,
                    "Preview unavailable\nClick to inspect LRI",
                    Color32::from_rgb(225, 125, 125),
                );
                response.clone().on_hover_text(error);
            }
        }

        let name_rect = egui::Rect::from_min_max(
            egui::pos2(content.left(), image_rect.bottom() + 9.0),
            content.right_bottom(),
        );
        let metadata = match &item.state {
            ItemState::Ready { metadata, .. } => Some(metadata),
            _ => None,
        };
        let icon_width = metadata.map_or(0.0, capture_icon_width);
        let filename_font = FontId::proportional(14.0);
        let filename_color = Color32::from_gray(225);
        ui.painter().text(
            name_rect.left_top(),
            egui::Align2::LEFT_TOP,
            elide_text(
                ui,
                &item.source.name,
                (name_rect.width() - icon_width).max(30.0),
                &filename_font,
                filename_color,
            ),
            filename_font,
            filename_color,
        );
        if let Some(metadata) = metadata {
            paint_capture_icons(ui, name_rect, metadata, fill);
            paint_metadata(ui, name_rect, metadata);
        }
        let clicked = response.clicked();
        response.on_hover_text(&item.source.location_label);
        clicked
    }
}

fn gallery_layout(available: f32, preview_size: f32) -> (usize, f32) {
    let available = available.max(1.0);
    let card_width = preview_size.min(available).max(1.0);
    let columns = ((available + CARD_GAP) / (card_width + CARD_GAP))
        .floor()
        .max(1.0) as usize;
    (columns, card_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wider_gallery_adds_columns_without_shrinking_cards() {
        assert_eq!(gallery_layout(700.0, 210.0), (3, 210.0));
        assert_eq!(gallery_layout(920.0, 210.0), (4, 210.0));
        assert_eq!(gallery_layout(1_400.0, 210.0), (6, 210.0));
    }

    #[test]
    fn card_shrinks_only_when_the_gallery_is_narrower_than_it() {
        assert_eq!(gallery_layout(100.0, 210.0), (1, 100.0));
    }
}
