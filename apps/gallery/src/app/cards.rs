use super::*;

/// Side of the selection checkbox drawn in a card's image corner.
const CHECKBOX_SIZE: f32 = 22.0;
/// Pointer distance (from the checkbox edge) at which a hidden checkbox fades in.
const CHECKBOX_REVEAL_DISTANCE: f32 = 40.0;

/// What a click on a card asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CardAction {
    /// Open the contact sheet (plain click on the photo).
    OpenPreview,
    /// Flip this card's selection (click on the checkbox or card chrome).
    Toggle,
    /// Apply the anchor's selection state to every card between the anchor and
    /// this one (shift-click).
    Range,
}

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
        let selected = &self.current_view.selected;
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
                            if let Some(action) =
                                Self::card(ui, item, card_width, selected.contains(&item.id))
                            {
                                clicked = Some((index, action));
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
        if let Some((index, action)) = clicked {
            self.apply_card_action(index, action);
        }
    }

    /// Resolve a card click into a preview or a selection change.
    pub(super) fn apply_card_action(&mut self, index: usize, action: CardAction) {
        if action == CardAction::OpenPreview {
            self.open_contact_sheet(index);
            return;
        }
        let view = &mut self.current_view;
        let ids = view.items.iter().map(|item| item.id).collect::<Vec<_>>();
        update_selection(
            &mut view.selected,
            &mut view.selection_anchor,
            &ids,
            index,
            action,
        );
    }

    pub(super) fn clear_selection(&mut self) {
        self.current_view.selected.clear();
        self.current_view.selection_anchor = None;
    }

    pub(super) fn selected_count(&self) -> usize {
        self.current_view.selected.len()
    }

    pub(super) fn card(
        ui: &mut egui::Ui,
        item: &GalleryItem,
        width: f32,
        selected: bool,
    ) -> Option<CardAction> {
        let height = width + CARD_EXTRA_HEIGHT;
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click());
        let fill = if selected {
            Color32::from_rgb(36, 44, 60)
        } else if response.hovered() {
            Color32::from_rgb(37, 40, 47)
        } else {
            Color32::from_rgb(29, 32, 38)
        };
        let accent = ui.visuals().selection.stroke.color;
        let accent_fill = ui.visuals().selection.bg_fill;
        ui.painter().rect(
            rect,
            CornerRadius::same(10),
            fill,
            if selected {
                Stroke::new(1.5_f32, accent)
            } else {
                Stroke::new(1.0_f32, Color32::from_rgb(55, 59, 68))
            },
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

        // Selection checkbox: hidden until the pointer comes close, always
        // shown once the card is selected.
        let checkbox_rect = egui::Rect::from_min_size(
            image_rect.left_top() + Vec2::splat(8.0),
            Vec2::splat(CHECKBOX_SIZE),
        );
        let pointer = ui.input(|input| input.pointer.latest_pos());
        let pointer_distance = pointer.map(|pos| checkbox_rect.distance_to_pos(pos));
        let reveal = pointer_distance.is_some_and(|distance| distance <= CHECKBOX_REVEAL_DISTANCE);
        if selected || reveal {
            paint_checkbox(ui, checkbox_rect, selected, reveal, accent_fill);
        }

        let action = if response.clicked() {
            let shift = ui.input(|input| input.modifiers.shift);
            let position = response.interact_pointer_pos();
            let on_checkbox = position.is_some_and(|pos| checkbox_rect.contains(pos));
            let on_photo = position.is_some_and(|pos| image_rect.contains(pos));
            if shift {
                Some(CardAction::Range)
            } else if on_photo && !on_checkbox {
                Some(CardAction::OpenPreview)
            } else {
                Some(CardAction::Toggle)
            }
        } else {
            None
        };
        response.on_hover_text(&item.source.location_label);
        action
    }
}

fn paint_checkbox(
    ui: &egui::Ui,
    rect: egui::Rect,
    checked: bool,
    highlighted: bool,
    accent: Color32,
) {
    let painter = ui.painter();
    let (fill, stroke) = if checked {
        (accent, Color32::from_white_alpha(235))
    } else if highlighted {
        (
            Color32::from_black_alpha(170),
            Color32::from_white_alpha(220),
        )
    } else {
        (
            Color32::from_black_alpha(120),
            Color32::from_white_alpha(150),
        )
    };
    painter.rect(
        rect,
        CornerRadius::same(5),
        fill,
        Stroke::new(1.5_f32, stroke),
        StrokeKind::Inside,
    );
    if checked {
        let left = rect.left_top() + Vec2::new(rect.width() * 0.26, rect.height() * 0.53);
        let middle = rect.left_top() + Vec2::new(rect.width() * 0.43, rect.height() * 0.71);
        let right = rect.left_top() + Vec2::new(rect.width() * 0.76, rect.height() * 0.32);
        let tick = Stroke::new(2.4_f32, Color32::WHITE);
        painter.line_segment([left, middle], tick);
        painter.line_segment([middle, right], tick);
    }
}

/// Selection state machine shared by checkbox, card-chrome, and shift clicks.
///
/// `Toggle` flips one item and makes it the anchor. `Range` copies the anchor's
/// current state onto every item between the anchor and `index`, leaving the
/// anchor in place so further shift-clicks re-range from the same origin and
/// selections outside the range untouched, which is what allows several
/// disjoint ranges. Without an anchor, `Range` selects the item and anchors it.
pub(super) fn update_selection(
    selected: &mut HashSet<u64>,
    anchor: &mut Option<u64>,
    ids: &[u64],
    index: usize,
    action: CardAction,
) {
    let Some(&id) = ids.get(index) else {
        return;
    };
    match action {
        CardAction::OpenPreview => {}
        CardAction::Toggle => {
            if !selected.remove(&id) {
                selected.insert(id);
            }
            *anchor = Some(id);
        }
        CardAction::Range => {
            let anchor_index = anchor.and_then(|anchor| ids.iter().position(|id| *id == anchor));
            let Some(anchor_index) = anchor_index else {
                selected.insert(id);
                *anchor = Some(id);
                return;
            };
            let select = selected.contains(&ids[anchor_index]);
            let (start, end) = if anchor_index <= index {
                (anchor_index, index)
            } else {
                (index, anchor_index)
            };
            for id in &ids[start..=end] {
                if select {
                    selected.insert(*id);
                } else {
                    selected.remove(id);
                }
            }
        }
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

    fn sorted(selected: &HashSet<u64>) -> Vec<u64> {
        let mut ids = selected.iter().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn toggle_flips_one_card_and_moves_the_anchor() {
        let ids = [10, 11, 12, 13];
        let mut selected = HashSet::new();
        let mut anchor = None;
        update_selection(&mut selected, &mut anchor, &ids, 1, CardAction::Toggle);
        assert_eq!(sorted(&selected), [11]);
        assert_eq!(anchor, Some(11));
        update_selection(&mut selected, &mut anchor, &ids, 1, CardAction::Toggle);
        assert!(selected.is_empty());
        assert_eq!(anchor, Some(11));
    }

    #[test]
    fn shift_click_selects_a_range_in_either_direction_and_keeps_other_ranges() {
        let ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut selected = HashSet::new();
        let mut anchor = None;
        update_selection(&mut selected, &mut anchor, &ids, 1, CardAction::Toggle);
        update_selection(&mut selected, &mut anchor, &ids, 3, CardAction::Range);
        assert_eq!(sorted(&selected), [1, 2, 3]);

        // A second, disjoint range to the right of the first.
        update_selection(&mut selected, &mut anchor, &ids, 8, CardAction::Toggle);
        update_selection(&mut selected, &mut anchor, &ids, 6, CardAction::Range);
        assert_eq!(sorted(&selected), [1, 2, 3, 6, 7, 8]);
        assert_eq!(anchor, Some(8), "anchor stays on the toggled card");
    }

    #[test]
    fn shift_click_from_a_deselected_anchor_deselects_the_range() {
        let ids = [0, 1, 2, 3, 4, 5];
        let mut selected = ids.iter().copied().collect::<HashSet<_>>();
        let mut anchor = None;
        update_selection(&mut selected, &mut anchor, &ids, 4, CardAction::Toggle);
        assert!(!selected.contains(&4));
        update_selection(&mut selected, &mut anchor, &ids, 2, CardAction::Range);
        assert_eq!(sorted(&selected), [0, 1, 5]);
    }

    #[test]
    fn shift_click_without_an_anchor_starts_a_selection() {
        let ids = [7, 8, 9];
        let mut selected = HashSet::new();
        let mut anchor = None;
        update_selection(&mut selected, &mut anchor, &ids, 2, CardAction::Range);
        assert_eq!(sorted(&selected), [9]);
        assert_eq!(anchor, Some(9));
        update_selection(&mut selected, &mut anchor, &ids, 9, CardAction::Toggle);
        assert_eq!(sorted(&selected), [9], "out-of-range index is ignored");
    }
}
