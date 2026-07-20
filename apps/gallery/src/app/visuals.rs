use super::*;

pub(super) fn draw_texture(
    ui: &egui::Ui,
    rect: egui::Rect,
    texture: &egui::TextureHandle,
    dimensions: [usize; 2],
) {
    let source = Vec2::new(dimensions[0] as f32, dimensions[1] as f32);
    let scale = (rect.width() / source.x).min(rect.height() / source.y);
    let draw_rect = egui::Rect::from_center_size(rect.center(), source * scale);
    ui.painter().image(
        texture.id(),
        draw_rect,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        Color32::WHITE,
    );
}

pub(super) fn placeholder(ui: &egui::Ui, rect: egui::Rect, text: &str, color: Color32) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        FontId::proportional(14.0),
        color,
    );
}

pub(super) fn elide_text(
    ui: &egui::Ui,
    text: &str,
    available_width: f32,
    font: &FontId,
    color: Color32,
) -> String {
    let fits = |candidate: &str| {
        ui.painter()
            .layout_no_wrap(candidate.to_owned(), font.clone(), color)
            .size()
            .x
            <= available_width
    };
    if fits(text) {
        return text.to_owned();
    }
    let characters = text.chars().collect::<Vec<_>>();
    let mut lower = 0usize;
    let mut upper = characters.len();
    while lower < upper {
        let middle = (lower + upper).div_ceil(2);
        let candidate = characters[..middle]
            .iter()
            .chain(std::iter::once(&'…'))
            .collect::<String>();
        if fits(&candidate) {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    characters[..lower]
        .iter()
        .chain(std::iter::once(&'…'))
        .collect()
}

pub(super) fn paint_metadata(ui: &egui::Ui, rect: egui::Rect, metadata: &CaptureMetadata) {
    let color = Color32::from_gray(158);
    let font = FontId::proportional(11.5);
    let iso = metadata
        .iso
        .map(|value| format!("ISO {value}"))
        .unwrap_or_else(|| "ISO -".to_owned());
    let shutter = metadata
        .shutter_ns
        .map(format_shutter)
        .unwrap_or_else(|| "-".to_owned());
    let focal = metadata
        .focal_length_mm
        .map(|value| format!("{value} mm"))
        .unwrap_or_else(|| "- mm".to_owned());
    let x = rect.left();
    let top = rect.top() + 23.0;
    let exposure = format!("{iso}   {shutter}");
    ui.painter().text(
        egui::pos2(x, top),
        egui::Align2::LEFT_TOP,
        elide_text(ui, &exposure, rect.width(), &font, color),
        font.clone(),
        color,
    );
    ui.painter().text(
        egui::pos2(x, top + 16.0),
        egui::Align2::LEFT_TOP,
        focal,
        font.clone(),
        color,
    );

    if let Some(captured_at) = &metadata.captured_at {
        let captured_at = format_capture_time(captured_at);
        let date_color = Color32::from_gray(138);
        ui.painter().text(
            egui::pos2(x, top + 34.0),
            egui::Align2::LEFT_TOP,
            elide_text(ui, &captured_at, rect.width(), &font, date_color),
            font,
            date_color,
        );
    }
}

pub(super) fn capture_icon_width(metadata: &CaptureMetadata) -> f32 {
    let count = usize::from(metadata.night_mode) + usize::from(metadata.tripod == Some(true));
    count as f32 * 19.0
}

pub(super) fn paint_capture_icons(
    ui: &egui::Ui,
    rect: egui::Rect,
    metadata: &CaptureMetadata,
    background: Color32,
) {
    let color = Color32::from_gray(210);
    let mut center = egui::pos2(rect.right() - 7.0, rect.top() + 7.0);
    if metadata.tripod == Some(true) {
        paint_tripod_icon(ui, center, color);
        center.x -= 19.0;
    }
    if metadata.night_mode {
        paint_moon_icon(ui, center, color, background);
    }
}

pub(super) fn show_capture_icons(
    ui: &mut egui::Ui,
    metadata: &CaptureMetadata,
    background: Color32,
) {
    if metadata.night_mode {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::hover());
        paint_moon_icon(ui, rect.center(), Color32::from_gray(210), background);
        response.on_hover_text("Night mode");
    }
    if metadata.tripod == Some(true) {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::hover());
        paint_tripod_icon(ui, rect.center(), Color32::from_gray(210));
        response.on_hover_text("Tripod detected");
    }
}

pub(super) fn paint_moon_icon(
    ui: &egui::Ui,
    center: egui::Pos2,
    color: Color32,
    background: Color32,
) {
    ui.painter().circle_filled(center, 5.5, color);
    ui.painter()
        .circle_filled(center + egui::vec2(3.0, -1.2), 5.4, background);
}

pub(super) fn paint_tripod_icon(ui: &egui::Ui, center: egui::Pos2, color: Color32) {
    let stroke = Stroke::new(1.25_f32, color);
    let body = egui::Rect::from_center_size(center + egui::vec2(0.0, -2.5), egui::vec2(10.0, 6.0));
    ui.painter()
        .rect_stroke(body, CornerRadius::same(1), stroke, StrokeKind::Inside);
    ui.painter()
        .circle_filled(center + egui::vec2(-2.2, -2.5), 0.9, color);
    ui.painter()
        .circle_filled(center + egui::vec2(2.2, -2.5), 0.9, color);
    ui.painter().line_segment(
        [center + egui::vec2(0.0, 0.5), center + egui::vec2(0.0, 7.0)],
        stroke,
    );
    ui.painter().line_segment(
        [
            center + egui::vec2(0.0, 3.0),
            center + egui::vec2(-5.0, 7.0),
        ],
        stroke,
    );
    ui.painter().line_segment(
        [center + egui::vec2(0.0, 3.0), center + egui::vec2(5.0, 7.0)],
        stroke,
    );
}

pub(super) fn format_shutter(nanoseconds: u64) -> String {
    let seconds = nanoseconds as f64 / 1_000_000_000.0;
    if seconds >= 1.0 {
        format!("{seconds:.1} s")
    } else if seconds > 0.0 {
        format!("1/{:.0} s", 1.0 / seconds)
    } else {
        "-".to_owned()
    }
}

pub(super) fn format_capture_time(value: &CaptureDateTime) -> String {
    let timezone = value
        .timezone_offset_minutes
        .map_or_else(String::new, |offset| {
            let sign = if offset < 0 { '-' } else { '+' };
            let absolute = offset.unsigned_abs();
            format!(" {sign}{:02}:{:02}", absolute / 60, absolute % 60)
        });
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}{timezone}",
        value.year, value.month, value.day, value.hour, value.minute, value.second
    )
}

pub(super) fn metadata_summary(metadata: &CaptureMetadata) -> String {
    let mut parts = Vec::new();
    if let Some(iso) = metadata.iso {
        parts.push(format!("ISO {iso}"));
    }
    if let Some(shutter) = metadata.shutter_ns {
        parts.push(format_shutter(shutter));
    }
    if let Some(focal) = metadata.focal_length_mm {
        parts.push(format!("{focal} mm"));
    }
    if let Some(captured_at) = &metadata.captured_at {
        parts.push(format_capture_time(captured_at));
    }
    parts.join("  |  ")
}

pub(super) fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

pub(super) fn folder_tab_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| path.display().to_string())
}
