use super::*;

const LOGO_PNG: &[u8] = include_bytes!("../../../../assets/logo_128.png");
const WORDMARK_PNG: &[u8] = include_bytes!("../../../../assets/chiaro_128.png");
const DESIGN_HEIGHT: f32 = 42.0;
const MAX_PHYSICAL_HEIGHT: f32 = 36.0;
const BRAND_TEXTURE_OPTIONS: egui::TextureOptions =
    egui::TextureOptions::LINEAR.with_mipmap_mode(Some(egui::TextureFilter::Linear));

pub(super) struct BrandTextures {
    logo: egui::TextureHandle,
    wordmark: egui::TextureHandle,
}

impl BrandTextures {
    pub(super) fn new(ctx: &egui::Context) -> Self {
        Self {
            logo: load_monochrome_texture(ctx, "chiaro-logo", LOGO_PNG, true),
            wordmark: load_monochrome_texture(ctx, "chiaro-wordmark", WORDMARK_PNG, false),
        }
    }

    pub(super) fn show(&self, ui: &mut egui::Ui) {
        const WORDMARK_ASPECT: f32 = 468.0 / 128.0;

        let brand_height = Self::height(ui.ctx());
        let scale = brand_height / DESIGN_HEIGHT;
        let top_row_height = 26.0 * scale;
        let version_row_height = brand_height - top_row_height;
        let text_width = 158.0 * scale;
        let wordmark_height = 20.0 * scale;
        let logo_size = self.logo.size();
        let logo_width = brand_height * logo_size[0] as f32 / logo_size[1] as f32;
        let brand_width = logo_width + ui.spacing().item_spacing.x + text_width;

        ui.allocate_ui_with_layout(
            Vec2::new(brand_width, brand_height),
            Layout::left_to_right(Align::Center),
            |ui| {
                ui.add(
                    egui::Image::new(&self.logo)
                        .fit_to_exact_size(Vec2::new(logo_width, brand_height))
                        .alt_text("Chiaro logo"),
                );
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    ui.allocate_ui_with_layout(
                        Vec2::new(text_width, top_row_height),
                        Layout::left_to_right(Align::Min),
                        |ui| {
                            ui.spacing_mut().item_spacing.x = 7.0 * scale;
                            ui.add(
                                egui::Image::new(&self.wordmark)
                                    .fit_to_exact_size(Vec2::new(
                                        wordmark_height * WORDMARK_ASPECT,
                                        wordmark_height,
                                    ))
                                    .alt_text("chiaro"),
                            );
                            ui.label(RichText::new("gallery").size(20.0 * scale));
                        },
                    );
                    ui.allocate_ui_with_layout(
                        Vec2::new(text_width, version_row_height),
                        Layout::left_to_right(Align::Max),
                        |ui| {
                            ui.label(
                                RichText::new(format!("version {}", env!("CARGO_PKG_VERSION")))
                                    .size(13.0 * scale)
                                    .color(Color32::from_gray(165)),
                            );
                        },
                    );
                });
            },
        );
    }

    pub(super) fn height(ctx: &egui::Context) -> f32 {
        capped_brand_height(ctx.pixels_per_point())
    }

    pub(super) fn toolbar_margin(ctx: &egui::Context) -> Margin {
        let pixels_per_point = ctx.pixels_per_point().max(1.0);
        let base_inset = (8.0 / pixels_per_point).max(1.0);
        let tab_height = ctx.style().spacing.interact_size.y.max(20.0) + 2.0;
        let centering_space = ((tab_height - Self::height(ctx)) / 2.0).max(0.0);
        let horizontal = base_inset.max(centering_space + 1.0).round();
        let vertical = (horizontal - centering_space).max(1.0).round();
        Margin {
            left: horizontal as i8,
            right: horizontal as i8,
            top: vertical as i8,
            bottom: vertical as i8,
        }
    }
}

fn capped_brand_height(pixels_per_point: f32) -> f32 {
    DESIGN_HEIGHT.min(MAX_PHYSICAL_HEIGHT / pixels_per_point.max(1.0))
}

fn load_monochrome_texture(
    ctx: &egui::Context,
    name: &'static str,
    png: &[u8],
    trim: bool,
) -> egui::TextureHandle {
    let image = monochrome_image(png);
    let image = if trim { trim_transparent(image) } else { image };
    ctx.load_texture(name, image, BRAND_TEXTURE_OPTIONS)
}

fn trim_transparent(image: egui::ColorImage) -> egui::ColorImage {
    let width = image.size[0];
    let height = image.size[1];
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;

    for (index, pixel) in image.pixels.iter().enumerate() {
        if pixel.a() == 0 {
            continue;
        }
        let x = index % width;
        let y = index / width;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    if min_x > max_x || min_y > max_y {
        return image;
    }

    let trimmed_width = max_x - min_x + 1;
    let trimmed_height = max_y - min_y + 1;
    let mut pixels = Vec::with_capacity(trimmed_width * trimmed_height);
    for y in min_y..=max_y {
        let row = y * width;
        pixels.extend_from_slice(&image.pixels[row + min_x..=row + max_x]);
    }
    egui::ColorImage::new([trimmed_width, trimmed_height], pixels)
}

fn monochrome_image(png: &[u8]) -> egui::ColorImage {
    let icon = eframe::icon_data::from_png_bytes(png).expect("embedded Chiaro asset is valid PNG");
    let mut rgba = icon.rgba;
    for pixel in rgba.chunks_exact_mut(4) {
        let luminance = (u16::from(pixel[0]) + u16::from(pixel[1]) + u16::from(pixel[2])) / 3;
        let ink = 255 - luminance;
        pixel[3] = ((u16::from(pixel[3]) * ink) / 255) as u8;
        pixel[0] = 238;
        pixel[1] = 238;
        pixel[2] = 238;
    }
    egui::ColorImage::from_rgba_unmultiplied([icon.width as usize, icon.height as usize], &rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_brand_assets_decode_at_expected_aspects() {
        let logo = eframe::icon_data::from_png_bytes(LOGO_PNG).unwrap();
        let wordmark = eframe::icon_data::from_png_bytes(WORDMARK_PNG).unwrap();

        assert_eq!((logo.width, logo.height), (128, 128));
        assert_eq!((wordmark.width, wordmark.height), (468, 128));
    }

    #[test]
    fn logo_fill_becomes_transparent_but_ink_remains_visible() {
        let logo = monochrome_image(LOGO_PNG);
        let pixel = |x, y| logo.pixels[y * logo.size[0] + x];

        assert_eq!(pixel(48, 64).a(), 0);
        assert!(pixel(28, 64).a() > 150);
        assert!(pixel(64, 64).a() > 200);
    }

    #[test]
    fn brand_height_is_capped_in_physical_pixels() {
        assert_eq!(capped_brand_height(1.0), 36.0);
        assert_eq!(capped_brand_height(2.0), 18.0);
        assert_eq!(capped_brand_height(3.0), 12.0);
    }

    #[test]
    fn transparent_logo_padding_is_trimmed_symmetrically_by_layout() {
        let logo = trim_transparent(monochrome_image(LOGO_PNG));
        assert_eq!(logo.size, [126, 122]);
    }
}
