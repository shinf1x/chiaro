mod app;
mod gallery;
mod parallel;
mod source;

use app::GalleryApp;

fn main() -> eframe::Result {
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../../../assets/logo_128.png"))
        .expect("embedded Chiaro logo is valid PNG");
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([720.0, 480.0])
            .with_icon(icon),
        ..Default::default()
    };

    eframe::run_native(
        "chiaro gallery",
        options,
        Box::new(|cc| Ok(Box::new(GalleryApp::new(cc)))),
    )
}
