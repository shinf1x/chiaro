//! Inspect an LRI and decode one calibrated preview entirely in memory.

use std::{env, error::Error, path::Path};

use chiaro::lri::{decode_camera_preview, decode_reference_preview, inspect_capture};

fn main() {
    if let Err(error) = run() {
        eprintln!("inspect_lri: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: inspect_lri <capture.lri> [camera] [max-edge]")?;
    let camera = args.next();
    let max_edge = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(320);
    if args.next().is_some() {
        return Err("usage: inspect_lri <capture.lri> [camera] [max-edge]".into());
    }

    let summary = inspect_capture(Path::new(&path))?;
    println!(
        "capture: reference {}, {} usable cameras, metadata {:?}",
        summary.reference_camera,
        summary.cameras.len(),
        summary.metadata,
    );
    let result = match camera.as_deref() {
        Some(camera) => decode_camera_preview(Path::new(&path), camera, max_edge),
        None => decode_reference_preview(Path::new(&path), max_edge),
    };
    let preview = result?;
    println!(
        "{}: camera {}, preview {}x{} ({} RGB bytes, calibrated color: {})",
        path,
        preview.camera,
        preview.size[0],
        preview.size[1],
        preview.rgb.len(),
        preview.color_calibrated,
    );
    Ok(())
}
