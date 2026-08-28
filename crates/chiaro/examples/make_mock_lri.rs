//! Write a folder of synthetic Light L16 captures for UI development and tests.
//!
//! ```bash
//! cargo run -p chiaro --features mock --example make_mock_lri -- \
//!     /tmp/mock_lris [count] [edge] [cameras]
//! ```
//!
//! `edge` is the long RAW edge (default 4160 for real sensor dimensions; pass
//! 416 for quick small files). `cameras` is a comma-separated list such as
//! `A1,A2,B1,C6` (default: a typical five-module firing set). Each file is a
//! smooth night-sky-like gradient with a handful of bright "stars" and a
//! scattering of planted hot pixels, so hot-pixel export has something to fix.

use std::{env, error::Error, fs, path::PathBuf};

use chiaro::lri::SensorPattern;
use chiaro::mock::{MockCamera, MockCapture};

fn main() {
    if let Err(error) = run() {
        eprintln!("make_mock_lri: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: make_mock_lri <output-dir> [count] [edge] [cameras]")?,
    );
    let count: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(12);
    let edge: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(4160);
    let cameras = args
        .next()
        .unwrap_or_else(|| "A1,A2,B1,B4,C6".to_owned())
        .split(',')
        .map(|name| name.trim().to_ascii_uppercase())
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    // Keep the L16's 4:3 aspect and a width divisible by four for RAW10.
    let width = edge / 4 * 4;
    let height = (edge * 3 / 4) / 4 * 4;
    fs::create_dir_all(&output)?;

    for index in 0..count {
        let number = 4480 + index;
        let mut capture = MockCapture {
            captured_at: Some((2024, 8, 23, 1, 30 + (index % 30) as u32, 0)),
            ..MockCapture::default()
        };
        for (slot, name) in cameras.iter().enumerate() {
            let pattern = match name.as_str() {
                "A2" | "C6" => SensorPattern::Mono,
                "A1" | "A3" | "A4" | "A5" => SensorPattern::Bggr,
                "B1" | "B2" | "B3" | "B4" | "B5" => SensorPattern::Grbg,
                _ => SensorPattern::Rggb,
            };
            let seed = (index * 31 + slot * 7) as u64;
            let low = 40 + (seed % 20) as u16;
            let mut camera = MockCamera::gradient(name, width, height, pattern, low, low + 160);
            let mut defects = Vec::new();
            // Bright "stars": a few saturated 3x3 blobs.
            for star in 0..6u64 {
                let x = ((seed + star * 7919) * 2654435761 % width as u64) as usize;
                let y = ((seed + star * 104729) * 2246822519 % height as u64) as usize;
                for dy in 0..3usize {
                    for dx in 0..3usize {
                        defects.push((x + dx, y + dy, 1023));
                    }
                }
            }
            // Hot pixels: isolated bright singles scattered pseudo-randomly.
            for hot in 0..(width * height / 20_000) as u64 {
                let x = ((seed + hot * 40503) * 3266489917 % width as u64) as usize;
                let y = ((seed + hot * 7727) * 668265263 % height as u64) as usize;
                defects.push((x, y, 600 + (hot % 400) as u16));
            }
            camera = camera.with_defects(&defects);
            camera.sensor_temperature_c = Some(36 + (index % 12) as i32);
            capture.cameras.push(camera);
        }
        let bytes = capture.encode()?;
        let path = output.join(format!("L16_{number:05}.lri"));
        fs::write(&path, &bytes)?;
        println!(
            "{} ({} cameras, {}x{}, {:.1} MB)",
            path.display(),
            cameras.len(),
            width,
            height,
            bytes.len() as f64 / 1e6
        );
    }
    Ok(())
}
