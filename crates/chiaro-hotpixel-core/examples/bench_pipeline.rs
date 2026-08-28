//! Per-stage timing of the hot-pixel pipeline on a real capture.
//!
//! ```bash
//! cargo run -p chiaro-hotpixel-core --release --example bench_pipeline -- \
//!     capture.lri hotpixel.rec [camera] [repeats]
//! ```
//!
//! Prints the median wall time of every stage so optimisation work can be
//! targeted and verified instead of guessed.

use std::{collections::HashMap, env, path::Path, time::Instant};

use chiaro::lri::{SensorPattern, parse_raw_layout};
use chiaro_hotpixel_core::{
    correct::{
        correct_hot_pixels_threaded, correct_hot_pixels_with_forced_map, demosaic_bilinear,
        demosaic_bilinear_threaded,
    },
    hotpixel::HotpixelRec,
    pipeline::{FramePipeline, OutputMode, extract_raw_plane, extract_raw_plane_threaded},
    png16::{write_gray16_native_atomic, write_rgb16_native_atomic},
    scan::mmap_file,
    thermal::ThermalProfile,
    universal_hotpixel::UniversalHotpixelProfile,
};

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn timed<T>(label: &str, repeats: usize, mut run: impl FnMut() -> T) -> T {
    let mut times = Vec::with_capacity(repeats);
    let mut result = None;
    for _ in 0..repeats {
        let started = Instant::now();
        result = Some(run());
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    println!("{label:<34} {:>9.1} ms", median(&mut times));
    result.expect("at least one repeat")
}

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let lri = args.next().expect("capture.lri");
    let rec = args.next().expect("hotpixel.rec");
    let camera_name = args.next().unwrap_or_else(|| "A1".to_owned());
    let repeats: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(3);

    let data = mmap_file(Path::new(&lri))?;
    let layout = parse_raw_layout(&data, &HashMap::new())?;
    let camera = layout
        .cameras
        .iter()
        .find(|camera| camera.name.eq_ignore_ascii_case(&camera_name))
        .expect("camera in capture")
        .clone();
    let rec = HotpixelRec::open(&rec)?;
    let universal = UniversalHotpixelProfile::bundled()?;
    let thermal = ThermalProfile::bundled()?;
    println!(
        "{} {}x{} {:?} temp={:?} exposure={}ns (median of {repeats})",
        camera.name,
        camera.width,
        camera.height,
        camera.pattern,
        camera.sensor_temperature_c,
        camera.exposure_ns
    );

    let map = timed("load_rotated_map", repeats, || {
        rec.load_rotated_map(camera.id, camera.width, camera.height)
            .unwrap()
    });
    timed("unpack RAW10 (1 thread)", repeats, || {
        extract_raw_plane(&data, &camera).unwrap()
    });
    let raw = timed("unpack RAW10 (all threads)", repeats, || {
        extract_raw_plane_threaded(&data, &camera, 0).unwrap()
    });
    let (forced, _) = timed("universal active_map", repeats, || {
        universal.active_map(&camera, &map, 4.0)
    });
    let config = Default::default();
    let mut corrected = raw.clone();
    timed("correct_hot_pixels (1 thread)", repeats, || {
        corrected.copy_from_slice(&raw);
        correct_hot_pixels_with_forced_map(
            &mut corrected,
            camera.width,
            camera.height,
            camera.pattern,
            &map,
            Some(&forced),
            &config,
        )
        .unwrap()
    });
    let stats = timed("correct_hot_pixels (all threads)", repeats, || {
        corrected.copy_from_slice(&raw);
        correct_hot_pixels_threaded(
            &mut corrected,
            camera.width,
            camera.height,
            camera.pattern,
            &map,
            Some(&forced),
            &config,
            0,
        )
        .unwrap()
    });
    println!(
        "{:<34} candidates={} corrected={}",
        "", stats.candidates, stats.corrected
    );
    let q6 = corrected.iter().map(|s| s << 6).collect::<Vec<u16>>();
    let mut glow = q6.clone();
    timed("thermal glow (1 thread)", repeats, || {
        glow.copy_from_slice(&q6);
        thermal.correct_raw_plane_q6(&camera, &mut glow, 1).unwrap();
    });
    timed("thermal glow (all threads)", repeats, || {
        glow.copy_from_slice(&q6);
        thermal.correct_raw_plane_q6(&camera, &mut glow, 0).unwrap();
    });
    let rgb = if camera.pattern != SensorPattern::Mono {
        timed("demosaic (1 thread)", repeats, || {
            demosaic_bilinear(&glow, camera.width, camera.height, camera.pattern).unwrap()
        });
        Some(timed("demosaic (all threads)", repeats, || {
            demosaic_bilinear_threaded(&glow, camera.width, camera.height, camera.pattern, 0)
                .unwrap()
        }))
    } else {
        None
    };
    let out = env::temp_dir().join("chiaro-bench.png");
    timed("png encode + write", repeats, || match &rgb {
        Some(rgb) => write_rgb16_native_atomic(&out, camera.width, camera.height, rgb).unwrap(),
        None => write_gray16_native_atomic(&out, camera.width, camera.height, &glow).unwrap(),
    });
    println!(
        "{:<34} {} bytes",
        "png size",
        std::fs::metadata(&out)?.len()
    );

    let pipeline = FramePipeline {
        universal_hotpixel: Some(&universal),
        thermal: Some(&thermal),
        ..FramePipeline::default()
    };
    let frame = timed("FramePipeline::correct_lri (total)", repeats, || {
        pipeline.correct_lri(&data, &camera, &map).unwrap()
    });
    timed("CorrectedFrame::write_png (total)", repeats, || {
        frame.write_png(&out, OutputMode::Rgb).unwrap()
    });
    let _ = std::fs::remove_file(out);
    Ok(())
}
