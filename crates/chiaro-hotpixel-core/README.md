# Chiaro Hotpixel core library

Reusable, UI-independent correction for Light L16 RAW frames. This crate is
the processing engine shared by the `chiaro-hotpixel` command-line application
and Chiaro Gallery.

## Capabilities

- parse the camera's factory `hotpixel.rec` and select the matching module map;
- decode Light's packed RAW10 surfaces;
- repair factory-listed hot and dead pixels with CFA-aware interpolation;
- apply bundled condition-dependent hot-pixel and corner-glow models;
- apply optional camera-specific cleanup profiles;
- preserve Bayer mosaics or produce linear 16-bit RGB output; and
- process and encode one frame across multiple CPU cores.

The bundled sensor-family assets are compiled into the library. They contain no
camera-specific defect coordinates, so callers still need the `hotpixel.rec`
belonging to the camera being processed.

## API

`FramePipeline` defines the enabled stages and processing options.
`correct_lri` returns a `CorrectedFrame` with Q6 linear samples and per-stage
statistics; `write_png` writes the same output used by the CLI.

```rust,no_run
use chiaro::lri::parse_raw_layout;
use chiaro_hotpixel_core::hotpixel::HotpixelRec;
use chiaro_hotpixel_core::pipeline::{FramePipeline, OutputMode};
use chiaro_hotpixel_core::scan::mmap_file;
use chiaro_hotpixel_core::thermal::ThermalProfile;
use chiaro_hotpixel_core::universal_hotpixel::UniversalHotpixelProfile;

fn main() -> anyhow::Result<()> {
    let factory = HotpixelRec::open("/camera/hotpixel.rec")?;
    let universal = UniversalHotpixelProfile::bundled()?;
    let glow = ThermalProfile::bundled()?;
    let pipeline = FramePipeline {
        universal_hotpixel: Some(&universal),
        thermal: Some(&glow),
        ..FramePipeline::default()
    };

    let lri = mmap_file("/captures/L16_04480.lri".as_ref())?;
    let layout = parse_raw_layout(&lri, &Default::default())?;
    for camera in &layout.cameras {
        let map = factory.load_rotated_map(camera.id, camera.width, camera.height)?;
        let frame = pipeline.correct_lri(&lri, camera, &map)?;
        let output = format!("/frames/{}.png", camera.name);
        frame.write_png(output.as_ref(), OutputMode::Rgb)?;
    }
    Ok(())
}
```

The crate does not print to the terminal. It accesses the filesystem only when
a caller requests profile loading, PNG output, or cleanup-profile training.

See [Chiaro Hotpixel](../../apps/hotpixel/README.md) for correction behavior,
calibration requirements, and command-line usage.

## Tests

```bash
cargo test -p chiaro-hotpixel-core
```
