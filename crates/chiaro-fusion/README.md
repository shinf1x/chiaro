# Chiaro fusion library

Multi-camera alignment and high-resolution synthesis for Light L16 captures.
This crate powers Chiaro Gallery's fused export and the `chiaro-fuse`
command-line application.

## Pipeline

1. Decode each participating RAW module and optionally run the shared
   hot-pixel and corner-glow correction pipeline.
2. Resolve the factory camera model, project each module into a reference view,
   and refine the alignment with image correlation.
3. Apply module-specific colour and flat-field calibration, match overlapping
   modules photometrically, and blend them into a 16-bit PNG.

Every run also writes a `.fusion.json` report with alignment, coverage,
photometric, and timing diagnostics.

## Calibration

Captures embed part of the camera model and take priority when their calibration
is newer. Device `calibration.lri` and `zoom_calib_v0.lri` files fill important
gaps, including mirror-aiming data. Supply both whenever possible; alignment is
likely to be poor without them.

`hotpixel.rec` is optional at the fusion API level, but when enabled it must
belong to the same physical camera.

## Output modes

- **Native** renders approximately the reference sensor's 13 MP resolution.
- **Maximum** uses the finest participating module that covers the view, capped
  by a caller-provided megapixel limit.
- **Scale** specifies canvas pixels per reference pixel directly.

The output is cropped to the focal length framed by the photographer by
default. Full-reference rendering can be requested instead. Monochrome modules
contribute luminance unless explicitly excluded.

## Important limitations

- Alignment uses one refined homography per module and is intended for distant
  scenes. Near objects can ghost because their depth-dependent parallax is not
  modelled.
- Correlation needs useful scene texture, and the photometric matcher needs
  tonal range. Night-sky captures may retain visible module boundaries; use
  per-camera Hotpixel output and a stacker for astrophotography.
- Factory geometry alone is not sufficiently accurate for normal output;
  disabling correlation refinement is intended mainly for diagnostics.

## API and diagnostics

`pipeline::fuse` accepts an in-memory LRI, `FusionOptions`, an output path, and
a progress callback. The processing stages use plain data structures so
alignment and synthesis can also be inspected independently.

Set `FusionOptions::debug_dir` to write per-module alignment checkerboards.
Continuous scene edges across checkerboard boundaries are a quick visual check
of the resulting warp.

See [Chiaro Fuse](../../apps/fuse/README.md) for command-line usage.

## Tests

```bash
cargo test -p chiaro-fusion
```
