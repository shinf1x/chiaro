# Chiaro Stack

Chiaro Stack uses every temporal frame and physical module in a Light L16 night
capture. It first creates a motion-safe denoised mosaic for each module, then
uses device geometry and colour calibration to synthesise the final image.
Moving or inconsistently aligned temporal samples fall back to the selected
sharp reference frame instead of ghosting.

```bash
chiaro-stack capture.lri \
  --hotpixel-rec /path/to/hotpixel.rec \
  --calibration /path/to/calibration.lri \
  --calibration /path/to/zoom_calib_v0.lri \
  --output capture-night.png
```

Use `--camera A1 --diagnostics` to inspect only one physical camera and write
its reference and effective-frame-count images.

The row-indexed gyroscope packets provide a rolling-shutter-aware rotation
seed. Correlation refines that seed, and the reference module's measured motion
is projected into the remaining modules according to focal length. The same
temporal reference is used throughout the device.

```bash
chiaro-stack capture.lri --camera A1 --output A1-night.png --diagnostics
```

The factory hot-pixel file should come from the camera that made the capture.
`--diagnostics` also writes the selected reference image, an effective-frame
count map, and a JSON report with alignment quality. Use `--linear` to preserve
linear camera RGB for downstream processing. `--demosaic` accepts `simple`, `amaze`, `rcd`, `lmmse`, or `igv`. AMaZE is the default.

Temporal rejection is per pixel. Cross-module alignment currently uses one
refined homography per physical module, so strong depth-dependent parallax can
still reduce detail or cause softness in the affected region.
