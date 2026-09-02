# Chiaro Stack

Chiaro Stack uses every temporal frame and physical module in a Light L16 night
capture. It first creates a motion-safe denoised mosaic for each module, then
uses device geometry and colour calibration to synthesise the final image.
Moving or inconsistently aligned temporal samples fall back to the selected
sharp reference frame instead of ghosting.

Cross-module alignment first refines a global homography, then builds a dense
multi-camera inverse-depth cost field through the calibrated camera models.
Semi-global matching proposes coarse hypotheses, but every accepted finite
node is remeasured on a finer grid with edge-aware photometric support; gaps
are not spatially completed. Each module continuously refines the shared depth
and a bounded local residual where its own evidence supports the surface;
likely occlusions and contradictory edge samples are suppressed, while distant
or ambiguous regions retain the global warp. Thin reference structures use
centre-surround consistency and a combined-weight floor so bright, displaced
background samples cannot erase them. Use `--no-depth` for an A/B
diagnostic comparison, or adjust `--depth-near` and `--depth-far` when the scene
lies outside the default 0.5 metre–10 kilometre interval.

```bash
chiaro-stack capture.lri \
  --hotpixel-rec /path/to/hotpixel.rec \
  --calibration /path/to/calibration.lri \
  --calibration /path/to/zoom_calib_v0.lri \
  --output capture-night.png
```

For an all-module run, `--diagnostics` writes the reconstructed control grid as
`*.depth-inverse.png`, `*.depth-visualization.png`, and
`*.depth-provenance.png`. Inverse depth is quantitative 16-bit grayscale
(nearer is brighter; zero is global/infinite or unsupported); the visualization
uses a log-scaled far-blue to near-red palette. The provenance map uses green
for direct measurements, amber for a regularized finite node if one is
explicitly supplied, blue for deliberate global/infinite fallback, and black
outside calibrated support. Default final finite nodes are directly remeasured
and therefore green.
With `--camera A1`, diagnostics instead write that module's reference and
effective-frame-count images.

The row-indexed gyroscope packets provide a rolling-shutter-aware rotation
seed. Correlation refines that seed, and the reference module's measured motion
is projected into the remaining modules according to focal length. The same
temporal reference is used throughout the device.

Temporal rejection is normalized by the factory signal-dependent noise model
for each frame's recorded gain. Models are interpolated between calibration
points; monochrome modules use their panchromatic characterization. The
capture's embedded table is preferred and a device-matched `calibration.lri`
can fill missing entries, including in single-camera mode.

```bash
chiaro-stack capture.lri --camera A1 --output A1-night.png --diagnostics
```

The factory hot-pixel file should come from the camera that made the capture.
`--diagnostics` also writes the selected reference image, an effective-frame
count map, and a JSON report with alignment quality. Use `--linear` to preserve
linear camera RGB for downstream processing. `--demosaic` accepts `simple`,
`amaze`, `rcd`, `lmmse`, or `igv`. LMMSE is the night-stack default; IGV is the
alternative for noise- and moire-prone detail. AMaZE remains the general-photo
default elsewhere, while RCD is intended for individual night-sky exposures.
`--highlight-recovery` accepts `none`, `local-bayer`, `multiscale-bayer`, or
`multi-camera`; all-module mode defaults to confidence-gated multi-camera RAW
recovery before demosaic.

Temporal rejection is per pixel. Dense depth reconstruction is classical and
confidence-gated; it does not yet estimate non-rigid subject motion between
temporal exposures.
