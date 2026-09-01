# Chiaro fusion library

Multi-camera alignment and high-resolution synthesis for Light L16 captures.
This crate powers Chiaro Gallery's fused export and the `chiaro-fuse`
command-line application.

## Pipeline

1. Decode each participating RAW module and optionally run the shared
   hot-pixel and corner-glow correction pipeline.
2. Resolve the factory camera model, project each module into a reference view,
   refine the global alignment with image correlation, build a dense calibrated
   inverse-depth cost field, and regularise coarse search hypotheses with
   edge-aware eight-direction semi-global matching. A finer 4-pixel grid then
   independently remeasures every accepted node with small or adaptive
   bilateral support; coarse values and holes are never copied into the final
   map. Each camera subsequently refines the shared depth continuously and
   applies a bounded residual correction.
3. Reconstruct Bayer colour with the selected demosaicing method, apply
   module-specific colour and flat-field calibration, match overlapping
   modules photometrically, and blend them into a 16-bit PNG. AMaZE is the
   default; Simple, RCD, LMMSE, and IGV are also available. Warp
   discontinuities are not interpolated across visible scene edges, and a
   reference-guided robust weight rejects contradictory edge samples without
   discarding agreeing high-resolution samples. Fine structure remains
   reference-anchored unless another module reproduces its direction at least
   as sharply, so mildly defocused or sub-pixel-misaligned lenses cannot soften
   a well-resolved reference edge. Centre-surround contrast and an aggregate
   reference-weight floor preserve thin branches and wires even when their
   centres have little directional gradient. For display-ready output, a smooth
   sensor-white shoulder neutralises false colour from unequally clipped raw
   channels without introducing a hard highlight boundary.

Every run also writes a `.fusion.json` report with alignment, coverage,
photometric, and timing diagnostics.

## Calibration

Captures embed part of the camera model and take priority when their calibration
is newer. Device `calibration.lri` and `zoom_calib_v0.lri` files fill important
gaps, including mirror-aiming data. Supply both whenever possible; alignment is
likely to be poor without them. An overlay is merged only when its physical
device id matches the capture.

Focus-dependent intrinsics are interpolated in lens Hall space and continued
linearly just beyond the factory samples, matching the validated reconstruction
model. The CLI's diagnostic `--intrinsics clamp` mode freezes out-of-range
captures at the nearest sample instead.

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

- Alignment uses a global homography followed by classical dense multi-view
  reconstruction. Textureless or contradictory areas remain explicit
  global/far fallback rather than receiving spatially completed depth. Small
  disconnected finite-depth islands are rejected as chance correlations; the
  filter never grows a measured surface into an unsupported region. A finite
  label must also improve measurably on the fitted global warp, preventing a
  shallow distant-scene cost curve from being reported as physical depth.
  The default finite search spans 0.5 m to 10 km so distant landscape detail
  is not collapsed onto a 100 m boundary.
  Per-camera consistency either applies a directly supported finite surface,
  retains the global warp, or suppresses an occluded view. Robust
  synthesis prevents most contradictory samples from producing double edges,
  but it cannot recover detail that moved differently in every exposure.
- Correlation needs useful scene texture, and the photometric matcher needs
  tonal range. Night-sky captures may retain visible module boundaries; use
  per-camera Hotpixel output and a stacker for astrophotography.
- Factory geometry alone is not sufficiently accurate for normal output;
  disabling correlation refinement is intended mainly for diagnostics.
- Highlight reconstruction can remove false clipping colour, but it cannot
  recover true colour or texture where every raw channel is saturated. Disable
  it when preserving the channel response for a dedicated raw processor.

## API and diagnostics

`pipeline::fuse` accepts an in-memory LRI, `FusionOptions`, an output path, and
a progress callback. The processing stages use plain data structures so
alignment and synthesis can also be inspected independently.

Set `FusionOptions::debug_dir` to write per-module alignment checkerboards.
Continuous scene edges across checkerboard boundaries are a quick visual check
of the resulting warp. The same directory receives `depth-inverse.png`,
`depth-visualization.png`, and `depth-provenance.png`. The first is quantitative
16-bit inverse depth; the second is a log-scaled far-blue to near-red rendering.
The provenance image marks directly remeasured nodes green, a regularized
finite node amber if one is explicitly supplied, global/infinite fallback blue,
and unsupported nodes black. With default settings, finite final nodes are
green: SGM proposes where to search but cannot create final depth by itself.

See [Chiaro Fuse](../../apps/fuse/README.md) for command-line usage.

## Tests

```bash
cargo test -p chiaro-fusion
```
