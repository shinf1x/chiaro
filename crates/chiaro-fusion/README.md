# Chiaro fusion library

Multi-camera alignment and high-resolution synthesis for Light L16 captures.
This crate powers Chiaro Gallery's fused export and the `chiaro-fuse`
command-line application.

## Pipeline

1. Decode each participating RAW module and optionally run the shared
   hot-pixel and corner-glow correction pipeline. Detect near-clipped CFA
   measurements, reconstruct small edges locally, and use a clipping-aware RGB
   pyramid for larger low-confidence regions. One bit of fractional RAW
   precision is reserved as radiometric headroom above sensor white.
2. Resolve the factory camera model, project each module into a reference view,
   refine the global alignment with image correlation, build a dense calibrated
   inverse-depth cost field, and regularise coarse search hypotheses with
   edge-aware eight-direction semi-global matching. A finer 4-pixel grid then
   independently remeasures every accepted node with small or adaptive
   bilateral support; coarse values and holes are never copied into the final
   map. Each camera subsequently refines the shared depth continuously and
   applies a bounded residual correction.
3. When requested, blend a low-confidence spatial highlight estimate toward a
   donor field only when at least two accepted, aligned modules provide
   consistent unclipped RAW radiance. Donors are regularised within each CFA
   phase and feathered at coverage boundaries; per-channel overlap ratios
   account for exposure/transmission differences. Treat the factory 17x13
   four-phase crosstalk mesh as a prior and, in the default adaptive mode, fit
   five small residual modes from smooth aligned overlap. The fit is performed
   in the capture's white-balance domain, tested on held-out observations, and
   falls back independently to the factory mesh for any module without enough
   evidence or a measurable validation improvement. Then reconstruct Bayer
   colour with the selected demosaicing method and apply
   module-specific colour and flat-field calibration, match overlapping
   modules photometrically, and blend them into a 16-bit PNG. AMaZE is the
   default; Simple, RCD, LMMSE, and IGV are also available. Warp
   discontinuities are not interpolated across visible scene edges, and a
   reference-guided robust weight rejects contradictory edge samples without
   discarding agreeing high-resolution samples. Luminance and chroma are
   weighted independently, preventing a defocused colour fringe with plausible
   total brightness from entering the result. Fine structure remains
   reference-anchored unless another module reproduces its direction more
   sharply; that agreeing zoom module may then own the high-frequency detail.
   Strong near-side residual parallax relative to a magnified module's
   calibrated focus plane suppresses that module locally. Centre-surround
   contrast preserves thin branches and wires even when their centres have
   little directional gradient. For display-ready output, a smooth sensor-white
   shoulder neutralises false colour from unequally clipped raw channels without
   introducing a hard highlight boundary.

Every run also writes a `.fusion.json` report with alignment, RAW highlight
confidence/counts, per-module crosstalk fit/validation measurements, coverage,
photometric, and timing diagnostics.

## Calibration

Captures embed part of the camera model and take priority when their calibration
is newer. Device `calibration.lri` and `zoom_calib_v0.lri` files fill important
gaps, including mirror-aiming data. Supply both whenever possible; alignment is
likely to be poor without them. An overlay is merged only when its physical
device id matches the capture.

Focus-dependent intrinsics and object-space focus distance are interpolated in
lens Hall space and continued linearly just beyond the factory samples,
matching the validated reconstruction model. Capture autofocus success,
disparity/contrast estimates, ROI, and actuator timeouts are retained in the
report. Image evidence remains authoritative: autofocus success describes the
selected focus plane, not whether every scene depth is sharp. The CLI's
diagnostic `--intrinsics clamp` mode freezes out-of-range captures at the
nearest sample instead.

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
- Spatial highlight reconstruction cannot recover true colour or texture where
  every local raw channel is saturated. Multi-camera mode can recover such
  samples only inside reliable overlap where at least two unclipped modules
  agree. The final smooth shoulder remains as a neutral fallback.
- Capture-adaptive crosstalk estimates only a strongly regularised residual on
  the supplied factory mesh. The reference module remains the colour anchor,
  because one scene cannot identify its absolute error. Capture gain, exposure,
  and AWB are recorded for analysis, but Chiaro does not yet select among
  multiple factory families because no reliable family mapping is available.
  Spatial leakage kernels are intentionally deferred until residual diagnostics
  demonstrate leakage beyond the existing phase matrix.

## API and diagnostics

`pipeline::fuse` accepts an in-memory LRI, `FusionOptions`, an output path, and
a progress callback. The processing stages use plain data structures so
alignment and synthesis can also be inspected independently.

With a debug directory, `<camera>_highlight-uncertainty.png` marks recovered
RAW samples by inverse confidence in addition to the depth, alignment, and
source-ownership diagnostics.

Set `FusionOptions::debug_dir` to write per-module alignment checkerboards.
Continuous scene edges across checkerboard boundaries are a quick visual check
of the resulting warp. The same directory receives `source-luminance-ownership.png`
and `source-color-ownership.png`; their camera-to-colour legend and exact owner
fractions are recorded under `synthesis.source_contributions` in the JSON
report. It also receives `depth-inverse.png`, `depth-visualization.png`, and
`depth-provenance.png`. The first is quantitative 16-bit inverse depth; the
second is a log-scaled far-blue to near-red rendering.
The provenance image marks directly remeasured nodes green, a regularized
finite node amber if one is explicitly supplied, global/infinite fallback blue,
and unsupported nodes black. With default settings, finite final nodes are
green: SGM proposes where to search but cannot create final depth by itself.

See [Chiaro Fuse](../../apps/fuse/README.md) for command-line usage.

## Tests

```bash
cargo test -p chiaro-fusion
```
