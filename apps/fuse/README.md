# Chiaro Fuse

Chiaro Fuse aligns the camera modules of one Light L16 capture and synthesises
a high-resolution 16-bit PNG. It is the command-line front end for
[`chiaro-fusion`](../../crates/chiaro-fusion/README.md).

## Installation

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-fuse
```

## Usage

```bash
chiaro-fuse capture.lri -o fused.png \
  --hotpixel-rec hotpixel.rec \
  --cleanup-profile camera.chiaro-cleanup \
  --calibration calibration.lri \
  --calibration zoom_calib_v0.lri
```

The command writes `fused.png` and a neighbouring `fused.fusion.json`
diagnostic report. Its `color` section records the available factory
illuminants, selected weights, confidence, and whether a held-out-validated
Macbeth refit was used. `array_color` records the CCT prior, selected simplex
blend, best and runner-up scores, reliable sample/module/spatial coverage, and
fallback reason when the array evidence is weak.

Supply `calibration.lri` and `zoom_calib_v0.lri` whenever possible. Capture
headers contain only part of the camera model; without device mirror-aiming and
remaining geometry data, cross-module alignment is likely to be poor.
If `hotpixel.rec` is used, it must belong to the same physical camera as the
capture. `--cleanup-profile` optionally applies learned temperature-, exposure-,
and gain-dependent defect and row/column correction before highlight recovery
and alignment. It requires the exact `hotpixel.rec` used to train the profile.

Common options include:

- `--canvas native|max|<scale>` controls output resolution; maximum is the
  default;
- `--max-megapixels` caps maximum-mode output at 82 MP by default;
- `--cleanup-profile` applies an optional `.chiaro-cleanup` calibration and
  records per-module availability and correction statistics in the report;
- `--resolution-reconstruction resample|multi-camera|joint-cfa` selects ordinary
  pull resampling, the legacy locally aligned multiscale reconstruction, or
  the default pre-demosaic Joint-CFA solver. Joint CFA solves calibrated
  physical samples jointly and falls back to the production MultiCamera result
  wherever independent local support is insufficient. MultiCamera remains
  explicitly selectable for comparison and rollback;
- `--joint-cfa-solve-flat` is a diagnostic-only switch that restores solver
  attempts where the reference structure gate guarantees a zero-weight update;
  ordinary Joint-CFA rendering skips those attempts and reports the saved
  fraction;
- `--cfa-held-out CAMERA` excludes a physical Bayer module from both baseline
  and joint reconstruction, then reports how well each predicts that camera's
  unseen real CFA measurements. Repeat it for camera-wise cross-validation;
- `--factory-profile cct-only|array-aware|a|f11|d65` controls factory colour
  selection. D65 preserves the original processing behavior and is the
  default. CCT-only and array-aware remain experimental; A and F11 are fixed
  diagnostic profiles;
- `--no-crop` renders the full reference view instead of the framed field;
- `--crop X,Y,W,H` selects a reproducible reference-raster region for matched
  diagnostic renders (use `--canvas 1` for one output pixel per reference
  pixel);
- `--camera` selects modules and can be repeated;
- `--demosaic simple|amaze|rcd|lmmse|igv` selects Bayer reconstruction; AMaZE
  is the default, RCD is intended for individual night photos, and LMMSE or IGV
  can be used when preparing night-stack output;
- `--highlight-recovery none|local-bayer|multiscale-bayer|multi-camera`
  selects pre-demosaic RAW recovery; multi-camera is the default;
- `--crosstalk none|factory|adaptive` controls the four-phase spatial
  correction; adaptive is the default and safely falls back per camera when a
  capture-specific residual does not improve held-out overlap measurements;
- `--exclude-mono` omits monochrome luminance;
- `--no-depth` keeps the global homography for depth-refinement comparisons;
- `--depth-near` and `--depth-far` set the calibrated local search interval
  (0.5 m to 10 km by default);
- `--no-highlight-correction` disables only the final display-oriented smooth
  shoulder; use `--highlight-recovery none` to preserve the RAW mosaic too;
- `--color display|linear` controls output encoding; and
- `--debug-dir` writes per-module alignment checkerboards, luminance/colour
  source-ownership maps, quantitative inverse-depth, log-colour depth, and
  colour-coded provenance control grids.

Factory colour records can be inspected independently:

```bash
chiaro-color-profile calibration.lri -o colour-profile.json
```

The report preserves ColorMatrix, ForwardMatrix, grey ratios, all 24 Macbeth
measurements, illuminant spectra, sensor spectra, and separate `gold_cc`
records. It also compares the existing D65-only path, each illuminant's factory
matrix, a robust linear refit, and a small regularized nonlinear candidate using
leave-one-patch-out CIEDE2000 statistics. `--raw-only` skips the fitting pass.

Run `chiaro-fuse --help` for all options.

## Joint CFA reconstruction and cross-camera validation

Joint CFA reconstruction operates on corrected physical R/Gr/Gb/B sites before
demosaicing. It robustly fits a compact local XYZ field from calibrated camera
response rows, sensor-noise variance, highlight provenance, alignment
confidence, and an edge-aligned Hann window. A local affine field is used so
the test does not obtain lower error merely by averaging away edges. The
implementation is deterministic, CPU-only, and independently tileable.

The real-capture validation keeps a MultiCamera render as the comparison
baseline and withholds one non-reference module:

```bash
chiaro-fuse capture.lri -o held-out-B2.png \
  --resolution-reconstruction multi-camera \
  --cfa-held-out B2
```

The neighbouring report contains noise-normalized robust prediction loss for a
fixed, deduplicated list of physical sensor sites. Solver failures remain in
the overall population as baseline fallbacks; `common_region` reports the
supported subset separately. Results are split into literal R/Gr/Gb/B,
green-phase, red/blue-phase, independently defined flat/structured, and
inverse-projection-residual populations. Empty populations carry
`measured: false`. Positive relative improvement means the emitted Joint-CFA
or fallback value predicted a camera it never observed more accurately.
Contributor losses are explicitly in-sample diagnostics, not evidence of added
resolution. Joint diagnostics also record attempted/support counts, response
and spatial conditioning, application confidence, peak resident memory, and
total/synthesis time per megapixel. Held-out reports estimate uncertainty over
64x64 sensor blocks instead of treating neighboring CFA sites as independent.
They also retain deterministic per-site phase, reference-only structure,
held-out SNR, prediction/loss, and solver-support diagnostics for paired subset
analysis without using in-sample contributor residuals as quality evidence.

Fusion builds a calibrated multi-camera inverse-depth cost field after global
alignment. Eight-direction semi-global matching proposes coarse hypotheses;
every accepted final node is independently remeasured on a finer grid using
small or adaptive edge-aware support. Missing regions are not completed.
Continuous per-camera depth and bounded residual refinement follow. Warp
discontinuities are suppressed at visible scene edges, and robust
reference-guided synthesis rejects contradictory edge samples while retaining
agreeing module detail. Luminance and colour consistency are evaluated
separately so defocused chromatic fringes cannot survive merely by matching
brightness. Focus distance interpolated from the captured lens position and
supported near-side residual parallax suppress a magnified source focused
behind nearby content. Fine structure stays reference-anchored unless a sharper source
reproduces the same direction, in which case the zoom module can own the detail.
Thin branches and wires retain centre-surround protection. Distant or ambiguous
areas retain the global warp;
detail that moved differently in every exposure cannot be reconstructed.
Before crosstalk and demosaic, RAW highlight recovery combines edge-aware local
colour-difference reconstruction with a clipping-aware multiscale pyramid.
Multi-camera mode additionally borrows radiance only when at least two aligned,
unclipped modules agree. The default adaptive crosstalk stage retains the
factory 17x13 matrix mesh as a prior and fits only a small, white-balance-aware
residual from smooth aligned regions. Display-ready output retains the smooth
sensor-white shoulder as a final neutral safeguard.
Factory colour conversion searches non-negative A/F11/D65 blends against
reliable aligned inter-module chroma while retaining capture white balance/CCT
as a soft prior and fallback. This distinguishes warm LED spectra from tungsten
when their nominal colour temperatures are similar. A conservative
Macbeth refit is reported but is promoted only if held-out accuracy, neutral
stability, and inter-module consistency all improve; current device data keeps
the supplied factory matrices.
Night-sky captures may retain visible module boundaries. Use Chiaro Hotpixel
and a stacker for astrophotography.
