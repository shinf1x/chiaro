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
diagnostic report.

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
- `--resolution-reconstruction resample|multi-camera` selects ordinary pull
  resampling or locally aligned, multiscale physical-sample reconstruction.
  The latter gives high-frequency ownership to the finest verified optical
  tier while retaining the reference camera's tone and colour. Locally sound
  detail may be recovered even from a module rejected for ordinary fusion;
- `--no-crop` renders the full reference view instead of the framed field;
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

Run `chiaro-fuse --help` for all options.

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
Night-sky captures may retain visible module boundaries. Use Chiaro Hotpixel
and a stacker for astrophotography.
