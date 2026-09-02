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
  --calibration calibration.lri \
  --calibration zoom_calib_v0.lri
```

The command writes `fused.png` and a neighbouring `fused.fusion.json`
diagnostic report.

Supply `calibration.lri` and `zoom_calib_v0.lri` whenever possible. Capture
headers contain only part of the camera model; without device mirror-aiming and
remaining geometry data, cross-module alignment is likely to be poor.
If `hotpixel.rec` is used, it must belong to the same physical camera as the
capture.

Common options include:

- `--canvas native|max|<scale>` controls output resolution;
- `--max-megapixels` caps maximum-mode output;
- `--no-crop` renders the full reference view instead of the framed field;
- `--camera` selects modules and can be repeated;
- `--demosaic simple|amaze|rcd|lmmse|igv` selects Bayer reconstruction; AMaZE
  is the default, RCD is intended for individual night photos, and LMMSE or IGV
  can be used when preparing night-stack output;
- `--exclude-mono` omits monochrome luminance;
- `--no-depth` keeps the global homography for depth-refinement comparisons;
- `--depth-near` and `--depth-far` set the calibrated local search interval
  (0.5 m to 10 km by default);
- `--no-highlight-correction` disables display-oriented clipped-highlight
  reconstruction and preserves the unequal raw-channel response for downstream
  processing;
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
Display-ready output uses a smooth shoulder near sensor white to neutralise the
false magenta produced when raw colour channels clip at different effective
levels after white balance. This avoids a hard boundary in the highlight, but
cannot recover colour or texture once every channel is saturated. Disable it
when exporting material intended for a dedicated raw processor.
Night-sky captures may retain visible module boundaries. Use Chiaro Hotpixel
and a stacker for astrophotography.
