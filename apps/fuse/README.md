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
- `--exclude-mono` omits monochrome luminance;
- `--no-highlight-correction` preserves unequal clipped-channel colour for
  downstream processing;
- `--color display|linear` controls output encoding; and
- `--debug-dir` writes per-module alignment checkerboards.

Run `chiaro-fuse --help` for all options.

Fusion currently assumes distant scenes. Near subjects can show parallax or
ghosting, and night-sky captures may retain visible module boundaries. Use
Chiaro Hotpixel and a stacker for astrophotography.
