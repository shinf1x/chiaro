# Chiaro Hotpixel

Chiaro Hotpixel extracts Light L16 RAW frames, corrects sensor defects, and
writes one stack of linear 16-bit PNGs per physical camera. It is intended for
astrophotography workflows in Siril and similar tools; it does not align or
stack the frames.

The command-line application uses the shared
[`chiaro-hotpixel-core`](../../crates/chiaro-hotpixel-core/README.md) pipeline,
which is also used by Chiaro Gallery.

## Installation

Download a prebuilt archive from [GitHub
Releases](https://github.com/shinf1x/chiaro/releases), install with Cargo, or
build the workspace package:

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-hotpixel
cargo build -p chiaro-hotpixel --release
```

## Extract frames

```bash
chiaro-hotpixel extract \
  --input /data/night_lris \
  --output /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec
```

`hotpixel.rec` must come from the same physical L16 as the captures. The
default RGB mode writes demosaiced Bayer frames and grayscale monochrome
frames. AMaZE is the default demosaicing method. Output is organised by camera:

```text
night_frames/
├── A1/
│   └── L16_04480.png
├── B1/
│   └── L16_04480.png
├── C3/
│   └── L16_04480.png
├── manifest.json
└── README.txt
```

Do not combine different camera directories into one initial stack: the
modules have different optics, viewpoints, and sometimes sensor types.

### Common options

- `--recursive` scans nested input directories.
- `--resume` keeps existing PNGs and generates missing frames.
- `--overwrite` replaces the existing output directory.
- `--camera NAME` restricts processing and can be repeated.
- `--mode mosaic` preserves corrected Bayer mosaics as grayscale PNGs. Generic
  PNG readers do not know their CFA pattern, so RGB is the safer default.
- `--demosaic simple|amaze|rcd|lmmse|igv` selects Bayer reconstruction. AMaZE is the default, RCD is intended for individual night photos, and LMMSE or IGV can be used for frames destined for night stacks.
- `--continue-on-error` processes remaining frames and exits nonzero afterward.
- `--threads` and `--png-level` control CPU use and compression.
- `--pattern CAMERA=PATTERN` overrides missing or unusual sensor metadata.

Run `chiaro-hotpixel extract --help` for the complete option list.

## Corrections

The default pipeline:

1. decodes each captured RAW module;
2. uses the matching factory map from `hotpixel.rec` to repair hot and dead
   pixels with same-colour neighbourhood samples;
3. applies bundled sensor-family models for condition-dependent hot pixels and
   low-frequency corner glow when the required capture metadata is available;
4. optionally applies a cleanup profile for camera-specific line and residual
   defects; and
5. writes linear 16-bit RGB or mosaic PNGs.

The bundled models contain no camera-specific defect coordinates and do not
replace `hotpixel.rec`. If required temperature, exposure, or gain metadata is
missing or incompatible, the relevant universal correction is skipped while
factory-map correction continues. Applied and skipped stages are recorded in
`manifest.json`.

Use `--no-universal-hotpixel-model` or `--no-glow-correction` to disable the
bundled models. Compatible replacement profiles can be supplied with
`--universal-hotpixel-profile` and `--glow-profile`.

## Optional camera cleanup profile

The `calibrate` command builds one portable `.chiaro-cleanup` profile from dark
captures:

```bash
chiaro-hotpixel calibrate \
  --input /data/l16_cleanup_training \
  --output /data/my-camera.chiaro-cleanup \
  --hotpixel-rec /data/hotpixel.rec
```

Training captures should come from one physical camera at the same exposure
and ISO, with at least two distinct recorded sensor temperatures. The profile
is tied to the supplied `hotpixel.rec` and refuses incompatible sensor layouts
or factory-map hashes.

Enable it during extraction with:

```bash
chiaro-hotpixel extract \
  --input /data/night_lris \
  --output /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --cleanup-profile /data/my-camera.chiaro-cleanup
```

Cleanup profiles are optional. Validate a trained profile against dark
captures that were not used for training before relying on it for important
work.
