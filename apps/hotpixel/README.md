# Chiaro Hotpixel

A Chiaro command-line application for astrophotography workflows
with the Light L16. It uses the workspace's shared `chiaro::lri`
parser rather than maintaining a second LRI implementation.

It scans a folder of `.lri` captures and creates one clean output directory per physical
camera. Each camera directory contains only corrected 16-bit PNG frames, so the complete
directory can be selected directly for further processing in Siril or other stacking
software.

Chiaro Hotpixel prepares the frames; it does not align or stack them.

## Installation

Download the latest prebuilt Linux archive from [GitHub
Releases](https://github.com/shinf1x/chiaro/releases), or install Hotpixel
directly from the repository:

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-hotpixel
```

## Output layout

```text
frames/
├── A1/
│   ├── L16_04480.png
│   └── L16_04481.png
├── B1/
│   ├── L16_04480.png
│   └── L16_04481.png
├── C3/
│   └── L16_04480.png
├── manifest.json
└── README.txt
```

Camera folders contain PNG files only. Run metadata stays at the output root.

## Processing performed

1. Parse each LRI directly in Rust.
2. Extract every captured RAW10 camera surface.
3. Select the corresponding factory map from `hotpixel.rec`:
   - `A1..A5` → records `0..4`
   - `B1..B5` → records `5..9`
   - `C1..C6` → records `10..15`
4. Rotate the factory map 180 degrees to match decoded RAW coordinates.
5. Correct factory-calibrated hot/dead pixels with same-color interpolation.
6. Write linear 16-bit PNGs into the physical camera's directory.

The default adaptive correction requires both:

- factory severity `>= 16` or the special value `255`; and
- a local outlier larger than `max(4 RAW codes, 6 × local MAD)`.

Factory values `1..254` correct positive hot outliers only. Value `255` is treated as a
known-defect class and may correct either hot or dead outliers.

For Bayer sensors, all hot-pixel calculations use only samples of the same CFA color.

## What is deliberately not applied

- no black-level subtraction;
- no dark-frame subtraction;
- no flat-field correction;
- no white balance;
- no color matrix;
- no exposure normalization;
- no gamma or tone curve;
- no stretch;
- no denoising or sharpening;
- no alignment or geometric correction.

The default RGB output performs only simple linear bilinear demosaicing after hot-pixel
correction. Use `--mode mosaic` to preserve the corrected Bayer mosaic instead.

## Build

```bash
cargo build -p chiaro-hotpixel --release
```

The binary is:

```text
chiaro-hotpixel
```

## Typical run

```bash
chiaro-hotpixel \
  /data/night_lrIs \
  /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --overwrite
```

This processes all `.lri` files directly inside `/data/night_lrIs`.

For nested input directories:

```bash
chiaro-hotpixel \
  /data/night_lrIs \
  /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --recursive \
  --overwrite
```

Nested source names are flattened safely. For example:

```text
session_1/L16_04480.lri → B1/session_1__L16_04480.png
```

## Resume an interrupted extraction

```bash
chiaro-hotpixel \
  /data/night_lrIs \
  /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --resume
```

Existing PNG files are left untouched and missing frames are generated.

## Preserve Bayer mosaics

```bash
chiaro-hotpixel \
  /data/night_lrIs \
  /data/night_frames_raw \
  --hotpixel-rec /data/hotpixel.rec \
  --mode mosaic \
  --overwrite
```

These are grayscale PNGs containing the corrected sensor mosaic. Most generic stackers
will treat them as monochrome because PNG has no standard CFA metadata; RGB mode is the
safer default for ordinary stacking software.

## Process selected cameras only

```bash
chiaro-hotpixel \
  /data/night_lrIs \
  /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --camera B1 \
  --camera B2 \
  --camera C3 \
  --overwrite
```

## Sensor metadata override

Monochrome Light modules are detected through the LRI hardware metadata. An explicit
override is available for unusual files:

```bash
--pattern A2=MONO
--pattern B1=RGGB
```

## Using the output in stacking software

Open each physical-camera directory separately in Siril or another stacking
application. Do not mix B1, B2, C3, and other camera folders in one initial
stack because they have different optics, viewpoints, and sometimes sensor
types. After processing each camera independently, align and combine the
resulting stacks if desired.
