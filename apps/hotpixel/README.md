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
5. Use the bundled sensor-family model to identify factory-listed pixels that
   should be active at this temperature, exposure, and gain.
6. Optionally remove camera-specific row/column fixed-pattern bias and add its
   residual active-pixel decisions.
7. Correct factory-calibrated hot/dead pixels with same-color interpolation.
8. Subtract the bundled low-frequency corner-glow model when its metadata is
   available.
9. Write linear 16-bit PNGs into the physical camera's directory.

The default adaptive correction requires both:

- factory severity `>= 16` or the special value `255`; and
- a local outlier larger than `max(4 RAW codes, 6 × local MAD)`.

Factory values `1..254` correct positive hot outliers only. Value `255` is treated as a
known-defect class and may correct either hot or dead outliers.

For Bayer sensors, all hot-pixel calculations use only samples of the same CFA color.

## Universal factory-hotpixel response

Hotpixel correction has two deliberately separate inputs:

- the supplied `hotpixel.rec` identifies the defect coordinates for this
  particular camera and sensor module;
- a bundled coordinate-free model predicts which factory severity classes are
  reliably active under the capture conditions.

The bundled model is enabled by default and was fitted from physical sensors in
camera groups A, B, and C. It has separate color and monochrome response curves,
but contains no camera IDs or pixel positions. Consequently, it transfers the
shared sensor behavior without pretending that two dies have defects in the
same locations. A predicted active coordinate is replaced with a same-color
local median; the predicted dark value is never subtracted from the photograph.
Coordinates not forced by the universal model still use the normal adaptive
factory-guided test, including the ambiguous hot/dead factory class `255`.

For factory severity `s`, the model is:

```text
predicted excess = response(sensor family, s, temperature)
                   × capture_time/14.999805952s
                   × analogue_gain/6.25
                   × digital_gain/1.015625
```

Temperature is not assumed to be proportional: the fitted response is linearly
interpolated between 25, 30, 35, 40, 45, 50, and 55°C nodes and clamps outside
that range. Exposure and both recorded gains scale linearly. The exposure and
digital-gain factors follow the RAW signal path; analogue-gain transfer is the
current model assumption and should be extended with more multi-ISO dark data.
The bundled coefficients use a conservative per-camera 25th percentile before
combining cameras, leaving uncertain pixels to the adaptive fallback.

If temperature is absent, exposure is zero, or either gain is absent/invalid,
only this universal forcing step is skipped. The supplied factory map and local
adaptive correction continue to work, so missing metadata does not disable
hotpixel cleanup. The reason and all applied scale factors are written per frame
to `manifest.json`.

Use `--no-universal-hotpixel-model` to disable the bundled prior, or
`--universal-hotpixel-profile FILE` to supply another compatible model. The
validated runtime model is bundled as `assets/l16-universal-hotpixel-v1.json`.

### Held-out validation

The following results compare the normal adaptive factory correction against
the same pipeline with the universal decision model enabled. The metric is RMS
of the persistent residual at factory-listed, modeled hot-only coordinates, in
original RAW10 codes; lower is better. The A/B captures were not used to fit
their exposure profile. For the stronger C-group transfer test, C1 and C6 were
excluded entirely while fitting the evaluated model.

| Exposure | Camera | Held-out frames | Before | After | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 s | A2 mono | 39 | 2.141 | 0.806 | 62.3% |
| 1 s | A3 color | 39 | 2.446 | 1.627 | 33.5% |
| 1 s | B1 color | 39 | 2.871 | 0.913 | 68.2% |
| 5 s | A2 mono | 44 | 0.696 | 0.312 | 55.2% |
| 5 s | A3 color | 44 | 1.058 | 0.404 | 61.8% |
| 5 s | B1 color | 44 | 2.688 | 0.330 | 87.7% |
| 15 s | A2 mono | 40 | 0.610 | 0.419 | 31.3% |
| 15 s | A3 color | 40 | 0.409 | 0.371 | 9.1% |
| 15 s | B1 color | 40 | 0.899 | 0.535 | 40.5% |
| 15 s, C1 excluded | C1 color | 13 | 0.565 | 0.467 | 17.3% |
| 15 s, C6 excluded | C6 mono | 13 | 3.944 | 0.968 | 75.5% |

Row/column profiles, non-defect spatial fixed-pattern RMS, and temporal-noise
RMS were unchanged in these comparisons. That is intentional: this model only
changes interpolation decisions at coordinates supplied by `hotpixel.rec`.

## Universal temperature-conditioned corner-glow correction

The validated low-frequency sensor-family coefficients are bundled into
`chiaro-hotpixel` and apply by default when the exposure and sensor metadata
match:

```bash
chiaro-hotpixel extract \
  --input /data/night_lris \
  --output /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --overwrite
```

`--glow-profile /external/experimental-profile` overrides the bundled model.
Use `--no-glow-correction` to perform only device-specific hot-pixel removal.

At extraction time, the supplied `hotpixel.rec` first corrects that particular
camera's defects. The universal profile then subtracts only the smooth spatial
glow. CFA orientation and analogue/digital gain are handled automatically.
A2 and C6 use the same field with their measured module mountings (horizontal
flip for A2, 180-degree rotation for C6); neither monochrome camera contributed
to the bundled fit.
Correction is applied only when dimensions and exposure match. Out-of-range
temperatures clamp to the trained range and are reported in `manifest.json`;
missing or mismatched metadata safely falls back to hot-pixel correction only.

For each position, the current model is:

```text
glow = (reference_field + temperature_slope × (sensor_temperature - 43°C))
       × capture_time/14.999805952s
       × analogue_gain/6.25 × digital_gain/1.015625
```

The two fields are bilinearly interpolated from the 64×48 grid. Temperature is
linear only within the trained 36–49°C range and clamps outside it. Capture
time scales continuously over the validated 4.999935488–14.999805952-second
range. Shorter captures are left unchanged: at one second the transferable
field is below one RAW code and readout/quantization structure dominates.
Glow correction is evaluated in Q6 RAW-code precision and written directly to
the linear 16-bit PNG. This avoids creating whole-RAW-code contour bands around
the smooth corner gradient; unmodified RAW10 samples remain exact multiples of
64.

Glow subtraction does not remove device-specific fine fixed-pattern noise or
random read/shot noise. Those are separate from corner glow; stacking or a
scene-aware denoiser is still needed for the random component.

## Optional device-specific defect and line cleanup

`chiaro-hotpixel calibrate` trains a compact residual profile from dark LRIs
taken at one exposure and ISO across a useful sensor-temperature range. Unlike
the universal hotpixel and corner-glow models, this profile is specific to the
physical camera modules and factory `hotpixel.rec` used for training. Users do
not need to make one for ordinary hotpixel cleanup; its main value is correcting
module-specific rows, columns, and wider bands:

```bash
chiaro-hotpixel calibrate \
  --input /external/l16_cleanup_training \
  --output /external/my-camera.chiaro-cleanup \
  --hotpixel-rec /data/hotpixel.rec \
  --overwrite
```

No camera list is needed: every module found in the folder is included. Use
repeatable `--camera` options only when intentionally calibrating a subset.
Individual LRIs do not need to contain every group; the output contains the
union of A/B/C modules found across the folder.

The trainer stores two kinds of coefficients:

- a linear or quadratic temperature response for factory-listed hot-pixel
  coordinates;
- temperature-conditioned additive offsets for every row and
  column, high-pass isolated within the matching monochrome or Bayer parity.
  The default 32-neighbor radius captures both single-pixel lines and wider
  bands of roughly 10 pixels while leaving the much broader corner glow to the
  universal model.

The residual per-pixel fit can add camera-specific active decisions to the
universal decisions. Other factory coordinates retain the normal adaptive
outlier test. The fitted dark value is deliberately not subtracted from a
photograph: held-out tests showed that direct dark-frame subtraction can
overcorrect individual pixels. The factory map remains the coordinate
authority and every replacement value comes from the photographed
neighborhood. Repair runs after line cleanup, which prevents the line model
from leaving a residual at an already interpolated coordinate.

Enable the profile explicitly during extraction:

```bash
chiaro-hotpixel extract \
  --input /data/night_lris \
  --output /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --cleanup-profile /external/my-camera.chiaro-cleanup \
  --overwrite
```

The output is one portable ZIP-based `.chiaro-cleanup` file. It contains the
manifest plus every selected A/B/C module's defect, row, column, and wide-band
coefficients; users do not need to merge per-camera results. Put all dark LRIs
from one calibration session in the input folder and run the calibrator once.
Legacy directory profiles remain readable for compatibility.

At least two distinct recorded sensor temperatures are required. Two
temperatures produce a linear response; three or more produce the normal
quadratic response. The chosen response type is recorded per module in the
file's manifest.

The profile refuses a different `hotpixel.rec` hash. Dimensions and sensor
pattern must match. Exposure, analogue gain, and digital gain scale the fitted
defect and line/band amplitudes relative to the calibration capture, and all
three factors are recorded per frame in `manifest.json`. Missing metadata or an
absent camera entry safely skips the optional cleanup. Temperatures outside the
training range clamp to its nearest endpoint. Monochrome modules use a
one-pixel parity step, so C6 can be trained and applied by including C6 dark
frames; it is not excluded by the model design.

Line correction is calculated in Q6 precision before hot-pixel interpolation.
For each row or column, the model is:

```text
offset = reference + slope × Δtemperature + curvature × Δtemperature²
```

Validate a cleanup profile with dark captures that were not used to create it.
Use one shared stretch for before/after images and inspect row/column profiles;
independent auto-stretches or isolated bright pixels are misleading.

## What is deliberately not applied by default

- no black-level subtraction;
- no device-specific fine fixed-pattern subtraction;
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
chiaro-hotpixel extract \
  --input /data/night_lrIs \
  --output /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --overwrite
```

This processes all `.lri` files directly inside `/data/night_lrIs`.

For nested input directories:

```bash
chiaro-hotpixel extract \
  --input /data/night_lrIs \
  --output /data/night_frames \
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
chiaro-hotpixel extract \
  --input /data/night_lrIs \
  --output /data/night_frames \
  --hotpixel-rec /data/hotpixel.rec \
  --resume
```

Existing PNG files are left untouched and missing frames are generated.

## Preserve Bayer mosaics

```bash
chiaro-hotpixel extract \
  --input /data/night_lrIs \
  --output /data/night_frames_raw \
  --hotpixel-rec /data/hotpixel.rec \
  --mode mosaic \
  --overwrite
```

These are grayscale PNGs containing the corrected sensor mosaic. Most generic stackers
will treat them as monochrome because PNG has no standard CFA metadata; RGB mode is the
safer default for ordinary stacking software.

## Process selected cameras only

```bash
chiaro-hotpixel extract \
  --input /data/night_lrIs \
  --output /data/night_frames \
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
