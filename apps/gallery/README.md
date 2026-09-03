# Chiaro Gallery

Chiaro Gallery is a native Linux application for browsing and processing Light
L16 captures from local folders or a connected camera over PTP/MTP.

![Chiaro Gallery displaying Light L16 captures](../../assets/docs/gallery_ui.png)

## Installation

Download a prebuilt archive from [GitHub
Releases](https://github.com/shinf1x/chiaro/releases), install with Cargo, or
run the workspace package directly:

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-gallery
cargo run -p chiaro-gallery --release
```

## Browsing captures

Connected Light L16 cameras appear automatically. Local folders can be opened
with **+ Folder...** or by dropping a folder or LRI file onto the application.
Folder scans are non-recursive.

Each capture card shows its main metadata and framed preview. Opening a card
shows all calibrated camera frames together on one translucent composite
canvas. Embedded camera intrinsics, distortion, and module pose remap usable
frames into the reference view; modules whose embedded geometry is incomplete
fall back to focal-group scaling. Pointing at a frame in the list on the right
draws that layer last without hiding the other frames; clicking either the list
entry or canvas opens the existing full-resolution pan-and-zoom preview for
that frame. A gold outline marks the final crop recorded by the camera, relative
to the reference focal group. Colour previews use the calibration matrix and
white balance embedded in the LRI. The traditional grid remains available by
turning off **Show a combined calibrated frame overlay** in Settings. Damaged
or unsupported captures remain visible as error cards.

![Image Preview](../../assets/docs/gallery_image_preview.png)

Decoded previews are cached in the platform cache directory. The **Settings**
tab controls whether caching is enabled, its size limit, and cache clearing.
Persistent settings are stored in the platform configuration directory as
`chiaro/gallery.json`.

An inspectable `gallery.sqlite3` database beside `gallery.json` indexes capture
identity hashes, source and thumbnail paths, and successful exports. Thumbnail
files are split across two hash-prefix directory levels so large collections do
not put thousands of files in one directory.

## Exporting

Select cards with their checkbox or non-image area; Shift-click extends the
current selection range. Starting an export clears the selection. Additional
exports can be added while another job is running and are processed in order.
Cards that have been exported successfully show a muted bronze output icon
beside their filename; the state persists through `gallery.sqlite3`.

Available pipelines:

- **Hot-pixel corrected frames** writes linear 16-bit PNGs into one directory
  per physical camera. It uses the same correction pipeline as
  [`chiaro-hotpixel`](../hotpixel/README.md). RGB export offers Simple, AMaZE,
  RCD, LMMSE, and IGV demosaicing; AMaZE is the default.
- **Fused high-resolution frame** aligns the participating modules and writes
  one 16-bit PNG plus a `.fusion.json` diagnostic report per capture. It uses
  the same pipeline as [`chiaro-fuse`](../fuse/README.md), defaults to the
  camera-justified maximum canvas (up to 82 MP), and exposes selectable
  multi-camera resolution reconstruction, demosaicing, and an optional
  `.chiaro-cleanup` profile. AMaZE is the default.
- **Night stack** accepts only captures marked as night mode. It motion-aligns
  and denoises every temporal burst, then performs calibrated multi-camera
  fusion through [`chiaro-stack`](../stack/README.md). Its settings expose the
  motion rejection threshold, gyro seed, alignment refinement, calibration,
  output resolution and colour, multi-camera reconstruction, demosaicing,
  optional per-frame `.chiaro-cleanup`, adaptive RAW crosstalk,
  flat-field/highlight corrections, compression, and resume behavior. Each PNG is accompanied by a
  `.night-fusion.json` diagnostic report. LMMSE is the night-stack default;
  IGV is the other noise-tolerant stack option.

Fused and night exports default to adaptive RAW crosstalk. The factory spatial
four-phase calibration remains the prior; the capture-specific residual is
accepted per camera only when it improves held-out smooth overlap samples.

Exports report progress, can be cancelled, check available disk space, and
record failures in a pipeline-specific export log. Standard pipelines skip
night-mode captures, while Night stack skips standard captures; a mixed
selection clearly lists the files that will be excluded before the job starts.

### Camera calibration

When a camera is indexed, Gallery looks for these device-specific files in
`DCIM/Camera/lightcal`:

- `hotpixel.rec`
- `calibration.lri`
- `zoom_calib_v0.lri`

They are copied automatically to the platform cache when a camera is first
indexed. Gallery matches the calibration files to captures by the physical
camera's 128-bit device id and uses the completed geometry for the combined
preview and export defaults. Calibration from a different camera is ignored;
manually selected paths are not overwritten.

Use the `hotpixel.rec` belonging to the same physical camera. The geometric
calibration files are also strongly recommended: capture headers contain only
part of the camera model, and cross-module alignment is likely to be poor
without the remaining geometry and mirror-aiming data.

## Important limitations

- Fusion combines a refined global homography with fine-grid direct
  multi-camera depth measurements. Semi-global matching supplies search seeds
  but cannot fill final gaps; distant or ambiguous regions retain global
  alignment. Robust reference-guided blending suppresses contradictory double
  edges, protects thin reference structures, and prevents weaker, misaligned
  detail from softening the reference, but detail that moved differently in
  every exposure cannot be reconstructed.
- Disconnecting USB during an active PTP operation can leave some L16 cameras
  unresponsive. Recovery may require holding the camera power button for about
  30 seconds.
