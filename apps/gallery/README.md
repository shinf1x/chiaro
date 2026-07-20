# Chiaro Gallery

Chiaro Gallery is the suite's native Linux application for browsing and
inspecting Light L16 captures from local folders or directly from a camera over
PTP/MTP. Its dark-themed UI uses the immediate-mode
[`egui`](https://github.com/emilk/egui) framework through `eframe`.

The current release focuses on browsing and inspection. Copying, moving, and
exporting captures are not available yet.

![Chiaro Gallery displaying Light L16 captures](../../assets/docs/gallery_ui.png)

The header uses the shared Chiaro logo and wordmark assets, with the application
name on its upper row and the workspace package version below it. Raster exports
are embedded for display so the app does not need a runtime SVG renderer.

## Installation

Download the latest prebuilt Linux archive from [GitHub
Releases](https://github.com/shinf1x/chiaro/releases), or install the gallery
directly from the repository:

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-gallery
```

## Run from source

```bash
cargo run -p chiaro-gallery --release
```

Sources appear as tabs across the top:

- Every connected Light L16 becomes an **L16 - PTP** or **L16 - MTP** tab.
  Discovery starts with the app and continues in the background, so tabs appear
  and disappear as cameras connect or disconnect. A newly detected L16 is
  selected and loaded automatically, including when it changes USB mode.
- **+ Folder...** opens the folder-path view, which accepts a manual path and
  exposes the system folder picker. Choosing a folder creates its own closable
  tab. Dropping a folder or an LRI file from an explorer does the same thing.

The tab strip scrolls horizontally when space is tight. Source controls and the
preview-size slider live on separate responsive rows, so neither can overlap the
other at half-screen widths. Folder tabs can be closed from the tab strip. Each
card shows the file name, ISO, shutter speed, focal length, capture time, and
selected reference-camera ID. Active night/tripod states appear as compact
icons beside the filename. The initial size fits five columns in the default
window. Images follow the orientation
stored in the LRI metadata. Global transfer state and progress live in a bottom
status bar so the gallery does not jump as work starts and finishes.

Click a capture card to open its contact sheet. Complete LELR blocks populate
camera cards while the remainder of the LRI is still transferring. Click a
camera card after the transfer completes to decode its full native
resolution image in memory. Drag to pan, and use the mouse wheel or slider for
pointer-centered zoom. The image fits the viewer when first opened. Escape
returns to the contact sheet; a second Escape closes the contact sheet.
The contact modal and its dimmed backdrop stop above the status bar, leaving
transfer progress unobscured. Both gallery and contact-sheet scrollbars reserve
their own right-side gutter instead of overlapping image cards.

The app scans the selected directory itself (not subdirectories). It supports
both `RAW_PACKED_10BPP` surfaces and the L16 night-mode `RAW_BAYER_JPEG` layouts:
four interleaved Bayer JPEG planes or one full-resolution mono JPEG. Captures
with repeated night frames use frame zero for the gallery and contact sheet. An
unsupported or damaged capture remains as an error card; hover it for details.

## How previews work

- The LELR container and the required protobuf fields are parsed read-only.
- `LightHeader.image_reference_camera` selects the camera module; it is not
  hard-coded.
- The source is memory-mapped. Packed 10-bit captures sample only pixels needed
  for a maximum 720-pixel thumbnail; night-mode JPEG planes are decoded and
  interleaved in memory before thumbnail sampling.
- Bayer frames receive a lightweight demosaic, embedded black/white levels,
  capture AWB gains, a per-camera D65/F7 forward color matrix, exposure
  normalization, and display gamma. Mono camera frames are also supported.
- Preview pixels live in RAM and are uploaded directly to an egui texture. No
  PNG, JPEG, cache, sidecar, or temporary preview file is created.
- PTP/MTP use the pure-Rust `mtp-rs` protocol implementation directly; no
  desktop mount or external camera command is involved. Camera cards prefer the
  small companion JPEG exposed by recent L16 firmware.
  Opening a card streams its full LRI once into a shared memory buffer for the
  contact sheet and full-camera views. An older PTP capture with no companion
  JPEG remains clickable and loads its LRI on demand.
- Every discovered capture enters the preview queue. Pending cards entering the
  viewport are promoted ahead of unloaded off-screen cards without discarding
  either task or duplicating its decode.
- Camera captures are added to the gallery as their object records arrive;
  their cards and preview jobs appear dynamically, but camera indexing has
  exclusive transport priority over thumbnail reads. The queued thumbnails
  begin when the complete DCIM index has been received.
  If a companion JPEG is encountered after its LRI, the card is upgraded to use
  it automatically. A per-card revision invalidates any queued, active, or
  already displayed LRI thumbnail so the JPEG reliably wins the indexing race.
- Companion JPEGs inherit capture metadata and orientation from sparse reads of
  their matching LRI, but their already-correct display color is left unchanged.
  Those metadata-only reads retain just protobuf messages, not zero-filled
  stand-ins for the skipped RAW payloads.
- Captures without a companion JPEG are read sparsely. The gallery fetches each
  32-byte LELR header and trailing protobuf message, skips unrelated RAW data,
  then requests only the declared reference camera's packed or Bayer-JPEG
  payload. A continuous incremental stream remains as a compatibility fallback
  for devices that reject partial-object reads.
  Bayer RAW thumbnails continue scanning the small metadata messages until the
  reference camera's AWB/color profile is available, so they use the same color
  treatment as the completed contact-sheet RAW previews.
- Two background workers decode previews so folder selection and resizing stay
  responsive. A requested contact sheet or full view is prioritized over queued
  card previews as soon as each worker finishes its current task; active card
  work is never interrupted. Camera indexing pauses at an object boundary for
  the contact transfer, while the free worker immediately accepts the modal job;
  closing the modal resumes indexing and preserves the thumbnail queue.
  Contact-sheet camera cards are emitted after each newly
  completed LELR payload while the same stream continues into memory.
- Closing a contact sheet or replacing an LRI task with its JPEG cancels that
  transport cleanly. Switching tabs preserves each tab's items, camera-index
  progress, and preview state while background work continues. A stale
  transaction-ID response during the next camera open is retried with a fresh
  session and a short backoff.
- A newly enumerated L16 gets a short USB settle period before its tab opens.
  Camera open operations use bounded timeouts and expose their current stage in
  the UI. A disconnected, timed-out, or malformed initial PTP/MTP response
  triggers a protocol reset (with USB re-enumeration as a fallback), then the
  selected device tab keeps retrying while that same camera remains connected.
  Brief discovery gaps keep the active camera tab and every loaded texture for
  a grace period, avoiding a full gallery reset during transient USB events.

## Known bugs

- Interrupting an active PTP operation by disconnecting USB can leave the Light
  L16 camera stuck. Recovery currently requires a hard camera reboot by holding
  its power button for approximately 30 seconds. This is documented for now and
  is not being fixed in the current work.

## Dependencies

The reusable library depends only on `memmap2`, `thiserror`, and the small
`zune-jpeg` decoder required by night captures. It is a separate package from
the gallery, so library consumers never pull in a UI or USB stack.

The gallery uses `eframe`, `mtp-rs` plus its runtime-agnostic `futures`
executor, `nusb`, and `rfd`.
`eframe` is built with a reduced feature set: only the default fonts, OpenGL
renderer, Wayland, and X11 are enabled. `rfd` uses the desktop portal on Linux;
this is what allows the requested native picker without hard-coding a desktop
environment. `mtp-rs` uses pure-Rust `nusb`, with no libmtp/libusb FFI or
external `gphoto2` executable.

## Structure

```text
src/
  app/
    mod.rs      application state and top-level immediate-mode update
    branding.rs embedded Chiaro logo, wordmark, and version header
    tabs.rs     tab strip, source controls, and status bar
    cards.rs    responsive gallery card grid
    events.rs   background-result handling and texture upload routing
    modal.rs    contact sheet and full-resolution image viewer
    visuals.rs  shared image, metadata, progress, and icon widgets
  gallery/
    mod.rs      preview scheduler, queues, and public gallery state
    worker.rs   background capture loading and preview decoding
  source/
    mod.rs       source models, device monitor, and public source API
    discovery.rs camera discovery, indexing, and local folder listing
    transfer.rs  sparse/continuous LRI reads and preview orientation
    jpeg.rs      companion JPEG decoding and local path helpers
  main.rs        native application entry point
```
