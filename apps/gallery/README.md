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
camera cards while the remainder of the LRI is still transferring. Ready frame
batches, including later color-calibration updates, decode across the available
CPU cores without making camera transport concurrent. Click a camera card after
the transfer completes to decode its full native resolution image in memory.
Drag to pan, and use the mouse wheel or slider for pointer-centered zoom. The
image fits the viewer when first opened. Escape returns to the contact sheet; a
second Escape closes the contact sheet.
The contact modal and its dimmed backdrop stop above the status bar, leaving
transfer progress unobscured. Both gallery and contact-sheet scrollbars reserve
their own right-side gutter instead of overlapping image cards.

The app scans the selected directory itself (not subdirectories). It supports
both `RAW_PACKED_10BPP` surfaces and the L16 night-mode `RAW_BAYER_JPEG` layouts:
four interleaved Bayer JPEG planes or one full-resolution mono JPEG. The contact
sheet shows every frame in a repeated night capture and labels each frame under
its physical camera. An unsupported or damaged capture remains as an error
card; hover it for details.

## Known bugs

- Interrupting an active PTP operation by disconnecting USB can leave the Light
  L16 camera stuck. Recovery currently requires a hard camera reboot by holding
  its power button for approximately 30 seconds. This is documented for now and
  is not being fixed in the current work.

## Dependencies

The reusable library depends only on `memmap2`, `thiserror`, and the small
`zune-jpeg` decoder required by night captures. It is a separate package from
the gallery, so library consumers never pull in a UI or USB stack.

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
  parallel.rs   dependency-free bounded CPU parallelism
  source/
    mod.rs       source models, device monitor, and public source API
    discovery.rs camera discovery, indexing, and local folder listing
    transfer.rs  sparse/windowed LRI reads and streamed preview batches
    jpeg.rs      companion JPEG decoding and local path helpers
  main.rs        native application entry point
```
