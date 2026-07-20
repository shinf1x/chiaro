# Chiaro

Chiaro is an open-source suite for the Light L16, growing toward a portable
replacement for the proprietary Lumen workflow - from browsing `.lri` captures
to processing and export. Linux is the current focus; support for additional
platforms, possibly original L16 Android, is being explored.

The suite currently includes:

- **Chiaro Gallery** for browsing and inspecting Light L16 captures.
- **Chiaro Hotpixel** for extracting each camera into separate files and
  removing hot pixels from nighttime-sky captures, ready for further
  processing in Siril or other stacking software.

![Chiaro Gallery displaying Light L16 captures](assets/docs/gallery_ui.png)

## Feature comparison

✅ Available · 🟡 In development · 🚧 Planned · ❌ Not planned

| Feature | Chiaro Gallery | Light Lumen |
| --- | --- | --- |
| Desktop platform | ✅ Native Linux | ✅ Windows and macOS |
| Getting to captures | ✅ Browse folders or a connected camera directly over PTP/MTP | ✅ Transfer captures from the camera into Lumen |
| Individual camera views | ✅ Contact sheet and full-resolution preview for every camera | ❌ Not available |
| Captures without companion `.lris` files | ✅ Builds a color-calibrated preview from the LRI RAW data | ❌ Not detected or processed |
| Computational fusion | 🟡 In development | ✅ Fuses images from ten or more cameras |
| Depth and focus editing | 🚧 Planned | ✅ Depth effect, focus adjustment, and depth-map repair tools |
| Finished-image export | 🚧 Planned | ✅ JPG and DNG |
| Android support | Under consideration | ❌ Not available; x86 desktop application |
| Best fit | Fast browsing and inspection | Full computational processing and editing |

## Workspace

| Package | Path | Purpose |
| --- | --- | --- |
| `chiaro-proto` | `crates/chiaro-proto` | Shared recovered schemas and generated Light metadata bindings |
| `chiaro` | `crates/chiaro` | Portable LELR parsing, metadata, RAW layout, and preview decoding |
| `chiaro-gallery` | `apps/gallery` | Native egui gallery with folder and PTP/MTP sources |
| `chiaro-hotpixel` | `apps/hotpixel` | Per-camera extraction and hot-pixel correction for nighttime-sky captures |

Production packages depend on the shared `chiaro` crate instead of
implementing the LRI protocol independently.

## Installation

Download the latest prebuilt Linux archive from [GitHub
Releases](https://github.com/shinf1x/chiaro/releases), or install either app
directly from the repository with Cargo:

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-gallery
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-hotpixel
```

## Build

```bash
cargo build --workspace --release
```

The application binaries are:

```text
target/release/chiaro-gallery
target/release/chiaro-hotpixel
```

Published GitHub releases automatically build a Linux x86-64 archive containing
both binaries, this README, and the license. The same archive can be produced as
a workflow artifact by manually running the **Release Binaries** workflow.

Package-specific usage is documented in
[`apps/gallery/README.md`](apps/gallery/README.md) and
[`apps/hotpixel/README.md`](apps/hotpixel/README.md).

## Acknowledgements

- [`ookami125/lri-cpp`](https://github.com/ookami125/lri-cpp) by
  [`@ookami125`](https://github.com/ookami125).
- [`gennyble/lri-rs`](https://github.com/gennyble/lri-rs) by
  [`@gennyble`](https://github.com/gennyble).
- [`dllu/lri-rs`](https://github.com/dllu/lri-rs) by
  [`@dllu`](https://github.com/dllu), who recovered the original Protocol
  Buffer definitions from Lumen.
