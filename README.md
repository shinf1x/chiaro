# Chiaro

Chiaro is an open-source toolkit for browsing and processing Light L16
captures. It is intended to provide a portable alternative to parts of the
proprietary Lumen workflow. Linux is currently the primary supported platform.

![Chiaro Gallery displaying Light L16 captures](assets/docs/gallery_ui.png)

## Applications

- **Chiaro Gallery** browses local or camera-resident `.lri` captures, previews
  individual camera modules, and runs export pipelines.
- **Chiaro Hotpixel** extracts corrected 16-bit per-camera frames for
  astrophotography and stacking workflows.
- **Chiaro Fuse** aligns the modules of a capture and synthesises one
  high-resolution frame.

The fusion pipeline currently assumes distant scenes. Near subjects may show
parallax or ghosting, and depth-aware fusion is not yet implemented. Night
captures are best processed as per-camera stacks with Chiaro Hotpixel.

## Installation

Download a prebuilt Linux archive from [GitHub
Releases](https://github.com/shinf1x/chiaro/releases), or install individual
applications with Cargo:

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-gallery
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-hotpixel
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-fuse
```

To build the complete workspace:

```bash
cargo build --workspace --release
```

The resulting binaries are written to `target/release/`.

## Workspace

| Package | Purpose |
| --- | --- |
| [`chiaro-gallery`](apps/gallery/README.md) | Native gallery and export application |
| [`chiaro-hotpixel`](apps/hotpixel/README.md) | Per-camera extraction and correction CLI |
| [`chiaro-fuse`](apps/fuse/README.md) | Multi-camera fusion CLI |
| [`chiaro`](crates/chiaro/README.md) | Shared LRI parsing, metadata, and preview decoding |
| [`chiaro-hotpixel-core`](crates/chiaro-hotpixel-core/README.md) | Reusable RAW correction pipeline |
| [`chiaro-fusion`](crates/chiaro-fusion/README.md) | Alignment and synthesis library |
| [`chiaro-proto`](crates/chiaro-proto/README.md) | Recovered Light metadata bindings |

## Acknowledgements

Chiaro builds on Light L16 format research from
[`ookami125/lri-cpp`](https://github.com/ookami125/lri-cpp),
[`gennyble/lri-rs`](https://github.com/gennyble/lri-rs), and
[`dllu/lri-rs`](https://github.com/dllu/lri-rs). The latter recovered the
original Protocol Buffer definitions from Lumen.
