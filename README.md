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
- **Chiaro Stack** denoises every temporal camera burst and combines the
  resulting modules through calibrated multi-camera fusion.

## Feature comparison

✅ Available · 🟡 Limited/baseline · 🚧 Planned · ❌ Not available

| Feature | Chiaro | Light Lumen |
| --- | --- | --- |
| Desktop support | ✅ Linux is primary; Windows and macOS release builds | ✅ Windows and macOS |
| Access to captures | ✅ Browse folders or a connected L16 directly over PTP/MTP | ✅ Import captures from the camera |
| Individual camera views | ✅ Contact sheet and full-resolution preview for every module | ❌ Not exposed |
| Captures without companion `.lris` files | ✅ Builds a colour-calibrated preview from LRI RAW data | ❌ Not detected or processed |
| Local catalog | ✅ Inspectable SQLite capture, thumbnail, and export history | ✅ Proprietary imported-photo library |
| Per-camera batch export | ✅ Corrected 16-bit RGB or Bayer PNG stacks; five demosaicing methods | ❌ Not exposed |
| Night-mode denoising | 🟡 Gallery and CLI export with gyro-seeded, motion-aware temporal and multi-camera fusion | ✅ Integrated night processing |
| Computational fusion | 🟡 Calibrated global and dense depth alignment, locally verified multi-camera subpixel reconstruction, and up to 82 MP PNG synthesis | ✅ Depth-aware multi-camera fusion |
| Depth and focus editing | 🚧 Planned | ✅ Focus adjustment, depth effect, and depth-map repair |
| Finished-image formats | 🟡 Fused 16-bit PNG; JPG and DNG planned | ✅ JPG and DNG |
| Best fit | Open browsing, inspection, research, and per-module workflows | Finished-photo fusion and depth editing |

Chiaro Fuse also includes an experimental `--rig-strategy anchor-graph` path
for highly repetitive multi-camera scenes. It grows correspondence identity from
leave-one-out-validated anchor constellations on an adaptive factory-overlap
camera graph for up to four bounded rounds. From round two it performs a
global structureless bundle solve over every observable camera parameter while
re-triangulating the shared scene landmarks inside the objective. Candidate
pairs default to ≥20% overlap of the smaller FOV; the graph targets
connected degree-3+ topology and short cycles rather than a reference-camera
star. Newly activated edges are directly searched in the native image data. The
factory geometry is used as a wide bootstrap proposal; once an intermediate
rig exists, sparse active edges are re-seeded against that capture-specific
geometry while retaining mutual, appearance-margin, reverse-closure, and
constellation-validation gates. An "active" edge is therefore never merely
transitive bookkeeping. See
[`ANCHOR_GRAPH_ROUNDS.md`](ANCHOR_GRAPH_ROUNDS.md).

The factory mirror-quadratic interpretation and CRA-derived focus/pupil model
have also been audited against real held-out correspondences. See
[`CALIBRATION_METADATA_AB.md`](CALIBRATION_METADATA_AB.md) for the recovered
inverse-root semantics, A/B results, diagnostics, and conservative defaults.

The fusion pipeline builds a calibrated multi-view cost field, uses
semi-global matching to seed a finer direct-measurement pass, and accepts only
finite depths reproduced by independent camera evidence. Distant, ambiguous,
or unsupported regions safely retain the global warp rather than receiving
completed depth. Per-camera local refinement and reference-guided robust
blending protect object boundaries from double edges. Motion seen differently
by every exposure can still lose detail. A classical, bounded-memory resolution
stage matches modules at common bandwidth, selects the finest locally verified
optical tier, and combines its multiscale coefficients. A denser tele observation
can transfer real detail directly; same-resolution reconstruction still requires
coherent subpixel phases. Night captures can be processed with
temporal and multi-camera fusion in Gallery or Chiaro Stack.

## Installation

Download a prebuilt Linux archive from [GitHub
Releases](https://github.com/shinf1x/chiaro/releases), or install individual
applications with Cargo:

```bash
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-gallery
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-hotpixel
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-fuse
cargo install --git https://github.com/shinf1x/chiaro.git chiaro-stack-app
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
| [`chiaro-stack`](apps/stack/README.md) | Motion-aware night-frame stacking CLI |
| [`chiaro`](crates/chiaro/README.md) | Shared LRI parsing, metadata, and preview decoding |
| [`chiaro-hotpixel-core`](crates/chiaro-hotpixel-core/README.md) | Reusable RAW correction pipeline |
| [`chiaro-fusion`](crates/chiaro-fusion/README.md) | Alignment and synthesis library |
| [`chiaro-stack`](crates/chiaro-stack/README.md) | Temporal RAW denoising library |
| [`chiaro-proto`](crates/chiaro-proto/README.md) | Recovered Light metadata bindings |

## Acknowledgements

Chiaro builds on Light L16 format research from
[`ookami125/lri-cpp`](https://github.com/ookami125/lri-cpp),
[`gennyble/lri-rs`](https://github.com/gennyble/lri-rs), and
[`dllu/lri-rs`](https://github.com/dllu/lri-rs). The latter recovered the
original Protocol Buffer definitions from Lumen.

### Matching/depth V7 experiment

The V7 experimental patch adds capture-geometry reseeding for sparse anchor
edges, projectively distinct multimode depth refinement, image-space depth
uncertainty, local slanted-plane scoring, global rig-rank filtering, and
geometry-gated high-frequency reconstruction. See
[`MATCHING_DEPTH_V7.md`](MATCHING_DEPTH_V7.md).

### Landmark-constellation V8 experiment

V8 promotes the anchor graph to a persistent scene-landmark identity graph.
Whole-image distinctiveness is used only to prioritize bootstrap markers; final
identity is camera/FOV-conditional and requires local appearance, reverse
closure, and an independently fitted multi-landmark constellation. From round
two all observable camera parameters participate in one structureless bundle
objective. Observation membership is reversible: leave-one-observation-out
reprojection and leave-one-landmark-out constellation checks can demote one bad
camera observation without deleting the physical landmark, while coherent bad
subsets are split into alternate landmark identities rather than discarded. No
capture-wide camera quality multiplier is used. See
[`LANDMARK_CONSTELLATION_V8.md`](LANDMARK_CONSTELLATION_V8.md).
