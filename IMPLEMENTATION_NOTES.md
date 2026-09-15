# Implementation notes

This tree adds `--rig-strategy anchor-graph` without replacing the existing
`physical` strategy.

## Main code changes

- `crates/chiaro-fusion/src/rig/anchor_graph.rs`
  - factory-overlap camera graph (all eligible pairs, not a reference star);
  - sparse, constellation-validated bootstrap anchors;
  - true leave-one-out local affine pair fields;
  - direct-edge/cycle tracking to prevent transitivity from proving itself;
  - scale-aware warped ZNCC on full available 2x2-CFA-cell luminance images;
  - iterative missing-observation completion;
  - iterative spawning of new tracks;
  - up to four propagation rounds by default;
  - stricter promotion/search gates in later rounds;
  - per-round orientation-only bearing bootstrap + structureless bundle fit;
  - round-2+ 3-D projection used only when it agrees with independent 2-D
    constellation prediction;
  - dual early-stop criterion on observation growth and cycle-supported 3+
    track growth, with adaptive edge activation preventing premature stop;
  - 8x6 per-camera spatial-coverage accounting and weak-camera-aware edge
    selection;
  - final cycle gate before tracks reach the physical rig fitter;
  - non-reference B<->C/C<->C tracks retained in the final metric solve.
- `crates/chiaro-fusion/src/rig.rs`
  - `RigRefinementStrategy::{Physical, AnchorGraph}`;
  - anchor-graph options/report integration;
  - anchor mode keeps factory camera centres and mirror state fixed while
    permitting observable orientation/sensor-raster refinement.
- `crates/chiaro-fusion/src/align.rs`
  - exposes the existing Shi-Tomasi corner detector internally to the rig
    child module; no new external API.
- `apps/fuse/src/main.rs`
  - CLI selection, camera-graph budget and convergence controls;
  - per-round console diagnostics.
- `crates/chiaro-fusion/src/pipeline.rs`
  - anchor-graph integration with the fusion pipeline.

## Defaults

```text
--rig-strategy physical                 # backward-compatible default
--rig-anchor-rounds 4                   # when anchor-graph is selected
--rig-anchor-min-overlap 0.20
--rig-anchor-initial-edges 18
--rig-anchor-max-edges 24
--rig-anchor-min-degree 3
--rig-anchor-edges-per-round 3
observation growth early stop: 1%
strong 3+ track growth early stop: 1%
```

Round 1 is driven by image constellations. Every round refits orientation from
fit-only tracks. From round 2 onward, a triangulated 3-D projection can tighten
a missing-observation proposal, but never override a disagreeing independent
2-D constellation.

## Validation performed in this environment

The environment used to prepare this patch does not contain `cargo`, `rustc`,
`rustfmt`, or `clippy`, and external package/toolchain installation is blocked.
Therefore no claim is made that the Rust workspace was compiled here.

Static checks performed:

- balanced Rust delimiters in all modified `.rs` files;
- anchor-graph round/early-stop/3-D/bundle code-path markers present;
- CLI wiring present;
- internal corner-detector visibility present;
- source tree diff checked so modifications are confined to the documented
  files plus the new anchor module/docs.

The anchor module contains unit tests for local affine prediction, rejection of
a 20-pixel periodic alias under true leave-one-out validation, the rule that a
reference-camera star is not a cycle while a closed triangle is, and balanced
redundant camera-edge selection.

## Run the real checks

```bash
cargo fmt --all -- --check
cargo test -p chiaro-fusion
cargo test -p chiaro-fuse
cargo clippy -p chiaro-fusion -p chiaro-fuse --all-targets -- -D warnings
```

Then compare the same capture:

```bash
target/release/chiaro-fuse capture.lri -o physical.png \
  --calibration calibration.lri \
  --calibration zoom_calib_v0.lri \
  --rig-strategy physical

target/release/chiaro-fuse capture.lri -o anchor.png \
  --calibration calibration.lri \
  --calibration zoom_calib_v0.lri \
  --rig-strategy anchor-graph \
  --rig-anchor-rounds 4 \
  --rig-anchor-min-overlap 0.20 \
  --rig-anchor-initial-edges 18 \
  --rig-anchor-max-edges 24 \
  --rig-anchor-min-degree 3 \
  --rig-anchor-edges-per-round 3
```
