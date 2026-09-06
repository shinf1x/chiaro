# Real-capture cross-camera CFA experiment

Status: experimental, updated 2026-09-06. `MultiCamera` remains the production
default.

> The numeric tables below are historical pre-correction results. Their
> populations changed with solver coverage and some predictions were evaluated
> at different spatial locations, so they must not be used as quality or
> camera-count evidence. Re-run the commands with the corrected protocol
> described below before drawing conclusions.

## Question

Can a compact joint reconstruction of physical Bayer measurements predict an
unseen L16 camera better than the existing per-camera demosaic followed by
`ResolutionReconstruction::MultiCamera`?

No synthetic scene or image was used. Algebra-only unit tests cover the solver;
all quality measurements below come from recorded L16 RAW samples.

## Implementation under test

Each observation retains the physical camera and sensor coordinate, R/Gr/Gb/B
phase, corrected linear value, calibrated noise variance, highlight provenance,
geometry confidence, visibility, camera response row, and its position in the
output lattice. Observations enter after the normal defect, thermal/glow, RAW
highlight, crosstalk, and flat-field stages and before demosaic.

The reconstruction fits a nine-parameter local-affine D50 XYZ field: centre XYZ
plus X/Y derivatives for each component. Three Huber IRLS iterations use
sensor-noise, highlight, spatial Hann-window, and geometry weights. A weak prior
anchors the centre to the production fused result. A constant-colour fit was
rejected during development because it blurred edges and made the initial
high-frequency held-out result 13.8% worse.

The experimental estimate is feathered over the production baseline only where
the reference contains supported luminance or chromatic structure. Flat regions
therefore remain bit-equivalent to the production estimate. Chromaticity edge
confidence fades out in dark regions where colour ratios are noise-sensitive.

Factory noise is propagated through the physical bilinear interpolation
footprint, local 4x4 crosstalk row, and flat-field gain. Reused physical sites
inflate neighboring observation variance conservatively. Highlight provenance
uses the same full nonzero footprint. The sensor's physical code range is kept
separate from working headroom, making the quantization floor representation
invariant.

Held-out cameras are excluded before cross-camera highlight recovery, common
profile selection, adaptive crosstalk, and scene photometric fitting. Camera
ablation runs load the same geometry population and freeze crop, scale, and
contributor-side radiometry. Validation deduplicates a fixed list of physical
held-out sites, inverse-projects each site to its continuous reference
coordinate, and scores baseline fallback at solver failures. Reports separate
overall and solver-supported common-region loss and include the sample IDs and
rejection counts.

Solver admission requires independently conditioned XYZ responses, 2D spatial
support, and robust support from multiple cameras before regularization can
stabilize remaining affine directions. Production border, warp, focus,
edge/detail, and chroma safeguards apply to every gathered footprint. If the
baseline contains monochrome or resolution-only information unavailable to the
Bayer equations, Joint CFA preserves its luminance. Contributor loss compares
the affine fit with the actual production baseline field at each observation
location and is labeled as an in-sample diagnostic.

## Historical primary held-out results (invalidated; rerun required)

The baseline and experiment use the same cameras, preprocessing, calibration,
geometry, crop, and output size. The named Bayer module participates in
geometry but is excluded from both reconstructions. Loss is robust error in
units of calibrated sensor noise; positive percentages mean lower Joint-CFA
loss.

| Capture | Held out | Real CFA samples | Overall | Luminance-like | Chroma-like | High frequency | Flat control |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L16_00563` | B2 | 10,497 | +5.42% | +5.32% | +5.51% | +5.70% | 0.00% (3) |
| `L16_00035` | B2 | 19,464 | +8.98% | +6.29% | +11.61% | +9.15% | 0.00% (3) |
| `L16_00054` | B2 | 7,655 | +2.71% | +0.15% | +4.97% | +2.58% | 0.00% (20) |
| `L16_01732` | A3 | 10,523 | +21.47% | +18.25% | +24.35% | +22.57% | 0.00% (51) |
| `L16_04364` | B2 | 53,980 | +7.16% | +6.53% | +7.94% | +10.68% | 0.00% (1,412) |

These figures describe the old evaluator only. The corrected report uses
literal phase groups, independently selected flat/structured bins, and an
explicit `measured` flag for empty bins.

## Historical camera-count and focal-tier ablation (invalidated)

`L16_04364`, reference B4, held-out B2:

| Reconstruction contributors | Kind | Overall improvement |
| --- | --- | ---: |
| B4 + B1 | 2, same B tier | +13.40% |
| B4 + B1 + B3 | 3, same B tier | +8.36% |
| B4 + B1 + B3 + B5 | 4, same B tier | +6.74% |
| B4 + B1 + B3 + B5 + C5 + C6 | 6, mixed B/C tiers | +6.82% |

The old runs changed both the scored population and preprocessing geometry.
They do not establish a camera-count trend. Corrected runs must compare the
identical `sample_ids` list and report overall fallback-inclusive loss beside
`common_region` coverage.

For the two-camera case, tightening the maximum held-out mapping error preserves
the result:

| Maximum sensor-space error | Samples | Improvement |
| --- | ---: | ---: |
| 0.10 px | 2,696 | +12.71% |
| 0.20 px | 10,905 | +13.16% |
| 0.40 px | 43,829 | +13.40% |

These were nearest-coordinate distances, not registration errors. The corrected
field is named `projection_error_bins` and measures only numerical inverse-warp
residual.

## Historical natural crop (invalidated)

The reproducible wire/railing crop from `L16_04364` is:

```text
reference crop: 1400,700,1200,1200
contributors: B4, B1
canvas scale: 1
```

The old evaluator reported an 11.84% improvement on 8,792 samples. Re-run this
crop with the corrected fixed-site protocol before interpreting that number.
The visual observations remain useful as historical artifact inspection, but
they do not validate the old loss calculation.

## Resources

| Run | Peak RSS | Total time/MP | Synthesis time/MP |
| --- | ---: | ---: | ---: |
| `L16_00563`, 11 modules, native | 1,949 MiB | 8.91 s | 0.96 s |
| `L16_00035`, 11 modules, native | 2,113 MiB | 11.38 s | 1.83 s |
| `L16_00054`, 11 modules, native | 2,020 MiB | 10.60 s | 2.21 s |
| `L16_01732`, 10 modules, native | 1,935 MiB | 10.81 s | 1.91 s |
| `L16_04364`, 11 modules, native | 2,124 MiB | 10.94 s | 2.89 s |
| `L16_04364`, two-camera Joint-CFA 1.44 MP crop | 609 MiB | 10.63 s | 2.14 s |

The crop's total time/MP includes fixed full-sensor alignment and is not a
throughput estimate. Comparing matched crop synthesis, Joint CFA takes 3.1 s
versus 1.3 s for MultiCamera (about 2.4x). Full many-camera runs already exceed
2 GiB because of the surrounding alignment pipeline, not because Joint CFA
allocates a full-frame reconstruction state. On-camera work therefore needs a
separate alignment-memory pass even if this experiment advances.

## Reproduction

```bash
chiaro-fuse capture.lri -o validation.png \
  --resolution-reconstruction multi-camera \
  --cfa-held-out B2 \
  --canvas native

chiaro-fuse capture.lri -o joint-crop.png \
  --camera B4 --camera B1 \
  --resolution-reconstruction joint-cfa \
  --crop 1400,700,1200,1200 \
  --canvas 1
```

Repeat `--camera` to test fixed subsets and `--cfa-held-out` to validate more
than one non-reference Bayer module. Monochrome or absent held-out modules fail
with an explicit error.

## Current decision

The historical quality conclusion is withdrawn pending corrected full-capture
reruns. Keep `joint-cfa` explicitly experimental and keep `MultiCamera` as the
default. Use fallback-inclusive `overall`, identical `sample_ids`, and
same-population `common_region` results for any new decision.
