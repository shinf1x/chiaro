# Real-capture cross-camera CFA experiment

Status: experimental, 2026-09-04. `MultiCamera` remains the production default.

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

Factory noise is propagated through the squared coefficients of the actual
local 4x4 crosstalk row and the flat-field gain. The solver is tile-local and
does not retain a frame-sized coefficient or normal-equation buffer.

## Primary held-out results

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

Every capture improves overall and in the high-frequency population. The
weakest case is `L16_00054`; its Gb phase still regresses by 0.51% even though
the other phases and aggregate score improve. This prevents a claim that every
phase is universally better.

## Camera-count and focal-tier ablation

`L16_04364`, reference B4, held-out B2:

| Reconstruction contributors | Kind | Overall improvement |
| --- | --- | ---: |
| B4 + B1 | 2, same B tier | +13.40% |
| B4 + B1 + B3 | 3, same B tier | +8.36% |
| B4 + B1 + B3 + B5 | 4, same B tier | +6.74% |
| B4 + B1 + B3 + B5 + C5 + C6 | 6, mixed B/C tiers | +6.82% |

Results are positive at every subset size but are not monotonic: the two-camera
pair is best, and both baseline and Joint CFA worsen as less compatible cameras
enter. This is evidence that contributor admission and residual
geometry/photometry need more work; camera count is not a quality proxy. The
mixed C-tier observations recover a small amount relative to four B cameras,
but do not recover the two-camera result.

For the two-camera case, tightening the maximum held-out mapping error preserves
the result:

| Maximum sensor-space error | Samples | Improvement |
| --- | ---: | ---: |
| 0.10 px | 2,696 | +12.71% |
| 0.20 px | 10,905 | +13.16% |
| 0.40 px | 43,829 | +13.40% |

The gain therefore does not depend on the loosest nearest-output matches.

## Natural crop

The reproducible wire/railing crop from `L16_04364` is:

```text
reference crop: 1400,700,1200,1200
contributors: B4, B1
canvas scale: 1
```

On 8,792 held-out B2 samples in this crop, Joint CFA improves prediction by
11.84%. The full Joint-CFA render solves 59.82% of output locations and applies
an average 81.0% structure confidence. Side-by-side inspection shows changes
concentrated on railings and other edges rather than the flat sky. No broad tone
shift or obvious block boundary is visible; amplified differences do reveal
small colour-edge changes, so a larger artifact corpus is still required before
production use.

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

The stop condition has not been met: Joint CFA repeatedly predicts unseen real
high-frequency CFA measurements better than the matched production baseline,
and the gain is larger than a tiny inconsistent fluctuation. However, it is not
ready to replace MultiCamera. Contributor count is non-monotonic, one phase in
the weakest capture regresses slightly, more held-out cameras remain to be
checked, and natural false-colour/moire inspection is still too small. Keep
`joint-cfa` explicitly experimental.
