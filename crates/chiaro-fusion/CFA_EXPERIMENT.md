# Real-capture cross-camera CFA experiment

Status: promoted for standard fusion, updated 2026-09-06. `JointCfa` is the
production default with local MultiCamera fallback. Night fusion remains on
explicit `MultiCamera` pending temporal-noise work.

> Tables explicitly labeled historical are pre-correction results. Their
> populations changed with solver coverage and some predictions were evaluated
> at different spatial locations, so they must not be used as quality or
> camera-count evidence. Corrected fixed-site results are reported separately.

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

The Joint-CFA estimate is feathered over the production baseline only where
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

## Corrected four-B phase, structure, and SNR diagnostic

The corrected evaluator was run on `L16_04364` with reference B4,
contributors B4+B1+B3+B5, held-out B2, crop `1400,700,1200,1200`, and canvas
scale 1. All 22,295 fixed physical B2 sites were retained; the aggregate result
reproduced the corrected four-B measurement exactly: baseline 5.066304, emitted
Joint-CFA/fallback 4.912339, or +3.039%.

Each fixed site now records phase, held-out signal/noise, reference-only
structure, both predictions and losses, and solver support. Structure is the
reference B4 luminance/chroma structure measure and never uses Joint-CFA
success, application weight, or confidence. Low/medium/high are fixed tertiles
of the held-out population:

- structure: `<0.0324933`, `0.0324933..0.0636651`, `>=0.0636651`;
- R SNR: `<28.4932`, `28.4932..38.1266`, `>=38.1266`;
- Gr SNR: `<28.8304`, `28.8304..35.7753`, `>=35.7753`;
- Gb SNR: `<29.1747`, `29.1747..39.0804`, `>=39.0804`;
- B SNR: `<28.1600`, `28.1600..37.0249`, `>=37.0249`.

SNR tertiles are phase-specific, so “low B SNR” means the weakest third of
valid B measurements rather than applying a threshold learned from green.
Intervals below are 95% delete-one-block jackknife intervals over physical
64x64 held-out-sensor blocks.

### R

| Structure | SNR | Sites | Baseline | Joint/fallback | Improvement | 95% block CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | Low | 263 | 2.523428 | 2.386505 | +5.426% | +2.089% .. +8.763% |
| Low | Medium | 213 | 2.321565 | 2.226937 | +4.076% | +0.977% .. +7.175% |
| Low | High | 1,462 | 1.821477 | 1.798995 | +1.234% | +0.265% .. +2.204% |
| Medium | Low | 793 | 2.911313 | 2.701508 | +7.207% | +1.783% .. +12.630% |
| Medium | Medium | 778 | 3.369572 | 3.020786 | +10.351% | +6.731% .. +13.971% |
| Medium | High | 174 | 5.214731 | 5.004044 | +4.040% | -1.403% .. +9.484% |
| High | Low | 797 | 5.926080 | 5.239318 | +11.589% | +7.668% .. +15.509% |
| High | Medium | 861 | 8.555607 | 7.725791 | +9.699% | +6.712% .. +12.686% |
| High | High | 217 | 11.407861 | 10.437283 | +8.508% | +3.594% .. +13.422% |

### Gr

| Structure | SNR | Sites | Baseline | Joint/fallback | Improvement | 95% block CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | Low | 260 | 3.639756 | 3.566903 | +2.002% | -0.041% .. +4.044% |
| Low | Medium | 328 | 1.938450 | 1.861565 | +3.966% | +1.433% .. +6.500% |
| Low | High | 1,207 | 2.544834 | 2.494255 | +1.988% | +1.064% .. +2.911% |
| Medium | Low | 848 | 3.671557 | 3.533129 | +3.770% | +1.578% .. +5.963% |
| Medium | Medium | 801 | 2.812366 | 2.704026 | +3.852% | +0.115% .. +7.589% |
| Medium | High | 274 | 3.850733 | 3.486854 | +9.450% | +3.722% .. +15.177% |
| High | Low | 746 | 6.087818 | 5.564298 | +8.599% | +5.865% .. +11.334% |
| High | Medium | 725 | 8.314265 | 7.699086 | +7.399% | +4.757% .. +10.042% |
| High | High | 373 | 19.431524 | 18.424273 | +5.184% | +3.158% .. +7.209% |

### Gb

| Structure | SNR | Sites | Baseline | Joint/fallback | Improvement | 95% block CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | Low | 225 | 2.821683 | 2.753312 | +2.423% | +0.603% .. +4.243% |
| Low | Medium | 242 | 2.812471 | 2.693837 | +4.218% | +1.736% .. +6.701% |
| Low | High | 1,406 | 1.752197 | 1.741376 | +0.618% | -0.026% .. +1.261% |
| Medium | Low | 871 | 3.349232 | 3.248222 | +3.016% | +0.534% .. +5.498% |
| Medium | Medium | 776 | 3.531574 | 3.353907 | +5.031% | +2.021% .. +8.040% |
| Medium | High | 185 | 5.491193 | 5.155150 | +6.120% | +1.695% .. +10.545% |
| High | Low | 793 | 7.505491 | 6.915515 | +7.861% | +5.810% .. +9.911% |
| High | Medium | 870 | 9.782797 | 8.924649 | +8.772% | +6.839% .. +10.705% |
| High | High | 298 | 23.744897 | 23.024891 | +3.032% | +1.520% .. +4.545% |

### B

| Structure | SNR | Sites | Baseline | Joint/fallback | Improvement | 95% block CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Low | Low | 283 | 2.513613 | 2.506278 | +0.292% | -4.350% .. +4.934% |
| Low | Medium | 346 | 2.818280 | 2.953326 | -4.792% | -10.850% .. +1.266% |
| Low | High | 1,197 | 2.186000 | 2.277572 | -4.189% | -8.090% .. -0.288% |
| Medium | Low | 835 | 3.452078 | 4.085292 | -18.343% | -26.131% .. -10.555% |
| Medium | Medium | 787 | 4.115005 | 4.685440 | -13.862% | -21.454% .. -6.271% |
| Medium | High | 309 | 6.359450 | 6.908192 | -8.629% | -15.549% .. -1.708% |
| High | Low | 718 | 5.469912 | 6.278006 | -14.773% | -30.107% .. +0.560% |
| High | Medium | 703 | 9.899695 | 10.270939 | -3.750% | -7.393% .. -0.107% |
| High | High | 331 | 13.805543 | 13.377050 | +3.104% | -0.677% .. +6.885% |

The B regression does **not** persist in high-structure/high-SNR B sites: that
cell improves by 3.10%, although its interval includes zero. Low-SNR B sites
account for 55.3% of the total B excess loss, medium-SNR sites for 37.8%, and
high-SNR sites for only 6.9%. Medium-structure sites account for 57.3% of the
excess, high-structure sites for 35.0%, and low-structure sites for 7.7%.
Therefore the regression is primarily a weak/medium-blue-signal problem, not a
smooth-sky-only problem; high-structure B still regresses when SNR is low or
medium.

## Independent C-contribution support analysis

C4 was completely held out while comparing B4+B1+B3+B5 against +C5, +C6, and
+C5+C6. This is a leave-one-C-camera-out test: C4 supplied 8,627 fixed physical
measurements and never entered reconstruction, profile selection, crosstalk,
highlight recovery, or scene photometric fitting. The physical IDs, phases,
measurements, noise sigmas, and independent reference-structure values matched
bit-for-bit across all four runs. No in-sample contributor residual is used.

Structure bands are C4-population tertiles: low `<0.0249925`, medium
`0.0249925..0.0526623`, and high `>=0.0526623`. “Changed” means that the emitted
prediction moved by at least 0.25 held-out C4 sigma relative to the B-only
output; `|change|/sigma` reports the mean absolute movement. Improvement compares
the paired B-only Joint-CFA/fallback loss with the C-enabled Joint-CFA/fallback
loss at the same C4 sites. Intervals use the same 64x64-block jackknife.

| C input | Structure | Sites | Changed | Mean change/sigma | B-only loss | With-C loss | Improvement | 95% block CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C5 | All | 8,627 | 58.5% | 1.822 | 41.810169 | 41.144896 | +1.591% | +0.231% .. +2.952% |
| C5 | Low | 2,876 | 64.8% | 0.958 | 38.592181 | 39.496341 | -2.343% | -2.733% .. -1.953% |
| C5 | Medium | 2,875 | 53.8% | 1.587 | 33.089551 | 32.259684 | +2.508% | +0.228% .. +4.788% |
| C5 | High | 2,876 | 56.9% | 2.921 | 53.745743 | 51.675574 | +3.852% | +1.330% .. +6.374% |
| C6 | All | 8,627 | 55.6% | 2.663 | 41.810169 | 40.458739 | +3.232% | +0.964% .. +5.501% |
| C6 | Low | 2,876 | 66.5% | 3.407 | 38.592181 | 37.948745 | +1.667% | -0.861% .. +4.196% |
| C6 | Medium | 2,875 | 47.2% | 1.590 | 33.089551 | 32.576750 | +1.550% | -0.546% .. +3.645% |
| C6 | High | 2,876 | 53.1% | 2.991 | 53.745743 | 50.847980 | +5.392% | +0.838% .. +9.945% |
| C5+C6 | All | 8,627 | 72.6% | 3.250 | 41.810169 | 39.685628 | +5.081% | +2.383% .. +7.779% |
| C5+C6 | Low | 2,876 | 79.8% | 2.531 | 38.592181 | 38.508036 | +0.218% | -1.464% .. +1.900% |
| C5+C6 | Medium | 2,875 | 67.4% | 2.430 | 33.089551 | 31.951934 | +3.438% | +0.525% .. +6.351% |
| C5+C6 | High | 2,876 | 70.6% | 4.790 | 53.745743 | 48.594226 | +9.585% | +4.139% .. +15.031% |

| Inputs | Solver-supported C4 sites | Coverage | Change from B-only |
| --- | ---: | ---: | ---: |
| B only | 6,249 | 72.44% | — |
| +C5 | 6,576 | 76.23% | +327 sites |
| +C6 | 6,249 | 72.44% | net zero (one gained, one lost) |
| +C5+C6 | 6,576 | 76.23% | +327 sites |

C-induced changes improve prediction specifically in high-structure regions:
C5 gives +3.85%, C6 +5.39%, and both together +9.59%, with all three block
intervals above zero. The combined low-structure result is statistically
neutral, while C5 alone degrades that band. C5 expands solver support; C6's
benefit comes mainly from changing estimates at already-supported sites. These
results are evidence that the C changes contain useful structured information
for an independent C sensor. They do not establish a B/C PSF or bandwidth
model, nor imply that every C-induced change should predict B2 better.

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

### CFA-14 early structure gate

Ordinary Joint-CFA rendering now computes the existing reference-only
luminance/chroma application gate before gathering contributor observations.
Locations with an exactly zero application weight retain the MultiCamera
baseline without running the local solver. Held-out validation still forces its
fixed sample locations through the solver, and `--joint-cfa-solve-flat`
explicitly restores the old all-pixel diagnostic behavior. Reports distinguish
candidate locations, actual solver attempts, and structure-gated skips.

Matched four-B runs on `L16_04364` used B4+B1+B3+B5 and canvas scale 1. Times
are single-run internal synthesis timings, so the avoided-solve fractions are
the more stable measure of work reduction:

| Crop | Reference window | Pixels | Solves skipped | Diagnostic synthesis | Gated synthesis | Observed change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Flat sky | `1720,850,256,256` | 65,536 | 17.19% | 4.1 s | 3.5 s | -14.6% |
| Edge-rich stairs | `1900,1100,256,256` | 65,536 | 2.10% | 8.5 s | 8.5 s | 0.0% at 0.1 s precision |
| Mixed held-out crop | `1400,700,1200,1200` | 1,440,000 | 4.48% | 72.8 s | 63.4 s | -12.9% |

Each gated PNG was byte-identical to its `--joint-cfa-solve-flat` counterpart.
The mixed PNG also matched the pre-CFA-14 artifact (SHA-256
`e588434bc59f52a495544db672117d4d157c4419ed6fb343f13f52802bc34e4f`).
Its corrected B2 population and result stayed unchanged: 22,295 physical sites,
baseline 5.066304, Joint-CFA/fallback 4.912339, +3.039%, with 14,918
solver-supported validation samples.

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

# Restore pre-gate flat-region solves for solver diagnostics or timing A/Bs.
chiaro-fuse capture.lri -o joint-crop-diagnostic.png \
  --camera B4 --camera B1 \
  --resolution-reconstruction joint-cfa \
  --joint-cfa-solve-flat \
  --crop 1400,700,1200,1200 \
  --canvas 1
```

Repeat `--camera` to test fixed subsets and `--cfa-held-out` to validate more
than one non-reference Bayer module. Monochrome or absent held-out modules fail
with an explicit error.

## Current decision

Promote `joint-cfa` to the standard-fusion default. The corrected fixed-site
results show a four-B overall improvement, isolate the remaining B weakness to
lower-SNR measurements, and independently validate useful structured C detail.
Keep `MultiCamera` explicitly selectable and retain it as the Night default
until merged temporal noise/provenance reaches the Joint-CFA observation model.
Continue using fallback-inclusive `overall`, identical `sample_ids`, and
same-population `common_region` results for future validation.
