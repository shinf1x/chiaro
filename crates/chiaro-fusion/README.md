# Chiaro fusion library

Multi-camera alignment and high-resolution synthesis for Light L16 captures.
This crate powers Chiaro Gallery's fused export and the `chiaro-fuse`
command-line application.

## Pipeline

1. Decode each participating RAW module and optionally run the shared
   hot-pixel, camera-specific cleanup, and corner-glow correction pipeline.
   A `.chiaro-cleanup` archive is validated against `hotpixel.rec`, opened once,
   and its selected camera entries are applied before highlight analysis or
   alignment. Detect near-clipped CFA
   measurements, reconstruct small edges locally, and use a clipping-aware RGB
   pyramid for larger low-confidence regions. One bit of fractional RAW
   precision is reserved as radiometric headroom above sensor white.
2. Resolve the factory camera model, project each module into a reference view,
   refine the global alignment with image correlation, build a dense calibrated
   inverse-depth cost field, and regularise coarse search hypotheses with
   edge-aware eight-direction semi-global matching. A finer 4-pixel grid then
   independently remeasures every accepted node with small or adaptive
   bilateral support; coarse values and holes are never copied into the final
   map. Each camera subsequently refines the shared depth continuously and
   applies a bounded residual correction.
3. When requested, blend a low-confidence spatial highlight estimate toward a
   donor field only when at least two accepted, aligned modules provide
   consistent unclipped RAW radiance. Donors are regularised within each CFA
   phase and feathered at coverage boundaries; per-channel overlap ratios
   account for exposure/transmission differences. Treat the factory 17x13
   four-phase crosstalk mesh as a prior and, in the default adaptive mode, fit
   five small residual modes from smooth aligned overlap. The fit is performed
   in the capture's white-balance domain, tested on held-out observations, and
   falls back independently to the factory mesh for any module without enough
   evidence or a measurable validation improvement. Then reconstruct Bayer
   colour with the selected demosaicing method and apply
   module-specific colour and flat-field calibration. By default, Chiaro uses
   the original D65 factory matrix with the capture's recorded white balance.
   The experimental CCT-only mode interpolates between factory A, F11, and D65
   anchors. The experimental array-aware mode additionally samples unclipped,
   sufficiently bright and geometrically reliable aligned overlap, rejects
   depth boundaries and unstable structure, and searches the 5%-step
   A/F11/D65 simplex for the common profile blend with the lowest robust
   inter-module chroma disagreement. This sparse search does not run complete
   fusion for each candidate. It falls back to the CCT prior when
   sample/module/spatial coverage is insufficient or the score surface is too
   weak to support an override. It also reports per-profile chroma
   distributions, D65-relative ratios, and chroma-normalized disagreement so
   contraction bias can be evaluated without affecting selection. Match overlapping
   modules photometrically, and blend them into a 16-bit PNG. AMaZE is the
   default; Simple, RCD, LMMSE, and IGV are also available. Warp
   discontinuities are not interpolated across visible scene edges, and a
   reference-guided robust weight rejects contradictory edge samples without
   discarding agreeing high-resolution samples. Luminance and chroma are
   weighted independently, preventing a defocused colour fringe with plausible
   total brightness from entering the result. Fine structure remains
   reference-anchored unless another module reproduces its direction more
   sharply; that agreeing zoom module may then own the high-frequency detail.
   Strong near-side residual parallax relative to a magnified module's
   calibrated focus plane suppresses that module locally. Centre-surround
   contrast preserves thin branches and wires even when their centres have
   little directional gradient. In multi-camera resolution mode, a separate
   locally verified warp matches each source at reference-camera bandwidth,
   retrieves the original physical samples, and selects the finest locally
   available optical tier. Edge-aligned compact Hann kernels form two detail
   bands; agreeing sources bias toward the strongest reliable coefficient.
   A denser tele module may transfer detail directly, while same-resolution
   reconstruction requires distinct subpixel phases. Colour and low-frequency
   tone remain on the robust fusion path. A reference-only luminance/chroma
   structure gate rejects guaranteed-zero Joint-CFA updates before gathering
   contributor observations; held-out validation and an explicit diagnostic
   option can still request flat-region solver statistics. A module rejected
   from ordinary fusion may still contribute resolution-only islands when this
   independent local registration is strong. For display-ready output, a smooth
   sensor-white shoulder neutralises false colour from unequally clipped raw
   channels without introducing a hard highlight boundary.

Every run also writes a `.fusion.json` report with alignment, RAW highlight
confidence/counts, cleanup availability and correction statistics, per-module
crosstalk fit/validation measurements, the CCT prior and selected colour-profile
weights, best/runner-up array scores, evidence coverage and confidence,
report-only forced-profile chroma distributions and normalized disagreement,
coverage, photometric, and timing diagnostics.

The CCT-only and array-aware selectors and forced A/F11 modes are retained as
experimental diagnostics. On the initial four-capture validation set, F11
slightly reduced median chroma relative to D65 in every capture, while its
inter-module disagreement advantage became small or reversed after normalizing
by scene chroma. A increased both chroma and disagreement substantially. Thus
none of the experimental choices has yet demonstrated a consistent improvement
over D65, so the original fixed-D65 path remains the production default pending
a broader calibrated corpus.

## Calibration

Captures embed part of the camera model and take priority when their calibration
is newer. Device `calibration.lri` and `zoom_calib_v0.lri` files fill important
gaps, including mirror-aiming data. Supply both whenever possible; alignment is
likely to be poor without them. An overlay is merged only when its physical
device id matches the capture.

Focus-dependent intrinsics and object-space focus distance are interpolated in
lens Hall space and continued linearly just beyond the factory samples,
matching the validated reconstruction model. Capture autofocus success,
disparity/contrast estimates, ROI, and actuator timeouts are retained in the
report. Image evidence remains authoritative: autofocus success describes the
selected focus plane, not whether every scene depth is sharp. The CLI's
diagnostic `--intrinsics clamp` mode freezes out-of-range captures at the
nearest sample instead.

`hotpixel.rec` is optional at the fusion API level, but when enabled it must
belong to the same physical camera. A cleanup profile is also optional and can
only be supplied as part of that hot-pixel stage because its manifest is tied
cryptographically to the factory map.

## Output modes

- **Native** renders approximately the reference sensor's 13 MP resolution.
- **Maximum** uses the finest participating module that covers the view, capped
  by a caller-provided megapixel limit. The applications default this cap to
  82 MP so the measured A/B magnification is not truncated below the L16's
  approximately 81.6 MP wide-output class.
- **Scale** specifies canvas pixels per reference pixel directly.

The output is cropped to the focal length framed by the photographer by
default. Full-reference rendering can be requested instead. Monochrome modules
contribute luminance unless explicitly excluded.

Resolution reconstruction is classical and deterministic: it uses no learned
model or GPU. Local registration renders one half-resolution matching buffer
at a time, and synthesis streams narrow output bands, so an 82 MP canvas does
not require an 82 MP floating-point output allocation.

`ResolutionReconstruction::JointCfa` is the standard-fusion default. It retains
physical CFA phase, sensor position, calibrated noise, highlight provenance,
geometry confidence, and camera identity, then solves a robust local-affine
D50 XYZ field directly from corrected mosaic observations. The affine terms
are important: a constant-colour neighbourhood would reduce error by blurring
fine detail. The solver is tileable and does not retain a frame-sized normal
system. Unsupported locations retain the production MultiCamera fallback;
`ResolutionReconstruction::MultiCamera` remains explicitly selectable.

`FusionOptions::cfa_held_out` provides the primary validation path. Named
non-reference cameras remain available for geometry but are excluded from all
reconstruction. Sparse output locations are projected into each held-out
sensor and compared with its real, measured, unclipped CFA sites in units of
calibrated noise. Reports retain fixed physical sample IDs and deterministic
site-level phase, reference-structure, SNR, prediction/loss, and solver-support
diagnostics for the MultiCamera baseline and joint solver. Resource diagnostics
include peak resident memory, runtime per output megapixel, observations/cameras
per solve, iterations, and physical sampling-phase spread.

## Important limitations

- Alignment uses a global homography followed by classical dense multi-view
  reconstruction. Textureless or contradictory areas remain explicit
  global/far fallback rather than receiving spatially completed depth. Small
  disconnected finite-depth islands are rejected as chance correlations; the
  filter never grows a measured surface into an unsupported region. A finite
  label must also improve measurably on the fitted global warp, preventing a
  shallow distant-scene cost curve from being reported as physical depth.
  The default finite search spans 0.5 m to 10 km so distant landscape detail
  is not collapsed onto a 100 m boundary.
  Per-camera consistency either applies a directly supported finite surface,
  retains the global warp, or suppresses an occluded view. Robust
  synthesis prevents most contradictory samples from producing double edges,
  but it cannot recover detail that moved differently in every exposure.
- Correlation needs useful scene texture, and the photometric matcher needs
  tonal range. Night-sky captures may retain visible module boundaries; use
  per-camera Hotpixel output and a stacker for astrophotography.
- Factory geometry alone is not sufficiently accurate for normal output;
  disabling correlation refinement is intended mainly for diagnostics.
- Spatial highlight reconstruction cannot recover true colour or texture where
  every local raw channel is saturated. Multi-camera mode can recover such
  samples only inside reliable overlap where at least two unclipped modules
  agree. The final smooth shoulder remains as a neutral fallback.
- Array-aware profile selection can identify the factory blend that makes the
  participating modules agree; it cannot prove absolute scene colour from an
  unknown spectrum. CCT/AWB therefore remains a soft plausibility prior and the
  fallback whenever aligned evidence is sparse or ambiguous.
- Capture-adaptive crosstalk estimates only a strongly regularised residual on
  the supplied factory mesh. The reference module remains the colour anchor,
  because one scene cannot identify its absolute error. Capture gain, exposure,
  and AWB are recorded for analysis. Colour conversion does select among the
  explicitly labelled A/F11/D65 ForwardMatrix profiles, but crosstalk does not
  invent or select undocumented matrix families.
  Spatial leakage kernels are intentionally deferred until residual diagnostics
  demonstrate leakage beyond the existing phase matrix.

## API and diagnostics

`pipeline::fuse` accepts an in-memory LRI, `FusionOptions`, an output path, and
a progress callback. The processing stages use plain data structures so
alignment and synthesis can also be inspected independently.

`chiaro-color-profile` exports all factory and `gold_cc` colour records and
performs leave-one-patch-out CIEDE2000 evaluation of the supplied ForwardMatrix,
a robust linear fit, and a strongly regularized quadratic candidate. Candidate
fits remain diagnostic unless held-out accuracy, neutrals, inter-module
agreement, and real-image stability all improve.

With a debug directory, `<camera>_highlight-uncertainty.png` marks recovered
RAW samples by inverse confidence in addition to the depth, alignment, and
source-ownership diagnostics.

Set `FusionOptions::debug_dir` to write per-module alignment checkerboards.
Continuous scene edges across checkerboard boundaries are a quick visual check
of the resulting warp. The same directory receives `source-luminance-ownership.png`
and `source-color-ownership.png`; their camera-to-colour legend and exact owner
fractions are recorded under `synthesis.source_contributions` in the JSON
report. It also receives `depth-inverse.png`, `depth-visualization.png`, and
`depth-provenance.png`. The first is quantitative 16-bit inverse depth; the
second is a log-scaled far-blue to near-red rendering.
The provenance image marks directly remeasured nodes green, a regularized
finite node amber if one is explicitly supplied, global/infinite fallback blue,
and unsupported nodes black. With default settings, finite final nodes are
green: SGM proposes where to search but cannot create final depth by itself.

See [Chiaro Fuse](../../apps/fuse/README.md) for command-line usage.
See [the real-capture CFA experiment](CFA_EXPERIMENT.md) for the current
held-out measurements, ablations, resource costs, and production decision.

## Tests

```bash
cargo test -p chiaro-fusion
```
