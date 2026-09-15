# Repository inefficiencies and imaging review

Reviewed 2026-09-05, against commit `2ca7664`.

This review covers RAW decoding/correction, temporal stacking, alignment/depth,
multi-camera synthesis, PNG output, Gallery export/cache behavior, and release
automation. Generated protobuf bindings were inspected as metadata context, not
treated as hand-maintained duplication. No processing code was changed.

**Start with findings 1, 2, 3, 12, and 13:** these concern discarded sensor
measurements, highlight corruption, lost clipping provenance, parser panics,
and outputs being confused with other captures. Optimize after establishing
regression cases for those behaviors.

Labels:

- **Confirmed / reproduced:** exercised through current public Rust APIs with
  small synthetic inputs.
- **Confirmed / code:** the behavior follows from the implementation; its
  prevalence or visual magnitude on real captures has not been measured.
- **Opportunity:** a concrete improvement to evaluate, not a claimed measured
  speedup or guaranteed image-quality improvement.
- **P1:** correctness, information loss, or failed processing; **P2:** quality,
  performance, or reliability improvement; **P3:** maintenance/distribution.

## Imaging correctness and extractable information

### 1. Temporal stacking discards one of the two measured green lattices

**P1 — Confirmed / reproduced.**

`Mosaic::sample_channel` searches the 2×2 CFA for the *first* occurrence of an
RGB channel. Both greens have channel number 1, so every green lookup uses
only that first stride-2 lattice. `stack_mosaic_burst` uses this sampler for
both the reference and donors; `reference_mosaic_u16` does too.

An RGGB 8×8 mosaic containing 100 everywhere except a measured green value of
900 at `(x=2, y=3)`, with black 0 and white 1000, returns **0.1 instead of 0.9**
at that position. Thus even an integer-position reference lookup replaces an
available measurement with interpolation. Half the green sites—one quarter
of all Bayer sites—are never read directly by this stacking sampler. This
also makes the exported comparison reference an unreliable ground truth.

**Improve:** read the reference's actual CFA sample directly. Preserve the
two green phases explicitly in temporal sampling, or use a green reconstruction
that uses both lattices and preserves measured values at integer positions.
Validate identity warps using textured, unequal green phases for all four CFA
patterns; constant-color tests cannot expose this.

Evidence: [sampler](crates/chiaro-fusion/src/image.rs#L228),
[merge](crates/chiaro-stack/src/lib.rs#L443),
[reference conversion](crates/chiaro-stack/src/lib.rs#L795).

### 2. Reserving highlight headroom can turn maximum white into zero

**P1 — Confirmed / reproduced in debug; release consequence follows from arithmetic.**

`reserve_highlight_headroom` evaluates `(*sample + 1) / 2` as `u16`.
At 65535 the addition panics with overflow checks enabled; with ordinary
release overflow behavior it wraps to zero. Night fusion calls this function
on a normalized mosaic that can legitimately contain 65535. This makes a
bright or clipped night pixel a crash trigger or a black sample.

**Improve:** widen before rounding the division, e.g.
`((u32::from(*sample) + 1) / 2) as u16`. Check 0, 1, 65534, and 65535, with
monotonicity and debug/release agreement; 65535 should become 32768.

Evidence: [conversion](crates/chiaro-fusion/src/image.rs#L69),
[night caller](crates/chiaro-stack/src/fusion.rs#L285).

### 3. Night highlights are classified after merging, and headroom arrives too late

**P1 — Confirmed / code.**

Temporal fusion averages samples and clamps the result into `[0,1]`, stores
it across the entire `u16` range, then calls highlight reconstruction with
white set to 65535. There is almost no storage above that white point for
reconstruction. All-module night fusion reserves headroom only *after* this
spatial recovery. Single-frame fusion correctly reserves it beforehand.

The temporal merge also has no original clipping masks: it computes weights
from brightness residuals and noise alone. A clipped input averaged with a
lower input can fall below the clipping guard, so later classification cannot
identify it as a saturated lower-bound measurement. Conversely, an unclipped
shorter exposure can be rejected as a mismatch to the clipped reference.

**Improve:** classify each original frame using its own sensor white, retain
those masks through alignment, and normalize exposure before a saturation-aware
merge. Reserve headroom before recovery and preserve merged confidence into
cross-camera fusion. Validate a bright patch with clipped long exposures and
unclipped shorter exposures, plus an all-clipped control case.

Evidence: [temporal merge and recovery](crates/chiaro-stack/src/lib.rs#L443),
[late headroom](crates/chiaro-stack/src/fusion.rs#L285),
[single-frame ordering](crates/chiaro-fusion/src/pipeline.rs#L1048).

### 4. Black-subtracted samples are clipped before temporal averaging

**P2 — Confirmed / code.**

`sample_channel` applies `max(0.0)` immediately after black subtraction.
Near black, a pair of equally plausible measurements at `black−d` and
`black+d` therefore averages to a positive value instead of zero. This
destroys negative noise excursions before they can cancel and biases faint
signals upward. The effect matters particularly for night and scientific
stacking; unsigned RAW storage itself is not the problem.

**Improve:** use signed or floating-point black-subtracted values through
temporal estimation and defer clipping to the requested output conversion.
Validate the mean and variance of a synthetic dark sequence centered on the
black level, including per-channel and spatial residuals.

Evidence: [normalization](crates/chiaro-fusion/src/image.rs#L228),
[merge](crates/chiaro-stack/src/lib.rs#L443).

### 5. Sensor noise controls rejection, but does not supply averaging weights

**P2 — Confirmed / code; estimator improvement needs evaluation.**

The temporal merge uses `weight = (1 − residual² / (sigma² × variance))²`,
clamped to its supported interval. Noise variance determines how readily a
donor is accepted, but the accepted contribution is not weighted by inverse
variance. The reference always starts with weight 1. A noisier frame gets a
wider acceptance interval without the corresponding reliability penalty.

**Improve:** separate motion/inconsistency confidence from measurement
precision, and combine them in the averaging weight. Propagate exposure-gain
scaling and interpolation weights into variance. For fixed independent
weights, check the output against
`sum(w_i² × variance_i) / sum(w_i)²`; do not promise this formula fully models
signal-dependent robust rejection. Evaluate heterogeneous exposures/gains and
motion before changing the production estimator.

Evidence: [weights](crates/chiaro-stack/src/lib.rs#L475),
[variance model](crates/chiaro-stack/src/lib.rs#L767).

### 6. Effective-frame counts saturate at four and distort downstream noise estimates

**P2 — Confirmed / reproduced.**

The count image stores `weight_sum × 16384` in `u16`, capped at 65535.
The report computes `mean_effective_frames` from that capped image and derives
`dark_noise_variance` from the report mean. Six identical constant mono frames
with distinct frame indices and identity motion seeds were all accepted but
reported **3.999939 effective frames**.

Even below the cap, the sum of robust weights is not the usual effective sample
count `(sum w)² / sum(w²)`. The resulting scalar variance feeds cross-camera
`noise_confidence`, so this affects synthesis weights, not only diagnostics.

**Improve:** retain uncapped weight sums, squared-weight sums, and propagated
variance for computation. Quantize only the diagnostic image with an explicit
scale. Check 2-, 4-, 6-, and longer-frame identical bursts and unequal-weight
cases separately.

Evidence: [count encoding/report](crates/chiaro-stack/src/lib.rs#L491),
[noise confidence](crates/chiaro-stack/src/fusion.rs#L682).

### 7. Final highlight neutralization ignores whether color was successfully recovered

**P2 — Confirmed / code.**

`ModuleColor::to_xyz_clipped` blends all white-balanced channels toward their
median when any raw channel approaches sensor white. At or above white the
blend becomes complete. It receives RGB and sensor-white values, but no
highlight recovery confidence or donor provenance. With the default shoulder
enabled, successfully reconstructed bright color can consequently be
neutralized along with genuinely unresolved clipping.

**Improve:** carry recovery reliability into synthesis and apply the neutral
fallback to unresolved clipping. Keep a separate tone-compression policy for
valid above-white radiance. Test colored highlights reconstructed from reliable
donors alongside all-clipped neutral and colored patches.

Evidence: [shoulder](crates/chiaro-fusion/src/synth.rs#L190),
[source representation](crates/chiaro-fusion/src/synth.rs#L346),
[default](crates/chiaro-fusion/src/synth.rs#L98).

### 8. “Linear” fused output still clips recoverable range and gamut

**P2 — Confirmed / code; output-format opportunity.**

`ColorPipeline::apply` converts XYZ to linear RGB and clamps each channel to
`[0,1]`; the final PNG loop quantizes this to 16 bits. Thus linear export is
not an unrestricted scene-linear intermediate: negative out-of-gamut components
and above-range radiance are irreversibly discarded. The high-precision
internal reconstruction cannot be recovered from that PNG later.

**Improve:** keep the current finished-image mode, but offer an explicitly
scene-linear floating-point intermediate with documented primaries, exposure
scale, and white point. Alternatively provide a documented range-preserving
integer encoding. Validate round trips of above-white and out-of-gamut test
vectors. A lossless PNG codec does not make earlier clipping lossless.

Evidence: [color conversion](crates/chiaro-fusion/src/synth.rs#L280),
[quantization](crates/chiaro-fusion/src/synth.rs#L1037).

### 9. PNGs omit the metadata needed to distinguish their color interpretation

**P2 — Confirmed / code.**

The shared custom encoder writes IHDR, IDAT, and IEND. It has no color-profile,
transfer-function, or processing-metadata parameter. The same encoder carries
display RGB, linear RGB, and RAW mosaics, so the image file alone does not
declare the correct interpretation. A grayscale mosaic also lacks CFA,
black/white-level, and orientation metadata inside the PNG.

**Improve:** tag finished RGB with the appropriate color/transfer metadata;
give linear output an accurate profile. Use a RAW-capable container or a
versioned, inseparable sidecar for mosaics. Do not label camera-native mosaic
or uncalibrated RGB as sRGB merely to add a tag. Verify exported files through
an independent metadata reader and color-managed viewer.

Evidence: [PNG header/chunks](crates/chiaro-hotpixel-core/src/png16.rs#L215),
[Hotpixel output modes](crates/chiaro-hotpixel-core/src/pipeline.rs#L43).

### 10. Temporal subpixel observations are collapsed before resolution reconstruction

**P2 — Opportunity.**

Night processing bilinearly samples each temporal CFA frame onto one module's
reference mosaic. It passes only the merged mosaic onward to multi-camera
resolution synthesis. Original temporal sampling positions and observations
are then unavailable to that stage. Denoising benefits remain, but repeated
subpixel sampling cannot be used directly for later resolution reconstruction.

**Improve:** evaluate retaining selected temporal observations and their
warps/uncertainty for a joint reconstruction, or reconstructing a module's
resolution before collapsing its burst. Preserve the current conservative
path for moving or ambiguous regions. Compare shifted high-frequency targets
against zero-shift controls and moving scenes; extra output pixels alone are
not evidence of recovered detail. Bound memory with tiles or retained-frame
limits.

Evidence: [resampled temporal merge](crates/chiaro-stack/src/lib.rs#L443),
[merged-only module input](crates/chiaro-stack/src/fusion.rs#L285).

### 11. Small previews point-sample the sensor without scale-appropriate filtering

**P2 — Confirmed / code; visual impact needs measurement.**

`render_sampled` chooses one sensor location per output pixel and reconstructs
RGB from its local neighborhood. The sampled footprint does not grow when
reducing a 4160-pixel sensor to a small thumbnail. Fine repeating patterns and
small bright objects can alias or disappear depending on sampling phase.
This affects previews, not the exported full-resolution pipeline.

**Improve:** use a CFA-aware area reduction or suitable image pyramid before
thumbnail sampling. Retain sparse decoding where useful, but filter over the
actual downsampling footprint. Compare checkerboards, fine wires, and stars at
multiple preview sizes and subpixel phases; benchmark the USB/read tradeoff.

Evidence: [preview resize](crates/chiaro/src/lri/mod.rs#L1828),
[local RGB sampling](crates/chiaro/src/lri/mod.rs#L2044).

## Reliability, exports, and calibration

### 12. The fusion metadata parser panics on a truncated LRI

**P1 — Confirmed / reproduced.**

`LriMessages::parse` validates a declared block header, then directly slices
`data[message_start..message_start + message_length]`. The descriptor validates
the message against the *declared block*, not the supplied byte slice.
Passing just the first 32 bytes of an otherwise valid mock LRI caused an
out-of-bounds panic instead of an error. This parser is also used while probing
Gallery export metadata, so failure is not confined to a command-line export.

**Improve:** require the block and message ranges to fit the supplied buffer
before slicing. Return contextual errors and share checked block iteration
with the core parser. Test truncation at every header/message boundary and
valid headers with overstated lengths.

Evidence: [unchecked slice](crates/chiaro-fusion/src/calibration.rs#L72),
[descriptor contract](crates/chiaro/src/lri/mod.rs#L819),
[Gallery probe](apps/gallery/src/export/mod.rs#L136).

### 13. Export names and resume checks can confuse captures or accept incomplete results

**P1 — Confirmed / code.**

Gallery output names use only a sanitized filename stem. Captures named
`L16_00001.lri` from different folders/devices collide; sanitization can also
collapse distinct names. Fusion and Hotpixel skip based on PNG existence,
while Night checks existence of both PNG and report. None of these checks
matches source identity, settings, calibration hashes, or software version.
Skipped files are recorded as successful exports of the current capture.

PNG writing is atomic individually, but fusion reports are written afterward
with `fs::write`. An interruption between the two leaves a PNG that ordinary
fusion resume considers complete. Reusing an output directory after changing
demosaicing or calibration can similarly retain old output silently.

**Improve:** include a capture identifier in output naming and a processing
fingerprint in a completion manifest. Commit that manifest after all outputs
are complete; skip only a matching completed result. Test same-name captures,
changed settings, missing/corrupt reports, and interrupted writes. Preserve an
explicit overwrite policy for intentionally replacing an export.

Evidence: [stem](apps/gallery/src/export/mod.rs#L84),
[fusion resume](apps/gallery/src/export/fusion.rs#L499),
[night resume](apps/gallery/src/export/night.rs#L508),
[Hotpixel resume](apps/gallery/src/export/hotpixel.rs#L575),
[report write](crates/chiaro-fusion/src/pipeline.rs#L1558).

### 14. Manually supplied defect maps are not bound to the capture's physical device

**P2 — Confirmed / code.**

`HotpixelRec` retains a file hash, record indices, and dimensions, but no
validated device identity. `load_rotated_map` checks only the selected record
and dimensions. A cleanup archive is tied to that map's hash, which proves
the archive/map pairing but does not prove the map belongs to the capture.
Automatic Gallery calibration lookup does use device identity; manually
supplied files and the library entry points do not establish the same link.

An unrelated same-model map can therefore target valid detail while missing
the capture's actual defects.

**Improve:** retain a verified device-to-map-hash association when importing
calibration, and check it at processing boundaries. Where the opaque factory
format cannot establish provenance, report that status explicitly rather than
implying dimensions establish a match. Do not reject unknown provenance as a
mismatch without evidence.

Evidence: [map validation](crates/chiaro-hotpixel-core/src/hotpixel.rs#L27),
[fusion loading](crates/chiaro-fusion/src/pipeline.rs#L978),
[automatic device lookup](apps/gallery/src/gallery/calibration_cache.rs#L53).

### 15. Fusion cancellation stops between captures, not during expensive processing

**P2 — Confirmed / code.**

Gallery checks cancellation around capture loading and between job items.
Once `fuse` or `fuse_night` runs, callbacks only update progress; their return
type cannot request termination, and the processing options have no cancellation
token. A cancelled job can continue through depth, reconstruction, and encoding
for the entire current capture.

**Improve:** pass cooperative cancellation through stage boundaries and
long-running row/tile loops. Return a distinct cancelled result, clean up
partial output, and avoid recording success. Verify cancellation during RAW
preparation, depth, and synthesis, including the longest stage's response time.

Evidence: [Gallery fusion callback](apps/gallery/src/export/fusion.rs#L550),
[night callback](apps/gallery/src/export/night.rs#L567),
[library API](crates/chiaro-fusion/src/pipeline.rs#L915).

### 16. The PNG encoder's memory bound does not hold under backpressure

**P2 — Confirmed / code.**

Workers send compressed bands over an unbounded `mpsc::channel`; the writer
also stores out-of-order results in an unbounded `BTreeMap`. If disk writing is
slow, compressed bands accumulate in the channel. If an early render band is
slow, later bands accumulate in the reorder map. Memory can approach the
compressed image size, rather than a fixed number of worker bands as the
comments claim. At compression level 0 this is close to the raw output size.

**Improve:** bound the total number of issued-but-not-written bands, including
the reorder window. Merely replacing the channel with a bounded one does not
bound a receiver that continuously drains into `pending`. Test with a delayed
first band and a throttled writer; assert a high-water mark for outstanding
bytes/bands.

Evidence: [workers/channel](crates/chiaro-hotpixel-core/src/png16.rs#L171),
[reorder buffer](crates/chiaro-hotpixel-core/src/png16.rs#L215).

## Performance and duplicated work

### 17. Streaming output still retains large full-frame input/intermediate buffers

**P2 — Confirmed / code; memory-reduction opportunity.**

Advanced demosaicing stores one RGB16 cache per accepted color module while
keeping its mosaic. At 4160×3120 that is about **78 MB RGB + 26 MB mosaic per
module**, before confidence maps, alignment data, and scratch space. Ten such
color modules would consume about **1.04 GB** for those two representations
alone; actual usage depends on participating sensor types.

Temporal stacking retains every corrected mosaic and half-resolution `f32`
luminance plane until merging, approximately **39 MB per frame** at the same
dimensions. It also constructs and highlight-processes a full reference mosaic
even when all-module night fusion discards it. Streaming the output canvas
does not remove these costs.

**Improve:** make reference/comparison products optional, explicitly release
stage-only buffers, and evaluate tiled demosaic caches with correct halos or
an alignment pass followed by streamed temporal accumulation. Measure peak RSS
over module/frame counts; avoid simply parallelizing more full-frame work.

Evidence: [RGB cache](crates/chiaro-fusion/src/image.rs#L25),
[cache preparation](crates/chiaro-fusion/src/pipeline.rs#L1357),
[retained temporal frames](crates/chiaro-stack/src/lib.rs#L301),
[unconditional reference work](crates/chiaro-stack/src/lib.rs#L503).

### 18. Synthesis performs shared atomic updates inside the pixel/source loops

**P2 — Confirmed / code; speedup unmeasured.**

Source ownership and resolution diagnostics call shared `fetch_add` repeatedly
for each output pixel and contributing source. On a large canvas this produces
many contended updates to a small counter set. Other counters in the same
function already accumulate locally per band and update once afterward.

**Improve:** use band-local arrays for all diagnostic counters and reduce them
after rendering. Preserve exact integer totals and image bytes. Benchmark
native and maximum canvases with multiple worker counts to separate contention
from the actual reconstruction cost.

Evidence: [per-source atomics](crates/chiaro-fusion/src/synth.rs#L710),
[resolution atomics](crates/chiaro-fusion/src/synth.rs#L975),
[existing band-local reduction](crates/chiaro-fusion/src/synth.rs#L1044).

### 19. Thread limits do not cover all stages, and crosstalk preparation is serial

**P2 — Confirmed / code; optimization needs profiling.**

The fusion pipeline spawns a thread per alignment input, while depth chooses
workers from `available_parallelism` without a caller thread budget.
`FusionOptions::threads = 1` therefore does not constrain the whole operation.
The RAW row helpers also create fresh scoped OS threads for each invocation.

Meanwhile, `Mosaic::prepare_demosaic` performs its full-resolution four-phase
crosstalk correction in a serial nested loop before invoking the threaded
demosaicer. Each pixel computes interpolated planes and a spatial matrix.

**Improve:** use a shared execution budget/pool across stages and apply row
parallelism to measured serial bottlenecks. Document which settings limit
which stages until that is unified. Check total active workers and stage
timings, including small images where thread startup may cost more than work.

Evidence: [alignment workers](crates/chiaro-fusion/src/pipeline.rs#L1131),
[depth workers](crates/chiaro-fusion/src/depth.rs#L1204),
[thread helper](crates/chiaro-hotpixel-core/src/parallel.rs#L46),
[serial crosstalk](crates/chiaro-fusion/src/image.rs#L95).

### 20. Batch fusion reloads reusable calibration, and night fusion reparses the capture

**P2 — Confirmed / code; impact depends on workload.**

Every call to `fuse` reads/parses overlay files, opens/hashes `hotpixel.rec`,
and loads bundled correction models again. Gallery invokes it once per
capture. `fuse_night` already shares bundled models between its modules, but
each `stack_mosaic_burst` call reparses the full frame layout and merges cloned
noise profiles even though the outer function has parsed the layout already.

**Improve:** introduce a reusable processing context for immutable external
calibration/models and a prepared-capture structure for parsed metadata.
Keep capture-specific calibration precedence and device matching intact;
invalidate cached external inputs when their content changes. Benchmark cold
and warm batches separately. This is metadata/model duplication, not a claim
that each parse decodes every RAW payload.

Evidence: [fusion setup](crates/chiaro-fusion/src/pipeline.rs#L930),
[night layout](crates/chiaro-stack/src/fusion.rs#L145),
[per-module reparse](crates/chiaro-stack/src/lib.rs#L216),
[Gallery loop](apps/gallery/src/export/fusion.rs#L491).

### 21. RAW10 unpacking has separate implementations and uneven parallelism

**P3 — Confirmed / code.**

The same reversed five-byte-to-four-sample decoding is implemented in
`chiaro::lri::decode_packed10` and the Hotpixel core unpacker, with a third
random-access bit extraction path for previews. Temporal RAW10 decoding uses
the serial core-parser implementation, while single-frame correction uses the
threaded Hotpixel implementation.

**Improve:** put the common low-level group decoding and bounds conventions
in the lower-level crate, with scalar, random-access, and optional parallel
entry points. Keep optimized bulk and sparse interfaces where useful; they
need not allocate identically. Cross-check them against one byte-level corpus
with padding/orientation cases and malformed lengths.

Evidence: [parser unpacker](crates/chiaro/src/lri/mod.rs#L720),
[Hotpixel unpacker](crates/chiaro-hotpixel-core/src/raw10.rs#L53),
[preview sampler](crates/chiaro/src/lri/mod.rs#L1984),
[temporal decoder](crates/chiaro/src/lri/mod.rs#L616).

### 22. Ordinary and night fusion duplicate orchestration and have already diverged

**P3 — Confirmed / code.**

Both pipelines independently assemble calibration, module representations,
alignment, depth/resolution warps, crosstalk, demosaicing, photometric matching,
synthesis inputs, and reports. They also implement separate module-color and
alignment-confidence handling. Gallery repeats substantial transfer/progress,
job-loop, logging, and resume code across exporters.

The highlight-headroom ordering in finding 3 and PNG/report resume difference
in finding 13 are concrete examples of behavior drifting across these paths.
Night's D65 color helper is also separate from ordinary fusion's profile-blend
machinery; that is a feature difference to make explicit, not automatically a
reason to force identical defaults.

**Improve:** separate frame preparation from a shared prepared-module fusion
stage, and share export transport/completion handling. Keep temporal policies
and intentional color defaults explicit. Test the common invariants through
both front ends rather than merely consolidating similar-looking functions.

Evidence: [ordinary pipeline](crates/chiaro-fusion/src/pipeline.rs#L915),
[night pipeline](crates/chiaro-stack/src/fusion.rs#L124),
[night color helper](crates/chiaro-stack/src/fusion.rs#L691),
[Gallery exporters](apps/gallery/src/export).

### 23. Thumbnail identity duplicates capture identity with weaker invalidation

**P2 — Confirmed / code.**

`CaptureIdentity` hashes a local file's nanosecond modification time, but
`ThumbnailKey` independently hashes whole seconds. A same-size edit within
one second changes the catalog identity while retaining the old thumbnail
key. For a device without a serial number, capture identity falls back to
USB location while the thumbnail key uses the shared string `unknown-serial`;
same-name, same-size objects on such devices can collide in the thumbnail
cache. Cache load does not reject an indexed entry whose existing capture hash
differs from the requested one.

**Improve:** derive the source portion of a versioned thumbnail key from one
shared identity implementation, then add preview-source and rendering settings.
Invalidate or migrate the old cache format deliberately. Test same-size rapid
file edits and multiple devices lacking serial numbers.

Evidence: [thumbnail hashing](apps/gallery/src/gallery/cache.rs#L65),
[capture hashing](apps/gallery/src/gallery/database.rs#L29),
[cache lookup](apps/gallery/src/gallery/cache.rs#L430).

## Validation and distribution

### 24. Release archives omit the advertised Stack application

**P3 — Confirmed / code.**

The workspace and top-level README include four applications, but release CI
builds only `chiaro-gallery`, `chiaro-hotpixel`, and `chiaro-fuse`. Both Unix
and Windows packaging lists omit `chiaro-stack`. Downloaded archives therefore
do not contain the complete advertised application set.

**Improve:** include package `chiaro-stack-app` in the build and binary
`chiaro-stack` in both packaging paths. Maintain one checked application list
or validate archive contents against an explicit manifest. Smoke-test each
packaged CLI with `--help`.

Evidence: [workspace](Cargo.toml#L1), [application list](README.md#L11),
[release build](.github/workflows/release.yml#L59),
[Unix packaging](.github/workflows/release.yml#L80),
[Windows packaging](.github/workflows/release.yml#L119),
[Stack package](apps/stack/Cargo.toml).

### 25. Current automated coverage does not protect end-to-end information preservation

**P2 — Confirmed / code and review checks.**

The only checked-in GitHub workflow is release building/packaging; it does not
run tests. The repository has useful kernel and geometry tests, but all 137
reviewed library tests pass despite findings 1 and 2. The mock encoder assigns
every module `frame_index = 0`, including repeated physical modules. Existing
repeated-frame/cleanup tests verify frame counts and correction, but do not
prove temporal averaging: the merge skips donors whose frame index equals
the reference. New temporal fixtures must assign distinct indices.

**Improve:** add pull-request test coverage and small end-to-end invariants:
identity-warp sample preservation for each CFA phase, maximum-white arithmetic,
real averaging across distinct temporal indices, dark-mean preservation,
clipping provenance, and export interruption/resume. Retain the real geometry
fixture. Build a representative image-quality corpus for parameter changes,
including motion, fine detail, clipped color, dark scenes, mixed gains, and
calibration gaps; the tests here do not establish a real-scene quality ranking
between demosaicers or reconstruction policies.

Evidence: [workflow](.github/workflows/release.yml),
[mock frame index](crates/chiaro/src/mock.rs#L225),
[temporal tests](crates/chiaro-stack/src/lib.rs#L960),
[donor exclusion](crates/chiaro-stack/src/lib.rs#L456),
[geometry fixture](crates/chiaro-fusion/tests/geometry_fixture.rs).

## Checks performed and limits

- `cargo test -p chiaro-fusion -p chiaro-stack -p chiaro-hotpixel-core -p chiaro --lib --locked --offline`:
  **137 passed** (60 fusion, 10 stack, 53 Hotpixel core, 14 parser/mock).
- `cargo test -p chiaro-fusion --test geometry_fixture --locked --offline`:
  **4 passed**.
- A temporary Rust harness outside the repository called the current public
  APIs. It reproduced second-green sampling as `0.1` for a measured `0.9`, a
  debug panic for headroom conversion of 65535, a count of `3.999939` for six
  identical accepted constant frames with distinct indices and identity seeds,
  and a parser panic when a valid LRI was truncated to its 32-byte header.
  Panics were caught by the harness; they are findings, not failures of the
  existing test suites.
- No full-resolution real-capture quality study, peak-RSS benchmark, CPU
  profiling run, or Windows/macOS release execution was performed. Performance
  recommendations above identify work/allocation patterns to measure; none
  claims an observed speedup. Existing conservative depth, motion, and
  independent-donor gates should be retained unless evidence supports a change.

Suggested order: fix the reproduced corruption/panic paths and temporal
clipping provenance; make export completion trustworthy; then establish
estimator/quality fixtures before changing weighting or reconstruction.
Profile memory, atomics, serial crosstalk, and repeated setup on those same
fixtures before choosing the largest performance refactor.

## Experimental branch review: cross-camera-cfa-experiment

Resolved findings CFA-01 through CFA-10, CFA-12, and CFA-16 were removed after implementation and verification on 2026-09-06. The remaining findings below are intentionally deferred.

### CFA-11. One affine field does not model different optical footprints or curved edges

**P2 — Opportunity; not a demonstrated cause of the saved trend.**

The design row contains a camera color response and first-order spatial
offsets. It has no per-camera blur/pixel-integration model, and one affine
field spans the compact support. Near discontinuities and fine repeating
structure, differently focused or magnified cameras need not observe that
same first-order field. More measurements can increase model disagreement
without adding the kind of independent detail the model can express.

**Improve:** evaluate same-tier, matched-sharpness subsets before mixed optical
tiers, and inspect residuals by camera and edge orientation. Compare adaptive
support or an optical-footprint-aware observation model on controlled detail
fixtures. Measure the benefit against the current affine model; adding a more
complex solver without those controls could increase cost and artifacts.

Evidence: [affine design](https://github.com/shinf1x/chiaro/blob/df5c5c54a4ae24220bf0d87b9fb95002ba1da98d/crates/chiaro-fusion/src/cfa.rs#L240),
[fixed output support](https://github.com/shinf1x/chiaro/blob/df5c5c54a4ae24220bf0d87b9fb95002ba1da98d/crates/chiaro-fusion/src/synth.rs#L1704).

### CFA-13. Night Joint CFA has no propagated temporal noise model

**P2 — Confirmed / code.**

The new mode is available through the shared reconstruction enum, including
Gallery's Night selector. Night synthesis supplies `noise_model: None` for
every merged module. Joint CFA therefore falls back to a quantization floor
derived from normalized mosaic storage, rather than the merged burst's
signal-dependent variance and spatial effective support. The existing scalar
dark-noise confidence is not a substitute for the residual variance used by
IRLS. A stacked interpolated pixel is also not an original physical RAW site.

**Improve:** propagate merged variance and provenance maps into the joint
observation interface, with a representation flag separating physical and
temporally reconstructed inputs. Until validated, make Night support's limits
explicit or restrict that combination. Compare bursts with varying accepted
frame counts, motion fallback, and exposure/gain before interpreting confidence.

Evidence: [missing noise model](https://github.com/shinf1x/chiaro/blob/df5c5c54a4ae24220bf0d87b9fb95002ba1da98d/crates/chiaro-stack/src/fusion.rs#L570),
[shared Night selector](https://github.com/shinf1x/chiaro/blob/df5c5c54a4ae24220bf0d87b9fb95002ba1da98d/apps/gallery/src/export/night.rs#L337).

### CFA-15. The per-pixel solver repeatedly allocates and recomputes invariant work

**P2 — Confirmed / code; profiling opportunity.**

Each attempted pixel allocates an observation vector; the solver allocates
filtered references and a closest-camera vector. Neighboring pixels repeatedly
visit the same physical sites, recomputing four-plane interpolation, crosstalk,
noise, gain fields, and a 3×3 inverse for each response row. Each IRLS iteration
recomputes design rows/base weights and all 81 normal-matrix entries although
the matrix is symmetric. New per-pixel shared atomic counters add to the
contention identified in main finding 18.

**Improve:** reuse worker-local scratch, precompute invariant design/base-weight
terms within a solve, accumulate one symmetric triangle, reduce diagnostics
per band, and evaluate bounded tile caches for corrected sites/response rows.
Cache keys must preserve spatial calibration changes. Validate numerical
agreement and measure allocation count, synthesis time, and peak RSS; do not
replace the tile-local design with an unbounded full-frame cache.

Evidence: [gathering/repeated response inversion](https://github.com/shinf1x/chiaro/blob/df5c5c54a4ae24220bf0d87b9fb95002ba1da98d/crates/chiaro-fusion/src/synth.rs#L1705),
[solver loop](https://github.com/shinf1x/chiaro/blob/df5c5c54a4ae24220bf0d87b9fb95002ba1da98d/crates/chiaro-fusion/src/cfa.rs#L105),
[new counters](https://github.com/shinf1x/chiaro/blob/df5c5c54a4ae24220bf0d87b9fb95002ba1da98d/crates/chiaro-fusion/src/synth.rs#L1189).
