# Capture-specific physical rig refinement

## Status

Chiaro contains an experimental capture-specific optimizer for the physical
L16 camera and mirror model. It is enabled by default. A candidate that
improves the physical fit and passes the physical safety checks replaces the
factory rig; the later image-space alignment comparison is diagnostic and no
longer overrides that decision.

The preferred observation source is now a physically guided, semi-dense
matcher. Legacy homography/RANSAC correspondences do not define the physical
matching locus; they are retained only as a deterministic fallback when the
physical matcher cannot provide enough usable tracks. The
provisional stage-2 warp supplies only a proposal for displacement
perpendicular to the physical epipolar locus; it never supplies a physical
observation or decides which depth is accepted.

The current matcher uses a **projected-motion-bounded inverse-depth
hierarchy**. Forty-eight inverse-depth samples provide coarse discovery, but
they are no longer the final depth quantization. For each reference location,
the calibrated rig measures the largest target-camera motion between adjacent
hypotheses. The search starts at the corresponding luminance-pyramid level and
refines a beam of separated depth modes until adjacent final hypotheses move by
at most 8 target-sensor pixels (subject to a six-level safety cap).

## Position in the pipeline

The intended data flow is:

```text
factory intrinsics/extrinsics + focus/mirror state
                    |
                    v
      resolved generalized-camera rig
                    |
                    v
 provisional measured residual alignment
 (perpendicular search proposal only)
                    |
                    v
  physical semi-dense matching and triangulation
                    |
                    v
       constrained capture-rig optimization
                    |
          physical safety/fit gate
                    |
                    v
 rerun residual image alignment from candidate rig
                    |
     residual-correction diagnostic
                    |
                    v
       depth, Joint CFA and final synthesis
```

For deterministic fallback and comparison, the implementation computes the
factory-seeded legacy alignments before attempting rig refinement. Those
alignments contribute only a perpendicular epipolar search proposal to the
preferred physical matcher. Their along-locus component is discarded so they
cannot choose scene depth, and their correspondences do not become physical
tracks. If the physical candidate passes its first gate, residual alignment is
rerun from the refined rig and that result continues downstream. The comparison
against the factory-seeded residual correction remains in the report as a
warning signal, but it is not ground truth and cannot restore the factory
cameras. Only failure of the physical solve itself retains factory cameras and
alignments.

Rig-refinement rejection does **not** make homography geometry the normal depth
model. Dense depth still prefers the resolved factory physical rig. The old
warp-seeded depth path is used only if physical depth cannot establish enough
usable calibrated views.

## 1. Inputs and evidence isolation

Each captured module contributes:

- its factory camera calibration;
- capture-specific focus and movable-mirror state;
- a half-resolution log-luminance plane;
- its name and whether its image may supply matching evidence.

A camera selected with `--cfa-held-out` keeps its calibrated geometry for
projection and later evaluation, but its pixels cannot influence physical
matching, rig fitting, or dense-depth evidence. A held-out camera cannot be the
reference camera. This prevents CFA validation data from leaking into the
geometry solution.

## 2. Semi-dense reference sampling

The reference image is sampled on a 32-full-resolution-pixel grid. Candidate
structure is measured over a 7x7 patch in the half-resolution log-luminance
plane. Patches below a log-luminance standard deviation of 0.008 are skipped.

At most 6,000 candidates are kept. If more are available, selection is thinned
uniformly in raster order to retain field coverage instead of retaining only
the strongest edges from one part of the image. Six thousand is therefore a
candidate budget, not the expected number of final tracks.

## 3. Physically guided depth and cross-camera matching

For every reference candidate, the matcher begins with 48 depths uniformly
spaced in inverse depth between 500 and 10,000,000 calibration units. On the
L16 calibration this spans the useful finite range through an effectively
distant hypothesis.

Forty-eight samples alone are too sparse for the high-focal-length C cameras:
on a wide baseline, adjacent planes can move a projection by tens of sensor
pixels. The matcher now evaluates that motion directly through every eligible
camera model. It chooses a box-filtered luminance-pyramid level where the
coarse samples remain discoverable, retains the six strongest separated depth
modes, and repeatedly inserts half-step inverse-depth samples around those
modes while moving to the next finer image level. At the native half-resolution
luminance level, adjacent final samples are required to differ by no more than
8 full-resolution target pixels unless the explicit six-level cap is reached.
Thus the nearest final hypothesis is normally within 4 target pixels of the
continuous physical locus.

The motion calculation uses only projections inside the actual target sensor.
This is stricter than the generalized camera's deliberately generous
half-frame distortion guard: an off-sensor point cannot provide a luminance
patch, and extrapolated distortion outside that domain must not force valid
image content onto an excessively coarse pyramid level.

At each depth, the reference ray defines a local fronto-parallel surface. The
surface centre and two nearby reference-raster offsets are projected through
each target's full `ResolvedCamera` model. This includes:

- camera pose and baseline;
- focus-dependent intrinsics;
- lens distortion;
- movable-mirror geometry;
- the local projection Jacobian.

The Jacobian maps the reference patch directly into the target raster. B/C
focal-length and magnification differences are therefore handled by physical
projection and local scale, not by assuming equal-sized patches or applying a
global 2x resize. The same mechanism handles any A/B/C pairing supported by
the calibrated cameras.

### Bounded residual search

Factory geometry defines the depth trajectory and remains the first search
centre. When that local search does not reach the 0.50 support threshold,
stage 2 supplies a fallback perpendicular proposal. For each depth and target camera, the
difference between the measured stage-2 mapping and physical projection is
decomposed into directions parallel and perpendicular to the physical
epipolar/depth locus. The parallel component is discarded because it contains
unknown scene parallax: applying it to every hypothesis would collapse all
depths onto the same measured pixel. Only the perpendicular component seeds
the residual search.

Around the active factory or fallback proposal, candidate offsets are searched
mainly perpendicular to the physical epipolar/depth locus:

- normal offsets: -32, -16, 0, +16 and +32 sensor pixels;
- tangential offsets: -4, 0 and +4 sensor pixels.

When the tangent is degenerate at an effectively distant depth, a bounded 5x5
two-dimensional search using -32, -16, 0, +16 and +32 pixels is used. This
allows orientation error to bootstrap without permitting an unrestricted warp
to absorb parallax. Equal-score candidates receive a small penalty for their
distance from the measured proposal rather than their distance from the
uncorrected factory projection.

Projected patches are compared with ZNCC. A target camera supports a depth only
when its score is at least 0.50. Nonmatching or occluded cameras are neutral;
they do not vote against a hypothesis.

### Multi-camera consensus

A depth requires at least three cameras including the reference. Its ranking
is the sum of positive evidence from all supporting target cameras:

```text
sum(max(0, ZNCC - 0.50) + 0.05)
```

The bounded support bonus lets broad independent support outrank a small set of
exceptionally strong accidental matches. The winning final hypothesis must
beat the best non-adjacent hypothesis by at least 3%. After depth selection,
every admitted target observation receives one final 3x3 localization
refinement at a 4-pixel step. Depth remains discrete in this bootstrap matcher,
but its local spacing is now set by projected image motion instead of the
original fixed 48-plane quantization.

Candidate processing is parallel but deterministic: the raster-ordered
candidates are divided into chunks and merged in the same chunk order. The
global `--threads` value is passed into this stage; for example, `--threads 8`
caps the physical matcher at eight workers.

## 4. Physical tracks and triangulation

All target observations supporting the selected shared depth are joined with
the reference observation into one track. Each observation retains:

- camera and sensor coordinate;
- ZNCC-derived confidence;
- reference-patch structure;
- local cross-focal scale;
- depth-margin reliability;
- an anisotropic localization covariance aligned with the epipolar tangent.

Before a track can reach the optimizer, it is triangulated under the factory
rig after subtracting the stage-2 perpendicular proposal from each bootstrap
observation. This prevents the pre-solve factory check from rejecting the
capture-specific displacement that the optimizer is intended to explain. The
original measured pixels are preserved and are the only coordinates used by
optimization and held-out evaluation. The proposal-normalized track must
satisfy all of the following:

- at least three retained cameras;
- positive depth in every participating camera;
- maximum pairwise ray angle of at least 0.003 degrees;
- triangulation condition number no greater than `1e9`;
- initial track RMS no greater than 100 pixels;
- every retained observation within 12 reference-equivalent pixels of the
  repeatedly retriangulated point.

If one target observation violates the last condition, the worst non-reference
observation is removed and the point is retriangulated. The reference is never
removed. The process repeats until the track is consistent or fewer than three
views remain. Thus the bounded residual search cannot create a valid track
solely from unrelated image peaks.

This is a factory-rig bootstrap gate, not a post-optimization gate. The report
therefore records both the +/-32 target-pixel local radius around the active
factory/stage-2 proposal and the 12-reference-pixel pre-solve consistency
threshold. The depth sweep itself can
move a prediction by hundreds of pixels, so displacement from the infinity
projection is not evidence that these bounds were exceeded. Only a residual
outside the correct depth-conditioned prediction indicates that the affected
camera cannot bootstrap itself into the optimizer.

The physical population is used only if it contains at least 80 usable tracks
and 20 validation tracks after the spatial split. Otherwise the optimizer receives the finest reliable legacy
correspondence population as a deterministic fallback. The JSON report states
which source was actually used.

## 5. Fit/validation split and what held-out RMS means

Tracks are always split by deterministic hashes of 256x256 full-resolution
reference blocks. The validation fraction is 20%, and all tracks in one block
go to the same side. Validation tracks never contribute to parameter
observability, optimization, or fit-track membership updates.

The reported held-out RMS does require a depth, but it does not use a depth map
estimated from the fit region. For each held-out multi-view track and for each
rig being evaluated:

1. triangulate that track's nuisance 3D point from its own fixed observations;
2. project the point into every camera in that track;
3. measure the multi-view reprojection residual.

This is a spatially held-out **geometric self-consistency** measurement. It
tests whether parameters learned from other image blocks explain unseen tracks
better. It is not ground-truth depth, and it is not leave-one-camera-out error,
because all observations in a validation track help triangulate that track's
3D point. A stronger future diagnostic would triangulate from a subset of
cameras and predict a camera excluded from that point solve.

Residuals are reported in three forms:

- native target sensor pixels;
- reference-equivalent pixels, divided by clamped local projection scale;
- angular error in degrees.

## 6. Observable physical parameterization

The reference camera is the fixed rotation, position, and raster gauge. Each
eligible non-reference camera may expose:

- three small world-frame axis-angle orientation components, each bounded to
  +/-0.5 degrees;
- three effective optical-centre offsets in world calibration units, each
  bounded to +/-5 (millimetres on L16);
- two sensor-raster origin offsets, each bounded to +/-64 pixels. The raster
  correction translates both K's principal point and the calibrated distortion
  centre while leaving the distortion coefficients unchanged;
- one additive movable-mirror state correction, bounded to +/-0.35 degrees.

The factory prior has a 0.20-degree orientation scale, a 1-unit optical-centre
scale, a 12-pixel raster scale, and a 0.08-degree mirror scale, with normalized
quadratic weight 0.02. For movable-mirror modules, centre correction applies to
the resolved virtual viewpoint: one capture cannot independently identify the
physical sensor location, mirror-plane distance, and mirror-axis position. No
arbitrary homography, polynomial warp, distortion-coefficient correction, or
per-track camera parameter exists inside this solve.

A camera first needs 150 fit observations. Every candidate parameter is then
tested independently with finite-difference Jacobians of retriangulated
reprojection residuals. Parameters with inadequate sensitivity or near-collinear
Jacobians are fixed at the factory value. Mirror-angle versus generic-orientation
degeneracy is checked explicitly, preferring the physical mirror degree of
freedom when it has adequate information. Raster offset versus yaw/pitch
degeneracy is also checked; the established bearing correction is retained
unless field-dependent evidence independently identifies the raster offset.
Cameras without an observable parameter remain outside the nuisance-point solve
so sparse views cannot move the 3D points of well-constrained cameras.

## 7. Robust solve and persistent membership

The solve uses only the fit side of the spatial split. It begins with a calibrated
epipolar/coplanarity objective and positive-depth checks. That initialization
is retained only if it improves the actual finite-depth objective.

The main objective robustly minimizes covariance-, confidence- and
structure-weighted reprojection residuals plus the factory prior. Every trial
rig retriangulates each track's nuisance 3D point. A Huber loss with transition
2.5 suppresses remaining mismatches, and bounded coordinate-Newton performs up
to six optimization sweeps.

After the initial solve, up to three fit-only membership passes are performed.
Each pass resolves the current rig, retriangulates the stored observations,
removes observations above the 6-reference-pixel consistency threshold, drops
tracks that can no longer retain three cameras, and reoptimizes. Images are not
reread and matches are not changed. Validation membership remains fixed.

## 8. Physical acceptance and fallback

The physical candidate is accepted only if:

- fit reprojection RMS decreases;
- at least 80% of fit tracks remain at positive depth;
- no fitted parameter reaches 98% of its bound;
- held-out RMS improves by at least 0.5%, at least 80% of
  held-out tracks remain at positive depth, and no affected camera with at
  least 12 validation samples regresses by more than 5% plus 0.02 pixels.

Held-out p95 reference lines of 6 reference-equivalent pixels and 0.10 degrees
are recorded in the JSON report, but they are not absolute truth and do not veto
a candidate that improves the independent split.

After acceptance, Chiaro installs the candidate rig and reruns the existing
residual image alignment. The report compares patch-supported correction
magnitudes against the factory-seeded run and marks whether the old 5%
reference threshold was reached. A shortfall or lost measurement is a warning
only. The physical candidate and its newly computed residual alignment still
reach depth, Joint CFA, resolution reconstruction, and synthesis. Use
`--no-rig-refine` for an explicit factory-only control.

## 9. JSON diagnostics

The fusion report records:

- candidate, physical-track and target-observation counts;
- mutually exclusive early rejection counts for unsupported depth, ambiguous
  depth, insufficient native-resolution views and failed physical consistency;
- whether physical or legacy correspondences drove the solve;
- per-camera physical-match support;
- inconsistent observation/track pruning;
- fit and debug-only held-out track counts;
- persistent-membership removals and pass count;
- objective and RMS before/after;
- native-pixel, reference-equivalent and angular residual percentiles;
- positive-depth fractions;
- per-camera fit and validation residuals;
- a 4x3 residual-vector field per reported camera;
- triangulation condition and ray-angle statistics;
- parameter sensitivities, correlations and observability decisions;
- proposed orientation/mirror corrections and bound status;
- downstream residual-correction measurements, reference-threshold result and
  any non-vetoing warning;
- the exact acceptance or fallback reason.

The dense physical stage does not treat stage 4 as an infinity observation. A
scene-fitted homography already contains parallax from whichever surfaces
dominated its fit; adding physical parallax to it would double-count that
motion and make the homography falsely win as a far-depth hypothesis. Instead,
for every physical depth hypothesis, stage 02 supplies only displacement
perpendicular to the calibrated epipolar trajectory:

```text
proposal(depth) = perpendicular(measured_warp - physical(depth))
candidate(depth) = physical(depth) + proposal(depth)
```

The along-trajectory component remains controlled by physical depth. The
measured warp is used unchanged only as the output fallback after finite depth
remains unresolved; it does not compete as evidence that the scene is at
infinity.

The physical and final stages also include fixed-scale disagreement maps against
the measured alignment, scalar confidence maps, and categorical visibility maps.
The fixed error colours are green at or below 1 pixel, yellow near 4 pixels,
orange near 16 pixels, and magenta at or above 32 pixels. Black means the two
warps do not share an in-sensor domain. In the factory and candidate stages the
error is an infinity-plane displacement from the measured capture warp, not a
held-out rig residual; finite-scene parallax is part of that displacement.
Visibility is green for directly visible, amber for in-sensor but unknown, red
for occluded, magenta for a depth boundary, and black for outside the target
sensor or undefined. White confidence in the two pre-depth physical stages is
only binary in-sensor projection validity, not measured alignment confidence.

The rig summary also reports the actual evaluated depth hypotheses per
reference candidate, a `level:candidate-count` refinement histogram, and both
the configured and worst observed final projected-motion bound. This makes a
six-level safety-cap hit or an unexpectedly expensive search visible instead
of presenting the original coarse-plane count as the effective resolution.

The trace reports the reference-space depth reconstruction once because it is a
single shared field; repeating that percentage for every camera would imply
independent depth solves that do not exist. Per-camera rows instead report how
much of the shared field that view directly refined, rejected, or retained as a
far fallback. `overlap` is purely the fraction of the reference raster mapping
inside that camera. `warp-defined` is the fraction retaining any finite final
mapping after depth evidence gates, while `direct-support` is the fraction
directly verified by that camera as a finite or far hypothesis. These quantities
must not be conflated: a camera can have roughly 90% optical overlap but much
less supported scene correspondence. Blend `weight share` is the actual
normalized contribution; `owner` is only the strongest individual source at a
pixel and is not an exclusive assignment.

When physical depth retains a directly verified far hypothesis, its synthesis
confidence inherits the measured alignment confidence. Depth uncertainty must
not replace an image-validated far mapping's confidence with a fixed low value;
doing so suppresses every non-reference camera and makes the reference dominate
the actual colour blend even when alignment is sound.

## 10. Historical fixed-48 result: `L16_04364`

Before the projected-motion hierarchy was introduced, the fixed-48
implementation was run as:

```sh
target/release/chiaro-fuse \
  '/run/media/dantistnfs/hdd_win/lightl16/alina_elina_nikita_halifax_walking_01_07_2026/L16_04364.lri' \
  --output /tmp/L16_04364_rig_48_parallel.png \
  --reference B4 \
  --cfa-held-out B2 \
  --crop 1900,1100,256,256 \
  --canvas native \
  --threads 8
```

The physical matcher used the full available reference raster; the crop limits
the requested synthesis/evaluation region. B2 supplied no matching or depth
evidence because it was the CFA-held-out camera.

### Observation population

| Metric | Result |
| --- | ---: |
| Reference candidates | 6,000 |
| Valid physical tracks | 875 |
| Tracks with 3+ cameras | 875 |
| Retained target observations | 2,073 |
| Inconsistent observations pruned during track construction | 1,137 |
| Candidate tracks rejected as inconsistent | 1,518 |
| Final fit tracks | 238 |
| Final spatially held-out tracks | 142 |
| Fit observations removed by persistent membership | 1,443 |
| Fit tracks removed by persistent membership | 476 |
| Membership passes | 3 |

The 875 tracks are the survivors of depth ambiguity, view-support,
triangulation, positive-depth and reprojection checks. They should not be
compared directly with feature counts such as “1,000 keypoints per megapixel”:
one retained track can contain several target observations, while this stage
needs only a spatially distributed calibration population and deliberately
rejects points that do not identify one common physical surface.

Per-camera support before the later observability/membership filtering was:

| Camera | Observations | Fraction of 875 tracks |
| --- | ---: | ---: |
| B5 | 439 | 50.17% |
| B1 | 358 | 40.91% |
| B3 | 358 | 40.91% |
| C2 | 236 | 26.97% |
| C3 | 215 | 24.57% |
| C6 | 156 | 17.83% |
| C5 | 131 | 14.97% |
| C1 | 116 | 13.26% |
| C4 | 64 | 7.31% |
| B2 | 0 | 0.00% (held out) |

### Geometric result

| Metric | Factory | Candidate | Change |
| --- | ---: | ---: | ---: |
| Fit reprojection RMS | 3.530 px | 3.346 px | improved 5.24% |
| Held-out reprojection RMS | 8.834 px | 8.905 px | **regressed 0.80%** |
| Fit positive-depth fraction | 100% | 100% | unchanged |
| Held-out positive-depth fraction | 100% | 100% | unchanged |

Held-out residual distributions were:

| Units | Factory median / p75 / p90 / p95 | Candidate median / p75 / p90 / p95 |
| --- | --- | --- |
| Sensor pixels | 6.296 / 9.637 / 13.417 / 18.145 | 6.291 / 9.669 / 12.816 / 18.962 |
| Reference-equivalent pixels | 5.770 / 8.574 / 10.484 / 11.366 | 5.557 / 8.779 / 10.467 / 11.124 |
| Angular degrees | 0.03954 / 0.05809 / 0.07110 / 0.07613 | 0.03808 / 0.05916 / 0.07127 / 0.07626 |

Triangulation itself was numerically usable: median condition number 7.54,
p90 condition number 69.34, and median maximum ray angle 1.137 degrees. All
tracks retained positive depth.

The solver proposed these small corrections:

| Camera | Orientation x / y / z (degrees) | Mirror (degrees) |
| --- | --- | ---: |
| B5 | +0.0083 / +0.0095 / +0.0123 | -0.0010 |
| B1 | +0.0022 / -0.0044 / -0.0052 | -0.0021 |
| B3 | +0.0030 / -0.0078 / +0.0050 | -0.0012 |
| C2 | +0.0187 / +0.0077 / +0.0910 | -0.0142 |
| C3 | -0.0112 / -0.0097 / +0.0689 | -0.0027 |

No proposed parameter approached its configured bound. Other cameras remained
fixed because they were the reference, explicitly held out, or did not retain
an observable parameter population.

### Acceptance outcome

**The candidate was rejected and the factory rig was retained.** Its fit RMS
improved, but:

- held-out RMS changed from 8.834 to 8.905 pixels, a -0.801% improvement versus
  the required +0.500%;
- held-out p95 remained 11.124 reference-equivalent pixels, above the 6.0-pixel
  absolute limit.

Angular p95 was within its 0.10-degree limit, and positive-depth support was
100%, but all gates are mandatory. Because the first geometric gate failed,
the downstream residual-correction gate evaluated zero cameras and no proposed
correction affected the rendered output.

The run spent approximately 319.4 seconds in alignment and 20.7 seconds in
synthesis, with peak resident memory around 2.00 GiB. These timings describe
this capture, crop, build and machine only.

## 11. Projected-motion hierarchy verification

A subsequent all-sensor B4-reference run exercised the new hierarchy over the
same 6,000 full-frame reference candidates. It evaluated 567,385 depth
hypotheses, or 94.6 per candidate. The refinement histogram was 1,791 candidates
at one level, 243 at two levels, and 3,966 at three levels. No candidate reached
the six-level safety cap, and the worst measured final adjacent-hypothesis
motion was 7.774 target pixels against the configured 8-pixel limit.

That run retained 481 physical tracks and 1,133 target observations. After
persistent membership it evaluated 99 fit and 36 held-out tracks. Training RMS
improved from 2.685 to 1.486 pixels, but held-out RMS regressed from 6.087 to
6.117 pixels and held-out p95 remained 10.071 reference-equivalent pixels. The
candidate was therefore correctly rejected. Alignment took about 351 seconds,
roughly 10% longer than the historical fixed-48 run rather than the many-fold
cost of a globally dense sweep.

This verifies that fixed-plane holes are now closed at the configured image
motion scale. It also demonstrates that depth quantization was not the only
problem: cross-field/cross-camera correspondence consistency or physical-model
mismatch still prevents the proposed rig correction from generalizing.

### Factory/candidate dense-depth comparison

The first forced dual-depth audit selected factory `PhysicalRig` for production
but also ran the rejected candidate through the complete dense solver. Both
branches accepted all ten target views and produced nearly identical sparse
finite-depth fields:

| Branch | Measured finite nodes | Tested nodes | Reconstructed | Regularized |
| --- | ---: | ---: | ---: | ---: |
| Factory `PhysicalRig` | 26,643 | 794,320 | 3.354% | 0 |
| Candidate `PhysicalRig` | 26,822 | 794,340 | 3.377% | 0 |

The candidate gained only 179 measured nodes, or 0.022 percentage points. Its
per-camera defined coverage differed from factory by at most 0.02 percentage
points, and finite accepted nodes changed only slightly in either direction.
This is direct evidence that the proposed small B5 correction does not address
the fundamental 96.6% finite-depth coverage deficit.

The initial verification run rebuilt factory, candidate and selected depth
independently and took about 1,047 seconds in alignment. The implementation now
reuses the selected factory result for `05a` when they are identical, leaving
only the candidate audit as additional dense-solve cost in the common rejected-
candidate case.

## 12. Stage-2 proposal recovery verification

An all-sensor B4-reference control then added the stage-2 alignment as an
epipolar proposal. The original factory-centred search remains authoritative
whenever it already reaches the 0.50 ZNCC support threshold; the proposal is
consulted only to recover otherwise unsupported camera/depth pairs. This
factory-first rule proved important: replacing every factory-centred result
with a measured proposal reduced the track population, whereas additive
recovery increased it.

The final candidate funnel was fully accounted for:

| Outcome | Reference candidates |
| --- | ---: |
| No supported multi-view depth | 2,910 |
| Ambiguous separated depth modes | 666 |
| Insufficient views after native localization | 0 |
| Failed proposal-normalized physical consistency | 1,855 |
| Retained physical tracks | 569 |
| **Total** | **6,000** |

Compared with the preceding projected-motion-only run, retained physical
tracks increased from 481 to 569 (+18.3%), target observations from 1,133 to
1,353 (+19.4%), and final optimization/validation tracks from 135 to 170
(+25.9%). C1 support increased from 46 to 81 observations and C4 from 26 to
58, while existing factory-supported matches remained the first choice.

The larger population produced a candidate that reduced held-out RMS from
11.221 to 10.148 reference-equivalent pixels (+9.56%). It was nevertheless
correctly rejected: held-out p95 was 20.573 pixels and 0.13939 degrees, above
the mandatory 6-pixel and 0.10-degree limits. This improves search coverage
without weakening acceptance, and identifies the remaining issue more
precisely as the long-tail consistency/generalization of recovered tracks.

The matching full debug audit retained factory `PhysicalRig` for production.
Its independent candidate dense branch initially remained almost identical to
factory. That comparison was subsequently superseded by the false-anchor and
dense-sampling correction below.

## 13. Dense false-anchor and direct-sampling correction

The first physical dense implementation still composed finite physical
parallax on top of stage 02 as though stage 02 measured an infinity plane. That
assumption was invalid: the fitted image warp already contains the parallax of
the scene surfaces that dominate its matches. It made the dominant structures
look artificially correct at the far label and then double-counted their
motion at every finite label.

The corrected implementation uses stage 02 only for displacement perpendicular
to the physical epipolar trajectory. The along-trajectory coordinate now comes
entirely from the tested physical depth. In an all-sensor control before direct
sampling refinement, this made factory and candidate geometry meaningfully
different for the first time:

| Branch | Direct selected | Neighbour-consistent | Component-consistent measured | Reconstructed |
| --- | ---: | ---: | ---: | ---: |
| Factory `PhysicalRig` | 45,122 | 35,512 | 20,455 | 2.57% |
| Candidate `PhysicalRig` | 74,552 | 61,159 | 35,369 | 4.45% |

The remaining direct search still inherited the fixed coarse inverse-depth
spacing. A ZNCC maximum could therefore lie between adjacent planes even when
the correct broad depth mode had been found. Direct verification now bisects
the winning neighbouring inverse-depth intervals for up to three levels, but
only while any calibrated target moves by more than 1.5 full-resolution pixels
between them. It adds at most six local hypotheses per node instead of
globally multiplying the 96-plane volume.

On the same capture and crop this bounded hierarchy produced:

| Branch | Direct selected | Neighbour-consistent | Component-consistent measured | Reconstructed |
| --- | ---: | ---: | ---: | ---: |
| Factory `PhysicalRig` | 76,825 | 64,073 | 41,710 | 5.25% |
| Candidate `PhysicalRig` | 100,641 | 83,982 | 50,093 | 6.30% |

Factory finite coverage more than doubled relative to the corrected fixed-step
control (+21,255 measured nodes), while the candidate gained 14,724 nodes. The
complete alignment time rose from 788.1 to 853.3 seconds, about 8.3%. Visual
inspection shows coherent new support on the left structure, roofline, stairs,
crane edges and foreground rather than isolated speckle.

This is still not a filled dense field. The production factory path remains at
5.25%, the rejected candidate reaches 6.30%, and both report zero regularized
nodes. Blue in the provenance image therefore still mostly means an
image-validated stage-02 output fallback, not a finite metric depth. The next
separate problem is conservative propagation from directly measured surfaces
through weakly textured interiors without crossing depth boundaries; relaxing
the photometric or component gates alone would relabel unsupported matches as
depth.

## 14. Interpretation and next validation work

This run validates the safety and wiring of the stage more strongly than it
validates calibration improvement:

- the physical matcher produced a sizeable, multi-view population without a
  homography-defined locus;
- cross-focal B/C observations were handled by the generalized-camera
  projection Jacobian;
- the constrained optimizer found small, bounded corrections and reduced fit
  error;
- spatial validation detected that those corrections did not generalize;
- the pipeline retained factory geometry, as designed.

It does **not** establish a useful capture-specific correction for
`L16_04364`. The current evidence suggests remaining correspondence
inconsistency or model mismatch across the image field: training improves while
held-out error does not, and the held-out reference-equivalent p95 remains well
above the absolute gate.

Before treating the feature as beneficial, repeat the matched factory/refined
experiment on several captures and require stable gains in held-out geometry
and the second residual-correction gate. Useful additions would be:

1. leave-one-camera-out reprojection validation;
2. validation grouped by camera tier, image-field cell, structure and depth;
3. explicit analysis of the many observations removed by persistent
   membership;
4. compare the projected-motion hierarchy against the historical fixed-plane
   population, especially C-camera support and observation-pruning rates;
5. repeat the forced factory/candidate dense-depth audit across captures;
6. add separately-provenanced, edge-aware depth propagation through weakly
   textured interiors, and validate it against held-out views before allowing
   it to replace the stage-02 output fallback.

Until those tests show repeatable improvement, the correct current conclusion
is: **the physical refinement mechanism is operational and safely gated, but
the tested capture remains on factory calibration.**

## 15. Resolution-recovery work plan

A registered comparison against the Light/Lumen export of `L16_04364` shows
that the remaining difference is not primarily global alignment. At preview
scale the median registration residual is 0.273 pixels, but coherent
high-frequency energy in the current result is only about 36% of Lumen's.
Disabling dense depth increases it to about 59%, while using only B4 falls to
about 30%. This identifies two separate losses: sparse or incorrect depth
removes otherwise useful cameras, and the final reconstruction still extracts
less detail than Lumen after those cameras are admitted.

The required changes, in dependency order, are:

1. **Retain uncertain physical depth instead of discarding it.** Preserve the
   edge-regularized coarse SGM solution on the fine output grid wherever a
   strict direct measurement is absent, then complete only tested,
   depth-consistent neighbourhoods. Keep this provenance separate and its
   confidence below directly measured depth. It may guide reconstruction but
   must never create hard occlusion.
2. **Initialize physical depth from measured epipolar displacement.** Project
   the measured correspondence onto the calibrated epipolar trajectory and
   invert that coordinate to seed the local depth search. The perpendicular
   component remains only a residual proposal. This should reduce missed
   narrow peaks and the amount of exhaustive search.
3. **Separate unknown depth/visibility from unavailable geometry.** A camera
   with valid projection but unresolved depth must retain a usable,
   low-confidence mapping rather than becoming `Undefined`. Selection can then
   choose between finite and far mappings without removing the camera from the
   contributor set.
4. **Replace the reference-only synthesis detail gate.** Structure visible in
   sharper auxiliary cameras must be able to activate Joint-CFA/resolution
   reconstruction even when B4 is locally blurred. Gate on robust multi-view
   structure and agreement instead of B4 structure alone.
5. **Improve CFA-precision local registration.** Dense physical geometry gets
   a source into the correct neighbourhood, but Joint CFA needs substantially
   tighter phase agreement. Diagnose and replace the current resolution-warp
   refinement where magnified cameras retain large corrections or almost no
   verified cells.
6. **Add calibrated detail restoration.** Once contributor geometry is dense
   and stable, introduce per-camera PSF/MTF-aware reconstruction or a
   conservative post-reconstruction deconvolution. This must be evaluated on
   real repeated structure and noise, not only on global sharpness scores.

Item 1 is implemented in the current branch. The default completion pass now:

- retains edge-compatible SGM estimates at reduced confidence and marks them
  `Regularized` even when their coarse source passed its own measurement gate;
- never replaces a surviving fine-grid direct measurement;
- leaves untested nodes unsupported;
- chooses a single locally compatible inverse-depth mode rather than blending
  foreground and background estimates;
- performs bounded edge-aware propagation through remaining tested holes; and
- continues to exclude every regularized node from the target-view hard
  z-buffer, so only directly measured surfaces can assert occlusion.

Validation must report measured and regularized coverage separately. A higher
regularized fraction is expected; it is useful only if per-camera refined
support, Joint-CFA participation and registered detail improve without visible
double edges or boundary leakage.

The first full-canvas `L16_04364` validation produced this comparison against
the immediately preceding optimized build:

| Metric | Before | With item 1 |
| --- | ---: | ---: |
| Direct measured nodes | 195,229 | 224,736 |
| Regularized nodes | 0 | 532,464 |
| Combined finite-depth coverage | 24.56% | 95.31% |
| Resolution reconstruction | 3.19% | 20.30% |
| Joint-CFA reconstructed output | 6.48% | 3.95% |
| Rejected non-reference edge samples | 68.14% | 55.24% |
| B4 luminance weight | 94.38% | 91.86% |
| Alignment time | 78.3 s | 87.2 s |
| Synthesis time | 57.1 s | 78.7 s |

No obvious double edges or boundary leakage appeared in the four registered
crane, mesh and railing inspection crops, and the complete outputs differ by
43.56 dB PSNR. The much broader field therefore behaves conservatively, but it
does not yet recover visibly Lumen-like detail. In particular, the lower
Joint-CFA success rate despite much higher resolution-reconstruction coverage
means that items 2 through 4 remain necessary: dense availability alone does
not make the admitted physical samples mutually consistent at CFA precision.

Item 2 is also implemented. At every fine-grid node, accepted stage-2 mappings
are converted to target-camera rays and paired with the physical reference ray.
Their closest positive intersection supplies an inverse-depth proposal. Seeds
from different cameras are clustered in inverse depth, require the normal
multi-view support count, and are confidence-weighted within the strongest
compatible mode. The resulting seed adds a second local hypothesis band beside
the SGM seed; it never changes the perpendicular proposal and does not bypass
any photometric, far-baseline, neighbourhood or component acceptance gate.

On the same full-canvas capture, 349,012 of 794,486 tested nodes (43.93%) had a
robust measured epipolar seed. Relative to item 1 alone:

| Metric | Item 1 | Items 1+2 |
| --- | ---: | ---: |
| Direct finite selections | 258,016 | 265,147 |
| Component-consistent measured nodes | 224,736 | 231,814 |
| Regularized nodes | 532,464 | 528,250 |
| Combined finite-depth coverage | 95.31% | 95.67% |
| Resolution reconstruction | 20.30% | 20.10% |
| Joint-CFA reconstructed output | 3.95% | 4.04% |
| Dense-depth time | 70.3 s | 71.3 s |

The seed therefore converts about 7,100 uncertain nodes into direct
measurements for roughly one second of dense-depth cost. The complete images
differ by 53.47 dB PSNR and the registered inspection crops show no new double
edges. Its resolution effect is intentionally modest: it improves search
initialization, while the remaining contributor loss is now primarily the
unknown/unavailable mapping distinction and CFA-precision consistency.

Item 3 is implemented as a distinct geometry state rather than a relaxed
visibility decision. If an in-sensor stage-2 mapping exists but neither finite
nor far depth is established, the per-camera warp now retains that mapping as
low-confidence `Unknown`; only an absent or out-of-sensor projection becomes
`Undefined`. When regularized finite depth fails per-camera visibility and the
shared far hypothesis is supported, the finite physical coordinate remains the
sampling point and the measured far coordinate is retained only as global
diagnostic context. Explicit measured-depth occlusion and boundary states
remain blocking.

Relative to items 1+2, this raised B-camera defined warp coverage from
88.2-89.7% to 90.2-94.8%. Resolution reconstruction increased from 20.10% to
23.44%, Joint-CFA output from 4.04% to 4.30%, and B4 luminance weight fell from
91.74% to 90.27%. The tradeoff is synthesis time: 77.0 seconds became 106.2
seconds because more valid low-confidence mappings reach downstream sampling.
The CLI and JSON report now expose unknown nodes separately from defined and
directly supported nodes.

Item 4 is implemented conservatively. Reference structure still activates
Joint CFA directly, but detail missing from B4 may now activate it when at
least two explicitly visible cameras with at least 1.25x magnification agree
on edge direction or ridge sign. The weaker agreeing magnitude controls the
gate, so a single sharp noise realization, an `Unknown` mapping, or an
occluded/boundary sample cannot open it.

This capture shows that the old reference-only gate was not a material
bottleneck: it already ran the solver at 93.49% of output pixels. Multi-view
activation added only 1,314 solver attempts and 433 successful
reconstructions; the outputs differ by 86.35 dB PSNR. The next engineering
priority should therefore be CFA-precision local registration and contributor
agreement, where C-camera verified fractions remain below 0.5% in most views,
before PSF/MTF restoration is attempted.

The first phase of item 5 is implemented, and it changes the diagnosis. A
finite reference depth and calibrated camera pose now always produce the
target coordinate directly; failure of an independent photometric check no
longer substitutes an infinity/far mapping. Visibility authority and coordinate
confidence are kept separate, and spatial-consensus demotion preserves the
finite coordinate. The remaining residual refinement is deliberately bounded:
the physical matcher fits a continuous subpixel peak inside its existing 3x3
basin, while resolution verification rejects corrections larger than 2.25
reference-raster pixels. A broad image-space search is not a substitute for
correct depth and camera geometry and would admit repeated-texture false
matches.

Joint-CFA reporting now exposes its rejection funnel, and weak but
photometrically consistent regularized geometry is retained as weak evidence
instead of being hard-gated twice. On `L16_04364`, 62.32% of attempted pixels
have at least two geometrically eligible calibrated colour-camera projections,
which is close to the expected C-group footprint but is not the same quantity
as a successful CFA solve. Only 8.09% reconstruct through Joint CFA: 15.11% of
attempts fail after collecting the CFA footprint and 38.55% of solver attempts
fail robust conditioning. Loosening those checks would merge mutually
inconsistent samples rather than recover resolution.

The major upstream omission was the rig observation gate. The default minimum
of 150 observations excluded every C camera on this capture (their counts were
56, 107, 81, 84, 58 and 80), so the supposedly known capture-specific C-camera
corrections were actually zero. Reducing the default to 48 lets the existing
observability, robust-membership, prior and bound checks decide which
parameters are supportable. Nine cameras are now fitted, the production-track
RMS improves from 6.499 to 3.530 pixels, downstream residual alignment improves
by 25.48%, resolution reconstruction reaches 24.25%, and Joint-CFA output rises
from 6.86% to 8.09%.

This expanded fit also improves independently held-out observations overall:
RMS falls from 12.425 to 11.176 pixels (10.05%). That validates the direction,
but the absolute residual is still much larger than the subpixel accuracy
needed for CFA fusion. The next part of item 5 must therefore refine physical
camera/depth geometry from dense, independently measured correspondences (with
held-out acceptance), then regenerate depth and warps. It should not expand the
image-space search radius or relax Joint-CFA conditioning.
