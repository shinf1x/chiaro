# Calibration metadata A/B

This note records the `L16_04364` experiments prompted by calibration metadata
that the fusion geometry previously discarded. All measured runs used the same
release build and at most eight worker threads.

## Mirror quadratic semantics

The old implementation interpreted the six quadratic coefficients as two
candidate Hall-to-angle polynomials and selected whichever better reproduced
the five explicit calibration pairs. That does not match the remaining
protobuf fields.

On every movable module in this device calibration, coefficients 0-2 are zero
and coefficients 3-5 encode normalized actuator position as a quadratic of
normalized mirror angle:

```text
normalized_hall = (actuator_length_offset - hall_code) / actuator_length_scale
normalized_hall = a*normalized_angle^2 + b*normalized_angle + c
angle_degrees = normalized_angle*mirror_angle_scale + mirror_angle_offset
```

The stored `inflection_value` independently confirms the mapping direction. It
equals the quadratic vertex transformed back to raw Hall counts for every
movable module. For B2, the value derived from the coefficients is
`10230.319287`; the protobuf stores `10230.319336`.

Chiaro now solves this quadratic and uses
`use_rplus_for_left_segment`/`use_rplus_for_right_segment` to select its inverse
root. `calibration-quadratic-inverse` is the default. The CLI retains
`current-quadratic` for rollback and `calibration-pairs-linear` for controlled
A/B experiments.

### End-to-end result

| Model | Tracks | Target RMS | Median | P90 | P95 | >50 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy quadratic control | 10,009 | 1.456 | 0.829 | 2.111 | 2.791 | 0 |
| Factory quadratic inverse | 10,060 | 1.336 | 0.822 | 2.099 | 2.811 | 0 |
| Measured-pair linear | 10,162 | 5.241 | 0.825 | 2.563 | 3.812 | 8 |

The inverse model improves same-binary target held-out RMS by about 8.2% and
substantially improves C1 (`3.991` to `2.218` px), C3 (`2.328` to `1.992` px),
and C5 (`1.960` to `1.689` px). Its P95 is essentially flat/slightly worse,
and several B cameras regress by a few hundredths of a pixel. The measured-pair
model is decisively rejected, including two roughly 125 px C2 failures.

Factory geometry participates in epipolar and cycle gates, so each model
produces a different correspondence graph. These are end-to-end comparisons,
not fixed-track reprojection tests. Neither optimized candidate passes the
existing absolute per-camera/target acceptance gates; the rig still falls back
as designed.

## CRA focus travel and optical centre

Resolved calibration now retains the CRA center, sensor and exit-pupil
distances, pixel size, lens Hall, Hall-to-distance ratio, complete radial
sample/fit-coefficient curves, fit cost, and valid ROI. The diagnostic report
records those values alongside the capture lens Hall, calibrated Hall span and
extrapolation, resolved `K`, AF mirror Hall, and all three mirror-angle
predictions.

The exact metadata relation

```text
sensor_distance = (lens_hall + 1) * distance_hall_ratio
```

supports a physically constrained experiment: one scalar per B/C focal group
scales the implied focus travel along each physical camera's optical axis. For
movable-mirror modules, the real camera moves before it is reflected into the
virtual camera. Enable this only with `--rig-max-focus-pupil-scale`; the default
zero leaves geometry unchanged. `--rig-focus-pupil-prior-sigma` controls its
prior.

The test fitted B `-0.0183` and C `+0.1569`, representing roughly 1.3-1.6 mm of
C-group axial motion. Target RMS was `1.360` px versus the legacy control's
`1.456` px, but the per-camera result was mixed: C1 improved from `3.991` to
`3.002` px, while C5 and C6 regressed slightly. It still failed the absolute
held-out gates. The metadata and implementation are therefore retained for
future multi-capture work, but a single group-shared pupil model is not enabled
by default.

## Reproducing the comparisons

Use the normal fusion command with `--threads 8`, plus one of:

```text
--mirror-angle-model calibration-quadratic-inverse
--mirror-angle-model current-quadratic
--mirror-angle-model calibration-pairs-linear
```

For the pupil experiment, add:

```text
--rig-max-center-offset 0
--rig-max-focus-pupil-scale 1
```

The neighboring `.fusion.json` report contains `rig_refinement.mirror_angle_mode`,
`rig_refinement.factory_model`, and per-correction `focus_pupil_scale` values.
