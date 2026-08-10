# M4 — Segmentation + Metrics (pipeline Stages 6-7)

Source: `docs/spec/No_Layups_V1_Specification.pdf`, Sections 6.3, 7.6, 7.7,
7.8, 8, 12 (row M4), 13.1 (`test_segment.py`, `test_metrics.py`).

## Goal

Complete key-event detection (address/top/impact) and compute the 8 swing
metrics + 2 normalized trajectories, so the CLI's `swing.json` output is
fully populated instead of `keyframes: null`.

## Target behavior

- `segment.py` gains `detect_impact`, `detect_top`, and an orchestrating
  `detect_keyframes()` implementing Section 7.6 steps 3-5 (already had steps
  1-2 from M2).
- Two-pass ordering (7.5) is completed: Pass A now runs the *full* Stage 6 on
  the smoothed, untransformed lead-wrist series (not just address), then
  Stage 5's canonical transform anchors on the detected address, then Stage 7
  (new `metrics.py`) computes the 8 metrics + trajectories in canonical space.
- `no_full_swing` becomes a genuine **soft failure**: the pipeline no longer
  raises past it. `frames` is still fully populated (canonical transform
  falls back to address=0 if even address-detection fails, so the viewer and
  a future manual-keyframe editor always have something to render);
  `keyframes`, `metrics`, `trajectories` are `null`; `"no_full_swing"` is
  appended to `meta.warnings`.
- `metrics.py` adds `angle_at`, `turn_angle`, `spine_tilt` (Section 6.3/8.1/8.2
  formulas, copied exactly) and `compute_metrics()` / `compute_trajectories()`.
  Any metric whose required joint is missing at the needed frame is `null`.

## Spec deviations (all found against real footage, all documented in code)

1. **Detection order — top before impact.** The spec finds impact (step 3)
   then bounds the top search by it (step 4). Top is the more robust of the
   two (a clear height maximum), so it is found first and used to bound
   impact. This also matches the physical order.
2. **Impact by height, not by speed.** The spec takes the lead-wrist speed
   peak inside a fixed `[0.4N, 0.9N]` slice. On the reference clip impact
   falls at ~98% of the clip, outside that window entirely, and the speed peak
   is easily won by tracking jitter during the blurred part of the swing.
   Impact is instead the lead wrist's height minimum after the top.
3. **Height sign.** Section 7.6 step 4 takes `argmax` of raw wrist Y, assuming
   Y grows upward. MediaPipe world landmarks grow *downward*, so Y is negated
   in `wrist_height()`. Segmentation is Pass A of the 7.5 two-pass ordering,
   so it runs before the canonical transform exists and cannot borrow its up
   axis.
4. **Playback-rate invariance (two places).** Speed in m/s is a function of
   the playback timeline, so slow-motion footage — extremely common for swing
   video — reads far slower than reality (the reference clip peaks at ~1.5
   m/s). Both of the spec's absolute m/s gates are therefore replaced with
   scale-invariant ones: the step-3 "did a swing happen?" gate (3.0 m/s)
   becomes a distance gate, `MIN_BACKSWING_RISE_M`; the step-1 takeaway
   trigger (0.4 m/s) becomes `min(0.4, 5% of the clip's own peak speed)`, so
   normal-speed footage keeps exactly the spec's behaviour while slow-motion
   footage scales down with it.
5. **`spine_tilt_address` is null, not a number** (Section 8.2). Section 6.1
   defines canonical +Y *as* the address trunk vector, so Section 8.2's formula
   evaluates to exactly 0.0 at address for every clip. Since the reference is
   anchored identically its value is also 0.0, so a reported 0.0 would score
   every golfer a permanent "good" (Section 9.3, |delta| <= 3 deg) on something
   never measured. `null` is an already-supported state that omits the rating
   and leaves the Section 5.2 schema intact. `spine_tilt_impact` is kept — in
   canonical space it reads as tilt *relative to address*, which is the
   coachable quantity.
6. **`head_sway_top` uses canonical X only** (Section 8.3). The spec's Z term
   is monocular depth: on the reference clip it contributed 37.9 cm of a 39.4
   cm total. Section 9.3 rates this metric on absolute thresholds rather than
   against the reference, so it gets no cancellation and a depth-inflated value
   reads "attention" for everyone. Canonical +X is body-derived (the address
   hip line, roughly the target line), so it is both the direction golf sway
   means and stable across camera angles. Reference value drops 39.4 -> 10.7 cm.
7. **Quality gate scope.** Section 7.4 step 3 is applied over `address..impact`
   by the pipeline once keyframes are known, not over the whole clip inside
   `filtering.process`. Tracking loss during the follow-through is very common
   and says nothing about whether the swing itself was measurable.

## Files to change

- `src/no_layups/config.py` — Stage 6 steps 3-5 thresholds (top window
  margins, minimum argmax window size) plus the two scale-invariant gates
  (`MIN_BACKSWING_RISE_M`, `TAKEAWAY_SPEED_FRACTION_OF_PEAK`).
- `src/no_layups/pipeline/filtering.py` — move the step-3 quality gate out of
  `process()` into `check_quality(smoothed, start, end)`; smooth a
  fully-filled copy so Savitzky-Golay cannot bleed NaN out of a long gap into
  neighbouring frames that tracked fine.
- `src/no_layups/pipeline/segment.py` — `detect_impact`, `detect_top`,
  `detect_keyframes`.
- `src/no_layups/pipeline/metrics.py` (new) — Section 8 formulas +
  `compute_metrics`/`compute_trajectories`.
- `src/no_layups/pipeline/__init__.py` — wire Pass A's full detection, the
  soft-failure fallback, and Stage 7.
- `tests/test_segment.py` (new, the spec-named file) — full
  address/top/impact detection against a synthetic wrist path with
  deterministic (piecewise-linear/constant-velocity) truth values, at both
  30 and 60 fps; stationary series raises `no_full_swing`. Also absorbs the
  three step-1/step-2 cases from M2's `tests/test_pipeline_segment.py`, which
  is deleted — the spec names one segmentation test file, and two files had
  begun duplicating the stationary-series case.
- `tests/test_metrics.py` (new, the spec-named file) — the 5 prescribed unit
  cases (`angle_at`, `turn_angle`, `spine_tilt` × 2, straight-arm 180°) plus
  a `compute_metrics` integration case (missing joint → null metric,
  `x_factor` = shoulder_turn − hip_turn).

## Test plan

See Section 13.1's `test_segment.py`/`test_metrics.py` bullets, reproduced
above. Plus a real-data check: rerun the CLI against the (already-trimmed)
Rory clip from M2 and eyeball detected keyframes against the viewer, and
confirm metric values fall in the plausible ranges from Section 13.2
(shoulder turn 60-110°, lead elbow 150-180°).

## Out of scope

- `compare.py` / reference comparison / ratings (Section 9) — M6.
- Exposing metrics/keyframes through the API or viewer UI — M5/M6 wire the
  CLI output into the app; this milestone only changes what `run_pipeline`
  returns.
