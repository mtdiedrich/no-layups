# Bundled reference swing

`reference.json` is the Section 9.1 reference artifact: a real swing in the
`swing.json` format (Section 5.2), used by `compare.py` (M6) as the baseline
every uploaded swing is rated against.

## Provenance

Rory McIlroy slow-motion driver swing, face-on, static camera.
Source: `RORY.mov` — 770x480, 60 fps, 677 frames, 11.28 s, h264.

It passes Stage 1 validation as-is, so there is no preprocessing step:

```bash
uv run no-layups process RORY.mov --handedness right -o reference/reference.json
```

An earlier 20.69 fps copy of the same footage sat below `config.MIN_FPS` (24)
and had to be retimed with ffmpeg before the pipeline would touch it. That
workaround is gone — do not reintroduce it. If you ever do need to normalize a
sub-24 fps clip, **retime** it (relabel the existing frames) rather than
duplicating frames up to the target rate: duplicated frames have zero
inter-frame speed and defeat Section 7.6 step 1's "3 consecutive frames above
threshold" test.

Keyframes were verified manually as Section 9.1 requires — `address: 38`,
`top: 330`, `impact: 445` — checked against the video frame by frame rather
than trusted from the detector. Backswing:downswing is 2.5:1.

`src/no_layups/static/sample.swing.json` is the same artifact, serving as the
viewer's demo asset. Note it is ~1.1 MB, roughly 3x the old 30 fps version,
and the browser fetches it on page load.

## Why a real reference and not the synthetic placeholder

Section 9.1 allows `tools/make_synthetic_reference.py` to build a placeholder
from the Appendix B poses. Do not use it for rating. Appendix B is a table of
hand-written canonical coordinates with exact rotations, so it carries **zero**
monocular depth error, while every real upload carries a lot of it. Running the
Appendix B poses through the real `metrics.py` yields `shoulder_turn_top` =
90.0 deg against this clip's 11.4 deg — a delta of ~79 deg, four times Section
9.3's "attention" threshold. Rated against the synthetic reference, a tour
professional's swing scores "attention" on 6 of 8 metrics. The synthetic
reference also fails its own absolute head-sway threshold (15.8 cm against an
"ok" ceiling of 10 cm).

Section 9.2's `delta = user - reference` only cancels MediaPipe's systematic
depth bias when **both** sides carry it. That is the whole reason the rating is
reference-relative, and it is why the reference must come from real footage
through the normal pipeline.

Cancellation is still only partial: foreshortening is view-dependent, so a
down-the-line upload does not cancel against this face-on reference as cleanly
as a face-on one does. See Section 13.2, which requires acceptance on both.

## Known soft spot

The turn metrics are baselined on a single address frame, and MediaPipe's
frame-to-frame depth noise makes that baseline unstable: across three
defensible address frames on this clip, `shoulder_turn_top` moved between 4.1
and 14.2 deg while the golfer was demonstrably motionless. Detection now picks
a settled address frame, but the underlying sensitivity is a property of
single-frame baselining, not of the detector. Median-filtering the address
joint positions over a small window would damp it. Worth doing before the
ratings in Section 9.3 are taken seriously.

## Note

`reference.json` contains derived numeric joint coordinates only — no video,
audio, or imagery. Worth a second look before any public distribution of the
repo, since the source clip is third-party broadcast footage.
