# Bundled reference swing

`reference.json` is the Section 9.1 reference artifact: a real swing in the
`swing.json` format (Section 5.2), used by `compare.py` (M6) as the baseline
every uploaded swing is rated against.

## Provenance

Rory McIlroy driver swing, slow motion, static camera, angled view.
Source: `RORY.mov` — 1440x1080, 60 fps, 873 frames, 14.55 s, h264.
(Watermarked "Video copyright Michael Field".)

It passes Stage 1 validation as-is, so there is no preprocessing step:

```bash
uv run no-layups process RORY.mov --handedness right -o reference/reference.json
```

Frames are downscaled to 1280x960 on the way in (`MAX_LONG_EDGE`). Duration is
14.55 s against the 15 s ceiling — 0.45 s of headroom, so do not re-trim it
longer.

Keyframes were verified manually as Section 9.1 requires — `address: 49`,
`top: 306`, `impact: 409` — checked against the video frame by frame rather
than trusted from the detector. Backswing:downswing is 2.5:1.

`meta.warnings` carries `partial_tracking`: the trail elbow is missing 33.5% of
address..impact because this view hides it behind the torso through the
backswing. It feeds no Section 8 metric, so the analysis is unaffected and only
the rendered skeleton degrades — see `filtering.check_quality`. Every joint
that does feed a metric is missing **0%** of the swing.

`src/no_layups/static/sample.swing.json` is the same artifact, serving as the
viewer's demo asset. It is ~1.4 MB and the browser fetches it on page load.

## Why the camera angle matters more than the pixels

An earlier reference used a face-on clip of the same golfer. Swapping to this
angled view changed the metrics far more than resolution did:

| metric | face-on clip | this clip | plausible (13.2) |
| --- | --- | --- | --- |
| shoulder_turn_top | 11.4 deg | **54.6 deg** | 60-110 |
| hip_turn_top | 9.2 deg | 22.0 deg | — |
| x_factor | 2.2 deg | **32.6 deg** | — |
| head_sway_top | 12.5 cm | 6.4 cm | <= 5 good |

Turn metrics are read off the depth axis (Section 8.1 drops Y and measures
rotation in the ground plane). On a face-on clip the shoulder line points
almost straight at the camera at the top of the backswing, so the rotation
lives entirely in the least reliable channel and reads as a fraction of its
true value. An angled view puts the same rotation partly in the image plane
where MediaPipe is accurate.

So "monocular depth cannot measure rotation" was too strong a conclusion. It
measures rotation poorly *from a face-on view*, which is the view golfers
default to filming.

Section 13.2's ranges are still not fully met — shoulder turn 54.6 against
60-110, lead elbow 139.5 against 150-180 — so absolute turn values remain
under-read. Compare the deltas, not the absolutes.

## Why a real reference and not the synthetic placeholder

Section 9.1 allows `tools/make_synthetic_reference.py` to build a placeholder
from the Appendix B poses. Do not use it for rating. Appendix B is hand-written
canonical coordinates with exact rotations, so it carries **zero** monocular
depth error while every real upload carries a lot of it. Its `shoulder_turn_top`
is exactly 90.0 deg, and it fails its own absolute head-sway threshold (15.8 cm
against an "ok" ceiling of 10 cm).

Section 9.2's `delta = user - reference` only cancels MediaPipe's systematic
bias when **both** sides carry it. That is the whole reason the rating is
reference-relative, and it is why the reference must come from real footage
through the normal pipeline.

Cancellation is still only partial, and now demonstrably view-dependent: an
upload shot face-on will not cancel against this angled reference. That is the
strongest argument for eventually keeping one reference per camera view, since
Section 13.2 requires acceptance on both face-on and down-the-line.

## Known soft spot

The turn metrics are baselined on a single address frame, and MediaPipe's depth
noise makes that baseline unstable: on the previous clip, three defensible
address frames moved `shoulder_turn_top` between 4.1 and 14.2 deg while the
golfer was motionless. Median-filtering the address joint positions over a
small window would damp it. Worth doing before the Section 9.3 ratings are
taken seriously.

## Note

`reference.json` contains derived numeric joint coordinates only — no video,
audio, or imagery. Worth a second look before any public distribution of the
repo, since the source clip is third-party watermarked footage.
