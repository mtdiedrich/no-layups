# Bundled reference swing

`reference.json` is the Section 9.1 reference artifact: a real swing in the
`swing.json` format (Section 5.2), used by `compare.py` as the baseline every
uploaded swing is rated against.

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

`meta.warnings` is now empty. It previously carried `partial_tracking` for the
trail elbow, which this view hides behind the torso through the backswing —
that flag came from masking frames on MediaPipe's `visibility` score, which
`filtering.process` no longer does. Visibility turned out to be uninformative
on golf footage rather than merely strict: a plainly-visible lead arm on
another clip scored a *median* visibility of 0.02 while being tracked
correctly. See the SPEC DEVIATION in `filtering.process`, and
`tools/tracking_report.py` to inspect the scores on any clip.

`src/no_layups/static/sample.swing.json` is the same artifact, serving as the
viewer's demo asset. It is ~1.4 MB and the browser fetches it on page load.

## Why the camera angle matters more than the pixels

An earlier reference used a face-on clip of the same golfer. Swapping to this
angled view changed the metrics far more than resolution did:

| metric | face-on clip | this clip | plausible (13.2) |
| --- | --- | --- | --- |
| shoulder_turn_top | 11.4 deg | **59.9 deg** | 60-110 |
| hip_turn_top | 9.2 deg | 25.9 deg | — |
| x_factor | 2.2 deg | **34.0 deg** | — |
| head_sway_top | 12.5 cm | 1.2 cm | <= 5 good |

(The face-on column predates the canonical-frame fix below and is kept only
for the angle comparison; the middle column is current.)

Turn metrics are read off the depth axis (Section 8.1 drops Y and measures
rotation in the ground plane). On a face-on clip the shoulder line points
almost straight at the camera at the top of the backswing, so the rotation
lives entirely in the least reliable channel and reads as a fraction of its
true value. An angled view puts the same rotation partly in the image plane
where MediaPipe is accurate.

So "monocular depth cannot measure rotation" was too strong a conclusion. It
measures rotation poorly *from a face-on view*, which is the view golfers
default to filming.

Section 13.2's ranges are all but met now — shoulder turn 59.9 against 60-110,
lead elbow 140.0 against 150-180 — but absolute turn values still read low.
Compare the deltas, not the absolutes.

Part of that gap was the canonical frame, not the camera. Section 6.1 defined
+Y as the address trunk vector, i.e. the golfer's *spine*, which at address
leans ~30 deg forward — so Section 8.1's "project onto the ground plane (drop
Y)" was dropping a plane tilted 30 deg off the ground, and the viewer rendered
the whole skeleton rotated by that angle (feet 0.394 m behind the hips, ankles
0.042 m apart in height while both were flat on the ground). `canonical.py`
now anchors +Y to world up; see its SPEC DEVIATION note. That moved
shoulder_turn_top 54.6 -> 59.0 and hip_turn_top 22.0 -> 25.9 on this clip, and
made `spine_tilt_address` measurable for the first time (5.8 deg here) — it
was previously degenerate and hardcoded to null.

World up was not the end of it. Assuming MediaPipe's Y *is* gravity left the
golfer falling forward: at address the pelvis sat 0.093 m in front of the ankle
line, the shoulders 0.384 m, the nose 0.709 m, putting the centre of mass
outside the toes. `canonical._vertical_direction` now takes +Y from the LEGS at
address — a golfer stands on their legs, so that line is near-vertical, whereas
the trunk genuinely leans and is what the previous fix moved away from.

Solving only for "pelvis back over the feet" pulls two quantities it was not
fitted to into range, which is the corroboration:

| at address | assumed camera vertical | leg-derived vertical | plausible |
| --- | --- | --- | --- |
| hip fwd of feet | 0.093 m | 0.002 m | ~0 *(this is the fitted one)* |
| spine from vertical | 39.3 deg | 32.5 deg | 25-35 for a driver |
| shoulder fwd of feet | 0.384 m | 0.250 m | 0.25-0.35 |

The rotation this removes is -7.0 deg here and -6.2 deg on an unrelated clip
shot on a different camera. Two cameras do not agree by accident, so this is a
systematic bias in MediaPipe's leg/pelvis depth rather than camera pitch —
which is why it is worth correcting generally rather than per clip. It must be
measured in a window around address only: the pelvis really does travel over
the feet during the backswing, and averaging across that left 4.4 of the 7 deg
uncorrected.

What it does not fix, because no rotation can: the nose sits 0.328 m in front
of the shoulder line and level with it, before and after. A head is ~0.15 m
forward and ~0.15 m above. That is the same non-rigid world-landmark head
described at the end of this section, and it is the reason the rendered
skeleton's head still reads wrong even with the body balanced.

+X moved for the same underlying reason. Its azimuth — the frame's rotation
about vertical — is an atan2 of a depth difference over a body width, so
taken from the hip line at one frame it is ~5 cm of depth error over 0.19 m,
about 15 deg of rotation applied to the whole skeleton. It showed: the feet
rendered 13.9 deg out of square in the viewer's face-on preset, where they
should be parallel to the screen. `canonical._stance_direction` now takes it
from the ankles, median-averaged over address..top where the feet are planted.

The feet land square by construction, so the corroboration is what else moved
without being asked to:

| at address | hip-derived +X | ankle-derived +X | sanity |
| --- | --- | --- | --- |
| ankle line | +13.9 deg | -1.2 deg | should be square |
| shoulder line | +25.8 deg | +11.6 deg | a few deg closed is normal |
| head_sway_top | 11.3 cm | 1.7 cm | 11.3 rated "attention" for McIlroy |
| spine_tilt_address | 18.6 deg | 5.3 deg | ~5-10 deg is normal |
| pelvis X drift | 0.176 m | 0.122 m | real is 0.05-0.15 m |

Every turn metric and joint angle is untouched: `turn_angle` and `angle_at`
measure angles BETWEEN lines, which no rotation about vertical can change.
Only the three quantities that read the X component moved, and all three
landed in range.

The origin moved too, for a related reason. MediaPipe emits *hip-centred*
coordinates: the mid-hip is (0,0,0) in every frame before the pipeline touches
it, so Section 6.1's "origin: midpoint of the hips at address" was subtracting
zero and the golfer could never translate. All pelvis motion came out instead
as the feet skating around the pinned hips — the ankles ranged 0.169 m in X
and 0.260 m in Z on a golfer whose feet are planted. `canonical._ground_anchor`
now anchors each frame to the ground contact, which inverts that:

| over address..impact | hip-anchored | ground-anchored |
| --- | --- | --- |
| lead ankle travel (X/Y/Z) | 0.169 / 0.073 / 0.260 | 0.033 / 0.005 / 0.073 |
| mid-hip travel (X/Y/Z) | 0.001 / 0.001 / 0.001 | 0.122 / 0.088 / 0.230 |

The vertical anchor tracks the *lower* ankle rather than the midpoint, because
feet only start level: the trail heel lifts through the follow-through, and a
midpoint anchor would rise with it and sink the whole body. It works — the
lead ankle holds to 0.005 m vertically while the trail ankle keeps its 0.041 m
of genuine heel lift.

This is a per-frame translation, so it is rigid within each frame and every
single-frame metric is unchanged to the decimal. Only `head_sway_top` moves:
it compares the nose across two frames, so it now measures head movement
against the ground rather than against the pelvis, which is what Section 8.3
asks for.

The remaining shortfall is the depth axis, and one problem sits behind all of
it: MediaPipe's world-landmark head is not rigid. Measured on this clip, the
upper arm's length varies 92% of its own mean across frames and the shoulder
width 79%, while the image landmarks it is derived from stay accurate. No
choice of axis or origin repairs that — it is the input. It is why hip-width
spread stays at 38% through every one of these changes, and why ground-anchored
pelvis Z travel reads 0.230 m when a golfer barely moves toward the ball at
all. X and Y are now in range; Z is not, and will not be until that input
improves.

## Smoothing

Depth gets a second, wider Savitzky-Golay pass that X and Y do not
(`filtering.smooth_depth`), because the noise is not isotropic: per-joint depth
jitter measured 3-8x the in-image jitter (right wrist 3.28 mm/frame^2 in Z
against 0.45 in X, 0.59 in Y). Mean jitter over address..impact fell from 1.39
to 0.52 mm/frame^2, and Z is no longer the dominant axis.

That pass runs ONLY on the rendering/metrics path, never on the series
segmentation reads. The two want opposite things -- `detect_takeaway_start`
looks for the frame the lead wrist starts moving, and a wide window smears
exactly that onset. Sharing one window put a cliff in the middle of the
parameter: sweeping the depth scale, this clip's address held at the verified
49 through 2.0 and collapsed to 31 at 2.5, while top and impact never moved.
Split, the render window is free to be as wide as the picture wants.

The ground anchor is smoothed harder still (`GROUND_ANCHOR_WINDOW_SCALE`). It
is a translation applied to every joint, so its noise lands on the whole
skeleton, and where a golfer stands changes far more slowly than their limbs
move. The hips are driven entirely by it -- MediaPipe pins them at the origin
-- and their jitter dropped 0.79 -> 0.13 mm/frame^2.

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

The turn metrics are still baselined on a single address frame, and MediaPipe's
depth noise makes that baseline unstable: on the previous clip, three
defensible address frames moved `shoulder_turn_top` between 4.1 and 14.2 deg
while the golfer was motionless. The frame's own +X axis no longer has this
problem — `_stance_direction` median-averages it over the planted span — but
the metrics still read the shoulder and hip lines at exactly one frame.
Applying the same median-over-a-window treatment there is the obvious next
step, and worth doing before the Section 9.3 ratings are taken seriously.

## Note

`reference.json` contains derived numeric joint coordinates only — no video,
audio, or imagery. Worth a second look before any public distribution of the
repo, since the source clip is third-party watermarked footage.
