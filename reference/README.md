# Bundled reference swing

`reference.json` is the Section 9.1 reference artifact: a real swing in the
`swing.json` format (Section 5.2), used by `compare.py` (M6) as the baseline
every uploaded swing is rated against.

## Provenance

Rory McIlroy slow-motion driver swing, face-on, static camera. Source clip:
`YTDown.com_YouTube_RORY-MCILROY-SLOW-MOTION-DRIVER-SWING-FA_Media_6VWsMdWYMo0_001_480p.mp4`
(770x480, 234 frames, **20.69 fps**).

The source is below `config.MIN_FPS` (24), so Stage 1 rejects it as-is. It is
**retimed** to 30 fps — every real frame kept, relabelled — rather than
frame-duplicated to 30 fps, because duplicated frames insert zero-speed frames
that defeat Section 7.6 step 1's "3 consecutive frames above threshold" test:

```bash
ffmpeg -i <source>.mp4 -filter:v "setpts=(20.69/30)*PTS" -r 30 -an \
       -c:v libx264 -crf 18 rory_30fps.mp4
uv run no-layups process rory_30fps.mp4 --handedness right -o reference/reference.json
```

The footage is already slow motion, so the nominal frame rate is arbitrary;
retiming changes no geometry, and every metric except the (unused) absolute
speed series is playback-rate invariant.

Keyframes were verified manually as Section 9.1 requires — `address: 20`,
`top: 121`, `impact: 153`, checked frame-by-frame against the video rather than
trusted from the detector.

`src/no_layups/static/sample.swing.json` is the same artifact, serving as the
viewer's demo asset.

## Why a real reference and not the synthetic placeholder

Section 9.1 allows `tools/make_synthetic_reference.py` to build a placeholder
from the Appendix B poses. Do not use it for rating. Appendix B is a table of
hand-written canonical coordinates with exact rotations, so it carries **zero**
monocular depth error, while every real upload carries a lot of it. Running the
Appendix B poses through the real `metrics.py` yields `shoulder_turn_top` =
90.0 deg against this clip's 11.8 deg — a delta of 78 deg, four times Section
9.3's "attention" threshold. Rated against the synthetic reference, a tour
professional's swing scores "attention" on 6 of 8 metrics.

Section 9.2's `delta = user - reference` only cancels MediaPipe's systematic
depth bias when **both** sides carry it. That is the whole reason the rating is
reference-relative, and it is why the reference must come from real footage
through the normal pipeline.

Cancellation is still only partial: foreshortening is view-dependent, so a
down-the-line upload does not cancel against this face-on reference as cleanly
as a face-on one does. See Section 13.2, which requires acceptance on both.

## Note

`reference.json` contains derived numeric joint coordinates only — no video,
audio, or imagery. Worth a second look before any public distribution of the
repo, since the source clip is third-party broadcast footage.
