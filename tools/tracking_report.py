"""Report per-joint MediaPipe visibility for a clip.

`overlay_pose.py` answers "did MediaPipe put the landmarks in the right
place?". This answers a different question that looks the same from the
outside: "did MediaPipe think it could SEE them?".

Those come apart. The Section 7.4 quality gate masks frames on
`visibility < 0.4` (config.VISIBILITY_THRESHOLD), and visibility is a
sigmoid predicting occlusion, not a measure of positional accuracy. The
tracker will follow a limb accurately while reporting low visibility for
it, so a clip can produce a clean overlay and still fail poor_tracking.

Use this to tell two very different situations apart:

  * visibility clustered just under the threshold (~0.3) -- the gate is
    marginally too strict and is discarding usable tracking.
  * visibility near zero -- MediaPipe genuinely believes the joint is
    hidden and is predicting its position. The overlay looks smooth
    because the prediction is smooth, not because it is right. Lowering
    the threshold here feeds invented coordinates into Section 8.

    uv run python tools/tracking_report.py YOURCLIP.mov
    uv run python tools/tracking_report.py YOURCLIP.mov --handedness left

Joints marked (metric) are the ones filtering.metric_critical_joints
treats as fatal; the rest only degrade the rendered skeleton.
"""

import argparse
from pathlib import Path

import numpy as np

from no_layups import config
from no_layups.pipeline import filtering, pose, video_io


def _longest_run_below(vis: np.ndarray, threshold: float) -> tuple[int, int]:
    """Returns (length, start_index) of the longest consecutive below-threshold run.

    The start index is the useful half: it says which frames to scrub to in
    overlay.mp4. A run that spans the whole backswing reads as the torso
    occluding the limb; one that begins right at the top of the backswing
    reads as the hands leaving a tight crop. Visibility alone cannot tell
    those apart -- MediaPipe reports ~0 for both.
    """
    longest = current = start = best_start = 0
    for i, below in enumerate(vis < threshold):
        if below:
            if current == 0:
                start = i
            current += 1
            if current > longest:
                longest, best_start = current, start
        else:
            current = 0
    return longest, best_start


def main() -> None:
    parser = argparse.ArgumentParser(prog="tracking_report")
    parser.add_argument("video")
    parser.add_argument("--handedness", choices=["right", "left"], default="right")
    args = parser.parse_args()

    video_path = Path(args.video)
    probe = video_io.probe(video_path)
    video_io.validate(probe)
    decodable = video_io.resolve_decodable_path(video_path)

    print(f"{video_path.name}: {probe.width}x{probe.height} @ {probe.fps:.2f}fps")
    print("extracting pose (this runs MediaPipe over every frame)...\n")
    raw = pose.extract(video_io.frames(decodable))

    threshold = config.VISIBILITY_THRESHOLD
    gate = config.POOR_TRACKING_MAX_MISSING_FRACTION
    critical = set(filtering.metric_critical_joints(args.handedness))

    # Scope to address..impact where possible -- that is the span the gate
    # actually judges. Fall back to the whole clip when segmentation cannot
    # place the keyframes, exactly as the pipeline does.
    smoothed = filtering.process(raw, probe.fps)
    span = f"whole clip (0..{raw.frame_count - 1})"
    lo, hi = 0, raw.frame_count - 1
    try:
        from no_layups.pipeline import segment

        keyframes = segment.detect_keyframes(
            smoothed[config.lead("wrist", args.handedness)], probe.fps
        )
        lo, hi = keyframes["address"], keyframes["impact"]
        span = f"address..impact ({lo}..{hi})"
    except Exception as exc:  # noqa: BLE001 - diagnostic tool, any failure just widens the span
        print(f"note: could not detect keyframes ({exc}); reporting over the whole clip\n")

    print(f"visibility threshold {threshold}, fails above {gate:.0%} missing, over {span}\n")
    header = (
        f"{'joint':<22}{'mean':>6}{'median':>8}{'<thr':>7}{'longest gap':>14}  verdict"
    )
    print(header)
    print("-" * len(header))

    for joint in config.JOINTS:
        vis = raw.vis[joint][lo : hi + 1]
        fraction = float((vis < threshold).mean())
        is_critical = joint in critical
        if fraction > gate:
            verdict = "FATAL" if is_critical else "warn"
        else:
            verdict = "ok"
        label = f"{joint} (metric)" if is_critical else joint
        run_len, run_start = _longest_run_below(vis, threshold)
        gap = "-" if run_len == 0 else f"{run_len}f @{lo + run_start}"
        print(
            f"{label:<22}{vis.mean():>6.2f}{np.median(vis):>8.2f}"
            f"{fraction:>6.0%}{gap:>14}  {verdict}"
        )

    print(
        "\nmean >> median means a bimodal split (some frames confident, most blind),"
        "\nso trust the median. Scrub overlay.mp4 to the longest-gap frame above to"
        "\nsee whether the joint is behind the torso or outside the frame."
    )


if __name__ == "__main__":
    main()
