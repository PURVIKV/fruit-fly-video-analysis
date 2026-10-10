"""
Frame-level fly count on the test videos.

The arena always holds two flies (one male, one female: the source is a
courtship video, and every annotated image has exactly one mp and one
fp), so the correct count in every frame is 2. For each
input/videos/test*.mp4 this runs the current detector + count model
(exactly as src/pipeline.py does) and reports:

  - frames decoded vs. the frame count in the file's metadata
  - % of frames whose predicted count == 2, and the count distribution
  - the longest run of consecutive wrong frames
  - frames with a touching-flies contour (predicted to hold 2 flies):
    counted correctly, but no sex / orientation / wing is predicted

Outputs:
    results/metrics/video_counts.md
    results/debug/video_counts.csv     one row per frame (git-ignored)
    results/debug/fail_*.png           example failure frames (git-ignored)

Run from the repository root:  python eval_video_counts.py
"""

import argparse
import struct

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from src.detection import (
    DETECTOR,
    MIN_AREA,
    arena_mask,
    find_candidates,
    threshold_frame
)
from src.pipeline import FLY_COUNT_MODEL, predict_fly_count


VIDEO_DIR = Path("input/videos")
REPORT_PATH = Path("results/metrics/video_counts.md")
DEBUG_DIR = Path("results/debug")

TRUE_COUNT = 2
N_FAILURE_EXAMPLES = 6

# Which videos overlap the annotated training images
# (input/images/README and input/videos/README.txt).
HELD_OUT = {
    "test1": "NO: session 12-08_22-00-00 was cut from test1",
    "test4": "NO: session 12-08_11-15-00 was cut from test4",
}
HELD_OUT_DEFAULT = (
    "probably: no annotated session is documented as cut from it "
    "(12-04_17-54-43 was 'sampled uniformly from video', source unstated)"
)


# -------------------------------------------------------------------------
# Why decoded frames != metadata frame count
# -------------------------------------------------------------------------

def mp4_video_track_info(path):
    """
    Read the MP4 sample table of the video track: number of stored frames
    and how many leading frames the edit list tells players to skip.

    The test clips were cut from a longer recording without re-encoding,
    so each file starts at the previous keyframe and uses an edit list to
    hide those pre-roll frames. OpenCV's CAP_PROP_FRAME_COUNT reports the
    stored frames; decoding honours the edit list.
    """

    data = Path(path).read_bytes()

    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}

    tracks = []
    current = {}

    def walk(start, end):

        nonlocal current

        i = start

        while i + 8 <= end:

            size, box = struct.unpack(">I4s", data[i:i + 8])
            header = 8

            if size == 1:
                size = struct.unpack(">Q", data[i + 8:i + 16])[0]
                header = 16
            elif size == 0:
                size = end - i

            body = data[i + header:i + size]

            if box in containers:
                walk(i + header, i + size)
            elif box == b"hdlr":
                current["type"] = body[8:12]
            elif box == b"elst":
                version = body[0]
                if version == 0:
                    current["media_time"] = struct.unpack(">i", body[12:16])[0]
                else:
                    current["media_time"] = struct.unpack(">q", body[16:24])[0]
            elif box == b"stts":
                current["sample_delta"] = struct.unpack(">I", body[12:16])[0]
            elif box == b"stsz":
                current["samples"] = struct.unpack(">I", body[8:12])[0]
                tracks.append(current)
                current = {}

            i += size

    walk(0, len(data))

    for track in tracks:

        if track.get("type") == b"vide":

            skipped = track.get("media_time", 0) // track.get("sample_delta", 1)

            return track["samples"], skipped

    return None, None


# -------------------------------------------------------------------------
# Counting
# -------------------------------------------------------------------------

def count_video(video_path, count_model):
    """Per-frame predicted counts for one video (same code as the pipeline)."""

    capture = cv2.VideoCapture(str(video_path))

    rows = []

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        candidates = find_candidates(frame)

        count, _ = predict_fly_count(count_model, candidates)

        rows.append({
            "video": Path(video_path).stem,
            "frame": len(rows),
            "time_s": round(len(rows) / capture.get(cv2.CAP_PROP_FPS), 3),
            "contours": len(candidates),
            "count": count,
            "correct": count == TRUE_COUNT,
            "touching": any(c["fly_count"] == 2 for c in candidates)
        })

    capture.release()

    return rows


def longest_wrong_run(correct):
    """(length, first frame) of the longest run of consecutive wrong frames."""

    best = (0, None)
    run_start = None

    for index, ok in enumerate(list(correct) + [True]):

        if not ok and run_start is None:
            run_start = index

        if ok and run_start is not None:
            best = max(best, (index - run_start, run_start), key=lambda item: item[0])
            run_start = None

    return best


def evaluate_videos(count_model, video_dir=VIDEO_DIR):
    """
    Run all test videos. Returns (summary DataFrame, per-frame DataFrame).
    """

    summaries = []
    frames = []

    for video_path in sorted(Path(video_dir).glob("test*.mp4")):

        capture = cv2.VideoCapture(str(video_path))
        metadata_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()

        stored, skipped = mp4_video_track_info(video_path)

        rows = count_video(video_path, count_model)

        frames += rows

        counts = np.array([row["count"] for row in rows])
        correct = counts == TRUE_COUNT

        run_length, run_start = longest_wrong_run(correct)

        touching = np.array([row["touching"] for row in rows])

        summaries.append({
            "video": video_path.stem,
            "held_out": HELD_OUT.get(video_path.stem, HELD_OUT_DEFAULT),
            "metadata_frames": metadata_frames,
            "edit_list_skipped": skipped,
            "decoded_frames": len(rows),
            "count_eq_2_pct": 100 * correct.mean(),
            "count_0": int(np.sum(counts == 0)),
            "count_1": int(np.sum(counts == 1)),
            "count_2": int(np.sum(counts == 2)),
            "count_3plus": int(np.sum(counts >= 3)),
            "touching_frames": int(touching.sum()),
            "touching_pct": 100 * touching.mean(),
            "longest_wrong_run": run_length,
            "longest_wrong_run_s": run_length / fps,
            "longest_wrong_run_start": run_start
        })

    return pd.DataFrame(summaries), pd.DataFrame(frames)


# -------------------------------------------------------------------------
# Debug images and report
# -------------------------------------------------------------------------

def save_failure_examples(frames, count_model, n_examples=N_FAILURE_EXAMPLES):
    """
    Save example wrong frames, spread over videos and time. Green = kept
    contours (with predicted count), red = fly-sized contours dropped
    because they touch the image border, yellow circle = arena mask.
    """

    wrong = frames[~frames["correct"]]

    if wrong.empty:
        return []

    picks = wrong.iloc[np.linspace(0, len(wrong) - 1, min(n_examples, len(wrong))).astype(int)]

    saved = []

    for _, row in picks.iterrows():

        capture = cv2.VideoCapture(str(VIDEO_DIR / f"{row['video']}.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = capture.read()
        capture.release()

        if not ok:
            continue

        image = frame.copy()

        contours, _ = cv2.findContours(
            threshold_frame(frame),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        large = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

        cv2.drawContours(image, large, -1, (0, 0, 255), 3)

        candidates = find_candidates(frame)

        predict_fly_count(count_model, candidates)

        for candidate in candidates:

            cv2.drawContours(image, [candidate["contour"]], -1, (0, 200, 0), 3)

            cv2.putText(
                image,
                str(candidate["fly_count"]),
                (candidate["x"], max(candidate["y"] - 8, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 140, 0),
                3
            )

        if DETECTOR["arena_mask"]:
            mask_contours, _ = cv2.findContours(
                arena_mask(frame.shape),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, mask_contours, -1, (0, 220, 255), 2)

        cv2.rectangle(image, (0, 0), (900, 60), (30, 30, 30), -1)

        cv2.putText(
            image,
            f"{row['video']} frame {row['frame']}: predicted {row['count']}, true 2",
            (12, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2
        )

        path = DEBUG_DIR / f"fail_{row['video']}_{int(row['frame']):04d}.png"

        cv2.imwrite(str(path), cv2.resize(image, (765, 765)))

        saved.append(path)

    return saved


def write_report(summary, examples):

    total_correct = np.sum(summary["count_eq_2_pct"] * summary["decoded_frames"] / 100)

    total_frames = summary["decoded_frames"].sum()

    lines = [
        "# Frame-level fly count on the test videos",
        "",
        "Generated by `python eval_video_counts.py`. The arena always holds two "
        "flies (courtship video; every annotated image has exactly one male "
        "`mp` and one female `fp`), so the correct count is 2 in every frame.",
        "",
        f"Detector: `{DETECTOR}`. Count model: `{FLY_COUNT_MODEL}`.",
        "",
        "| Video | Held out? | Metadata frames | Edit-list skipped | Decoded | "
        "Count == 2 | 0 / 1 / 2 / 3+ | Longest wrong run | Touching-flies frames |",
        "|---|---|---|---|---|---|---|---|---|"
    ]

    for _, row in summary.iterrows():

        run = (
            f"{row['longest_wrong_run']} frames ({row['longest_wrong_run_s']:.1f} s) "
            f"from frame {int(row['longest_wrong_run_start'])}"
            if row["longest_wrong_run"] else "none"
        )

        lines.append(
            f"| {row['video']} | {row['held_out']} | {row['metadata_frames']} | "
            f"{row['edit_list_skipped']} | {row['decoded_frames']} | "
            f"{row['count_eq_2_pct']:.1f}% | {row['count_0']} / {row['count_1']} / "
            f"{row['count_2']} / {row['count_3plus']} | {run} | "
            f"{row['touching_frames']} ({row['touching_pct']:.1f}%) |"
        )

    lines += [
        "",
        f"All videos: count == 2 in {total_correct:.0f} / {total_frames} frames "
        f"({100 * total_correct / total_frames:.1f}%).",
        "",
        "Touching-flies frames: frames where the two flies form one contour "
        "(area above the 10,442 px split), predicted as 2 flies. The count is "
        "right, but sex, orientation and wing angle are not predicted there.",
        "",
        "## Decoded vs. metadata frame count",
        "",
        "The clips were cut from the original recording without re-encoding. "
        "Each file therefore starts at the keyframe before the cut and stores "
        "the frames in between, and an MP4 edit list tells players to skip "
        "them. OpenCV's `CAP_PROP_FRAME_COUNT` reports all stored frames "
        "(test1: 368), while decoding honours the edit list (test1 skips 18 "
        "frames, leaving 350 for the 10.06 s edit; OpenCV returns 351 because "
        "of rounding at the end of the edit). No frames are lost.",
        "",
        "## Example failure frames",
        "",
        "Saved in `results/debug/` (git-ignored). Green = kept contour with its "
        "predicted count; red = fly-sized contour dropped because it touches "
        "the image border; yellow = arena mask (if enabled).",
        ""
    ]

    lines += [f"- `{path}`" for path in examples]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    parser.add_argument("--model", default=FLY_COUNT_MODEL)

    args = parser.parse_args()

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    count_model = joblib.load(args.model)

    summary, frames = evaluate_videos(count_model)

    frames.to_csv(DEBUG_DIR / "video_counts.csv", index=False)

    for old in DEBUG_DIR.glob("fail_*.png"):
        old.unlink()

    examples = save_failure_examples(frames, count_model)

    write_report(summary, examples)

    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
