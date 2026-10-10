import cv2
import json
import math
import platform
import shlex
import shutil
import subprocess
import time
import joblib
import numpy as np

from pathlib import Path

from src.detection import (
    find_candidates,
    orientation_from_contour
)
from src.features import (
    count_features,
    orientation_features,
    to_gray,
    wing_features
)


VIDEO_PATH = "input/videos/test1.mp4"
OUTPUT_DIR = Path("results/videos")
STATS_DIR = Path("results/metrics/pipeline_stats")


def output_paths(video_path):
    """Annotated video and timing file, named after the input video."""

    stem = Path(video_path).stem

    return (
        OUTPUT_DIR / f"final_demo_{stem}.mp4",
        STATS_DIR / f"{stem}.json"
    )

FLY_COUNT_MODEL = "models/fly_count_model.pkl"
SEX_MODEL = "models/sex_model.pkl"
ORIENTATION_MODEL = "models/orientation_model.pkl"
WING_MODEL = "models/wing_angle_model.pkl"


def load_models():

    models = {}

    print("\nLoading trained models...")

    models["count"] = joblib.load(
        FLY_COUNT_MODEL
    )

    models["sex"] = joblib.load(
        SEX_MODEL
    )

    models["orientation"] = joblib.load(
        ORIENTATION_MODEL
    )

    models["wing"] = joblib.load(
        WING_MODEL
    )

    print("All 4 models loaded successfully.")

    return models


def predict_fly_count(
    model,
    candidates
):
    """
    Predict the number of flies in every contour.

    Returns the frame's fly count (sum over contours) and the contours
    that contain at least one fly, each tagged with "fly_count".
    """

    if not candidates:
        return 0, []

    counts = model.predict(
        np.array([
            count_features(candidate)
            for candidate in candidates
        ])
    )

    flies = []

    for candidate, count in zip(candidates, counts):

        candidate["fly_count"] = int(count)

        if count > 0:
            flies.append(candidate)

    return int(np.sum(counts)), flies


def predict_sex(
    model,
    fly
):

    features = np.array([[
        fly["area"],
        fly["aspect_ratio"]
    ]])

    prediction = model.predict(
        features
    )[0]

    if prediction == 1:
        return "Male"

    return "Female"


def predict_orientation(
    model,
    gray,
    fly
):
    """
    Heading of the fly (abdomen -> head) in image-coordinate degrees.

    The moment axis gives the body line; the flip classifier (HOG of the
    upright patch -> PCA -> LogisticRegression) decides which end is the
    head. Same feature code as train/train_orientation.py.
    """

    axis = orientation_from_contour(
        fly["contour"]
    )

    features = orientation_features(
        gray,
        fly,
        axis
    )

    flip = int(
        model.predict(
            features.reshape(1, -1)
        )[0]
    )

    return (axis + 180 * flip) % 360


def display_angle(heading):
    """Image-coordinate heading -> counter-clockwise degrees (y up)."""

    return (-heading) % 360


def draw_heading(
    image,
    fly,
    heading,
    length=60
):

    start = (
        int(fly["center_x"]),
        int(fly["center_y"])
    )

    end = (
        int(start[0] + length * math.cos(math.radians(heading))),
        int(start[1] + length * math.sin(math.radians(heading)))
    )

    cv2.arrowedLine(
        image,
        start,
        end,
        (0, 0, 255),
        2,
        tipLength=0.3
    )


def predict_wing_angle(
    model,
    gray,
    fly,
    heading
):
    """
    Right and left wing angles (deg, 0-90) of a male fly.

    The patch is rotated upright with the predicted heading, split into
    halves (left half mirrored), and each half goes through
    HOG -> PCA -> LinearRegression, as in train/train_wing_angle.py.
    """

    right, left = wing_features(
        gray,
        fly,
        heading
    )

    angles = np.clip(
        model.predict(
            np.array([right, left])
        ),
        0,
        90
    )

    return float(angles[0]), float(angles[1])


LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.7
LABEL_THICKNESS = 2
LABEL_PADDING = 6
LABEL_LINE_GAP = 8


def overlaps(a, b):
    """True if rectangles a and b, given as (left, top, right, bottom), overlap."""

    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_text_block(
    image,
    lines,
    box,
    placed=None
):
    """
    Draw white text lines on a filled dark rectangle next to a fly.

    The block goes just above the fly's bounding box (x, y, w, h), or
    below it if there is no room above or that spot overlaps a block in
    `placed` (blocks already drawn in this frame). If both spots are
    taken it is pushed down until it is clear. It is always clamped
    fully inside the frame, and its rectangle is appended to `placed`.
    """

    if placed is None:
        placed = []

    frame_height, frame_width = image.shape[:2]

    sizes = [
        cv2.getTextSize(
            text,
            LABEL_FONT,
            LABEL_SCALE,
            LABEL_THICKNESS
        )[0]
        for text in lines
    ]

    line_height = max(size[1] for size in sizes) + LABEL_LINE_GAP

    block_width = max(size[0] for size in sizes) + 2 * LABEL_PADDING
    block_height = line_height * len(lines) + 2 * LABEL_PADDING

    x, y, w, h = box

    left = min(max(x, 0), frame_width - block_width)

    def rect(top):
        top = min(max(top, 0), frame_height - block_height)
        return (left, top, left + block_width, top + block_height)

    above = y - block_height - 6
    below = y + h + 6

    candidates = [rect(below), rect(above)] if above < 0 else [rect(above), rect(below)]

    chosen = next(
        (
            candidate for candidate in candidates
            if not any(overlaps(candidate, other) for other in placed)
        ),
        None
    )

    if chosen is None:

        chosen = candidates[0]

        while any(overlaps(chosen, other) for other in placed):

            lowest = max(other[3] for other in placed if overlaps(chosen, other))

            if lowest + 2 + block_height > frame_height:
                break

            chosen = rect(lowest + 2)

    placed.append(chosen)

    left, top = chosen[0], chosen[1]

    cv2.rectangle(
        image,
        (left, top),
        (left + block_width, top + block_height),
        (30, 30, 30),
        -1
    )

    for line_number, text in enumerate(lines):

        baseline_y = (
            top +
            LABEL_PADDING +
            (line_number + 1) * line_height -
            LABEL_LINE_GAP // 2
        )

        cv2.putText(
            image,
            text,
            (left + LABEL_PADDING, baseline_y),
            LABEL_FONT,
            LABEL_SCALE,
            (255, 255, 255),
            LABEL_THICKNESS,
            cv2.LINE_AA
        )


def draw_results(
    frame,
    flies,
    count_prediction,
    models
):

    result = frame.copy()

    placed_labels = []

    gray = to_gray(frame)

    for index, fly in enumerate(flies):

        contour = fly["contour"]

        x = fly["x"]
        y = fly["y"]
        w = fly["width"]
        h = fly["height"]

        center_x = int(
            fly["center_x"]
        )

        center_y = int(
            fly["center_y"]
        )

        # -----------------------------------------------------
        # ML predictions
        # -----------------------------------------------------

        # A contour predicted to hold two touching flies cannot be
        # split, so per-fly models are not run on it.

        if fly["fly_count"] == 1:

            sex = predict_sex(
                models["sex"],
                fly
            )

            heading = predict_orientation(
                models["orientation"],
                gray,
                fly
            )

            draw_heading(
                result,
                fly,
                heading
            )

            # The wing model is trained on male flies only.

            if sex == "Male":

                right_wing, left_wing = predict_wing_angle(
                    models["wing"],
                    gray,
                    fly,
                    heading
                )

                wing_text = (
                    f"Wings R/L: {right_wing:.0f} / "
                    f"{left_wing:.0f} deg"
                )

            else:

                wing_text = "Wings: n/a (female)"

            information = [
                f"Fly {index + 1}",
                f"Sex: {sex}",
                f"Orientation: {display_angle(heading):.1f} deg",
                wing_text
            ]

        else:

            information = [
                f"Blob {index + 1}: "
                f"{fly['fly_count']} flies touching"
            ]

        # -----------------------------------------------------
        # Draw contour
        # -----------------------------------------------------

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.circle(
            result,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        # -----------------------------------------------------
        # Text
        # -----------------------------------------------------

        draw_text_block(
            result,
            information,
            (x, y, w, h),
            placed_labels
        )

    # ---------------------------------------------------------
    # Main information panel
    # ---------------------------------------------------------

    cv2.rectangle(
        result,
        (10, 10),
        (420, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result,
        f"ML Fly Count: {count_prediction}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        "Fruit Fly ML Analysis",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return result


def reencode_h264(path):
    """
    Re-encode a video to H.264 / yuv420p (plays in PowerPoint, browsers
    and QuickTime) with ffmpeg. If ffmpeg is not installed, print the
    exact command instead. Returns the new path, or None.
    """

    source = Path(path)
    target = source.with_name(source.stem + "_h264.mp4")

    command = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(target)
    ]

    if shutil.which("ffmpeg") is None:
        print("\nffmpeg is not installed. To make an H.264 copy, install it")
        print("(e.g. brew install ffmpeg) and run:")
        print("  " + shlex.join(command))
        return None

    print("\nRe-encoding to H.264:", target)

    subprocess.run(command, check=True, capture_output=True)

    return target


def machine_description():

    cpu = platform.processor() or "unknown CPU"

    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass

    return (
        f"{cpu}, {platform.system()} {platform.machine()}, "
        f"Python {platform.python_version()}, OpenCV {cv2.__version__}"
    )


def main(
    video_path=VIDEO_PATH,
    max_frames=None,
    display=True,
    h264=False
):
    """
    Run all models on a video and write the annotated output.

    max_frames stops early (quick smoke test); display=False skips the
    preview window. Timing is written to results/metrics/pipeline_stats.json.
    """

    print("=" * 70)
    print("FRUIT FLY ML VIDEO ANALYSIS")
    print("=" * 70)

    output_path, stats_path = output_paths(video_path)

    Path(
        "results/videos"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    models = load_models()

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "\nERROR: Could not open video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("\nVideo:")
    print(
        "FPS:",
        fps
    )
    print(
        "Resolution:",
        width,
        "x",
        height
    )
    print(
        "Total frames:",
        total_frames
    )

    # ---------------------------------------------------------
    # Output writer
    # ---------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print(
            "\nERROR: Could not create output video."
        )

        cap.release()

        return

    print(
        "\nOutput:",
        output_path
    )

    print(
        "\nStarting ML analysis..."
    )

    print(
        "Press Q to stop."
    )

    frame_count = 0
    total_time = 0.0
    wall_start = time.perf_counter()

    while max_frames is None or frame_count < max_frames:

        start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------------------
        # Fly-count model: one prediction per detected contour
        # (0, 1 or 2 flies); the frame count is their sum.
        # -----------------------------------------------------

        count_prediction, flies = predict_fly_count(
            models["count"],
            find_candidates(frame)
        )

        # -----------------------------------------------------
        # Draw all ML results
        # -----------------------------------------------------

        result = draw_results(
            frame,
            flies,
            count_prediction,
            models
        )

        elapsed = (
            time.perf_counter() -
            start
        )

        total_time += elapsed

        frame_count += 1

        processing_fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0
        )

        draw_text_block(
            result,
            [f"Processing FPS: {processing_fps:.1f}"],
            (10, height, 0, 0)
        )

        writer.write(
            result
        )

        if display:

            cv2.imshow(
                "Fruit Fly ML Analysis",
                result
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    wall_time = time.perf_counter() - wall_start

    cap.release()
    writer.release()

    if display:
        cv2.destroyAllWindows()

    average_fps = (
        frame_count /
        total_time
        if total_time > 0
        else 0
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL PIPELINE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nFrames processed:",
        frame_count
    )

    wall_fps = (
        frame_count /
        wall_time
        if wall_time > 0
        else 0
    )

    # Processing FPS times read + detection + all models + drawing;
    # wall-clock FPS also includes writing the video and the preview.

    print(
        "Average processing FPS:",
        f"{average_fps:.2f}"
    )

    print(
        "Wall-clock FPS (incl. video writing/display):",
        f"{wall_fps:.2f}"
    )

    stats = {
        "video": str(video_path),
        "resolution": f"{width} x {height}",
        "input_fps": fps,
        "frames_processed": frame_count,
        "max_frames": max_frames,
        "display": display,
        "processing_fps": round(average_fps, 2),
        "wall_clock_fps": round(wall_fps, 2),
        "machine": machine_description(),
        "output": str(output_path)
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    if h264:
        reencode_h264(output_path)

    print(
        "Output video:",
        output_path
    )

    print(
        "\nComplete ML pipeline finished."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()