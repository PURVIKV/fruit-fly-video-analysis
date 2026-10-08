import cv2
import math
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
OUTPUT_PATH = "results/videos/final_demo.mp4"

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


def draw_results(
    frame,
    flies,
    count_prediction,
    models
):

    result = frame.copy()

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

        label_y = max(
            y - 100,
            20
        )

        for line_number, text in enumerate(
            information
        ):

            cv2.putText(
                result,
                text,
                (
                    x,
                    label_y +
                    line_number * 22
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
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


def main():

    print("=" * 70)
    print("FRUIT FLY ML VIDEO ANALYSIS")
    print("=" * 70)

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
        VIDEO_PATH
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
        OUTPUT_PATH,
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
        OUTPUT_PATH
    )

    print(
        "\nStarting ML analysis..."
    )

    print(
        "Press Q to stop."
    )

    frame_count = 0
    total_time = 0.0

    while True:

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

        cv2.putText(
            result,
            f"Processing FPS: {processing_fps:.1f}",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(
            result
        )

        cv2.imshow(
            "Fruit Fly ML Analysis",
            result
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    writer.release()
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

    print(
        "Average processing FPS:",
        f"{average_fps:.2f}"
    )

    print(
        "Output video:",
        OUTPUT_PATH
    )

    print(
        "\nComplete ML pipeline finished."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()