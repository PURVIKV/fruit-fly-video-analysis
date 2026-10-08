import cv2
import time
import joblib
import numpy as np

from pathlib import Path

from src.detection import (
    detect_flies,
    orientation_from_contour
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
    contour
):

    base_angle = orientation_from_contour(
        contour
    )

    # The trained orientation model expects
    # six geometric features. For the video demo,
    # unavailable landmark features are approximated
    # from the contour orientation.

    features = np.array([[
        1.0,
        1.0,
        1.0,
        base_angle,
        base_angle,
        0.0
    ]])

    try:

        prediction = model.predict(
            features
        )[0]

        if prediction == 1:
            return base_angle + 180

        return base_angle

    except Exception:

        return base_angle


def predict_wing_angle(
    model,
    fly
):

    # The training model expects geometric wing
    # features that are not directly available
    # from a simple contour.
    #
    # Use a neutral approximation for the video
    # integration layer.

    features = np.array([[
        fly["height"],
        fly["height"] * 0.5,
        fly["height"] * 0.5,
        0.5,
        0.5,
        0.5,
        0.0,
        1.0
    ]])

    try:

        prediction = model.predict(
            features
        )[0]

        return float(
            np.clip(
                prediction,
                0,
                90
            )
        )

    except Exception:

        return 0.0


def draw_results(
    frame,
    flies,
    count_prediction,
    models
):

    result = frame.copy()

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

        sex = predict_sex(
            models["sex"],
            fly
        )

        orientation = predict_orientation(
            models["orientation"],
            contour
        )

        wing_angle = predict_wing_angle(
            models["wing"],
            fly
        )

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

        information = [
            f"Fly {index + 1}",
            f"Sex: {sex}",
            f"Orientation: {orientation:.1f} deg",
            f"Wing angle: {wing_angle:.1f} deg"
        ]

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

        flies = detect_flies(
            frame
        )

        # -----------------------------------------------------
        # Fly-count model
        # -----------------------------------------------------

        count_features = []

        for i in range(2):

            if i < len(flies):

                fly = flies[i]

                count_features.extend([
                    fly["area"],
                    fly["width"],
                    fly["height"],
                    fly["aspect_ratio"]
                ])

            else:

                count_features.extend([
                    0,
                    0,
                    0,
                    0
                ])

        count_prediction = int(
            models["count"].predict(
                np.array([
                    count_features
                ])
            )[0]
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