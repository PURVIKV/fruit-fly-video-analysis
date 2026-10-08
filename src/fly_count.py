# DEPRECATED - do not use for training or evaluation.
#
# This script labels each video frame with min(number of detected
# contours, 2) and then uses the sizes of those same contours as features,
# so the label is a deterministic function of the input (circular; the
# 100% accuracy it reports is guaranteed by construction). It also never
# saves a model.
#
# The fly-count model is now trained by train/train_fly_count.py on
# annotated images: one example per detected contour, labelled with the
# number of annotated flies (mp/fp points) inside it. Kept for reference.

import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


VIDEO_PATH = "input/videos/test1.mp4"


def preprocess_frame(frame):
    """
    Convert a frame to grayscale and threshold it.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary


def extract_contour_features(binary):
    """
    Extract contour-based features from a thresholded frame.

    The largest non-border contours are treated as possible flies.
    """

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    height, width = binary.shape

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small objects/noise.
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore objects touching the image border.
        if (
            x <= 0
            or y <= 0
            or x + w >= width
            or y + h >= height
        ):
            continue

        aspect_ratio = w / h if h != 0 else 0

        candidates.append(
            {
                "area": area,
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio
            }
        )

    # Sort largest objects first.
    candidates.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    return candidates


def create_features(video_path):
    """
    Extract numerical features from video frames.

    Since the videos contain frames with different visible
    fly configurations, the number of detected candidate
    contours is used as the target label.

    The first two candidate areas are used as features.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return None, None

    X = []
    y = []

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        binary = preprocess_frame(frame)

        candidates = extract_contour_features(binary)

        # Number of detected fly-like objects.
        count = min(len(candidates), 2)

        # First candidate features.
        if len(candidates) >= 1:
            area1 = candidates[0]["area"]
            width1 = candidates[0]["width"]
            height1 = candidates[0]["height"]
            aspect1 = candidates[0]["aspect_ratio"]
        else:
            area1 = 0
            width1 = 0
            height1 = 0
            aspect1 = 0

        # Second candidate features.
        if len(candidates) >= 2:
            area2 = candidates[1]["area"]
            width2 = candidates[1]["width"]
            height2 = candidates[1]["height"]
            aspect2 = candidates[1]["aspect_ratio"]
        else:
            area2 = 0
            width2 = 0
            height2 = 0
            aspect2 = 0

        features = [
            area1,
            width1,
            height1,
            aspect1,
            area2,
            width2,
            height2,
            aspect2
        ]

        X.append(features)
        y.append(count)

    cap.release()

    return np.array(X), np.array(y)


def main():

    print("=" * 70)
    print("FRUIT FLY COUNT MODEL")
    print("=" * 70)

    print("\nVideo:")
    print(VIDEO_PATH)

    # ---------------------------------------------------------
    # Create dataset
    # ---------------------------------------------------------

    X, y = create_features(VIDEO_PATH)

    if X is None:
        return

    if len(X) == 0:
        print("\nERROR: No frames were extracted.")
        return

    print("\nFrames processed:", len(X))

    print("\nClass distribution:")
    print("-" * 50)

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(
            f"{label} flies : {count} frames"
        )

    # ---------------------------------------------------------
    # Check whether multiple classes exist
    # ---------------------------------------------------------

    if len(unique) < 2:
        print("\nERROR: Only one fly-count class was detected.")
        print("A classifier requires at least two classes.")
        return

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # ---------------------------------------------------------
    # Decision Tree
    # ---------------------------------------------------------

    print("\nTraining Decision Tree...")

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    feature_names = [
        "area_1",
        "width_1",
        "height_1",
        "aspect_ratio_1",
        "area_2",
        "width_2",
        "height_2",
        "aspect_ratio_2"
    ]

    print("\nFeature Importance:")
    print("-" * 50)

    for name, importance in zip(
        feature_names,
        model.feature_importances_
    ):
        print(
            f"{name:20s}: {importance:.4f}"
        )

    print("\n" + "=" * 70)
    print("FLY COUNT MODEL COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()