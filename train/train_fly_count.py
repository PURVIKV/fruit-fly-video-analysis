"""
Per-contour fly-count classifier.

Dataset: every contour that the shared detector (src/detection.py
find_candidates, i.e. the pipeline preprocessing without the top-2 cut)
finds in an annotated image. The label is the number of annotated fly
body points (mp = male, fp = female; mp2 is a second male point and is
not counted) that lie inside the contour:

    0 = no fly (shadow / debris), 1 = one fly, 2 = both flies touching

The labels come from the human annotation, not from the detector, so the
model is no longer trained to reproduce its own input (the old
video-based labelling in src/fly_count.py was circular).

The frame's fly count in the pipeline is the sum of the per-contour
predictions.
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.annotations import iter_annotated_images, points_inside
from src.detection import find_candidates
from src.features import COUNT_FEATURE_NAMES, count_features
from src.model_io import save_model


MODEL_PATH = "models/fly_count_model.pkl"
TREE_PATH = "results/metrics/fly_count_tree.txt"

COUNT_LABELS = ["mp", "fp"]

MAX_DEPTH = 2


def build_dataset():
    """
    One row per detected contour.

    Returns a dict with X, y, image (group id per row), session and
    n_images_without_contours.
    """

    X = []
    y = []
    images = []
    sessions = []
    empty_images = 0

    for item in iter_annotated_images():

        fly_points = []

        for label in COUNT_LABELS:
            fly_points.extend(item["labels"].get(label, []))

        candidates = find_candidates(item["image"])

        if not candidates:
            empty_images += 1

        for candidate in candidates:

            X.append(count_features(candidate))
            y.append(min(points_inside(candidate["contour"], fly_points), 2))
            images.append(item["image_id"])
            sessions.append(item["session"])

    return {
        "X": np.array(X, dtype=float),
        "y": np.array(y),
        "image": np.array(images),
        "session": np.array(sessions),
        "n_images_without_contours": empty_images
    }


def make_model():

    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        random_state=42
    )


def image_count_accuracy(y_pred, images, true_count=2):
    """
    Fraction of images whose summed per-contour prediction equals the
    true number of flies (every annotated image contains both flies).
    """

    totals = {}

    for prediction, image in zip(y_pred, images):
        totals[image] = totals.get(image, 0) + int(prediction)

    return np.mean([total == true_count for total in totals.values()])


def main():

    print("=" * 70)
    print("FRUIT FLY COUNT (PER CONTOUR) - DECISION TREE")
    print("=" * 70)

    data = build_dataset()

    X = data["X"]
    y = data["y"]
    images = data["image"]

    print("\nContours:", len(X))
    print("Images with at least one contour:", len(np.unique(images)))
    print("Images with no contour:", data["n_images_without_contours"])

    print("\nClass distribution (flies inside contour):")

    for label in [0, 1, 2]:
        print(f"{label}: {np.sum(y == label)}")

    # ---------------------------------------------------------
    # Hold-out check, grouped by image (all contours of one image
    # stay together). evaluate.py reports the full CV results.
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=images))

    model = make_model()

    model.fit(X[train_idx], y[train_idx])

    predictions = model.predict(X[test_idx])

    print("\nTraining contours:", len(train_idx))
    print("Testing contours:", len(test_idx))

    print(
        "\nPer-contour accuracy:",
        f"{accuracy_score(y[test_idx], predictions) * 100:.2f}%"
    )

    print(
        "Per-image count accuracy (sum == 2):",
        f"{image_count_accuracy(predictions, images[test_idx]) * 100:.2f}%"
    )

    print("\nConfusion Matrix (rows = true 0/1/2):")
    print(confusion_matrix(y[test_idx], predictions, labels=[0, 1, 2]))

    print(
        classification_report(
            y[test_idx],
            predictions,
            labels=[0, 1, 2],
            target_names=["0 flies", "1 fly", "2 flies"],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Final model: fit on all contours, save model and tree rules.
    # ---------------------------------------------------------

    model = make_model()

    model.fit(X, y)

    rules = export_text(model, feature_names=COUNT_FEATURE_NAMES)

    Path(TREE_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(TREE_PATH, "w", encoding="utf-8") as file:
        file.write(
            "Fly-count decision tree (fit on all "
            f"{len(X)} contours, max_depth={MAX_DEPTH})\n"
            "class = number of flies inside the contour\n\n"
        )
        file.write(rules)

    print("Tree rules:")
    print(rules)

    save_model(model, MODEL_PATH, __file__)

    print("Model saved to:", MODEL_PATH)
    print("Tree rules saved to:", TREE_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
