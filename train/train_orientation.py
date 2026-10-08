"""
Image-based fly orientation (Stanford CS229 fruit-fly approach).

1. Axis angle (0-180 deg) from the image moments of the fly's contour
   (src/detection.py orientation_from_contour).
2. Fixed-size patch around the contour centroid, rotated so that axis
   points up (src/features.py).
3. HOG -> PCA -> LogisticRegression predicts whether the head is at the
   bottom of the patch, i.e. the heading is axis + 180 deg ("flip").

Ground truth heading = direction abdomen -> head from the annotated
head/abdomen points (male: mh/ma, female: fh/fa). The features use only
the frame and the detected contour, never the annotated points.

Errors reported:
  (a) flip-classifier error
  (b) final angular error, |predicted heading - true heading| on the circle
  (c) baseline: always predict the training set's majority flip
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.annotations import FLIES, iter_annotated_images, match_isolated_flies
from src.detection import find_candidates, orientation_from_contour
from src.features import (
    angle_difference,
    heading_from_points,
    orientation_features,
    to_gray
)
from src.model_io import save_model


MODEL_PATH = "models/orientation_model.pkl"

N_COMPONENTS = 20


def build_dataset():
    """
    One example per isolated, head/abdomen-annotated fly (both sexes).

    Returns a dict with X (HOG), y (1 = flip), image, session, sex,
    axis (moment axis angle) and heading (true heading).
    """

    rows = {
        "X": [], "y": [], "image": [], "session": [],
        "sex": [], "axis": [], "heading": []
    }

    for item in iter_annotated_images():

        labels = item["labels"]

        candidates = find_candidates(item["image"])

        matched = match_isolated_flies(candidates, labels)

        gray = to_gray(item["image"])

        for sex, (_, head, abdomen) in FLIES.items():

            candidate = matched[sex]

            if candidate is None or not labels.get(head) or not labels.get(abdomen):
                continue

            heading = heading_from_points(labels[head][0], labels[abdomen][0])

            axis = orientation_from_contour(candidate["contour"])

            rows["X"].append(orientation_features(gray, candidate, axis))
            rows["y"].append(int(angle_difference(heading, axis) > 90))
            rows["image"].append(item["image_id"])
            rows["session"].append(item["session"])
            rows["sex"].append(sex)
            rows["axis"].append(axis)
            rows["heading"].append(heading)

    return {key: np.array(value) for key, value in rows.items()}


def make_model(n_components=N_COMPONENTS):

    return Pipeline([
        (
            "pca",
            PCA(
                n_components=n_components,
                random_state=42
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ])


def predicted_heading(axis, flip):

    return (np.asarray(axis) + 180 * np.asarray(flip)) % 360


def angular_errors(heading, axis, flip):
    """Circular error (deg) of heading = axis + 180 * flip."""

    return np.array([
        angle_difference(true, predicted)
        for true, predicted in zip(heading, predicted_heading(axis, flip))
    ])


def evaluate_split(data, train_idx, test_idx, n_components=N_COMPONENTS):
    """Fit on train_idx; return metrics on test_idx (model + baseline)."""

    y = data["y"]

    model = make_model(n_components)

    model.fit(data["X"][train_idx], y[train_idx])

    flip = model.predict(data["X"][test_idx])

    majority = int(np.mean(y[train_idx]) > 0.5)

    baseline_flip = np.full(len(test_idx), majority)

    errors = angular_errors(data["heading"][test_idx], data["axis"][test_idx], flip)

    baseline_errors = angular_errors(
        data["heading"][test_idx],
        data["axis"][test_idx],
        baseline_flip
    )

    oracle_errors = angular_errors(
        data["heading"][test_idx],
        data["axis"][test_idx],
        y[test_idx]
    )

    return {
        "flip_error": np.mean(flip != y[test_idx]),
        "angle_mae": np.mean(errors),
        "angle_median": np.median(errors),
        "baseline_flip_error": np.mean(baseline_flip != y[test_idx]),
        "baseline_angle_mae": np.mean(baseline_errors),
        "axis_only_mae": np.mean(oracle_errors),
        "flip_pred": flip
    }


def main():

    print("=" * 70)
    print("FRUIT FLY ORIENTATION - MOMENTS + HOG/PCA/LOGISTIC FLIP")
    print("=" * 70)

    data = build_dataset()

    X = data["X"]
    y = data["y"]

    print("\nExamples (isolated annotated flies):", len(X))

    for sex in FLIES:
        print(f"  {sex}: {np.sum(data['sex'] == sex)}")

    print("HOG feature length:", X.shape[1])
    print("Flip labels: 0 (head = axis):", np.sum(y == 0),
          " 1 (head = axis + 180):", np.sum(y == 1))

    # ---------------------------------------------------------
    # Hold-out check grouped by image. evaluate.py reports the full
    # random / GroupKFold / leave-one-session-out results.
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=data["image"]))

    result = evaluate_split(data, train_idx, test_idx)

    print("\nTraining examples:", len(train_idx))
    print("Testing examples:", len(test_idx))

    print(f"\n(a) Flip error:              {result['flip_error'] * 100:.2f}%")
    print(f"(b) Angular error:           mean {result['angle_mae']:.1f} deg, "
          f"median {result['angle_median']:.1f} deg")
    print(f"(c) Baseline (majority flip): flip error "
          f"{result['baseline_flip_error'] * 100:.2f}%, "
          f"angular error {result['baseline_angle_mae']:.1f} deg")
    print(f"    Moment axis with perfect flip (lower bound): "
          f"{result['axis_only_mae']:.1f} deg")

    for sex in FLIES:

        mask = data["sex"][test_idx] == sex

        if mask.any():
            errors = angular_errors(
                data["heading"][test_idx][mask],
                data["axis"][test_idx][mask],
                result["flip_pred"][mask]
            )
            print(f"    {sex}: flip error "
                  f"{np.mean(result['flip_pred'][mask] != y[test_idx][mask]) * 100:.1f}%, "
                  f"angular error {np.mean(errors):.1f} deg (n={mask.sum()})")

    # ---------------------------------------------------------
    # Final model: fit on all examples and save.
    # ---------------------------------------------------------

    model = make_model()

    model.fit(X, y)

    pca = model.named_steps["pca"]

    print("\nPCA components:", pca.n_components_)
    print("Explained variance:",
          f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

    save_model(model, MODEL_PATH, __file__)

    print("\nModel saved to:", MODEL_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
