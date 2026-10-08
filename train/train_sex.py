"""
Sex classification of a detected fly from its contour shape.

Features: contour area and bounding-box aspect ratio, exactly what the
pipeline computes for every detected fly (src/detection.py). Each
annotated male (mp) / female (fp) body point is matched to the nearest
detected contour.
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.annotations import iter_annotated_images
from src.detection import find_candidates
from src.model_io import save_model


MODEL_PATH = "models/sex_model.pkl"


def distance(
    x1,
    y1,
    x2,
    y2
):

    return np.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


def match_flies(
    contours,
    labels
):

    male_point = labels.get(
        "mp",
        []
    )

    female_point = labels.get(
        "fp",
        []
    )

    if not male_point or not female_point:
        return None

    if len(contours) < 2:
        return None

    male_x, male_y = male_point[0]
    female_x, female_y = female_point[0]

    # Match closest contour to male annotation.
    male_index = min(
        range(len(contours)),
        key=lambda i: distance(
            contours[i]["center_x"],
            contours[i]["center_y"],
            male_x,
            male_y
        )
    )

    # Match closest remaining contour to female annotation.
    remaining_indices = [
        i for i in range(len(contours))
        if i != male_index
    ]

    if not remaining_indices:
        return None

    female_index = min(
        remaining_indices,
        key=lambda i: distance(
            contours[i]["center_x"],
            contours[i]["center_y"],
            female_x,
            female_y
        )
    )

    return (
        contours[male_index],
        contours[female_index]
    )


def build_dataset():
    """
    Two examples (male = 1, female = 0) per image in which both flies
    are matched to different contours.

    Returns a dict with X ([area, aspect_ratio]), y, image, session.
    """

    X = []
    y = []
    images = []
    sessions = []

    for item in iter_annotated_images():

        matched = match_flies(
            find_candidates(item["image"]),
            item["labels"]
        )

        if matched is None:
            continue

        for fly, label in zip(matched, [1, 0]):

            X.append([
                fly["area"],
                fly["aspect_ratio"]
            ])
            y.append(label)
            images.append(item["image_id"])
            sessions.append(item["session"])

    return {
        "X": np.array(X),
        "y": np.array(y),
        "image": np.array(images),
        "session": np.array(sessions)
    }


def make_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ])


def main():

    print("=" * 70)
    print("FRUIT FLY SEX CLASSIFICATION")
    print("=" * 70)

    data = build_dataset()

    X = data["X"]
    y = data["y"]
    groups = data["image"]
    usable_images = len(np.unique(groups))

    if len(X) == 0:

        print(
            "\nERROR: No usable examples found."
        )

        return

    print(
        "\nUsable annotated images:",
        usable_images
    )

    print(
        "Total ML examples:",
        len(X)
    )

    print(
        "Feature matrix:",
        X.shape
    )

    print(
        "\nClass distribution:"
    )

    print(
        "Female:",
        np.sum(y == 0)
    )

    print(
        "Male:",
        np.sum(y == 1)
    )

    # ---------------------------------------------------------
    # Grouped train/test split.
    # Both flies from one image remain in the same split.
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(
        "\nTraining examples:",
        len(X_train)
    )

    print(
        "Testing examples:",
        len(X_test)
    )

    # ---------------------------------------------------------
    # StandardScaler + Logistic Regression.
    # ---------------------------------------------------------

    model = make_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nAccuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Female",
                "Male"
            ],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Save model.
    # ---------------------------------------------------------

    Path("models").mkdir(
        parents=True,
        exist_ok=True
    )

    save_model(
        model,
        MODEL_PATH,
        __file__
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nSex classification training completed."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()