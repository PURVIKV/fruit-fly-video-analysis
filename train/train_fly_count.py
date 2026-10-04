import cv2
import json
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


IMAGE_DIR = Path("input/images")
MODEL_PATH = "models/sex_model.pkl"


def load_annotation(json_path):
    """Read annotation points from one JSON file."""

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    labels = {}

    for shape in data.get("shapes", []):

        label = shape.get("label")
        points = shape.get("points", [])

        if not points:
            continue

        x, y = points[0]

        if label not in labels:
            labels[label] = []

        labels[label].append(
            (x, y)
        )

    return labels


def get_image_path(json_path):
    """Find the image corresponding to an annotation."""

    stem = json_path.stem

    candidates = [
        json_path.with_suffix(".bmp"),
        json_path.with_suffix(".png"),
        json_path.with_suffix(".jpg"),
        json_path.with_suffix(".jpeg")
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def find_fly_contours(image):
    """Detect possible fly contours."""

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

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

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = gray.shape

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if (
            x <= 0
            or y <= 0
            or x + w >= width
            or y + h >= height
        ):
            continue

        if h == 0:
            continue

        candidates.append({
            "contour": contour,
            "area": area,
            "aspect_ratio": w / h,
            "center_x": x + w / 2,
            "center_y": y + h / 2
        })

    return candidates


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


def match_contours_to_annotations(
    contours,
    labels
):
    """
    Match detected contours to male/female annotation points.
    """

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

    male_x, male_y = male_point[0]
    female_x, female_y = female_point[0]

    if len(contours) < 2:
        return None

    # Find the contour closest to the male annotation.
    male_contour = min(
        contours,
        key=lambda c: distance(
            c["center_x"],
            c["center_y"],
            male_x,
            male_y
        )
    )

    remaining = [
        c for c in contours
        if c is not male_contour
    ]

    if not remaining:
        return None

    # Find the remaining contour closest to female annotation.
    female_contour = min(
        remaining,
        key=lambda c: distance(
            c["center_x"],
            c["center_y"],
            female_x,
            female_y
        )
    )

    return male_contour, female_contour


def build_dataset():

    X = []
    y = []
    groups = []

    json_files = sorted(
        IMAGE_DIR.rglob("*.json")
    )

    print(
        "JSON files found:",
        len(json_files)
    )

    usable_images = 0

    for json_path in json_files:

        image_path = get_image_path(
            json_path
        )

        if image_path is None:
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        labels = load_annotation(
            json_path
        )

        matched = match_contours_to_annotations(
            find_fly_contours(image),
            labels
        )

        if matched is None:
            continue

        male, female = matched

        # -----------------------------------------------------
        # Original order:
        # Male = class 1
        # Female = class 0
        # -----------------------------------------------------

        male_features = [
            male["area"],
            male["aspect_ratio"]
        ]

        female_features = [
            female["area"],
            female["aspect_ratio"]
        ]

        pair_features = (
            male_features +
            female_features
        )

        # Male example.
        X.append(pair_features)
        y.append(1)
        groups.append(str(json_path))

        # Female example.
        X.append(pair_features)
        y.append(0)
        groups.append(str(json_path))

        usable_images += 1

    return (
        np.array(X),
        np.array(y),
        np.array(groups),
        usable_images
    )


def main():

    print("=" * 70)
    print("FRUIT FLY SEX CLASSIFICATION")
    print("=" * 70)

    X, y, groups, usable_images = build_dataset()

    if len(X) == 0:

        print(
            "\nERROR: No usable training examples found."
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
    #
    # Both examples from the same image stay in the same split.
    # This avoids image-level data leakage.
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
    # Logistic Regression pipeline.
    # ---------------------------------------------------------

    model = Pipeline([
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

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # Evaluation.
    # ---------------------------------------------------------

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

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        "Model saved to:"
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