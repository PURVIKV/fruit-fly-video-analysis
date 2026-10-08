import cv2
import json
import numpy as np
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_io import save_model
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

    for extension in [
        ".bmp",
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        image_path = json_path.with_suffix(
            extension
        )

        if image_path.exists():
            return image_path

    return None


def find_contours(image):

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

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        center_x = (
            moments["m10"] /
            moments["m00"]
        )

        center_y = (
            moments["m01"] /
            moments["m00"]
        )

        candidates.append({
            "area": area,
            "aspect_ratio": w / h,
            "center_x": center_x,
            "center_y": center_y
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

        contours = find_contours(
            image
        )

        matched = match_flies(
            contours,
            labels
        )

        if matched is None:
            continue

        male, female = matched

        # -----------------------------------------------------
        # One ML example for the male fly.
        # Class 1 = Male
        # -----------------------------------------------------

        X.append([
            male["area"],
            male["aspect_ratio"]
        ])

        y.append(1)

        groups.append(
            str(json_path)
        )

        # -----------------------------------------------------
        # One ML example for the female fly.
        # Class 0 = Female
        # -----------------------------------------------------

        X.append([
            female["area"],
            female["aspect_ratio"]
        ])

        y.append(0)

        groups.append(
            str(json_path)
        )

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