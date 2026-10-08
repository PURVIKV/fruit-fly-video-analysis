import json
import math
import sys
import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_io import save_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


ANNOTATION_DIR = Path("input/images")
MODEL_PATH = "models/orientation_model.pkl"


def load_annotation(json_path):
    """Read annotation points from a JSON file."""

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


def calculate_angle(
    x1,
    y1,
    x2,
    y2
):
    """Calculate angle from point 1 to point 2."""

    dx = x2 - x1
    dy = y2 - y1

    angle = math.degrees(
        math.atan2(
            -dy,
            dx
        )
    )

    if angle < 0:
        angle += 360

    return angle


def calculate_distance(
    x1,
    y1,
    x2,
    y2
):
    """Calculate Euclidean distance."""

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def create_features(labels):
    """
    Create geometric features from male head,
    body point and abdomen.
    """

    if (
        not labels.get("mh")
        or not labels.get("mp")
        or not labels.get("ma")
    ):
        return None

    head_x, head_y = labels["mh"][0]
    point_x, point_y = labels["mp"][0]
    abdomen_x, abdomen_y = labels["ma"][0]

    # ---------------------------------------------------------
    # Body-axis orientation.
    # ---------------------------------------------------------

    orientation = calculate_angle(
        head_x,
        head_y,
        abdomen_x,
        abdomen_y
    )

    # ---------------------------------------------------------
    # Geometric features.
    # ---------------------------------------------------------

    head_to_point = calculate_distance(
        head_x,
        head_y,
        point_x,
        point_y
    )

    point_to_abdomen = calculate_distance(
        point_x,
        point_y,
        abdomen_x,
        abdomen_y
    )

    head_to_abdomen = calculate_distance(
        head_x,
        head_y,
        abdomen_x,
        abdomen_y
    )

    head_to_point_angle = calculate_angle(
        head_x,
        head_y,
        point_x,
        point_y
    )

    point_to_abdomen_angle = calculate_angle(
        point_x,
        point_y,
        abdomen_x,
        abdomen_y
    )

    # ---------------------------------------------------------
    # Relative angular information.
    # ---------------------------------------------------------

    angle_difference = (
        head_to_point_angle -
        point_to_abdomen_angle
    )

    # Normalize to [-180, 180].
    while angle_difference > 180:
        angle_difference -= 360

    while angle_difference < -180:
        angle_difference += 360

    features = [
        head_to_point,
        point_to_abdomen,
        head_to_abdomen,
        head_to_point_angle,
        point_to_abdomen_angle,
        angle_difference
    ]

    return features, orientation


def build_dataset():

    X = []
    y = []

    json_files = sorted(
        ANNOTATION_DIR.rglob("*.json")
    )

    print(
        "JSON files found:",
        len(json_files)
    )

    usable = 0

    for json_path in json_files:

        labels = load_annotation(
            json_path
        )

        result = create_features(
            labels
        )

        if result is None:
            continue

        features, orientation = result

        X.append(
            features
        )

        # -----------------------------------------------------
        # Two-class formulation:
        #
        # 0 = orientation from 0 to <180 degrees
        # 1 = orientation from 180 to <360 degrees
        # -----------------------------------------------------

        if orientation < 180:
            target = 0
        else:
            target = 1

        y.append(
            target
        )

        usable += 1

    return (
        np.array(X),
        np.array(y),
        usable
    )


def main():

    print("=" * 70)
    print("FRUIT FLY ORIENTATION CLASSIFICATION")
    print("=" * 70)

    X, y, usable = build_dataset()

    if len(X) == 0:

        print(
            "\nERROR: No usable annotations found."
        )

        return

    print(
        "\nUsable annotated images:",
        usable
    )

    print(
        "Feature matrix:",
        X.shape
    )

    print(
        "\nClass distribution:"
    )

    print(
        "0 - orientation < 180°:",
        np.sum(y == 0)
    )

    print(
        "1 - orientation >= 180°:",
        np.sum(y == 1)
    )

    # ---------------------------------------------------------
    # Train/test split.
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # ---------------------------------------------------------
    # Standardization + PCA + Logistic Regression.
    # ---------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "pca",
            PCA(
                n_components=0.95,
                random_state=42
            )
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
                "0-180 degrees",
                "180-360 degrees"
            ],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # PCA information.
    # ---------------------------------------------------------

    pca = model.named_steps["pca"]

    print(
        "PCA components:",
        pca.n_components_
    )

    print(
        "Explained variance:",
        f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
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
        "\nOrientation training completed."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()