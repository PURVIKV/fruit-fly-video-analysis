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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


ANNOTATION_DIR = Path("input/images")
MODEL_PATH = "models/wing_angle_model.pkl"


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


def calculate_distance(
    x1,
    y1,
    x2,
    y2
):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def calculate_angle(
    x1,
    y1,
    x2,
    y2
):
    """
    Calculate the angle of a vector.
    """

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


def calculate_wing_angle(
    body_angle,
    wing_angle
):
    """
    Calculate the acute angle between the
    body axis and wing direction.

    Result is restricted to 0-90 degrees.
    """

    difference = abs(
        wing_angle -
        body_angle
    )

    difference = difference % 360

    if difference > 180:
        difference = 360 - difference

    if difference > 90:
        difference = 180 - difference

    return difference


def create_features(
    labels,
    wing_point
):
    """
    Create geometric features for one wing.
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

    wing_x, wing_y = wing_point

    # ---------------------------------------------------------
    # Body geometry.
    # ---------------------------------------------------------

    body_length = calculate_distance(
        head_x,
        head_y,
        abdomen_x,
        abdomen_y
    )

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

    # ---------------------------------------------------------
    # Wing geometry.
    # ---------------------------------------------------------

    wing_distance = calculate_distance(
        point_x,
        point_y,
        wing_x,
        wing_y
    )

    wing_to_head_distance = calculate_distance(
        head_x,
        head_y,
        wing_x,
        wing_y
    )

    wing_to_abdomen_distance = calculate_distance(
        abdomen_x,
        abdomen_y,
        wing_x,
        wing_y
    )

    body_angle = calculate_angle(
        head_x,
        head_y,
        abdomen_x,
        abdomen_y
    )

    wing_angle = calculate_angle(
        point_x,
        point_y,
        wing_x,
        wing_y
    )

    relative_angle = (
        wing_angle -
        body_angle
    )

    while relative_angle > 180:
        relative_angle -= 360

    while relative_angle < -180:
        relative_angle += 360

    # ---------------------------------------------------------
    # Normalized features.
    # ---------------------------------------------------------

    if body_length > 0:

        normalized_wing_distance = (
            wing_distance /
            body_length
        )

        normalized_head_distance = (
            wing_to_head_distance /
            body_length
        )

        normalized_abdomen_distance = (
            wing_to_abdomen_distance /
            body_length
        )

    else:

        normalized_wing_distance = 0
        normalized_head_distance = 0
        normalized_abdomen_distance = 0

    features = [
        body_length,
        head_to_point,
        point_to_abdomen,
        normalized_wing_distance,
        normalized_head_distance,
        normalized_abdomen_distance,
        math.sin(math.radians(relative_angle)),
        math.cos(math.radians(relative_angle))
    ]

    target = calculate_wing_angle(
        body_angle,
        wing_angle
    )

    return features, target


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

    usable_images = 0
    images_missing_wings = 0
    images_missing_body = 0

    for json_path in json_files:

        labels = load_annotation(
            json_path
        )

        if (
            not labels.get("mh")
            or not labels.get("mp")
            or not labels.get("ma")
        ):

            images_missing_body += 1
            continue

        wing_points = labels.get(
            "mw",
            []
        )

        if len(wing_points) < 2:

            images_missing_wings += 1
            continue

        usable_images += 1

        # -----------------------------------------------------
        # Create one regression example for each wing.
        # -----------------------------------------------------

        for wing_point in wing_points[:2]:

            result = create_features(
                labels,
                wing_point
            )

            if result is None:
                continue

            features, target = result

            X.append(
                features
            )

            y.append(
                target
            )

    return (
        np.array(X),
        np.array(y),
        usable_images,
        images_missing_wings,
        images_missing_body
    )


def main():

    print("=" * 70)
    print("FRUIT FLY WING ANGLE REGRESSION")
    print("=" * 70)

    (
        X,
        y,
        usable_images,
        missing_wings,
        missing_body
    ) = build_dataset()

    if len(X) == 0:

        print(
            "\nERROR: No usable wing annotations found."
        )

        return

    print(
        "\nImages with usable wing annotations:",
        usable_images
    )

    print(
        "Images missing wing annotations:",
        missing_wings
    )

    print(
        "Images missing body annotations:",
        missing_body
    )

    print(
        "Total regression examples:",
        len(X)
    )

    print(
        "Feature matrix:",
        X.shape
    )

    print(
        "\nTarget wing-angle range:"
    )

    print(
        "Minimum:",
        f"{np.min(y):.2f} degrees"
    )

    print(
        "Maximum:",
        f"{np.max(y):.2f} degrees"
    )

    # ---------------------------------------------------------
    # Train/test split.
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        "\nTraining examples:",
        len(X_train)
    )

    print(
        "Testing examples:",
        len(X_test)
    )

    # ---------------------------------------------------------
    # StandardScaler + PCA + Linear Regression.
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
            "regressor",
            LinearRegression()
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    # Keep predictions within physical range.
    predictions = np.clip(
        predictions,
        0,
        90
    )

    # ---------------------------------------------------------
    # Evaluation.
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    absolute_errors = np.abs(
        y_test -
        predictions
    )

    max_error = np.max(
        absolute_errors
    )

    std_error = np.std(
        y_test -
        predictions
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
        "\nMean Absolute Error:",
        f"{mae:.2f} degrees"
    )

    print(
        "Root Mean Squared Error:",
        f"{rmse:.2f} degrees"
    )

    print(
        "Standard Error:",
        f"{std_error:.2f} degrees"
    )

    print(
        "Maximum Absolute Error:",
        f"{max_error:.2f} degrees"
    )

    # ---------------------------------------------------------
    # PCA information.
    # ---------------------------------------------------------

    pca = model.named_steps["pca"]

    print(
        "\nPCA components:",
        pca.n_components_
    )

    print(
        "Explained variance:",
        f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Example predictions.
    # ---------------------------------------------------------

    print(
        "\nSample predictions:"
    )

    for actual, predicted in zip(
        y_test[:10],
        predictions[:10]
    ):

        error = abs(
            actual -
            predicted
        )

        print(
            f"Actual: {actual:6.2f}° | "
            f"Predicted: {predicted:6.2f}° | "
            f"Error: {error:6.2f}°"
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
        "\nWing-angle training completed."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()