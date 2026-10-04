import cv2
import json
import math
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


IMAGE_DIR = Path("input/images")


def get_points(annotation, label):
    """Return all points belonging to a label."""

    points = []

    for shape in annotation.get("shapes", []):

        if shape.get("label") == label:

            shape_points = shape.get("points", [])

            for point in shape_points:
                points.append(tuple(point))

    return points


def get_point(annotation, label):
    """Return the first point for a label."""

    points = get_points(
        annotation,
        label
    )

    if points:
        return points[0]

    return None


def calculate_angle(point1, point2, reference):
    """
    Calculate the angle between the wing vector and the
    body reference vector.

    The result is restricted to approximately 0-90 degrees.
    """

    wing_vector = np.array([
        point1[0] - reference[0],
        point1[1] - reference[1]
    ], dtype=float)

    reference_vector = np.array([
        point2[0] - reference[0],
        point2[1] - reference[1]
    ], dtype=float)

    wing_norm = np.linalg.norm(
        wing_vector
    )

    reference_norm = np.linalg.norm(
        reference_vector
    )

    if (
        wing_norm == 0
        or reference_norm == 0
    ):
        return None

    cosine = np.dot(
        wing_vector,
        reference_vector
    ) / (
        wing_norm * reference_norm
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = math.degrees(
        math.acos(cosine)
    )

    # Convert obtuse angles to the equivalent
    # smaller wing angle.
    if angle > 90:
        angle = 180 - angle

    return angle


def create_features(
    wing1,
    wing2,
    male_head,
    male_point,
    male_abdomen
):
    """
    Create geometric features from the annotated wing
    and body points.
    """

    hx, hy = male_head
    px, py = male_point
    ax, ay = male_abdomen

    w1x, w1y = wing1
    w2x, w2y = wing2

    body_dx = ax - hx
    body_dy = ay - hy

    body_length = math.sqrt(
        body_dx ** 2 +
        body_dy ** 2
    )

    if body_length == 0:
        return None

    wing1_dx = w1x - px
    wing1_dy = w1y - py

    wing2_dx = w2x - px
    wing2_dy = w2y - py

    wing1_length = math.sqrt(
        wing1_dx ** 2 +
        wing1_dy ** 2
    )

    wing2_length = math.sqrt(
        wing2_dx ** 2 +
        wing2_dy ** 2
    )

    features = [
        body_dx / body_length,
        body_dy / body_length,

        wing1_dx / body_length,
        wing1_dy / body_length,

        wing2_dx / body_length,
        wing2_dy / body_length,

        wing1_length / body_length,
        wing2_length / body_length
    ]

    return features


def main():

    print("=" * 70)
    print("FRUIT FLY WING ANGLE REGRESSION")
    print("=" * 70)

    json_files = sorted(
        IMAGE_DIR.rglob("*.json")
    )

    print(
        "\nJSON files found:",
        len(json_files)
    )

    if len(json_files) == 0:

        print(
            "ERROR: No JSON files found."
        )

        return

    features = []
    wing_angles = []

    usable = 0
    missing_wings = 0
    missing_body = 0

    for json_path in json_files:

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as file:

                annotation = json.load(file)

        except Exception:

            continue

        # ---------------------------------------------------------
        # Body points
        # ---------------------------------------------------------

        male_head = get_point(
            annotation,
            "mh"
        )

        male_point = get_point(
            annotation,
            "mp"
        )

        male_abdomen = get_point(
            annotation,
            "ma"
        )

        if (
            male_head is None
            or male_point is None
            or male_abdomen is None
        ):

            missing_body += 1

            continue

        # ---------------------------------------------------------
        # Wing points
        # ---------------------------------------------------------

        wing_points = get_points(
            annotation,
            "mw"
        )

        if len(wing_points) < 2:

            missing_wings += 1

            continue

        wing1 = wing_points[0]
        wing2 = wing_points[1]

        # ---------------------------------------------------------
        # Calculate geometric wing angles.
        # ---------------------------------------------------------

        angle1 = calculate_angle(
            wing1,
            male_abdomen,
            male_point
        )

        angle2 = calculate_angle(
            wing2,
            male_abdomen,
            male_point
        )

        if (
            angle1 is None
            or angle2 is None
        ):

            continue

        # ---------------------------------------------------------
        # Features
        # ---------------------------------------------------------

        feature_vector = create_features(
            wing1,
            wing2,
            male_head,
            male_point,
            male_abdomen
        )

        if feature_vector is None:

            continue

        # ---------------------------------------------------------
        # Store TWO training examples.
        #
        # Each wing gets its own target angle.
        # ---------------------------------------------------------

        features.append(
            feature_vector
        )

        wing_angles.append(
            angle1
        )

        features.append(
            feature_vector
        )

        wing_angles.append(
            angle2
        )

        usable += 1

    print(
        "\nUsable wing annotations:",
        usable
    )

    print(
        "Images missing wing points:",
        missing_wings
    )

    print(
        "Images missing body points:",
        missing_body
    )

    if usable < 20:

        print(
            "\nERROR: Not enough usable wing data."
        )

        return

    X = np.array(
        features,
        dtype=float
    )

    y = np.array(
        wing_angles,
        dtype=float
    )

    print(
        "\nTotal regression examples:",
        len(X)
    )

    print(
        "Feature matrix shape:",
        X.shape
    )

    print(
        "\nWing angle range:"
    )

    print(
        "Minimum:",
        round(float(np.min(y)), 2),
        "degrees"
    )

    print(
        "Maximum:",
        round(float(np.max(y)), 2),
        "degrees"
    )

    # -------------------------------------------------------------
    # Train/test split
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
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

    # -------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # -------------------------------------------------------------
    # Linear Regression
    # -------------------------------------------------------------

    print(
        "\nTraining Linear Regression..."
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    # Keep predictions in valid 0-90 degree range.
    predictions = np.clip(
        predictions,
        0,
        90
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

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

    errors = np.abs(
        y_test - predictions
    )

    std_error = np.std(
        y_test - predictions
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "WING ANGLE REGRESSION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nMAE: {mae:.2f} degrees"
    )

    print(
        f"RMSE: {rmse:.2f} degrees"
    )

    print(
        f"Error standard deviation: {std_error:.2f} degrees"
    )

    print(
        f"Maximum absolute error: {np.max(errors):.2f} degrees"
    )

    print(
        "\nSample predictions:"
    )

    print(
        "-" * 50
    )

    sample_count = min(
        10,
        len(y_test)
    )

    for i in range(sample_count):

        print(
            f"Actual: {y_test[i]:6.2f}°   "
            f"Predicted: {predictions[i]:6.2f}°   "
            f"Error: {errors[i]:6.2f}°"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "WING ANGLE REGRESSION COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()