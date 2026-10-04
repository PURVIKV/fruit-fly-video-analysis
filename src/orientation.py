import cv2
import json
import math
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


IMAGE_DIR = Path("input/images")


def get_point(annotation, label):
    for shape in annotation.get("shapes", []):
        if shape.get("label") == label:
            points = shape.get("points", [])
            if points:
                return tuple(points[0])
    return None


def get_male_points(annotation):
    head = get_point(annotation, "mh")
    body = get_point(annotation, "mp")
    abdomen = get_point(annotation, "ma")

    return head, body, abdomen


def calculate_orientation(head, abdomen):
    """
    Calculate the basic body orientation from the male head
    and abdomen annotation points.
    """

    dx = abdomen[0] - head[0]
    dy = abdomen[1] - head[1]

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    if angle < 0:
        angle += 360

    return angle


def create_orientation_features(head, body, abdomen):
    """
    Create geometric features describing the fly orientation.
    """

    hx, hy = head
    px, py = body
    ax, ay = abdomen

    features = [
        px - hx,
        py - hy,
        ax - hx,
        ay - hy,
        ax - px,
        ay - py
    ]

    return features


def main():

    print("=" * 70)
    print("FRUIT FLY ORIENTATION ANALYSIS")
    print("=" * 70)

    json_files = sorted(
        IMAGE_DIR.rglob("*.json")
    )

    print(
        "\nJSON files found:",
        len(json_files)
    )

    if len(json_files) == 0:
        print("ERROR: No JSON files found.")
        return

    features = []
    angles = []
    image_names = []

    usable = 0

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

        head, body, abdomen = get_male_points(
            annotation
        )

        if (
            head is None
            or body is None
            or abdomen is None
        ):
            continue

        orientation = calculate_orientation(
            head,
            abdomen
        )

        feature_vector = create_orientation_features(
            head,
            body,
            abdomen
        )

        features.append(
            feature_vector
        )

        angles.append(
            orientation
        )

        image_names.append(
            json_path.name
        )

        usable += 1

    print(
        "\nUsable orientation annotations:",
        usable
    )

    if usable < 20:
        print(
            "ERROR: Not enough orientation data."
        )
        return

    X = np.array(
        features,
        dtype=float
    )

    angles = np.array(
        angles,
        dtype=float
    )

    print(
        "Feature matrix shape:",
        X.shape
    )

    print(
        "\nOrientation range:"
    )

    print(
        "Minimum angle:",
        round(float(np.min(angles)), 2)
    )

    print(
        "Maximum angle:",
        round(float(np.max(angles)), 2)
    )

    # -------------------------------------------------------------
    # Handle the 180-degree ambiguity.
    #
    # The same body axis can point in either direction.
    # We create a binary target representing whether 180 degrees
    # needs to be added to the basic orientation.
    # -------------------------------------------------------------

    orientation_class = np.where(
        angles >= 180,
        1,
        0
    )

    print(
        "\nOrientation classes:"
    )

    print(
        "0 = 0-180 degrees:",
        int(np.sum(orientation_class == 0))
    )

    print(
        "1 = 180-360 degrees:",
        int(np.sum(orientation_class == 1))
    )

    # -------------------------------------------------------------
    # Train/test split
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test, angle_train, angle_test = (
        train_test_split(
            X,
            orientation_class,
            angles,
            test_size=0.2,
            random_state=42,
            stratify=orientation_class
        )
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
    # PCA
    #
    # The reference project uses PCA before classification.
    # Our current geometric feature vector is small, so we use
    # the maximum valid number of components.
    # -------------------------------------------------------------

    n_components = min(
        X_train.shape[0],
        X_train.shape[1]
    )

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_train_pca = pca.fit_transform(
        X_train
    )

    X_test_pca = pca.transform(
        X_test
    )

    print(
        "\nPCA components:",
        n_components
    )

    print(
        "Explained variance:",
        round(
            float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ) * 100,
            2
        ),
        "%"
    )

    # -------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_pca,
        y_train
    )

    predictions = model.predict(
        X_test_pca
    )

    # -------------------------------------------------------------
    # Classification results
    # -------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ORIENTATION CLASSIFICATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "0-180 degrees",
                "180-360 degrees"
            ],
            zero_division=0
        )
    )

    # -------------------------------------------------------------
    # Convert predicted class back to orientation
    # -------------------------------------------------------------

    predicted_angles = []

    for original_angle, predicted_class in zip(
        angle_test,
        predictions
    ):

        # Body-axis angle has a 180 degree ambiguity.
        base_angle = original_angle % 180

        if predicted_class == 1:
            final_angle = base_angle + 180
        else:
            final_angle = base_angle

        predicted_angles.append(
            final_angle % 360
        )

    predicted_angles = np.array(
        predicted_angles
    )

    # -------------------------------------------------------------
    # Circular angular error
    # -------------------------------------------------------------

    angular_errors = np.abs(
        predicted_angles - angle_test
    )

    angular_errors = np.minimum(
        angular_errors,
        360 - angular_errors
    )

    print(
        "\nMean angular error:",
        round(
            float(np.mean(angular_errors)),
            2
        ),
        "degrees"
    )

    print(
        "Median angular error:",
        round(
            float(np.median(angular_errors)),
            2
        ),
        "degrees"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ORIENTATION ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()