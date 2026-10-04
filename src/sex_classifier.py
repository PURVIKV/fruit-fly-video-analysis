import cv2
import json
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


IMAGE_DIR = Path("input/images")


def get_point(annotation, label):
    """Get the first point belonging to a specified annotation label."""

    for shape in annotation.get("shapes", []):

        if shape.get("label") == label:

            points = shape.get("points", [])

            if points:
                return tuple(points[0])

    return None


def find_contours(image):
    """Convert image to binary and find possible fly contours."""

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

    candidates = []

    height, width = binary.shape

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore objects touching the image border
        if (
            x <= 0
            or y <= 0
            or x + w >= width
            or y + h >= height
        ):
            continue

        if h == 0:
            continue

        aspect_ratio = w / h

        candidates.append({
            "contour": contour,
            "area": area,
            "aspect_ratio": aspect_ratio
        })

    return candidates


def find_contour(point, contours):
    """Find the contour containing or closest to an annotation point."""

    if point is None:
        return None

    best = None
    best_distance = -float("inf")

    for candidate in contours:

        distance = cv2.pointPolygonTest(
            candidate["contour"],
            (
                float(point[0]),
                float(point[1])
            ),
            True
        )

        if distance > best_distance:

            best_distance = distance
            best = candidate

    # Allow a small distance because the annotation point
    # may not lie exactly inside the detected contour.
    if best_distance >= -30:
        return best

    return None


def main():

    print("=" * 70)
    print("FRUIT FLY SEX CLASSIFICATION")
    print("ANNOTATION-GUIDED CONTOUR FEATURES")
    print("=" * 70)

    # Search recursively so all five image folders are included.
    json_files = sorted(
        IMAGE_DIR.rglob("*.json")
    )

    print(
        "\nJSON files found:",
        len(json_files)
    )

    if len(json_files) == 0:

        print(
            "ERROR: No JSON annotation files found."
        )

        return

    examples = []

    processed = 0
    usable = 0

    missing_images = 0
    unreadable_images = 0
    missing_annotations = 0
    insufficient_contours = 0

    for json_path in json_files:

        # ---------------------------------------------------------
        # The dataset contains both BMP and PNG images.
        # Try BMP first, then PNG.
        # ---------------------------------------------------------

        image_path = json_path.with_suffix(".bmp")

        if not image_path.exists():

            image_path = json_path.with_suffix(".png")

        if not image_path.exists():

            missing_images += 1

            continue

        processed += 1

        # ---------------------------------------------------------
        # Read image
        # ---------------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            unreadable_images += 1

            continue

        # ---------------------------------------------------------
        # Read annotation JSON
        # ---------------------------------------------------------

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as file:

                annotation = json.load(file)

        except Exception as error:

            print(
                "\nWARNING: Could not read:",
                json_path
            )

            print(error)

            continue

        # ---------------------------------------------------------
        # Get male and female reference points
        # ---------------------------------------------------------

        male_point = get_point(
            annotation,
            "mp"
        )

        female_point = get_point(
            annotation,
            "fp"
        )

        if (
            male_point is None
            or female_point is None
        ):

            missing_annotations += 1

            continue

        # ---------------------------------------------------------
        # Detect fly contours
        # ---------------------------------------------------------

        contours = find_contours(
            image
        )

        if len(contours) < 2:

            insufficient_contours += 1

            continue

        # ---------------------------------------------------------
        # Match annotation points to contours
        # ---------------------------------------------------------

        male = find_contour(
            male_point,
            contours
        )

        female = find_contour(
            female_point,
            contours
        )

        if (
            male is None
            or female is None
        ):

            continue

        # Make sure both points do not identify
        # the same contour.
        if male["contour"] is female["contour"]:

            continue

        # ---------------------------------------------------------
        # Create male example
        #
        # Label:
        # 1 = Male
        # ---------------------------------------------------------

        examples.append([
            male["area"],
            male["aspect_ratio"],
            1
        ])

        # ---------------------------------------------------------
        # Create female example
        #
        # Label:
        # 0 = Female
        # ---------------------------------------------------------

        examples.append([
            female["area"],
            female["aspect_ratio"],
            0
        ])

        usable += 1

    # -------------------------------------------------------------
    # Dataset summary
    # -------------------------------------------------------------

    print("\nDataset summary:")
    print("-" * 70)

    print(
        "Images successfully processed:",
        processed
    )

    print(
        "Missing images:",
        missing_images
    )

    print(
        "Unreadable images:",
        unreadable_images
    )

    print(
        "Images with missing annotations:",
        missing_annotations
    )

    print(
        "Images with insufficient contours:",
        insufficient_contours
    )

    print(
        "Usable annotated images:",
        usable
    )

    if usable < 20:

        print(
            "\nERROR: Not enough usable images."
        )

        return

    # -------------------------------------------------------------
    # Convert examples to NumPy arrays
    # -------------------------------------------------------------

    data = np.array(
        examples,
        dtype=float
    )

    X = data[:, :2]
    y = data[:, 2]

    print(
        "\nTotal ML examples:",
        len(X)
    )

    print(
        "Male examples:",
        int(np.sum(y == 1))
    )

    print(
        "Female examples:",
        int(np.sum(y == 0))
    )

    # -------------------------------------------------------------
    # Train/test split
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
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
        X_train,
        y_train
    )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Female",
            "Male"
        ],
        zero_division=0
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
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    print(
        "\nClassification Report:"
    )

    print(report)

    # -------------------------------------------------------------
    # Model coefficients
    # -------------------------------------------------------------

    print(
        "\nModel coefficients:"
    )

    print(
        "Area:",
        round(
            model.coef_[0][0],
            4
        )
    )

    print(
        "Aspect ratio:",
        round(
            model.coef_[0][1],
            4
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SEX CLASSIFICATION COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()