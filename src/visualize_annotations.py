import cv2
import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path(
    "input/images/12-04_17-54-43/1.bmp"
)

JSON_PATH = Path(
    "input/images/12-04_17-54-43/1.json"
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Could not load image.")
        print("Path:", image_path)
        return None

    return image


# ============================================================
# LOAD JSON ANNOTATIONS
# ============================================================

def load_annotations(json_path):

    if not json_path.exists():

        print("ERROR: JSON file not found.")
        print("Path:", json_path)

        return None

    with open(json_path, "r", encoding="utf-8") as file:

        data = json.load(file)

    return data


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def draw_annotations(image, annotations):

    result = image.copy()

    shapes = annotations.get("shapes", [])

    for index, shape in enumerate(shapes):

        label = shape.get("label", "unknown")

        points = shape.get("points", [])

        for point in points:

            x = int(point[0])
            y = int(point[1])

            # ------------------------------------------------
            # Draw point
            # ------------------------------------------------

            cv2.circle(
                result,
                (x, y),
                7,
                (0, 0, 255),
                -1
            )

            # ------------------------------------------------
            # Draw small white outline
            # ------------------------------------------------

            cv2.circle(
                result,
                (x, y),
                9,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Draw label
            # ------------------------------------------------

            text_position = (
                x + 10,
                y - 10
            )

            cv2.putText(
                result,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    return result


# ============================================================
# PRINT ANNOTATIONS
# ============================================================

def print_annotations(annotations):

    shapes = annotations.get("shapes", [])

    print("\nAnnotations:")
    print("-" * 60)

    for index, shape in enumerate(shapes):

        label = shape.get("label", "unknown")
        points = shape.get("points", [])

        print(
            f"{index + 1}. "
            f"{label:5s} -> {points}"
        )

    print("-" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FRUIT FLY ANNOTATION VISUALIZER")
    print("=" * 70)

    print("\nImage:")
    print(IMAGE_PATH)

    print("\nAnnotation:")
    print(JSON_PATH)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(IMAGE_PATH)

    if image is None:
        return

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    annotations = load_annotations(JSON_PATH)

    if annotations is None:
        return

    # --------------------------------------------------------
    # Print annotations
    # --------------------------------------------------------

    print_annotations(annotations)

    # --------------------------------------------------------
    # Draw annotations
    # --------------------------------------------------------

    result = draw_annotations(
        image,
        annotations
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "Fruit Fly Ground Truth Annotations",
        result
    )

    print("\nAnnotation image displayed.")
    print("Press Q to close.")

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
