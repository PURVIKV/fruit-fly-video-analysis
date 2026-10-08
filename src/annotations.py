"""
Loading of the annotated images in input/images.

Each recording session is one folder; every image has a labelme JSON with
point labels:

    mh, ma   male head / abdomen          (277 images)
    fh, fa   female head / abdomen        (217 images)
    mp, fp   male / female body point     (326 images, one per fly)
    mp2      second male body point (thorax, between mp and mh)
    mw       male wing tips               (257 images, two per image)
"""

import json

from pathlib import Path

import cv2


IMAGE_DIR = Path("input/images")

IMAGE_EXTENSIONS = [".bmp", ".png", ".jpg", ".jpeg"]


def load_annotation(json_path):
    """Return {label: [(x, y), ...]} for one JSON file."""

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    labels = {}

    for shape in data.get("shapes", []):

        points = shape.get("points", [])

        if not points:
            continue

        x, y = points[0]

        labels.setdefault(shape.get("label"), []).append((x, y))

    return labels


def get_image_path(json_path):
    """Find the image that belongs to an annotation file."""

    for extension in IMAGE_EXTENSIONS:

        image_path = json_path.with_suffix(extension)

        if image_path.exists():
            return image_path

    return None


def iter_annotated_images(image_dir=IMAGE_DIR):
    """
    Yield one dict per annotated image, in a fixed (sorted) order:

        json_path, image (BGR), labels, image_id, session

    image_id is the JSON path (unique per image); session is the name of
    the recording folder, used for leave-one-session-out evaluation.
    """

    for json_path in sorted(Path(image_dir).rglob("*.json")):

        image_path = get_image_path(json_path)

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        yield {
            "json_path": json_path,
            "image": image,
            "labels": load_annotation(json_path),
            "image_id": str(json_path),
            "session": json_path.parent.name
        }


def points_inside(contour, points):
    """Number of points that lie inside (or on) a contour."""

    return sum(
        cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
        for x, y in points
    )


FLIES = {
    # sex: (body point, head, abdomen)
    "male": ("mp", "mh", "ma"),
    "female": ("fp", "fh", "fa")
}


def match_isolated_flies(candidates, labels):
    """
    For each sex, the detected contour that contains that fly's body point
    and not the other fly's (i.e. a single, isolated fly), or None.
    Contours holding both touching flies are skipped: they cannot be
    oriented or measured as one fly.
    """

    matched = {}

    for sex, (body, _, _) in FLIES.items():

        other = "fp" if body == "mp" else "mp"

        matched[sex] = None

        for candidate in candidates:

            contour = candidate["contour"]

            if (
                points_inside(contour, labels.get(body, [])) > 0
                and points_inside(contour, labels.get(other, [])) == 0
            ):
                matched[sex] = candidate
                break

    return matched
