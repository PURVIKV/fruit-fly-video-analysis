"""
Fly detection shared by the training scripts and the video pipeline.

Every model is trained on contours produced by find_candidates(), and the
pipeline calls the same function on each video frame, so training and
inference see exactly the same preprocessing.
"""

import math

import cv2


MIN_AREA = 500


def threshold_frame(frame):
    """Grayscale -> Gaussian blur -> inverted Otsu threshold."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary


def find_candidates(frame):
    """
    Return every fly-sized contour that does not touch the image border,
    sorted by area (largest first).
    """

    binary = threshold_frame(frame)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    height, width = binary.shape

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if (
            x <= 0
            or y <= 0
            or x + w >= width
            or y + h >= height
        ):
            continue

        if h == 0:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        candidates.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "center_x": moments["m10"] / moments["m00"],
            "center_y": moments["m01"] / moments["m00"]
        })

    candidates.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    return candidates


def detect_flies(frame):
    """The two largest candidates (at most two flies are in the arena)."""

    return find_candidates(frame)[:2]


def orientation_from_contour(contour):
    """
    Body-axis angle in degrees [0, 180) from second-order image moments.

    The angle is measured in image coordinates (x right, y down), so a
    positive angle rotates clockwise on screen. It gives the axis only;
    head vs. tail (the 180 degree ambiguity) is resolved separately.
    """

    moments = cv2.moments(contour)

    denominator = moments["mu20"] - moments["mu02"]

    numerator = 2 * moments["mu11"]

    if denominator == 0 and numerator == 0:
        return 0.0

    angle = math.degrees(
        0.5 * math.atan2(numerator, denominator)
    )

    if angle < 0:
        angle += 180

    return angle
