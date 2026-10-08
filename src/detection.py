"""
Fly detection shared by the training scripts and the video pipeline.

Every model is trained on contours produced by find_candidates(), and the
pipeline calls the same function on each video frame, so training and
inference see exactly the same preprocessing.

The detector has three switches (DETECTOR below):

    arena_mask     AND the binary image with a filled circle of radius
                   (rows - 1) // 2 centred in the square frame, so
                   nothing outside the arena can form or join a blob
                   (img_to_mask in the original Stanford repo)
    threshold      "otsu"  Gaussian blur + Otsu over the whole frame
                   "core"  fixed threshold 115: only the dark body core
                           (head/thorax/abdomen) remains, legs and wings
                           drop out (threshold_core in the original)
                   "wings" median blur, threshold at (mean intensity in
                           the arena - 5), 9x9 erosion: keeps the wings
                           (threshold_wings in the original)
    border_filter  drop contours that touch the image border
"""

import math

import cv2
import numpy as np


MIN_AREA = 500

CORE_THRESHOLD = 115
WINGS_MEAN_OFFSET = 5
WINGS_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

DETECTOR = {
    "arena_mask": False,
    "threshold": "otsu",
    "border_filter": True
}


def set_detector(**settings):
    """Change detector switches (used by the experiments in J)."""

    unknown = set(settings) - set(DETECTOR)

    if unknown:
        raise KeyError(f"unknown detector settings: {unknown}")

    DETECTOR.update(settings)


_MASKS = {}


def arena_mask(shape):
    """Filled circle of radius (rows - 1) // 2 centred in a square frame."""

    rows, cols = shape[:2]

    if (rows, cols) not in _MASKS:

        radius = (rows - 1) // 2

        mask = np.zeros((rows, cols), np.uint8)

        cv2.circle(mask, (cols // 2, rows // 2), radius, 255, -1)

        _MASKS[(rows, cols)] = mask

    return _MASKS[(rows, cols)]


def threshold_frame(frame):
    """Binary image (fly = 255) according to the DETECTOR settings."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = arena_mask(gray.shape)

    if DETECTOR["threshold"] == "otsu":

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    elif DETECTOR["threshold"] == "core":

        _, binary = cv2.threshold(
            gray,
            CORE_THRESHOLD,
            255,
            cv2.THRESH_BINARY_INV
        )

    elif DETECTOR["threshold"] == "wings":

        blurred = cv2.medianBlur(gray, 5)

        level = gray[mask == 255].mean() - WINGS_MEAN_OFFSET

        _, binary = cv2.threshold(
            blurred,
            level,
            255,
            cv2.THRESH_BINARY_INV
        )

        binary = cv2.erode(binary, WINGS_ERODE_KERNEL, iterations=1)

    else:
        raise ValueError(f"unknown threshold {DETECTOR['threshold']}")

    if DETECTOR["arena_mask"]:
        binary = cv2.bitwise_and(binary, mask)

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

        if DETECTOR["border_filter"] and (
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
