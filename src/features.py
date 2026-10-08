"""
Features computed from a video frame / detected contour only.

Shared by the training scripts and src/pipeline.py so that training and
inference use exactly the same code.
"""

import math

import cv2
import numpy as np

from skimage.feature import hog


# -------------------------------------------------------------------------
# Fly count
# -------------------------------------------------------------------------

COUNT_FEATURE_NAMES = ["area"]


def count_features(candidate):
    """Features of one detected contour for the per-contour count model."""

    return [candidate["area"]]


# -------------------------------------------------------------------------
# Upright patches + HOG (orientation and wing angle)
#
# Angles are in image coordinates, in degrees: 0 = +x (right), 90 = +y
# (down on screen), i.e. measured clockwise on screen. A "heading" is the
# direction from the abdomen towards the head.
# -------------------------------------------------------------------------

PATCH_PX = 240           # side of the square crop in frame pixels
PATCH_OUT = 96           # crop is resized to PATCH_OUT x PATCH_OUT
WING_PATCH_PX = 288      # larger crop so spread wing tips stay inside
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (12, 12),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}


def to_gray(frame):

    if frame.ndim == 2:
        return frame

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def heading_from_points(head, tail):
    """Image-coordinate angle of the vector tail -> head, in [0, 360)."""

    return math.degrees(
        math.atan2(head[1] - tail[1], head[0] - tail[0])
    ) % 360


def angle_difference(a, b):
    """Absolute circular difference between two angles, in [0, 180]."""

    difference = abs(a - b) % 360

    return min(difference, 360 - difference)


def patch_transform(center, heading, patch_px=PATCH_PX):
    """
    2x3 affine matrix mapping frame pixels to the upright patch: the
    point `center` goes to the patch centre and the direction `heading`
    points up (towards row 0). Coordinates are in the resized patch.
    """

    scale = PATCH_OUT / patch_px

    # cv2 rotates counter-clockwise on screen for positive angles, which
    # subtracts from an image-coordinate angle; heading -> -90 (up).
    matrix = cv2.getRotationMatrix2D(
        (float(center[0]), float(center[1])),
        heading + 90,
        scale
    )

    matrix[0, 2] += PATCH_OUT / 2 - center[0]
    matrix[1, 2] += PATCH_OUT / 2 - center[1]

    return matrix


def upright_patch(gray, center, heading, patch_px=PATCH_PX):
    """Fixed-size grayscale patch around center, rotated so heading is up."""

    return cv2.warpAffine(
        gray,
        patch_transform(center, heading, patch_px),
        (PATCH_OUT, PATCH_OUT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def hog_features(patch):

    return hog(patch, feature_vector=True, **HOG_PARAMS)


def orientation_features(gray, candidate, axis_angle):
    """
    HOG of the patch rotated so the moment axis (axis_angle, 0-180)
    points up. The flip classifier decides whether the head is at the
    top (heading = axis_angle) or the bottom (axis_angle + 180).
    """

    center = (candidate["center_x"], candidate["center_y"])

    return hog_features(upright_patch(gray, center, axis_angle))


def wing_side(center, heading, point):
    """
    Signed cross product of the heading direction with (point - center),
    in image coordinates. Positive = the fly's right-hand side of the
    upright patch (patch column > centre), negative = left.
    """

    ux = math.cos(math.radians(heading))
    uy = math.sin(math.radians(heading))

    return ux * (point[1] - center[1]) - uy * (point[0] - center[0])


def wing_features(gray, candidate, heading):
    """
    HOG of the right half and of the mirrored left half of the upright
    male patch. Mirroring makes a left wing look like a right wing, so
    one regressor serves both. Returns (right, left).
    """

    center = (candidate["center_x"], candidate["center_y"])

    patch = upright_patch(gray, center, heading, WING_PATCH_PX)

    half = PATCH_OUT // 2

    right = patch[:, half:]
    left = np.fliplr(patch[:, :half])

    return hog_features(right), hog_features(left)
