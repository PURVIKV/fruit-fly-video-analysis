"""
Features computed from a video frame / detected contour only.

Shared by the training scripts and src/pipeline.py so that training and
inference use exactly the same code.
"""


# -------------------------------------------------------------------------
# Fly count
# -------------------------------------------------------------------------

COUNT_FEATURE_NAMES = ["area"]


def count_features(candidate):
    """Features of one detected contour for the per-contour count model."""

    return [candidate["area"]]
