"""
Image-based male wing angle regression.

Target (unchanged from the original script, so numbers stay comparable):
the acute angle (0-90 deg) between the body axis (head mh -> abdomen ma)
and the line from the male body point mp to the wing tip mw.

Features (frame only): the male patch is rotated upright, split into
right / left halves, and the left half is mirrored so one regressor
serves both wings; HOG -> PCA -> LinearRegression (Ridge also reported).
Each annotated wing tip is assigned to a half by the sign of the cross
product of the body axis with (wing tip - centroid); if both tips fall
on the same side, the one further right is called right.

Training uses patches oriented with the ground-truth heading. The
end-to-end error orients the patch with the predicted heading from the
orientation model instead (moment axis + flip classifier), as the
pipeline does.
"""

import math
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.annotations import iter_annotated_images, match_isolated_flies
from src.detection import find_candidates, orientation_from_contour
from src.features import (
    angle_difference,
    heading_from_points,
    orientation_features,
    to_gray,
    wing_features,
    wing_side
)
from src.model_io import save_model


MODEL_PATH = "models/wing_angle_model.pkl"

N_COMPONENTS = 80

RIDGE_ALPHA = 10.0


def calculate_angle(x1, y1, x2, y2):
    """Angle of the vector from point 1 to point 2 (y axis up), 0-360."""

    angle = math.degrees(math.atan2(-(y2 - y1), x2 - x1))

    if angle < 0:
        angle += 360

    return angle


def calculate_wing_angle(body_angle, wing_angle):
    """Acute angle between the body axis and the wing line, 0-90 deg."""

    difference = abs(wing_angle - body_angle) % 360

    if difference > 180:
        difference = 360 - difference

    if difference > 90:
        difference = 180 - difference

    return difference


def wing_targets(labels):
    """Target angle for each of the first two annotated wing tips."""

    body_angle = calculate_angle(*labels["mh"][0], *labels["ma"][0])

    return [
        calculate_wing_angle(
            body_angle,
            calculate_angle(*labels["mp"][0], *wing)
        )
        for wing in labels["mw"][:2]
    ]


def side_targets(center, heading, wings, targets):
    """[right target, left target] relative to the given heading."""

    sides = [wing_side(center, heading, wing) for wing in wings]

    right = int(np.argmax(sides))

    return [targets[right], targets[1 - right]]


def build_dataset():
    """
    One entry per isolated male fly with head, abdomen, body point and
    two wing tips. Per fly (arrays of shape (n_flies, 2, d) hold the
    [right, left-mirrored] halves):

      X_gt, y_gt        halves / targets with the ground-truth heading
      X_h0, y_h0        ... with heading = moment axis
      X_h180, y_h180    ... with heading = moment axis + 180
      X_orient, axis, flip   orientation features / labels of the fly
      image, session
    """

    rows = {
        key: [] for key in [
            "X_gt", "y_gt", "X_h0", "y_h0", "X_h180", "y_h180",
            "X_orient", "axis", "flip", "image", "session"
        ]
    }

    for item in iter_annotated_images():

        labels = item["labels"]

        if (
            not labels.get("mh")
            or not labels.get("ma")
            or not labels.get("mp")
            or len(labels.get("mw", [])) < 2
        ):
            continue

        candidate = match_isolated_flies(
            find_candidates(item["image"]),
            labels
        )["male"]

        if candidate is None:
            continue

        gray = to_gray(item["image"])

        center = (candidate["center_x"], candidate["center_y"])

        wings = labels["mw"][:2]

        targets = wing_targets(labels)

        heading = heading_from_points(labels["mh"][0], labels["ma"][0])

        axis = orientation_from_contour(candidate["contour"])

        for key, used_heading in [
            ("gt", heading),
            ("h0", axis),
            ("h180", axis + 180)
        ]:
            rows["X_" + key].append(wing_features(gray, candidate, used_heading))
            rows["y_" + key].append(side_targets(center, used_heading, wings, targets))

        rows["X_orient"].append(orientation_features(gray, candidate, axis))
        rows["axis"].append(axis)
        rows["flip"].append(int(angle_difference(heading, axis) > 90))
        rows["image"].append(item["image_id"])
        rows["session"].append(item["session"])

    return {key: np.array(value) for key, value in rows.items()}


def flatten(data, fly_idx, key="gt"):
    """(X, y) rows for both halves of the given flies."""

    X = data["X_" + key][fly_idx]
    y = data["y_" + key][fly_idx]

    return X.reshape(-1, X.shape[-1]), y.reshape(-1)


def make_model(n_components=N_COMPONENTS, regressor="linear"):

    if regressor == "ridge":
        final = Ridge(alpha=RIDGE_ALPHA)
    else:
        final = LinearRegression()

    return Pipeline([
        (
            "pca",
            PCA(
                n_components=n_components,
                random_state=42
            )
        ),
        (
            "regressor",
            final
        )
    ])


def predict_angles(model, X):

    return np.clip(model.predict(X), 0, 90)


def regression_metrics(y_true, y_pred):

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred))
    }


def evaluate_split(
    data,
    train_idx,
    test_idx,
    n_components=N_COMPONENTS,
    regressor="linear",
    orientation_model=None
):
    """
    Fit on the train flies (ground-truth-oriented halves) and report the
    test error with ground-truth orientation, a train-mean baseline and,
    if an orientation model (already fit without the test flies) is
    given, the end-to-end error with the predicted orientation.
    """

    X_train, y_train = flatten(data, train_idx)
    X_test, y_test = flatten(data, test_idx)

    model = make_model(n_components, regressor)

    model.fit(X_train, y_train)

    result = {
        "gt": regression_metrics(y_test, predict_angles(model, X_test)),
        "baseline": regression_metrics(
            y_test,
            np.full(len(y_test), np.mean(y_train))
        )
    }

    if orientation_model is not None:

        flip = orientation_model.predict(data["X_orient"][test_idx])

        X_h0, y_h0 = data["X_h0"][test_idx], data["y_h0"][test_idx]
        X_h180, y_h180 = data["X_h180"][test_idx], data["y_h180"][test_idx]

        choose = flip.astype(bool)[:, None, None]

        X_pred = np.where(choose, X_h180, X_h0)
        y_pred_side = np.where(choose[:, :, 0], y_h180, y_h0)

        result["end_to_end"] = regression_metrics(
            y_pred_side.reshape(-1),
            predict_angles(model, X_pred.reshape(-1, X_pred.shape[-1]))
        )

        result["flip_error"] = np.mean(flip != data["flip"][test_idx])

    return result


def fit_orientation_without(orient_data, test_groups, group_key):
    """Orientation model fit on the orientation examples outside test_groups."""

    from train.train_orientation import make_model as make_orientation_model

    keep = ~np.isin(orient_data[group_key], list(test_groups))

    model = make_orientation_model()

    model.fit(orient_data["X"][keep], orient_data["y"][keep])

    return model


def main():

    from train.train_orientation import build_dataset as build_orientation_dataset

    print("=" * 70)
    print("FRUIT FLY MALE WING ANGLE - UPRIGHT HALF PATCH HOG/PCA REGRESSION")
    print("=" * 70)

    data = build_dataset()

    n_flies = len(data["image"])

    X_all, y_all = flatten(data, np.arange(n_flies))

    print("\nIsolated male flies with two wing tips:", n_flies)
    print("Regression examples (one per wing):", len(X_all))
    print("HOG feature length per half:", X_all.shape[1])
    print(f"Target: mean {np.mean(y_all):.1f} deg, std {np.std(y_all):.1f} deg")

    # ---------------------------------------------------------
    # Hold-out check grouped by image (both wings of a fly stay
    # together). evaluate.py reports the full CV results.
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(data["image"], groups=data["image"])
    )

    orientation_model = fit_orientation_without(
        build_orientation_dataset(),
        set(data["image"][test_idx]),
        "image"
    )

    print("\nTraining flies:", len(train_idx), " wings:", 2 * len(train_idx))
    print("Testing flies:", len(test_idx), " wings:", 2 * len(test_idx))

    for regressor in ["linear", "ridge"]:

        result = evaluate_split(
            data,
            train_idx,
            test_idx,
            regressor=regressor,
            orientation_model=orientation_model
        )

        print(f"\n{regressor}:")
        print(f"  ground-truth orientation: MAE {result['gt']['mae']:.2f} deg, "
              f"RMSE {result['gt']['rmse']:.2f} deg")
        print(f"  predicted orientation:    MAE {result['end_to_end']['mae']:.2f} deg, "
              f"RMSE {result['end_to_end']['rmse']:.2f} deg "
              f"(flip error {result['flip_error'] * 100:.1f}%)")

    print(f"\nBaseline (predict train mean): MAE {result['baseline']['mae']:.2f} deg, "
          f"RMSE {result['baseline']['rmse']:.2f} deg")

    # ---------------------------------------------------------
    # Final model: LinearRegression on all flies.
    # ---------------------------------------------------------

    model = make_model()

    model.fit(X_all, y_all)

    pca = model.named_steps["pca"]

    print("\nPCA components:", pca.n_components_)
    print("Explained variance:",
          f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

    save_model(model, MODEL_PATH, __file__)

    print("\nModel saved to:", MODEL_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
