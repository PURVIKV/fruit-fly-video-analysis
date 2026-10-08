"""
Evaluation of all four models under three splitting schemes.

    random  original-style random 80/20 split over rows, 10 seeds
    image   5-fold GroupKFold, group = annotated image
    session leave-one-session-out, group = recording folder (strictest:
            frames of one session are near-duplicates of each other)

Dataset builders, model factories and per-split scoring are imported from
the trainers, so nothing here re-implements a model.

Outputs:
    results/metrics/evaluation.csv   one row per task / split / metric
    results/metrics/evaluation.md    the same as tables (mean +- std)
    results/plots/orientation_pca.png, results/plots/wing_pca.png
        error vs number of PCA components, leave-one-session-out

Run from the repository root:  python evaluate.py
"""

import warnings

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    train_test_split
)

import train.train_fly_count as fly_count
import train.train_orientation as orientation
import train.train_sex as sex
import train.train_wing_angle as wing


METRICS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")

SPLITS = ["random", "image", "session"]

SPLIT_NAMES = {
    "random": "Random 80/20 (10 seeds)",
    "image": "GroupKFold by image (5)",
    "session": "Leave-one-session-out"
}

N_SEEDS = 10

PCA_GRID = [2, 5, 10, 20, 40, 80, 120]

# Plot colours (validated categorical slots 1-2, neutral baseline).
COLOR_1 = "#2a78d6"
COLOR_2 = "#eb6834"
COLOR_BASELINE = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def split_indices(n_rows, images, sessions, scheme):
    """(train_idx, test_idx) pairs for one splitting scheme."""

    rows = np.arange(n_rows)

    if scheme == "random":
        return [
            train_test_split(rows, test_size=0.20, random_state=seed)
            for seed in range(N_SEEDS)
        ]

    if scheme == "image":
        return list(GroupKFold(n_splits=5).split(rows, groups=images))

    return list(LeaveOneGroupOut().split(rows, groups=sessions))


# -------------------------------------------------------------------------
# Per-task fold scoring
# -------------------------------------------------------------------------

def score_fly_count(data, scheme):

    folds = []

    for train_idx, test_idx in split_indices(
        len(data["y"]), data["image"], data["session"], scheme
    ):

        model = fly_count.make_model()
        model.fit(data["X"][train_idx], data["y"][train_idx])

        prediction = model.predict(data["X"][test_idx])

        fold = {
            "contour accuracy (%)":
                100 * accuracy_score(data["y"][test_idx], prediction)
        }

        # Per-image count only makes sense when an image's contours are
        # all in the test fold.
        if scheme != "random":
            fold["image count accuracy (%)"] = 100 * fly_count.image_count_accuracy(
                prediction,
                data["image"][test_idx]
            )

        folds.append(fold)

    return folds


def score_sex(data, scheme):

    folds = []

    for train_idx, test_idx in split_indices(
        len(data["y"]), data["image"], data["session"], scheme
    ):

        model = sex.make_model()
        model.fit(data["X"][train_idx], data["y"][train_idx])

        folds.append({
            "accuracy (%)": 100 * accuracy_score(
                data["y"][test_idx],
                model.predict(data["X"][test_idx])
            )
        })

    return folds


def score_orientation(data, scheme, n_components=orientation.N_COMPONENTS):

    folds = []

    for train_idx, test_idx in split_indices(
        len(data["y"]), data["image"], data["session"], scheme
    ):

        result = orientation.evaluate_split(
            data, train_idx, test_idx, min(n_components, len(train_idx))
        )

        folds.append({
            "flip error (%)": 100 * result["flip_error"],
            "angular error, mean (deg)": result["angle_mae"],
            "angular error, median (deg)": result["angle_median"],
            "baseline flip error (%)": 100 * result["baseline_flip_error"],
            "baseline angular error (deg)": result["baseline_angle_mae"],
            "axis-only error, perfect flip (deg)": result["axis_only_mae"]
        })

    return folds


def score_wing_random(data):
    """Original style: random split over individual wings (rows)."""

    X, y = wing.flatten(data, np.arange(len(data["image"])))

    folds = []

    for seed in range(N_SEEDS):

        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.20, random_state=seed
        )

        fold = {}

        for regressor, name in [("linear", "MAE linear"), ("ridge", "MAE ridge")]:

            model = wing.make_model(regressor=regressor)
            model.fit(X[train_idx], y[train_idx])

            fold[f"{name}, true orientation (deg)"] = wing.regression_metrics(
                y[test_idx], wing.predict_angles(model, X[test_idx])
            )["mae"]

        fold["MAE baseline, train mean (deg)"] = np.mean(
            np.abs(y[test_idx] - np.mean(y[train_idx]))
        )

        folds.append(fold)

    return folds


def score_wing(data, orient_data, scheme, n_components=wing.N_COMPONENTS):
    """
    Grouped splits over flies (both wings stay together). The end-to-end
    error uses an orientation model fit without the test images/sessions.
    """

    if scheme == "random":
        return score_wing_random(data)

    group_key = "image" if scheme == "image" else "session"

    folds = []

    for train_idx, test_idx in split_indices(
        len(data["image"]), data["image"], data["session"], scheme
    ):

        orientation_model = wing.fit_orientation_without(
            orient_data,
            set(data[group_key][test_idx]),
            group_key
        )

        n_components = min(n_components, 2 * len(train_idx))

        linear = wing.evaluate_split(
            data, train_idx, test_idx, n_components, "linear",
            orientation_model
        )

        ridge = wing.evaluate_split(
            data, train_idx, test_idx, n_components, "ridge"
        )

        folds.append({
            "MAE linear, true orientation (deg)": linear["gt"]["mae"],
            "MAE ridge, true orientation (deg)": ridge["gt"]["mae"],
            "MAE linear, predicted orientation (deg)": linear["end_to_end"]["mae"],
            "RMSE linear, predicted orientation (deg)": linear["end_to_end"]["rmse"],
            "MAE baseline, train mean (deg)": linear["baseline"]["mae"]
        })

    return folds


# -------------------------------------------------------------------------
# Aggregation and reports
# -------------------------------------------------------------------------

def summarise(task, scheme, folds):

    rows = []

    for metric in folds[0]:

        values = np.array([fold[metric] for fold in folds if metric in fold])

        rows.append({
            "task": task,
            "split": scheme,
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(),
            "n_folds": len(values)
        })

    return rows


def write_markdown(table, sizes, sessions, path):

    lines = [
        "# Evaluation",
        "",
        "Generated by `python evaluate.py`. Values are mean ± std over the "
        "folds / seeds of each scheme.",
        "",
        "| Scheme | What is held out |",
        "|---|---|",
        f"| {SPLIT_NAMES['random']} | random 20% of rows (original-style; "
        "rows from one image or session can be on both sides) |",
        f"| {SPLIT_NAMES['image']} | whole images (all contours / both wings "
        "of an image stay together) |",
        f"| {SPLIT_NAMES['session']} | a whole recording folder (strictest) |",
        ""
    ]

    titles = {
        "fly count": "Fly count (per contour, decision tree on area)",
        "sex": "Sex (logistic regression on area + aspect ratio)",
        "orientation": "Orientation (moment axis + HOG/PCA/logistic flip, both sexes)",
        "wing": "Male wing angle (upright half-patch HOG/PCA/regression)"
    }

    for task in titles:

        subset = table[table["task"] == task]

        lines += [f"## {titles[task]}", "", sizes[task], ""]

        header = "| Metric | " + " | ".join(SPLIT_NAMES[s] for s in SPLITS) + " |"

        lines += [header, "|---" * (len(SPLITS) + 1) + "|"]

        for metric in dict.fromkeys(subset["metric"]):

            cells = []

            for scheme in SPLITS:

                row = subset[(subset["metric"] == metric) & (subset["split"] == scheme)]

                if row.empty:
                    cells.append("—")
                else:
                    cells.append(
                        f"{row['mean'].iloc[0]:.1f} ± {row['std'].iloc[0]:.1f}"
                    )

            lines.append(f"| {metric} | " + " | ".join(cells) + " |")

        lines.append("")

    lines += [
        "Notes:",
        "",
        "- Fly count: the per-image count (sum of contour predictions == 2) "
        "is only computed when all contours of an image are in the test "
        "fold. Images with no usable contour (flies merged with the arena "
        "wall) are not in the dataset, so it is an upper bound.",
        "- Wing: the random scheme splits individual wings, as the original "
        "script did, so it has no end-to-end column (that needs whole flies).",
        "- Leave-one-session-out only has a fold for sessions that contain "
        "examples of that task:",
        ""
    ]

    for task, task_sessions in sessions.items():
        lines.append(
            f"  - {task}: {len(task_sessions)} sessions "
            f"({', '.join(task_sessions)})"
        )

    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def style_axis(axis, title, ylabel):

    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY)
    axis.set_xlabel("PCA components", color=TEXT_SECONDARY)
    axis.set_ylabel(ylabel, color=TEXT_SECONDARY)
    axis.set_xscale("log")
    axis.set_xticks(PCA_GRID)
    axis.set_xticklabels([str(n) for n in PCA_GRID])
    axis.minorticks_off()
    axis.tick_params(colors=TEXT_SECONDARY)
    axis.grid(axis="y", color="#e4e3df", linewidth=0.8)
    axis.set_ylim(bottom=0)

    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        axis.spines[side].set_color("#c9c8c2")


def line(axis, x, y, color, label):

    axis.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)


def baseline_line(axis, value, label):

    axis.axhline(value, color=COLOR_BASELINE, linewidth=1.5, linestyle="--")
    axis.annotate(label, (PCA_GRID[-1], value), xytext=(0, 4), ha="right",
                  textcoords="offset points", color=TEXT_SECONDARY, fontsize=9)


def chosen_marker(axis, n_components):

    axis.axvline(n_components, color="#c9c8c2", linewidth=1, linestyle=":")
    axis.annotate(f"used: {n_components}", (n_components, 0), xytext=(4, 4),
                  textcoords="offset points", color=TEXT_SECONDARY, fontsize=9)


def plot_pca_curves(orient_data, wing_data):
    """Error vs number of PCA components, leave-one-session-out."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Orientation: two measures with different units -> two panels.
    flip, angle, baseline_flip, baseline_angle = [], [], [], []

    for n in PCA_GRID:
        folds = score_orientation(orient_data, "session", n)
        flip.append(np.mean([f["flip error (%)"] for f in folds]))
        angle.append(np.mean([f["angular error, mean (deg)"] for f in folds]))
        baseline_flip.append(np.mean([f["baseline flip error (%)"] for f in folds]))
        baseline_angle.append(np.mean([f["baseline angular error (deg)"] for f in folds]))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)

    line(axes[0], PCA_GRID, flip, COLOR_1, "HOG/PCA/logistic")
    baseline_line(axes[0], baseline_flip[0], "majority-flip baseline")
    style_axis(axes[0], "Orientation: flip error", "flip error (%)")
    chosen_marker(axes[0], orientation.N_COMPONENTS)

    line(axes[1], PCA_GRID, angle, COLOR_1, "HOG/PCA/logistic")
    baseline_line(axes[1], baseline_angle[0], "majority-flip baseline")
    style_axis(axes[1], "Orientation: mean angular error", "degrees")
    chosen_marker(axes[1], orientation.N_COMPONENTS)

    figure.suptitle("Leave-one-session-out, mean over "
                    f"{len(np.unique(orient_data['session']))} sessions "
                    "(components capped at the training-set size)",
                    x=0.01, ha="left", fontsize=9, color=TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "orientation_pca.png", dpi=150)
    plt.close(figure)

    # Wing: same unit (degrees) -> one panel, two series + baseline.
    true_mae, predicted_mae, baseline = [], [], []

    for n in PCA_GRID:
        folds = score_wing(wing_data, orient_data, "session", n)
        true_mae.append(np.mean([f["MAE linear, true orientation (deg)"] for f in folds]))
        predicted_mae.append(np.mean([f["MAE linear, predicted orientation (deg)"] for f in folds]))
        baseline.append(np.mean([f["MAE baseline, train mean (deg)"] for f in folds]))

    figure, axis = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)

    line(axis, PCA_GRID, true_mae, COLOR_1, "true orientation")
    line(axis, PCA_GRID, predicted_mae, COLOR_2, "predicted orientation")
    baseline_line(axis, baseline[0], "baseline: train mean")
    style_axis(axis, "Male wing angle: MAE (LinearRegression)", "degrees")
    chosen_marker(axis, wing.N_COMPONENTS)

    axis.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower left")

    figure.suptitle("Leave-one-session-out, mean over "
                    f"{len(np.unique(wing_data['session']))} sessions "
                    "(components capped at the training-set size)",
                    x=0.01, ha="left", fontsize=9, color=TEXT_SECONDARY)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "wing_pca.png", dpi=150)
    plt.close(figure)


def main():

    warnings.filterwarnings("ignore")

    print("Building datasets...")

    data = {
        "fly count": fly_count.build_dataset(),
        "sex": sex.build_dataset(),
        "orientation": orientation.build_dataset(),
        "wing": wing.build_dataset()
    }

    sizes = {
        "fly count": f"{len(data['fly count']['y'])} contours from "
                     f"{len(np.unique(data['fly count']['image']))} images.",
        "sex": f"{len(data['sex']['y'])} flies from "
               f"{len(np.unique(data['sex']['image']))} images.",
        "orientation": f"{len(data['orientation']['y'])} isolated flies "
                       f"({np.sum(data['orientation']['sex'] == 'male')} male, "
                       f"{np.sum(data['orientation']['sex'] == 'female')} female).",
        "wing": f"{len(data['wing']['image'])} isolated males, "
                f"{2 * len(data['wing']['image'])} wings."
    }

    for task, text in sizes.items():
        print(f"  {task}: {text}")

    rows = []

    for scheme in SPLITS:

        print("Evaluating:", SPLIT_NAMES[scheme])

        rows += summarise("fly count", scheme, score_fly_count(data["fly count"], scheme))
        rows += summarise("sex", scheme, score_sex(data["sex"], scheme))
        rows += summarise("orientation", scheme, score_orientation(data["orientation"], scheme))
        rows += summarise("wing", scheme, score_wing(data["wing"], data["orientation"], scheme))

    table = pd.DataFrame(rows)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    table.to_csv(METRICS_DIR / "evaluation.csv", index=False, float_format="%.4f")

    sessions = {
        task: [str(name) for name in np.unique(task_data["session"])]
        for task, task_data in data.items()
    }

    write_markdown(table, sizes, sessions, METRICS_DIR / "evaluation.md")

    print("Plotting PCA curves (leave-one-session-out)...")

    plot_pca_curves(data["orientation"], data["wing"])

    print((METRICS_DIR / "evaluation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
