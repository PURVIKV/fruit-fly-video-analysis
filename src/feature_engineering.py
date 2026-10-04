import math
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/annotations_dataset.csv")
OUTPUT_FILE = Path("data/features_dataset.csv")


def calculate_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points.
    """
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        return float("nan")

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def calculate_angle(x1, y1, x2, y2):
    """
    Calculate the angle from point 1 to point 2.

    The result is normalized to the range 0-360 degrees.
    """
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        return float("nan")

    dx = x2 - x1
    dy = y2 - y1

    angle = math.degrees(math.atan2(dy, dx))

    if angle < 0:
        angle += 360

    return angle


def calculate_features(row):
    """
    Calculate geometric features for one annotated image.
    """

    # ---------------------------------------------------------
    # Male body geometry
    # ---------------------------------------------------------

    male_head_to_point = calculate_distance(
        row["male_head_x"],
        row["male_head_y"],
        row["male_point_x"],
        row["male_point_y"]
    )

    male_point_to_abdomen = calculate_distance(
        row["male_point_x"],
        row["male_point_y"],
        row["male_abdomen_x"],
        row["male_abdomen_y"]
    )

    male_head_to_abdomen = calculate_distance(
        row["male_head_x"],
        row["male_head_y"],
        row["male_abdomen_x"],
        row["male_abdomen_y"]
    )

    male_orientation_angle = calculate_angle(
        row["male_head_x"],
        row["male_head_y"],
        row["male_abdomen_x"],
        row["male_abdomen_y"]
    )

    # ---------------------------------------------------------
    # Male body secondary point
    # ---------------------------------------------------------

    male_point_to_point2 = calculate_distance(
        row["male_point_x"],
        row["male_point_y"],
        row["male_point2_x"],
        row["male_point2_y"]
    )

    # ---------------------------------------------------------
    # Male wing geometry
    # ---------------------------------------------------------

    wing_distance = calculate_distance(
        row["wing1_x"],
        row["wing1_y"],
        row["wing2_x"],
        row["wing2_y"]
    )

    wing1_angle = calculate_angle(
        row["male_point_x"],
        row["male_point_y"],
        row["wing1_x"],
        row["wing1_y"]
    )

    wing2_angle = calculate_angle(
        row["male_point_x"],
        row["male_point_y"],
        row["wing2_x"],
        row["wing2_y"]
    )

    # ---------------------------------------------------------
    # Female/Male relative position
    # ---------------------------------------------------------

    male_female_distance = calculate_distance(
        row["male_point_x"],
        row["male_point_y"],
        row["female_point_x"],
        row["female_point_y"]
    )

    male_to_female_angle = calculate_angle(
        row["male_point_x"],
        row["male_point_y"],
        row["female_point_x"],
        row["female_point_y"]
    )

    # ---------------------------------------------------------
    # Return all engineered features
    # ---------------------------------------------------------

    return pd.Series({
        "male_head_to_point_distance": male_head_to_point,
        "male_point_to_abdomen_distance": male_point_to_abdomen,
        "male_head_to_abdomen_distance": male_head_to_abdomen,

        "male_orientation_angle": male_orientation_angle,

        "male_point_to_point2_distance": male_point_to_point2,

        "wing_distance": wing_distance,

        "wing1_angle": wing1_angle,
        "wing2_angle": wing2_angle,

        "male_female_distance": male_female_distance,
        "male_to_female_angle": male_to_female_angle
    })


def main():

    print("=" * 70)
    print("FRUIT FLY FEATURE ENGINEERING")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():
        print("\nERROR: Input dataset not found.")
        print("Expected:", INPUT_FILE)
        return

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    print("\nLoading dataset:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("\nOriginal dataset:")
    print("Rows   :", len(df))
    print("Columns:", len(df.columns))

    # ---------------------------------------------------------
    # Calculate engineered features
    # ---------------------------------------------------------

    print("\nCalculating geometric features...")

    feature_df = df.apply(
        calculate_features,
        axis=1
    )

    # ---------------------------------------------------------
    # Combine original identifiers with features
    # ---------------------------------------------------------

    result = pd.concat(
        [
            df[["image"]],
            feature_df
        ],
        axis=1
    )

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE DATASET CREATED")
    print("=" * 70)

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nRows:", len(result))
    print("Columns:", len(result.columns))

    print("\nFeatures:")
    for column in result.columns:
        print(" -", column)

    print("\nFirst 10 rows:")
    print("-" * 70)
    print(
        result.head(10).to_string(index=False)
    )

    print("\nMissing values:")
    print("-" * 70)

    missing = result.isna().sum()

    for column, count in missing.items():
        print(f"{column:40s}: {count}")

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()