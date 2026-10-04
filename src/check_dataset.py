import pandas as pd

DATASET_PATH = "data/annotations_dataset.csv"


def main():
    print("=" * 70)
    print("CHECKING EXTRACTED DATASET")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:")
    print("Rows   :", len(df))
    print("Columns:", len(df.columns))

    print("\nFirst 10 rows:")
    print("-" * 70)
    print(df.head(10).to_string(index=False))

    print("\nMissing values:")
    print("-" * 70)

    missing = df.isnull().sum()

    for column, count in missing.items():
        print(f"{column:20s}: {count}")

    print("\nAvailable annotation counts:")
    print("-" * 70)

    print(
        "Male head:",
        df["male_head_x"].notna().sum()
    )

    print(
        "Male point:",
        df["male_point_x"].notna().sum()
    )

    print(
        "Male abdomen:",
        df["male_abdomen_x"].notna().sum()
    )

    print(
        "Female point:",
        df["female_point_x"].notna().sum()
    )

    print(
        "Wing 1:",
        df["wing1_x"].notna().sum()
    )

    print(
        "Wing 2:",
        df["wing2_x"].notna().sum()
    )

    print(
        "Male point 2:",
        df["male_point2_x"].notna().sum()
    )

    print("\nDataset statistics:")
    print("-" * 70)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    print(
        df[numeric_columns].describe().round(2).to_string()
    )

    print("\n" + "=" * 70)
    print("DATASET CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()