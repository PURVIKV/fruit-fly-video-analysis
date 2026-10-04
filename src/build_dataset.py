import json
import csv
from pathlib import Path


INPUT_DIR = Path("input/images")
OUTPUT_FILE = Path("data/annotations_dataset.csv")


def get_point(shape):
    """
    Extract the first point from a LabelMe annotation.
    """
    points = shape.get("points", [])

    if not points:
        return None, None

    x = points[0][0]
    y = points[0][1]

    return x, y


def process_json(json_file):
    """
    Read one JSON annotation file and convert
    its labels into a single row of data.
    """

    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    row = {
        "image": data.get("imagePath", json_file.stem),

        "male_head_x": "",
        "male_head_y": "",

        "male_point_x": "",
        "male_point_y": "",

        "male_abdomen_x": "",
        "male_abdomen_y": "",

        "female_point_x": "",
        "female_point_y": "",

        "wing1_x": "",
        "wing1_y": "",

        "wing2_x": "",
        "wing2_y": "",

        "male_point2_x": "",
        "male_point2_y": "",
    }

    wing_count = 0

    for shape in data.get("shapes", []):

        label = shape.get("label", "")
        x, y = get_point(shape)

        if x is None or y is None:
            continue

        if label == "mh":
            row["male_head_x"] = x
            row["male_head_y"] = y

        elif label == "mp":
            row["male_point_x"] = x
            row["male_point_y"] = y

        elif label == "ma":
            row["male_abdomen_x"] = x
            row["male_abdomen_y"] = y

        elif label == "fp":
            row["female_point_x"] = x
            row["female_point_y"] = y

        elif label == "mp2":
            row["male_point2_x"] = x
            row["male_point2_y"] = y

        elif label == "mw":

            if wing_count == 0:
                row["wing1_x"] = x
                row["wing1_y"] = y

            elif wing_count == 1:
                row["wing2_x"] = x
                row["wing2_y"] = y

            wing_count += 1

    return row


def main():

    print("=" * 70)
    print("BUILDING FRUIT FLY ANNOTATION DATASET")
    print("=" * 70)

    json_files = sorted(INPUT_DIR.rglob("*.json"))

    print("\nInput directory:")
    print(INPUT_DIR)

    print("\nJSON files found:", len(json_files))

    if not json_files:
        print("ERROR: No JSON files found.")
        return

    rows = []

    for index, json_file in enumerate(json_files, start=1):

        try:
            row = process_json(json_file)
            rows.append(row)

        except Exception as error:
            print(
                f"ERROR processing {json_file}: {error}"
            )

        if index % 50 == 0:
            print(
                f"Processed {index}/{len(json_files)} files..."
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "image",

        "male_head_x",
        "male_head_y",

        "male_point_x",
        "male_point_y",

        "male_abdomen_x",
        "male_abdomen_y",

        "female_point_x",
        "female_point_y",

        "wing1_x",
        "wing1_y",

        "wing2_x",
        "wing2_y",

        "male_point2_x",
        "male_point2_y",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nRows:", len(rows))
    print("Columns:", len(fieldnames))

    print("\nColumns:")
    for column in fieldnames:
        print(" -", column)

    print("\nDataset building completed.")


if __name__ == "__main__":
    main()