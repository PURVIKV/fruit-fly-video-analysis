import json
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

ANNOTATION_DIR = Path("input/images")


# ============================================================
# FIND ALL JSON FILES
# ============================================================

json_files = list(ANNOTATION_DIR.rglob("*.json"))

print("=" * 70)
print("FRUIT FLY DATASET ANNOTATION ANALYSIS")
print("=" * 70)

print("\nAnnotation directory:")
print(ANNOTATION_DIR)

print("\nTotal JSON files found:")
print(len(json_files))


if not json_files:
    print("\nERROR: No JSON annotation files found.")
    exit()


# ============================================================
# ANALYZE LABELS
# ============================================================

label_counter = Counter()

image_counter = Counter()

shape_type_counter = Counter()

total_shapes = 0


for json_file in json_files:

    try:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:

        print("\nCould not read:")
        print(json_file)
        print("Error:", e)
        continue


    shapes = data.get("shapes", [])

    total_shapes += len(shapes)

    for shape in shapes:

        label = shape.get("label", "UNKNOWN")
        shape_type = shape.get("shape_type", "UNKNOWN")

        label_counter[label] += 1
        shape_type_counter[shape_type] += 1

        image_counter[json_file.parent] += 1


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nTotal annotation files:")
print(len(json_files))

print("\nTotal annotated shapes:")
print(total_shapes)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("LABEL FREQUENCY")
print("-" * 70)

for label, count in label_counter.most_common():

    print(f"{label:10s} : {count}")


# ------------------------------------------------------------
# Shape types
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SHAPE TYPES")
print("-" * 70)

for shape_type, count in shape_type_counter.most_common():

    print(f"{shape_type:15s} : {count}")


# ------------------------------------------------------------
# Number of shapes per image
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SHAPES PER IMAGE")
print("-" * 70)

shape_counts = []

for json_file in json_files:

    try:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = len(data.get("shapes", []))
        shape_counts.append(count)

    except Exception:
        pass


if shape_counts:

    print("Minimum shapes:", min(shape_counts))
    print("Maximum shapes:", max(shape_counts))
    print("Average shapes:", round(sum(shape_counts) / len(shape_counts), 2))


# ============================================================
# SHOW FIRST 10 ANNOTATION FILES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE ANNOTATION FILES")
print("=" * 70)

for json_file in json_files[:10]:

    try:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = [
            shape.get("label", "UNKNOWN")
            for shape in data.get("shapes", [])
        ]

        print("\n", json_file)
        print("Labels:", labels)

    except Exception:
        pass


print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
