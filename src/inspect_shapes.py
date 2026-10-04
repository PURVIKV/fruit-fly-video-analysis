import json
from pathlib import Path


# Find the first annotation file
json_files = list(Path("input/images").rglob("*.json"))

if not json_files:
    print("ERROR: No JSON files found.")
    exit()

json_file = json_files[0]

print("=" * 70)
print("FRUIT FLY ANNOTATION INSPECTOR")
print("=" * 70)

print("\nFile:")
print(json_file)

# Load JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nImage:")
print(data.get("imagePath"))

print("\nNumber of annotated shapes:")
print(len(data.get("shapes", [])))

print("\nANNOTATIONS")
print("-" * 70)

for i, shape in enumerate(data.get("shapes", [])):

    label = shape.get("label")
    shape_type = shape.get("shape_type")
    points = shape.get("points", [])

    print(f"\nShape {i + 1}")
    print("  Label     :", label)
    print("  Shape type:", shape_type)
    print("  Points    :", len(points))

    # Print only the points, which should be manageable
    for point in points:
        print("     ", point)

print("\n" + "=" * 70)
print("Inspection complete.")
print("=" * 70)