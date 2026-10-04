import json
from pathlib import Path


# Find the first JSON annotation file
json_files = list(
    Path("input/images").rglob("*.json")
)

if not json_files:
    print("ERROR: No JSON files found.")
    exit()

json_file = json_files[0]

print("=" * 70)
print("ANNOTATION INSPECTOR")
print("=" * 70)

print("\nFile:")
print(json_file)

print("\nLoading JSON...")

try:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

except Exception as e:
    print("ERROR while reading JSON:")
    print(e)
    exit()


print("\nJSON loaded successfully!")

print("\nTop-level type:")
print(type(data).__name__)


# ------------------------------------------------------------
# If the JSON is a dictionary
# ------------------------------------------------------------

if isinstance(data, dict):

    print("\nTop-level keys:")
    
    for key in data.keys():
        print("  -", key)

    print("\nDetailed structure:")
    print("-" * 70)

    for key, value in data.items():

        if isinstance(value, dict):
            print(f"{key}: dictionary")
            print("    keys:", list(value.keys())[:20])

        elif isinstance(value, list):
            print(f"{key}: list")
            print("    number of items:", len(value))

            if len(value) > 0:
                print("    first item type:", type(value[0]).__name__)

                if isinstance(value[0], dict):
                    print(
                        "    first item keys:",
                        list(value[0].keys())[:20]
                    )

        elif isinstance(value, str):

            # Do NOT print huge encoded strings
            print(
                f"{key}: string "
                f"(length = {len(value)})"
            )

            if len(value) < 300:
                print("    value:", value)

        else:
            print(
                f"{key}: {type(value).__name__} = {value}"
            )


# ------------------------------------------------------------
# If the JSON is something else
# ------------------------------------------------------------

else:

    print("\nJSON content is not a dictionary.")
    print("Type:", type(data).__name__)

    if isinstance(data, list):
        print("Number of items:", len(data))

        if len(data) > 0:
            print(
                "First item type:",
                type(data[0]).__name__
            )


print("\n" + "=" * 70)
print("Inspection complete.")
print("=" * 70)