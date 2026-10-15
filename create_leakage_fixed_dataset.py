from pathlib import Path
from collections import defaultdict, Counter
from shutil import copy2
import hashlib
import random
import yaml

from sklearn.model_selection import GroupShuffleSplit

SOURCE = Path("object_dataset_fixed")
DEST = Path("object_dataset_leakage_fixed")

SEED = 42
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = {
    0: "Bowl",
    1: "Cup",
    2: "Fork",
    3: "Knife",
    4: "Plate",
    5: "Spoon",
}

# Never overwrite an existing dataset.
if DEST.exists():
    raise SystemExit(
        f"\nSTOP: {DEST} already exists.\n"
        "Nothing was changed. Rename or inspect that folder first."
    )

# Collect images and group related filename variants.
records = []
groups = defaultdict(list)
exact_hashes = defaultdict(list)

for split in SPLITS:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"Missing images or labels in {split}")

    for image_path in sorted(image_dir.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise SystemExit(f"Missing label file: {label_path}")

        # Match the source identifier used in our previous audit.
        original_id = image_path.stem.split("_jpg.rf.")[0]

        # Count classes represented in this image.
        image_classes = set()
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if parts:
                try:
                    class_id = int(parts[0])
                except ValueError:
                    raise SystemExit(f"Invalid class ID in {label_path}")

                if class_id not in CLASS_NAMES:
                    raise SystemExit(
                        f"Unexpected class {class_id} in {label_path}"
                    )

                image_classes.add(class_id)

        record = {
            "image": image_path,
            "label": label_path,
            "group": original_id,
            "classes": image_classes,
        }

        records.append(record)
        groups[original_id].append(record)

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        exact_hashes[digest].append(original_id)

if not records:
    raise SystemExit("No images found.")

# Check whether the same exact image content has multiple group IDs.
hash_conflicts = [
    ids for ids in exact_hashes.values()
    if len(set(ids)) > 1
]

if hash_conflicts:
    print(
        "\nWARNING: Exact duplicate images with different group IDs "
        f"found: {len(hash_conflicts)} groups."
    )
    print("These require additional review; filename grouping alone may not suffice.")

# Split groups, never individual images.
group_ids = sorted(groups)
dummy_x = list(range(len(group_ids)))
group_array = group_ids

first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_idx, temp_idx = next(
    first_split.split(dummy_x, groups=group_array)
)

train_groups = {group_array[i] for i in train_idx}
temp_groups = [group_array[i] for i in temp_idx]

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=SEED,
)

valid_idx, test_idx = next(
    second_split.split(
        list(range(len(temp_groups))),
        groups=temp_groups,
    )
)

valid_groups = {temp_groups[i] for i in valid_idx}
test_groups = {temp_groups[i] for i in test_idx}

assignment = {}

for group_id in train_groups:
    assignment[group_id] = "train"

for group_id in valid_groups:
    assignment[group_id] = "valid"

for group_id in test_groups:
    assignment[group_id] = "test"

# Copy images and labels into the new dataset.
for split in SPLITS:
    (DEST / split / "images").mkdir(parents=True)
    (DEST / split / "labels").mkdir(parents=True)

for record in records:
    split = assignment[record["group"]]

    copy2(
        record["image"],
        DEST / split / "images" / record["image"].name,
    )
    copy2(
        record["label"],
        DEST / split / "labels" / record["label"].name,
    )

# Write a dataset configuration using an absolute path.
data_config = {
    "path": str(DEST.resolve()),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": CLASS_NAMES,
}

with (DEST / "data.yaml").open("w") as f:
    yaml.safe_dump(data_config, f, sort_keys=False)

# Verify image/label pairing and report class coverage.
print("\n========== NEW DATASET REPORT ==========")
print("Dataset:", DEST.resolve())
print("Total images:", len(records))
print("Original filename groups:", len(group_ids))

for split in SPLITS:
    image_dir = DEST / split / "images"
    label_dir = DEST / split / "labels"

    images = {
        p.stem for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {p.stem for p in label_dir.glob("*.txt")}

    counts = Counter()
    for path in label_dir.glob("*.txt"):
        for line in path.read_text().splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(images - labels))
    print("Orphan labels:", len(labels - images))
    print("Annotated instances:", sum(counts.values()))

    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_name}: {counts[class_id]} instances")

    if images != labels:
        raise SystemExit(f"Image/label pairing check failed for {split}")

# Confirm that filename groups do not cross the new splits.
group_splits = defaultdict(set)

for record in records:
    group_splits[record["group"]].add(assignment[record["group"]])

leaking_groups = {
    group_id: split_set
    for group_id, split_set in group_splits.items()
    if len(split_set) > 1
}

print("\nGroups crossing new splits:", len(leaking_groups))

if leaking_groups:
    raise SystemExit("Group leakage check failed.")

# Check exact image hashes across the resulting splits.
new_hashes = defaultdict(set)

for split in SPLITS:
    for path in (DEST / split / "images").iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            new_hashes[digest].add(split)

duplicate_leaks = [
    digest for digest, split_set in new_hashes.items()
    if len(split_set) > 1
]

print("Exact duplicate groups crossing new splits:", len(duplicate_leaks))

if duplicate_leaks:
    raise SystemExit("Exact duplicate leakage check failed.")

print("\nCreated dataset successfully.")
print("Configuration:", DEST / "data.yaml")
print("Existing dataset and trained model were not modified.")
