import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_coarse_dataset.csv"

REALISTIC_DIR = PROJECT_ROOT / "data/processed/pitt_ads/splits_realistic"
BALANCED_TRAIN_DIR = PROJECT_ROOT / "data/processed/pitt_ads/splits_balanced_train_realistic_eval"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to write: {path}")

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_split_column(rows, split_name):
    new_rows = []

    for row in rows:
        new_row = dict(row)
        new_row["split"] = split_name
        new_rows.append(new_row)

    return new_rows


def print_distribution(title, rows):
    print()
    print(title)
    print("-" * len(title))

    counter = Counter(row["coarse_label"] for row in rows)

    for label in LABELS:
        print(f"{label}: {counter[label]}")

    print(f"Total: {len(rows)}")


def create_realistic_splits(rows):
    rows_by_label = defaultdict(list)

    for row in rows:
        rows_by_label[row["coarse_label"]].append(row)

    train_rows = []
    val_rows = []
    test_rows = []

    for label, label_rows in rows_by_label.items():
        random.shuffle(label_rows)

        n = len(label_rows)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_rows.extend(label_rows[:train_end])
        val_rows.extend(label_rows[train_end:val_end])
        test_rows.extend(label_rows[val_end:])

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    train_rows = add_split_column(train_rows, "train")
    val_rows = add_split_column(val_rows, "val")
    test_rows = add_split_column(test_rows, "test")

    return train_rows, val_rows, test_rows


def balance_train_by_downsampling_safe(train_rows):
    sensitive_rows = [
        row for row in train_rows
        if row["coarse_label"] != "safe_or_irrelevant"
    ]

    safe_rows = [
        row for row in train_rows
        if row["coarse_label"] == "safe_or_irrelevant"
    ]

    safe_to_keep = min(len(safe_rows), len(sensitive_rows))
    sampled_safe_rows = random.sample(safe_rows, safe_to_keep)

    balanced_train_rows = sensitive_rows + sampled_safe_rows
    random.shuffle(balanced_train_rows)

    return balanced_train_rows


def write_summary(output_dir, split_rows):
    summary_path = output_dir / "split_summary.csv"

    with open(summary_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "coarse_label", "count"])

        for split_name, rows in split_rows:
            counter = Counter(row["coarse_label"] for row in rows)

            for label in LABELS:
                writer.writerow([split_name, label, counter[label]])


def save_split_dir(output_dir, train_rows, val_rows, test_rows):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "val.csv", val_rows)
    write_csv(output_dir / "test.csv", test_rows)

    write_summary(
        output_dir,
        [
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        ],
    )


def main():
    random.seed(RANDOM_SEED)

    rows = read_csv(INPUT_CSV)

    rows = [
        row for row in rows
        if row["coarse_label"] in LABELS
    ]

    print_distribution("Original realistic dataset", rows)

    realistic_train, realistic_val, realistic_test = create_realistic_splits(rows)

    save_split_dir(
        REALISTIC_DIR,
        realistic_train,
        realistic_val,
        realistic_test,
    )

    print()
    print(f"Saved realistic splits to: {REALISTIC_DIR}")

    print_distribution("Realistic train", realistic_train)
    print_distribution("Realistic val", realistic_val)
    print_distribution("Realistic test", realistic_test)

    balanced_train = balance_train_by_downsampling_safe(realistic_train)

    save_split_dir(
        BALANCED_TRAIN_DIR,
        balanced_train,
        realistic_val,
        realistic_test,
    )

    print()
    print(f"Saved balanced-train realistic-eval splits to: {BALANCED_TRAIN_DIR}")

    print_distribution("Balanced train", balanced_train)
    print_distribution("Natural val", realistic_val)
    print_distribution("Natural test", realistic_test)


if __name__ == "__main__":
    main()
