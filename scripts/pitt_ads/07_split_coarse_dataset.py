import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(".")
INPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_coarse_balanced.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/pitt_ads/splits"
TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"
SUMMARY_CSV = OUTPUT_DIR / "split_summary.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_split_column(rows, split_name):
    new_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["split"] = split_name
        new_rows.append(new_row)
    return new_rows


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(INPUT_CSV)

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

        label_train = label_rows[:train_end]
        label_val = label_rows[train_end:val_end]
        label_test = label_rows[val_end:]

        train_rows.extend(label_train)
        val_rows.extend(label_val)
        test_rows.extend(label_test)

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    train_rows = add_split_column(train_rows, "train")
    val_rows = add_split_column(val_rows, "val")
    test_rows = add_split_column(test_rows, "test")

    fieldnames = list(train_rows[0].keys())

    write_csv(TRAIN_CSV, train_rows, fieldnames)
    write_csv(VAL_CSV, val_rows, fieldnames)
    write_csv(TEST_CSV, test_rows, fieldnames)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "coarse_label", "count"])

        for split_name, split_rows in [
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        ]:
            counter = Counter(row["coarse_label"] for row in split_rows)

            for label, count in counter.most_common():
                writer.writerow([split_name, label, count])

    print(f"Saved: {TRAIN_CSV}")
    print(f"Saved: {VAL_CSV}")
    print(f"Saved: {TEST_CSV}")
    print(f"Saved: {SUMMARY_CSV}")

    print("\nSplit sizes:")
    print(f"train: {len(train_rows)}")
    print(f"val:   {len(val_rows)}")
    print(f"test:  {len(test_rows)}")

    print("\nTrain label distribution:")
    for label, count in Counter(row["coarse_label"] for row in train_rows).most_common():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
