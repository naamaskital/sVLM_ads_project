import csv
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(".")
INPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_coarse_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_coarse_balanced.csv"
SUMMARY_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_coarse_balanced_summary.csv"

RANDOM_SEED = 42
SAFE_LABEL = "safe_or_irrelevant"


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(RANDOM_SEED)

    rows = read_csv(INPUT_CSV)

    safe_rows = [row for row in rows if row["coarse_label"] == SAFE_LABEL]
    sensitive_rows = [row for row in rows if row["coarse_label"] != SAFE_LABEL]

    num_sensitive = len(sensitive_rows)

    sampled_safe_rows = random.sample(
        safe_rows,
        k=min(len(safe_rows), num_sensitive)
    )

    final_rows = sensitive_rows + sampled_safe_rows
    random.shuffle(final_rows)

    fieldnames = list(final_rows[0].keys())
    write_csv(OUTPUT_CSV, final_rows, fieldnames)

    label_counter = Counter(row["coarse_label"] for row in final_rows)
    fine_counter = Counter(row["fine_label"] for row in final_rows)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "count"])

        for label, count in label_counter.most_common():
            writer.writerow(["coarse_label", label, count])

        for label, count in fine_counter.most_common():
            writer.writerow(["fine_label", label, count])

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Original rows: {len(rows)}")
    print(f"Sensitive rows kept: {len(sensitive_rows)}")
    print(f"Safe rows sampled: {len(sampled_safe_rows)}")
    print(f"Final balanced rows: {len(final_rows)}")

    print("\nBalanced coarse label distribution:")
    for label, count in label_counter.most_common():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
