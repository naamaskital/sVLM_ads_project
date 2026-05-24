import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(".")
INPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_coarse_dataset.csv"
SUMMARY_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_coarse_summary.csv"

FINE_TO_COARSE = {
    "safe_or_irrelevant": ("safe_or_irrelevant", "yes"),
    "alcohol": ("substance", "yes"),
    "smoking_alcohol_abuse": ("substance", "yes"),
    "animal_abuse": ("violence_or_abuse", "yes"),
    "violence": ("violence_or_abuse", "yes"),
    "bullying": ("bullying", "yes"),
    "gambling": ("exclude", "no"),
}


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_csv(INPUT_CSV)

    final_rows = []
    excluded_rows = []

    for row in rows:
        fine_label = row["moderation_label"]

        if fine_label not in FINE_TO_COARSE:
            excluded_rows.append(row)
            continue

        coarse_label, include = FINE_TO_COARSE[fine_label]

        if include != "yes":
            excluded_rows.append(row)
            continue

        new_row = dict(row)
        new_row["fine_label"] = fine_label
        new_row["coarse_label"] = coarse_label
        final_rows.append(new_row)

    fieldnames = list(final_rows[0].keys())

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    coarse_counter = Counter(row["coarse_label"] for row in final_rows)
    fine_counter = Counter(row["fine_label"] for row in final_rows)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "count"])

        for label, count in coarse_counter.most_common():
            writer.writerow(["coarse_label", label, count])

        for label, count in fine_counter.most_common():
            writer.writerow(["fine_label", label, count])

        writer.writerow(["excluded", "excluded_rows", len(excluded_rows)])

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Included rows: {len(final_rows)}")
    print(f"Excluded rows: {len(excluded_rows)}")

    print("\nCoarse label distribution:")
    for label, count in coarse_counter.most_common():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
