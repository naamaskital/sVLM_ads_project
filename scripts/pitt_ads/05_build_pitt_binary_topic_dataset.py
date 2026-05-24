import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(".")
INPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_dataset.csv"

OUTPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_binary_topic_dataset.csv"
SUMMARY_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_binary_topic_summary.csv"

# fine moderation label -> (safety_label, topic_label, include)
LABEL_MAPPING = {
    "safe_or_irrelevant": ("safe", "safe_or_irrelevant", "yes"),

    "alcohol": ("unsafe", "substance", "yes"),
    "smoking_alcohol_abuse": ("unsafe", "substance", "yes"),

    "animal_abuse": ("unsafe", "violence_or_abuse", "yes"),
    "violence": ("unsafe", "violence_or_abuse", "yes"),

    "bullying": ("unsafe", "bullying", "yes"),

    # Too few examples for now, so keep it out of the first clean training version.
    "gambling": ("unsafe", "gambling", "no"),
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

        if fine_label not in LABEL_MAPPING:
            excluded_rows.append(row)
            continue

        safety_label, topic_label, include = LABEL_MAPPING[fine_label]

        if include != "yes":
            excluded_rows.append(row)
            continue

        new_row = dict(row)
        new_row["safety_label"] = safety_label
        new_row["topic_label"] = topic_label
        new_row["fine_label"] = fine_label

        final_rows.append(new_row)

    fieldnames = list(final_rows[0].keys())

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    safety_counter = Counter(row["safety_label"] for row in final_rows)
    topic_counter = Counter(row["topic_label"] for row in final_rows)
    fine_counter = Counter(row["fine_label"] for row in final_rows)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "count"])

        for name, count in safety_counter.most_common():
            writer.writerow(["safety_label", name, count])

        for name, count in topic_counter.most_common():
            writer.writerow(["topic_label", name, count])

        for name, count in fine_counter.most_common():
            writer.writerow(["fine_label", name, count])

        writer.writerow(["excluded", "excluded_rows", len(excluded_rows)])

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Included rows: {len(final_rows)}")
    print(f"Excluded rows: {len(excluded_rows)}")

    print("\nSafety distribution:")
    for label, count in safety_counter.most_common():
        print(f"{label}: {count}")

    print("\nTopic distribution:")
    for label, count in topic_counter.most_common():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
