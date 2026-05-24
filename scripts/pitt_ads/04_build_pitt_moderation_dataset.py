import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(".")
INPUT_TOPICS_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_topics.csv"
MAPPING_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_topic_to_moderation_label.csv"

OUTPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_dataset.csv"
SUMMARY_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_moderation_summary.csv"
MISSING_TOPICS_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_missing_topic_mapping_review.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    topic_rows = read_csv(INPUT_TOPICS_CSV)
    mapping_rows = read_csv(MAPPING_CSV)

    topic_to_label = {
        row["majority_topic_name"]: row["moderation_label"]
        for row in mapping_rows
    }

    topic_to_include = {
        row["majority_topic_name"]: row["include_in_dataset"]
        for row in mapping_rows
    }

    final_rows = []
    excluded_rows = []
    missing_mapping_rows = []

    for row in topic_rows:
        topic_name = row["majority_topic_name"]

        # If the topic is a free-text "Other" value, keep it for review and exclude for now.
        if topic_name not in topic_to_label:
            missing_mapping_rows.append({
                "topic_name": topic_name,
                "relative_image_path": row["relative_image_path"],
                "image_path": row["image_path"],
                "topic_annotations_raw": row["topic_annotations_raw"],
                "majority_count": row["majority_count"],
                "num_annotators": row["num_annotators"],
            })
            excluded_rows.append(row)
            continue

        include = topic_to_include[topic_name]
        moderation_label = topic_to_label[topic_name]

        if include != "yes":
            excluded_rows.append(row)
            continue

        if row["image_exists"] != "True":
            excluded_rows.append(row)
            continue

        final_rows.append({
            "relative_image_path": row["relative_image_path"],
            "image_path": row["image_path"],
            "pitt_topic_id": row["majority_topic_id"],
            "pitt_topic_name": topic_name,
            "moderation_label": moderation_label,
            "majority_count": row["majority_count"],
            "num_annotators": row["num_annotators"],
            "image_exists": row["image_exists"],
        })

    write_rows(
        OUTPUT_CSV,
        [
            "relative_image_path",
            "image_path",
            "pitt_topic_id",
            "pitt_topic_name",
            "moderation_label",
            "majority_count",
            "num_annotators",
            "image_exists",
        ],
        final_rows,
    )

    missing_counter = Counter(row["topic_name"] for row in missing_mapping_rows)
    missing_summary_rows = [
        {
            "topic_name": topic_name,
            "count": count,
        }
        for topic_name, count in missing_counter.most_common()
    ]

    write_rows(
        MISSING_TOPICS_CSV,
        ["topic_name", "count"],
        missing_summary_rows,
    )

    label_counter = Counter(row["moderation_label"] for row in final_rows)
    topic_counter = Counter(row["pitt_topic_name"] for row in final_rows)

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "count"])

        for label, count in label_counter.most_common():
            writer.writerow(["moderation_label", label, count])

        for topic, count in topic_counter.most_common():
            writer.writerow(["pitt_topic", topic, count])

        writer.writerow(["excluded", "free_text_topics_for_review", len(missing_mapping_rows)])
        writer.writerow(["excluded", "total_excluded_rows", len(excluded_rows)])

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {MISSING_TOPICS_CSV}")
    print(f"Included rows: {len(final_rows)}")
    print(f"Excluded rows: {len(excluded_rows)}")
    print(f"Free-text topic rows for review: {len(missing_mapping_rows)}")

    print("\nLabel distribution:")
    for label, count in label_counter.most_common():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
