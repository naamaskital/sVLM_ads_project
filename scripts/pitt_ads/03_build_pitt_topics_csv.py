import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(".")
ANNOTATION_DIR = PROJECT_ROOT / "data/raw/pitt_ads/annotations/image"
IMAGE_ROOT = PROJECT_ROOT / "data/raw/pitt_ads/images"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/pitt_ads"

TOPICS_JSON = ANNOTATION_DIR / "Topics.json"
OUTPUT_CSV = OUTPUT_DIR / "pitt_topics.csv"
SUMMARY_CSV = OUTPUT_DIR / "pitt_topics_summary.csv"

TOPIC_ID_TO_NAME = {
    "1": "Restaurants, cafe, fast food",
    "2": "Chocolate, cookies, candy, ice cream",
    "3": "Chips, snacks, nuts, fruit, gum, cereal, yogurt, soups",
    "4": "Seasoning, condiments, ketchup",
    "5": "Pet food",
    "6": "Alcohol",
    "7": "Coffee, tea",
    "8": "Soda, juice, milk, energy drinks, water",
    "9": "Cars, automobiles",
    "10": "Electronics",
    "11": "Phone, TV and internet service providers",
    "12": "Financial services",
    "13": "Education",
    "14": "Security and safety services",
    "15": "Software",
    "16": "Other services",
    "17": "Beauty products and cosmetics",
    "18": "Healthcare and medications",
    "19": "Clothing and accessories",
    "20": "Baby products",
    "21": "Games and toys",
    "22": "Cleaning products",
    "23": "Home improvements and repairs",
    "24": "Home appliances",
    "25": "Vacation and travel",
    "26": "Media and arts",
    "27": "Sports equipment and activities",
    "28": "Shopping",
    "29": "Gambling",
    "30": "Environment, nature, pollution, wildlife",
    "31": "Animal rights, animal abuse",
    "32": "Human rights",
    "33": "Safety, safe driving, fire safety",
    "34": "Smoking, alcohol abuse",
    "35": "Domestic violence",
    "36": "Self esteem, bullying, cyber bullying",
    "37": "Political candidates",
    "38": "Charities",
    "39": "Unclear",
}


def normalize_annotation(annotation):
    """
    Convert a raw topic annotation into topic id and topic name.
    Numeric labels are mapped using the official Pitt topics list.
    Free-text labels are kept as OTHER_TEXT.
    """
    annotation = str(annotation).strip()

    if annotation in TOPIC_ID_TO_NAME:
        return annotation, TOPIC_ID_TO_NAME[annotation]

    if annotation.isdigit():
        return annotation, f"UNKNOWN_TOPIC_ID_{annotation}"

    return "OTHER_TEXT", annotation


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TOPICS_JSON, "r", encoding="utf-8") as f:
        topics_data = json.load(f)

    rows = []

    for relative_image_path, annotations in topics_data.items():
        topic_ids = []
        topic_names = []

        for annotation in annotations:
            topic_id, topic_name = normalize_annotation(annotation)
            topic_ids.append(topic_id)
            topic_names.append(topic_name)

        counts = Counter(topic_names)
        majority_topic_name, majority_count = counts.most_common(1)[0]

        majority_topic_id = ""
        for topic_id, topic_name in zip(topic_ids, topic_names):
            if topic_name == majority_topic_name:
                majority_topic_id = topic_id
                break

        image_path = IMAGE_ROOT / relative_image_path

        rows.append({
            "relative_image_path": relative_image_path,
            "image_path": str(image_path),
            "topic_annotations_raw": json.dumps(annotations, ensure_ascii=False),
            "topic_ids": ";".join(topic_ids),
            "topic_names": ";".join(topic_names),
            "majority_topic_id": majority_topic_id,
            "majority_topic_name": majority_topic_name,
            "majority_count": majority_count,
            "num_annotators": len(annotations),
            "image_exists": image_path.exists(),
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "relative_image_path",
            "image_path",
            "topic_annotations_raw",
            "topic_ids",
            "topic_names",
            "majority_topic_id",
            "majority_topic_name",
            "majority_count",
            "num_annotators",
            "image_exists",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    topic_counter = Counter(row["majority_topic_name"] for row in rows)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["topic_name", "count"])
        writer.writeheader()

        for topic_name, count in topic_counter.most_common():
            writer.writerow({
                "topic_name": topic_name,
                "count": count,
            })

    print("Loaded 39 topic labels.")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Number of annotated images: {len(rows)}")
    print(f"Images found on disk: {sum(row['image_exists'] for row in rows)}")
    print(f"Images missing on disk: {sum(not row['image_exists'] for row in rows)}")

    print("\nTop topics:")
    for topic_name, count in topic_counter.most_common(20):
        print(f"{topic_name}: {count}")


if __name__ == "__main__":
    main()
