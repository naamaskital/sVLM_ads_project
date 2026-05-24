import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(".")
SPLIT_DIR = PROJECT_ROOT / "data/processed/pitt_ads/splits"
REPORT_DIR = PROJECT_ROOT / "reports/pitt_ads/coarse_split_viewer"
REPORT_HTML = REPORT_DIR / "index.html"

SPLIT_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}

MAX_EXAMPLES_PER_LABEL_PER_SPLIT = 12


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rel_path_from_report(path):
    return Path("../../../") / path


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    missing_images = []

    for split_name, split_path in SPLIT_FILES.items():
        rows = read_csv(split_path)

        for row in rows:
            row["split"] = split_name

            image_path = PROJECT_ROOT / row["image_path"]
            if not image_path.exists():
                missing_images.append(row)

            all_rows.append(row)

    print(f"Total rows: {len(all_rows)}")
    print(f"Missing images: {len(missing_images)}")

    print("\nDistribution by split and coarse label:")
    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in all_rows if row["split"] == split_name]
        counter = Counter(row["coarse_label"] for row in split_rows)

        print(f"\n{split_name}: {len(split_rows)}")
        for label, count in counter.most_common():
            print(f"  {label}: {count}")

    examples = defaultdict(list)

    for row in all_rows:
        key = (row["split"], row["coarse_label"])
        if len(examples[key]) < MAX_EXAMPLES_PER_LABEL_PER_SPLIT:
            examples[key].append(row)

    html_parts = []
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pitt Ads Coarse Dataset Viewer</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
h1, h2, h3 { margin-top: 28px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 10px; }
.card img { width: 100%; height: 180px; object-fit: contain; background: #f5f5f5; }
.meta { font-size: 13px; margin-top: 8px; }
.label { font-weight: bold; }
</style>
</head>
<body>
<h1>Pitt Ads Coarse Dataset Viewer</h1>
<p>This viewer shows sample images from each split and coarse label.</p>
""")

    for split_name in ["train", "val", "test"]:
        html_parts.append(f"<h2>{html.escape(split_name)}</h2>")

        labels = sorted({row["coarse_label"] for row in all_rows if row["split"] == split_name})

        for label in labels:
            html_parts.append(f"<h3>{html.escape(label)}</h3>")
            html_parts.append('<div class="grid">')

            for row in examples[(split_name, label)]:
                img_src = rel_path_from_report(Path(row["image_path"]))

                html_parts.append(f"""
<div class="card">
<img src="{html.escape(str(img_src))}">
<div class="meta">
<div><span class="label">coarse:</span> {html.escape(row["coarse_label"])}</div>
<div><span class="label">fine:</span> {html.escape(row["fine_label"])}</div>
<div><span class="label">Pitt topic:</span> {html.escape(row["pitt_topic_name"])}</div>
<div><span class="label">path:</span> {html.escape(row["relative_image_path"])}</div>
</div>
</div>
""")

            html_parts.append("</div>")

    html_parts.append("</body></html>")

    REPORT_HTML.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\nSaved viewer: {REPORT_HTML}")


if __name__ == "__main__":
    main()
