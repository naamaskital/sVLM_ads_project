import csv
import html
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(".")
INPUT_CSV = PROJECT_ROOT / "data/processed/pitt_ads/pitt_binary_topic_dataset.csv"

REPORT_DIR = PROJECT_ROOT / "reports/pitt_ads/binary_topic_viewer"
REPORT_HTML = REPORT_DIR / "index.html"

MAX_EXAMPLES_PER_GROUP = 20

# Windows path to the USB drive.
WINDOWS_USB_IMAGE_ROOT = "file:///E:/sVLM_ads_data/pitt_ads/images"


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_windows_img_src(relative_image_path):
    relative_image_path = relative_image_path.replace("\\", "/")
    return f"{WINDOWS_USB_IMAGE_ROOT}/{relative_image_path}"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(INPUT_CSV)

    groups = defaultdict(list)

    for row in rows:
        key = (row["safety_label"], row["topic_label"])
        if len(groups[key]) < MAX_EXAMPLES_PER_GROUP:
            groups[key].append(row)

    html_parts = []

    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pitt Ads Binary + Topic Viewer</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 24px;
}
h1, h2 {
    margin-top: 28px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}
.card {
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 10px;
}
.card img {
    width: 100%;
    height: 220px;
    object-fit: contain;
    background: #f5f5f5;
    border: 1px solid #ccc;
}
.meta {
    font-size: 14px;
    margin-top: 8px;
}
.label {
    font-weight: bold;
}
.safe {
    background: #eef9ee;
}
.unsafe {
    background: #fff1f1;
}
</style>
</head>
<body>
<h1>Pitt Ads Viewer: Safe / Unsafe + Topic</h1>
<p>This viewer is used to manually inspect the mapping from Pitt Ads topics to the project labels.</p>
""")

    for (safety_label, topic_label), group_rows in sorted(groups.items()):
        html_parts.append(
            f"<h2>{html.escape(safety_label)} / {html.escape(topic_label)}</h2>"
        )
        html_parts.append('<div class="grid">')

        for row in group_rows:
            img_src = make_windows_img_src(row["relative_image_path"])
            card_class = "unsafe" if row["safety_label"] == "unsafe" else "safe"

            html_parts.append(f"""
<div class="card {card_class}">
<img src="{html.escape(img_src)}">
<div class="meta">
<div><span class="label">safety:</span> {html.escape(row["safety_label"])}</div>
<div><span class="label">topic:</span> {html.escape(row["topic_label"])}</div>
<div><span class="label">fine:</span> {html.escape(row["fine_label"])}</div>
<div><span class="label">Pitt topic:</span> {html.escape(row["pitt_topic_name"])}</div>
<div><span class="label">path:</span> {html.escape(row["relative_image_path"])}</div>
</div>
</div>
""")

        html_parts.append("</div>")

    html_parts.append("</body></html>")

    REPORT_HTML.write_text("\n".join(html_parts), encoding="utf-8")

    print(f"Saved viewer: {REPORT_HTML}")
    print("Open it with:")
    print(f"explorer.exe {REPORT_DIR}")


if __name__ == "__main__":
    main()
