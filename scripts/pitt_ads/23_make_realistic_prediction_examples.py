from pathlib import Path
import argparse
import html
import shutil

import numpy as np
import pandas as pd


SAFE_LABEL = "safe_or_irrelevant"


def load_image_paths(args, n_rows):
    npz_path = Path(args.embeddings_dir) / f"{args.split}_clip_vit_b32_embeddings.npz"

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        if "image_paths" in data.files:
            paths = [str(x) for x in data["image_paths"]]
            if len(paths) == n_rows:
                return paths

    split_csv = Path(args.splits_dir) / f"{args.split}.csv"
    df = pd.read_csv(split_csv)

    possible_cols = [
        "image_path",
        "path",
        "file_path",
        "filename",
        "image",
        "image_file",
        "ad_image",
    ]

    for col in possible_cols:
        if col in df.columns:
            paths = df[col].astype(str).tolist()
            if len(paths) == n_rows:
                return paths

    print("Could not find image paths automatically.")
    print("Split columns:", list(df.columns))
    return ["" for _ in range(n_rows)]


def resolve_image_path(raw_path, image_root):
    if not raw_path or raw_path == "nan":
        return None

    p = Path(raw_path)

    if p.exists():
        return p

    image_root = Path(image_root)

    candidate = image_root / raw_path
    if candidate.exists():
        return candidate

    candidate = image_root / p.name
    if candidate.exists():
        return candidate

    return None


def copy_images(df, category_dir, image_root):
    category_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        raw_path = str(row.get("image_path", ""))
        src = resolve_image_path(raw_path, image_root)

        if src is None:
            copied_paths.append("")
            continue

        suffix = src.suffix if src.suffix else ".jpg"
        safe_name = f"{rank:02d}_{row['true_label']}__pred_{row['predicted_label']}{suffix}"
        safe_name = safe_name.replace("/", "_").replace(" ", "_")

        dst = category_dir / safe_name
        shutil.copy2(src, dst)
        copied_paths.append(str(dst))

    df = df.copy()
    df["copied_image_path"] = copied_paths
    return df


def save_category(df, name, output_dir, image_root):
    category_dir = output_dir / "example_images" / name
    df_with_images = copy_images(df, category_dir, image_root)

    csv_path = output_dir / f"mlp_{name}.csv"
    df_with_images.to_csv(csv_path, index=False)

    return df_with_images, csv_path


def make_html(categories, html_path):
    parts = []
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<style>")
    parts.append("body { font-family: Arial, sans-serif; margin: 24px; }")
    parts.append("table { border-collapse: collapse; width: 100%; margin-bottom: 40px; }")
    parts.append("td, th { border: 1px solid #ddd; padding: 8px; font-size: 14px; }")
    parts.append("th { background: #f2f2f2; }")
    parts.append("img { max-width: 220px; max-height: 220px; }")
    parts.append("</style></head><body>")
    parts.append("<h1>Realistic Split - MLP Prediction Examples</h1>")

    for title, df in categories.items():
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append("<table>")
        parts.append(
            "<tr>"
            "<th>image</th>"
            "<th>true_label</th>"
            "<th>predicted_label</th>"
            "<th>confidence</th>"
            "<th>safety</th>"
            "<th>unsafe</th>"
            "<th>prob_bullying</th>"
            "<th>prob_substance</th>"
            "<th>prob_violence</th>"
            "</tr>"
        )

        for _, row in df.iterrows():
            img_path = row.get("copied_image_path", "")
            if img_path:
                rel = Path(img_path).relative_to(html_path.parent)
                img_html = f"<img src='{html.escape(str(rel))}'>"
            else:
                img_html = "image not found"

            parts.append("<tr>")
            parts.append(f"<td>{img_html}</td>")
            parts.append(f"<td>{html.escape(str(row['true_label']))}</td>")
            parts.append(f"<td>{html.escape(str(row['predicted_label']))}</td>")
            parts.append(f"<td>{float(row['confidence_score']):.3f}</td>")
            parts.append(f"<td>{float(row['safety_score']):.3f}</td>")
            parts.append(f"<td>{float(row['unsafe_score']):.3f}</td>")
            parts.append(f"<td>{float(row.get('prob_bullying', 0)):.3f}</td>")
            parts.append(f"<td>{float(row.get('prob_substance', 0)):.3f}</td>")
            parts.append(f"<td>{float(row.get('prob_violence_or_abuse', 0)):.3f}</td>")
            parts.append("</tr>")

        parts.append("</table>")

    parts.append("</body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="mlp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--image-root", default="data/raw/pitt_ads/images")
    parser.add_argument("--splits-dir", default="data/processed/pitt_ads/splits_balanced_train_realistic_eval")
    parser.add_argument("--embeddings-dir", default="reports/pitt_ads/realistic_train_clip_embeddings")
    parser.add_argument("--predictions-dir", default="reports/pitt_ads/realistic_train_clip_classifiers")
    parser.add_argument("--output-dir", default="reports/pitt_ads/realistic_prediction_examples")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_dir) / f"{args.classifier}_{args.split}_predictions_with_scores.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)
    df.insert(0, "row_id", range(len(df)))

    image_paths = load_image_paths(args, len(df))
    df["image_path"] = image_paths

    errors = df[df["true_label"] != df["predicted_label"]]
    unsafe_true = df[df["true_label"] != SAFE_LABEL]

    categories_raw = {
        "top_unsafe": df.sort_values("unsafe_score", ascending=False).head(args.n),
        "most_uncertain": df.sort_values("confidence_score", ascending=True).head(args.n),
        "high_confidence_errors": errors.sort_values("confidence_score", ascending=False).head(args.n),
        "false_positives_safe_predicted_unsafe": df[
            (df["true_label"] == SAFE_LABEL) & (df["predicted_label"] != SAFE_LABEL)
        ].sort_values("unsafe_score", ascending=False).head(args.n),
        "false_negatives_unsafe_predicted_safe": unsafe_true[
            unsafe_true["predicted_label"] == SAFE_LABEL
        ].sort_values("safety_score", ascending=False).head(args.n),
    }

    saved_categories = {}

    for name, cat_df in categories_raw.items():
        saved_df, csv_path = save_category(cat_df, name, output_dir, args.image_root)
        saved_categories[name] = saved_df
        print(f"Saved {name}: {csv_path}")

    html_path = output_dir / "mlp_test_examples_viewer.html"
    make_html(saved_categories, html_path)

    print()
    print("Saved HTML viewer:", html_path)
    print("Open it from VSCode / browser to see the examples with images.")


if __name__ == "__main__":
    main()
