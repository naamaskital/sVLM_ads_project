from pathlib import Path
import html

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]

HARMFUL_LABELS = [
    "bullying",
    "substance",
    "violence_or_abuse",
]


def find_label_column(df):
    for col in ["true_label", "coarse_label", "label", "moderation_label", "class"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find true label column. Columns: {list(df.columns)}")


def find_prediction_column(df):
    for col in ["predicted_label", "zero_shot_prediction", "prediction", "pred"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find prediction column. Columns: {list(df.columns)}")


def find_image_column(df):
    for col in ["image_path", "path", "filepath", "file_path", "filename", "image_name", "id"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find image column. Columns: {list(df.columns)}")


def load_predictions(path, model_name):
    df = pd.read_csv(path)

    true_col = find_label_column(df)
    pred_col = find_prediction_column(df)
    image_col = find_image_column(df)

    out = pd.DataFrame({
        "model": model_name,
        "image_path": df[image_col].astype(str),
        "true_label": df[true_col].astype(str),
        "predicted_label": df[pred_col].astype(str),
    })

    out["is_error"] = out["true_label"] != out["predicted_label"]
    out["error_type"] = out["true_label"] + " -> " + out["predicted_label"]

    return out


def make_confusion_matrix(df, output_path):
    cm = confusion_matrix(
        df["true_label"],
        df["predicted_label"],
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )

    cm_df.to_csv(output_path, index=True)
    return cm_df


def make_metrics_row(df, model_name):
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    row = {
        "model": model_name,
        "num_examples": len(df),
        "num_errors": int((y_true != y_pred).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    for label in LABELS:
        row[f"{label}_precision"] = report[label]["precision"]
        row[f"{label}_recall"] = report[label]["recall"]
        row[f"{label}_f1"] = report[label]["f1-score"]

    return row


def browser_image_path(image_path, project_root):
    raw = str(image_path)

    if raw.startswith(str(project_root)):
        rel = Path(raw).relative_to(project_root)
        return "../../../" + rel.as_posix()

    p = Path(raw)

    if not p.is_absolute():
        return "../../../" + p.as_posix()

    return ""


def write_html_viewer(all_errors, output_path, project_root, max_per_model=80):
    parts = []

    parts.append("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Pitt Ads Error Analysis</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 24px;
    background: #f7f7f7;
}
h1, h2 {
    color: #222;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 16px;
}
.card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 12px;
}
.card img {
    width: 100%;
    max-height: 220px;
    object-fit: contain;
    background: #eee;
}
.meta {
    font-size: 14px;
    line-height: 1.4;
}
.true {
    color: #b00020;
    font-weight: bold;
}
.pred {
    color: #0057b8;
    font-weight: bold;
}
.path {
    font-size: 11px;
    color: #666;
    word-break: break-all;
}
</style>
</head>
<body>
<h1>Pitt Ads Error Analysis</h1>
<p>This viewer shows examples where the model prediction differs from the true label.</p>
""")

    for model_name in sorted(all_errors["model"].unique()):
        model_errors = all_errors[all_errors["model"] == model_name].head(max_per_model)

        parts.append(f"<h2>{html.escape(model_name)} errors</h2>")
        parts.append('<div class="grid">')

        for _, row in model_errors.iterrows():
            img_src = browser_image_path(row["image_path"], project_root)

            parts.append('<div class="card">')

            if img_src:
                parts.append(f'<img src="{html.escape(img_src)}" loading="lazy">')
            else:
                parts.append("<p><b>Image path is outside project root.</b></p>")

            parts.append('<div class="meta">')
            parts.append(f'<p>True: <span class="true">{html.escape(row["true_label"])}</span></p>')
            parts.append(f'<p>Predicted: <span class="pred">{html.escape(row["predicted_label"])}</span></p>')
            parts.append(f'<p>Error type: {html.escape(row["error_type"])}</p>')
            parts.append(f'<p class="path">{html.escape(row["image_path"])}</p>')
            parts.append("</div>")
            parts.append("</div>")

        parts.append("</div>")

    parts.append("""
</body>
</html>
""")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    project_root = Path(__file__).resolve().parents[2]

    classical_dir = project_root / "reports" / "pitt_ads" / "classical_heads"
    zero_shot_dir = project_root / "reports" / "pitt_ads" / "clip_zero_shot"

    output_dir = project_root / "reports" / "pitt_ads" / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = {}

    for path in classical_dir.glob("*_test_predictions.csv"):
        model_name = path.name.replace("_test_predictions.csv", "")
        prediction_files[model_name] = path

    zero_shot_path = zero_shot_dir / "test_predictions.csv"
    if zero_shot_path.exists():
        prediction_files["clip_zero_shot"] = zero_shot_path

    if not prediction_files:
        raise ValueError("No prediction files found.")

    all_predictions = []
    summary_rows = []

    for model_name, path in sorted(prediction_files.items()):
        print(f"Reading {model_name}: {path}")

        df = load_predictions(path, model_name)
        all_predictions.append(df)

        summary_rows.append(make_metrics_row(df, model_name))

        make_confusion_matrix(
            df,
            output_dir / f"{model_name}_confusion_matrix.csv",
        )

        errors = df[df["is_error"]].copy()
        errors.to_csv(output_dir / f"{model_name}_errors.csv", index=False)

        error_counts = (
            errors["error_type"]
            .value_counts()
            .reset_index()
        )
        error_counts.columns = ["error_type", "count"]
        error_counts.to_csv(output_dir / f"{model_name}_errors_by_type.csv", index=False)

        bullying_errors = errors[errors["true_label"] == "bullying"].copy()
        bullying_errors.to_csv(output_dir / f"{model_name}_bullying_errors.csv", index=False)

        harmful_predicted_as_safe = df[
            (df["true_label"].isin(HARMFUL_LABELS)) &
            (df["predicted_label"] == "safe_or_irrelevant")
        ].copy()
        harmful_predicted_as_safe.to_csv(
            output_dir / f"{model_name}_harmful_predicted_as_safe.csv",
            index=False,
        )

        safe_predicted_as_harmful = df[
            (df["true_label"] == "safe_or_irrelevant") &
            (df["predicted_label"].isin(HARMFUL_LABELS))
        ].copy()
        safe_predicted_as_harmful.to_csv(
            output_dir / f"{model_name}_safe_predicted_as_harmful.csv",
            index=False,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("macro_f1", ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(output_dir / "all_predictions.csv", index=False)

    all_errors = combined[combined["is_error"]].copy()
    all_errors.to_csv(output_dir / "all_errors.csv", index=False)

    write_html_viewer(
        all_errors,
        output_dir / "index.html",
        project_root,
    )

    print()
    print("Saved error analysis to:")
    print(output_dir)

    print()
    print("Summary:")
    print(summary_df[["model", "accuracy", "macro_f1", "weighted_f1", "num_errors"]])


if __name__ == "__main__":
    main()
