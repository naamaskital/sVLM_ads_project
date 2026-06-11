from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]


def read_if_exists(path, experiment_group):
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None

    df = pd.read_csv(path)
    df.insert(0, "experiment_group", experiment_group)
    return df


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(f"Could not find any of {possible_names}. Columns: {list(df.columns)}")


def summarize_predictions(path, model_name, experiment_group):
    if not path.exists():
        print(f"Skipping missing prediction file: {path}")
        return None

    df = pd.read_csv(path)

    true_col = find_column(df, ["true_label", "coarse_label", "label", "moderation_label"])
    pred_col = find_column(df, ["predicted_label", "zero_shot_prediction", "prediction", "pred"])

    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    row = {
        "experiment_group": experiment_group,
        "model": model_name,
        "split": "test",
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }

    for label in LABELS:
        row[f"{label}_precision"] = report[label]["precision"]
        row[f"{label}_recall"] = report[label]["recall"]
        row[f"{label}_f1"] = report[label]["f1-score"]

    return pd.DataFrame([row])


def save_final_confusion_matrix(project_root, best_row):
    model_name = best_row["model"]
    group = best_row["experiment_group"]

    prediction_paths = [
        project_root / "reports" / "pitt_ads" / "simple_baselines" / f"{model_name}_test_predictions.csv",
        project_root / "reports" / "pitt_ads" / "classical_heads" / f"{model_name}_test_predictions.csv",
        project_root / "reports" / "pitt_ads" / "ocr_text_baseline" / f"{model_name}_test_predictions.csv",
        project_root / "reports" / "pitt_ads" / "image_text_fusion" / f"{model_name}_test_predictions.csv",
        project_root / "reports" / "pitt_ads" / "clip_zero_shot" / "test_predictions.csv",
    ]

    selected_path = None
    for path in prediction_paths:
        if path.exists():
            if model_name == "clip_zero_shot" and "clip_zero_shot" in str(path):
                selected_path = path
                break
            if path.name == f"{model_name}_test_predictions.csv":
                selected_path = path
                break

    if selected_path is None:
        print(f"Could not find prediction file for best model: {model_name} ({group})")
        return

    df = pd.read_csv(selected_path)
    true_col = find_column(df, ["true_label", "coarse_label", "label", "moderation_label"])
    pred_col = find_column(df, ["predicted_label", "zero_shot_prediction", "prediction", "pred"])

    cm = confusion_matrix(
        df[true_col].astype(str),
        df[pred_col].astype(str),
        labels=LABELS,
    )

    output_dir = project_root / "reports" / "pitt_ads" / "final_results"
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
    cm_df.to_csv(output_dir / "best_model_confusion_matrix.csv")


def main():
    project_root = Path(__file__).resolve().parents[2]
    final_dir = project_root / "reports" / "pitt_ads" / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)

    result_tables = []

    candidates = [
        (
            project_root / "reports" / "pitt_ads" / "simple_baselines" / "simple_baseline_model_comparison.csv",
            "simple_baselines",
        ),
        (
            project_root / "reports" / "pitt_ads" / "classical_heads" / "model_comparison.csv",
            "clip_image_embeddings",
        ),
        (
            project_root / "reports" / "pitt_ads" / "ocr_text_baseline" / "ocr_text_model_comparison.csv",
            "ocr_text_only",
        ),
        (
            project_root / "reports" / "pitt_ads" / "image_text_fusion" / "image_text_fusion_model_comparison.csv",
            "image_text_fusion",
        ),
    ]

    for path, group in candidates:
        table = read_if_exists(path, group)
        if table is not None:
            result_tables.append(table)

    zero_shot_summary = summarize_predictions(
        project_root / "reports" / "pitt_ads" / "clip_zero_shot" / "test_predictions.csv",
        "clip_zero_shot",
        "zero_shot",
    )

    if zero_shot_summary is not None:
        result_tables.append(zero_shot_summary)

    if not result_tables:
        raise ValueError("No result files found. Run the experiment scripts first.")

    final_results = pd.concat(result_tables, ignore_index=True)

    # Keep a clean column order.
    preferred_columns = [
        "experiment_group",
        "model",
        "split",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "bullying_f1",
        "safe_or_irrelevant_f1",
        "substance_f1",
        "violence_or_abuse_f1",
        "bullying_precision",
        "bullying_recall",
        "safe_or_irrelevant_precision",
        "safe_or_irrelevant_recall",
        "substance_precision",
        "substance_recall",
        "violence_or_abuse_precision",
        "violence_or_abuse_recall",
    ]

    existing_columns = [col for col in preferred_columns if col in final_results.columns]
    other_columns = [col for col in final_results.columns if col not in existing_columns]
    final_results = final_results[existing_columns + other_columns]

    final_results = final_results.sort_values(
        by=["split", "macro_f1", "accuracy"],
        ascending=[True, False, False],
    )

    full_path = final_dir / "all_experiment_results.csv"
    final_results.to_csv(full_path, index=False)

    test_results = final_results[final_results["split"] == "test"].copy()
    test_results = test_results.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False,
    )

    test_path = final_dir / "test_results_ranked.csv"
    test_results.to_csv(test_path, index=False)

    markdown_path = final_dir / "test_results_ranked.md"
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(test_results.to_markdown(index=False))

    best_row = test_results.iloc[0]
    save_final_confusion_matrix(project_root, best_row)

    summary_path = final_dir / "final_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Final experiment summary\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Best model: {best_row['model']}\n")
        file.write(f"Experiment group: {best_row['experiment_group']}\n")
        file.write(f"Test accuracy: {best_row['accuracy']:.4f}\n")
        file.write(f"Test macro F1: {best_row['macro_f1']:.4f}\n")
        file.write(f"Test weighted F1: {best_row['weighted_f1']:.4f}\n")

    print()
    print("Saved final results to:")
    print(final_dir)
    print()
    print("Best test model:")
    print(best_row[["experiment_group", "model", "accuracy", "macro_f1", "weighted_f1"]])
    print()
    print("Ranked test results:")
    print(test_results[["experiment_group", "model", "accuracy", "macro_f1", "weighted_f1"]])


if __name__ == "__main__":
    main()
