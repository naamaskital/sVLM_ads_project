from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]


def find_label_column(df):
    for col in ["coarse_label", "label", "moderation_label", "class"]:
        if col in df.columns:
            return col

    raise ValueError(f"Could not find label column. Columns: {list(df.columns)}")


def load_split(project_root, split):
    path = project_root / "data" / "processed" / "pitt_ads" / "splits" / f"{split}.csv"
    df = pd.read_csv(path)

    label_col = find_label_column(df)

    # We only need the labels for simple baselines.
    y = df[label_col].astype(str)

    # DummyClassifier expects some X, but it ignores the values.
    X = [[0] for _ in range(len(y))]

    return X, y


def evaluate(model_name, model, X, y, split, output_dir):
    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0)

    report_text = classification_report(
        y,
        pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y,
        pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_path = output_dir / f"{model_name}_{split}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Split: {split}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1: {macro_f1:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        file.write(report_text)

    cm = confusion_matrix(y, pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
    cm_df.to_csv(output_dir / f"{model_name}_{split}_confusion_matrix.csv")

    predictions_path = output_dir / f"{model_name}_{split}_predictions.csv"
    pd.DataFrame(
        {
            "true_label": y,
            "predicted_label": pred,
        }
    ).to_csv(predictions_path, index=False)

    row = {
        "model": model_name,
        "split": split,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    for label in LABELS:
        row[f"{label}_precision"] = report_dict[label]["precision"]
        row[f"{label}_recall"] = report_dict[label]["recall"]
        row[f"{label}_f1"] = report_dict[label]["f1-score"]

    return row


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "reports" / "pitt_ads" / "simple_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(project_root, "train")
    X_val, y_val = load_split(project_root, "val")
    X_test, y_test = load_split(project_root, "test")

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent", random_state=42),
        "stratified_random_baseline": DummyClassifier(strategy="stratified", random_state=42),
    }

    rows = []

    for model_name, model in models.items():
        print()
        print("=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)

        model.fit(X_train, y_train)

        val_row = evaluate(model_name, model, X_val, y_val, "val", output_dir)
        test_row = evaluate(model_name, model, X_test, y_test, "test", output_dir)

        rows.append(val_row)
        rows.append(test_row)

        print(f"Validation Macro F1: {val_row['macro_f1']:.4f}")
        print(f"Test Macro F1: {test_row['macro_f1']:.4f}")
        print(f"Test Accuracy: {test_row['accuracy']:.4f}")

    results = pd.DataFrame(rows)
    results_path = output_dir / "simple_baseline_model_comparison.csv"
    results.to_csv(results_path, index=False)

    print()
    print("Saved simple baseline results to:")
    print(results_path)
    print()
    print(results[["model", "split", "accuracy", "macro_f1", "weighted_f1"]])


if __name__ == "__main__":
    main()
