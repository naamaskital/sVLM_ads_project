import argparse
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB


LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]


def load_ocr_split(project_root, split):
    path = project_root / "data" / "processed" / "pitt_ads" / "ocr" / f"{split}_ocr.csv"
    df = pd.read_csv(path)

    df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)

    return df


def make_vectorizer():
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        lowercase=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        lowercase=True,
    )

    return FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer),
    ])


def build_models(selected_models):
    models = {
        "ocr_logistic_regression": make_pipeline(
            make_vectorizer(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "ocr_linear_svm": make_pipeline(
            make_vectorizer(),
            LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                random_state=42,
            ),
        ),
        "ocr_complement_nb": make_pipeline(
            make_vectorizer(),
            ComplementNB(),
        ),
    }

    return {name: models[name] for name in selected_models}


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
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report_text)

    cm = confusion_matrix(y, pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
    cm_df.to_csv(output_dir / f"{model_name}_{split}_confusion_matrix.csv")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "ocr_logistic_regression",
            "ocr_linear_svm",
            "ocr_complement_nb",
        ],
        choices=[
            "ocr_logistic_regression",
            "ocr_linear_svm",
            "ocr_complement_nb",
        ],
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "reports" / "pitt_ads" / "ocr_text_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_ocr_split(project_root, "train")
    val_df = load_ocr_split(project_root, "val")
    test_df = load_ocr_split(project_root, "test")

    print("OCR text availability:")
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(
            split,
            "rows:",
            len(df),
            "with_text:",
            int((df["ocr_text"].str.len() > 0).sum()),
        )

    X_train = train_df["ocr_text"]
    y_train = train_df["label"]

    X_val = val_df["ocr_text"]
    y_val = val_df["label"]

    X_test = test_df["ocr_text"]
    y_test = test_df["label"]

    models = build_models(args.models)
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
    results_path = output_dir / "ocr_text_model_comparison.csv"
    results.to_csv(results_path, index=False)

    print()
    print("Saved OCR text baseline results to:")
    print(results_path)

    print()
    print(results[["model", "split", "accuracy", "macro_f1", "weighted_f1"]])


if __name__ == "__main__":
    main()
