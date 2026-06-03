import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]


def load_clip_embeddings(project_root, split):
    path = (
        project_root
        / "reports"
        / "pitt_ads"
        / "clip_embeddings"
        / f"{split}_clip_vit_b32_embeddings.npz"
    )

    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"].astype(str)
    image_paths = data["image_paths"].astype(str)

    return X, y, image_paths


def load_ocr_split(project_root, split):
    path = (
        project_root
        / "data"
        / "processed"
        / "pitt_ads"
        / "ocr"
        / f"{split}_ocr.csv"
    )

    df = pd.read_csv(path)
    df["ocr_text"] = df["ocr_text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)

    return df


def make_text_vectorizer():
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


def build_dense_fusion_features(
    X_image_train,
    X_image_val,
    X_image_test,
    text_train,
    text_val,
    text_test,
    text_components,
):
    vectorizer = make_text_vectorizer()

    print("Fitting OCR TF-IDF vectorizer...")
    X_text_train_sparse = vectorizer.fit_transform(text_train)
    X_text_val_sparse = vectorizer.transform(text_val)
    X_text_test_sparse = vectorizer.transform(text_test)

    max_components = min(
        text_components,
        X_text_train_sparse.shape[1] - 1,
        X_text_train_sparse.shape[0] - 1,
    )

    if max_components < 2:
        raise ValueError("Not enough OCR text features for TruncatedSVD.")

    print(f"Reducing OCR text features with TruncatedSVD: {max_components} components")

    svd = TruncatedSVD(
        n_components=max_components,
        random_state=42,
    )

    X_text_train_dense = svd.fit_transform(X_text_train_sparse)
    X_text_val_dense = svd.transform(X_text_val_sparse)
    X_text_test_dense = svd.transform(X_text_test_sparse)

    X_train = np.hstack([X_image_train, X_text_train_dense])
    X_val = np.hstack([X_image_val, X_text_val_dense])
    X_test = np.hstack([X_image_test, X_text_test_dense])

    print("Fusion feature shapes:")
    print("train:", X_train.shape)
    print("val:", X_val.shape)
    print("test:", X_test.shape)

    return X_train, X_val, X_test


def build_models(selected_models):
    all_models = {
        "fusion_logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "fusion_linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                random_state=42,
            ),
        ),
        "fusion_mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128,),
                max_iter=300,
                early_stopping=False,
                random_state=42,
            ),
        ),
    }

    return {name: all_models[name] for name in selected_models}


def evaluate(model_name, model, X, y, split, output_dir, image_paths):
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

    predictions_path = output_dir / f"{model_name}_{split}_predictions.csv"
    pd.DataFrame({
        "image_path": image_paths,
        "true_label": y,
        "predicted_label": pred,
    }).to_csv(predictions_path, index=False)

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


def check_alignment(split, y_clip, ocr_df):
    y_ocr = ocr_df["label"].to_numpy().astype(str)

    if len(y_clip) != len(y_ocr):
        raise ValueError(
            f"Length mismatch for {split}: "
            f"CLIP has {len(y_clip)} rows, OCR has {len(y_ocr)} rows"
        )

    if not np.array_equal(y_clip, y_ocr):
        print(f"Warning: label order mismatch detected in {split}.")
        print("The script will continue, but check data alignment if results look strange.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-components", type=int, default=128)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "fusion_logistic_regression",
            "fusion_linear_svm",
            "fusion_mlp",
        ],
        choices=[
            "fusion_logistic_regression",
            "fusion_linear_svm",
            "fusion_mlp",
        ],
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "reports" / "pitt_ads" / "image_text_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_image_train, y_train, train_paths = load_clip_embeddings(project_root, "train")
    X_image_val, y_val, val_paths = load_clip_embeddings(project_root, "val")
    X_image_test, y_test, test_paths = load_clip_embeddings(project_root, "test")

    train_ocr = load_ocr_split(project_root, "train")
    val_ocr = load_ocr_split(project_root, "val")
    test_ocr = load_ocr_split(project_root, "test")

    check_alignment("train", y_train, train_ocr)
    check_alignment("val", y_val, val_ocr)
    check_alignment("test", y_test, test_ocr)

    print("OCR text availability:")
    for split, df in [("train", train_ocr), ("val", val_ocr), ("test", test_ocr)]:
        print(
            split,
            "rows:",
            len(df),
            "with_text:",
            int((df["ocr_text"].str.len() > 0).sum()),
        )

    X_train, X_val, X_test = build_dense_fusion_features(
        X_image_train,
        X_image_val,
        X_image_test,
        train_ocr["ocr_text"],
        val_ocr["ocr_text"],
        test_ocr["ocr_text"],
        args.text_components,
    )

    models = build_models(args.models)

    rows = []

    for model_name, model in models.items():
        print()
        print("=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)

        model.fit(X_train, y_train)

        val_row = evaluate(
            model_name,
            model,
            X_val,
            y_val,
            "val",
            output_dir,
            val_paths,
        )

        test_row = evaluate(
            model_name,
            model,
            X_test,
            y_test,
            "test",
            output_dir,
            test_paths,
        )

        rows.append(val_row)
        rows.append(test_row)

        print(f"Validation Macro F1: {val_row['macro_f1']:.4f}")
        print(f"Test Macro F1: {test_row['macro_f1']:.4f}")
        print(f"Test Accuracy: {test_row['accuracy']:.4f}")

    results = pd.DataFrame(rows)
    results_path = output_dir / "image_text_fusion_model_comparison.csv"
    results.to_csv(results_path, index=False)

    print()
    print("Saved image + OCR fusion results to:")
    print(results_path)

    print()
    print(results[["model", "split", "accuracy", "macro_f1", "weighted_f1"]])


if __name__ == "__main__":
    main()
