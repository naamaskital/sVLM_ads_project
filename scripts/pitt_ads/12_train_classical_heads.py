import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from transformers import CLIPModel, CLIPProcessor


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


def find_image_column(df):
    for col in ["image_path", "path", "filepath", "file_path", "filename", "image_name", "id"]:
        if col in df.columns:
            return col

    raise ValueError(f"Could not find image column. Columns: {list(df.columns)}")


def resolve_image_path(value, project_root):
    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return None

    path = Path(value)

    if path.is_absolute() and path.exists():
        return path

    candidate = project_root / value

    if candidate.exists():
        return candidate

    return None


def extract_clip_features(split_name, project_root, model, processor, device, batch_size, force_extract):
    cache_dir = project_root / "reports" / "pitt_ads" / "clip_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{split_name}_clip_vit_b32_embeddings.npz"

    if cache_path.exists() and not force_extract:
        data = np.load(cache_path, allow_pickle=True)
        print(f"Loaded cached embeddings for {split_name}: {cache_path}")
        return data["X"], data["y"], data["image_paths"]

    split_path = project_root / "data" / "processed" / "pitt_ads" / "splits" / f"{split_name}.csv"
    df = pd.read_csv(split_path)

    label_col = find_label_column(df)
    image_col = find_image_column(df)

    rows = df.to_dict("records")

    features = []
    labels = []
    image_paths = []

    missing_images = 0
    unreadable_images = 0

    for start in tqdm(range(0, len(rows), batch_size), desc=f"Extracting {split_name}"):
        batch_rows = rows[start:start + batch_size]

        images = []
        valid_rows = []
        valid_paths = []

        for row in batch_rows:
            image_path = resolve_image_path(row[image_col], project_root)

            if image_path is None:
                missing_images += 1
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                unreadable_images += 1
                continue

            images.append(image)
            valid_rows.append(row)
            valid_paths.append(str(image_path))

        if not images:
            continue

        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            image_features = model.visual_projection(vision_outputs.pooler_output)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        features.append(image_features.cpu().numpy())

        for row, image_path in zip(valid_rows, valid_paths):
            labels.append(row[label_col])
            image_paths.append(image_path)

    if len(features) == 0:
        raise ValueError(
            f"No features were extracted for split {split_name}. "
            f"Missing images: {missing_images}, unreadable images: {unreadable_images}"
        )

    X = np.vstack(features)
    y = np.array(labels)
    image_paths = np.array(image_paths)

    np.savez_compressed(cache_path, X=X, y=y, image_paths=image_paths)

    print(f"Saved embeddings for {split_name}: {cache_path}")
    print(f"Examples: {len(y)}")
    print(f"Missing images: {missing_images}")
    print(f"Unreadable images: {unreadable_images}")

    return X, y, image_paths


def build_classifiers(selected_names):
    all_classifiers = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                max_iter=10000,
                random_state=42
            )
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=42
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128,),
                max_iter=300,
                early_stopping=False,
                random_state=42
            )
        ),
    }

    return {name: all_classifiers[name] for name in selected_names}


def evaluate_model(model_name, clf, X, y, split_name, output_dir, image_paths):
    pred = clf.predict(X)

    report_dict = classification_report(
        y,
        pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y,
        pred,
        labels=LABELS,
        digits=4,
        zero_division=0
    )

    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0)

    report_path = output_dir / f"{model_name}_{split_name}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report_text)

    predictions_path = output_dir / f"{model_name}_{split_name}_predictions.csv"

    pd.DataFrame({
        "image_path": image_paths,
        "true_label": y,
        "predicted_label": pred,
    }).to_csv(predictions_path, index=False)

    row = {
        "model": model_name,
        "split": split_name,
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=[
            "logistic_regression",
            "linear_svm",
            "random_forest",
            "gradient_boosting",
            "mlp",
        ],
        choices=[
            "logistic_regression",
            "linear_svm",
            "random_forest",
            "gradient_boosting",
            "mlp",
        ],
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "reports" / "pitt_ads" / "classical_heads"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    X_train, y_train, train_paths = extract_clip_features(
        "train",
        project_root,
        model,
        processor,
        device,
        args.batch_size,
        args.force_extract,
    )

    X_val, y_val, val_paths = extract_clip_features(
        "val",
        project_root,
        model,
        processor,
        device,
        args.batch_size,
        args.force_extract,
    )

    X_test, y_test, test_paths = extract_clip_features(
        "test",
        project_root,
        model,
        processor,
        device,
        args.batch_size,
        args.force_extract,
    )

    classifiers = build_classifiers(args.classifiers)

    all_results = []

    for model_name, clf in classifiers.items():
        print()
        print("=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)

        clf.fit(X_train, y_train)

        val_row = evaluate_model(
            model_name,
            clf,
            X_val,
            y_val,
            "val",
            output_dir,
            val_paths,
        )

        test_row = evaluate_model(
            model_name,
            clf,
            X_test,
            y_test,
            "test",
            output_dir,
            test_paths,
        )

        all_results.append(val_row)
        all_results.append(test_row)

        print(f"Validation macro F1: {val_row['macro_f1']:.4f}")
        print(f"Test macro F1: {test_row['macro_f1']:.4f}")
        print(f"Test accuracy: {test_row['accuracy']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print()
    print("Saved model comparison to:")
    print(results_path)

    print()
    print("Summary:")
    print(results_df[["model", "split", "accuracy", "macro_f1", "weighted_f1"]])


if __name__ == "__main__":
    main()
