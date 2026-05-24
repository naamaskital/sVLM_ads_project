"""
Train a CLIP-based baseline for Pitt Ads moderation.

Pipeline:
Image -> CLIP image encoder -> Logistic Regression classifier -> coarse_label

Run from project root:
    python scripts/pitt_ads/10_train_clip_baseline.py
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data/processed/pitt_ads/splits"
OUTPUT_DIR = PROJECT_ROOT / "reports/pitt_ads/clip_baseline"
MODEL_DIR = PROJECT_ROOT / "models/pitt_ads/clip_baseline"

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

IMAGE_COL = "image_path"
LABEL_COL = "coarse_label"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def load_split(csv_path):
    """
    Load one dataset split CSV.
    """
    df = pd.read_csv(csv_path)
    df = df[[IMAGE_COL, LABEL_COL]].dropna()
    return df


def load_image(image_path):
    """
    Load one image and convert it to RGB.
    """
    full_path = PROJECT_ROOT / image_path

    try:
        image = Image.open(full_path).convert("RGB")
        return image
    except Exception as error:
        print(f"Warning: failed to load image: {full_path}")
        print(error)
        return None


def extract_clip_embeddings(df, processor, model, device):
    """
    Extract CLIP image embeddings for all images in a dataframe.
    """
    model.eval()

    all_embeddings = []
    valid_labels = []

    image_paths = df[IMAGE_COL].tolist()
    labels = df[LABEL_COL].tolist()

    for start_idx in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Extracting CLIP embeddings"):
        batch_paths = image_paths[start_idx:start_idx + BATCH_SIZE]
        batch_labels = labels[start_idx:start_idx + BATCH_SIZE]

        images = []
        kept_labels = []

        for image_path, label in zip(batch_paths, batch_labels):
            image = load_image(image_path)
            if image is not None:
                images.append(image)
                kept_labels.append(label)

        if not images:
            continue

        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            pixel_values = inputs["pixel_values"]

            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output

            image_features = model.visual_projection(pooled_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu().numpy())
        valid_labels.extend(kept_labels)

    embeddings = np.vstack(all_embeddings)
    labels = np.array(valid_labels)

    return embeddings, labels


def save_text(path, text):
    """
    Save text content to a file.
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VAL_CSV)
    test_df = load_split(TEST_CSV)

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)

    print("\nExtracting train embeddings...")
    x_train, y_train_text = extract_clip_embeddings(train_df, processor, model, device)

    print("\nExtracting validation embeddings...")
    x_val, y_val_text = extract_clip_embeddings(val_df, processor, model, device)

    print("\nExtracting test embeddings...")
    x_test, y_test_text = extract_clip_embeddings(test_df, processor, model, device)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_text)
    y_test = label_encoder.transform(y_test_text)

    print("\nClasses:")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"{idx}: {class_name}")

    print("\nTraining Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    classifier.fit(x_train, y_train)

    print("\nEvaluating on validation set...")
    val_preds = classifier.predict(x_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_report = classification_report(
        y_val,
        val_preds,
        target_names=label_encoder.classes_,
    )

    print(f"Validation accuracy: {val_acc:.4f}")
    print(val_report)

    print("\nEvaluating on test set...")
    test_preds = classifier.predict(x_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_report = classification_report(
        y_test,
        test_preds,
        target_names=label_encoder.classes_,
    )
    test_cm = confusion_matrix(y_test, test_preds)

    print(f"Test accuracy: {test_acc:.4f}")
    print(test_report)
    print("Confusion matrix:")
    print(test_cm)

    save_text(
        OUTPUT_DIR / "validation_report.txt",
        f"Validation accuracy: {val_acc:.4f}\n\n{val_report}\n",
    )

    save_text(
        OUTPUT_DIR / "test_report.txt",
        f"Test accuracy: {test_acc:.4f}\n\n{test_report}\n\nConfusion matrix:\n{test_cm}\n",
    )

    np.save(OUTPUT_DIR / "test_confusion_matrix.npy", test_cm)

    joblib.dump(classifier, MODEL_DIR / "logistic_regression_classifier.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    print("\nSaved outputs:")
    print(f"- {OUTPUT_DIR / 'validation_report.txt'}")
    print(f"- {OUTPUT_DIR / 'test_report.txt'}")
    print(f"- {MODEL_DIR / 'logistic_regression_classifier.joblib'}")
    print(f"- {MODEL_DIR / 'label_encoder.joblib'}")


if __name__ == "__main__":
    main()
