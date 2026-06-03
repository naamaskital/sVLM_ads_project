import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from transformers import CLIPModel, CLIPProcessor


LABEL_PROMPTS = {
    "safe_or_irrelevant": [
        "a safe advertisement",
        "a harmless advertisement",
        "an ordinary product advertisement"
    ],
    "substance": [
        "an advertisement about alcohol, smoking, tobacco or substances",
        "an advertisement for beer, wine, cigarettes or tobacco"
    ],
    "violence_or_abuse": [
        "an advertisement showing violence or abuse",
        "an advertisement related to weapons, fighting, injury or abuse"
    ],
    "bullying": [
        "an advertisement showing bullying or harassment",
        "an advertisement related to humiliation, insults or social abuse"
    ],
}


def find_label_column(df):
    possible = ["coarse_label", "label", "moderation_label", "class"]
    for col in possible:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find label column. Columns are: {list(df.columns)}")


def find_image_column(df):
    possible = [
        "image_path",
        "path",
        "filepath",
        "file_path",
        "filename",
        "file_name",
        "image_name",
        "image_id",
        "id",
    ]
    for col in possible:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find image column. Columns are: {list(df.columns)}")


def resolve_image_path(value, images_root):
    value = str(value)
    path = Path(value)

    if path.exists():
        return path

    candidate = images_root / value
    if candidate.exists():
        return candidate

    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_root / f"{value}{ext}"
        if candidate.exists():
            return candidate

    return None


def build_text_features(model, processor, labels, device):
    all_label_features = []

    for label in labels:
        prompts = LABEL_PROMPTS[label]

        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask")
            )
            text_features = model.text_projection(text_outputs.pooler_output)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        label_feature = text_features.mean(dim=0)
        label_feature = label_feature / label_feature.norm()

        all_label_features.append(label_feature)

    return torch.stack(all_label_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    split_path = project_root / "data" / "processed" / "pitt_ads" / "splits" / f"{args.split}.csv"
    images_root = project_root
    output_dir = project_root / "reports" / "pitt_ads" / "clip_zero_shot"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_path)

    label_col = find_label_column(df)
    image_col = find_image_column(df)

    labels = ["bullying", "safe_or_irrelevant", "substance", "violence_or_abuse"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    text_features = build_text_features(model, processor, labels, device)

    y_true = []
    y_pred = []
    used_rows = []

    rows = df.to_dict("records")

    for start in tqdm(range(0, len(rows), args.batch_size)):
        batch_rows = rows[start:start + args.batch_size]

        images = []
        valid_rows = []

        for row in batch_rows:
            image_path = resolve_image_path(row[image_col], images_root)

            if image_path is None:
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            images.append(image)
            valid_rows.append(row)

        if not images:
            continue

        inputs = processor(
            text=None,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            image_features = model.visual_projection(vision_outputs.pooler_output)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarities = image_features @ text_features.T
        predictions = similarities.argmax(dim=1).cpu().tolist()

        for row, pred_idx in zip(valid_rows, predictions):
            true_label = row[label_col]
            pred_label = labels[pred_idx]

            y_true.append(true_label)
            y_pred.append(pred_label)

            new_row = dict(row)
            new_row["zero_shot_prediction"] = pred_label
            used_rows.append(new_row)

    predictions_path = output_dir / f"{args.split}_predictions.csv"
    pd.DataFrame(used_rows).to_csv(predictions_path, index=False)

    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / f"{args.split}_confusion_matrix.csv")

    summary_path = output_dir / f"{args.split}_classification_report.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CLIP zero-shot experiment\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Number of evaluated examples: {len(y_true)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)

    print()
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved report to: {summary_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
