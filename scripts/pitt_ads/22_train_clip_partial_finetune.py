import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModelWithProjection


IMAGE_COLUMNS = [
    "image_path",
    "path",
    "filepath",
    "file_path",
    "image",
    "image_filename",
    "filename",
]

LABEL_COLUMNS = [
    "coarse_label",
    "moderation_label",
    "label",
    "class",
    "target",
]


def find_column(df, possible_columns, column_type):
    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {column_type} column. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_image_path(value, project_root, image_root):
    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidate = project_root / path
    if candidate.exists():
        return candidate

    if image_root is not None:
        image_root = Path(image_root)

        candidate = image_root / path
        if candidate.exists():
            return candidate

        candidate = image_root / path.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find image: {value}")


class AdsDataset(Dataset):
    def __init__(self, csv_path, label_to_id, project_root, image_root=None):
        self.df = pd.read_csv(csv_path)
        self.image_col = find_column(self.df, IMAGE_COLUMNS, "image")
        self.label_col = find_column(self.df, LABEL_COLUMNS, "label")

        self.label_to_id = label_to_id
        self.project_root = Path(project_root)
        self.image_root = image_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = resolve_image_path(
            row[self.image_col],
            self.project_root,
            self.image_root,
        )

        image = Image.open(image_path).convert("RGB")
        label_name = str(row[self.label_col])
        label_id = self.label_to_id[label_name]

        return {
            "image": image,
            "label": label_id,
            "label_name": label_name,
            "image_path": str(image_path),
        }


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        image_paths = [item["image_path"] for item in batch]
        label_names = [item["label_name"] for item in batch]

        encoded = processor(images=images, return_tensors="pt")

        return {
            "pixel_values": encoded["pixel_values"],
            "labels": labels,
            "image_paths": image_paths,
            "label_names": label_names,
        }

    return collate_fn


class CLIPPartialFineTuner(nn.Module):
    def __init__(self, model_name, num_labels, unfreeze_last_n):
        super().__init__()

        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name, use_safetensors=True)
        projection_dim = self.vision_model.config.projection_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, num_labels),
        )

        self.freeze_clip()

        if unfreeze_last_n > 0:
            self.unfreeze_last_layers(unfreeze_last_n)

    def freeze_clip(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self, unfreeze_last_n):
        encoder_layers = self.vision_model.vision_model.encoder.layers

        for layer in encoder_layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.vision_model.vision_model.post_layernorm.parameters():
            param.requires_grad = True

        for param in self.vision_model.visual_projection.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        image_features = outputs.image_embeds
        logits = self.classifier(image_features)
        return logits


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def compute_class_weights(train_csv, label_col, label_to_id):
    df = pd.read_csv(train_csv)
    counts = df[label_col].value_counts().to_dict()

    weights = []
    total = len(df)
    num_classes = len(label_to_id)

    for label_name, label_id in sorted(label_to_id.items(), key=lambda x: x[1]):
        count = counts.get(label_name, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(pixel_values)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        losses.append(loss.item())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, id_to_label):
    model.eval()

    losses = []
    all_preds = []
    all_labels = []
    all_paths = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(pixel_values)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        losses.append(loss.item())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_paths.extend(batch["image_paths"])

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    target_names = [id_to_label[i] for i in range(len(id_to_label))]

    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(id_to_label))),
    )

    predictions_df = pd.DataFrame(
        {
            "image_path": all_paths,
            "true_id": all_labels,
            "pred_id": all_preds,
            "true_label": [id_to_label[i] for i in all_labels],
            "pred_label": [id_to_label[i] for i in all_preds],
        }
    )

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "report": report,
        "confusion_matrix": cm,
        "predictions": predictions_df,
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        default="data/processed/pitt_ads/splits_balanced_train_realistic_eval",
        help="Directory that contains train.csv, val.csv, and test.csv",
    )

    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image root directory if paths in CSV are relative",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/pitt_ads/clip_partial_finetune",
        help="Where to save model and reports",
    )

    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="Learning rate for classifier head",
    )

    parser.add_argument(
        "--lr-clip",
        type=float,
        default=1e-5,
        help="Learning rate for unfrozen CLIP layers",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=0,
        help="How many last CLIP vision transformer layers to unfreeze. 0 means train only the classifier head.",
    )

    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use class weights in the loss function.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = Path.cwd()
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing val CSV: {val_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    label_col = find_column(train_df, LABEL_COLUMNS, "label")

    all_labels = sorted(
        set(train_df[label_col].astype(str))
        | set(val_df[label_col].astype(str))
        | set(test_df[label_col].astype(str))
    )

    label_to_id = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    save_json(output_dir / "label_to_id.json", label_to_id)
    save_json(output_dir / "id_to_label.json", id_to_label)
    save_json(output_dir / "args.json", vars(args))

    print("Labels:")
    for label, idx in label_to_id.items():
        print(f"  {idx}: {label}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = CLIPProcessor.from_pretrained(args.model_name)

    train_dataset = AdsDataset(
        train_csv,
        label_to_id,
        project_root,
        image_root=args.image_root,
    )
    val_dataset = AdsDataset(
        val_csv,
        label_to_id,
        project_root,
        image_root=args.image_root,
    )
    test_dataset = AdsDataset(
        test_csv,
        label_to_id,
        project_root,
        image_root=args.image_root,
    )

    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = CLIPPartialFineTuner(
        model_name=args.model_name,
        num_labels=len(label_to_id),
        unfreeze_last_n=args.unfreeze_last_n,
    )
    model.to(device)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    head_params = []
    clip_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("classifier"):
            head_params.append(param)
        else:
            clip_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": clip_params, "lr": args.lr_clip},
        ],
        weight_decay=args.weight_decay,
    )

    if args.class_weights:
        class_weights = compute_class_weights(train_csv, label_col, label_to_id)
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights.detach().cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            id_to_label,
        )

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }

        history.append(epoch_info)

        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.4f} | "
            f"Train macro F1: {train_metrics['macro_f1']:.4f}"
        )

        print(
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"Val macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val weighted F1: {val_metrics['weighted_f1']:.4f}"
        )

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "args": vars(args),
                    "best_val_macro_f1": best_val_macro_f1,
                },
                output_dir / "best_model.pt",
            )

            print(f"Saved new best model with val Macro-F1: {best_val_macro_f1:.4f}")

    print("\nLoading best model for final test evaluation...")
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        id_to_label,
    )

    final_summary = {
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "best_val_macro_f1": best_val_macro_f1,
        "num_labels": len(label_to_id),
        "labels": id_to_label,
    }

    save_json(output_dir / "test_summary.json", final_summary)
    save_json(output_dir / "test_classification_report.json", test_metrics["report"])

    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=[id_to_label[i] for i in range(len(id_to_label))],
        columns=[id_to_label[i] for i in range(len(id_to_label))],
    )
    cm_df.to_csv(output_dir / "test_confusion_matrix.csv")

    test_metrics["predictions"].to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    print("\nFinal test results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")

    print("\nSaved outputs to:")
    print(output_dir)


if __name__ == "__main__":
    main()
