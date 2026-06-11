from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data/processed/pitt_ads/splits_realistic"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/pitt_ads/realistic_train_experiment/clip_embeddings"

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_image(path):
    image = Image.open(path)
    return image.convert("RGB")


def resolve_image_path(row):
    image_path = Path(str(row["image_path"]))

    if image_path.is_absolute():
        return image_path

    return PROJECT_ROOT / image_path


def extract_embeddings(df, processor, model, device, batch_size):
    all_embeddings = []
    all_labels = []
    all_paths = []
    failed = 0

    rows = df.to_dict("records")

    for start in tqdm(range(0, len(rows), batch_size), desc="Extracting CLIP embeddings"):
        batch_rows = rows[start:start + batch_size]

        images = []
        labels = []
        paths = []

        for row in batch_rows:
            path = resolve_image_path(row)

            try:
                image = load_image(path)
            except Exception:
                failed += 1
                continue

            images.append(image)
            labels.append(str(row["coarse_label"]))
            paths.append(str(row["image_path"]))

        if not images:
            continue

        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu().numpy().astype("float32"))
        all_labels.extend(labels)
        all_paths.extend(paths)

    if not all_embeddings:
        raise ValueError("No embeddings were extracted.")

    X = np.vstack(all_embeddings)
    y = np.array(all_labels)
    image_paths = np.array(all_paths)

    return X, y, image_paths, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_path = Path(args.splits_dir) / f"{args.split}.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.split}_clip_vit_b32_embeddings.npz"

    if output_path.exists() and not args.force:
        print(f"Output already exists: {output_path}")
        print("Use --force to overwrite.")
        return

    print(f"Reading split: {split_path}")
    df = pd.read_csv(split_path)

    if "image_exists" in df.columns:
        df = df[df["image_exists"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    print(f"Rows to process: {len(df)}")
    print("Label distribution:")
    print(df["coarse_label"].value_counts())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.to(device)
    model.eval()

    X, y, image_paths, failed = extract_embeddings(
        df=df,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        image_paths=image_paths,
    )

    print()
    print(f"Saved: {output_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"failed images: {failed}")


if __name__ == "__main__":
    main()
