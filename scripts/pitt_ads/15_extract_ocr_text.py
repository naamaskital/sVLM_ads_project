import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import pytesseract


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


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    split_path = (
        project_root
        / "data"
        / "processed"
        / "pitt_ads"
        / "splits"
        / f"{args.split}.csv"
    )

    output_dir = project_root / "data" / "processed" / "pitt_ads" / "ocr"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_images is None:
        output_path = output_dir / f"{args.split}_ocr.csv"
    else:
        output_path = output_dir / f"{args.split}_ocr_first_{args.max_images}.csv"

    if output_path.exists() and not args.overwrite:
        print(f"Output already exists: {output_path}")
        print("Use --overwrite if you want to recreate it.")
        return

    df = pd.read_csv(split_path)

    label_col = find_label_column(df)
    image_col = find_image_column(df)

    if args.max_images is not None:
        df = df.head(args.max_images).copy()

    rows = []

    missing_images = 0
    unreadable_images = 0
    ocr_errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"OCR {args.split}"):
        image_path = resolve_image_path(row[image_col], project_root)

        if image_path is None:
            missing_images += 1
            ocr_text = ""
            status = "missing_image"
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                unreadable_images += 1
                ocr_text = ""
                status = "unreadable_image"
            else:
                try:
                    ocr_text = pytesseract.image_to_string(image, lang=args.lang)
                    ocr_text = clean_text(ocr_text)
                    status = "ok"
                except Exception:
                    ocr_errors += 1
                    ocr_text = ""
                    status = "ocr_error"

        rows.append({
            "image_path": row[image_col],
            "label": row[label_col],
            "ocr_text": ocr_text,
            "ocr_status": status,
            "ocr_text_length": len(ocr_text),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    print()
    print(f"Saved OCR file to: {output_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Missing images: {missing_images}")
    print(f"Unreadable images: {unreadable_images}")
    print(f"OCR errors: {ocr_errors}")
    print(f"Rows with text: {(out_df['ocr_text_length'] > 0).sum()}")

    print()
    print("Sample OCR results:")
    sample = out_df[out_df["ocr_text_length"] > 0].head(10)
    for _, row in sample.iterrows():
        print("-" * 80)
        print(f"Label: {row['label']}")
        print(f"Text: {row['ocr_text'][:300]}")


if __name__ == "__main__":
    main()
