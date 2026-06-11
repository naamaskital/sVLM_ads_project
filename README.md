# sVLM Ads Project

This project focuses on automatic advertisement content classification for safety-oriented moderation.

The goal is to classify advertisement images into coarse moderation categories:

- `safe_or_irrelevant`
- `substance`
- `violence_or_abuse`
- `bullying`

The project uses the University of Pittsburgh Ads Dataset.  
The original advertisement topics were mapped into moderation-oriented labels, and several image, text, and fusion-based models were evaluated.

---

## Project Goal

Advertisements can contain both visual and textual signals that may indicate unsafe or sensitive content.

The main research question in this project is:

**Can visual-language representations help classify advertisements into moderation-related categories?**

To answer this, the project compares:

1. Simple baselines
2. CLIP zero-shot classification
3. CLIP image embeddings with classical classifiers
4. OCR text-only classifiers
5. Image + text fusion models

---

## Dataset

The project is based on the Pittsburgh Ads Dataset.

The original advertisement topics were mapped into coarse moderation labels:

| Coarse label | Meaning |
|---|---|
| `safe_or_irrelevant` | Regular ads that do not match the sensitive categories |
| `substance` | Ads related to alcohol, smoking, drugs, or similar products |
| `violence_or_abuse` | Ads related to violence, weapons, abuse, or aggressive content |
| `bullying` | Ads related to bullying, humiliation, or hostile social content |

---

## Final Data Splits

The final evaluation uses a realistic imbalanced test set, because in real moderation settings most ads are safe.

Final realistic test distribution:

| Label | Test count |
|---|---:|
| `bullying` | 43 |
| `safe_or_irrelevant` | 7951 |
| `substance` | 522 |
| `violence_or_abuse` | 137 |

A balanced-train / realistic-evaluation setup was also created.  
In this setup, the training set is less dominated by the safe class, while validation and test remain realistic and imbalanced.

Balanced training distribution:

| Label | Train count |
|---|---:|
| `bullying` | 195 |
| `safe_or_irrelevant` | 3261 |
| `substance` | 2429 |
| `violence_or_abuse` | 637 |

---

## Pipeline

The main pipeline includes the following stages:

1. Build topic-level CSV files from the Pittsburgh Ads annotations.
2. Map original advertisement topics into moderation labels.
3. Create coarse moderation labels.
4. Create balanced and realistic train/validation/test splits.
5. Generate local HTML viewers for manual inspection.
6. Extract CLIP image embeddings.
7. Train image-only classifiers on CLIP embeddings.
8. Run CLIP zero-shot classification.
9. Extract OCR text from advertisements.
10. Train OCR text-only classifiers.
11. Train image + text fusion classifiers.
12. Collect final results and error analysis.

---

## Models

### 1. Simple Baselines

Two simple baselines were used:

- Majority baseline
- Stratified random baseline

These baselines are important because the final test set is highly imbalanced.  
For example, the majority baseline achieves high accuracy by predicting mostly `safe_or_irrelevant`, but it fails on the sensitive categories.

### 2. CLIP Zero-Shot

CLIP zero-shot classification was tested using text prompts for the target moderation categories.

This model does not train on the project dataset.  
It directly compares the image representation to textual class descriptions.

The zero-shot result was weak, showing that generic prompts are not enough for this task.

### 3. CLIP Image Embeddings + Classical Classifiers

CLIP was used as a frozen image feature extractor.

The extracted image embeddings were used to train several classifiers:

- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting
- MLP

This approach tests whether strong visual embeddings are enough for advertisement moderation.

### 4. OCR Text-Only Models

OCR was used to extract text from advertisement images.

Then text-only classifiers were trained on the extracted text.

This checks whether the text inside the advertisement is useful by itself.

The OCR-only models helped in some cases, but text alone was not enough because many ads contain important visual information.

### 5. Image + Text Fusion

The final fusion models combine:

- CLIP image embeddings
- OCR-based text features

The best model was:

**Image + Text Fusion MLP**

This model performed best according to macro-F1, which is the most important metric for this project because the dataset is imbalanced.

---

## Final Results

The best final model is:

| Model | Group | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| `fusion_mlp` | `image_text_fusion` | 0.9092 | 0.6493 | 0.9211 |

Even though some models achieved higher accuracy, macro-F1 is more important here because it gives more weight to all classes, including small sensitive classes such as `bullying`.

Comparison of selected test results:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| `fusion_mlp` | 0.9092 | 0.6493 |
| `gradient_boosting` | 0.9203 | 0.6390 |
| `random_forest` | 0.9401 | 0.6234 |
| `majority_baseline` | 0.9189 | 0.2394 |

The majority baseline proves why accuracy alone is misleading in this project.

---

## Best Model Confusion Matrix

Best model: `fusion_mlp`

| True label | Pred bullying | Pred safe | Pred substance | Pred violence |
|---|---:|---:|---:|---:|
| `bullying` | 25 | 12 | 2 | 4 |
| `safe_or_irrelevant` | 43 | 7275 | 518 | 115 |
| `substance` | 1 | 64 | 454 | 3 |
| `violence_or_abuse` | 3 | 16 | 5 | 113 |

The model performs well on the large safe class and also detects many `substance` and `violence_or_abuse` examples.

The hardest class is `bullying`, mainly because it has very few examples in the dataset and can be visually subtle.

---

## Error Analysis

The most common error types for the image-only MLP were:

| Error type | Count |
|---|---:|
| `substance -> safe_or_irrelevant` | 56 |
| `safe_or_irrelevant -> substance` | 49 |
| `violence_or_abuse -> safe_or_irrelevant` | 21 |
| `bullying -> safe_or_irrelevant` | 14 |
| `safe_or_irrelevant -> violence_or_abuse` | 8 |

These errors show that the model sometimes misses harmful content and sometimes over-detects sensitive categories in safe ads.

---

## Main Conclusion

The best approach is to combine visual and textual information.

CLIP image embeddings provide strong visual features, while OCR text adds useful information from slogans, product names, and written content inside the ad.

The final Image + Text Fusion MLP achieved the best macro-F1 and is therefore the selected final model.

---

## Future Work

Future extensions of this project can include partial fine-tuning of CLIP.

Instead of using CLIP only as a frozen feature extractor, a future experiment can freeze most of the CLIP model and train only a small classification head or the last visual layers.

This would allow comparing several levels of model adaptation:

1. CLIP zero-shot classification
2. Frozen CLIP embeddings with classical classifiers
3. Image + text fusion using CLIP and OCR
4. Partial fine-tuning of CLIP on the advertisement moderation dataset

This experiment should preferably be run on a GPU, because fine-tuning CLIP on many advertisement images is computationally expensive.

---

## Repository Structure

```text
data/
  processed/pitt_ads/
    splits/
    splits_realistic/
    splits_balanced_train_realistic_eval/
    ocr/

reports/
  pitt_ads/
    simple_baselines/
    clip_zero_shot/
    classical_heads/
    ocr_text_baseline/
    image_text_fusion/
    error_analysis/
    final_results/
    report_assets/

scripts/
  pitt_ads/
    03_build_pitt_topics_csv.py
    04_build_pitt_moderation_dataset.py
    06_create_balanced_coarse_dataset.py
    07_split_coarse_dataset.py
    10_train_clip_baseline.py
    11_clip_zero_shot.py
    12_train_classical_heads.py
    13_error_analysis.py
    14_make_report_assets.py
    15_extract_ocr_text.py
    16_train_ocr_text_baseline.py
    17_train_image_text_fusion.py
    18_train_simple_baselines.py
    19_collect_final_results.py
    20_create_realistic_splits.py
```

---

## How to Run

Activate the environment:

```bash
source .venv/bin/activate
```

Run the final result collection script:

```bash
python3 scripts/pitt_ads/19_collect_final_results.py
```

The final results are saved under:

```text
reports/pitt_ads/final_results/
```

