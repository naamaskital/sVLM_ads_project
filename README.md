# Ads Moderation with Vision-Language Models

This project develops an automatic moderation pipeline for advertisement images using vision-language models, OCR, classical machine learning classifiers, and confidence-based manual review.

The goal is to classify advertisement images into moderation-related categories such as safe content, substance-related content, violence or abuse, and bullying. The project focuses not only on achieving high classification performance, but also on building a realistic moderation system that can decide when to classify automatically and when to send uncertain cases to human review.

## Project Motivation

Online advertisements appear at very large scale across websites, social platforms, and recommendation systems. Since ads can include visual content, text, branding, products, and contextual cues, moderation is more complex than ordinary image classification.

A practical moderation system should be able to:

* Detect unsafe or sensitive advertisement content.
* Handle both visual and textual signals inside an ad.
* Work under realistic class imbalance.
* Provide a confidence score for each prediction.
* Route uncertain or risky predictions to manual review.

This project explores these goals using CLIP-based image representations, OCR-based text features, image-text fusion, partial fine-tuning, and confidence calibration.

## Dataset

The project is based on the University of Pittsburgh Ads Dataset, which contains approximately 64K advertisement images. The original dataset contains many advertisement topics, and in this project they are mapped into a moderation-oriented label space.

The final classification task contains four labels:

| Label                | Meaning                                                                         |
| -------------------- | ------------------------------------------------------------------------------- |
| `safe_or_irrelevant` | Ads that are safe or not relevant to the unsafe categories                      |
| `substance`          | Ads related to alcohol, smoking, drugs, or similar substance-related content    |
| `violence_or_abuse`  | Ads containing violent, abusive, or threatening visual/textual cues             |
| `bullying`           | Ads that may include bullying, harassment, humiliation, or aggressive messaging |

A realistic split was used, where the training set is more balanced while validation and test sets remain closer to the natural imbalanced distribution. This makes the evaluation more realistic for moderation, where unsafe classes are usually much less common than safe content.

## Main Research Question

Can vision-language models, OCR text extraction, and confidence-based routing improve advertisement moderation under realistic class imbalance?

More specifically, the project investigates:

1. How well CLIP image embeddings perform for ad moderation.
2. Whether OCR text improves moderation performance.
3. Whether image-text fusion improves over image-only models.
4. Whether partial fine-tuning of CLIP improves performance.
5. Whether confidence scores can support a practical manual review mechanism.

## Methodology

The project was built as a multi-stage experimental pipeline.

### 1. Dataset Construction

The raw advertisement dataset was processed and mapped into moderation categories. The pipeline builds CSV files containing image paths, labels, and split information.

Main scripts:

```bash
python scripts/pitt_ads/03_build_pitt_topics_csv.py
python scripts/pitt_ads/04_build_pitt_moderation_dataset.py
python scripts/pitt_ads/20_create_realistic_splits.py
```

### 2. CLIP Image Embeddings

The first main representation is based on CLIP ViT-B/32. Each advertisement image is passed through CLIP, and the resulting image embedding is used as a feature vector for classical classifiers.

Main script:

```bash
python scripts/pitt_ads/21_extract_realistic_train_clip_embeddings.py
```

### 3. Classical Classifiers on CLIP Embeddings

Several classifiers were trained on top of frozen CLIP embeddings:

* Logistic Regression
* Linear SVM
* MLP classifier

Main script:

```bash
python scripts/pitt_ads/22_train_realistic_embedding_classifiers.py
```

These models provide strong baselines and show how much information can be extracted from frozen CLIP image features.

### 4. Partial Fine-Tuning of CLIP

In addition to frozen embeddings, the project tested partial fine-tuning of CLIP. The model is initialized from a pretrained CLIP vision encoder and trained with a small classification head. Some experiments train only the head, while others unfreeze the last CLIP vision layers.

Main script:

```bash
python scripts/pitt_ads/22_train_clip_partial_finetune.py
```

Final partial fine-tuning result:

| Model                    | Test Accuracy | Macro-F1 | Weighted-F1 |
| ------------------------ | ------------: | -------: | ----------: |
| CLIP partial fine-tuning |        0.9067 |   0.6397 |      0.9189 |

### 5. OCR Text Baseline

Advertisements often contain important textual information, such as product names, slogans, warning words, prices, and sensitive phrases. Therefore, OCR text was extracted from the images and used as an additional signal.

Main scripts:

```bash
python scripts/pitt_ads/15_extract_ocr_text.py
python scripts/pitt_ads/16_train_ocr_text_baseline.py
```

The OCR baseline helps evaluate whether text alone can provide useful moderation information.

### 6. Image + OCR Fusion

The final and best-performing approach combines:

* CLIP image embeddings
* OCR-based text features

The combined feature vector is used to train fusion classifiers.

Main script:

```bash
python scripts/pitt_ads/17_train_image_text_fusion.py
```

Final image-text fusion results:

| Model                      | Split |   Accuracy |   Macro-F1 | Weighted-F1 |
| -------------------------- | ----- | ---------: | ---------: | ----------: |
| Fusion Logistic Regression | Test  |     0.8520 |     0.5285 |      0.8805 |
| Fusion Linear SVM          | Test  |     0.8565 |     0.5105 |      0.8842 |
| Fusion MLP                 | Test  | **0.9100** | **0.6551** |  **0.9217** |

The best final model is:

```text
fusion_mlp
```

It achieved:

```text
Test Accuracy: 91.00%
Test Macro-F1: 65.51%
Test Weighted-F1: 92.17%
```

## Final Model Performance

The best model was the Image + OCR Fusion MLP.

Final summary:

| Metric      |  Value |
| ----------- | -----: |
| Accuracy    | 0.9100 |
| Macro-F1    | 0.6551 |
| Weighted-F1 | 0.9217 |

Per-class performance for the best model:

| Class                |     F1 |
| -------------------- | -----: |
| `safe_or_irrelevant` | 0.9503 |
| `substance`          | 0.6065 |
| `violence_or_abuse`  | 0.6075 |
| `bullying`           | 0.4561 |

The model performs very well on the majority safe class and achieves meaningful detection ability on the minority unsafe classes. The `bullying` class remains the hardest class, mainly because it is smaller, more subjective, and often depends on subtle visual or textual context.

## Confidence Calibration and Manual Review

A moderation system should not only output a class label, but also estimate how confident it is. This project therefore includes confidence analysis and manual review thresholding.

Main script:

```bash
python scripts/pitt_ads/25_confidence_calibration_analysis.py
```

Confidence analysis results:

| Metric                                       |  Value |
| -------------------------------------------- | -----: |
| Total test examples                          |  8,653 |
| Overall accuracy                             | 0.9100 |
| Expected Calibration Error                   | 0.0524 |
| High-confidence examples, confidence >= 0.90 |  7,667 |
| High-confidence errors, confidence >= 0.90   |    371 |
| Low-confidence examples, confidence < 0.60   |    243 |

Manual review threshold analysis:

| Confidence Threshold | Sent to Review | Review Percent | Automatic Accuracy | Automatic Macro-F1 |
| -------------------: | -------------: | -------------: | -----------------: | -----------------: |
|                 0.50 |             43 |          0.50% |             0.9124 |             0.6644 |
|                 0.60 |            243 |          2.81% |             0.9221 |             0.6921 |
|                 0.70 |            411 |          4.75% |             0.9302 |             0.7133 |
|                 0.80 |            649 |          7.50% |             0.9400 |             0.7436 |
|                 0.90 |            986 |         11.39% |             0.9516 |             0.7777 |

A threshold of 0.80 gives a strong practical tradeoff: only 7.5% of test examples are sent to manual review, while the automatic decisions reach 94.00% accuracy and 74.36% Macro-F1.

## Error Analysis

The project includes analysis of:

* False positives: safe ads predicted as unsafe.
* False negatives: unsafe ads predicted as safe.
* High-confidence mistakes.
* Most uncertain predictions.
* Top unsafe predictions.

Main scripts:

```bash
python scripts/pitt_ads/13_error_analysis.py
python scripts/pitt_ads/23_make_realistic_prediction_examples.py
```

Important output folders:

```bash
reports/pitt_ads/error_analysis/
reports/pitt_ads/realistic_prediction_examples/
```

The error analysis shows that many mistakes occur in visually ambiguous ads, ads with weak or misleading OCR text, and cases where unsafe categories overlap semantically, such as substance-related ads that visually resemble ordinary food or lifestyle advertisements.

## Final Outputs

The main generated reports and assets are saved under:

```bash
reports/pitt_ads/
```

Important final result files:

```bash
reports/pitt_ads/final_results/final_summary.txt
reports/pitt_ads/final_results/test_results_ranked.csv
reports/pitt_ads/final_results/best_model_confusion_matrix.csv
reports/pitt_ads/image_text_fusion/image_text_fusion_model_comparison.csv
reports/pitt_ads/confidence_calibration/summary.txt
reports/pitt_ads/manual_review_threshold_analysis.csv
reports/pitt_ads/realistic_prediction_examples/
reports/pitt_ads/report_assets/
```

Useful report assets:

```bash
reports/pitt_ads/report_assets/final_model_comparison.csv
reports/pitt_ads/report_assets/final_model_comparison.md
reports/pitt_ads/report_assets/mlp_confusion_matrix.png
reports/pitt_ads/report_assets/top_mlp_error_types.png
```

## Repository Structure

```text
scripts/pitt_ads/
├── 03_build_pitt_topics_csv.py
├── 04_build_pitt_moderation_dataset.py
├── 15_extract_ocr_text.py
├── 16_train_ocr_text_baseline.py
├── 17_train_image_text_fusion.py
├── 19_collect_final_results.py
├── 20_create_realistic_splits.py
├── 21_extract_realistic_train_clip_embeddings.py
├── 22_train_realistic_embedding_classifiers.py
├── 22_train_clip_partial_finetune.py
├── 23_make_realistic_prediction_examples.py
├── 25_confidence_calibration_analysis.py
├── 13_error_analysis.py
└── 14_make_report_assets.py
```

Generated output folders:

```text
reports/pitt_ads/
├── clip_partial_finetune/
├── clip_partial_finetune_head_only/
├── clip_partial_finetune_unfreeze1/
├── confidence_calibration/
├── error_analysis/
├── final_results/
├── image_text_fusion/
├── realistic_prediction_examples/
├── realistic_train_clip_classifiers/
└── report_assets/
```

## How to Reproduce the Main Final Pipeline

The expected image folder is:

```bash
data/raw/pitt_ads/images/
```

Then run:

```bash
python scripts/pitt_ads/20_create_realistic_splits.py
python scripts/pitt_ads/21_extract_realistic_train_clip_embeddings.py
python scripts/pitt_ads/22_train_realistic_embedding_classifiers.py
python scripts/pitt_ads/22_train_clip_partial_finetune.py
python scripts/pitt_ads/15_extract_ocr_text.py
python scripts/pitt_ads/16_train_ocr_text_baseline.py
python scripts/pitt_ads/17_train_image_text_fusion.py
python scripts/pitt_ads/23_make_realistic_prediction_examples.py
python scripts/pitt_ads/25_confidence_calibration_analysis.py
python scripts/pitt_ads/19_collect_final_results.py
python scripts/pitt_ads/13_error_analysis.py
python scripts/pitt_ads/14_make_report_assets.py
```

## Main Conclusions

1. Frozen CLIP embeddings already provide a strong baseline for advertisement moderation.
2. Partial fine-tuning improves the task-specific adaptation of CLIP and reaches strong results.
3. OCR text adds useful information because many ads include meaningful textual cues.
4. The best model is the Image + OCR Fusion MLP, which combines visual and textual representations.
5. Confidence-based routing makes the model more practical for real moderation systems.
6. A manual review threshold can improve the quality of automatic decisions while reviewing only a small percentage of examples.
7. Minority classes remain challenging, especially bullying, due to class imbalance, ambiguity, and subjective interpretation.

## Final Result

The final system combines image understanding, OCR text, and confidence-based decision routing.

Best model:

```text
Image + OCR Fusion MLP
```

Final test performance:

```text
Accuracy: 91.00%
Macro-F1: 65.51%
Weighted-F1: 92.17%
```

With a confidence threshold of 0.80, only 7.5% of examples are routed to manual review, while automatic decisions reach 94.00% accuracy.

## Future Work

Possible future improvements include:

* Using stronger vision-language models.
* Fine-tuning larger CLIP variants.
* Improving OCR quality and text preprocessing.
* Adding multimodal transformer fusion instead of feature concatenation.
* Expanding the moderation label set.
* Adding human-in-the-loop active learning.
* Testing the system on additional advertisement datasets.
