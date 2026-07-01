# Advertisement Moderation Classification Using Visual and Textual Representations

This project focuses on automatic advertisement content classification for safety-oriented moderation.

The goal is to classify advertisement images into four coarse moderation categories:

- `safe_or_irrelevant`
- `substance`
- `violence_or_abuse`
- `bullying`

The project is based on the University of Pittsburgh Ads Dataset.  
The original advertisement topics were mapped into broader moderation-oriented labels, and several visual, textual, and multimodal approaches were evaluated.

---

## Final Report

The final report is available here:

[docs/final_report.pdf](docs/final_report.pdf)

---

## Main Idea

Advertisements often contain both visual and textual information.  
A single advertisement may include objects, people, products, slogans, brand names, prices, and other written content.

Therefore, advertisement moderation is not only an image classification task and not only a text classification task.  
A useful moderation model should be able to use both:

- visual information from the advertisement image
- textual information extracted from the advertisement using OCR

The main research question is:

**Can visual and textual representations help classify advertisements into moderation-related categories?**

---

## Dataset and Splits

The project uses the University of Pittsburgh Ads Dataset.

The original advertisement topics were mapped into four coarse moderation categories:

| Label | Meaning |
|---|---|
| `safe_or_irrelevant` | Regular advertisements that do not match the sensitive categories |
| `substance` | Advertisements related to alcohol, smoking, drugs, or similar products |
| `violence_or_abuse` | Advertisements related to violence, weapons, abuse, or aggressive content |
| `bullying` | Advertisements related to bullying, humiliation, or hostile social content |

Two evaluation settings were used:

1. **Controlled evaluation setting**  
   Used to compare image-only, text-only, and multimodal models under a less extreme imbalance.

2. **Realistic evaluation setting**  
   Used as the final evaluation setup. The test set is highly imbalanced, because in real moderation systems most advertisements are safe.

Final realistic test distribution:

| Label | Test Count |
|---|---:|
| `safe_or_irrelevant` | 7951 |
| `substance` | 522 |
| `violence_or_abuse` | 137 |
| `bullying` | 43 |

---

## Methods

The project compares several approaches:

- Simple baselines
- CLIP zero-shot classification
- CLIP image embeddings with classical classifiers
- OCR text-only classifiers
- Image and OCR text fusion models
- Partial fine-tuning of CLIP
- Confidence-based manual review analysis
- Confidence calibration analysis

---

## Main Evaluation Metric

Because the dataset is highly imbalanced, accuracy alone can be misleading.

For example, a model can achieve high accuracy by predicting most examples as `safe_or_irrelevant`, while still failing to detect smaller sensitive classes.

Therefore, **Macro-F1** was selected as the main metric.  
Macro-F1 gives equal importance to all classes, including small classes such as `bullying`.

---

## Final Results

The best final model was:

**Image and Text Fusion MLP**

This model combines:

- CLIP image embeddings
- OCR-based text features

Final realistic evaluation:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Image and Text Fusion MLP | 0.9092 | 0.6493 | 0.9211 |

The fusion model achieved the best Macro-F1 score and was selected as the final model.

---

## Partial Fine-Tuning Results

Partial fine-tuning of CLIP was also tested.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| CLIP head-only fine-tuning | 0.8506 | 0.5124 | 0.8827 |
| CLIP partial fine-tuning, last visual block | 0.9040 | 0.6033 | 0.9188 |

Partial fine-tuning improved over head-only fine-tuning, but did not outperform the final Image and Text Fusion MLP.

---

## Manual Review Threshold Analysis

A confidence-based manual review policy was evaluated.  
Predictions with confidence below a threshold are sent to manual review, while high-confidence predictions are handled automatically.

| Confidence Threshold | Sent to Review | Review % | Auto Accuracy | Auto Macro-F1 |
|---:|---:|---:|---:|---:|
| 0.50 | 48 | 0.55% | 0.9117 | 0.6613 |
| 0.60 | 223 | 2.58% | 0.9214 | 0.6901 |
| 0.70 | 417 | 4.82% | 0.9298 | 0.7146 |
| 0.80 | 640 | 7.40% | 0.9396 | 0.7417 |
| 0.90 | 991 | 11.45% | 0.9522 | 0.7793 |

This shows the trade-off between automation and reliability.  
A higher threshold sends more examples to manual review, but improves the quality of automatic decisions.

---

## Confidence Calibration Analysis

A confidence calibration analysis was performed for the final Image and Text Fusion MLP.

| Measure | Value |
|---|---:|
| Total examples | 8653 |
| Overall accuracy | 0.9092 |
| Expected Calibration Error | 0.0532 |
| High-confidence examples, confidence >= 0.90 | 7662 |
| High-confidence errors, confidence >= 0.90 | 366 |
| Low-confidence examples, confidence < 0.60 | 223 |

The analysis shows that confidence scores are useful, but not perfect.  
Even when the model is highly confident, mistakes can still occur.  
Therefore, confidence should be used as a decision-support signal rather than as a complete replacement for human review.

---

## Repository Structure

```text
data/
  processed/
    pitt_ads/

docs/
  final_report.tex
  final_report.pdf

reports/
  pitt_ads/
    final_results/
    image_text_fusion/
    clip_partial_finetune_summary/
    confidence_calibration/
    realistic_prediction_examples/

scripts/
  pitt_ads/
    17_train_image_text_fusion.py
    22_train_clip_partial_finetune.py
    23_make_realistic_prediction_examples.py
    25_confidence_calibration_analysis.py
Important Scripts
Script	Purpose
17_train_image_text_fusion.py	Trains image-text fusion models and saves prediction confidence scores
22_train_clip_partial_finetune.py	Runs CLIP head-only and partial fine-tuning experiments
23_make_realistic_prediction_examples.py	Generates confidence, safety, error, and manual review outputs
25_confidence_calibration_analysis.py	Performs confidence calibration analysis
Main Conclusion

The main conclusion is that combining visual and textual information is the best approach for advertisement moderation.

CLIP image embeddings provide strong visual representations.
OCR adds useful textual information from written content inside advertisements.
The final Image and Text Fusion MLP achieved the best Macro-F1 score in the realistic evaluation setting.

The project also shows that high accuracy is not enough for moderation tasks.
Because the data is imbalanced, Macro-F1 and per-class behavior are more important for understanding whether the model detects smaller sensitive categories.

Author

Naama Skital
B.Sc. Computer Science
Ariel University