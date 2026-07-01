# CLIP Partial Fine-Tuning Results

This folder summarizes the partial fine-tuning experiments used in the final report.

The fine-tuning experiments were run in an external environment with stronger hardware.
The reproducible training code is available in:

`scripts/pitt_ads/22_train_clip_partial_finetune.py`

Two variants were evaluated:

1. Head-only fine-tuning:
   - The CLIP visual encoder was frozen.
   - Only the classification head was trained.

2. Partial fine-tuning:
   - The classification head was trained.
   - The last CLIP visual block was unfrozen.

## Results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| CLIP head-only fine-tuning | 0.8506 | 0.5124 | 0.8827 |
| CLIP partial fine-tuning, last visual block | 0.9040 | 0.6033 | 0.9188 |

The partial fine-tuning model improved over the head-only model, but did not outperform the final Image and Text Fusion MLP.
