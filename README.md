# sVLM Ads Project

This project focuses on automatic advertisement content classification.

The goal is to classify advertisement images into coarse safety-related categories:
- safe_or_irrelevant
- substance
- violence_or_abuse
- bullying

The project uses the Pittsburgh Ads Dataset and maps its original advertisement topics into moderation-oriented labels.

## Current pipeline

1. Build topic-level CSV from the Pittsburgh Ads annotations.
2. Map Pittsburgh advertisement topics into moderation labels.
3. Create coarse safety labels.
4. Balance the dataset.
5. Split the data into train, validation, and test sets.
6. Inspect the split using a local HTML viewer.
7. Train an image-only CLIP baseline with a classical classifier.

## Research direction

The project compares whether general visual representations, such as CLIP image embeddings, are sufficient for advertisement content classification.

Later experiments will compare:
- CLIP zero-shot classification
- CLIP embeddings + classical classifiers
- different classification heads
- error analysis by class
