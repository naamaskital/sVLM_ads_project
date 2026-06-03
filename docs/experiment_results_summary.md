# Experiment Results Summary

## Project goal

The goal of this project is to classify advertisement images into safety-oriented moderation categories.

The labels used in this project are:

- bullying
- safe_or_irrelevant
- substance
- violence_or_abuse

The project uses the Pittsburgh Ads Dataset. The original advertisement topics were mapped into moderation-oriented labels.

## Main research question

The main question is whether general visual representations, such as CLIP image embeddings, are useful for advertisement content moderation.

The project compares:

1. CLIP zero-shot classification
2. Frozen CLIP image embeddings with several classification heads
3. Error analysis of the best-performing model

## Model architecture

The main architecture is modular:

Image  
↓  
Frozen CLIP image encoder  
↓  
Image embedding  
↓  
Classification head  
↓  
Moderation label  

CLIP is used as a fixed visual feature extractor. The classification head is trained separately on top of the extracted embeddings.

## Experiments

The following models were tested:

| Model | Description |
|---|---|
| CLIP zero-shot | CLIP predicts labels using text prompts, without training on our data |
| Logistic Regression | Linear classification head over frozen CLIP embeddings |
| Linear SVM | Linear SVM head over frozen CLIP embeddings |
| Random Forest | Tree-based classification head |
| Gradient Boosting | Boosted tree-based classification head |
| MLP | Small neural classification head over frozen CLIP embeddings |

## Main results

| Model | Accuracy | Macro-F1 | Number of errors |
|---|---:|---:|---:|
| MLP | 0.8744 | 0.7937 | 176 |
| Logistic Regression | 0.8351 | 0.7528 | 231 |
| Gradient Boosting | 0.8572 | 0.7251 | 200 |
| Linear SVM | 0.8258 | 0.7230 | 244 |
| Random Forest | 0.8073 | 0.6222 | 270 |
| CLIP zero-shot | 0.3947 | 0.3442 | 848 |

## Main conclusion

The best-performing model is the MLP classification head trained on top of frozen CLIP image embeddings.

This suggests that CLIP embeddings contain useful visual information for advertisement moderation, but the task still benefits strongly from supervised training on the project-specific labels.

CLIP zero-shot performs much worse than the trained classifiers. This means that general CLIP knowledge is not enough for this specific moderation task without adaptation.

## Confusion matrix for the best model: MLP

| True label | Pred bullying | Pred safe_or_irrelevant | Pred substance | Pred violence_or_abuse |
|---|---:|---:|---:|---:|
| bullying | 22 | 14 | 0 | 7 |
| safe_or_irrelevant | 3 | 639 | 49 | 8 |
| substance | 1 | 56 | 458 | 7 |
| violence_or_abuse | 5 | 21 | 5 | 106 |

## Per-class observations

The model performs well on:

- safe_or_irrelevant
- substance
- violence_or_abuse

The hardest class is bullying.

For bullying, the model correctly predicts 22 out of 43 examples. It misclassifies 14 bullying examples as safe_or_irrelevant and 7 as violence_or_abuse.

This suggests that bullying is visually more subtle than the other categories. It may require understanding text, social context, or implied meaning.

## Main error types for MLP

| Error type | Count |
|---|---:|
| substance -> safe_or_irrelevant | 56 |
| safe_or_irrelevant -> substance | 49 |
| violence_or_abuse -> safe_or_irrelevant | 21 |
| bullying -> safe_or_irrelevant | 14 |
| safe_or_irrelevant -> violence_or_abuse | 8 |
| bullying -> violence_or_abuse | 7 |
| substance -> violence_or_abuse | 7 |

## Error analysis

The main confusion is between substance and safe_or_irrelevant.

This is expected because some substance advertisements look visually similar to ordinary product advertisements. For example, an alcohol advertisement may visually look like a regular beverage advertisement.

Another important issue is bullying. Some bullying examples are predicted as violence_or_abuse, probably because the model focuses on aggressive visual cues instead of understanding the social or textual message.

Some harmful examples are also predicted as safe_or_irrelevant. This is important because in real moderation systems, false negatives are risky: harmful content may remain unfiltered.

## Limitation

The current model is image-only. It does not explicitly use the text that appears inside advertisements.

This is a limitation because many advertisements contain slogans, warnings, product names, or social messages. These textual elements may be important for distinguishing between visually similar categories.

## Next step

The next research step is to add OCR-based text features and compare:

1. Image-only classification
2. OCR/text-only classification
3. Image + OCR multimodal classification

This will help test whether text improves performance, especially for subtle categories such as bullying and for confusing cases such as substance versus safe_or_irrelevant.
