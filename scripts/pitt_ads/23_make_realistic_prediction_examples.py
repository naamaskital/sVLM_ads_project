from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]

SAFE_LABEL = "safe_or_irrelevant"


def main():
    project_root = Path(__file__).resolve().parents[2]

    predictions_path = (
        project_root
        / "reports"
        / "pitt_ads"
        / "image_text_fusion"
        / "fusion_mlp_test_predictions.csv"
    )

    output_dir = (
        project_root
        / "reports"
        / "pitt_ads"
        / "realistic_prediction_examples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manual_review_path = (
        project_root
        / "reports"
        / "pitt_ads"
        / "manual_review_threshold_analysis.csv"
    )

    df = pd.read_csv(predictions_path)

    required_columns = {
        "image_path",
        "true_label",
        "predicted_label",
        "confidence_score",
        "safety_score",
        "unsafe_score",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns in predictions file: "
            + ", ".join(sorted(missing_columns))
            + "\\nRun scripts/pitt_ads/17_train_image_text_fusion.py again first."
        )

    df["is_correct"] = df["true_label"] == df["predicted_label"]
    df["is_error"] = ~df["is_correct"]
    df["error_type"] = df["true_label"] + " -> " + df["predicted_label"]
    df["true_is_unsafe"] = df["true_label"] != SAFE_LABEL
    df["predicted_is_unsafe"] = df["predicted_label"] != SAFE_LABEL

    df.to_csv(output_dir / "fusion_mlp_test_predictions_with_scores.csv", index=False)

    # Qualitative examples for analysis.
    df.sort_values("unsafe_score", ascending=False).head(100).to_csv(
        output_dir / "top_unsafe_predictions.csv",
        index=False,
    )

    df.sort_values("confidence_score", ascending=True).head(100).to_csv(
        output_dir / "most_uncertain_predictions.csv",
        index=False,
    )

    errors = df[df["is_error"]].copy()

    errors.to_csv(output_dir / "all_errors.csv", index=False)

    errors.sort_values("confidence_score", ascending=False).head(100).to_csv(
        output_dir / "high_confidence_errors.csv",
        index=False,
    )

    false_negatives = df[
        (df["true_label"] != SAFE_LABEL)
        & (df["predicted_label"] == SAFE_LABEL)
    ].copy()

    false_negatives.sort_values("confidence_score", ascending=False).to_csv(
        output_dir / "unsafe_predicted_as_safe.csv",
        index=False,
    )

    false_positives = df[
        (df["true_label"] == SAFE_LABEL)
        & (df["predicted_label"] != SAFE_LABEL)
    ].copy()

    false_positives.sort_values("confidence_score", ascending=False).to_csv(
        output_dir / "safe_predicted_as_unsafe.csv",
        index=False,
    )

    error_summary = (
        errors.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    error_summary.to_csv(output_dir / "errors_by_type.csv", index=False)

    # Manual review threshold analysis.
    rows = []
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

    for threshold in thresholds:
        sent_to_review = df["confidence_score"] < threshold
        automatic = ~sent_to_review

        auto_df = df[automatic]

        if len(auto_df) > 0:
            auto_accuracy = accuracy_score(
                auto_df["true_label"],
                auto_df["predicted_label"],
            )
            auto_macro_f1 = f1_score(
                auto_df["true_label"],
                auto_df["predicted_label"],
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        else:
            auto_accuracy = 0.0
            auto_macro_f1 = 0.0

        rows.append({
            "confidence_threshold": threshold,
            "sent_to_review": int(sent_to_review.sum()),
            "review_percent": 100.0 * float(sent_to_review.mean()),
            "auto_accuracy": auto_accuracy,
            "auto_macro_f1": auto_macro_f1,
            "automatic_decisions": int(automatic.sum()),
        })

    review_df = pd.DataFrame(rows)
    review_df.to_csv(manual_review_path, index=False)
    review_df.to_csv(output_dir / "manual_review_threshold_analysis.csv", index=False)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Realistic prediction examples and manual review analysis\\n")
        f.write("=" * 60 + "\\n\\n")
        f.write(f"Input predictions: {predictions_path}\\n")
        f.write(f"Total examples: {len(df)}\\n")
        f.write(f"Errors: {int(df['is_error'].sum())}\\n")
        f.write(f"Unsafe predicted as safe: {len(false_negatives)}\\n")
        f.write(f"Safe predicted as unsafe: {len(false_positives)}\\n\\n")
        f.write("Manual review threshold analysis:\\n")
        f.write(review_df.to_string(index=False))
        f.write("\\n")

    print("Saved prediction examples to:", output_dir)
    print("Saved manual review analysis to:", manual_review_path)
    print(review_df)


if __name__ == "__main__":
    main()
