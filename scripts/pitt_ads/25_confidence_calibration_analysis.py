from pathlib import Path

import numpy as np
import pandas as pd


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
        / "confidence_calibration"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)

    required_columns = {
        "true_label",
        "predicted_label",
        "confidence_score",
        "safety_score",
        "unsafe_score",
        "is_correct",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
            + "\nRun scripts/pitt_ads/17_train_image_text_fusion.py first."
        )

    df["is_correct"] = df["true_label"] == df["predicted_label"]

    bins = np.linspace(0.0, 1.0, 11)
    df["confidence_bin"] = pd.cut(
        df["confidence_score"],
        bins=bins,
        include_lowest=True,
    )

    rows = []
    ece = 0.0
    total = len(df)

    for bin_name, group in df.groupby("confidence_bin", observed=True):
        if len(group) == 0:
            continue

        bin_count = len(group)
        avg_confidence = group["confidence_score"].mean()
        accuracy = group["is_correct"].mean()
        error_rate = 1.0 - accuracy

        ece += (bin_count / total) * abs(avg_confidence - accuracy)

        rows.append(
            {
                "confidence_bin": str(bin_name),
                "count": bin_count,
                "percent": 100.0 * bin_count / total,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "error_rate": error_rate,
                "calibration_gap": avg_confidence - accuracy,
            }
        )

    calibration_df = pd.DataFrame(rows)
    calibration_df.to_csv(output_dir / "confidence_bins.csv", index=False)

    high_conf_errors = df[
        (df["confidence_score"] >= 0.90)
        & (~df["is_correct"])
    ].copy()

    high_conf_errors.to_csv(
        output_dir / "high_confidence_errors_090.csv",
        index=False,
    )

    very_low_conf = df[df["confidence_score"] < 0.60].copy()
    very_low_conf.to_csv(
        output_dir / "low_confidence_predictions_under_060.csv",
        index=False,
    )

    summary = {
        "total_examples": total,
        "overall_accuracy": float(df["is_correct"].mean()),
        "expected_calibration_error_ece": float(ece),
        "high_confidence_090_count": int((df["confidence_score"] >= 0.90).sum()),
        "high_confidence_090_errors": int(len(high_conf_errors)),
        "low_confidence_under_060_count": int(len(very_low_conf)),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "confidence_summary.csv", index=False)

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Confidence calibration analysis\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Total examples: {summary['total_examples']}\n")
        f.write(f"Overall accuracy: {summary['overall_accuracy']:.4f}\n")
        f.write(f"Expected Calibration Error: {summary['expected_calibration_error_ece']:.4f}\n")
        f.write(f"High-confidence examples >= 0.90: {summary['high_confidence_090_count']}\n")
        f.write(f"High-confidence errors >= 0.90: {summary['high_confidence_090_errors']}\n")
        f.write(f"Low-confidence examples < 0.60: {summary['low_confidence_under_060_count']}\n\n")
        f.write("Confidence bins:\n")
        f.write(calibration_df.to_string(index=False))
        f.write("\n")

    print("Saved calibration analysis to:", output_dir)
    print(pd.DataFrame([summary]))
    print(calibration_df)


if __name__ == "__main__":
    main()
