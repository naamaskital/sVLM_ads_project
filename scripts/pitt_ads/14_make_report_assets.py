from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LABELS = [
    "bullying",
    "safe_or_irrelevant",
    "substance",
    "violence_or_abuse",
]


def plot_confusion_matrix(csv_path, output_path, title):
    cm = pd.read_csv(csv_path, index_col=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm.values)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parents[2]

    error_dir = project_root / "reports" / "pitt_ads" / "error_analysis"
    output_dir = project_root / "reports" / "pitt_ads" / "report_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = error_dir / "summary.csv"
    summary = pd.read_csv(summary_path)

    selected_columns = [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "num_errors",
        "bullying_f1",
        "safe_or_irrelevant_f1",
        "substance_f1",
        "violence_or_abuse_f1",
    ]

    report_table = summary[selected_columns].copy()
    report_table = report_table.sort_values("macro_f1", ascending=False)

    report_table.to_csv(output_dir / "final_model_comparison.csv", index=False)

    markdown_path = output_dir / "final_model_comparison.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(report_table.to_markdown(index=False))

    plot_confusion_matrix(
        error_dir / "mlp_confusion_matrix.csv",
        output_dir / "mlp_confusion_matrix.png",
        "MLP Head Confusion Matrix",
    )

    plot_confusion_matrix(
        error_dir / "clip_zero_shot_confusion_matrix.csv",
        output_dir / "clip_zero_shot_confusion_matrix.png",
        "CLIP Zero-Shot Confusion Matrix",
    )

    mlp_errors = pd.read_csv(error_dir / "mlp_errors_by_type.csv")
    mlp_errors.head(12).to_csv(output_dir / "top_mlp_error_types.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    top_errors = mlp_errors.head(10)
    ax.barh(top_errors["error_type"], top_errors["count"])
    ax.set_xlabel("Number of errors")
    ax.set_ylabel("Error type")
    ax.set_title("Top MLP Error Types")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "top_mlp_error_types.png", dpi=200)
    plt.close()

    print("Saved report assets to:")
    print(output_dir)

    print()
    print("Final model comparison:")
    print(report_table)


if __name__ == "__main__":
    main()
