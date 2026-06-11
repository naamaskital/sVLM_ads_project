from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


SAFE_LABEL = "safe_or_irrelevant"


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def predict_with_scores(model, X):
    pred = model.predict(X)

    if not hasattr(model, "predict_proba"):
        return pred, None, None

    proba = model.predict_proba(X)
    classes = list(model.classes_)

    confidence_score = proba.max(axis=1)

    if SAFE_LABEL in classes:
        safe_index = classes.index(SAFE_LABEL)
        safety_score = proba[:, safe_index]
    else:
        safety_score = np.full(len(pred), np.nan)

    unsafe_score = 1.0 - safety_score

    score_df = pd.DataFrame({
        "predicted_label": pred,
        "confidence_score": confidence_score,
        "safety_score": safety_score,
        "unsafe_score": unsafe_score,
    })

    for i, class_name in enumerate(classes):
        score_df[f"prob_{class_name}"] = proba[:, i]

    return pred, score_df, classes


def evaluate(model, X, y, labels):
    pred, score_df, classes = predict_with_scores(model, X)

    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "per_class_report": classification_report(
            y,
            pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }

    return metrics, score_df


def save_predictions(output_path, y_true, score_df):
    predictions = score_df.copy()
    predictions.insert(0, "true_label", y_true)
    predictions.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings-dir",
        default="reports/pitt_ads/realistic_train_clip_embeddings",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/pitt_ads/realistic_train_clip_classifiers",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["logistic_regression", "linear_svm", "mlp"],
        choices=[
            "logistic_regression",
            "linear_svm",
            "random_forest",
            "gradient_boosting",
            "mlp",
        ],
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_npz(embeddings_dir / "train_clip_vit_b32_embeddings.npz")
    X_val, y_val = load_npz(embeddings_dir / "val_clip_vit_b32_embeddings.npz")
    X_test, y_test = load_npz(embeddings_dir / "test_clip_vit_b32_embeddings.npz")

    labels = sorted(pd.Series(y_train).unique().tolist())

    print("Train:", X_train.shape, pd.Series(y_train).value_counts().to_dict())
    print("Val:", X_val.shape, pd.Series(y_val).value_counts().to_dict())
    print("Test:", X_test.shape, pd.Series(y_test).value_counts().to_dict())
    print("Labels:", labels)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),

        # Calibrated SVM gives probabilities, so we can compute confidence/safety scores.
        "linear_svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=42,
                ),
                cv=3,
            ),
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
        ),

        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256,),
                activation="relu",
                max_iter=500,
                early_stopping=False,
                random_state=42,
            ),
        ),
    }

    rows = []

    for name in args.classifiers:
        print()
        print("=" * 80)
        print("Training:", name)

        model = models[name]
        model.fit(X_train, y_train)

        val_metrics, val_scores = evaluate(model, X_val, y_val, labels)
        test_metrics, test_scores = evaluate(model, X_test, y_test, labels)

        print("VAL accuracy:", round(val_metrics["accuracy"], 4))
        print("VAL macro_f1:", round(val_metrics["macro_f1"], 4))
        print("TEST accuracy:", round(test_metrics["accuracy"], 4))
        print("TEST macro_f1:", round(test_metrics["macro_f1"], 4))

        rows.append({
            "classifier": name,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        })

        with open(output_dir / f"{name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "classifier": name,
                    "labels": labels,
                    "val": val_metrics,
                    "test": test_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        save_predictions(
            output_dir / f"{name}_val_predictions_with_scores.csv",
            y_val,
            val_scores,
        )

        save_predictions(
            output_dir / f"{name}_test_predictions_with_scores.csv",
            y_test,
            test_scores,
        )

    results = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    results.to_csv(output_dir / "realistic_classifier_results.csv", index=False)

    print()
    print("=" * 80)
    print("Final results:")
    print(results.to_string(index=False))
    print()
    print("Saved to:", output_dir / "realistic_classifier_results.csv")
    print("Saved prediction files with confidence_score, safety_score, unsafe_score")


if __name__ == "__main__":
    main()
