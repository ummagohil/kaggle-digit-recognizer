"""Digit classifier, built step by step.

Load images, evaluate a baseline, then train on all labels and export predictions.
Run from the project folder: python3 src/train.py
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


PROJECT_DIR = Path(__file__).resolve().parents[1]
PIXEL_COLUMNS = [f"pixel{i}" for i in range(784)]


def load_pixels(path: Path, *, labelled: bool):
    """Read one CSV and return normalised pixels plus labels, when available."""
    table = pd.read_csv(path)
    expected_columns = (["label"] if labelled else []) + PIXEL_COLUMNS
    if table.columns.tolist() != expected_columns:
        raise ValueError(f"{path.name}: unexpected column names or order")
    if table.empty:
        raise ValueError(f"{path.name}: contains no images")

    # Check values before conversion so invalid data cannot be silently rounded.
    pixels = table[PIXEL_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(pixels).all():
        raise ValueError(f"{path.name}: missing or nonfinite pixels")
    if ((pixels < 0) | (pixels > 255)).any():
        raise ValueError(f"{path.name}: pixels must be between 0 and 255")
    if (pixels != np.floor(pixels)).any():
        raise ValueError(f"{path.name}: pixels must be whole numbers")

    # X contains the evidence: one image per row, with 784 pixel features.
    # Scaling changes 0–255 into 0–1 while preserving the greyscale information.
    X = pixels.astype(np.float32) / 255.0

    # y contains the answers. Test images have no known answers.
    y = None
    if labelled:
        if not table["label"].isin(range(10)).all():
            raise ValueError(f"{path.name}: labels must be whole digits from 0 to 9")
        y = table["label"].to_numpy(dtype=np.int64)
    return X, y


def main():
    X, y = load_pixels(PROJECT_DIR / "data" / "train.csv", labelled=True)
    X_test, _ = load_pixels(PROJECT_DIR / "data" / "test.csv", labelled=False)

    print(f"Labelled images: {X.shape[0]:,}")
    print(f"Pixels per image: {X.shape[1]} (28 × 28)")
    print(f"Training labels: {np.unique(y).tolist()}")
    print(f"Pixel range after scaling: {X.min():.1f} to {X.max():.1f}")
    print(f"Test images to predict later: {X_test.shape[0]:,}")

    # Reserve 20% of the known answers. The model never sees these during fitting.
    # The audit found no exact duplicate images in this dataset.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(y_train):,} images; validating on {len(y_valid):,}.", flush=True)

    # Treat nonconvergence as an error so an unfinished fit is not silently used.
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=2):
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(X_train, y_train)
        valid_predictions = model.predict(X_valid)

        accuracy = accuracy_score(y_valid, valid_predictions)
        matrix = confusion_matrix(y_valid, valid_predictions, labels=range(10))
        print(f"Validation accuracy: {accuracy:.2%}", flush=True)
        print("Confusion matrix: rows = true digit, columns = predicted digit")
        print(matrix, flush=True)

        report_dir = PROJECT_DIR / "reports" / "baseline"
        report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            matrix,
            index=pd.Index(range(10), name="true_digit"),
            columns=[f"predicted_{i}" for i in range(10)],
        ).to_csv(report_dir / "confusion_matrix.csv")
        pd.DataFrame(classification_report(
            y_valid, valid_predictions, labels=list(range(10)),
            output_dict=True, zero_division=0,
        )).T.to_csv(report_dir / "classification_report.csv", index_label="digit_or_summary")
        (report_dir / "metrics.json").write_text(json.dumps({
            "model": "LogisticRegression", "max_iter": 1000,
            "solver": "lbfgs", "random_state": 42,
            "training_rows": len(y_train), "validation_rows": len(y_valid),
            "validation_accuracy": float(accuracy),
            "validation_errors": int((y_valid != valid_predictions).sum()),
        }, indent=2) + "\n")

        # Evaluation is complete. Use all known answers for the final model.
        # Its training accuracy would not be an independent validation score.
        print("Training final model on all labelled images...", flush=True)
        final_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        final_model.fit(X, y)
        test_predictions = final_model.predict(X_test)

    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(X_test) + 1),
        "Label": test_predictions,
    })
    output_path = PROJECT_DIR / "submissions" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
