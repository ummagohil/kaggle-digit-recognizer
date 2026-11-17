
This is a learning companion to `src/train.py`. It explains the script that produced the submission. 

## The beginning...

We start with examples that have answers, teach a model to associate pixels with digits, test it on answers withheld from training, and predict the digits for Kaggle's unlabelled images.

`train.csv → pixels and labels → training/validation split → learn → evaluate`

`all labelled images → fresh model → test.csv predictions → submission.csv`

The CSVs already contain flattened images. We do not flatten them again. A pixel at position 350 always refers to the same location in the 28 × 28 grid. The logistic regression model uses that position as a separate input feature.

## Tools Used

- `Path` builds filesystem paths 
- `json` saves evaluation settings and results in a structured text file
- `warnings` and `ConvergenceWarning` let us stop if the optimiser fails to converge.
- `numpy`, called `np`, handles numeric arrays
- `pandas`, called `pd`, reads and writes tables
- `LogisticRegression` is our first digit classifier
- `train_test_split` separates learning examples from validation examples
- `accuracy_score`, `classification_report`, and `confusion_matrix` evaluate predictions
- `threadpool_limits` limits the number of CPU threads used by underlying numerical libraries. It controls resource use, not the number of digits or training examples.

## Name the pixel columns

```python
PROJECT_DIR = Path(__file__).resolve().parents[1]
PIXEL_COLUMNS = [f"pixel{i}" for i in range(784)]
```

`__file__` is the location of `train.py`. `resolve()` gives its absolute path. For this file, `parents[0]` is `src` and `parents[1]` is `kaggle-digits`. This makes data paths work even when the script is launched from another directory.

`range(784)` generates integers 0 through 783. The list comprehension turns them into `pixel0`, `pixel1`, and so on. Keeping this order matters: rearranging the pixels would scramble the images.

## Load one dataset with `load_pixels`

```python
def load_pixels(path: Path, *, labelled: bool):
    table = pd.read_csv(path)
```

We call the `load_pixels` function for both training and test files. `path: Path` and `labelled: bool` are type hints. The `*` means we must name the second argument, for example `labelled=True`.

`pd.read_csv` loads a CSV into a DataFrame, a table with named columns. We expect the training table to have `label` followed by 784 pixel columns. The test table should have only the pixel columns.

The checks reject unexpected column names/order, empty files, missing or infinite pixels, values outside 0–255, and fractional pixel values. `raise ValueError(...)` stops execution with an explanation. We check before converting to smaller types so invalid values cannot be hidden by rounding.

`np.isfinite(pixels).all()` asks whether every pixel is finite. `.any()` asks whether at least one pixel breaks a rule. `np.floor` rounds down; comparing each value with its floor detects fractions.
```python
X = pixels.astype(np.float32) / 255.0
```
This converts pixels to 32-bit floating-point numbers and scales them. Black stays 0, white becomes 1. We preserve grey edges and stroke detail. The temporary `float64` array is used for validation and the model inputs are `float32`.
```python
y = None
if labelled:
    # Validate that every answer is an integer digit from 0 to 9.
    y = table["label"].to_numpy(dtype=np.int64)
return X, y
```
The actual script checks labels before converting them. `X` is the pixel input and `y` is the correct answer for each row. `None` means no labels are available. The labels are deliberately excluded from `X`: otherwise we would be giving the model the answer as an input.

## Load training and test inputs in `main`

```python
X, y = load_pixels(PROJECT_DIR / "data" / "train.csv", labelled=True)
X_test, _ = load_pixels(PROJECT_DIR / "data" / "test.csv", labelled=False)
```

The `/` operator joins path components. The returned values are unpacked into variables. `_` marks a returned value we do not need: the test file has no labels.

`X[0]` is the first image's flat pixel vector. `y[0]` is its answer. `X[0].reshape(28, 28)` would restore the grid for display but this classifier takes the flat vector.

## Reserve images for an honest check
```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

This keeps each image paired with its correct answer while splitting the labelled examples:

- 33,600 training images and answers are used to learn
- 8,400 validation images and answers are used to evaluate

`test_size=0.2` reserves 20%

`random_state=42` repeats the same split when data, ordering, and relevant software remain the same. The number 42 has no special learning power. `stratify=y` maintains approximately the same proportions of each digit in both partitions.

## Fit the baseline classifier
```python
warnings.filterwarnings("error", category=ConvergenceWarning)
with threadpool_limits(limits=2):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
```

Creating `model` sets up an untrained estimator. `fit` is where learning happens. It adjusts weights to make the correct training labels more probable, with regularisation discouraging excessively large weights.

Despite its name, logistic regression performs classification here. For each image it computes ten scores, one for each possible digit. Conceptually, for digit `k`:

```text
score[k] = bias[k] + sum(weight[k, j] * pixel[j] for j in 0…783)
probability[k] = exp(score[k]) / sum(exp(score[d]) for d in 0…9)
```

These probabilities sum to 1. The predicted digit is the class with the highest probability. A high probability is the model's estimate, not a guarantee of correctness. The script currently saves class predictions, not probabilities.

This model learns which pixel locations support each digit. It does not explicitly learn local curves and edges through convolutional filters; that would be a different model, such as a CNN.

`lbfgs` is the numerical optimiser. `max_iter=1000` is an iteration cap, not a number of images and not a promise of 1,000 complete passes through the data. If fitting does not converge, our warning policy stops the script before it writes a new submission. An older submission file, if present, would remain on disk.

## Predict the validation labels
```python
valid_predictions = model.predict(X_valid)
```
The fitted model receives only the withheld pixels. It returns an array of 8,400 predicted digits in the same order. We have not supplied `y_valid` to `fit` or `predict`. We only use those answers to score the predictions.

## 8. Measure accuracy and understand mistakes

```python
accuracy = accuracy_score(y_valid, valid_predictions)
matrix = confusion_matrix(y_valid, valid_predictions, labels=range(10))
```

Accuracy is the number of correct predictions divided by the number evaluated. Our saved run correctly classified **7,670 out of 8,400 images**, making **730 errors**, for **91.31% validation accuracy** when rounded to two decimal places.

This is a score for the model trained on 33,600 images on this particular validation split. It is not a Kaggle score and does not directly measure the final model trained on all 42,000 images.

In the confusion matrix, rows are true digits and columns are predicted digits. The diagonal contains correct predictions.

`classification_report` adds precision, recall, and F1 for each digit:

- Precision: among images predicted as that digit, how many were correct?
- Recall: among images truly showing that digit, how many did we recognise?
- F1: a combined measure of precision and recall.
- Support: how many validation examples had that true label.

## Save the evaluation records

The script creates the directory of `reports/baseline/`. 

- `confusion_matrix.csv` stores the 10 × 10 matrix with explicit true and predicted digit labels. Its DataFrame index is intentionally saved because it identifies the true digit.
- `classification_report.csv` stores per-digit and aggregate evaluation metrics. `.T` transposes the report so classes and summary names are rows.
- `metrics.json` stores the model name, selected settings, split seed, partition sizes, validation accuracy, and error count. `json.dumps(..., indent=2)` creates readable JSON. Explicit `float` and `int` conversions make NumPy-derived values serialisable.

These reports describe the evaluation model. They do not contain ground-truth test labels.

## Train a fresh model on all known answers

```python
final_model = LogisticRegression(max_iter=1000, solver="lbfgs")
final_model.fit(X, y)
test_predictions = final_model.predict(X_test)
```

Once we have evaluated the approach, we fit a fresh estimator using all 42,000 labelled images. It starts a new fit, this is not an incremental continuation of the earlier model.

The validation examples are now included in the final training data. We cannot reuse them to claim an independent validation score for this final model. A new independent score comes from Kaggle's hidden test answers.

`predict(X_test)` returns one digit for each of the 28,000 test rows. The script never reads the sample submission's placeholder labels to learn or evaluate.

## Build the exact submission table

```python
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(X_test) + 1),
    "Label": test_predictions,
})
```

`np.arange` excludes its upper endpoint, so this generates IDs 1 through 28,000. Each ID refers to the corresponding original test row. IDs are created after prediction; they were never model features.

## What we have learned so far

The CSV format is storage for images. The pixels are our evidence, the labels provide supervision, fitting learns weights, validation checks unseen examples, and prediction turns unlabelled images into digit guesses. A correctly formatted submission is the final output of those steps.

## Complete script used for the submission

The following is an exact snapshot of `src/train.py` when this guide was created.

```python
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
```
