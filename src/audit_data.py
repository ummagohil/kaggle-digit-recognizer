"""Audit original Kaggle CSVs and export readable tables without changing source data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'reports')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(root / 'data' / f'{name}.csv') for name in ['train', 'test', 'sample_submission']}
    pixels = [f'pixel{i}' for i in range(784)]
    train, test, sample = [frames[k] for k in ['train', 'test', 'sample_submission']]
    schemas = {'train': ['label'] + pixels, 'test': pixels, 'sample_submission': ['ImageId', 'Label']}
    checks, inventory, hashes = [], [], {}
    def check(dataset, name, value, expected):
        checks.append({'dataset': dataset, 'check': name, 'observed': str(value), 'expected': str(expected), 'passed': bool(value == expected)})
    for name, frame in frames.items():
        inventory.append({'dataset': name, 'rows': len(frame), 'columns': len(frame.columns), 'missing_cells': int(frame.isna().sum().sum())})
        check(name, 'Column names and order match schema', list(frame.columns) == schemas[name], True)
        check(name, 'Missing cells', int(frame.isna().sum().sum()), 0)
        numeric = frame.apply(pd.to_numeric, errors='coerce')
        check(name, 'Nonnumeric nonmissing cells', int((numeric.isna() & frame.notna()).sum().sum()), 0)
        vals = numeric.to_numpy(dtype=float)
        check(name, 'Nonfinite values', int((~np.isfinite(vals)).sum()), 0)
        check(name, 'Fractional values', int((np.isfinite(vals) & (vals != np.floor(vals))).sum()), 0)
        if name in ['train', 'test'] and set(pixels).issubset(frame.columns):
            x = numeric[pixels].to_numpy()
            check(name, 'Pixels outside 0–255', int(((x < 0) | (x > 255)).sum()), 0)
            check(name, 'Completely blank images', int((x == 0).all(axis=1).sum()), 0)
            # SHA-256 of exact pixel vectors gives stable grouping IDs, without using labels.
            hashes[name] = [hashlib.sha256(row.astype('<f8').tobytes()).hexdigest() for row in x]
            frame_summary = pd.DataFrame({'source_row': np.arange(1, len(frame)+1), 'image_group': hashes[name], 'nonzero_pixels': (x > 0).sum(axis=1), 'mean_intensity': x.mean(axis=1), 'max_intensity': x.max(axis=1)})
            if name == 'train':
                frame_summary.insert(1, 'label', train['label'])
            frame_summary.to_csv(out / f'{name}_image_summary.csv', index=False)
            check(name, 'Duplicate images beyond first occurrence', int(pd.Series(hashes[name]).duplicated().sum()), 0)
            check(name, 'Minimum pixel intensity', int(np.nanmin(x)), 0)
            check(name, 'Maximum pixel intensity', int(np.nanmax(x)), 255)
            frame.head(50).to_csv(out / f'{name}_preview.csv', index=False)
    check('train', 'Labels outside integer digits 0–9', int((~train.label.isin(range(10))).sum()), 0)
    counts = train.label.value_counts().reindex(range(10), fill_value=0).rename_axis('digit').reset_index(name='images')
    counts['percent_of_training'] = (100 * counts.images / len(train)).round(2)
    groups = pd.DataFrame({'group': hashes['train'], 'label': train.label})
    check('train', 'Identical image groups with conflicting labels', int((groups.groupby('group').label.nunique() > 1).sum()), 0)
    overlap = set(hashes['train']) & set(hashes['test'])
    check('train/test', 'Exact image groups present in both files', len(overlap), 0)
    check('sample_submission', 'Rows match test', len(sample), len(test))
    check('sample_submission', 'ImageId is 1 through test row count in order', sample.ImageId.tolist() == list(range(1, len(test)+1)), True)
    check('sample_submission', 'Labels outside digits 0–9', int((~sample.Label.isin(range(10))).sum()), 0)
    audit, datasets = pd.DataFrame(checks), pd.DataFrame(inventory)
    audit.to_csv(out / 'quality_checks.csv', index=False)
    datasets.to_csv(out / 'datasets.csv', index=False)
    counts.to_csv(out / 'label_counts.csv', index=False)
    dictionary = pd.DataFrame([
        ['label', 'train.csv', 'Correct digit, integer 0–9. Prediction target.'],
        ['pixel0 … pixel783', 'train.csv, test.csv', '784 intensity values, 0 (black) to 255 (white). Reshape into 28 × 28 in row order.'],
        ['ImageId', 'sample_submission.csv', '1-based test row number. Preserve test order. Never a model input.'],
        ['Label', 'sample_submission.csv', 'Example prediction column. Placeholder values are not known test answers.'],
        ['source_row', 'generated image summaries', '1-based source data row, excluding header. Traceability only.'],
        ['image_group', 'generated image summaries', 'SHA-256 of pixel vector. Group identical images together when splitting validation.'],
    ], columns=['field', 'location', 'meaning'])
    dictionary.to_csv(out / 'data_dictionary.csv', index=False)
    questions = '''<ol><li>Can we predict the correct digit (0–9) from the 28 × 28 pixel image? Measure accuracy on held-out labelled training rows.</li><li>Does every digit have enough examples? Compare counts and per-digit recall, not just overall accuracy.</li><li>Which digits are confused with each other? Inspect a confusion matrix and misclassified validation images after training.</li><li>Do faint strokes, slant, or off-centre writing cause errors? Compare errors by image properties before deciding on transformations.</li><li>Does a more complex model outperform a simple baseline on the same validation split?</li><li>Does augmentation help? Apply it only to the training partition and compare on untouched validation images.</li></ol>'''
    policy = '''<ul><li>Preserve original CSVs. These are already tabular data: one image per row, one pixel per feature column.</li><li>Zeros are valid black background. Do not impute them, drop faint images, binarise strokes, or guess replacement labels.</li><li>Fail validation on malformed schemas, missing/nonfinite values, fractional pixels, out-of-range pixels, or invalid labels. Investigate rather than silently repair.</li><li>Retain valid duplicate rows, but keep identical-image groups within one partition to avoid validation leakage. Keep all test rows and their order.</li><li>Use a reproducible, approximately 80/20 stratified split of the labelled training data, respecting duplicate-image groups. Keep the label and identifiers out of model inputs.</li><li>For models requiring normalised inputs, convert pixels to float32 and divide by 255. Fit any learned preprocessing on the training partition only. Keep the 28 × 28 spatial arrangement available.</li><li>Do not use sample submission labels as ground truth, or look up test labels from external MNIST copies. Local test accuracy cannot be measured because the answers are absent.</li></ul>'''
    table = lambda df: '<div class="scroll">' + df.to_html(index=False, border=0, escape=True) + '</div>'
    example_rows = []
    for digit in range(10):
        row = train[train.label == digit].iloc[0]
        a = row[pixels].to_numpy(dtype=int)
        rects = ''.join(f'<rect x="{i%28}" y="{i//28}" width="1" height="1" fill="rgb({v},{v},{v})"/>' for i,v in enumerate(a) if v)
        example_rows.append(f'<figure><svg role="img" aria-label="Training example labelled {digit}" viewBox="0 0 28 28"><rect width="28" height="28" fill="black"/>{rects}</svg><figcaption>Label {digit}</figcaption></figure>')
    preview = train.head(12).copy()
    preview.insert(0, 'source_row', range(1,13))
    compact = preview[['source_row', 'label', 'pixel0', 'pixel350', 'pixel351', 'pixel352', 'pixel353', 'pixel783']]
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Digit data review</title><style>body{font:16px system-ui;color:#202b33;background:#fafaf8;max-width:1160px;margin:40px auto;padding:0 24px;line-height:1.6}h1{font-size:30px}h2{margin-top:40px;font-size:22px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{text-align:left;padding:10px 14px;border-bottom:1px solid #ddd}th{background:#edf1f2;white-space:nowrap}.scroll{overflow:auto}li{margin:10px 0}.digits{display:flex;flex-wrap:wrap;gap:15px}figure{margin:0}svg{width:88px;height:88px;image-rendering:pixelated}figcaption{font-size:13px}p{max-width:900px}</style><h1>Digit Recognizer: data review</h1><p>Goal: predict the correct handwritten digit from its pixels. Source: the user-provided digit-recognizer.zip, extracted to kaggle-digits/data. Full source files inspected. No model trained and no rows removed.</p>'''
    html += '<h2>What we have</h2>' + table(datasets)
    html += '<h2>What the columns mean</h2>' + table(dictionary)
    html += '<h2>Training examples reconstructed from pixels</h2><p>First available training example for each label. These are training images, not test predictions.</p><div class="digits">' + ''.join(example_rows) + '</div>'
    html += '<h2>Training table preview</h2><p>First 12 training rows. Selected pixel columns shown to keep this readable; all 784 are retained in the CSVs. Centre pixels are included because edge pixels are often zero.</p>' + table(compact)
    html += '<h2>Training examples per digit</h2>' + table(counts)
    html += '<h2>Quality checks</h2><p>False flags require review. Duplicate or overlap flags are diagnostics, not automatic deletion instructions.</p>' + table(audit)
    html += '<h2>Cleaning and validation decisions</h2>' + policy
    html += '<h2>Questions to answer next</h2>' + questions
    html += '<h2>Next experiment</h2><p>Build a simple baseline on a fixed validation split. Report accuracy, per-digit recall and a confusion matrix. Then compare a convolutional neural network on the same split. No model accuracy is established by this audit.</p></html>'
    (out / 'data_review.html').write_text(html)
    print(datasets.to_string(index=False))
    print(counts.to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == '__main__':
    main()
