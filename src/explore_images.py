"""Step 1: reconstruct labelled training images and inspect handwriting variation.

Run from the project folder: python3 src/explore_images.py
Only train.csv is used. Original pixels and labels are never changed.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'reports' / 'step1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / 'data' / 'train.csv')
    columns = [f'pixel{i}' for i in range(784)]
    assert frame.columns.tolist() == ['label'] + columns, 'Unexpected training schema'
    values = frame[columns].to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 255)).all()
    assert (values == np.floor(values)).all()
    assert frame.label.isin(range(10)).all()
    images = values.astype(np.uint8).reshape(-1, 28, 28)
    assert np.array_equal(images.reshape(-1, 784), values)
    rng = np.random.default_rng(42)
    examples = []
    canvas = Image.new('RGB', (1000, 1480), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), 'STEP 1: HANDWRITING VARIATIONS', fill='black')
    draw.text((20, 40), 'Eight random training examples per digit. Fixed seed: 42. Row = 1-based CSV data row.', fill='black')
    for label in range(10):
        indices = rng.choice(np.flatnonzero(frame.label.to_numpy() == label), size=30, replace=False)
        for column, index in enumerate(indices):
            image = Image.fromarray(images[index])
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            # Mean intensity includes black background. It is not a confidence score.
            examples.append({'digit': label, 'row': int(index)+1, 'nonzero': int((images[index] > 0).sum()), 'mean': round(float(images[index].mean()), 2), 'png': base64.b64encode(buf.getvalue()).decode()})
            if column < 8:
                x, y = 100 + column * 110, 85 + label * 135
                canvas.paste(image.resize((84, 84), Image.Resampling.NEAREST).convert('RGB'), (x, y))
                draw.text((x, y+90), f'row {index+1}', fill='black')
        draw.text((20, 115+label*135), f'Digit {label}', fill='black')
    draw.text((20, 1450), 'Source: kaggle-digits/data/train.csv. Labels are supplied answers, not model predictions.', fill='black')
    canvas.save(args.output / 'digit_gallery.png')
    body = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Step 1: Explore the digit images</title>
<style>body{max-width:1080px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui;background:#fafaf8;color:#202b33}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:20px}figure{margin:0;padding:12px;background:white;border:1px solid #ddd}figure img{width:112px;height:112px;image-rendering:pixelated;display:block}figcaption{font-size:13px}select,button{font:inherit;padding:6px 10px;margin-right:10px}.controls{margin:20px 0}li{margin-bottom:8px}.note{border-left:3px solid #647c86;padding-left:16px}#count{font-size:14px}</style>
<h1>Step 1: See what the model will see</h1>
<p>We want to recognise a handwritten digit from 0 to 9. Each image below is reconstructed directly from 784 CSV pixel values arranged in 28 rows and 28 columns. Black = 0, white = 255, and grey values preserve the softer edges of each stroke.</p>
<p class="note">Source: train.csv from the supplied ZIP. This explorer contains 30 randomly selected training images per digit (300 total, seed 42). The labels are known training answers. No test images are labelled and no model has been trained.</p>
<h2>Explore the handwriting</h2><p>Compare loop shapes, slant, stroke thickness, and where the digit sits in its frame. Use the filter to focus on one digit. Row numbers let us trace each image to its original data.</p>
<div class="controls"><label>Digit <select id="digit"><option value="all">All digits</option></select></label><label>Order <select id="order"><option value="sample">Random sample order</option><option value="faint">Lowest mean intensity first</option><option value="bright">Highest mean intensity first</option></select></label></div>
<p id="count"></p><div id="gallery" class="grid"></div>
<h2>Questions for our first inspection</h2><ol><li>How much can the same digit vary between writers? Compare several examples with the same label.</li><li>Which shapes look similar across different digits? Compare 1 and 7, 3 and 8, or 4 and 9. These are candidate confusions to measure later, not proven model errors.</li><li>Are thin or faint strokes still visible? Mean intensity includes the black background, so a low value can mean a small digit or thin strokes as well as faint writing.</li><li>Are digits off-centre or slanted? Keep the original images while we establish a baseline. Later we can test whether training-only shifts or rotations improve validation accuracy.</li></ol>
<h2>What we do with this information</h2><p>Keep greyscale information and the original labels. Convert pixel values to float32 and divide by 255 when preparing model inputs. Reserve a stratified validation set before training. After fitting a baseline, use a confusion matrix and its mistaken predictions to discover which digits actually cause problems.</p>
<p>Looking unusual does not make an image invalid. This step makes no label corrections, removes no images, and claims no model accuracy.</p>
<script>const data=__DATA__;const digit=document.getElementById('digit');for(let i=0;i<10;i++){const opt=document.createElement('option');opt.value=i;opt.textContent=i;digit.append(opt)}function render(){let rows=data.filter(x=>digit.value==='all'||x.digit===Number(digit.value));const order=document.getElementById('order').value;if(order==='faint')rows.sort((a,b)=>a.mean-b.mean);if(order==='bright')rows.sort((a,b)=>b.mean-a.mean);document.getElementById('count').textContent=`Showing ${rows.length} labelled training images. Nonzero pixels and mean intensity describe the original 28 × 28 image.`;document.getElementById('gallery').innerHTML=rows.map(x=>`<figure><img alt="Training digit ${x.digit}, source row ${x.row}" src="data:image/png;base64,${x.png}"><figcaption><strong>Label ${x.digit}</strong> · row ${x.row}<br>${x.nonzero} nonzero pixels<br>Mean intensity ${x.mean} / 255</figcaption></figure>`).join('')}digit.onchange=render;document.getElementById('order').onchange=render;render();</script></html>'''
    (args.output / 'image_explorer.html').write_text(body.replace('__DATA__', json.dumps(examples)))
    # Numeric traceability for the selected samples, without embedding image bytes.
    pd.DataFrame([{k:v for k,v in e.items() if k != 'png'} for e in examples]).to_csv(args.output / 'sample_index.csv', index=False)
    print(f'Checked reshape round trip for {len(images):,} images. Exported 300 examples, gallery and explorer to {args.output}')


if __name__ == '__main__':
    main()
