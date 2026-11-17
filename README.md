# 🔢 Digits Recogniser 
A Python project for the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer), predicting handwritten digits from 0 to 9. 

### 📋 Aims of App

- User can run the script and get a submission that can be submitted
- User can read the code walk through file to understand the repo

### 👩‍💻 Technical Details

- Python
- NumPy and pandas for image data preparation
- scikit-learn for the logistic regression classifier
- Pillow for image previews

### 🖥️ Run the app

Run these commands in Terminal from the project folder:

```bash
cd ~/Documents/kaggle-digits
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the competition data from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data) and extract `train.csv` and `test.csv` into the `data/` folder if they are not already there.

Train the model and generate predictions:

```bash
python src/train.py
```

The script prints validation accuracy and a confusion matrix, then trains a final model on all labelled images. It saves the Kaggle submission to `submissions/submission.csv` and evaluation reports to `reports/baseline/`.

Upload `submissions/submission.csv` on the Kaggle competition submission page. The script does not upload it automatically. Each run retrains the models and replaces the generated submission file.

For later runs, activate the existing environment and run the script:

```bash
cd ~/Documents/kaggle-digits
source .venv/bin/activate
python src/train.py
```

### 💭 Future Improvements
None
