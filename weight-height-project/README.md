# Height & Weight Toolkit

Two scikit-learn models plus a BMI helper, with a Python CLI, a Streamlit app and a standalone web app.

| Task | Model |
|---|---|
| Predict height from weight | Linear regression |
| Predict gender from height + weight | Logistic regression |
| BMI, category, healthy weight range | Formula |

## Structure
```
generate_demo_data.py  create a synthetic weight-height.csv (skip if you have the real one)
common.py              loading, unit conversion (in -> cm, lb -> kg), BMI
train.py               trains and saves both models into models/
predict.py             command-line predictions
app.py                 Streamlit app
web/index.html         standalone browser app (open directly; can train on your CSV)
```

## Run
```bash
pip install -r requirements.txt
python generate_demo_data.py        # or place your weight-height.csv here
python train.py
python predict.py --height 170 --weight 70
streamlit run app.py
```

## Fixes vs. the original notebook
- Saves the trained **model** (the notebook pickled the DataFrame).
- Converts units **before** building X/y.
- Uses a train/test split for the classifier, so accuracy is not measured on training data.
- One unit system (cm / kg) across all models.

Dataset format: columns `Gender, Height (in), Weight (lb)`.
