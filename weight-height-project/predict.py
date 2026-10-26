"""Command-line predictions.   python predict.py --height 170 --weight 70"""
import argparse, pickle
import pandas as pd
from common import bmi, bmi_category

ap = argparse.ArgumentParser()
ap.add_argument("--height", type=float, required=True, help="cm")
ap.add_argument("--weight", type=float, required=True, help="kg")
a = ap.parse_args()

reg = pickle.load(open("models/weight_height_reg.pkl", "rb"))
clf = pickle.load(open("models/gender_class.pkl", "rb"))

b = bmi(a.height, a.weight)
lo, hi = 18.5 * (a.height / 100) ** 2, 24.9 * (a.height / 100) ** 2
p = clf.predict_proba(pd.DataFrame([[a.height, a.weight]], columns=["Height", "Weight"]))[0]
print(f"BMI: {b:.1f} ({bmi_category(b)})")
print(f"Healthy weight range: {lo:.0f}-{hi:.0f} kg")
print(f"Expected height at {a.weight:.0f} kg: {reg.predict(pd.DataFrame([[a.weight]], columns=['Weight']))[0]:.1f} cm")
print(f"Gender guess: {'Female' if p[1] >= .5 else 'Male'} (P female = {p[1]:.0%})")
