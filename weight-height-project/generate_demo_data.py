"""Create a synthetic weight-height.csv (Gender, Height in inches, Weight in pounds).
Use this only if you don't have the real dataset."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n, r = 5000, 0.85
rows = []
for gender, (mh, sh, mw, sw) in {"Male": (69.0, 2.86, 187.0, 19.8),
                                 "Female": (63.7, 2.70, 135.9, 19.2)}.items():
    zw = rng.standard_normal(n)
    zh = r * zw + np.sqrt(1 - r**2) * rng.standard_normal(n)
    rows.append(pd.DataFrame({"Gender": gender, "Height": mh + sh * zh, "Weight": mw + sw * zw}))
pd.concat(rows).sample(frac=1, random_state=42).to_csv("weight-height.csv", index=False)
print("Wrote weight-height.csv")
