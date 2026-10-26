"""Train and save both models.
  1) Linear regression : Weight (kg) -> Height (cm)
  2) Logistic regression: Height (cm) + Weight (kg) -> Gender (0 = Male, 1 = Female)
Usage: python train.py [path/to/weight-height.csv]"""
import sys, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from common import load_data

df = load_data(sys.argv[1] if len(sys.argv) > 1 else "weight-height.csv")
print(f"Data: {df.shape[0]} rows, missing values: {int(df.isna().sum().sum())}")

# ---- 1) Regression ----
X, y = df[["Weight"]], df["Height"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(Xtr, ytr)
pred = reg.predict(Xte)
print(f"\n[Regression] coef={reg.coef_[0]:.3f} intercept={reg.intercept_:.2f} "
      f"R2={r2_score(yte, pred):.3f} RMSE={mean_squared_error(yte, pred) ** 0.5:.2f} cm")

# ---- 2) Classification ----
X, y = df[["Height", "Weight"]], df["Gender"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"\n[Classification] test accuracy={accuracy_score(yte, pred):.3f}")
print(classification_report(yte, pred, target_names=["Male", "Female"]))

# ---- Save the MODELS (not the DataFrame) ----
pickle.dump(reg, open("models/weight_height_reg.pkl", "wb"))
pickle.dump(clf, open("models/gender_class.pkl", "wb"))
print("Saved models/weight_height_reg.pkl and models/gender_class.pkl")
