import pandas as pd

def load_data(path="weight-height.csv"):
    """Load CSV (Height in inches, Weight in pounds) and convert to cm / kg."""
    df = pd.read_csv(path)
    df["Height"] = df["Height"] * 2.54      # inches -> cm
    df["Weight"] = df["Weight"] / 2.2046    # pounds -> kg
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    return df.dropna()

def bmi(height_cm, weight_kg):
    return weight_kg / (height_cm / 100) ** 2

def bmi_category(b):
    return ("Underweight" if b < 18.5 else "Healthy" if b < 25
            else "Overweight" if b < 30 else "Obese")
