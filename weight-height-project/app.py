"""Streamlit app.   streamlit run app.py"""
import pickle
import pandas as pd
import streamlit as st
from common import bmi, bmi_category

st.set_page_config(page_title="Height & Weight Toolkit", layout="centered")
st.title("Height & Weight Toolkit")

reg = pickle.load(open("models/weight_height_reg.pkl", "rb"))
clf = pickle.load(open("models/gender_class.pkl", "rb"))

h = st.slider("Height (cm)", 120, 220, 170)
w = st.slider("Weight (kg)", 30, 150, 70)

b = bmi(h, w)
c1, c2 = st.columns(2)
c1.metric("BMI", f"{b:.1f}", bmi_category(b))
c2.metric("Healthy range", f"{18.5*(h/100)**2:.0f}–{24.9*(h/100)**2:.0f} kg")

ph = reg.predict(pd.DataFrame([[w]], columns=["Weight"]))[0]
st.metric("Expected height at this weight", f"{ph:.1f} cm", f"{h-ph:+.1f} cm vs. you")

p = clf.predict_proba(pd.DataFrame([[h, w]], columns=["Height", "Weight"]))[0]
st.subheader("Gender classification")
st.progress(float(p[1]), text=f"P(female) = {p[1]:.0%}   |   P(male) = {p[0]:.0%}")
st.caption("A statistical guess from body measurements only.")
