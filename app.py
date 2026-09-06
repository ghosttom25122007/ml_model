"""
Student Score Predictor - Streamlit Web App
-----------------------------------------------
Predicts a student's exam score based on hours studied,
using a Linear Regression model trained on sample data.
"""

import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("score_model.joblib")

# --- Page setup ---
st.set_page_config(page_title="Student Score Predictor", page_icon="📊")
st.title("📊 Student Score Predictor")
st.write(
    "This app predicts a student's exam score based on hours studied, "
    "using a Linear Regression model I trained on sample study data."
)

st.subheader("Enter Study Hours")

# --- Input ---
hours = st.slider("Hours Studied", 0.0, 12.0, 5.0, step=0.5)

# --- Predict button ---
if st.button("Predict Score"):
    input_data = np.array([[hours]])
    predicted_score = model.predict(input_data)[0]

    # Clamp between 0 and 100 for realism (model could technically go outside this range)
    predicted_score = max(0, min(100, predicted_score))

    st.success(f"Predicted Score: **{predicted_score:.1f} / 100**")

    # Show the learned relationship for transparency
    slope = model.coef_[0]
    intercept = model.intercept_
    st.write(f"Model's learned formula: **Score = {slope:.2f} × Hours + {intercept:.2f}**")

st.divider()
st.caption("Built with Python, Scikit-learn, and Streamlit. Model: Linear Regression.")
