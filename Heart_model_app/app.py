import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("heart_model.joblib")

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details to predict heart disease")

# ---------------- USER VISIBLE INPUTS ---------------- #

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=45
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    help="0: Typical | 1: Atypical | 2: Non-anginal | 3: Asymptomatic"
)

chol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=220,
    value=150
)

# ---------------- AUTO-FILLED (HIDDEN) ---------------- #
# These are not shown in the GUI

exang = 1      # Exercise induced angina: No
oldpeak = 3.5   # Normal ST depression
slope = 2      # Flat ST slope

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict"):
    input_data = np.array([[
        age,
        sex,
        cp,
        chol,
        thalach,
        exang,
        oldpeak,
        slope
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
