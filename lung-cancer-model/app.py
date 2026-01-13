import streamlit as st
import numpy as np
import joblib
import random
from datetime import datetime, timedelta

# Load trained model
model = joblib.load("lung_cancer_model.joblib")

st.set_page_config(page_title="Lung Cancer Survival Prediction", layout="centered")

st.title("🫁 Lung Cancer Survival Prediction")
st.write("Enter patient details")

# ---------------- USER INPUTS ---------------- #

age = st.number_input("Age", 1, 120, 55)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
family_history = st.selectbox("Family History of Cancer", ["Yes", "No"])
cancer_stage = st.selectbox(
    "Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"]
)

# ---------------- AUTO-FILLED (HIDDEN) ---------------- #

patient_id = random.randint(10000, 99999)
country = "India"
diagnosis_date = 0  # encoded placeholder
end_treatment_date = 0  # encoded placeholder

cholesterol_level = 220
hypertension = 0
asthma = 0
cirrhosis = 0
other_cancer = 0
treatment_type = 1  # Chemotherapy (encoded)

# ---------------- ENCODING ---------------- #

gender_val = 1 if gender == "Male" else 0
family_history_val = 1 if family_history == "Yes" else 0

smoking_map = {"Never": 0, "Former": 1, "Current": 2}
smoking_val = smoking_map[smoking_status]

stage_map = {
    "Stage I": 1,
    "Stage II": 2,
    "Stage III": 3,
    "Stage IV": 4
}
stage_val = stage_map[cancer_stage]

country_val = 1  # India (encoded)

# ---------------- MODEL INPUT (16 FEATURES – EXACT ORDER) ---------------- #

input_data = np.array([[
    patient_id,
    age,
    gender_val,
    country_val,
    diagnosis_date,
    stage_val,
    family_history_val,
    smoking_val,
    bmi,
    cholesterol_level,
    hypertension,
    asthma,
    cirrhosis,
    other_cancer,
    treatment_type,
    end_treatment_date
]])

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Patient Likely to Survive")
    else:
        st.error("⚠️ High Risk – Survival Unlikely")
