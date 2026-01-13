import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model + training columns
model = joblib.load("vehicle_price_model_updated.joblib")
model_columns = joblib.load("model_columns.joblib")

st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗")
st.title("🚗 Vehicle Price Prediction App")

# ---------------------- USER INPUT FIELDS ----------------------
year = st.number_input("Year", 1990, 2025, 2020)
cylinders = st.number_input("Cylinders", 2, 16, 4)
mileage = st.number_input("Mileage", 0, 300000, 50000)
doors = st.number_input("Doors", 2, 6, 4)

# Dynamic dropdowns based on training columns
def extract_options(prefix):
    return sorted([col.replace(prefix + "_", "") 
                   for col in model_columns if col.startswith(prefix + "_")])

make = st.selectbox("Make", extract_options("make"))
model_name = st.selectbox("Model", extract_options("model"))
engine = st.selectbox("Engine Type", extract_options("engine"))
fuel = st.selectbox("Fuel Type", extract_options("fuel"))
transmission = st.selectbox("Transmission", extract_options("transmission"))
trim = st.selectbox("Trim", extract_options("trim"))
body = st.selectbox("Body Type", extract_options("body"))
drivetrain = st.selectbox("Drivetrain", extract_options("drivetrain"))


# ---------------------- CONSTRUCT FEATURE ROW ----------------------
# Create an empty input row with all model columns = 0
input_row = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

# Fill numeric fields
input_row["year"] = year
input_row["cylinders"] = cylinders
input_row["mileage"] = mileage
input_row["doors"] = doors

# One-hot encoded fields
def set_flag(prefix, value):
    col = f"{prefix}_{value}"
    if col in input_row.columns:
        input_row.at[0, col] = 1

set_flag("make", make)
set_flag("model", model_name)
set_flag("engine", engine)
set_flag("fuel", fuel)
set_flag("transmission", transmission)
set_flag("trim", trim)
set_flag("body", body)
set_flag("drivetrain", drivetrain)


# ---------------------- PREDICT ----------------------
if st.button("Predict Price"):
    y_log = model.predict(input_row)[0]
    price = np.expm1(y_log)  # reverse log1p
    st.success(f"Estimated Vehicle Price: ₹ {round(price, 2)}")
