import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and column names
model = joblib.load("mobile_model.joblib")
scaler = joblib.load("mobile_scaler.joblib")
columns = joblib.load("mobile_columns.joblib")
template = joblib.load("template.joblib")

st.set_page_config(page_title="📱 Mobile Price Predictor", page_icon="📱")
st.title("📱 Mobile Price Range Prediction")

# ---------------------- PRICE RANGE DESCRIPTION ----------------------
st.markdown("""
### 💰 Price Range Categories  
These ranges correspond to the model's output classes:

| Predicted Class | Estimated Price Range |
|-----------------|-----------------------|
| **0** | ₹5,000 – ₹10,000 |
| **1** | ₹10,000 – ₹15,000 |
| **2** | ₹15,000 – ₹20,000 |
| **3** | ₹20,000 – ₹30,000 |

""")

# Mapping class → price range text
price_map = {
    0: "₹5,000 – ₹10,000",
    1: "₹10,000 – ₹15,000",
    2: "₹15,000 – ₹20,000",
    3: "₹20,000 – ₹30,000"
}

st.write("Enter only the essential features below:")

# ---------------------- USER INPUTS ----------------------
battery = st.number_input("Battery Power", 500, 2000, 1000)
ram = st.number_input("RAM (MB)", 256, 8000, 2000)
int_memory = st.number_input("Internal Memory (GB)", 2, 256, 64)
pc = st.number_input("Primary Camera (MP)", 0, 20, 10)
fc = st.number_input("Front Camera (MP)", 0, 20, 5)
talk_time = st.number_input("Talk Time (hrs)", 2, 20, 10)

# ---------------------- BUILD FULL FEATURE ROW ----------------------
input_row = template.copy()

input_row["battery_power"] = battery
input_row["ram"] = ram
input_row["int_memory"] = int_memory
input_row["pc"] = pc
input_row["fc"] = fc
input_row["talk_time"] = talk_time

# ---------------------- SCALE + PREDICT ----------------------
scaled = scaler.transform(input_row)

if st.button("Predict Price Range"):
    pred_class = model.predict(scaled)[0]
    price_range = price_map[pred_class]

    st.success(f"📌 **Predicted Class:** {pred_class}")
    st.success(f"💰 **Estimated Price Range:** {price_range}")
