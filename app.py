# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/regression_model.pkl")

# Title and description
st.title("🩺 MedInsurePredictor")
st.markdown("""
Predict your estimated **medical insurance charges** based on personal and health factors.
""")

# Input form on the right-hand side (sidebar)
st.sidebar.header("Enter Your Details:")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Do you smoke?", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# When button is clicked
if st.sidebar.button("Predict Insurance Charges"):
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Insurance Charges: ₹{prediction:,.2f}")
