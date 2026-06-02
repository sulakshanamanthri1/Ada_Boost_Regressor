import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/adaboost_model.pkl")

# Page Config
st.set_page_config(
    page_title="Tip Prediction App",
    page_icon="💰",
    layout="centered"
)

# Load CSS
with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.title("💰 Restaurant Tip Prediction")
st.write("Predict the expected tip using AdaBoost Regressor")

st.markdown("---")

# User Inputs

total_bill = st.number_input(
    "Total Bill Amount",
    min_value=0.0,
    value=20.0
)

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

day = st.selectbox(
    "Day",
    ["Thur", "Fri", "Sat", "Sun"]
)

time = st.selectbox(
    "Time",
    ["Lunch", "Dinner"]
)

size = st.slider(
    "Party Size",
    min_value=1,
    max_value=10,
    value=2
)

st.markdown("---")

if st.button("Predict Tip"):

    input_data = pd.DataFrame({
        "total_bill": [total_bill],
        "sex": [sex],
        "smoker": [smoker],
        "day": [day],
        "time": [time],
        "size": [size]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Tip Amount: ${prediction:.2f}"
    )