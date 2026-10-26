import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Page title
st.title("🌱Crop Recommendation System")
st.write("Enter the required information to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N) content in soil", min_value=0)
P = st.number_input("Phosphorus (P) content in soil", min_value=0)
K = st.number_input("Potassium (K) content in soil", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH level of the soil")
rainfall = st.number_input("Rainfall (mm)")

# Predict button
if st.button("Predict"):
    # Make prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]

    # Display the result
    st.success(f"🌾Recommended Crop: **{prediction}**")
