import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load Model and Scaler
# -------------------------------
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.write("Enter the house details below to predict the price.")

# -------------------------------
# User Inputs
# (⚠ Make sure these match your dataset columns exactly)
# -------------------------------

area = st.number_input("Area (in sqft)", min_value=0.0, step=10.0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
stories = st.number_input("Number of Stories", min_value=0, step=1)
parking = st.number_input("Parking Spaces", min_value=0, step=1)

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict Price"):
    
    # Create input array
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    
    # Scale input
    input_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_data)
    
    # Display Result
    st.success(f"🏷 Estimated House Price: ₹ {prediction[0]:,.2f}")
