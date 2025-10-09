import streamlit as st
import pickle
import numpy as np
import pandas as pd
with open('house_rent.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']

st.set_page_config(page_title="House Rent Predictor", page_icon="🏠", layout="centered")
st.title("House Rent Prediction App")
st.write("Enter property details below to predict the monthly rent:")

BHK = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
Size = st.number_input("Size (in Sq. Ft.)", min_value=100, max_value=10000, value=1000)
Bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Predict Rent"):
    Size_power = np.sqrt(Size)
    X_new = pd.DataFrame([[BHK, Size_power, Bathroom]], columns=['BHK', 'Size_power', 'Bathroom'])
    X_scaled = scaler.transform(X_new)

    predicted_log_rent = model.predict(X_scaled)
    predicted_rent = np.expm1(predicted_log_rent)[0]

    st.success(f"💰 Estimated Monthly Rent: ₹{predicted_rent:,.2f}")
