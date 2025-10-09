import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('house_rent.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
pt = data['pt']
scaler = data['scaler']
trained_columns = data['columns']

st.set_page_config(page_title=" House Rent Predictor", layout="centered")
st.title("House Rent Prediction App")
st.write("Enter the property details below to estimate monthly rent:")

BHK = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
Size = st.number_input("Size (in Sq. Ft.)", min_value=100, max_value=10000, value=1000)
City = st.selectbox("City", ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
Furnishing_Status = st.selectbox("Furnishing Status", ['Furnished', 'Semi-Furnished', 'Unfurnished'])

if st.button("Predict Rent"):
    bhk_transformed = pt.transform(np.array([[BHK]]))
    bhk_transformed = scaler.transform(bhk_transformed)

    size_transformed = pt.transform(np.array([[Size]]))
    size_transformed = scaler.transform(size_transformed)

    new_data = pd.DataFrame({
        'BHK': [BHK],
        'Size_transformed': [size_transformed[0][0]],
        'City': [City],
        'Furnishing Status': [Furnishing_Status],
    })

    new_data_encoded = pd.get_dummies(new_data, columns=['City', 'Furnishing Status'], drop_first=True)
    for col in trained_columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0

    new_data_encoded = new_data_encoded[trained_columns]

    predicted_log_rent = model.predict(new_data_encoded)
    predicted_rent = np.expm1(predicted_log_rent)[0]

    st.success(f"💰 Estimated Monthly Rent: ₹{predicted_rent:,.2f}")
