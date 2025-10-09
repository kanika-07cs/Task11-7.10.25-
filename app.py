import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('house_rent.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
pt_BHK = data['pt_BHK']
scaler_BHK = data['scaler_BHK']
pt_Size = data['pt_Size']
scaler_Size = data['scaler_Size']
scaler_Bathroom = data['scaler_Bathroom']
scaler_y = data['scaler_y']
trained_columns = data['columns']

st.set_page_config(page_title="House Rent Predictor", layout="centered")
st.title("House Rent Prediction App")
st.write("Enter the property details below to estimate monthly rent:")

BHK = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
Size = st.number_input("Size (in Sq. Ft.)", min_value=100, max_value=10000, value=1000)
Bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
City = st.selectbox("City", ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
Furnishing_Status = st.selectbox("Furnishing Status", ['Furnished', 'Semi-Furnished', 'Unfurnished'])
Area_Type = st.selectbox("Area Type", ['Super Area', 'Carpet Area', 'Built Area'])

if st.button("Predict Rent"):
    BHK_s = scaler_BHK.transform(pt_BHK.transform([[BHK]]))[0][0]
    Size_s = scaler_Size.transform(pt_Size.transform([[Size]]))[0][0]
    Bathroom_s = scaler_Bathroom.transform(np.log1p([[Bathroom]]))[0][0]

    row = np.zeros(len(trained_columns))
    row[trained_columns.index('BHK_transformed')] = BHK_s
    row[trained_columns.index('Size_transformed')] = Size_s
    row[trained_columns.index('Bathroom_transformed')] = Bathroom_s

    city_col = f'City_{City}'
    if city_col in trained_columns:
        row[trained_columns.index(city_col)] = 1

    furn_col = f'Furnishing Status_{Furnishing_Status}'
    if furn_col in trained_columns:
        row[trained_columns.index(furn_col)] = 1

    area_col = f'Area Type_{Area_Type}'
    if area_col in trained_columns:
        row[trained_columns.index(area_col)] = 1

    pred_scaled = model.predict([row])
    pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
    pred_rent = np.expm1(pred_log)

    st.success(f"💰 Estimated Monthly Rent: ₹{pred_rent:,.2f}")
