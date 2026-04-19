import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Prediction App")

# Load data directly from your repo
@st.cache_data
def load_data():
    df = pd.read_csv('Car_Price_Prediction.csv')
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("Vehicle Specifications")
make = st.sidebar.selectbox("Manufacturer", sorted(df['Make'].unique()))
model_list = sorted(df[df['Make'] == make]['Model'].unique())
car_model = st.sidebar.selectbox("Model", model_list)
year = st.sidebar.slider("Year", int(df['Year'].min()), 2026, 2020)
engine = st.sidebar.slider("Engine Size (L)", 1.0, 6.0, 2.0)
mileage = st.sidebar.number_input("Mileage", 0, 1000000, 50000)
fuel = st.sidebar.selectbox("Fuel Type", df['Fuel Type'].unique())
trans = st.sidebar.selectbox("Transmission", df['Transmission'].unique())

if st.button("Predict Price"):
    # Encoding
    df_encoded = df.copy()
    le = LabelEncoder()
    cat_cols = ['Make', 'Model', 'Fuel Type', 'Transmission']
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Quick Training
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Process Input
    input_data = pd.DataFrame([[make, car_model, year, engine, mileage, fuel, trans]], columns=X.columns)
    for col in cat_cols:
        le.fit(df[col])
        input_data[col] = le.transform(input_data[col])
        
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")
