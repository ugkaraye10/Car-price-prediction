import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# App Setup
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")
st.markdown("Enter vehicle details below to get an estimated market price.")

# 1. Load Data
@st.cache_data
def load_data():
    # Make sure 'Car_Price_Prediction.csv' is uploaded to your GitHub repo
    df = pd.read_csv('Car_Price_Prediction.csv')
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset 'Car_Price_Prediction.csv' not found in repository.")
    st.stop()

# 2. Sidebar UI for User Input
st.sidebar.header("Car Specifications")
make = st.sidebar.selectbox("Manufacturer", sorted(df['Make'].unique()))
model_list = sorted(df[df['Make'] == make]['Model'].unique())
car_model = st.sidebar.selectbox("Model", model_list)
year = st.sidebar.slider("Year", int(df['Year'].min()), 2026, 2020)
engine = st.sidebar.number_input("Engine Size (L)", 1.0, 6.0, 2.0)
mileage = st.sidebar.number_input("Mileage", 0, 1000000, 50000)
fuel = st.sidebar.selectbox("Fuel Type", df['Fuel Type'].unique())
trans = st.sidebar.selectbox("Transmission", df['Transmission'].unique())

# 3. Model Training (Simplified for App launch)
@st.cache_resource
def train_prediction_model(data):
    df_encoded = data.copy()
    encoders = {}
    cat_features = ['Make', 'Model', 'Fuel Type', 'Transmission']
    
    for col in cat_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, encoders

model, encoders = train_prediction_model(df)

# 4. Prediction Button
if st.button("Estimate Price"):
    input_data = pd.DataFrame([[make, car_model, year, engine, mileage, fuel, trans]], 
                              columns=['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission'])
    
    # Apply encoding to input
    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])
        
    prediction = model.predict(input_data)
    st.success(f"Estimated Market Value: ${prediction[0]:,.2f}")
