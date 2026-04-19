import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.title("Car Price Prediction App")

# Load data - assumes Car_Price_Prediction.csv is in the same GitHub folder
@st.cache_data
def load_data():
    df = pd.read_csv('Car_Price_Prediction.csv')
    # Basic cleaning matching your notebook
    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

df = load_data()

# Sidebar for user inputs
st.sidebar.header("User Input Features")

def user_input_features():
    make = st.sidebar.selectbox("Make", df['Make'].unique())
    model = st.sidebar.selectbox("Model", df['Model'].unique())
    year = st.sidebar.slider("Year", int(df['Year'].min()), int(df['Year'].max()), 2015)
    engine_size = st.sidebar.number_input("Engine Size", min_value=1.0, max_value=5.0, value=2.0)
    mileage = st.sidebar.number_input("Mileage", value=50000)
    fuel_type = st.sidebar.selectbox("Fuel Type", df['Fuel Type'].unique())
    transmission = st.sidebar.selectbox("Transmission", df['Transmission'].unique())
    
    data = {
        'Make': make,
        'Model': model,
        'Year': year,
        'Engine Size': engine_size,
        'Mileage': mileage,
        'Fuel Type': fuel_type,
        'Transmission': transmission
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Encoding categorical data
label_encoders = {}
for col in ['Make', 'Model', 'Fuel Type', 'Transmission']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    # Map input data using the same encoder
    input_df[col] = le.transform(input_df[col])

# Simple Model Training (for demo, in production you'd load a saved model)
X = df.drop('Price', axis=1)
y = df['Price']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"The estimated car price is: ${prediction[0]:,.2f}")