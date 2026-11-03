import streamlit as st
import pandas as pd
import joblib

# Load Trained Model
model = joblib.load("models/taxi_fare_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered"
)

# Header
st.title("🚕 NYC Taxi Fare Prediction System")
st.markdown("### Predict taxi fare using Machine Learning")

# Sidebar
st.sidebar.header("Trip Information")

# User Inputs
trip_distance = st.number_input(
    "Trip Distance (Miles)",
    min_value=0.1,
    max_value=100.0,
    value=5.0
)

passenger_count = st.number_input(
    "Passenger Count",
    min_value=1,
    max_value=6,
    value=1
)

trip_duration = st.number_input(
    "Trip Duration (Minutes)",
    min_value=1,
    max_value=500,
    value=15
)

pickup_hour = st.slider(
    "Pickup Hour",
    0,
    23,
    12
)

pickup_day = st.slider(
    "Pickup Day",
    1,
    31,
    15
)

pickup_month = st.slider(
    "Pickup Month",
    1,
    12,
    1
)

pickup_weekday = st.slider(
    "Pickup Weekday (0=Monday, 6=Sunday)",
    0,
    6,
    0
)

# Predict Button
if st.button("🚀 Predict Fare"):

    input_data = pd.DataFrame([{
        "VendorID": 2,
        "passenger_count": passenger_count,
        "trip_distance": trip_distance,
        "RatecodeID": 1,
        "store_and_fwd_flag": 0,
        "PULocationID": 132,
        "DOLocationID": 236,
        "payment_type": 1,
        "extra": 1.0,
        "mta_tax": 0.5,
        "tolls_amount": 0.0,
        "congestion_surcharge": 2.5,
        "airport_fee": 0.0,
        "pickup_hour": pickup_hour,
        "pickup_day": pickup_day,
        "pickup_month": pickup_month,
        "pickup_weekday": pickup_weekday,
        "trip_duration_minutes": trip_duration
    }])

    prediction = model.predict(input_data)

    st.success(
        f"💰 Estimated Taxi Fare: ${prediction[0]:.2f}"
    )

    st.balloons()