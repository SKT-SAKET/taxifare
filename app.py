import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
    border: none;
}

.stButton > button:hover {
    background-color: #E63939;
    color: white;
}

.prediction-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #E8F5E9;
    border: 2px solid #4CAF50;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/taxi_fare_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# -----------------------------
# HEADER
# -----------------------------
st.title("🚕 NYC Taxi Fare Prediction System")
st.markdown(
    "Predict taxi fares instantly using a Machine Learning model."
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("ℹ️ About")

st.sidebar.info(
    """
    This application predicts NYC taxi fares
    using Machine Learning.

    Tech Stack:
    - Python
    - Scikit-Learn
    - Streamlit
    """
)

st.sidebar.success("Model Status: Loaded ✅")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Enter Trip Details")

col1, col2 = st.columns(2)

with col1:
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

with col2:
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
    "Pickup Weekday (0=Mon, 6=Sun)",
    0,
    6,
    0
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
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

    try:
        prediction = model.predict(input_data)

        st.markdown(
            f"""
            <div class="prediction-box">
                💰 Estimated Fare: ${prediction[0]:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "Built by Saket using Machine Learning, Streamlit and Scikit-Learn 🚀"
)
    
