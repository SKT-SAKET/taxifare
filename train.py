import os
from pyexpat import model
import random
import warnings


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# Fix randomness
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import kagglehub

path = kagglehub.dataset_download(
    "madalagopichand/nyc-green-yellow-taxi-trip-records",
    force_download=False
)

print(path)

file_path = r"C:\Users\saket\.cache\kagglehub\datasets\madalagopichand\nyc-green-yellow-taxi-trip-records\versions\1\yellow\2023\yellow-2023-01.parquet"

df = pd.read_parquet(file_path)

print(df.shape)
print(df.head())

print("\nColumns:")
print(df.columns.tolist())


# Explore dataset structure
for root, dirs, files in os.walk(path):
    print("\nFolder:", root)
    print("Files:", files[:20])


import os

all_files = []

for root, dirs, files in os.walk(path):
    for file in files:
        all_files.append(os.path.join(root, file))

print("Total Files:", len(all_files))

file_path = r"C:\Users\saket\.cache\kagglehub\datasets\madalagopichand\nyc-green-yellow-taxi-trip-records\versions\1\yellow\2023\yellow-2023-01.parquet"

df = pd.read_parquet(file_path)

print(df.shape)

print(df.head())

print(df.columns.tolist())

# Dataset Summary

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print(df[['fare_amount']].describe())


#Datetime Feature Engineering

df['tpep_pickup_datetime'] = pd.to_datetime(
    df['tpep_pickup_datetime']
)

df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday

print(df.head())

print(df.columns.tolist())

#Data Cleaning

# Remove negative fares

df = df[df["fare_amount"] > 0]

# Remove negative distance

df = df[df["trip_distance"] > 0]

# Fill missing values

df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)

df["RatecodeID"] = df["RatecodeID"].fillna(
    df["RatecodeID"].mode()[0]
)

df["congestion_surcharge"] = df["congestion_surcharge"].fillna(0)

df["airport_fee"] = df["airport_fee"].fillna(0)

df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N")

print(df.shape)


##Trip Duration Feature

df["trip_duration_minutes"] = (
    (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    / 60
)

df = df[df["trip_duration_minutes"] > 0]

print(df["trip_duration_minutes"].describe())

print("Sampling Data...")

df = df.sample(
    n=50000,
    random_state=42
)

print(df.shape)

##Encode Categorical Feature
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["store_and_fwd_flag"] = le.fit_transform(
    df["store_and_fwd_flag"]
)
y = df["fare_amount"]

##Select Features
X = df[
    [
        "VendorID",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "extra",
        "mta_tax",
        "tolls_amount",
        "congestion_surcharge",
        "airport_fee",
        "pickup_hour",
        "pickup_day",
        "pickup_month",
        "pickup_weekday",
        "trip_duration_minutes"
    ]
]


##Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

##Strong Baseline Model

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

model.fit(X_train, y_train)

#Prediction
preds = model.predict(X_test)

##Evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, preds)

rmse = mean_squared_error(
    y_test,
    preds
) ** 0.5

r2 = r2_score(
    y_test,
    preds
)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

#Save Model
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/taxi_fare_model.pkl"
)

print("Model Saved")



import joblib

joblib.dump(
    list(X.columns),
    "models/features.pkl"
)

print("Features Saved")

import joblib
import pandas as pd

model = joblib.load("models/taxi_fare_model.pkl")

sample = pd.DataFrame([{
    "VendorID": 2,
    "passenger_count": 1,
    "trip_distance": 5.2,
    "RatecodeID": 1,
    "store_and_fwd_flag": 0,
    "PULocationID": 132,
    "DOLocationID": 236,
    "payment_type": 1,
    "extra": 1.0,
    "mta_tax": 0.5,
    "tolls_amount": 0,
    "congestion_surcharge": 2.5,
    "airport_fee": 0,
    "pickup_hour": 14,
    "pickup_day": 15,
    "pickup_month": 1,
    "pickup_weekday": 0,
    "trip_duration_minutes": 18
}])

prediction = model.predict(sample)

print("Predicted Fare: $", round(prediction[0], 2))

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

from sklearn.metrics import r2_score

print("Train R2:", r2_score(y_train, train_pred))
print("Test R2 :", r2_score(y_test, test_pred))



import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/taxi_fare_model.pkl")

st.title("🚖 NYC Taxi Fare Predictor")

st.write("Enter trip details below")

trip_distance = st.number_input(
    "Trip Distance (miles)",
    min_value=0.1,
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
    value=15
)

if st.button("Predict Fare"):

    data = pd.DataFrame([{
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
        "tolls_amount": 0,
        "congestion_surcharge": 2.5,
        "airport_fee": 0,
        "pickup_hour": 12,
        "pickup_day": 15,
        "pickup_month": 1,
        "pickup_weekday": 0,
        "trip_duration_minutes": trip_duration
    }])

    prediction = model.predict(data)

    st.success(
        f"Predicted Fare: ${prediction[0]:.2f}"
    )
    