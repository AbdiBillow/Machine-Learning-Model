import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("Machine Learning App with Streamlit")

# Upload Data Button
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

   
 if X_col and y_col:
    X = data[[X_col]]
    y = data[y_col]
 # Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model Button
if st.button("Train Model"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.success("Model Trained Successfully!")

# Predict Button
if st.button("Make Predictions"):
    y_pred = model.predict(X_test)
    st.write("Predictions:", y_pred[:5])

# Evaluate Model Button
if st.button("Evaluate Model"):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

          
                   
