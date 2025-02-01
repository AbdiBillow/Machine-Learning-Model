import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the model globally
model = None

# Streamlit UI
st.title("Linear Regression Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Dropdowns for selecting columns
    region_column = st.selectbox("Select Region Column", df.columns)
    district_column = st.selectbox("Select District Column", df.columns)
    month_column = st.selectbox("Select Month Column", df.columns)
    price_column = st.selectbox("Select Price Column (Target Variable)", df.columns)
    commodity_columns = st.multiselect("Select Commodity Columns (Features)", df.columns)

    if st.button("Train Model"):
        if price_column and commodity_columns:
            X = df[commodity_columns]
            y = df[price_column]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        def train_model():
            # Train the model
            global model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("Model Trained Successfully!")
            st.write("### Model Performance")
            st.write(f"**Mean Absolute Error:** {mae:.2f}")
            st.write(f"**Mean Squared Error:** {mse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

        else:
            st.error("Please select both price and commodity columns.")

    if st.button("Make Predictions"):
        if model is not None:
            st.write("### Enter values for prediction")
            input_data = {}
            for col in commodity_columns:
                input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)
            
            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.success(f"Predicted Price (USD): {prediction[0]:.2f}")
        else:
            st.error("Train the model first before making predictions.")

