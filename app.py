import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("Commodity Price Prediction App")

# Upload Data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    # Define the target column
    target_column = "Price(USD)"
    # Get all available columns
    available_columns = data.columns.tolist()
    # Select Feature Columns
    # Automatically select all columns except "Price in USD"
    feature_columns = [col for col in available_columns if col != target_column]
    # Let user refine selection
    selected_features = st.multiselect("Select Feature Columns", feature_columns, default=feature_columns)
    # Ensure "Price in USD" is present
    if "Price in USD" not in data.columns:
        st.error("Dataset must contain a 'Price in USD' column.")
    else:
        target_column = "Price in USD"

        if feature_columns:
            # Prepare Data
            X = data[feature_columns]
            y = data[target_column]

            # Convert categorical features to numeric
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            # Train Model Button
            if st.button("Train Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                st.success("Model Trained Successfully!")

                # Make Prediction Button
                st.subheader("Make a Prediction")
                input_data = {}
                for col in feature_columns:
                    if data[col].dtype == 'object':  # If categorical
                        options = data[col].unique().tolist()
                        input_data[col] = st.selectbox(f"Select {col}", options)
                    else:
                        input_data[col] = st.number_input(f"Enter {col} value", value=float(data[col].mean()))

                # Convert categorical inputs
                input_df = pd.DataFrame([input_data])
                for col in input_df.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    input_df[col] = le.fit_transform(input_df[col])

                if st.button("Predict Price in USD"):
                    prediction = model.predict(input_df)
                    st.success(f"Predicted Price in USD: ${prediction[0]:.2f}")

                # Evaluate Model Button
                if st.button("Evaluate Model"):
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"Mean Absolute Error: {mae:.4f}")
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"R² Score: {r2:.4f}")


