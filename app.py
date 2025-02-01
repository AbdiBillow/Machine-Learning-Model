import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize session state for the model
if "model" not in st.session_state:
    st.session_state.model = None

# Streamlit UI
st.title("Linear Regression Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Dropdowns for selecting columns
    all_columns = df.columns.tolist() #Use .tolist() to ensure the columns is a list
    commodity_columns = st.multiselect("Select Commodity Columns (Features)", all_columns)
    region_column = st.selectbox("Select Region Column", all_columns)
    district_column = st.selectbox("Select District Column", all_columns)
    month_column = st.selectbox("Select Month Column", all_columns)
    market_column = st.selectbox("Select Market Column", all_columns)
    price_column = st.selectbox("Select Price Column (Target Variable)", all_columns)
    if price_column and commodity_columns:
        def train_model():
            """Train the Linear Regression model"""
            
            X = df[commodity_columns]
            y = df[price_column]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            st.session_state.model = model # Store the trained model in session state

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("✅ Model Trained Successfully!")
            st.write("### 📊 Model Performance")
            st.write(f"**📌 Mean Absolute Error (MAE):** {mae:.2f}")
            st.write(f"**📌 Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**📌 R² Score:** {r2:.2f}")

        if st.button("Train Model"):
            train_model()
        
        def make_predictions():
           """Make predictions using the trained model"""
           if st.session_state.model is not None:
               st.write("### ✏️ Enter Values for Prediction")
               input_data = {}
               for col in commodity_columns:
                   input_data[col] = st.number_input(f"Enter value for {col}", value=0.0,format="%.2f")

               if st.button("Predict"):
                   input_df = pd.DataFrame([input_data])
                   prediction = st.session_state.model.predict(input_df) # Use the model from session state
                   st.success(f"💰 Predicted Price (USD): {prediction[0]:.2f}")
           else:
               st.error("⚠️ Train the model first before making predictions.")

        if st.button("Make Predictions"):
            make_predictions()
    else:
        st.warning("⚠️ Please select both the target price column and commodity feature columns.")
