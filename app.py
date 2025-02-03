import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt


# Initialize session state for the model
if "model" not in st.session_state:
    st.session_state.model = None

# Streamlit UI
st.title("Food Price Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())
        print(df['Price(USD)'].isnull().sum())
        
        # Extract commodity columns (features)
        commodity_columns = [col for col in df.columns if col.startswith('Commodity_')]
        price_column = 'Price(USD)'  # Assuming the target column is 'Price(USD)'

        if price_column and commodity_columns:
            if price_column in commodity_columns:
                st.error("⚠️ The target column (Price) should not be included in the feature columns.")
            elif not all(df[col].dtype in [int, float] for col in commodity_columns + [price_column]):
                st.error("⚠️ Selected feature and target columns must contain numeric data.")
            else:
                def train_model():
                    """Train the Linear Regression model"""
                    X = df[commodity_columns]
                    y = df[price_column]

                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                    # Train the model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    st.session_state.model = model

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

                    # Display coefficients
                    st.write("### 📈 Model Coefficients")
                    coefficients = pd.DataFrame({
                        "Feature": commodity_columns,
                        "Coefficient": model.coef_
                    })
                    st.write(coefficients)
                    st.write(f"**Intercept:** {model.intercept_:.2f}")

                    # Visualize predictions
                    st.write("### 📊 Actual vs Predicted Prices")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax.set_xlabel("Actual Prices")
                    ax.set_ylabel("Predicted Prices")
                    ax.set_title("Actual vs Predicted Prices")
                    st.pyplot(fig)

                if st.button("Train Model"):
                    train_model()

                def make_predictions():
                    """Make predictions using the trained model"""
                    if st.session_state.model is not None:
                        st.write("### ✏️ Enter Values for Prediction")

                        # Dropdowns for selecting month and commodity
                        selected_month = st.selectbox("Select Month", range(1, 13))
                        selected_commodity = st.selectbox("Select Commodity", commodity_columns)

                        if st.button("Predict"):
                            # Create input data for prediction
                            input_data = {col: 0 for col in commodity_columns}
                            input_data[selected_commodity] = 1  # Set the selected commodity to 1 (1 kg)

                            # Convert input data to DataFrame
                            input_df = pd.DataFrame([input_data])

                            # Predict the price
                            prediction = st.session_state.model.predict(input_df)
                            predicted_price = prediction[0]

                            # Ensure the predicted price is for 1 kg and is reasonable
                            if predicted_price < 1:  # Assuming prices are in USD per kg
                                st.success(f"💰 Predicted Price for 1 kg of {selected_commodity} in month {selected_month}: ${predicted_price:.2f}")
                            else:
                                st.warning(f"⚠️ Predicted price for 1 kg of {selected_commodity} in month {selected_month} is unusually high: ${predicted_price:.2f}. Please check the data or model.")
                    else:
                        st.error("⚠️ Train the model first before making predictions.")

                make_predictions()

                # Save and load model
                if st.session_state.model is not None:
                    if st.button("Save Model"):
                        joblib.dump(st.session_state.model, "linear_regression_model.pkl")
                        st.success("Model saved successfully as 'linear_regression_model.pkl'.")

                if st.button("Load Model"):
                    try:
                        st.session_state.model = joblib.load("linear_regression_model.pkl")
                        st.success("Model loaded successfully!")
                    except FileNotFoundError:
                        st.error("⚠️ No saved model found. Train and save a model first.")

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")
       
