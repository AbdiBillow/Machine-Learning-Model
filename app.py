import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Initialize session state for the model
if "model" not in st.session_state:
    st.session_state.model = None

# Streamlit UI
st.title("Food Price Prediction App")
st.sidebar.info("The Machine Learning Model will make Food Price Prediction for Common Food Groups in Somalia.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())

        # Auto-detect feature types
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Let user confirm/tweak detected features
        st.write("### Feature Selection")
        selected_features = st.multiselect(
            "Select Features (Categorical or Numeric)",
            df.columns.tolist(),
            default=categorical_features + numeric_features
        )
        target_column = st.selectbox("Select Target Column", df.columns.tolist())

        if target_column and selected_features:
            # Preprocessing pipeline
            if st.button('Data Preprocessing'):
            try:     
                preprocessor = ColumnTransformer(
                transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), [f for f in selected_features if f in categorical_features]),
                        ('num', StandardScaler(), [f for f in selected_features if f in numeric_features])
                    ],
                    remainder='drop'
                )
    
                # Create pipeline
                model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
                ])
    
                # Split data
                X = df[selected_features]
                y = df[target_column]
    
                # Handle missing values
                if X.isnull().sum().any() or y.isnull().any():
                    st.warning("⚠️ Missing values detected. Simple imputation applied.")
                    X = X.fillna(X.mean())  # Numeric: fill with mean
                    y = y.fillna(y.mean())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            if st.button("Train Model"):
                try:
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    st.success("✅ Model Trained Successfully!")

                    # Evaluate
                    y_pred = model.predict(X_test)
                    st.write("### 📊 Model Performance")
                    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
                    st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
                    st.write(f"**R²:** {r2_score(y_test, y_pred):.2f}")

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")

            # Prediction UI
            if st.session_state.model:
                st.write("### ✏️ Make Predictions")
                input_data = {}
                for feature in selected_features:
                    if feature in categorical_features:
                        input_data[feature] = st.selectbox(f"{feature}", df[feature].unique())
                    else:
                        input_data[feature] = st.number_input(f"{feature}", value=df[feature].mean())

                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        prediction = st.session_state.model.predict(input_df)
                        st.success(f"💰 Predicted {target_column}: **{prediction[0]:.2f}**")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {str(e)}")
        
        
       
                       
                   
                            

              
       
