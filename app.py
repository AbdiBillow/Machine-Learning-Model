import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Initialize session state
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
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

        # Remove target from selected features
        selected_features = [f for f in selected_features if f != target_column]

        # Data Preprocessing Button and Logic
        preprocess_button = st.button('Data Preprocessing', disabled=st.session_state.data_processed)

        if preprocess_button:
            try:
                # Validation checks
                if not selected_features:
                    st.error("❌ Please select at least one feature.")
                    st.stop()
                
                if not pd.api.types.is_numeric_dtype(df[target_column]):
                    st.error("❌ Target column must be numeric for regression.")
                    st.stop()

                # Preprocessing pipeline
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), 
                         [f for f in selected_features if f in categorical_features]),
                        ('num', StandardScaler(), 
                         [f for f in selected_features if f in numeric_features])
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
                    st.warning("⚠️ Missing values detected. Applying imputation...")
                    
                    # Numeric features
                    num_cols = X.select_dtypes(include=['number']).columns
                    if not num_cols.empty:
                        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
                    
                    # Categorical features
                    cat_cols = X.select_dtypes(exclude=['number']).columns
                    for col in cat_cols:
                        X[col] = X[col].fillna(X[col].mode().iloc[0])
                    
                    # Target variable
                    y = y.fillna(y.mean())

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.model = model

                st.success("✅ Data Preprocessing successful!")
                st.session_state.data_processed = True

            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")

        # Train Model Button and Logic
        train_button = st.button("Train Model", disabled=not st.session_state.data_processed or st.session_state.model_trained)

        if train_button:
            try:
                model = st.session_state.model
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                model.fit(X_train, y_train)
                st.session_state.model = model
                st.success("✅ Model Trained Successfully!")
                st.session_state.model_trained = True

                # Evaluate
                y_pred = model.predict(X_test)
                st.write("### 📊 Model Performance")
                st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
                st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"**R²:** {r2_score(y_test, y_pred):.2f}")

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

        # Prediction UI
        if st.session_state.model and st.session_state.model_trained:
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
        st.error(f"❌ Error loading file: {str(e)}")
        
        
       
                       
                   
                            

              
       
