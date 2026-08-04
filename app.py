import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Initialize session state with all required keys
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "categorical_features_selected" not in st.session_state:
    st.session_state.categorical_features_selected = []
if "numeric_features_selected" not in st.session_state:
    st.session_state.numeric_features_selected = []
if "target_column" not in st.session_state:
    st.session_state.target_column = None

# Streamlit UI
st.title("Food Price Prediction App")
st.sidebar.info("The Machine Learning Model will make Food Price Prediction for Common Food Groups in Somalia.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Check if a new file has been uploaded
        current_file_id = id(uploaded_file)
        if st.session_state.uploaded_file_id != current_file_id:
            # Reset session state for new file
            st.session_state.data_processed = False
            st.session_state.model_trained = False
            st.session_state.model = None
            st.session_state.selected_features = []
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.uploaded_file_id = current_file_id
        
        df = pd.read_csv(uploaded_file)
        
        # Validate that DataFrame is not empty
        if df.empty:
            st.error("❌ Uploaded CSV file is empty. Please upload a valid dataset.")
            st.stop()
        
        st.session_state.df = df
        st.write("### Dataset Preview")
        st.write(df.head())
        st.write(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        # Auto-detect feature types
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Let user confirm/tweak detected features
        st.write("### Feature Selection")
        st.info(f"📊 **Detected Features:**\n- Numeric: {numeric_features}\n- Categorical: {categorical_features}")
        
        selected_features = st.multiselect(
            "Select Features (Categorical or Numeric)",
            df.columns.tolist(),
            default=categorical_features + numeric_features
        )
        
        # Store selected features in session state
        st.session_state.selected_features = selected_features
        
        target_column = st.selectbox("Select Target Column", df.columns.tolist())

        # Validate target column is numeric
        if target_column in selected_features and not pd.api.types.is_numeric_dtype(df[target_column]):
            st.warning("⚠️ Warning: Selected target column is not numeric. For regression, please select a numeric column.")

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

                # Recompute feature types based on selected features
                categorical_features_selected = [
                    f for f in selected_features 
                    if pd.api.types.is_categorical_dtype(df[f]) 
                    or pd.api.types.is_object_dtype(df[f])
                ]
                numeric_features_selected = [
                    f for f in selected_features 
                    if pd.api.types.is_numeric_dtype(df[f])
                ]

                st.info(f"✓ **Processing Features:**\n- Categorical: {categorical_features_selected}\n- Numeric: {numeric_features_selected}")

                # Preprocessing pipeline
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_selected),
                        ('num', StandardScaler(), numeric_features_selected)
                    ],
                    remainder='drop'
                )

                # Create pipeline
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression())
                ])

                # Split data
                X = df[selected_features].copy()
                y = df[target_column].copy()

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
                        mode_val = X[col].mode()
                        if len(mode_val) > 0:
                            X[col] = X[col].fillna(mode_val.iloc[0])
                        else:
                            # If mode is empty, use a placeholder value
                            X[col] = X[col].fillna('Unknown')
                    
                    # Target variable - handle both numeric and edge cases
                    y_numeric = pd.to_numeric(y, errors='coerce')
                    if y_numeric.isnull().any():
                        y = y_numeric.fillna(y_numeric.mean())
                    else:
                        y = y_numeric

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Store variables in session state
                st.session_state.update({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'model': model,
                    'categorical_features_selected': categorical_features_selected,
                    'numeric_features_selected': numeric_features_selected,
                    'selected_features': selected_features,
                    'target_column': target_column
                })

                st.success("✅ Data Preprocessing successful!")
                st.session_state.data_processed = True

            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")

        # Train Model Button and Logic
        train_button = st.button(
            "Train Model", 
            disabled=not st.session_state.data_processed or st.session_state.model_trained
        )
        if train_button:
            try:
                # Validate that preprocessing was completed
                if st.session_state.model is None:
                    st.error("❌ Please complete Data Preprocessing first before training.")
                    st.stop()
                
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

        # Reset Model Button
        if st.session_state.model_trained:
            if st.button("🔄 Reset Model"):
                st.session_state.data_processed = False
                st.session_state.model_trained = False
                st.session_state.model = None
                st.success("✅ Model reset. You can now preprocess and train a new model.")
                st.rerun()

        # Prediction Section
        if st.session_state.model and st.session_state.model_trained:
            st.write("### ✏️ Predict Future Prices")
            
            # Retrieve selected features and target column from session state
            selected_features_pred = st.session_state.get('selected_features', [])
            target_column_pred = st.session_state.get('target_column', None)
            categorical_features = st.session_state.categorical_features_selected
            numeric_features = st.session_state.numeric_features_selected
            
            if not selected_features_pred or target_column_pred is None:
                st.error("❌ Error: Feature information not found. Please preprocess again.")
                st.stop()
            
            # Check if Year column exists
            has_year = "Year" in selected_features_pred
            
            input_data = {}

            for feature in selected_features_pred:
                if feature in categorical_features and feature != "Year":
                    unique_values = df[feature].astype(str).unique().tolist()
                    input_data[feature] = st.selectbox(
                        f"{feature}", 
                        unique_values
                    )
                elif feature == "Year" and has_year:
                    year_min = int(df["Year"].min())
                    year_max = int(df["Year"].max())
                    input_data[feature] = st.number_input(
                        f"{feature} (Enter Future Year)", 
                        min_value=year_min, 
                        max_value=year_max + 10, 
                        value=year_max + 1
                    )
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}", 
                        value=float(df[feature].mean())
                    )

            # Prediction button
            if st.button("Predict Future Price"):
                try:
                    # Convert input to DataFrame
                    input_df = pd.DataFrame([input_data])

                    # Make prediction using the full pipeline
                    prediction = st.session_state.model.predict(input_df)

                    # Determine year for display
                    display_year = input_data.get('Year', 'N/A')
                    st.success(f"📅 Predicted Price for {display_year}: **${prediction[0]:.2f} USD**")
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
