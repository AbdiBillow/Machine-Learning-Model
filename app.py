import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Session state initialization ----------
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "feature_types" not in st.session_state:
    st.session_state.feature_types = {}   # {'categorical': [...], 'numeric': [...]}
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ---------- Reset function ----------
def reset_app():
    """Reset all processing/training states."""
    st.session_state.data_processed = False
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.selected_features = []
    st.session_state.feature_types = {}
    # The button click will automatically rerun the app

# ---------- UI ----------
st.title("Food Price Prediction App")
st.sidebar.info("The Machine Learning Model will make Food Price Prediction for Common Food Groups in Somalia.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Detect file change to reset processing
if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        reset_app()
        st.session_state.uploaded_file_name = uploaded_file.name

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())

        # Auto-detect feature types
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Feature selection
        st.write("### Feature Selection")
        all_columns = df.columns.tolist()
        selected_features = st.multiselect(
            "Select Features (Categorical or Numeric)",
            all_columns,
            default=categorical_features + numeric_features
        )
        target_column = st.selectbox("Select Target Column", all_columns)

        # Remove target from features if present
        if target_column in selected_features:
            selected_features.remove(target_column)

        # ---------- Preprocessing ----------
        preprocess_button = st.button(
            "Data Preprocessing",
            disabled=st.session_state.data_processed
        )

        if preprocess_button:
            # Validations
            if not selected_features:
                st.error("❌ Please select at least one feature.")
            elif not pd.api.types.is_numeric_dtype(df[target_column]):
                st.error("❌ Target column must be numeric for regression.")
            else:
                with st.spinner("Preprocessing data..."):
                    try:
                        # Recompute feature types based on selected features only
                        cat_feats = [
                            f for f in selected_features
                            if pd.api.types.is_categorical_dtype(df[f]) or pd.api.types.is_object_dtype(df[f])
                        ]
                        num_feats = [
                            f for f in selected_features
                            if pd.api.types.is_numeric_dtype(df[f])
                        ]

                        # Build preprocessing pipeline with imputation
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('cat', Pipeline([
                                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                ]), cat_feats),
                                ('num', Pipeline([
                                    ('imputer', SimpleImputer(strategy='mean')),
                                    ('scaler', StandardScaler())
                                ]), num_feats)
                            ],
                            remainder='drop'
                        )

                        # Full pipeline
                        model = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('regressor', LinearRegression())
                        ])

                        # Split data
                        X = df[selected_features]
                        y = df[target_column]

                        # Handle missing target values (though pipeline already handles features)
                        if y.isnull().any():
                            st.warning("⚠️ Missing values in target column. Imputing with mean.")
                            y = y.fillna(y.mean())

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        # Store everything needed later in session state
                        st.session_state.update({
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test,
                            'model': model,
                            'selected_features': selected_features,
                            'feature_types': {
                                'categorical': cat_feats,
                                'numeric': num_feats
                            },
                            'target_column': target_column,
                            'data_processed': True,
                            'model_trained': False  # Reset training flag
                        })

                        st.success("✅ Data Preprocessing successful! You can now train the model.")

                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {str(e)}")

        # ---------- Training ----------
        train_button = st.button(
            "Train Model",
            disabled=not st.session_state.data_processed or st.session_state.model_trained
        )

        if train_button:
            with st.spinner("Training model..."):
                try:
                    model = st.session_state.model
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test

                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    st.session_state.model_trained = True

                    # Evaluate
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.success("✅ Model Trained Successfully!")
                    st.write("### 📊 Model Performance")
                    st.write(f"**MAE:** {mae:.2f}")
                    st.write(f"**MSE:** {mse:.2f}")
                    st.write(f"**R²:** {r2:.2f}")

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")

        # ---------- Prediction ----------
        if st.session_state.model_trained and st.session_state.model is not None:
            st.write("### ✏️ Predict Future Prices")

            selected_features = st.session_state.selected_features
            cat_feats = st.session_state.feature_types.get('categorical', [])
            num_feats = st.session_state.feature_types.get('numeric', [])

            input_data = {}
            for feature in selected_features:
                if feature in cat_feats:
                    # Dropdown for categorical features
                    options = df[feature].dropna().unique().tolist()
                    input_data[feature] = st.selectbox(
                        f"Select {feature}",
                        options,
                        key=f"pred_{feature}"
                    )
                else:
                    # Numeric input (default to mean)
                    default_val = df[feature].mean()
                    if pd.isna(default_val):
                        default_val = 0.0
                    input_data[feature] = st.number_input(
                        f"Enter {feature}",
                        value=float(default_val),
                        key=f"pred_{feature}"
                    )

            if st.button("Predict Future Price"):
                try:
                    # Build DataFrame with correct column order
                    input_df = pd.DataFrame([input_data])
                    # Reorder to match training columns
                    input_df = input_df[selected_features]

                    # Predict
                    prediction = st.session_state.model.predict(input_df)

                    # Display result
                    st.success(f"💰 Predicted Price: **{prediction[0]:.2f} USD**")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

        # ---------- Reset Button ----------
        if st.session_state.data_processed or st.session_state.model_trained:
            if st.button("Reset All (Clear Processing/Training)"):
                reset_app()

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
else:
    st.info("Please upload a CSV file to get started.")
