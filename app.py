# ==============================================================
# PriceSight — Commodity Price Forecasting
# Single-file Streamlit Application
# ==============================================================

import io
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# ==============================================================
# OPTIONAL XGBOOST
# ==============================================================

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ==============================================================
# PAGE CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="PriceSight — Commodity Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================
# CUSTOM CSS
# ==============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        background: #f8fafc;
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1d4ed8;
    }

    .metric-label {
        color: #64748b;
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    .prediction-box {
        border: 2px solid #2563eb;
        background: #eff6ff;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .prediction-amount {
        font-size: 2.7rem;
        font-weight: 800;
        color: #1d4ed8;
    }

    .prediction-label {
        color: #64748b;
        font-size: .85rem;
    }

    .best-model {
        display: inline-block;
        border-radius: 20px;
        padding: .25rem .75rem;
        background: #dcfce7;
        color: #15803d;
        font-weight: 600;
        font-size: .75rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# SESSION STATE
# ==============================================================

DEFAULT_SESSION_STATE = {
    "trained_models": {},
    "comparison_df": None,
    "best_model_name": None,
    "feature_cols": [],
    "categorical_cols": [],
    "numeric_cols": [],
    "target_col": None,
    "df_raw": None,
    "df_clean": None,
    "model_metadata": {},
    "test_results": {},
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==============================================================
# MONTH UTILITIES
# ==============================================================

MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================


def reset_models():
    """Clear model-related session state."""
    st.session_state.trained_models = {}
    st.session_state.comparison_df = None
    st.session_state.best_model_name = None
    st.session_state.test_results = {}
    st.session_state.model_metadata = {}


def clean_column_names(df):
    """Remove leading/trailing whitespace from column names."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_month_series(series):
    """
    Convert month values to numbers 1-12.

    Supports:
    January
    Jan
    1
    01
    etc.
    """

    def convert_month(value):

        if pd.isna(value):
            return np.nan

        text = str(value).strip().lower()

        if text in MONTH_MAP:
            return MONTH_MAP[text]

        try:
            number = int(float(text))

            if 1 <= number <= 12:
                return number

        except Exception:
            pass

        return np.nan

    return series.apply(convert_month)


def prepare_dataframe(
    df,
    feature_cols,
    categorical_cols,
    numeric_cols,
    target_col,
    month_col=None,
    year_col=None,
):
    """
    Prepare data while preserving categorical variables as text.

    No encoding happens here.
    OneHotEncoder is fitted later inside the ML pipeline.
    """

    required_cols = list(
        dict.fromkeys(feature_cols + [target_col])
    )

    data = df[required_cols].copy()

    # ----------------------------------------------------------
    # Target
    # ----------------------------------------------------------

    data[target_col] = pd.to_numeric(
        data[target_col],
        errors="coerce",
    )

    # Remove observations with missing target
    data = data.dropna(subset=[target_col])

    # ----------------------------------------------------------
    # Categorical features
    # ----------------------------------------------------------

    for col in categorical_cols:

        if col in data.columns:

            data[col] = (
                data[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            data[col] = data[col].replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "none": pd.NA,
                    "null": pd.NA,
                }
            )

    # ----------------------------------------------------------
    # Numeric features
    # ----------------------------------------------------------

    for col in numeric_cols:

        if col in data.columns:

            if month_col and col == month_col:
                data[col] = normalize_month_series(data[col])

            else:
                data[col] = pd.to_numeric(
                    data[col],
                    errors="coerce",
                )

    # ----------------------------------------------------------
    # Year
    # ----------------------------------------------------------

    if year_col and year_col in data.columns:

        data[year_col] = pd.to_numeric(
            data[year_col],
            errors="coerce",
        )

    data.reset_index(drop=True, inplace=True)

    return data


def remove_numeric_outliers(
    df,
    numeric_cols,
    target_col,
):
    """
    Remove outliers using IQR.

    Only numeric predictors + target are considered.
    Categorical variables are never filtered.
    """

    output = df.copy()

    columns = list(
        dict.fromkeys(numeric_cols + [target_col])
    )

    for col in columns:

        if col not in output.columns:
            continue

        numeric_series = pd.to_numeric(
            output[col],
            errors="coerce",
        )

        non_missing = numeric_series.dropna()

        if len(non_missing) < 5:
            continue

        q1 = non_missing.quantile(0.25)
        q3 = non_missing.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        keep_mask = (
            numeric_series.isna()
            | (
                (numeric_series >= lower)
                & (numeric_series <= upper)
            )
        )

        output = output[keep_mask]

    return output.reset_index(drop=True)


# ==============================================================
# PREPROCESSOR
# ==============================================================


def build_preprocessor(
    categorical_cols,
    numeric_cols,
    scale_numeric=True,
):

    transformers = []

    # ----------------------------------------------------------
    # CATEGORICAL PIPELINE
    # ----------------------------------------------------------

    if categorical_cols:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            )
        )

    # ----------------------------------------------------------
    # NUMERIC PIPELINE
    # ----------------------------------------------------------

    if numeric_cols:

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]

        if scale_numeric:

            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler(),
                )
            )

        numeric_pipeline = Pipeline(
            steps=numeric_steps
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ==============================================================
# MODEL PIPELINE
# ==============================================================


def build_model_pipeline(
    model_name,
    categorical_cols,
    numeric_cols,
):

    # Scaling useful for LR and SVR.
    # Not needed for tree-based models.

    scale_numeric = model_name in [
        "Linear Regression",
        "Support Vector Machine",
    ]

    preprocessor = build_preprocessor(
        categorical_cols,
        numeric_cols,
        scale_numeric=scale_numeric,
    )

    if model_name == "Linear Regression":

        model = LinearRegression()

    elif model_name == "Random Forest":

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        )

    elif model_name == "XGBoost":

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed."
            )

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective="reg:squarederror",
            eval_metric="rmse",
            verbosity=0,
            n_jobs=-1,
        )

    elif model_name == "Support Vector Machine":

        model = SVR(
            kernel="rbf",
            C=10,
            epsilon=0.1,
            gamma="scale",
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ==============================================================
# MODEL EVALUATION
# ==============================================================


def calculate_adjusted_r2(
    r2,
    n,
    p,
):
    """
    Adjusted R² using raw predictor count.

    This is intentionally based on original predictors,
    not one-hot-expanded columns.
    """

    if n <= p + 1:
        return np.nan

    return (
        1
        - (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )


def evaluate_model(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
):

    pipeline.fit(
        X_train,
        y_train,
    )

    train_predictions = pipeline.predict(
        X_train
    )

    test_predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions,
        )
    )

    train_r2 = r2_score(
        y_train,
        train_predictions,
    )

    test_r2 = r2_score(
        y_test,
        test_predictions,
    )

    adjusted_r2 = calculate_adjusted_r2(
        test_r2,
        len(y_test),
        X_test.shape[1],
    )

    # ----------------------------------------------------------
    # CROSS VALIDATION ONLY ON TRAINING DATA
    # ----------------------------------------------------------

    number_of_folds = min(
        5,
        len(X_train),
    )

    if number_of_folds >= 2:

        try:

            cv_strategy = KFold(
                n_splits=number_of_folds,
                shuffle=True,
                random_state=42,
            )

            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring="r2",
                n_jobs=-1,
            )

            cv_mean = np.nanmean(
                cv_scores
            )

            cv_std = np.nanstd(
                cv_scores
            )

        except Exception:

            cv_mean = np.nan
            cv_std = np.nan

    else:

        cv_mean = np.nan
        cv_std = np.nan

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Test": test_r2,
        "R2 Train": train_r2,
        "Adjusted R2": adjusted_r2,
        "CV R2 Mean": cv_mean,
        "CV R2 Std": cv_std,
        "_pipeline": pipeline,
        "_y_test": y_test,
        "_predictions": test_predictions,
    }


# ==============================================================
# MODEL CONCLUSION
# ==============================================================


def determine_conclusion(
    row,
    best_model,
):

    r2 = row["R2 Test"]

    train_r2 = row[
        "R2 Train"
    ]

    gap = train_r2 - r2

    parts = []

    if row["Model"] == best_model:
        parts.append(
            "Best overall model."
        )

    if r2 >= 0.85:

        parts.append(
            "Excellent fit."
        )

    elif r2 >= 0.65:

        parts.append(
            "Good fit."
        )

    elif r2 >= 0.40:

        parts.append(
            "Moderate fit."
        )

    elif r2 >= 0:

        parts.append(
            "Weak predictive fit."
        )

    else:

        parts.append(
            "Poor fit; worse than predicting the mean."
        )

    if gap > 0.20:

        parts.append(
            "Possible overfitting."
        )

    return " ".join(parts)


# ==============================================================
# FEATURE NAMES / IMPORTANCE
# ==============================================================


def get_transformed_feature_names(
    pipeline,
):

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    try:

        return list(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        return []


def get_feature_importance(
    pipeline,
):

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        get_transformed_feature_names(
            pipeline
        )
    )

    if not feature_names:
        return None

    if hasattr(
        model,
        "feature_importances_",
    ):

        values = (
            model.feature_importances_
        )

    elif hasattr(
        model,
        "coef_",
    ):

        values = np.ravel(
            model.coef_
        )

        values = np.abs(
            values
        )

    else:

        return None

    if len(values) != len(
        feature_names
    ):

        return None

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": values,
        }
    )

    importance_df[
        "Feature"
    ] = (
        importance_df[
            "Feature"
        ]
        .str.replace(
            "categorical__",
            "",
            regex=False,
        )
        .str.replace(
            "numeric__",
            "",
            regex=False,
        )
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return importance_df


# ==============================================================
# FORECAST DATE CALCULATION
# ==============================================================


def advance_month(
    month,
    year,
):

    month = int(month)
    year = int(year)

    month += 1

    if month > 12:

        month = 1
        year += 1

    return month, year


# ==============================================================
# SIDEBAR
# ==============================================================

with st.sidebar:

    st.title(
        "📈 PriceSight"
    )

    st.caption(
        "Commodity Price Forecasting"
    )

    st.divider()

    st.subheader(
        "Model Configuration"
    )

    test_size = (
        st.slider(
            "Test Split (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5,
        )
        / 100
    )

    drop_duplicates = st.checkbox(
        "Drop duplicate rows",
        value=True,
    )

    outlier_removal = st.checkbox(
        "Remove numeric outliers (IQR)",
        value=False,
    )

    st.divider()

    st.subheader(
        "Saved Model"
    )

    # ----------------------------------------------------------
    # DOWNLOAD MODEL
    # ----------------------------------------------------------

    if (
        st.session_state.best_model_name
        and st.session_state.trained_models
    ):

        best_model_name = (
            st.session_state.best_model_name
        )

        best_pipeline = (
            st.session_state.trained_models.get(
                best_model_name
            )
        )

        if best_pipeline:

            model_buffer = io.BytesIO()

            model_package = {
                "pipeline": best_pipeline,
                "best_model_name": best_model_name,
                "feature_cols": st.session_state.feature_cols,
                "categorical_cols": st.session_state.categorical_cols,
                "numeric_cols": st.session_state.numeric_cols,
                "target_col": st.session_state.target_col,
                "metadata": st.session_state.model_metadata,
            }

            joblib.dump(
                model_package,
                model_buffer,
            )

            model_buffer.seek(0)

            st.download_button(
                label=f"Download {best_model_name}",
                data=model_buffer,
                file_name="pricesight_best_model.pkl",
                mime="application/octet-stream",
            )

    # ----------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------

    uploaded_model = st.file_uploader(
        "Load saved model",
        type=["pkl"],
        key="model_uploader",
    )

    if (
        uploaded_model is not None
        and st.button(
            "Apply Loaded Model"
        )
    ):

        try:

            loaded = joblib.load(
                uploaded_model
            )

            model_name = loaded[
                "best_model_name"
            ]

            st.session_state.trained_models = {
                model_name: loaded[
                    "pipeline"
                ]
            }

            st.session_state.best_model_name = (
                model_name
            )

            st.session_state.feature_cols = (
                loaded.get(
                    "feature_cols",
                    [],
                )
            )

            st.session_state.categorical_cols = (
                loaded.get(
                    "categorical_cols",
                    [],
                )
            )

            st.session_state.numeric_cols = (
                loaded.get(
                    "numeric_cols",
                    [],
                )
            )

            st.session_state.target_col = (
                loaded.get(
                    "target_col"
                )
            )

            st.session_state.model_metadata = (
                loaded.get(
                    "metadata",
                    {},
                )
            )

            st.success(
                "Model loaded successfully."
            )

        except Exception as error:

            st.error(
                f"Could not load model: {error}"
            )


# ==============================================================
# HEADER
# ==============================================================

st.markdown(
    """
    <div class="main-title">
        PriceSight — Commodity Price Forecasting
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload commodity price data, compare machine-learning
        models, predict prices, and generate future forecasts.
    </div>
    """,
    unsafe_allow_html=True,
)

if not XGBOOST_AVAILABLE:

    st.info(
        "XGBoost is not currently installed. "
        "Linear Regression, Random Forest and SVM remain available. "
        "Install `xgboost` to enable the XGBoost model."
    )


# ==============================================================
# FILE UPLOAD
# ==============================================================

uploaded_file = st.file_uploader(
    "Upload Commodity Price CSV",
    type=["csv"],
    help=(
        "Your dataset should ideally contain Commodity, "
        "Market/Region, Month, Year and Price."
    ),
)


if uploaded_file is None:

    st.info(
        "Upload a CSV file above to begin."
    )

    st.stop()


# ==============================================================
# READ DATA
# ==============================================================

try:

    df_raw = pd.read_csv(
        uploaded_file
    )

except Exception as error:

    st.error(
        f"Could not read the CSV file: {error}"
    )

    st.stop()


df_raw = clean_column_names(
    df_raw
)


if drop_duplicates:

    df_raw = (
        df_raw
        .drop_duplicates()
        .reset_index(drop=True)
    )


st.session_state.df_raw = df_raw


if df_raw.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


# ==============================================================
# COLUMN MAPPING
# ==============================================================

st.divider()

st.subheader(
    "Step 1 — Map Dataset Columns"
)

st.caption(
    "Select the columns that represent the main forecasting variables."
)


all_columns = (
    df_raw.columns.tolist()
)


numeric_candidate_columns = []

for col in all_columns:

    converted = pd.to_numeric(
        df_raw[col],
        errors="coerce",
    )

    if converted.notna().mean() >= 0.70:

        numeric_candidate_columns.append(
            col
        )


map1, map2, map3, map4, map5 = (
    st.columns(5)
)


with map1:

    commodity_col = st.selectbox(
        "Commodity",
        ["(none)"] + all_columns,
        key="commodity_mapping",
    )


with map2:

    market_col = st.selectbox(
        "Market / Region",
        ["(none)"] + all_columns,
        key="market_mapping",
    )


with map3:

    month_col = st.selectbox(
        "Month",
        ["(none)"] + all_columns,
        key="month_mapping",
    )


with map4:

    year_col = st.selectbox(
        "Year",
        ["(none)"] + all_columns,
        key="year_mapping",
    )


with map5:

    target_options = (
        numeric_candidate_columns
        if numeric_candidate_columns
        else all_columns
    )

    target_col = st.selectbox(
        "Price / Target",
        target_options,
        key="target_mapping",
    )


# ==============================================================
# VALIDATION
# ==============================================================

selected_mapping = [
    col
    for col in [
        commodity_col,
        market_col,
        month_col,
        year_col,
    ]
    if col != "(none)"
]


if target_col in selected_mapping:

    st.error(
        "The target Price column cannot also be used "
        "as Commodity, Market, Month or Year."
    )

    st.stop()


duplicates = [
    col
    for col in selected_mapping
    if selected_mapping.count(col) > 1
]


if duplicates:

    st.error(
        "The same dataset column has been mapped "
        "to more than one feature. Please correct the mapping."
    )

    st.stop()


# ==============================================================
# FEATURE CLASSIFICATION
# ==============================================================

categorical_cols = []

if commodity_col != "(none)":

    categorical_cols.append(
        commodity_col
    )

if market_col != "(none)":

    categorical_cols.append(
        market_col
    )


numeric_cols = []

if month_col != "(none)":

    numeric_cols.append(
        month_col
    )

if year_col != "(none)":

    numeric_cols.append(
        year_col
    )


excluded = set(
    categorical_cols
    + numeric_cols
    + [target_col]
)


available_extra_numeric = [
    col
    for col in numeric_candidate_columns
    if col not in excluded
]


extra_features = st.multiselect(
    "Additional Numeric Features (optional)",
    available_extra_numeric,
)


numeric_cols.extend(
    extra_features
)


feature_cols = (
    categorical_cols
    + numeric_cols
)


if not feature_cols:

    st.warning(
        "Select at least one predictor column."
    )

    st.stop()


# ==============================================================
# PREPARE DATA
# ==============================================================

df_clean = prepare_dataframe(
    df=df_raw,
    feature_cols=feature_cols,
    categorical_cols=categorical_cols,
    numeric_cols=numeric_cols,
    target_col=target_col,
    month_col=(
        month_col
        if month_col != "(none)"
        else None
    ),
    year_col=(
        year_col
        if year_col != "(none)"
        else None
    ),
)


if outlier_removal:

    df_clean = remove_numeric_outliers(
        df_clean,
        numeric_cols,
        target_col,
    )


if len(df_clean) < 20:

    st.error(
        f"Only {len(df_clean)} usable observations remain. "
        "At least 20 observations are recommended."
    )

    st.stop()


st.session_state.df_clean = (
    df_clean
)

st.session_state.feature_cols = (
    feature_cols
)

st.session_state.categorical_cols = (
    categorical_cols
)

st.session_state.numeric_cols = (
    numeric_cols
)

st.session_state.target_col = (
    target_col
)


# ==============================================================
# DATA SUMMARY
# ==============================================================

missing_percentage = (
    df_raw[
        list(
            dict.fromkeys(
                feature_cols
                + [target_col]
            )
        )
    ]
    .isna()
    .mean()
    .mean()
    * 100
)


summary1, summary2, summary3, summary4 = (
    st.columns(4)
)


summary1.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-value">
            {len(df_raw):,}
        </div>
        <div class="metric-label">
            Original Rows
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


summary2.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-value">
            {len(df_clean):,}
        </div>
        <div class="metric-label">
            Training Rows
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


summary3.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-value">
            {len(feature_cols)}
        </div>
        <div class="metric-label">
            Predictors
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


summary4.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-value">
            {missing_percentage:.1f}%
        </div>
        <div class="metric-label">
            Missing Data
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================
# APPLICATION TABS
# ==============================================================

(
    tab_data,
    tab_train,
    tab_prediction,
    tab_forecast,
) = st.tabs(
    [
        "📋 Data Preview",
        "🤖 Train & Compare",
        "💰 Price Prediction",
        "📈 Future Forecast",
    ]
)


# ==============================================================
# TAB 1 — DATA
# ==============================================================

with tab_data:

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df_clean.head(200),
        use_container_width=True,
        height=350,
    )

    st.caption(
        f"Showing up to 200 of {len(df_clean):,} usable rows."
    )

    with st.expander(
        "Descriptive Statistics"
    ):

        numeric_description = (
            df_clean
            .select_dtypes(
                include=np.number
            )
            .describe()
            .T
        )

        if not numeric_description.empty:

            st.dataframe(
                numeric_description.style.format(
                    "{:.4f}"
                ),
                use_container_width=True,
            )

        else:

            st.info(
                "No numeric variables available."
            )

    with st.expander(
        "Missing Values"
    ):

        columns_for_missing = (
            feature_cols
            + [target_col]
        )

        missing_df = pd.DataFrame(
            {
                "Column": columns_for_missing,
                "Missing Count": [
                    df_raw[col]
                    .isna()
                    .sum()
                    for col
                    in columns_for_missing
                ],
            }
        )

        missing_df[
            "Missing %"
        ] = (
            missing_df[
                "Missing Count"
            ]
            / len(df_raw)
            * 100
        ).round(2)

        st.dataframe(
            missing_df,
            use_container_width=True,
        )


# ==============================================================
# TAB 2 — TRAIN & COMPARE
# ==============================================================

with tab_train:

    st.subheader(
        "Step 2 — Train & Compare Models"
    )

    st.caption(
        "Preprocessing is fitted only on the training data "
        "inside each model pipeline, preventing data leakage."
    )


    model_names = [
        "Linear Regression",
        "Random Forest",
        "Support Vector Machine",
    ]


    if XGBOOST_AVAILABLE:

        model_names.insert(
            2,
            "XGBoost",
        )


    st.write(
        "**Models:** "
        + " • ".join(
            model_names
        )
    )


    if st.button(
        "Train All Models",
        type="primary",
        key="train_models_button",
    ):

        X = df_clean[
            feature_cols
        ].copy()

        y = df_clean[
            target_col
        ].astype(float)


        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
            )
        )


        results = []

        trained_models = {}

        test_results = {}


        progress = st.progress(
            0,
            text="Training models..."
        )


        for index, model_name in enumerate(
            model_names
        ):

            progress.progress(
                index / len(model_names),
                text=f"Training {model_name}...",
            )

            try:

                pipeline = build_model_pipeline(
                    model_name,
                    categorical_cols,
                    numeric_cols,
                )


                result = evaluate_model(
                    pipeline,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    model_name,
                )


                trained_models[
                    model_name
                ] = result[
                    "_pipeline"
                ]


                test_results[
                    model_name
                ] = {
                    "y_test": np.array(
                        result["_y_test"]
                    ),
                    "predictions": np.array(
                        result["_predictions"]
                    ),
                }


                results.append(
                    {
                        key: value
                        for key, value
                        in result.items()
                        if not key.startswith("_")
                    }
                )


            except Exception as error:

                st.warning(
                    f"{model_name} failed: {error}"
                )


            progress.progress(
                (index + 1)
                / len(model_names),
                text=f"Finished {model_name}",
            )


        progress.empty()


        if results:

            comparison_df = pd.DataFrame(
                results
            )


            comparison_df = (
                comparison_df
                .sort_values(
                    "R2 Test",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )


            best_model_name = (
                comparison_df.iloc[0][
                    "Model"
                ]
            )


            comparison_df[
                "Conclusion"
            ] = comparison_df.apply(
                lambda row: determine_conclusion(
                    row,
                    best_model_name,
                ),
                axis=1,
            )


            st.session_state.trained_models = (
                trained_models
            )

            st.session_state.comparison_df = (
                comparison_df
            )

            st.session_state.best_model_name = (
                best_model_name
            )

            st.session_state.test_results = (
                test_results
            )


            st.session_state.model_metadata = {
                "commodity_col": commodity_col,
                "market_col": market_col,
                "month_col": month_col,
                "year_col": year_col,
                "extra_features": extra_features,
                "trained_at": datetime.now().isoformat(),
            }


            best_score = (
                comparison_df.iloc[0][
                    "R2 Test"
                ]
            )


            st.success(
                f"Training completed. "
                f"Best model: {best_model_name} "
                f"(Test R² = {best_score:.4f})"
            )


    # ----------------------------------------------------------
    # RESULTS
    # ----------------------------------------------------------

    if (
        st.session_state.comparison_df
        is not None
    ):

        comparison_df = (
            st.session_state.comparison_df.copy()
        )

        best_model_name = (
            st.session_state.best_model_name
        )


        comparison_df.insert(
            1,
            "Rank",
            range(
                1,
                len(comparison_df) + 1
            ),
        )


        st.markdown(
            "### Model Comparison"
        )


        display_columns = [
            "Rank",
            "Model",
            "MAE",
            "RMSE",
            "R2 Test",
            "Adjusted R2",
            "R2 Train",
            "CV R2 Mean",
            "CV R2 Std",
            "Conclusion",
        ]


        display_df = comparison_df[
            display_columns
        ].copy()


        def highlight_best(row):

            if (
                row["Model"]
                == best_model_name
            ):

                return [
                    "background-color: #dcfce7; font-weight: 600"
                ] * len(row)

            return [
                ""
            ] * len(row)


        styled_comparison = (
            display_df.style
            .apply(
                highlight_best,
                axis=1,
            )
            .format(
                {
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "R2 Test": "{:.4f}",
                    "Adjusted R2": "{:.4f}",
                    "R2 Train": "{:.4f}",
                    "CV R2 Mean": "{:.4f}",
                    "CV R2 Std": "{:.4f}",
                }
            )
        )


        st.dataframe(
            styled_comparison,
            use_container_width=True,
        )


        # ------------------------------------------------------
        # BEST MODEL METRICS
        # ------------------------------------------------------

        best_row = comparison_df.iloc[
            0
        ]


        st.markdown(
            f"### 🏆 Best Model: {best_model_name}"
        )


        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )


        metric1.metric(
            "MAE",
            f"{best_row['MAE']:.4f}",
        )

        metric2.metric(
            "RMSE",
            f"{best_row['RMSE']:.4f}",
        )

        metric3.metric(
            "Test R²",
            f"{best_row['R2 Test']:.4f}",
        )

        metric4.metric(
            "CV R²",
            f"{best_row['CV R2 Mean']:.4f}",
        )


        # ------------------------------------------------------
        # DOWNLOAD RESULTS
        # ------------------------------------------------------

        downloadable = (
            comparison_df.drop(
                columns=[],
                errors="ignore",
            )
        )

        comparison_csv = (
            downloadable
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            "Download Model Comparison CSV",
            comparison_csv,
            "pricesight_model_comparison.csv",
            "text/csv",
        )


        # ------------------------------------------------------
        # RESIDUAL ANALYSIS
        # ------------------------------------------------------

        st.markdown(
            "### Residual Diagnostics"
        )


        if (
            best_model_name
            in st.session_state.test_results
        ):

            actual = (
                st.session_state.test_results[
                    best_model_name
                ][
                    "y_test"
                ]
            )

            predicted = (
                st.session_state.test_results[
                    best_model_name
                ][
                    "predictions"
                ]
            )


            residuals = (
                actual
                - predicted
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.scatter(
                predicted,
                residuals,
                alpha=0.7,
            )

            ax.axhline(
                0,
                linestyle="--",
            )

            ax.set_xlabel(
                "Predicted Price"
            )

            ax.set_ylabel(
                "Residual (Actual - Predicted)"
            )

            ax.set_title(
                f"Residual Plot — {best_model_name}"
            )

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )


        # ------------------------------------------------------
        # FEATURE IMPORTANCE
        # ------------------------------------------------------

        best_pipeline = (
            st.session_state.trained_models[
                best_model_name
            ]
        )


        importance_df = (
            get_feature_importance(
                best_pipeline
            )
        )


        if (
            importance_df
            is not None
            and not importance_df.empty
        ):

            st.markdown(
                "### Feature Importance"
            )


            top_features = (
                importance_df.head(
                    20
                )
                .sort_values(
                    "Importance",
                    ascending=True,
                )
            )


            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            ax.barh(
                top_features[
                    "Feature"
                ],
                top_features[
                    "Importance"
                ],
            )

            ax.set_xlabel(
                "Importance"
            )

            ax.set_title(
                f"Top Features — {best_model_name}"
            )

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )


        with st.expander(
            "Understanding the Metrics"
        ):

            st.markdown(
                """
                | Metric | Meaning |
                |---|---|
                | **MAE** | Average absolute prediction error. Lower is better. |
                | **RMSE** | Prediction error that penalizes large mistakes more heavily. Lower is better. |
                | **Test R²** | Variance explained on unseen test observations. Higher is better. |
                | **Adjusted R²** | R² adjusted for the number of predictors. |
                | **Train R²** | Model performance on training observations. |
                | **CV R² Mean** | Average R² from cross-validation performed only on training data. |
                | **CV R² Std** | Variation in cross-validation performance. Lower indicates more stable performance. |
                """
            )


# ==============================================================
# TAB 3 — PRICE PREDICTION
# ==============================================================

with tab_prediction:

    st.subheader(
        "Step 3 — Predict Commodity Price"
    )


    if not st.session_state.trained_models:

        st.info(
            "Train at least one model first."
        )

    else:

        st.caption(
            "Enter the characteristics of the commodity observation. "
            "Categorical values are processed automatically by the fitted "
            "OneHotEncoder."
        )


        def get_options(
            column
        ):

            if (
                column == "(none)"
                or column not in df_raw.columns
            ):

                return []

            return sorted(
                df_raw[
                    column
                ]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )


        prediction_values = {}


        input_col1, input_col2 = (
            st.columns(2)
        )


        # ------------------------------------------------------
        # COMMODITY
        # ------------------------------------------------------

        if commodity_col != "(none)":

            commodity_options = (
                get_options(
                    commodity_col
                )
            )

            selected_commodity = (
                input_col1.selectbox(
                    "Commodity",
                    commodity_options,
                    key="prediction_commodity",
                )
            )

            prediction_values[
                commodity_col
            ] = selected_commodity


        # ------------------------------------------------------
        # MARKET
        # ------------------------------------------------------

        if market_col != "(none)":

            market_options = (
                get_options(
                    market_col
                )
            )

            selected_market = (
                input_col2.selectbox(
                    "Market / Region",
                    market_options,
                    key="prediction_market",
                )
            )

            prediction_values[
                market_col
            ] = selected_market


        input_col3, input_col4 = (
            st.columns(2)
        )


        # ------------------------------------------------------
        # MONTH
        # ------------------------------------------------------

        if month_col != "(none)":

            selected_month_name = (
                input_col3.selectbox(
                    "Month",
                    list(
                        MONTH_NAMES.values()
                    ),
                    key="prediction_month",
                )
            )

            selected_month = [
                key
                for key, value
                in MONTH_NAMES.items()
                if value
                == selected_month_name
            ][0]

            prediction_values[
                month_col
            ] = selected_month


        # ------------------------------------------------------
        # YEAR
        # ------------------------------------------------------

        if year_col != "(none)":

            available_years = pd.to_numeric(
                df_clean[
                    year_col
                ],
                errors="coerce",
            ).dropna()


            default_year = (
                int(
                    available_years.max()
                )
                if len(
                    available_years
                )
                else datetime.now().year
            )


            selected_year = (
                input_col4.number_input(
                    "Year",
                    min_value=1900,
                    max_value=2200,
                    value=default_year,
                    step=1,
                    key="prediction_year",
                )
            )


            prediction_values[
                year_col
            ] = selected_year


        # ------------------------------------------------------
        # EXTRA NUMERIC FEATURES
        # ------------------------------------------------------

        if extra_features:

            st.markdown(
                "**Additional Numeric Inputs**"
            )


            columns = st.columns(
                min(
                    4,
                    len(extra_features),
                )
            )


            for index, feature in enumerate(
                extra_features
            ):

                feature_series = (
                    pd.to_numeric(
                        df_clean[
                            feature
                        ],
                        errors="coerce",
                    )
                )


                median_value = (
                    float(
                        feature_series.median()
                    )
                    if feature_series.notna().any()
                    else 0.0
                )


                prediction_values[
                    feature
                ] = columns[
                    index
                    % len(columns)
                ].number_input(
                    feature,
                    value=median_value,
                    key=f"prediction_{feature}",
                )


        model_choices = list(
            st.session_state.trained_models.keys()
        )


        default_model = (
            st.session_state.best_model_name
        )


        model_index = (
            model_choices.index(
                default_model
            )
            if default_model
            in model_choices
            else 0
        )


        selected_model = (
            st.selectbox(
                "Prediction Model",
                model_choices,
                index=model_index,
                key="prediction_model",
            )
        )


        if st.button(
            "Predict Price",
            type="primary",
            key="predict_price_button",
        ):

            try:

                # Ensure every required feature exists
                for feature in feature_cols:

                    if feature not in prediction_values:

                        prediction_values[
                            feature
                        ] = np.nan


                input_df = pd.DataFrame(
                    [
                        prediction_values
                    ]
                )


                input_df = input_df[
                    feature_cols
                ]


                # Categorical cleanup
                for col in categorical_cols:

                    if col in input_df.columns:

                        input_df[col] = (
                            input_df[col]
                            .astype("string")
                            .str.strip()
                            .str.lower()
                        )


                # Numeric cleanup
                for col in numeric_cols:

                    if col in input_df.columns:

                        input_df[col] = pd.to_numeric(
                            input_df[col],
                            errors="coerce",
                        )


                pipeline = (
                    st.session_state.trained_models[
                        selected_model
                    ]
                )


                prediction = float(
                    pipeline.predict(
                        input_df
                    )[0]
                )


                prediction = max(
                    0,
                    prediction,
                )


                # --------------------------------------------------
                # APPROXIMATE PREDICTION RANGE
                # --------------------------------------------------

                model_comparison = (
                    st.session_state.comparison_df
                )


                if (
                    model_comparison
                    is not None
                ):

                    matching_row = (
                        model_comparison[
                            model_comparison[
                                "Model"
                            ]
                            == selected_model
                        ]
                    )


                    rmse_value = (
                        float(
                            matching_row.iloc[
                                0
                            ][
                                "RMSE"
                            ]
                        )
                        if not matching_row.empty
                        else 0
                    )

                else:

                    rmse_value = 0


                lower_bound = max(
                    0,
                    prediction
                    - 1.96
                    * rmse_value,
                )


                upper_bound = (
                    prediction
                    + 1.96
                    * rmse_value
                )


                st.markdown(
                    f"""
                    <div class="prediction-box">

                        <div class="prediction-label">
                            Predicted Commodity Price
                        </div>

                        <div class="prediction-amount">
                            USD {prediction:,.4f}
                        </div>

                        <div class="prediction-label">
                            Approximate prediction range:
                            USD {lower_bound:,.4f}
                            —
                            USD {upper_bound:,.4f}
                        </div>

                        <br>

                        <span class="best-model">
                            Model: {selected_model}
                        </span>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                dataset_prices = pd.to_numeric(
                    df_clean[
                        target_col
                    ],
                    errors="coerce",
                )


                context1, context2, context3 = (
                    st.columns(3)
                )


                context1.metric(
                    "Dataset Minimum",
                    f"USD {dataset_prices.min():,.4f}",
                )


                median_price = (
                    dataset_prices.median()
                )


                context2.metric(
                    "Dataset Median",
                    f"USD {median_price:,.4f}",
                    delta=(
                        f"{prediction - median_price:+.4f}"
                    ),
                )


                context3.metric(
                    "Dataset Maximum",
                    f"USD {dataset_prices.max():,.4f}",
                )


                st.caption(
                    "The displayed range is based on ±1.96 × test RMSE. "
                    "It is a useful uncertainty indicator but is not a formal "
                    "statistical prediction interval for every model."
                )


            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

                st.exception(
                    error
                )


# ==============================================================
# TAB 4 — FUTURE FORECAST
# ==============================================================

with tab_forecast:

    st.subheader(
        "Step 4 — Future Price Forecast"
    )


    if not st.session_state.trained_models:

        st.info(
            "Train the models before generating a forecast."
        )

    elif (
        month_col == "(none)"
        or year_col == "(none)"
    ):

        st.warning(
            "Future monthly forecasting requires both "
            "Month and Year columns to be mapped."
        )

    else:

        st.caption(
            "The forecast advances the calendar month-by-month. "
            "Commodity and market remain fixed instead of being "
            "artificially incremented."
        )


        forecast_col1, forecast_col2, forecast_col3 = (
            st.columns(3)
        )


        # ------------------------------------------------------
        # COMMODITY
        # ------------------------------------------------------

        forecast_commodity = None


        if commodity_col != "(none)":

            forecast_commodity_options = (
                get_options(
                    commodity_col
                )
            )


            forecast_commodity = (
                forecast_col1.selectbox(
                    "Commodity",
                    forecast_commodity_options,
                    key="forecast_commodity",
                )
            )


        # ------------------------------------------------------
        # MARKET
        # ------------------------------------------------------

        forecast_market = None


        if market_col != "(none)":

            forecast_market_options = (
                get_options(
                    market_col
                )
            )


            forecast_market = (
                forecast_col2.selectbox(
                    "Market / Region",
                    forecast_market_options,
                    key="forecast_market",
                )
            )


        # ------------------------------------------------------
        # STEPS
        # ------------------------------------------------------

        forecast_steps = (
            forecast_col3.number_input(
                "Forecast Months",
                min_value=1,
                max_value=60,
                value=12,
                step=1,
                key="forecast_steps",
            )
        )


        # ------------------------------------------------------
        # MODEL
        # ------------------------------------------------------

        forecast_model_options = (
            list(
                st.session_state.trained_models.keys()
            )
        )


        forecast_model_index = (
            forecast_model_options.index(
                st.session_state.best_model_name
            )
            if st.session_state.best_model_name
            in forecast_model_options
            else 0
        )


        forecast_model = (
            st.selectbox(
                "Forecast Model",
                forecast_model_options,
                index=forecast_model_index,
                key="forecast_model",
            )
        )


        # ------------------------------------------------------
        # STARTING DATE
        # ------------------------------------------------------

        available_years = pd.to_numeric(
            df_clean[
                year_col
            ],
            errors="coerce",
        ).dropna()


        if available_years.empty:

            default_start_year = (
                datetime.now().year
            )

        else:

            default_start_year = int(
                available_years.max()
            )


        latest_year_data = df_clean[
            pd.to_numeric(
                df_clean[
                    year_col
                ],
                errors="coerce",
            )
            == default_start_year
        ]


        available_months = pd.to_numeric(
            latest_year_data[
                month_col
            ],
            errors="coerce",
        ).dropna()


        default_start_month = (
            int(
                available_months.max()
            )
            if not available_months.empty
            else 12
        )


        date1, date2 = st.columns(
            2
        )


        start_month_name = (
            date1.selectbox(
                "Last Observed Month",
                list(
                    MONTH_NAMES.values()
                ),
                index=max(
                    0,
                    default_start_month
                    - 1,
                ),
                key="forecast_start_month",
            )
        )


        start_month = [
            month_number
            for month_number, month_name
            in MONTH_NAMES.items()
            if month_name
            == start_month_name
        ][0]


        start_year = (
            date2.number_input(
                "Last Observed Year",
                min_value=1900,
                max_value=2200,
                value=default_start_year,
                step=1,
                key="forecast_start_year",
            )
        )


        # ------------------------------------------------------
        # EXTRA FEATURE VALUES
        # ------------------------------------------------------

        forecast_extra_values = {}


        if extra_features:

            with st.expander(
                "Future values for additional numeric predictors"
            ):

                st.caption(
                    "These variables are held constant across the "
                    "forecast unless you regenerate the forecast with new values."
                )


                extra_columns = st.columns(
                    min(
                        4,
                        len(extra_features),
                    )
                )


                for index, feature in enumerate(
                    extra_features
                ):

                    series = pd.to_numeric(
                        df_clean[
                            feature
                        ],
                        errors="coerce",
                    )


                    median_value = (
                        float(
                            series.median()
                        )
                        if series.notna().any()
                        else 0.0
                    )


                    forecast_extra_values[
                        feature
                    ] = extra_columns[
                        index
                        % len(extra_columns)
                    ].number_input(
                        feature,
                        value=median_value,
                        key=f"forecast_extra_{feature}",
                    )


        # ------------------------------------------------------
        # GENERATE FORECAST
        # ------------------------------------------------------

        if st.button(
            "Generate Forecast",
            type="primary",
            key="generate_forecast_button",
        ):

            try:

                pipeline = (
                    st.session_state.trained_models[
                        forecast_model
                    ]
                )


                forecast_records = []


                current_month = int(
                    start_month
                )

                current_year = int(
                    start_year
                )


                for step in range(
                    1,
                    int(
                        forecast_steps
                    )
                    + 1,
                ):

                    current_month, current_year = (
                        advance_month(
                            current_month,
                            current_year,
                        )
                    )


                    row = {}


                    if commodity_col != "(none)":

                        row[
                            commodity_col
                        ] = forecast_commodity


                    if market_col != "(none)":

                        row[
                            market_col
                        ] = forecast_market


                    row[
                        month_col
                    ] = current_month


                    row[
                        year_col
                    ] = current_year


                    for feature in extra_features:

                        row[
                            feature
                        ] = forecast_extra_values.get(
                            feature,
                            np.nan,
                        )


                    for feature in feature_cols:

                        if feature not in row:

                            row[
                                feature
                            ] = np.nan


                    prediction_input = (
                        pd.DataFrame(
                            [row]
                        )[
                            feature_cols
                        ]
                    )


                    for col in categorical_cols:

                        if col in prediction_input.columns:

                            prediction_input[col] = (
                                prediction_input[
                                    col
                                ]
                                .astype(
                                    "string"
                                )
                                .str.strip()
                                .str.lower()
                            )


                    for col in numeric_cols:

                        if col in prediction_input.columns:

                            prediction_input[col] = (
                                pd.to_numeric(
                                    prediction_input[
                                        col
                                    ],
                                    errors="coerce",
                                )
                            )


                    predicted_price = float(
                        pipeline.predict(
                            prediction_input
                        )[0]
                    )


                    predicted_price = max(
                        0,
                        predicted_price,
                    )


                    forecast_records.append(
                        {
                            "Step": step,
                            "Year": current_year,
                            "Month": MONTH_NAMES[
                                current_month
                            ],
                            "Month Number": current_month,
                            "Predicted Price (USD)": predicted_price,
                        }
                    )


                forecast_df = pd.DataFrame(
                    forecast_records
                )


                # --------------------------------------------------
                # PRICE CHANGES
                # --------------------------------------------------

                forecast_df[
                    "Change (USD)"
                ] = (
                    forecast_df[
                        "Predicted Price (USD)"
                    ]
                    .diff()
                    .fillna(0)
                )


                forecast_df[
                    "Change (%)"
                ] = (
                    forecast_df[
                        "Predicted Price (USD)"
                    ]
                    .pct_change()
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .fillna(0)
                    * 100
                )


                st.markdown(
                    f"### Forecast — {forecast_commodity or 'Commodity'}"
                )


                st.caption(
                    f"Model: {forecast_model}"
                )


                forecast_display = (
                    forecast_df.copy()
                )


                st.dataframe(
                    forecast_display.style.format(
                        {
                            "Predicted Price (USD)": "USD {:,.4f}",
                            "Change (USD)": "{:+,.4f}",
                            "Change (%)": "{:+.2f}%",
                        }
                    ),
                    use_container_width=True,
                    height=min(
                        600,
                        80
                        + len(
                            forecast_df
                        )
                        * 35,
                    ),
                )


                # --------------------------------------------------
                # FORECAST CHART
                # --------------------------------------------------

                chart_df = (
                    forecast_df.copy()
                )


                chart_df[
                    "Period"
                ] = (
                    chart_df[
                        "Month"
                    ]
                    + " "
                    + chart_df[
                        "Year"
                    ].astype(str)
                )


                fig, ax = plt.subplots(
                    figsize=(11, 5)
                )


                ax.plot(
                    chart_df[
                        "Period"
                    ],
                    chart_df[
                        "Predicted Price (USD)"
                    ],
                    marker="o",
                )


                ax.set_xlabel(
                    "Forecast Period"
                )

                ax.set_ylabel(
                    "Predicted Price (USD)"
                )

                ax.set_title(
                    "Future Commodity Price Forecast"
                )


                ax.tick_params(
                    axis="x",
                    rotation=45,
                )


                fig.tight_layout()


                st.pyplot(
                    fig
                )


                plt.close(
                    fig
                )


                # --------------------------------------------------
                # FORECAST SUMMARY
                # --------------------------------------------------

                prices = forecast_df[
                    "Predicted Price (USD)"
                ]


                forecast_min = float(
                    prices.min()
                )

                forecast_max = float(
                    prices.max()
                )

                forecast_mean = float(
                    prices.mean()
                )


                if len(prices) > 1:

                    total_change = (
                        prices.iloc[-1]
                        - prices.iloc[0]
                    )

                    total_pct = (
                        (
                            total_change
                            / abs(
                                prices.iloc[0]
                            )
                        )
                        * 100
                        if prices.iloc[0]
                        != 0
                        else 0
                    )

                else:

                    total_change = 0
                    total_pct = 0


                forecast1, forecast2, forecast3, forecast4 = (
                    st.columns(4)
                )


                forecast1.metric(
                    "Forecast Minimum",
                    f"USD {forecast_min:,.4f}",
                )


                forecast2.metric(
                    "Forecast Maximum",
                    f"USD {forecast_max:,.4f}",
                )


                forecast3.metric(
                    "Forecast Mean",
                    f"USD {forecast_mean:,.4f}",
                )


                forecast4.metric(
                    "Total Change",
                    f"USD {total_change:+,.4f}",
                    delta=f"{total_pct:+.2f}%",
                )


                # --------------------------------------------------
                # DOWNLOAD
                # --------------------------------------------------

                forecast_csv = (
                    forecast_df
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )


                st.download_button(
                    "Download Forecast CSV",
                    forecast_csv,
                    "pricesight_forecast.csv",
                    "text/csv",
                )


                st.info(
                    "This forecast is an ML scenario projection: future Month "
                    "and Year values are supplied to the trained regression model. "
                    "It should not be interpreted as a classical ARIMA/Prophet-style "
                    "time-series forecast."
                )


            except Exception as error:

                st.error(
                    f"Forecast failed: {error}"
                )

                st.exception(
                    error
                )


# ==============================================================
# FOOTER
# ==============================================================

st.divider()

st.caption(
    "PriceSight — Machine Learning Commodity Price Forecasting"
)
