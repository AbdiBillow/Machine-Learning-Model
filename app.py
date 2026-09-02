
import io
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Somalia Food Price Predictor", page_icon="📈", layout="wide")
st.title("Somalia Food Price Prediction")
st.caption("Compare four regression algorithms and predict the USD price of 1 kg of a commodity.")

REQUIRED_COLUMNS = {
    "Admin 1", "Admin 2", "Market Name", "Commodity", "Price Date",
    "Price", "Unit", "Currency", "Data Type"
}
FEATURES = ["Admin 2", "Market Name", "Commodity", "Month", "Year", "Currency"]
CATEGORICAL = ["Admin 2", "Market Name", "Commodity", "Currency"]
NUMERIC = ["Month", "Year"]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_data(raw, sos_rate, sls_rate):
    data = raw.copy()
    data.columns = data.columns.str.strip()
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()
        data[col] = data[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    data["Price Date"] = pd.to_datetime(data["Price Date"], dayfirst=True, errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.drop_duplicates()
    data = data[data["Unit"].str.upper().eq("KG")]
    data = data[data["Data Type"].str.casefold().eq("aggregated")]
    data = data.dropna(subset=["Admin 2", "Market Name", "Commodity", "Price Date", "Price", "Currency"])
    data = data[data["Price"] > 0]

    rates = {"SOS": float(sos_rate), "SLS": float(sls_rate), "USD": 1.0}
    data["USD Rate"] = data["Currency"].str.upper().map(rates)
    data = data.dropna(subset=["USD Rate"])
    data["Price_USD"] = data["Price"] / data["USD Rate"]
    data["Month"] = data["Price Date"].dt.month.astype(int)
    data["Year"] = data["Price Date"].dt.year.astype(int)

    # Remove extreme recording errors within each commodity while retaining genuine market variation.
    def trim_group(group):
        if len(group) < 10:
            return group
        low, high = group["Price_USD"].quantile([0.01, 0.99])
        return group[group["Price_USD"].between(low, high)]

    data = data.groupby("Commodity", group_keys=False).apply(trim_group).reset_index(drop=True)
    if len(data) < 50:
        raise ValueError("Fewer than 50 usable 1-kg historical observations remain after cleaning.")
    return data


def build_preprocessor():
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL),
        ("numeric", numeric, NUMERIC),
    ])


@st.cache_resource(show_spinner=False)
def train_models(cleaned_csv):
    data = pd.read_csv(io.StringIO(cleaned_csv))
    data["Price Date"] = pd.to_datetime(data["Price Date"])
    data = data.sort_values("Price Date").reset_index(drop=True)

    # Chronological holdout gives a more realistic estimate of future prediction quality.
    split = max(1, int(len(data) * 0.8))
    train = data.iloc[:split]
    test = data.iloc[split:]
    if test.empty:
        raise ValueError("The dataset is too small to create a test set.")

    x_train, y_train = train[FEATURES], train["Price_USD"]
    x_test, y_test = test[FEATURES], test["Price_USD"]
    algorithms = {
        "XGBoost": XGBRegressor(
            n_estimators=350, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=42, n_jobs=-1
        ),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Support Vector Machine": SVR(C=20, epsilon=0.05, gamma="scale"),
    }

    trained, rows = {}, []
    for name, estimator in algorithms.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        prediction = np.maximum(pipeline.predict(x_test), 0)
        mae = mean_absolute_error(y_test, prediction)
        rmse = mean_squared_error(y_test, prediction) ** 0.5
        r2 = r2_score(y_test, prediction)
        nonzero = y_test != 0
        mape = np.mean(np.abs((y_test[nonzero] - prediction[nonzero]) / y_test[nonzero])) * 100
        rows.append({"Algorithm": name, "MAE (USD)": mae, "RMSE (USD)": rmse,
                     "R²": r2, "MAPE (%)": mape})
        trained[name] = pipeline

    metrics = pd.DataFrame(rows).sort_values("RMSE (USD)").reset_index(drop=True)
    best_name = metrics.iloc[0]["Algorithm"]
    # Retrain the selected algorithm on every cleaned historical record.
    best_estimator = algorithms[best_name]
    best_model = Pipeline([("preprocessor", build_preprocessor()), ("model", best_estimator)])
    best_model.fit(data[FEATURES], data["Price_USD"])
    return trained, metrics, best_name, best_model


with st.sidebar:
    st.header("Currency conversion")
    st.info("Edit these rates when the market exchange rate changes.")
    sos_rate = st.number_input("SOS per USD", min_value=1.0, value=26000.0, step=500.0)
    sls_rate = st.number_input("SLS per USD", min_value=1.0, value=10000.0, step=500.0)

uploaded = st.file_uploader("Upload the WFP food-price CSV", type=["csv"])

if uploaded is None:
    st.info("Upload the WFP CSV file to begin.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
    st.success(f"Data uploaded successfully — {len(raw_df):,} rows received.")
    cleaned = clean_data(raw_df, sos_rate, sls_rate)
    st.success(f"Data preprocessed successfully — {len(cleaned):,} valid 1-kg historical records retained.")
except Exception as exc:
    st.error(f"The data could not be prepared: {exc}")
    st.stop()

with st.expander("View cleaned data", expanded=False):
    st.dataframe(cleaned.head(200), use_container_width=True)

if st.button("Train and compare models", type="primary", use_container_width=True):
    st.session_state["train_requested"] = True

if not st.session_state.get("train_requested"):
    st.info("Select **Train and compare models** to evaluate all four algorithms.")
    st.stop()

try:
    with st.spinner("Training XGBoost, Linear Regression, Random Forest, and SVM..."):
        cleaned_csv = cleaned.to_csv(index=False)
        _, metrics, best_name, best_model = train_models(cleaned_csv)
    st.success(f"Models trained successfully. Best model: {best_name} (lowest test RMSE).")
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

st.subheader("Algorithm comparison")
display_metrics = metrics.copy()
for col in ["MAE (USD)", "RMSE (USD)", "R²", "MAPE (%)"]:
    display_metrics[col] = display_metrics[col].map(lambda x: f"{x:,.4f}")
st.dataframe(display_metrics, use_container_width=True, hide_index=True)

model_bytes = io.BytesIO()
joblib.dump(best_model, model_bytes)
st.download_button("Download best trained model", model_bytes.getvalue(),
                   file_name="best_food_price_model.joblib", mime="application/octet-stream")

st.subheader("Predict the USD price of 1 kg")
left, right = st.columns(2)
with left:
    town = st.selectbox("Town", sorted(cleaned["Admin 2"].unique()))
    town_data = cleaned[cleaned["Admin 2"] == town]
    market = st.selectbox("Market", sorted(town_data["Market Name"].unique()))
    market_data = town_data[town_data["Market Name"] == market]
    commodity = st.selectbox("Commodity", sorted(market_data["Commodity"].unique()))
with right:
    month = st.selectbox("Month", range(1, 13), format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
    min_year = int(cleaned["Year"].min())
    max_year = int(cleaned["Year"].max())
    year = st.number_input("Year", min_value=min_year, max_value=max_year + 10,
                           value=max_year + 1, step=1)

currency_mode = market_data[market_data["Commodity"] == commodity]["Currency"].mode()
currency = currency_mode.iloc[0] if not currency_mode.empty else cleaned["Currency"].mode().iloc[0]

if st.button("Predict price", type="primary", use_container_width=True):
    input_row = pd.DataFrame([{
        "Admin 2": town, "Market Name": market, "Commodity": commodity,
        "Month": int(month), "Year": int(year), "Currency": currency,
    }])
    predicted = max(float(best_model.predict(input_row)[0]), 0.0)
    st.success("Prediction made successfully.")
    st.metric(f"Predicted price: 1 kg of {commodity}", f"${predicted:,.2f} USD")
    if int(year) > max_year:
        st.caption("This is an extrapolation beyond the latest year in the uploaded data; uncertainty increases farther into the future.")

