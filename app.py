import hashlib
import io
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
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

st.set_page_config(
    page_title="Somalia Food Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REQUIRED_COLUMNS = {
    "Admin 1", "Admin 2", "Market Name", "Commodity", "Price Date",
    "Price", "Unit", "Currency", "Data Type",
}
FEATURES = ["Admin 2", "Market Name", "Commodity", "Month", "Year", "Currency"]
CATEGORICAL = ["Admin 2", "Market Name", "Commodity", "Currency"]
NUMERIC = ["Month", "Year"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--ink:#17213a;--muted:#68738a;--line:#dfe5ee;--bg:#f4f7fb;--blue:#1967d2;--green:#137a5b;}
html,body,[class*="css"]{font-family:'Inter',sans-serif}.stApp{background:var(--bg)}
[data-testid="stHeader"]{background:rgba(244,247,251,.88);backdrop-filter:blur(10px)}
.block-container{max-width:1180px;padding-top:1.4rem;padding-bottom:4rem}
.brand{display:flex;align-items:center;gap:12px;background:#fff;border:1px solid var(--line);border-radius:15px;padding:13px 17px;margin-bottom:26px;box-shadow:0 6px 22px rgba(24,39,75,.04)}
.brand-mark{width:42px;height:42px;border-radius:12px;background:#195bc2;color:#fff;display:grid;place-items:center;font-size:21px}.brand b{font-size:17px;color:var(--ink)}.brand small{display:block;color:var(--muted);font-size:12px;margin-top:2px}
.hero{display:flex;justify-content:space-between;align-items:end;gap:24px;margin:0 0 24px}.eyebrow{font-size:12px;letter-spacing:.14em;font-weight:800;color:var(--blue)}.hero h1{font-size:38px;line-height:1.12;letter-spacing:-.035em;color:var(--ink);margin:7px 0 10px}.hero p{color:var(--muted);font-size:16px;line-height:1.6;max-width:700px;margin:0}.rate-card{background:#eaf2ff;border:1px solid #cfe0fa;padding:15px 18px;border-radius:14px;min-width:235px}.rate-card span,.rate-card small{color:#5d6b82;font-size:12px}.rate-card b{display:block;color:var(--ink);font-size:15px;margin:3px 0}
.steps{display:grid;grid-template-columns:repeat(4,1fr);background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 22px;margin-bottom:20px}.step{display:flex;align-items:center;gap:9px;color:#8b94a6;font-weight:700;font-size:14px}.step i{font-style:normal;width:26px;height:26px;border-radius:50%;background:#edf0f5;display:grid;place-items:center;font-size:12px}.step.on{color:var(--green)}.step.on i{background:#ddf5ec}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:var(--line)!important;border-radius:16px!important;box-shadow:0 8px 24px rgba(24,39,75,.035)}
.section-head{display:flex;align-items:center;gap:11px;margin-bottom:12px}.section-icon{width:40px;height:40px;border-radius:11px;background:#eaf2ff;color:var(--blue);display:grid;place-items:center;font-size:19px}.section-head h2{font-size:17px;color:var(--ink);margin:0}.section-head p{font-size:13px;color:var(--muted);margin:3px 0 0}
.success-note{display:flex;gap:9px;background:#f0faf6;border:1px solid #cfeadd;color:var(--green);padding:11px 12px;border-radius:10px;font-size:13px;margin:10px 0}.success-note b{display:block}.success-note span{display:block;color:#5c786e;font-size:12px;margin-top:2px}
.best-note{background:#eef5ff;border:1px solid #d4e4fa;border-radius:10px;padding:12px;color:#24466f;font-size:13px;margin-top:10px}.best-note b{color:#1967d2}.best-note span{color:#5e7188}
.result{background:linear-gradient(105deg,#102b53,#184d79);color:#fff;border-radius:14px;padding:22px 24px;margin-top:16px}.result small{font-size:11px;letter-spacing:.12em;color:#b9d2eb;font-weight:800}.result strong{display:block;font-size:34px;margin:4px 0}.result strong span{font-size:14px;color:#c5d8e9}.result p{color:#c5d8e9;margin:0;font-size:13px}
.disclaimer{font-size:12px;color:#798398;line-height:1.55;margin-top:14px}
div.stButton>button{border-radius:10px;font-weight:700;min-height:43px}div.stButton>button[kind="primary"]{background:#1967d2;border-color:#1967d2}.stDownloadButton>button{border-radius:10px;font-weight:700}
[data-testid="stFileUploaderDropzone"]{background:#f9fbfe;border:1.5px dashed #b8c4d6;border-radius:13px}.stDataFrame{border:1px solid #e3e8f0;border-radius:10px;overflow:hidden}
@media(max-width:800px){.hero{display:block}.rate-card{display:none}.steps{padding:12px}.step{font-size:12px}.hero h1{font-size:30px}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""", unsafe_allow_html=True)


def section_header(icon, title, subtitle):
    st.markdown(
        f'<div class="section-head"><span class="section-icon">{icon}</span>'
        f'<div><h2>{title}</h2><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def success_note(title, detail):
    st.markdown(
        f'<div class="success-note">✓<div><b>{title}</b><span>{detail}</span></div></div>',
        unsafe_allow_html=True,
    )


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def clean_data(raw, sos_rate, sls_rate):
    data = raw.copy()
    data.columns = data.columns.str.strip()
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

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

    bounds = data.groupby("Commodity")["Price_USD"].quantile([0.01, 0.99]).unstack()
    bounds.columns = ["lower", "upper"]
    data = data.join(bounds, on="Commodity")
    group_size = data.groupby("Commodity")["Commodity"].transform("size")
    keep = (group_size < 10) | data["Price_USD"].between(data["lower"], data["upper"])
    data = data.loc[keep].drop(columns=["lower", "upper"]).reset_index(drop=True)
    if len(data) < 50:
        raise ValueError("Fewer than 50 usable historical 1-kg observations remain after cleaning.")
    return data


def build_preprocessor():
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot_encoder()),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL),
        ("numeric", numeric, NUMERIC),
    ])


def estimators():
    return {
        "XGBoost": XGBRegressor(
            n_estimators=350, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        ),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        "Support Vector Machine": SVR(C=20, epsilon=0.05, gamma="scale"),
    }


@st.cache_resource(show_spinner=False)
def train_models(cleaned_csv):
    data = pd.read_csv(io.StringIO(cleaned_csv), parse_dates=["Price Date"])
    data = data.sort_values("Price Date").reset_index(drop=True)
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    if train.empty or test.empty:
        raise ValueError("The dataset is too small to create chronological train and test sets.")

    x_train, y_train = train[FEATURES], train["Price_USD"]
    x_test, y_test = test[FEATURES], test["Price_USD"]
    rows = []
    models = estimators()
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", clone(estimator))])
        pipeline.fit(x_train, y_train)
        prediction = np.maximum(pipeline.predict(x_test), 0)
        nonzero = y_test != 0
        rows.append({
            "Algorithm": name,
            "MAE (USD)": mean_absolute_error(y_test, prediction),
            "RMSE (USD)": mean_squared_error(y_test, prediction) ** 0.5,
            "R²": r2_score(y_test, prediction),
            "MAPE (%)": np.mean(np.abs((y_test[nonzero] - prediction[nonzero]) / y_test[nonzero])) * 100,
        })

    metrics = pd.DataFrame(rows).sort_values("RMSE (USD)").reset_index(drop=True)
    best_name = metrics.loc[0, "Algorithm"]
    best_model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", clone(models[best_name])),
    ])
    best_model.fit(data[FEATURES], data["Price_USD"])
    return metrics, best_name, best_model


st.markdown('<div class="brand"><span class="brand-mark">⌁</span><div><b>FoodPrice ML</b><small>Somalia market intelligence</small></div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Currency conversion")
    st.caption("Update these assumptions when exchange rates change.")
    sos_rate = st.number_input("SOS per USD", min_value=1.0, value=26000.0, step=500.0)
    sls_rate = st.number_input("SLS per USD", min_value=1.0, value=10000.0, step=500.0)

st.markdown(
    f'<div class="hero"><div><span class="eyebrow">WFP FOOD PRICE DATA</span>'
    '<h1>Predict commodity prices with confidence.</h1>'
    '<p>Upload market data, compare four regression algorithms, and estimate the USD price of one kilogram—all in one workspace.</p></div>'
    f'<div class="rate-card"><span>Currency assumption</span><b>1 USD = {sos_rate:,.0f} SOS</b><small>Change it from the sidebar</small></div></div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload the WFP food-price CSV", type=["csv"], label_visibility="collapsed")
stage = 0
cleaned = None
signature = None

if uploaded is not None:
    stage = 1
    file_bytes = uploaded.getvalue()
    signature = hashlib.sha256(file_bytes + f"{sos_rate}:{sls_rate}".encode()).hexdigest()
    if st.session_state.get("dataset_signature") != signature:
        st.session_state["dataset_signature"] = signature
        st.session_state["train_requested"] = False
        st.session_state.pop("prediction", None)
    try:
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
        cleaned = clean_data(raw_df, sos_rate, sls_rate)
        stage = 2
    except Exception as exc:
        st.error(f"The data could not be prepared: {exc}")

if st.session_state.get("train_requested") and cleaned is not None:
    stage = 3
if st.session_state.get("prediction") is not None:
    stage = 4

st.markdown(
    '<div class="steps">' + ''.join(
        f'<div class="step {"on" if stage >= i else ""}"><i>{"✓" if stage >= i else i}</i>{label}</div>'
        for i, label in enumerate(["Upload", "Preprocess", "Train", "Predict"], start=1)
    ) + '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.88, 1.32], gap="large")
with left:
    with st.container(border=True):
        section_header("▣", "Dataset", "WFP food-price CSV")
        if uploaded is None:
            st.info("Choose a CSV file above to begin.")
        elif cleaned is not None:
            success_note("Data uploaded successfully", f"{len(raw_df):,} rows received.")
            success_note("Data preprocessed successfully", f"{len(cleaned):,} valid historical 1-kg records retained.")
            with st.expander("Preview cleaned data"):
                st.dataframe(cleaned.head(200), use_container_width=True, hide_index=True)
            if st.button("▶  Train & compare models", type="primary", use_container_width=True):
                st.session_state["train_requested"] = True
                st.session_state.pop("prediction", None)
                st.rerun()

with right:
    with st.container(border=True):
        section_header("▥", "Model comparison", "Chronological 20% test set")
        if cleaned is None or not st.session_state.get("train_requested"):
            st.info("Upload and preprocess data, then train the four algorithms.")
        else:
            try:
                with st.spinner("Training XGBoost, Linear Regression, Random Forest, and SVM…"):
                    metrics, best_name, best_model = train_models(cleaned.to_csv(index=False))
                success_note("Models trained successfully", f"{best_name} achieved the lowest test RMSE.")
                formatted = metrics.copy()
                formatted["MAE (USD)"] = formatted["MAE (USD)"].map(lambda x: f"${x:,.4f}")
                formatted["RMSE (USD)"] = formatted["RMSE (USD)"].map(lambda x: f"${x:,.4f}")
                formatted["R²"] = formatted["R²"].map(lambda x: f"{x:,.4f}")
                formatted["MAPE (%)"] = formatted["MAPE (%)"].map(lambda x: f"{x:,.2f}%")
                st.dataframe(formatted, use_container_width=True, hide_index=True)
                st.markdown(f'<div class="best-note"><b>✓ {best_name} selected</b><br><span>Lowest RMSE on unseen chronological data; retrained on all cleaned observations.</span></div>', unsafe_allow_html=True)
                model_bytes = io.BytesIO()
                joblib.dump(best_model, model_bytes)
                st.download_button("Download best trained model", model_bytes.getvalue(), "best_food_price_model.joblib", "application/octet-stream", use_container_width=True)
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
                best_model = None

if cleaned is not None and st.session_state.get("train_requested") and "best_model" in locals() and best_model is not None:
    with st.container(border=True):
        section_header("↗", "Price prediction", "Select market conditions for a 1 kg estimate.")
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.35, 1, 0.8])
        with c1:
            town = st.selectbox("Town", sorted(cleaned["Admin 2"].unique()))
        town_data = cleaned[cleaned["Admin 2"] == town]
        with c2:
            market = st.selectbox("Market", sorted(town_data["Market Name"].unique()))
        market_data = town_data[town_data["Market Name"] == market]
        with c3:
            commodity = st.selectbox("Commodity", sorted(market_data["Commodity"].unique()))
        with c4:
            month = st.selectbox("Month", range(1, 13), format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
        min_year, max_year = int(cleaned["Year"].min()), int(cleaned["Year"].max())
        with c5:
            year = st.number_input("Year", min_value=min_year, max_value=max_year + 10, value=max_year + 1, step=1)

        currency_mode = market_data[market_data["Commodity"] == commodity]["Currency"].mode()
        currency = currency_mode.iloc[0] if not currency_mode.empty else cleaned["Currency"].mode().iloc[0]
        if st.button("Predict price", type="primary", use_container_width=True):
            input_row = pd.DataFrame([{
                "Admin 2": town, "Market Name": market, "Commodity": commodity,
                "Month": int(month), "Year": int(year), "Currency": currency,
            }])
            predicted = max(float(best_model.predict(input_row)[0]), 0.0)
            st.session_state["prediction"] = {
                "price": predicted, "commodity": commodity, "market": market,
                "month": pd.Timestamp(2000, month, 1).strftime("%B"), "year": int(year),
            }
            st.rerun()

        result = st.session_state.get("prediction")
        if result:
            st.markdown(
                f'<div class="result"><small>PREDICTED RETAIL PRICE</small>'
                f'<strong>${result["price"]:,.2f} <span>USD / kg</span></strong>'
                f'<p>{result["commodity"]} · {result["market"]} · {result["month"]} {result["year"]}</p></div>',
                unsafe_allow_html=True,
            )
            st.success("Prediction made successfully.")
            if result["year"] > max_year:
                st.warning("This year is beyond the historical data range. Uncertainty increases farther into the future.")

st.markdown('<p class="disclaimer">Predictions are statistical estimates based on the uploaded historical data and exchange-rate assumptions. They should support—not replace—market monitoring and professional judgment.</p>', unsafe_allow_html=True)
