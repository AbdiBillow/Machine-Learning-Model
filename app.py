import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import io
import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PriceSight — Commodity Forecasting",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.metric-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    background: #f8fafc;
}
.metric-card .val { font-size: 1.4rem; font-weight: 700; color: #1e40af; }
.metric-card .lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; }
.pred-result {
    border: 2px solid #2563eb;
    border-radius: 12px;
    padding: 1.8rem;
    text-align: center;
    background: #eff6ff;
    margin: 1rem 0;
}
.pred-result .amount { font-size: 2.6rem; font-weight: 800; color: #1d4ed8; }
.pred-result .sublabel { font-size: 0.8rem; color: #64748b; margin-top: .4rem; }
.best-badge {
    display: inline-block;
    background: #dcfce7;
    color: #16a34a;
    border: 1px solid #86efac;
    border-radius: 20px;
    padding: .2rem .75rem;
    font-size: .75rem;
    font-weight: 600;
}
.warn-badge {
    display: inline-block;
    background: #fef9c3;
    color: #ca8a04;
    border: 1px solid #fde047;
    border-radius: 20px;
    padding: .2rem .75rem;
    font-size: .75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "trained_models":   {},
    "comparison_df":    None,
    "best_model_name":  None,
    "label_encoders":   {},
    "feature_cols":     [],
    "target_col":       None,
    "df_clean":         None,
    "df_raw":           None,
    "cat_cols":         [],
    "num_feature_cols": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_dataframe(df: pd.DataFrame, cat_cols: list, fit: bool = True) -> pd.DataFrame:
    df = df.copy()
    if fit:
        st.session_state.label_encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.lower()
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("unknown"))
            st.session_state.label_encoders[col] = le
        else:
            le = st.session_state.label_encoders.get(col)
            if le:
                val = df[col].values[0]
                if val in le.classes_:
                    df[col] = le.transform([val])
                else:
                    df[col] = le.transform([le.classes_[0]])
    return df


def build_pipeline(name: str) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="mean")),
             ("scaler",  StandardScaler())]
    if name == "Linear Regression":
        steps.append(("model", LinearRegression()))
    elif name == "Random Forest":
        steps.append(("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)))
    elif name == "XGBoost":
        steps.append(("model", XGBRegressor(n_estimators=200, random_state=42,
                                             verbosity=0, eval_metric="rmse")))
    elif name == "Support Vector Machine":
        steps.append(("model", SVR(kernel="rbf", C=10, epsilon=0.1)))
    return Pipeline(steps)


def evaluate_model(pipeline, X_train, X_test, y_train, y_test, name: str) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred       = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    r2_tr = r2_score(y_train, y_pred_train)
    try:
        cv = cross_val_score(pipeline, np.vstack([X_train, X_test]),
                             np.concatenate([y_train, y_test]),
                             cv=5, scoring="r2", n_jobs=-1)
        cv_mean = cv.mean()
    except Exception:
        cv_mean = np.nan
    return {
        "Model":        name,
        "MAE":          round(mae, 4),
        "RMSE":         round(rmse, 4),
        "R2 (Test)":    round(r2, 4),
        "R2 (Train)":   round(r2_tr, 4),
        "CV R2 (mean)": round(cv_mean, 4),
        "_pipeline":    pipeline,
    }


def determine_conclusion(row: pd.Series, best_name: str) -> str:
    name = row["Model"]
    r2   = row["R2 (Test)"]
    gap  = row["R2 (Train)"] - row["R2 (Test)"]
    if name == best_name:
        base = "Best overall model. "
    else:
        base = ""
    if r2 >= 0.85:
        quality = "Excellent fit."
    elif r2 >= 0.65:
        quality = "Good fit."
    elif r2 >= 0.40:
        quality = "Moderate fit."
    else:
        quality = "Poor fit — consider more data or features."
    overfit = " Possible overfitting." if gap > 0.20 else ""
    return base + quality + overfit


def style_comparison(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    display = df.drop(columns=["_pipeline"], errors="ignore")
    best_r2  = display["R2 (Test)"].max()

    def highlight(row):
        styles = [""] * len(row)
        if row["R2 (Test)"] == best_r2:
            styles = ["background-color: #f0fdf4; font-weight: 600"] * len(row)
        return styles

    return (
        display.style
        .apply(highlight, axis=1)
        .format({
            "MAE":          "{:.4f}",
            "RMSE":         "{:.4f}",
            "R2 (Test)":    "{:.4f}",
            "R2 (Train)":   "{:.4f}",
            "CV R2 (mean)": "{:.4f}",
        })
        .bar(subset=["R2 (Test)"], color="#bbf7d0", vmin=0, vmax=1)
        .bar(subset=["MAE"],       color="#fed7aa", vmin=0)
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("PriceSight")
    st.caption("Commodity Price Forecasting")
    st.divider()

    st.subheader("Data Configuration")
    test_size       = st.slider("Test Split (%)", 10, 40, 20) / 100
    drop_duplicates = st.checkbox("Drop duplicate rows", value=True)
    outlier_removal = st.checkbox("Remove outliers (IQR)", value=False)

    st.divider()
    st.subheader("Model Download")
    if st.session_state.best_model_name and st.session_state.trained_models:
        best_pipe = st.session_state.trained_models.get(st.session_state.best_model_name)
        if best_pipe:
            buf = io.BytesIO()
            joblib.dump({
                "pipeline":       best_pipe,
                "feature_cols":   st.session_state.feature_cols,
                "target_col":     st.session_state.target_col,
                "cat_cols":       st.session_state.cat_cols,
                "label_encoders": st.session_state.label_encoders,
                "best_model":     st.session_state.best_model_name,
            }, buf)
            buf.seek(0)
            st.download_button(
                f"Download Best Model ({st.session_state.best_model_name})",
                data=buf,
                file_name="pricesight_best_model.pkl",
                mime="application/octet-stream",
            )

    uploaded_model = st.file_uploader("Load Saved Model (.pkl)", type=["pkl"])
    if uploaded_model and st.button("Apply Loaded Model"):
        try:
            data = joblib.load(uploaded_model)
            st.session_state.trained_models  = {data["best_model"]: data["pipeline"]}
            st.session_state.feature_cols    = data["feature_cols"]
            st.session_state.target_col      = data["target_col"]
            st.session_state.cat_cols        = data["cat_cols"]
            st.session_state.label_encoders  = data["label_encoders"]
            st.session_state.best_model_name = data["best_model"]
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("PriceSight — Commodity Price Forecasting")
st.caption("Upload your commodity price dataset, train all four models, compare performance, and predict prices.")

if not XGBOOST_AVAILABLE:
    st.warning("XGBoost is not installed. Run `pip install xgboost` to enable it.")

st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your commodity price CSV",
    type=["csv"],
    help="CSV should contain columns for commodity, market, month, year, and a numeric price column.",
)

if not uploaded_file:
    st.info(
        "Upload a CSV file to get started. "
        "Expected columns: commodity name, market/region, month, year, and a price column."
    )
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if drop_duplicates:
    df_raw = df_raw.drop_duplicates()
df_raw.reset_index(drop=True, inplace=True)
st.session_state.df_raw = df_raw

# ── Column Mapping ────────────────────────────────────────────────────────────
st.subheader("Step 1 — Map Your Columns")
st.caption("Tell the app which columns represent commodity, market, month, year, and price.")

all_cols  = df_raw.columns.tolist()
num_cols  = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_raw[c])]
cat_cols_available = [c for c in all_cols if df_raw[c].dtype == object
                      or df_raw[c].nunique() < 50]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    commodity_col = st.selectbox("Commodity Column", ["(none)"] + all_cols)
with c2:
    market_col = st.selectbox("Market Column", ["(none)"] + all_cols)
with c3:
    month_col = st.selectbox("Month Column", ["(none)"] + all_cols)
with c4:
    year_col = st.selectbox("Year Column", ["(none)"] + all_cols)
with c5:
    target_col = st.selectbox("Price Column (Target)", num_cols if num_cols else all_cols)

# Build feature lists
mapped_cat   = [c for c in [commodity_col, market_col, month_col, year_col]
                if c != "(none)" and c in all_cols]
mapped_num   = [c for c in num_cols
                if c != target_col and c not in mapped_cat]

extra_features = st.multiselect(
    "Additional numeric features (optional)",
    options=mapped_num,
    default=[],
    help="Add any extra numeric columns as model inputs.",
)

all_features = mapped_cat + extra_features

if not all_features:
    st.warning("Map at least one column (commodity, market, month, or year) to continue.")
    st.stop()

# ── Identify categorical vs numeric among features ────────────────────────────
cat_cols  = [c for c in mapped_cat if df_raw[c].dtype == object
             or df_raw[c].nunique() < 80]
num_feats = [c for c in all_features if c not in cat_cols]

st.session_state.feature_cols   = all_features
st.session_state.target_col     = target_col
st.session_state.cat_cols       = cat_cols

# ── Prepare data ──────────────────────────────────────────────────────────────
df_work = df_raw[all_features + [target_col]].copy()
df_work = df_work.dropna(subset=[target_col])

# Encode categoricals
df_encoded = encode_dataframe(df_work, cat_cols, fit=True)
df_encoded[all_features] = df_encoded[all_features].apply(pd.to_numeric, errors="coerce")
df_encoded = df_encoded.dropna(subset=all_features)

if outlier_removal:
    for col in all_features + [target_col]:
        q1, q3 = df_encoded[col].quantile(0.25), df_encoded[col].quantile(0.75)
        iqr = q3 - q1
        df_encoded = df_encoded[
            (df_encoded[col] >= q1 - 1.5 * iqr) &
            (df_encoded[col] <= q3 + 1.5 * iqr)
        ]

df_encoded.reset_index(drop=True, inplace=True)
st.session_state.df_clean = df_encoded

if len(df_encoded) < 20:
    st.error(f"Only {len(df_encoded)} usable rows after cleaning. Need at least 20.")
    st.stop()

# ── Summary row ───────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
s1.markdown(f'<div class="metric-card"><div class="val">{len(df_raw):,}</div>'
            f'<div class="lbl">Total Rows</div></div>', unsafe_allow_html=True)
s2.markdown(f'<div class="metric-card"><div class="val">{len(df_encoded):,}</div>'
            f'<div class="lbl">Clean Rows</div></div>', unsafe_allow_html=True)
s3.markdown(f'<div class="metric-card"><div class="val">{len(all_features)}</div>'
            f'<div class="lbl">Features</div></div>', unsafe_allow_html=True)
missing_pct = df_raw[all_features + [target_col]].isnull().mean().mean() * 100
s4.markdown(f'<div class="metric-card"><div class="val">{missing_pct:.1f}%</div>'
            f'<div class="lbl">Missing Data</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_train, tab_predict, tab_forecast = st.tabs([
    "Data Preview", "Train & Compare Models", "Price Prediction", "Future Forecast"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — DATA PREVIEW
# ════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df_work.head(100), use_container_width=True, height=320)

    with st.expander("Descriptive Statistics"):
        st.dataframe(
            df_work.describe().T.style.format("{:.3f}"),
            use_container_width=True,
        )

    with st.expander("Missing Values per Column"):
        mv = df_raw[all_features + [target_col]].isnull().sum().reset_index()
        mv.columns = ["Column", "Missing Count"]
        mv["Missing %"] = (mv["Missing Count"] / len(df_raw) * 100).round(2)
        st.dataframe(mv, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN & COMPARE
# ════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Step 2 — Train All Models & Compare")
    st.caption(
        "All four algorithms are trained simultaneously on the same train/test split. "
        "Results are shown in a comparison table ranked by Test R²."
    )

    MODELS = ["Linear Regression", "Random Forest", "XGBoost", "Support Vector Machine"]
    if not XGBOOST_AVAILABLE:
        MODELS = [m for m in MODELS if m != "XGBoost"]

    st.markdown("**Models that will be trained:**")
    badge_html = " &nbsp; ".join(
        f'<span style="background:#e0e7ff;color:#3730a3;border-radius:6px;'
        f'padding:.2rem .7rem;font-size:.8rem;font-weight:600">{m}</span>'
        for m in MODELS
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Train All Models", type="primary"):
        X = df_encoded[all_features].values
        y = df_encoded[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        results  = []
        trained  = {}
        progress = st.progress(0, text="Training models...")

        for i, name in enumerate(MODELS):
            progress.progress((i) / len(MODELS), text=f"Training {name} ...")
            try:
                pipe   = build_pipeline(name)
                result = evaluate_model(pipe, X_train, X_test, y_train, y_test, name)
                results.append(result)
                trained[name] = result["_pipeline"]
            except Exception as e:
                st.warning(f"{name} failed: {e}")
            progress.progress((i + 1) / len(MODELS), text=f"Done: {name}")

        progress.empty()

        if results:
            comp_df = pd.DataFrame(results)
            best_idx  = comp_df["R2 (Test)"].idxmax()
            best_name = comp_df.loc[best_idx, "Model"]
            comp_df["Conclusion"] = comp_df.apply(
                lambda row: determine_conclusion(row, best_name), axis=1
            )

            st.session_state.trained_models  = trained
            st.session_state.comparison_df   = comp_df
            st.session_state.best_model_name = best_name

            st.success(f"All models trained. Best model: **{best_name}** "
                       f"(R² = {comp_df.loc[best_idx, 'R2 (Test)']:.4f})")

    if st.session_state.comparison_df is not None:
        st.markdown("### Model Comparison Table")
        st.caption("Rows highlighted in green = best-performing model by Test R².")

        comp = st.session_state.comparison_df.copy()
        best = st.session_state.best_model_name

        # Add visual best marker
        comp.insert(1, "Rank", "")
        comp.loc[comp["Model"] == best, "Rank"] = "Best"

        display_cols = ["Model", "Rank", "MAE", "RMSE",
                        "R2 (Test)", "R2 (Train)", "CV R2 (mean)", "Conclusion"]
        disp = comp[display_cols].copy()

        def highlight_best(row):
            if row["Rank"] == "Best":
                return ["background-color: #f0fdf4; font-weight: 600"] * len(row)
            return [""] * len(row)

        styled = (
            disp.style
            .apply(highlight_best, axis=1)
            .format({
                "MAE":          "{:.4f}",
                "RMSE":         "{:.4f}",
                "R2 (Test)":    "{:.4f}",
                "R2 (Train)":   "{:.4f}",
                "CV R2 (mean)": "{:.4f}",
            })
            .bar(subset=["R2 (Test)"], color="#bbf7d0", vmin=0, vmax=1)
            .bar(subset=["MAE"],       color="#fecaca", vmin=0)
        )
        st.dataframe(styled, use_container_width=True, height=220)

        # Metric cards for best model
        best_row = comp[comp["Model"] == best].iloc[0]
        st.markdown(f"#### Best Model: {best}")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="val">{best_row["MAE"]:.4f}</div>'
                    f'<div class="lbl">MAE</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="val">{best_row["RMSE"]:.4f}</div>'
                    f'<div class="lbl">RMSE</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="val">{best_row["R2 (Test)"]:.4f}</div>'
                    f'<div class="lbl">R2 (Test)</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="val">{best_row["CV R2 (mean)"]:.4f}</div>'
                    f'<div class="lbl">CV R2</div></div>', unsafe_allow_html=True)

        with st.expander("Metric Definitions"):
            st.markdown("""
| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average absolute difference between actual and predicted |
| **RMSE** | Root Mean Squared Error — penalises large errors more than MAE |
| **R2 (Test)** | Coefficient of determination on the unseen test set (1.0 = perfect) |
| **R2 (Train)** | R2 on training data — large gap with Test R2 suggests overfitting |
| **CV R2 (mean)** | 5-fold cross-validated R2 — more robust generalization estimate |
""")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE PREDICTION
# ════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Step 3 — Predict Commodity Price (USD)")

    if not st.session_state.trained_models:
        st.info("Train the models first in the 'Train & Compare Models' tab.")
        st.stop()

    st.caption(
        "Select the commodity, market, month, and year. "
        "The system will use the best-performing model to predict the price in USD."
    )

    # ── Dynamic dropdowns from actual data ───────────────────────────
    def get_unique(col):
        if col == "(none)" or col not in df_raw.columns:
            return []
        return sorted(df_raw[col].dropna().astype(str).str.strip().unique().tolist())

    p1, p2 = st.columns(2)
    p3, p4 = st.columns(2)

    commodity_options = get_unique(commodity_col)
    market_options    = get_unique(market_col)
    month_options     = get_unique(month_col)
    year_options      = get_unique(year_col)

    sel_commodity = p1.selectbox(
        "Commodity",
        commodity_options if commodity_options else ["N/A"],
        help="Select the commodity whose price you want to predict."
    ) if commodity_col != "(none)" else None

    sel_market = p2.selectbox(
        "Market / Region",
        market_options if market_options else ["N/A"],
        help="Select the market or region."
    ) if market_col != "(none)" else None

    sel_month = p3.selectbox(
        "Month",
        month_options if month_options else ["N/A"],
        help="Select the month."
    ) if month_col != "(none)" else None

    sel_year = p4.selectbox(
        "Year",
        year_options if year_options else ["N/A"],
        help="Select the year."
    ) if year_col != "(none)" else None

    # Extra numeric inputs
    extra_inputs = {}
    if extra_features:
        st.markdown("**Additional Feature Inputs:**")
        ex_cols = st.columns(min(4, len(extra_features)))
        for i, ef in enumerate(extra_features):
            col_med = float(df_encoded[ef].median())
            extra_inputs[ef] = ex_cols[i % len(ex_cols)].number_input(
                ef, value=col_med, format="%.4f", key=f"extra_{ef}"
            )

    # Model selector for prediction
    model_choices = list(st.session_state.trained_models.keys())
    sel_model = st.selectbox(
        "Model to use for prediction",
        model_choices,
        index=model_choices.index(st.session_state.best_model_name)
        if st.session_state.best_model_name in model_choices else 0,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Price", type="primary"):
        # Build input row
        input_dict = {}
        if commodity_col != "(none)" and sel_commodity:
            input_dict[commodity_col] = sel_commodity
        if market_col != "(none)" and sel_market:
            input_dict[market_col] = sel_market
        if month_col != "(none)" and sel_month:
            input_dict[month_col] = sel_month
        if year_col != "(none)" and sel_year:
            input_dict[year_col] = sel_year
        input_dict.update(extra_inputs)

        try:
            input_df = pd.DataFrame([input_dict])

            # Encode categoricals
            for col in cat_cols:
                if col in input_df.columns:
                    le = st.session_state.label_encoders.get(col)
                    if le:
                        val = str(input_df[col].values[0]).strip().lower()
                        if val in le.classes_:
                            input_df[col] = le.transform([val])
                        else:
                            st.warning(
                                f"'{val}' was not seen during training for column '{col}'. "
                                "Using closest known value."
                            )
                            input_df[col] = le.transform([le.classes_[0]])

            input_df[all_features] = input_df[all_features].apply(pd.to_numeric, errors="coerce")
            input_df = input_df[all_features]

            pipeline   = st.session_state.trained_models[sel_model]
            prediction = pipeline.predict(input_df)[0]
            prediction = max(prediction, 0)  # prices cannot be negative

            # CI from residuals
            comp_df   = st.session_state.comparison_df
            model_row = comp_df[comp_df["Model"] == sel_model]
            rmse_val  = float(model_row["RMSE"].values[0]) if len(model_row) else 0
            lo = max(0, prediction - 1.96 * rmse_val)
            hi = prediction + 1.96 * rmse_val

            is_best = sel_model == st.session_state.best_model_name
            badge   = ('<span class="best-badge">Best Model</span>'
                       if is_best
                       else '<span class="warn-badge">Not Best Model</span>')

            # Commodity and market label for display
            pred_label_parts = []
            if sel_commodity:
                pred_label_parts.append(sel_commodity)
            if sel_market:
                pred_label_parts.append(sel_market)
            if sel_month:
                pred_label_parts.append(str(sel_month))
            if sel_year:
                pred_label_parts.append(str(sel_year))
            pred_label = " | ".join(pred_label_parts) if pred_label_parts else "Selected inputs"

            st.markdown(f"""
            <div class="pred-result">
                <div style="font-size:.8rem;color:#64748b;margin-bottom:.5rem">{pred_label}</div>
                <div class="amount">USD {prediction:,.4f}</div>
                <div class="sublabel">
                    95% CI: USD {lo:,.4f} &mdash; USD {hi:,.4f}
                    &nbsp;&nbsp;|&nbsp;&nbsp; Model: <strong>{sel_model}</strong>
                    &nbsp; {badge}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Context: how does this compare to the dataset's price range?
            p_min = float(df_work[target_col].min())
            p_max = float(df_work[target_col].max())
            p_med = float(df_work[target_col].median())
            ctx1, ctx2, ctx3 = st.columns(3)
            ctx1.metric("Dataset Min Price", f"USD {p_min:,.4f}")
            ctx2.metric("Dataset Median Price", f"USD {p_med:,.4f}",
                        delta=f"{prediction - p_med:+.4f} from median")
            ctx3.metric("Dataset Max Price", f"USD {p_max:,.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — FUTURE FORECAST
# ════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.subheader("Step 4 — Future Price Forecast")

    if not st.session_state.trained_models:
        st.info("Train the models first in the 'Train & Compare Models' tab.")
        st.stop()

    st.caption(
        "Iterative future price forecast using the selected model. "
        "Features are extrapolated by their historical mean step-change per period."
    )

    fc1, fc2, fc3 = st.columns(3)

    # Commodity filter for context
    commodity_filter = None
    if commodity_col != "(none)":
        comm_opts = get_unique(commodity_col)
        commodity_filter = fc1.selectbox("Commodity to forecast", comm_opts,
                                          key="fc_commodity")

    n_steps = fc2.number_input("Forecast Steps (periods)", 1, 60, 12, step=1)

    forecast_model_choices = list(st.session_state.trained_models.keys())
    fc_model = fc3.selectbox(
        "Model",
        forecast_model_choices,
        index=forecast_model_choices.index(st.session_state.best_model_name)
        if st.session_state.best_model_name in forecast_model_choices else 0,
        key="fc_model",
    )

    if st.button("Generate Forecast", type="primary"):
        try:
            df_enc = st.session_state.df_clean.copy()

            # Filter by commodity if applicable
            if commodity_filter and commodity_col != "(none)":
                le  = st.session_state.label_encoders.get(commodity_col)
                enc_val = None
                if le and commodity_filter.strip().lower() in le.classes_:
                    enc_val = le.transform([commodity_filter.strip().lower()])[0]
                if enc_val is not None:
                    sub = df_enc[df_enc[commodity_col] == enc_val]
                    if len(sub) >= 5:
                        df_enc = sub

            # Compute per-feature deltas
            last_row = df_enc[all_features].iloc[-1].values.astype(float).copy()
            deltas   = []
            for col in all_features:
                series = df_enc[col].dropna().values
                deltas.append(np.mean(np.diff(series)) if len(series) >= 2 else 0.0)
            deltas = np.array(deltas)

            pipe     = st.session_state.trained_models[fc_model]
            current  = last_row.copy()
            records  = []

            for i in range(1, int(n_steps) + 1):
                current = current + deltas
                inp     = pd.DataFrame([current], columns=all_features)
                pred    = max(pipe.predict(inp)[0], 0)
                records.append({"Step": f"T+{i}", "Predicted Price (USD)": round(pred, 4)})

            forecast_df = pd.DataFrame(records)

            # Change columns
            prices  = forecast_df["Predicted Price (USD)"].values
            prev    = np.concatenate([[prices[0]], prices[:-1]])
            forecast_df["Change (USD)"] = np.round(prices - prev, 4)
            forecast_df["Change (%)"]   = np.where(
                prev != 0, np.round((prices - prev) / np.abs(prev) * 100, 2), 0.0
            )

            def color_change(v):
                if isinstance(v, (int, float)):
                    return "color: green" if v > 0 else ("color: red" if v < 0 else "")
                return ""

            styled_fc = (
                forecast_df.style
                .applymap(color_change, subset=["Change (USD)", "Change (%)"])
                .format({
                    "Predicted Price (USD)": "USD {:,.4f}",
                    "Change (USD)":          "{:+.4f}",
                    "Change (%)":            "{:+.2f}%",
                })
            )
            st.markdown(f"#### Forecast for: {commodity_filter or 'All'} | Model: {fc_model}")
            st.dataframe(styled_fc, use_container_width=True,
                         height=min(500, 60 + len(forecast_df) * 35))

            # Summary
            fc_min  = prices.min()
            fc_max  = prices.max()
            fc_mean = prices.mean()
            tot_chg = prices[-1] - prices[0]
            tot_pct = (tot_chg / abs(prices[0]) * 100) if prices[0] != 0 else 0

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.markdown(f'<div class="metric-card"><div class="val">USD {fc_min:,.4f}</div>'
                         f'<div class="lbl">Forecast Min</div></div>', unsafe_allow_html=True)
            sm2.markdown(f'<div class="metric-card"><div class="val">USD {fc_max:,.4f}</div>'
                         f'<div class="lbl">Forecast Max</div></div>', unsafe_allow_html=True)
            sm3.markdown(f'<div class="metric-card"><div class="val">USD {fc_mean:,.4f}</div>'
                         f'<div class="lbl">Forecast Mean</div></div>', unsafe_allow_html=True)
            color = "green" if tot_chg >= 0 else "red"
            sm4.markdown(
                f'<div class="metric-card">'
                f'<div class="val" style="color:{color}">{tot_chg:+.4f}</div>'
                f'<div class="lbl">Total Change ({tot_pct:+.2f}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            csv_bytes = forecast_df.to_csv(index=False).encode()
            st.download_button(
                "Download Forecast CSV",
                data=csv_bytes,
                file_name="pricesight_forecast.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.exception(e)
