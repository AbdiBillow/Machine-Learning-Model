import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PriceSight — ML Forecasting",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS  (dark, editorial aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;1,9..40,300&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d0f14;
    --surface:   #13161f;
    --surface2:  #1a1e2b;
    --border:    #252936;
    --accent:    #e8c547;
    --accent2:   #4fc3f7;
    --success:   #4ade80;
    --danger:    #f87171;
    --text:      #e8eaf0;
    --muted:     #7a8099;
    --font-head: 'DM Serif Display', Georgia, serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text);
}
.stApp { background: var(--bg); }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent);
    font-family: var(--font-head);
    letter-spacing: .02em;
}

/* ── Headings ── */
h1 { font-family: var(--font-head); font-size: 2.8rem !important; color: var(--text) !important; letter-spacing: -.01em; line-height: 1.1; }
h2 { font-family: var(--font-head); font-size: 1.7rem !important; color: var(--text) !important; }
h3 { font-family: var(--font-body); font-size: 1.1rem !important; color: var(--muted) !important; font-weight: 600; text-transform: uppercase; letter-spacing: .1em; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-blue   { border-left: 3px solid var(--accent2); }
.card-green  { border-left: 3px solid var(--success); }
.card-red    { border-left: 3px solid var(--danger); }

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-tile {
    flex: 1; min-width: 140px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-tile .label {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--muted);
    font-family: var(--font-mono);
    margin-bottom: .4rem;
}
.metric-tile .value {
    font-family: var(--font-mono);
    font-size: 1.55rem;
    font-weight: 500;
    color: var(--accent);
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: .2rem .65rem;
    border-radius: 20px;
    font-size: .72rem;
    font-family: var(--font-mono);
    font-weight: 500;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.badge-green { background: rgba(74,222,128,.12); color: var(--success); border: 1px solid rgba(74,222,128,.3); }
.badge-yellow{ background: rgba(232,197,71,.12); color: var(--accent);  border: 1px solid rgba(232,197,71,.3); }
.badge-red   { background: rgba(248,113,113,.12); color: var(--danger); border: 1px solid rgba(248,113,113,.3); }
.badge-blue  { background: rgba(79,195,247,.12);  color: var(--accent2);border: 1px solid rgba(79,195,247,.3); }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #13161f 0%, #1a1e2b 60%, #0d1219 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(232,197,71,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute; bottom: -40px; left: 40%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(79,195,247,.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    margin-top: .5rem;
    max-width: 560px;
    line-height: 1.6;
}
.hero-tag {
    font-family: var(--font-mono);
    font-size: .72rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: .15em;
    margin-bottom: .8rem;
}

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stNumberInput input, .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
.stSlider > div { accent-color: var(--accent); }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    letter-spacing: .03em;
    padding: .55rem 1.6rem !important;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Alerts ── */
.stSuccess { background: rgba(74,222,128,.08) !important; border: 1px solid rgba(74,222,128,.25) !important; border-radius: 8px !important; color: var(--success) !important; }
.stError   { background: rgba(248,113,113,.08) !important; border: 1px solid rgba(248,113,113,.25) !important; border-radius: 8px !important; }
.stWarning { background: rgba(232,197,71,.08)  !important; border: 1px solid rgba(232,197,71,.25)  !important; border-radius: 8px !important; }
.stInfo    { background: rgba(79,195,247,.08)  !important; border: 1px solid rgba(79,195,247,.25)  !important; border-radius: 8px !important; }

/* ── Dataframe ── */
.dataframe { font-family: var(--font-mono) !important; font-size: .8rem; }

/* ── Tab strip ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px;
    padding: .3rem;
    gap: .2rem;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0d0f14 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, rgba(232,197,71,.08), rgba(79,195,247,.05));
    border: 1px solid rgba(232,197,71,.3);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.pred-box .pred-label {
    font-family: var(--font-mono);
    font-size: .78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .14em;
    margin-bottom: .5rem;
}
.pred-box .pred-value {
    font-family: var(--font-head);
    font-size: 3rem;
    color: var(--accent);
    line-height: 1;
}
.pred-box .pred-range {
    font-family: var(--font-mono);
    font-size: .82rem;
    color: var(--muted);
    margin-top: .6rem;
}

/* ── Future table ── */
.future-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: .85rem; }
.future-table th {
    background: var(--surface2);
    color: var(--muted);
    text-align: left;
    padding: .6rem 1rem;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    border-bottom: 1px solid var(--border);
}
.future-table td { padding: .6rem 1rem; border-bottom: 1px solid var(--border); }
.future-table tr:last-child td { border-bottom: none; }
.future-table .up   { color: var(--success); }
.future-table .down { color: var(--danger); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#13161f",
    "axes.facecolor":    "#1a1e2b",
    "axes.edgecolor":    "#252936",
    "axes.labelcolor":   "#e8eaf0",
    "axes.titlecolor":   "#e8eaf0",
    "text.color":        "#e8eaf0",
    "xtick.color":       "#7a8099",
    "ytick.color":       "#7a8099",
    "grid.color":        "#252936",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#13161f",
    "legend.edgecolor":  "#252936",
    "font.family":       "sans-serif",
})
ACCENT  = "#e8c547"
ACCENT2 = "#4fc3f7"
SUCCESS = "#4ade80"
DANGER  = "#f87171"
MUTED   = "#7a8099"

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "model":        None,
    "scaler":       None,
    "pipeline":     None,
    "feature_cols": [],
    "target_col":   None,
    "metrics":      {},
    "cv_scores":    None,
    "df":           None,
    "model_type":   "Linear Regression",
    "poly_degree":  1,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def clean_dataframe(df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
    """Drop rows where ALL feature+target values are NaN, then impute remaining."""
    cols = feature_cols + [target_col]
    df_clean = df[cols].copy()
    # Drop rows where target is missing
    df_clean = df_clean.dropna(subset=[target_col])
    return df_clean


def build_pipeline(model_type: str, poly_degree: int, alpha: float = 1.0) -> Pipeline:
    steps = []
    steps.append(("imputer", SimpleImputer(strategy="mean")))
    if poly_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    steps.append(("scaler", StandardScaler()))
    if model_type == "Ridge Regression":
        steps.append(("model", Ridge(alpha=alpha)))
    elif model_type == "Lasso Regression":
        steps.append(("model", Lasso(alpha=alpha, max_iter=10_000)))
    else:
        steps.append(("model", LinearRegression()))
    return Pipeline(steps)


def get_r2_badge(r2: float) -> str:
    if r2 >= 0.85:
        return f'<span class="badge badge-green">Excellent — R² {r2:.3f}</span>'
    elif r2 >= 0.65:
        return f'<span class="badge badge-yellow">Good — R² {r2:.3f}</span>'
    elif r2 >= 0.4:
        return f'<span class="badge badge-blue">Fair — R² {r2:.3f}</span>'
    else:
        return f'<span class="badge badge-red">Poor — R² {r2:.3f}</span>'


def fig_to_st(fig):
    """Render a matplotlib figure in Streamlit without the white border flash."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def forecast_future(pipeline, last_known: pd.Series, feature_cols: list,
                    steps: int, method: str) -> pd.DataFrame:
    """
    Iterative future forecasting.
    - method='trend'  : uses linear time-trend to extrapolate features
    - method='manual' : uses user-supplied future feature values
    """
    records = []
    current = last_known[feature_cols].values.astype(float).copy()

    # Compute per-feature mean deltas from the training data for trend mode
    df = st.session_state.df
    deltas = []
    for col in feature_cols:
        series = df[col].dropna().values
        if len(series) >= 2:
            deltas.append(np.mean(np.diff(series)))
        else:
            deltas.append(0.0)
    deltas = np.array(deltas)

    for i in range(1, steps + 1):
        if method == "trend":
            current = current + deltas
        input_df = pd.DataFrame([current], columns=feature_cols)
        pred = pipeline.predict(input_df)[0]
        records.append({"Step": i, **dict(zip(feature_cols, current)), "Predicted Price": pred})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.2rem'>
        <div style='font-family:var(--font-mono);font-size:.7rem;color:#7a8099;
                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:.3rem'>
            ML Platform
        </div>
        <div style='font-family:var(--font-head);font-size:1.6rem;
                    color:#e8eaf0;line-height:1.1'>
            PriceSight
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Model Settings")

    model_type = st.selectbox(
        "Algorithm",
        ["Linear Regression", "Ridge Regression", "Lasso Regression"],
        index=["Linear Regression", "Ridge Regression", "Lasso Regression"].index(
            st.session_state.model_type
        ),
    )
    st.session_state.model_type = model_type

    poly_degree = st.slider("Polynomial Degree", 1, 4, st.session_state.poly_degree,
                            help="Degree 1 = standard linear. Higher degrees capture nonlinear patterns.")
    st.session_state.poly_degree = poly_degree

    alpha = 1.0
    if model_type in ("Ridge Regression", "Lasso Regression"):
        alpha = st.number_input("Regularization (alpha)", min_value=0.0001,
                                max_value=100.0, value=1.0, format="%.4f")

    test_size = st.slider("Test Split (%)", 10, 40, 20) / 100

    st.markdown("---")
    st.markdown("## Data Options")
    drop_duplicates = st.checkbox("Drop duplicate rows", value=True)
    outlier_removal = st.checkbox("Remove outliers (IQR method)", value=False)

    st.markdown("---")
    st.markdown("## Model I/O")

    if st.session_state.pipeline is not None:
        buf = io.BytesIO()
        joblib.dump({
            "pipeline":     st.session_state.pipeline,
            "feature_cols": st.session_state.feature_cols,
            "target_col":   st.session_state.target_col,
            "metrics":      st.session_state.metrics,
        }, buf)
        buf.seek(0)
        st.download_button(
            "Download Trained Model",
            data=buf,
            file_name="pricesight_model.pkl",
            mime="application/octet-stream",
        )

    uploaded_model = st.file_uploader("Load Saved Model (.pkl)", type=["pkl"])
    if uploaded_model and st.button("Apply Loaded Model"):
        try:
            data = joblib.load(uploaded_model)
            st.session_state.pipeline     = data["pipeline"]
            st.session_state.feature_cols = data["feature_cols"]
            st.session_state.target_col   = data["target_col"]
            st.session_state.metrics      = data.get("metrics", {})
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:var(--font-mono);font-size:.7rem;color:#7a8099;line-height:1.8'>
        PriceSight v2.0<br>
        Linear · Ridge · Lasso<br>
        Polynomial · Future Forecast
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">ML Price Forecasting Platform</div>
    <h1>PriceSight</h1>
    <div class="hero-sub">
        Upload your dataset, configure a regression pipeline, evaluate model
        performance, and forecast future prices — all in one place.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a CSV file here, or click to browse",
    type=["csv"],
    help="Your CSV should contain numeric feature columns and a numeric price/target column.",
)

if uploaded_file is None:
    st.markdown("""
    <div class="card card-accent" style="margin-top:1.5rem">
        <div style="font-family:var(--font-mono);font-size:.78rem;color:#7a8099;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem">
            Getting Started
        </div>
        <ul style="color:#e8eaf0;line-height:2;margin:0;padding-left:1.2rem;font-size:.95rem">
            <li>Upload a CSV with numeric columns</li>
            <li>Select your feature and target columns</li>
            <li>Train the model from the <strong>Training</strong> tab</li>
            <li>Forecast future prices in the <strong>Forecast</strong> tab</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if drop_duplicates:
    df_raw = df_raw.drop_duplicates()

df_raw.reset_index(drop=True, inplace=True)
st.session_state.df = df_raw

# ─────────────────────────────────────────────────────────────────────────────
#  COLUMN SELECTION
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols = [c for c in df_raw.columns if is_numeric_series(df_raw[c])]
all_cols     = df_raw.columns.tolist()

if len(numeric_cols) < 2:
    st.error("Your dataset must contain at least 2 numeric columns (features + target).")
    st.stop()

col_a, col_b = st.columns([3, 2])
with col_a:
    feature_cols = st.multiselect(
        "Feature Columns (predictors)",
        options=numeric_cols,
        default=st.session_state.feature_cols if st.session_state.feature_cols else [],
        help="Select one or more numeric columns to use as model inputs.",
    )
with col_b:
    remaining    = [c for c in numeric_cols if c not in feature_cols]
    target_col   = st.selectbox(
        "Target Column (price to predict)",
        options=remaining if remaining else numeric_cols,
        help="The numeric column you want the model to predict.",
    )

# Validate
if not feature_cols:
    st.info("Select at least one feature column to continue.")
    st.stop()

if target_col in feature_cols:
    st.error("The target column must not be selected as a feature.")
    st.stop()

non_numeric = [c for c in feature_cols if not is_numeric_series(df_raw[c])]
if non_numeric:
    st.error(f"Non-numeric feature columns detected: {non_numeric}. Please select numeric columns only.")
    st.stop()

st.session_state.feature_cols = feature_cols
st.session_state.target_col   = target_col

# Clean dataset
df_clean = clean_dataframe(df_raw, feature_cols, target_col)

if outlier_removal:
    for col in feature_cols + [target_col]:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR     = Q3 - Q1
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) &
            (df_clean[col] <= Q3 + 1.5 * IQR)
        ]

df_clean.reset_index(drop=True, inplace=True)

if len(df_clean) < 10:
    st.error(f"Only {len(df_clean)} usable rows remain after cleaning. Need at least 10.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA SUMMARY ROW
# ─────────────────────────────────────────────────────────────────────────────
missing_pct = df_raw[feature_cols + [target_col]].isnull().mean().mean() * 100
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-tile">
        <div class="label">Total Rows</div>
        <div class="value">{len(df_raw):,}</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-tile">
        <div class="label">Clean Rows</div>
        <div class="value">{len(df_clean):,}</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-tile">
        <div class="label">Features</div>
        <div class="value">{len(feature_cols)}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-tile">
        <div class="label">Missing %</div>
        <div class="value">{missing_pct:.1f}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_data, tab_train, tab_eval, tab_predict, tab_forecast = st.tabs([
    "Data Explorer", "Training", "Evaluation", "Single Prediction", "Future Forecast"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### Data Explorer")

    with st.expander("Dataset Preview (first 50 rows)", expanded=True):
        st.dataframe(
            df_clean[feature_cols + [target_col]].head(50),
            use_container_width=True,
            height=300,
        )

    with st.expander("Descriptive Statistics"):
        st.dataframe(
            df_clean[feature_cols + [target_col]].describe().T.style.format("{:.3f}"),
            use_container_width=True,
        )

    with st.expander("Missing Values"):
        mv = df_raw[feature_cols + [target_col]].isnull().sum().reset_index()
        mv.columns = ["Column", "Missing Count"]
        mv["Missing %"] = (mv["Missing Count"] / len(df_raw) * 100).round(2)
        st.dataframe(mv, use_container_width=True)

    st.markdown("---")
    st.markdown("### Distribution & Correlation")

    plot_col = st.selectbox("Inspect column distribution", feature_cols + [target_col])
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df_clean[plot_col].dropna(), bins=30, color=ACCENT, alpha=.85, edgecolor="#0d0f14")
        ax.set_title(f"Distribution — {plot_col}", fontsize=10, pad=10)
        ax.set_xlabel(plot_col, fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig_to_st(fig)

    with c2:
        if len(feature_cols) > 1:
            corr_df = df_clean[feature_cols + [target_col]].corr()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            mask = np.zeros_like(corr_df, dtype=bool)
            np.fill_diagonal(mask, True)
            sns.heatmap(
                corr_df, ax=ax, mask=mask,
                cmap=sns.diverging_palette(220, 45, as_cmap=True),
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=.4, linecolor="#252936",
                cbar_kws={"shrink": .75},
            )
            ax.set_title("Correlation Matrix", fontsize=10, pad=10)
            ax.tick_params(labelsize=7)
            fig.tight_layout()
            fig_to_st(fig)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(df_clean[feature_cols[0]], df_clean[target_col],
                       color=ACCENT2, alpha=.5, s=20, edgecolors="none")
            ax.set_xlabel(feature_cols[0], fontsize=8)
            ax.set_ylabel(target_col, fontsize=8)
            ax.set_title(f"{feature_cols[0]} vs {target_col}", fontsize=10, pad=10)
            ax.grid(True)
            fig.tight_layout()
            fig_to_st(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("### Train the Model")

    info_cols = st.columns(3)
    info_cols[0].markdown(f"""<div class="card card-accent">
        <div class="label" style="font-family:var(--font-mono);font-size:.72rem;
             color:#7a8099;text-transform:uppercase;letter-spacing:.1em">Algorithm</div>
        <div style="font-size:1rem;font-weight:600;margin-top:.3rem">{model_type}</div>
    </div>""", unsafe_allow_html=True)
    info_cols[1].markdown(f"""<div class="card card-blue">
        <div class="label" style="font-family:var(--font-mono);font-size:.72rem;
             color:#7a8099;text-transform:uppercase;letter-spacing:.1em">Poly Degree</div>
        <div style="font-size:1rem;font-weight:600;margin-top:.3rem">{poly_degree}</div>
    </div>""", unsafe_allow_html=True)
    info_cols[2].markdown(f"""<div class="card card-green">
        <div class="label" style="font-family:var(--font-mono);font-size:.72rem;
             color:#7a8099;text-transform:uppercase;letter-spacing:.1em">Test Split</div>
        <div style="font-size:1rem;font-weight:600;margin-top:.3rem">{int(test_size*100)}%</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Train Model Now", key="train_btn"):
        with st.spinner("Building pipeline and training ..."):
            try:
                X = df_clean[feature_cols]
                y = df_clean[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                pipeline = build_pipeline(model_type, poly_degree,
                                          alpha if model_type != "Linear Regression" else 1.0)
                pipeline.fit(X_train, y_train)

                y_pred       = pipeline.predict(X_test)
                y_pred_train = pipeline.predict(X_train)

                mae  = mean_absolute_error(y_test, y_pred)
                mse  = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2   = r2_score(y_test, y_pred)
                r2_train = r2_score(y_train, y_pred_train)

                # Cross-validation
                cv_scores = cross_val_score(pipeline, X, y, cv=min(5, len(X)//5),
                                            scoring="r2", n_jobs=-1)

                st.session_state.pipeline   = pipeline
                st.session_state.metrics    = {
                    "mae": mae, "mse": mse, "rmse": rmse,
                    "r2": r2, "r2_train": r2_train,
                    "y_test": y_test, "y_pred": y_pred,
                    "y_train": y_train, "y_pred_train": y_pred_train,
                }
                st.session_state.cv_scores = cv_scores

                # Overfitting check
                overfit_gap = r2_train - r2
                if overfit_gap > 0.25:
                    st.warning(
                        f"Possible overfitting detected: train R² = {r2_train:.3f}, "
                        f"test R² = {r2:.3f} (gap = {overfit_gap:.3f}). "
                        "Consider reducing polynomial degree or increasing alpha."
                    )

                st.success("Model trained successfully!")
                st.markdown(get_r2_badge(r2), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)

    if st.session_state.pipeline is not None and st.session_state.metrics:
        m = st.session_state.metrics
        st.markdown("#### Performance Snapshot")
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="label">MAE</div>
                <div class="value">{m['mae']:.4f}</div>
            </div>
            <div class="metric-tile">
                <div class="label">RMSE</div>
                <div class="value">{m['rmse']:.4f}</div>
            </div>
            <div class="metric-tile">
                <div class="label">MSE</div>
                <div class="value">{m['mse']:.4f}</div>
            </div>
            <div class="metric-tile">
                <div class="label">R² (Test)</div>
                <div class="value">{m['r2']:.4f}</div>
            </div>
            <div class="metric-tile">
                <div class="label">R² (Train)</div>
                <div class="value">{m['r2_train']:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.cv_scores is not None:
            cv = st.session_state.cv_scores
            st.markdown(f"""
            <div class="card card-blue" style="margin-top:.8rem">
                <div style="font-family:var(--font-mono);font-size:.78rem;color:#7a8099;
                     text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem">
                    5-Fold Cross-Validation R²
                </div>
                <div style="font-family:var(--font-mono);font-size:1.1rem;color:#4fc3f7">
                    {cv.mean():.4f} <span style="color:#7a8099;font-size:.8rem">
                    (+/- {cv.std()*2:.4f})</span>
                </div>
                <div style="font-size:.78rem;color:#7a8099;margin-top:.3rem">
                    Individual folds: {", ".join(f"{v:.3f}" for v in cv)}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### Model Evaluation")

    if st.session_state.pipeline is None:
        st.info("Train the model first to see evaluation charts.")
    else:
        m = st.session_state.metrics
        y_test  = m["y_test"]
        y_pred  = m["y_pred"]
        residuals = np.array(y_test) - np.array(y_pred)

        ecol1, ecol2 = st.columns(2)

        # Actual vs Predicted
        with ecol1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(y_test, y_pred, color=ACCENT, alpha=.65, s=25, edgecolors="none", zorder=3)
            mn = min(y_test.min(), y_pred.min())
            mx = max(y_test.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], color=DANGER, linewidth=1.5, linestyle="--", label="Perfect fit")
            ax.set_xlabel("Actual", fontsize=8)
            ax.set_ylabel("Predicted", fontsize=8)
            ax.set_title("Actual vs Predicted", fontsize=10, pad=10)
            ax.legend(fontsize=7)
            ax.grid(True)
            fig.tight_layout()
            fig_to_st(fig)

        # Residual plot
        with ecol2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(y_pred, residuals, color=ACCENT2, alpha=.65, s=25, edgecolors="none", zorder=3)
            ax.axhline(0, color=DANGER, linewidth=1.5, linestyle="--")
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Residual", fontsize=8)
            ax.set_title("Residual Plot", fontsize=10, pad=10)
            ax.grid(True)
            fig.tight_layout()
            fig_to_st(fig)

        ecol3, ecol4 = st.columns(2)

        # Residual histogram
        with ecol3:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.hist(residuals, bins=25, color=SUCCESS, alpha=.8, edgecolor="#0d0f14")
            ax.axvline(0, color=DANGER, linewidth=1.5, linestyle="--")
            ax.set_xlabel("Residual Value", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title("Residual Distribution", fontsize=10, pad=10)
            ax.grid(True, axis="y")
            fig.tight_layout()
            fig_to_st(fig)

        # Feature importance (coefficients)
        with ecol4:
            try:
                pipe = st.session_state.pipeline
                model_step = pipe.named_steps["model"]
                coef = model_step.coef_

                if "poly" in pipe.named_steps:
                    poly_step  = pipe.named_steps["poly"]
                    feat_names = poly_step.get_feature_names_out(feature_cols)
                else:
                    feat_names = feature_cols

                n_show  = min(15, len(feat_names))
                idx     = np.argsort(np.abs(coef))[-n_show:][::-1]
                top_feats = [feat_names[i] for i in idx]
                top_coefs = [coef[i] for i in idx]
                colors    = [SUCCESS if v > 0 else DANGER for v in top_coefs]

                fig, ax = plt.subplots(figsize=(6, 3.8))
                bars = ax.barh(range(len(top_feats)), top_coefs, color=colors, alpha=.85)
                ax.set_yticks(range(len(top_feats)))
                ax.set_yticklabels(top_feats, fontsize=7)
                ax.axvline(0, color=MUTED, linewidth=.8)
                ax.set_xlabel("Coefficient", fontsize=8)
                ax.set_title("Feature Coefficients", fontsize=10, pad=10)
                ax.grid(True, axis="x")
                fig.tight_layout()
                fig_to_st(fig)
            except Exception:
                st.info("Coefficient plot not available for this configuration.")

        # Coefficients table
        with st.expander("Full Coefficient Table"):
            try:
                coef_df = pd.DataFrame({
                    "Feature":     feat_names,
                    "Coefficient": coef,
                    "|Coeff|":     np.abs(coef),
                }).sort_values("|Coeff|", ascending=False)
                intercept = pipe.named_steps["model"].intercept_
                st.dataframe(
                    coef_df.style.format({"Coefficient": "{:.6f}", "|Coeff|": "{:.6f}"}),
                    use_container_width=True,
                )
                st.markdown(f"""
                <div class="card" style="margin-top:.6rem">
                    <span style="font-family:var(--font-mono);font-size:.8rem;color:#7a8099">
                        Intercept:
                    </span>
                    <span style="font-family:var(--font-mono);font-size:.9rem;color:#e8c547;
                                 margin-left:.5rem">
                        {intercept:.6f}
                    </span>
                </div>""", unsafe_allow_html=True)
            except Exception:
                st.info("Coefficient table not available.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Single Prediction")

    if st.session_state.pipeline is None:
        st.info("Train the model first to make predictions.")
    else:
        st.markdown(
            '<div class="card" style="margin-bottom:1rem">'
            '<div style="font-size:.88rem;color:#7a8099">Enter values for each feature below, '
            'then click <strong style="color:#e8eaf0">Predict Price</strong>.</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # Build input grid
        n_cols   = min(3, len(feature_cols))
        col_grid = st.columns(n_cols)
        input_data = {}
        for i, col in enumerate(feature_cols):
            col_min = float(df_clean[col].min())
            col_max = float(df_clean[col].max())
            col_med = float(df_clean[col].median())
            with col_grid[i % n_cols]:
                input_data[col] = st.number_input(
                    col,
                    min_value=col_min * 0.0 if col_min < 0 else col_min * 0.5,
                    max_value=col_max * 2.0,
                    value=col_med,
                    format="%.4f",
                    key=f"pred_input_{col}",
                )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Predict Price", key="predict_btn"):
            try:
                input_df   = pd.DataFrame([input_data])
                prediction = st.session_state.pipeline.predict(input_df)[0]

                # Confidence interval approximation via residual std
                residuals  = (np.array(st.session_state.metrics["y_test"]) -
                              np.array(st.session_state.metrics["y_pred"]))
                res_std    = residuals.std()
                lo, hi     = prediction - 1.96 * res_std, prediction + 1.96 * res_std

                st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-label">Predicted Price</div>
                    <div class="pred-value">{prediction:,.4f}</div>
                    <div class="pred-range">
                        95% confidence interval:
                        {lo:,.4f} &mdash; {hi:,.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show how close this prediction is to the data range
                target_min = float(df_clean[target_col].min())
                target_max = float(df_clean[target_col].max())
                target_med = float(df_clean[target_col].median())

                pos_cols = st.columns(3)
                pos_cols[0].metric("Dataset Min", f"{target_min:,.4f}")
                pos_cols[1].metric("Dataset Median", f"{target_med:,.4f}",
                                   delta=f"{prediction - target_med:+.4f} from median")
                pos_cols[2].metric("Dataset Max", f"{target_max:,.4f}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — FUTURE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown("### Future Price Forecast")

    if st.session_state.pipeline is None:
        st.info("Train the model first to generate future forecasts.")
    else:
        st.markdown("""
        <div class="card card-accent" style="margin-bottom:1.2rem">
            <div style="font-size:.88rem;color:#7a8099;line-height:1.7">
                <strong style="color:#e8eaf0">Trend Forecast</strong> extrapolates each
                feature by its historical mean step-change and projects prices forward.<br>
                <strong style="color:#e8eaf0">Manual Forecast</strong> lets you define
                each feature value per future step manually.
            </div>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2 = st.columns([2, 1])
        with fc1:
            forecast_method = st.radio(
                "Forecast Method",
                ["Trend Extrapolation", "Manual Step Entry"],
                horizontal=True,
            )
        with fc2:
            n_steps = st.number_input("Number of future steps", 1, 100, 10, step=1)

        last_row = df_clean[feature_cols].iloc[-1]

        if forecast_method == "Manual Step Entry":
            st.markdown("**Set feature values for the next step (repeated for all steps):**")
            manual_input = {}
            man_cols = st.columns(min(3, len(feature_cols)))
            for i, col in enumerate(feature_cols):
                with man_cols[i % len(man_cols)]:
                    manual_input[col] = st.number_input(
                        col, value=float(last_row[col]), format="%.4f",
                        key=f"fc_{col}"
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Generate Forecast", key="forecast_btn"):
            try:
                with st.spinner("Forecasting ..."):
                    method = "trend" if forecast_method == "Trend Extrapolation" else "manual"
                    if method == "manual":
                        last_row = pd.Series(manual_input)

                    forecast_df = forecast_future(
                        st.session_state.pipeline,
                        last_row,
                        feature_cols,
                        int(n_steps),
                        method,
                    )

                    # ── Historical + forecast chart ──────────────────────────
                    hist_prices = df_clean[target_col].values
                    hist_x      = np.arange(len(hist_prices))
                    fore_x      = np.arange(len(hist_prices) - 1,
                                            len(hist_prices) + len(forecast_df))
                    fore_y      = np.concatenate([[hist_prices[-1]],
                                                  forecast_df["Predicted Price"].values])

                    # Confidence band
                    residuals  = (np.array(st.session_state.metrics["y_test"]) -
                                  np.array(st.session_state.metrics["y_pred"]))
                    res_std    = residuals.std()
                    band_lo    = fore_y - 1.96 * res_std
                    band_hi    = fore_y + 1.96 * res_std

                    fig, ax = plt.subplots(figsize=(12, 4.5))
                    ax.plot(hist_x, hist_prices, color=ACCENT2, linewidth=1.8,
                            label="Historical", zorder=3)
                    ax.plot(fore_x, fore_y, color=ACCENT, linewidth=2.2,
                            linestyle="--", label="Forecast", zorder=4)
                    ax.fill_between(fore_x, band_lo, band_hi,
                                    color=ACCENT, alpha=.12, label="95% CI")
                    ax.axvline(len(hist_prices) - 1, color=MUTED,
                               linewidth=1, linestyle=":")
                    ax.set_xlabel("Time Step", fontsize=8)
                    ax.set_ylabel(target_col, fontsize=8)
                    ax.set_title(f"Price Forecast — next {n_steps} steps", fontsize=10, pad=10)
                    ax.legend(fontsize=7)
                    ax.grid(True)

                    # Highlight final forecast value
                    final_val = forecast_df["Predicted Price"].iloc[-1]
                    ax.annotate(
                        f"{final_val:,.3f}",
                        xy=(fore_x[-1], fore_y[-1]),
                        xytext=(-55, 18),
                        textcoords="offset points",
                        fontsize=7.5,
                        color=ACCENT,
                        fontfamily="monospace",
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1),
                    )

                    fig.tight_layout()
                    fig_to_st(fig)

                    # ── Forecast table ───────────────────────────────────────
                    st.markdown("#### Forecast Table")
                    forecast_display = forecast_df.copy()
                    forecast_display.insert(
                        0, "Step Label",
                        [f"T+{i}" for i in forecast_display["Step"]]
                    )
                    price_series = forecast_display["Predicted Price"]
                    prev         = [np.nan] + list(price_series.values[:-1])
                    forecast_display["Change"]   = price_series.values - np.array(prev)
                    forecast_display["Change %"] = (
                        forecast_display["Change"] / np.array(prev) * 100
                    )

                    st.dataframe(
                        forecast_display.style.format({
                            "Predicted Price": "{:,.4f}",
                            "Change":          "{:+.4f}",
                            "Change %":        "{:+.2f}%",
                            **{c: "{:.4f}" for c in feature_cols},
                        }).applymap(
                            lambda v: "color: #4ade80" if isinstance(v, float) and v > 0
                            else ("color: #f87171" if isinstance(v, float) and v < 0 else ""),
                            subset=["Change", "Change %"],
                        ),
                        use_container_width=True,
                        height=min(400, 80 + len(forecast_df) * 35),
                    )

                    # ── Summary stats ────────────────────────────────────────
                    fc_prices = forecast_df["Predicted Price"]
                    s1, s2, s3, s4 = st.columns(4)
                    s1.markdown(f"""<div class="metric-tile">
                        <div class="label">Forecast Min</div>
                        <div class="value">{fc_prices.min():,.4f}</div>
                    </div>""", unsafe_allow_html=True)
                    s2.markdown(f"""<div class="metric-tile">
                        <div class="label">Forecast Max</div>
                        <div class="value">{fc_prices.max():,.4f}</div>
                    </div>""", unsafe_allow_html=True)
                    s3.markdown(f"""<div class="metric-tile">
                        <div class="label">Forecast Mean</div>
                        <div class="value">{fc_prices.mean():,.4f}</div>
                    </div>""", unsafe_allow_html=True)
                    total_chg   = fc_prices.iloc[-1] - fc_prices.iloc[0]
                    total_chg_p = total_chg / abs(fc_prices.iloc[0]) * 100 if fc_prices.iloc[0] != 0 else 0
                    color_str   = "var(--success)" if total_chg >= 0 else "var(--danger)"
                    s4.markdown(f"""<div class="metric-tile">
                        <div class="label">Total Change</div>
                        <div class="value" style="color:{color_str};font-size:1.2rem">
                            {total_chg:+.4f}<br>
                            <span style="font-size:.85rem">({total_chg_p:+.2f}%)</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # CSV download
                    csv_buf = forecast_display.to_csv(index=False).encode()
                    st.download_button(
                        "Download Forecast CSV",
                        data=csv_buf,
                        file_name="pricesight_forecast.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.exception(e)
