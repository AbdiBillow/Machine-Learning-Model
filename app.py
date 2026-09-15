import hashlib
import io
import joblib
import pandas as pd
import streamlit as st
from forecasting import UNITS, KEYS, clean_data, make_panel, train_all, predict, explain

st.set_page_config(page_title='Somalia Food Price Predictor', page_icon='📈', layout='wide')


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


@st.cache_resource(show_spinner=False)
def fit_cached(csv, level):
    panel = pd.read_csv(io.StringIO(csv), parse_dates=['Origin', 'Target Date', 'last_observation'])
    return train_all(panel, level)


def show_table(frame, key):
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button('Download table', frame.to_csv(index=False), key+'.csv', 'text/csv', key=key)


st.title('Somalia food price forecasts')
st.write('Forecast each of the next three months in USD per kg, litre or head, with prediction intervals and decision context.')
with st.sidebar:
    st.header('Currency conversion')
    method = st.radio('Conversion method', ['Historical monthly rates', 'Fixed-rate scenario'])
    rates, fx, fx_bytes = None, None, b''
    if method == 'Historical monthly rates':
        st.caption('Rate direction: local-currency units per one USD. Exact district/market/currency/month matching.')
        f = st.file_uploader('Historical rates CSV (optional for USD records)', type='csv', key='fx')
        st.download_button('Download rate template', 'Admin 2,Market Name,Currency,Date,Local_per_USD\n',
                           'historical_rates_template.csv', 'text/csv')
        if f:
            fx_bytes = f.getvalue()
            try:
                fx = pd.read_csv(io.BytesIO(fx_bytes))
            except Exception as exc:
                st.error(f'Cannot read rate file: {exc}')
                st.stop()
    else:
        st.warning('Fixed rates are scenario assumptions, not verified historical dollar prices.')
        sos = st.number_input('SOS per USD', min_value=1., value=26000., step=500.)
        sls = st.number_input('SLS per USD', min_value=1., value=10000., step=500.)
        rates = {'SOS': sos, 'SLS': sls, 'USD': 1.}
    level = st.selectbox('Prediction interval level', [.9, .8, .95], format_func=lambda v: f'{v:.0%}')

uploaded = st.file_uploader('Upload WFP food-price CSV', type='csv')
if not uploaded:
    st.info('Upload your historical market-price data to begin.')
    st.stop()
try:
    cleaned, report = clean_data(pd.read_csv(io.BytesIO(uploaded.getvalue())), rates=rates, fx=fx)
except Exception as exc:
    st.error(f'Data preparation failed: {exc}')
    st.stop()
st.success('Data uploaded and preprocessing completed. Genuine price spikes are preserved.')
with st.expander('Data quality and currency coverage'):
    st.json(report)
    st.dataframe(cleaned.head(100), hide_index=True)
if report['Rows without currency conversion']:
    st.warning(f"{report['Rows without currency conversion']:,} rows excluded because currency conversion is unavailable.")
if method == 'Historical monthly rates' and fx is None:
    st.info('USD records are used. Upload verified historical rates to include local-currency prices.')
fuel = st.selectbox('Optional fuel input: choose a fuel commodity recorded in litres',
                    ['No fuel input'] + sorted(cleaned.loc[cleaned['Unit'].eq('L'), 'Commodity'].unique()))
st.caption('A selected fuel is tested with and without the input for each algorithm. Only observations available at the forecast origin are used. Rainfall is excluded.')
panel = make_panel(cleaned, fuel=None if fuel == 'No fuel input' else fuel)
signature = hashlib.sha256(uploaded.getvalue()+fx_bytes+repr((rates, level, fuel,
    pd.Timestamp.now(tz='UTC').strftime('%Y-%m'))).encode()).hexdigest()
if st.session_state.get('signature') != signature:
    st.session_state.update(signature=signature, trained=False, basket=[])
    st.session_state.pop('results', None)
if st.button('Train and compare models', type='primary'):
    st.session_state['trained'] = True
    st.session_state.pop('results', None)
if not st.session_state['trained']:
    st.info('Training compares four algorithms and two benchmarks separately for each unit and each 1-, 2- and 3-month horizon.')
    st.stop()
try:
    with st.spinner('Splitting history, training, calibrating intervals and evaluating all three horizons...'):
        bundles, skipped = fit_cached(panel.to_csv(index=False), level)
except Exception as exc:
    st.error(f'Training failed: {exc}')
    st.stop()
st.success('Chronological splits, training, interval calibration and final evaluation completed.')
for (u, h), reason in skipped.items():
    st.warning(f'USD/{UNITS[u]}, horizon {h}: {reason}')
st.subheader('Model comparison')
for (u, h), b in bundles.items():
    with st.expander(f'USD/{UNITS[u]} · {h} month(s) · {b["best"]}'):
        st.write('Selection period: four algorithms and two benchmarks')
        show_table(b['selection'], f'selection_{u}_{h}')
        st.write('Final evaluation: held out from model selection and interval calibration')
        show_table(b['test_metrics'], f'test_{u}_{h}')
        st.caption('MAE and RMSE are in USD per selected unit. Seasonal benchmark falls back to latest price when seasonal history is missing, keeping comparison samples equal.')
        st.dataframe(pd.DataFrame([{'Purpose': k, 'First forecast origin': v[0].date(),
                                   'Last forecast origin': v[-1].date()} for k, v in b['dates'].items()]), hide_index=True)
        show_table(b['details'], f'detailed_accuracy_{u}_{h}')
        st.caption('Each historical forecast trains only on target prices already observed at its origin. Two-month gaps between stages protect 3-month labels. Coverage is empirical and may change during shocks.')
        if pd.isna(b['q']):
            st.warning('At least 20 calibration residuals are needed for an interval at this unit and horizon.')
model_bytes = io.BytesIO()
joblib.dump({'bundles': bundles, 'currency_method': report['Currency method'], 'version': 2}, model_bytes)
st.download_button('Download trained models', model_bytes.getvalue(), 'food_price_forecasting_models.joblib', 'application/octet-stream')

st.subheader('Price prediction: next three months')
selection = {}
unit = st.selectbox('Prediction unit', sorted({u for u, h in bundles}), format_func=lambda u: 'USD per '+UNITS[u])
selection['Unit'] = unit
options = cleaned[cleaned['Unit'].eq(unit)]
for c in ['Admin 2', 'Market Name', 'Commodity', 'Currency']:
    selection[c] = st.selectbox(c, sorted(options[c].unique()))
    options = options[options[c].eq(selection[c])]
choice = tuple(selection[c] for c in KEYS)
if st.session_state.get('choice') != choice:
    st.session_state['choice'] = choice
    st.session_state.pop('results', None)
st.write(f"Forecast origin: **{panel['Origin'].max():%B %Y}**. Three forecasts use the same observed history.")
if st.button('Predict next three months', type='primary'):
    results = []
    for h in (1, 2, 3):
        if (unit, h) in bundles:
            try:
                results.append(predict(bundles[(unit, h)], panel, selection))
            except ValueError as exc:
                st.error(f'Month {h}: {exc}')
    st.session_state['results'] = results
results = st.session_state.get('results')
if results:
    st.success('Prediction completed.')
    table = pd.DataFrame([{'Month': r['Target Date'].strftime('%B %Y'), 'Horizon': r['Horizon'],
                          'Predicted USD/'+UNITS[unit]: r['prediction'], 'Lower': r['lower'], 'Upper': r['upper'],
                          'Change vs latest (%)': r['change_pct'], 'Method': r['algorithm']} for r in results])
    show_table(table, 'three_month_forecast')
    selected_month = st.selectbox('Month for decision context and procurement', range(len(results)),
                                 format_func=lambda i: results[i]['Target Date'].strftime('%B %Y'))
    r = results[selected_month]
    a, b = st.columns(2)
    a.metric('Predicted USD per '+UNITS[unit], f"${r['prediction']:.2f}", f"{r['change_pct']:+.1f}% vs latest", delta_color='inverse')
    if pd.notna(r['lower']):
        b.metric(f'{level:.0%} prediction interval', f"${r['lower']:.2f} – ${r['upper']:.2f}")
    else:
        b.warning('Interval unavailable: insufficient calibration history.')
    st.write(explain(r, report['Currency method']))
    quantity = st.number_input('Procurement quantity ('+UNITS[unit]+')', min_value=0., value=1.)
    threshold = st.number_input('Price increase alert threshold (%)', min_value=0., value=10.)
    if r['change_pct'] >= threshold:
        st.warning(f'The point forecast meets your {threshold:g}% increase threshold. Verify quotations.')
    else:
        st.info(f'The point forecast is below your {threshold:g}% increase threshold.')
    cost = {'Month': r['Target Date'].strftime('%Y-%m'), 'Market': selection['Market Name'],
            'Commodity': selection['Commodity'], 'Quantity': quantity, 'Unit': UNITS[unit],
            'Expected USD': quantity*r['prediction'], 'Lower USD': quantity*r['lower'], 'Upper USD': quantity*r['upper']}
    st.dataframe(pd.DataFrame([cost]), hide_index=True)
    if st.button('Add to planning basket'):
        st.session_state['basket'].append(cost)
if st.session_state.get('basket'):
    basket = pd.DataFrame(st.session_state['basket'])
    st.subheader('Planning basket')
    show_table(basket, 'planning_basket')
    st.metric('Expected total cost (USD)', f"${basket['Expected USD'].sum():,.2f}")
    st.caption('Item intervals are not a joint basket interval; prices may move together.')
    if st.button('Clear basket'):
        st.session_state['basket'] = []
        st.rerun()
st.caption('Backtests assume completed-month observations are available at month-end. Reporting delays and revisions may reduce real-time performance.')
