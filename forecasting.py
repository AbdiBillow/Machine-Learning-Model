"""Direct 1–3 month forecasts. No future observed price is used as an input."""
import math
import numpy as np
import pandas as pd
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

UNITS = {'KG': 'kg', 'L': 'litre', 'Head': 'head'}
KEYS = ['Admin 2', 'Market Name', 'Commodity', 'Unit', 'Currency']
NUMERIC = ['recent', 'lag_1', 'lag_2', 'seasonal', 'rolling_3', 'rolling_6', 'count', 'Month', 'Year']
REQUIRED = set(KEYS + ['Admin 1', 'Price Date', 'Price', 'Data Type'])
BASELINES = ['Latest price', 'Same month last year']


def month(value):
    return pd.Timestamp(value).to_period('M').to_timestamp()


def clean_data(raw, rates=None, fx=None, as_of=None):
    """Historical FX schema: district, market, currency, Date, Local_per_USD."""
    d = raw.copy()
    d.columns = d.columns.str.strip()
    missing = REQUIRED.difference(d.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    report = {'Input rows': len(d)}
    for c in d.select_dtypes(include=['object', 'string']):
        d[c] = d[c].astype('string').str.strip().replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    report['Duplicate rows'] = int(d.duplicated().sum())
    d = d.drop_duplicates()
    exchange = d['Commodity'].str.casefold().eq('exchange rate').fillna(False)
    report['Exchange-rate rows excluded from targets'] = int(exchange.sum())
    d = d[~exchange & d['Data Type'].str.casefold().eq('aggregated').fillna(False)].copy()
    d['Unit'] = d['Unit'].str.upper().map({'KG': 'KG', 'L': 'L', 'LITRE': 'L', 'LITER': 'L', 'HEAD': 'Head'})
    d['Currency'] = d['Currency'].str.upper()
    d['Price Date'] = pd.to_datetime(d['Price Date'], format='mixed', dayfirst=True, errors='coerce')
    d['Price'] = pd.to_numeric(d['Price'], errors='coerce')
    d = d.dropna(subset=KEYS + ['Price Date', 'Price'])
    d = d[np.isfinite(d['Price']) & d['Price'].gt(0)].copy()
    d['Date'] = d['Price Date'].dt.to_period('M').dt.to_timestamp()
    current = month(as_of if as_of is not None else pd.Timestamp.now(tz='UTC').tz_localize(None))
    report['Incomplete/future month rows excluded'] = int(d['Date'].ge(current).sum())
    d = d[d['Date'] < current].copy()
    if fx is not None:
        f = fx.copy()
        f.columns = f.columns.str.strip()
        keys = ['Admin 2', 'Market Name', 'Currency', 'Date']
        if set(keys + ['Local_per_USD']).difference(f.columns):
            raise ValueError('Rate file needs Admin 2, Market Name, Currency, Date, Local_per_USD.')
        for c in keys[:-1]:
            f[c] = f[c].astype('string').str.strip()
        f['Currency'] = f['Currency'].str.upper()
        f['Date'] = pd.to_datetime(f['Date'], format='mixed', dayfirst=True, errors='coerce').dt.to_period('M').dt.to_timestamp()
        f['Local_per_USD'] = pd.to_numeric(f['Local_per_USD'], errors='coerce')
        if (f[keys + ['Local_per_USD']].isna().any().any() or f.duplicated(keys).any()
                or not np.isfinite(f['Local_per_USD']).all() or not f['Local_per_USD'].gt(0).all()):
            raise ValueError('Rates must be positive, finite and unique per market/currency/month.')
        d = d.merge(f[keys + ['Local_per_USD']], on=keys, how='left', validate='many_to_one')
        d.loc[d['Currency'].eq('USD'), 'Local_per_USD'] = 1.
        report['Currency method'] = 'Historical rates matched exactly by district, market, currency and month'
    else:
        rates = {**(rates or {}), 'USD': 1.}
        if any(not np.isfinite(float(v)) or float(v) <= 0 for v in rates.values()):
            raise ValueError('Fixed rates must be positive and finite.')
        d['Local_per_USD'] = d['Currency'].map(rates)
        report['Currency method'] = 'Fixed-rate scenario: ' + str(rates) if len(rates) > 1 else 'USD records only'
    report['Rows without currency conversion'] = int(d['Local_per_USD'].isna().sum())
    d = d.dropna(subset=['Local_per_USD']).copy()
    d['Price_USD'] = d['Price'] / d['Local_per_USD']
    d = d[np.isfinite(d['Price_USD']) & d['Price_USD'].gt(0)]
    report['Retained rows'] = len(d)
    report['Price spikes'] = 'Preserved; no percentile filtering'
    if d.empty:
        raise ValueError('No usable completed-month prices remain. Check dates, units and currency coverage.')
    return d.reset_index(drop=True), report


def make_panel(cleaned, fuel=None):
    monthly = cleaned.groupby(KEYS + ['Date'], as_index=False).agg(
        Price_USD=('Price_USD', 'mean'), last_observation=('Price Date', 'max'))
    end = monthly['Date'].max()
    pieces = []
    for key, g in monthly.groupby(KEYS, sort=True):
        g = g.set_index('Date').reindex(pd.date_range(g['Date'].min(), end, freq='MS'))
        g.index.name = 'Origin'
        for c, value in zip(KEYS, key):
            g[c] = value
        y = g['Price_USD']
        g['recent'], g['lag_1'], g['lag_2'] = y, y.shift(1), y.shift(2)
        g['rolling_3'] = y.rolling(3, min_periods=1).mean()
        g['rolling_6'] = y.rolling(6, min_periods=1).mean()
        g['count'] = y.notna().cumsum()
        g['gaps'] = y.isna().cumsum()
        for h in (1, 2, 3):
            part = g.copy()
            part['Horizon'] = h
            part['Target Date'] = part.index + pd.offsets.MonthBegin(h)
            part['Month'] = part['Target Date'].dt.month
            part['Year'] = part['Target Date'].dt.year
            part['target'] = y.shift(-h)
            part['seasonal'] = y.shift(12-h)
            pieces.append(part.reset_index())
    panel = pd.concat(pieces, ignore_index=True)
    if fuel:
        f = monthly[monthly['Commodity'].eq(fuel) & monthly['Unit'].eq('L')].copy()
        if f.empty:
            raise ValueError('Selected fuel has no litre observations.')
        f = f.rename(columns={'Date': 'Origin', 'Price_USD': 'fuel_recent'})
        join = ['Admin 2', 'Market Name', 'Currency', 'Origin']
        panel = panel.merge(f[join + ['fuel_recent']], on=join, how='left', validate='many_to_one')
    return panel


def algorithms():
    return {'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1),
            'Support Vector Machine': SVR(C=20, epsilon=.05),
            'XGBoost': XGBRegressor(n_estimators=150, max_depth=5, learning_rate=.05,
                                    subsample=.85, colsample_bytree=.85, random_state=42, n_jobs=2)}


def model(estimator, fuel=False):
    numbers = NUMERIC + (['fuel_recent'] if fuel else [])
    pre = ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=True), KEYS),
        ('numbers', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                              ('scale', StandardScaler())]), numbers)])
    return Pipeline([('preprocessor', pre), ('estimator', clone(estimator))])


def usable(panel):
    return panel[panel['recent'].notna() & panel['count'].ge(12)].copy()


def schedule(panel):
    last = panel['Origin'].max()
    # Three origins per block; two-origin embargo ensures all three-month labels
    # are observed before selecting/calibrating the following block.
    return {name: pd.date_range(last-pd.offsets.MonthBegin(start), periods=3, freq='MS')
            for name, start in [('selection', 15), ('calibration', 10), ('test', 5)]}


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        return {'N': 0, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE (%)': np.nan, 'R2': np.nan}
    return {'N': len(y), 'MAE': mean_absolute_error(y, p), 'RMSE': mean_squared_error(y, p)**.5,
            'MAPE (%)': np.mean(np.abs((y-p)/y))*100,
            'R2': r2_score(y, p) if len(y)>1 and np.var(y)>0 else np.nan}


def walk(panel, origins, estimator=None, fuel=False, baseline='Latest price'):
    d = usable(panel)
    pieces = []
    for origin in origins:
        train = d[d['Target Date'].le(origin) & d['target'].notna()]
        test = d[d['Origin'].eq(origin) & d['target'].notna()].copy()
        if len(train) < 30 or test.empty:
            raise ValueError(f'Insufficient history at {origin:%Y-%m}: need 30 training rows and observed evaluation targets.')
        if estimator is None:
            test['prediction'] = test['recent'] if baseline == 'Latest price' else test['seasonal'].fillna(test['recent'])
        else:
            fitted = model(estimator, fuel).fit(train, train['target'])
            test['prediction'] = np.maximum(fitted.predict(test), 0)
        test['Training labels through'] = train['Target Date'].max()
        pieces.append(test)
    return pd.concat(pieces, ignore_index=True)


def quantile(errors, level):
    a = np.asarray(errors, dtype=float)
    a = a[np.isfinite(a)]
    rank = math.ceil((len(a)+1)*level)
    if len(a)<20 or rank>len(a):
        return np.nan
    return float(np.sort(a)[rank-1])


def intervals(rows, q, commodity_q):
    rows = rows.copy()
    width = rows['Commodity'].map(commodity_q).fillna(q)*rows['recent']
    rows['lower'] = np.maximum(0, rows['prediction']-width)
    rows['upper'] = rows['prediction']+width
    return rows


def detail(rows):
    output = []
    for keys in [['Unit', 'Horizon'], ['Unit', 'Horizon', 'Commodity'],
                 ['Unit', 'Horizon', 'Market Name'], KEYS + ['Horizon']]:
        for key, g in rows.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            valid = g['lower'].notna()
            output.append({'Grouping': ' / '.join(keys), **dict(zip(keys, key)),
                           **metrics(g['target'], g['prediction']), 'Interval N': int(valid.sum()),
                           'Coverage (%)': 100*g.loc[valid, 'target'].between(g.loc[valid, 'lower'], g.loc[valid, 'upper']).mean(),
                           'Mean interval width': (g['upper']-g['lower']).mean(),
                           'Latest-price MAE': metrics(g['target'], g['recent'])['MAE']})
    return pd.DataFrame(output)


def train_one(panel, level=.9, estimators=None):
    if panel['Unit'].nunique()!=1 or panel['Horizon'].nunique()!=1:
        raise ValueError('Each model must use one unit and one horizon.')
    if not 0<level<1:
        raise ValueError('Interval level must be between zero and one.')
    dates = schedule(panel)
    candidates = algorithms() if estimators is None else estimators
    configs, comparisons = {}, []
    variants = [False, True] if 'fuel_recent' in panel and panel['fuel_recent'].notna().any() else [False]
    for name, estimator in candidates.items():
        for fuel in variants:
            label = name + (' + fuel' if fuel else '')
            out = walk(panel, dates['selection'], estimator, fuel)
            configs[label] = (estimator, fuel)
            comparisons.append({'Algorithm': label, **metrics(out['target'], out['prediction'])})
    for baseline in BASELINES:
        out = walk(panel, dates['selection'], baseline=baseline)
        comparisons.append({'Algorithm': baseline, **metrics(out['target'], out['prediction'])})
    table = pd.DataFrame(comparisons).sort_values(['RMSE', 'Algorithm']).reset_index(drop=True)
    best = table.iloc[0]['Algorithm']

    def evaluate(origins):
        return walk(panel, origins, *configs[best]) if best in configs else walk(panel, origins, baseline=best)

    calibration = evaluate(dates['calibration'])
    errors = (calibration['target']-calibration['prediction']).abs()/calibration['recent']
    calibration['error'] = errors
    q = quantile(errors, level)
    commodity_q = {c: quantile(g['error'], level) for c, g in calibration.groupby('Commodity') if len(g)>=30}
    test = intervals(evaluate(dates['test']), q, commodity_q)
    fitted = None
    if best in configs:
        estimator, fuel = configs[best]
        train = usable(panel).dropna(subset=['target'])
        fitted = model(estimator, fuel).fit(train, train['target'])
    test_metrics = [{'Algorithm': best, **metrics(test['target'], test['prediction'])},
                    {'Algorithm': 'Latest price', **metrics(test['target'], test['recent'])},
                    {'Algorithm': 'Same month last year', **metrics(test['target'], test['seasonal'].fillna(test['recent']))}]
    return {'best': best, 'model': fitted, 'selection': table, 'calibration': calibration,
            'test': test, 'test_metrics': pd.DataFrame(test_metrics).drop_duplicates('Algorithm'),
            'details': detail(test), 'q': q, 'commodity_q': commodity_q, 'level': level, 'dates': dates,
            'origin': panel['Origin'].max(), 'unit': panel['Unit'].iloc[0], 'horizon': int(panel['Horizon'].iloc[0])}


def train_all(panel, level=.9, progress=None):
    bundles, skipped = {}, {}
    for (unit, h), part in panel.groupby(['Unit', 'Horizon']):
        if progress:
            progress(f'Training and validating USD/{UNITS[unit]}, horizon {h} month(s)...')
        try:
            bundles[(unit, int(h))] = train_one(part, level)
        except ValueError as exc:
            skipped[(unit, int(h))] = str(exc)
    if not bundles:
        raise ValueError('No forecasts could be trained: '+str(skipped))
    return bundles, skipped


def predict(bundle, panel, selection):
    d = panel.copy()
    for key in KEYS:
        if key not in selection:
            raise ValueError('Select '+key)
        d = d[d[key].eq(selection[key])]
    d = usable(d[d['Origin'].eq(bundle['origin']) & d['Horizon'].eq(bundle['horizon'])])
    if len(d)!=1 or selection['Unit']!=bundle['unit']:
        raise ValueError('This series needs 12 observed months and a price in the latest dataset month.')
    if bundle['model'] is None:
        d['prediction'] = d['recent'] if bundle['best']=='Latest price' else d['seasonal'].fillna(d['recent'])
    else:
        d['prediction'] = np.maximum(bundle['model'].predict(d), 0)
    result = intervals(d, bundle['q'], bundle['commodity_q']).iloc[0].to_dict()
    result['change_pct'] = 100*(result['prediction']/result['recent']-1)
    result['algorithm'] = bundle['best']
    result['level'] = bundle['level']
    local = bundle['test'].copy()
    for k in KEYS:
        local = local[local[k].eq(selection[k])]
    result['local_metrics'] = metrics(local['target'], local['prediction'])
    result['local_benchmark'] = metrics(local['target'], local['recent'])
    result['coverage'] = bundle['details'].iloc[0]['Coverage (%)']
    result['interval_n'] = bundle['details'].iloc[0]['Interval N']
    result['interval_scope'] = 'commodity within unit' if selection['Commodity'] in bundle['commodity_q'] else 'pooled within unit'
    return result


def explain(r, conversion=''):
    u = UNITS[r['Unit']]
    text = (f"{r['Commodity']} in {r['Market Name']} ({r['Admin 2']}) is forecast at ${r['prediction']:.2f}/{u} "
            f"for {r['Target Date']:%B %Y}, {r['Horizon']} month(s) after {r['Origin']:%B %Y}. "
            f"The latest monthly average is ${r['recent']:.2f}/{u}, giving an expected change of {r['change_pct']:+.1f}%. "
            f"The latest source observation is {r['last_observation']:%d %B %Y}. ")
    if pd.notna(r['lower']):
        text += (f"The estimated {r['level']:.0%} interval is ${r['lower']:.2f}–${r['upper']:.2f}/{u}, "
                 f"calibrated for this horizon using residuals {r['interval_scope']}. ")
        if r['interval_n']:
            text += f"Final-test coverage for this unit and horizon was {r['coverage']:.1f}% across {int(r['interval_n'])} predictions. "
    else:
        text += 'The interval is unavailable because calibration observations are insufficient. '
    m = r['local_metrics']
    if m['N']:
        text += (f"Local final-test MAE was ${m['MAE']:.3f}/{u} across {m['N']} forecasts, versus "
                 f"${r['local_benchmark']['MAE']:.3f}/{u} for the latest-price benchmark. This small sample limits reliability conclusions. ")
    else:
        text += 'Local final-test accuracy is unverified. '
    text += (f"Selected method: {r['algorithm']}. Historical missing months: {int(r['gaps'])}. "
             'Use these estimates for preliminary procurement budgeting, allow for uncertainty, and verify supplier quotations. '
             'Rainfall is excluded. The model does not establish the cause of price changes or guarantee accuracy during shocks. '
             'Intervals for the three months are separate, not a joint guarantee. '+conversion+'. ')
    if r['Origin'] < month(pd.Timestamp.now(tz='UTC').tz_localize(None))-pd.offsets.MonthBegin(1):
        text += 'The source data are stale; upload recent observations before making current decisions.'
    return text
