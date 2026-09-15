import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import forecasting as f


def raw_prices(markets=10, months=48, units=('KG',)):
    return pd.DataFrame([{'Admin 1':'Region','Admin 2':'District','Market Name':f'Market {m}',
        'Commodity':'Maize' if u=='KG' else 'Fuel','Price Date':d.strftime('%d/%m/%Y'),
        'Price':1+.02*i+.1*m+.04*np.sin(i),'Unit':u,'Currency':'USD','Data Type':'Aggregated'}
        for u in units for m in range(markets)
        for i,d in enumerate(pd.date_range('2020-01-01',periods=months,freq='MS'))])


def prepared(raw=None, fuel=None):
    d,_=f.clean_data(raw_prices() if raw is None else raw, as_of='2024-01-01')
    return f.make_panel(d, fuel)


def small_models():
    return {'Linear Regression':LinearRegression(),
            'Random Forest':RandomForestRegressor(n_estimators=5,random_state=42),
            'Support Vector Machine':SVR(),
            'XGBoost':XGBRegressor(n_estimators=5,random_state=42,n_jobs=1)}


def test_calendar_gaps_and_price_spikes():
    raw=raw_prices(markets=1)
    raw.loc[10,'Price']=10000
    raw=raw.drop(index=15)
    d,_=f.clean_data(raw,as_of='2024-01-01')
    assert d['Price_USD'].max()==10000
    p=f.make_panel(d)
    after=p[(p['Origin']==pd.Timestamp('2021-05-01'))&(p['Horizon']==3)].iloc[0]
    assert pd.isna(after['lag_1']) and after['lag_2']>0
    missing=p[p['Origin']==pd.Timestamp('2021-04-01')]
    assert f.usable(missing).empty


def test_currency_matching_and_invalid_rates():
    raw=raw_prices(markets=1,months=3)
    raw['Currency']='SOS';raw['Price']=26000
    fx=pd.DataFrame([{'Admin 2':'District','Market Name':'Market 0','Currency':'SOS',
                      'Date':'2020-01-01','Local_per_USD':26000}])
    d,report=f.clean_data(raw,fx=fx,as_of='2024-01-01')
    assert d['Price_USD'].tolist()==[1]
    assert report['Rows without currency conversion']==2
    with pytest.raises(ValueError,match='unique'):
        f.clean_data(raw,fx=pd.concat([fx,fx]),as_of='2024-01-01')
    with pytest.raises(ValueError,match='positive'):
        f.clean_data(raw,rates={'SOS':0},as_of='2024-01-01')


def test_current_future_forecast_and_invalid_rows_excluded():
    raw=raw_prices(markets=1,months=6)
    raw.loc[0,'Data Type']='Forecast';raw.loc[1,'Price']=np.inf
    d,r=f.clean_data(raw,as_of='2020-05-20')
    assert len(d)==2 and d['Date'].max()==pd.Timestamp('2020-04-01')
    assert r['Incomplete/future month rows excluded']==2


@pytest.mark.parametrize('h',[1,2,3])
def test_direct_horizon_does_not_use_future_prices(h):
    raw=raw_prices()
    origin=pd.Timestamp('2022-08-01')
    before=prepared(raw)
    raw.loc[pd.to_datetime(raw['Price Date'],dayfirst=True).gt(origin),'Price']*=100
    after=prepared(raw)
    mask=(before['Origin']==origin)&(before['Horizon']==h)
    pd.testing.assert_frame_equal(before.loc[mask,f.KEYS+f.NUMERIC],after.loc[mask,f.KEYS+f.NUMERIC])
    out=f.walk(before[before['Horizon']==h],[origin],LinearRegression())
    out2=f.walk(after[after['Horizon']==h],[origin],LinearRegression())
    np.testing.assert_allclose(out['prediction'],out2['prediction'])
    assert (out['Training labels through']<=origin).all()
    assert (out['Target Date']==origin+pd.offsets.MonthBegin(h)).all()


def test_test_labels_do_not_change_selection_or_calibration():
    raw=raw_prices();p=prepared(raw);p=p[p['Horizon']==3]
    b=f.train_one(p,estimators={'Linear Regression':LinearRegression()})
    raw.loc[pd.to_datetime(raw['Price Date'],dayfirst=True).gt(b['dates']['test'][0]),'Price']*=100
    p2=prepared(raw);p2=p2[p2['Horizon']==3]
    b2=f.train_one(p2,estimators={'Linear Regression':LinearRegression()})
    pd.testing.assert_frame_equal(b['selection'],b2['selection'])
    assert b['q']==b2['q'] and b['best']==b2['best']
    assert b['dates']['selection'][-1]+pd.offsets.MonthBegin(3)<=b['dates']['calibration'][0]
    assert b['dates']['calibration'][-1]+pd.offsets.MonthBegin(3)<=b['dates']['test'][0]


def test_three_month_predictions_all_algorithms_and_stale_series():
    p=prepared()
    selected=p.iloc[0][f.KEYS].to_dict()
    for h in (1,2,3):
        b=f.train_one(p[p['Horizon']==h],estimators=small_models())
        assert set(small_models()).issubset(set(b['selection']['Algorithm']))
        r=f.predict(b,p,selected)
        assert r['Target Date']==pd.Timestamp('2023-12-01')+pd.offsets.MonthBegin(h)
        assert 0<=r['lower']<=r['prediction']<=r['upper']
        assert r['local_metrics']['N']==3
        assert 'Rainfall is excluded' in f.explain(r)
    raw=raw_prices()
    raw=raw[~(raw['Market Name'].eq('Market 0')&raw['Price Date'].eq('01/12/2023'))]
    with pytest.raises(ValueError,match='latest dataset month'):
        f.predict(b,prepared(raw),selected)


def test_calibration_samples_and_fuel():
    assert np.isnan(f.quantile(range(19),.9))
    assert f.quantile(range(20),.9)==18
    raw=raw_prices(units=('KG','L'))
    p=prepared(raw,fuel='Fuel')
    row=p[(p['Unit']=='KG')&(p['Market Name']=='Market 0')&(p['Origin']==pd.Timestamp('2021-02-01'))].iloc[0]
    expected=raw[(raw['Unit']=='L')&(raw['Market Name']=='Market 0')&(raw['Price Date']=='01/02/2021')]['Price'].iloc[0]
    assert row['fuel_recent']==expected
    b=f.train_one(p[(p['Unit']=='KG')&(p['Horizon']==3)],estimators={'Linear Regression':LinearRegression()})
    assert 'Linear Regression + fuel' in b['selection']['Algorithm'].tolist()


def test_notebook_core_and_syntax():
    n=json.loads(Path('machine_learning_project.ipynb').read_text())
    code=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert Path('forecasting.py').read_text() in code
    for s in code:
        if '%pip' not in s:
            compile(s,'notebook','exec')


def test_streamlit_upload_train_predict(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(f,'algorithms',small_models)
    payload=raw_prices().to_csv(index=False).encode()
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k: None if k.get('key')=='fx' else io.BytesIO(payload))
    app=AppTest.from_file('app.py',default_timeout=120).run()
    assert not app.exception
    next(b for b in app.button if b.label=='Train and compare models').click().run()
    assert not app.exception
    next(b for b in app.button if b.label=='Predict next three months').click().run()
    assert not app.exception
    assert 'Predicted USD per kg' in [m.label for m in app.metric]
    assert len(app.session_state['results'])==3
