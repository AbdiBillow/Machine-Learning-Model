# Somalia Food Price Forecasting

Forecast each of the next three months in USD per kg, litre or head. The app and
standalone notebook share the same Python forecasting core. Rainfall is excluded.

[Open standalone notebook in Colab](https://colab.research.google.com/github/AbdiBillow/Machine-Learning-Model/blob/main/machine_learning_project.ipynb)

## Run
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
Upload the WFP CSV with Admin 1, Admin 2, Market Name, Commodity, Price Date,
Price, Unit, Currency and Data Type. Aggregated records from completed months
are retained. Duplicate, incomplete, unsupported-unit and non-positive/non-finite
records are excluded. Genuine price spikes remain. Monthly means are calculated
separately by district, market, commodity, unit and currency.

## Forecasts and validation
- Direct models predict horizons 1, 2 and 3 from the same observed history.
  No actual intermediate future price is used to predict month three.
- Inputs: latest monthly price, earlier prices, corresponding seasonal price,
  3- and 6-month rolling means, observed-month count and target month/year.
  Missing calendar months are inserted before creating features. Missing targets
  are never imputed; feature imputation and scaling are fitted inside each fold.
- Optional fuel commodity in litres is matched by district/market/currency at
  the forecast origin. Each algorithm is compared with and without that input.
- Each unit/horizon compares Linear Regression, Random Forest, SVR and XGBoost
  with latest-price and seasonal benchmarks. Missing seasonal prices fall back
  to latest price, consistently in selection, evaluation and live prediction.
- Three forecast origins select the lowest-RMSE method; three later origins
  calibrate intervals; three final origins evaluate it. Two-month gaps between
  stages ensure three-month labels are known before the next stage starts.
- At every historical origin, training uses only targets dated at or before
  that origin. Final-test results do not select the model or calibrate intervals.
  Later test origins may use earlier observed outcomes, simulating monthly refits.
- A series needs 12 observed months and a price in the dataset's latest month.
  Each fold requires 30 available training rows. Sparse unit/horizon groups are
  reported as unavailable. The selected method is refitted on available history.
- Intervals use corrected empirical quantiles of absolute calibration errors
  divided by the latest price, separately for every horizon and unit. At least
  20 residuals are required. Commodity calibration needs 30 observations;
  otherwise errors are pooled within that unit/horizon. Temporal dependence,
  revisions and shocks mean nominal coverage is not guaranteed.
- Detailed final-test tables report N, MAE, RMSE, MAPE, R², interval N/coverage/width
  and benchmark errors by unit, horizon, commodity, market and exact series.
- The prediction table shows one price and interval for each of the next three
  months. Context explains latest observations, expected change, local errors,
  missing months, uncertainty and procurement implications without claiming causes.
- Quantity estimates, configurable increase alerts and a planning basket are
  included. Item/month intervals are not a joint basket or three-month interval.

## Currency conversion
Default: USD observations plus a verified historical-rate CSV, if supplied.
Rate columns: Admin 2, Market Name, Currency, Date, Local_per_USD. The rate must
be local currency units per one USD and unique for each district/market/currency/month.
Matches are exact; missing matches are counted and excluded. No future/backfilled
rate is used. Embedded 'exchange rate' rows are counted and excluded from price
targets; their direction is not guessed. A fixed-rate scenario can be explicitly
enabled with user-supplied SOS/SLS rates; it is not verified historical USD pricing.

## Practical limits
Forecast origin is the latest completed month in the uploaded data. Stale datasets
produce dated forecasts with a warning; a stale individual series is rejected.
Backtests assume month-end availability of data; publication delays and revisions
require vintage data for proper real-time evaluation. Synthetic tests verify the
implementation, not Somalia forecast accuracy or superiority to FSNAU.

## Tests
```bash
python -m pip install pytest
python -m pytest -q
```
Tests cover calendar gaps, price spikes, currency coverage, future-label leakage
at all three horizons, selection/calibration isolation, four estimators, fuel,
intervals, stale series, notebook/core agreement and the Streamlit upload/train/predict flow.

The repository notebook contains the standalone Python core and preserves the
original notebook's cell IDs. Saved old results are cleared; rerun the cells with
your CSV. User data, old notebook outputs and trained models are not published in
this repository.
