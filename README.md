Somalia Food Price Prediction App
This Streamlit app cleans WFP food-price data, compares XGBoost, Linear Regression,
Random Forest, and Support Vector Regression, selects the model with the lowest
chronological test RMSE, retrains it on all historical observations, and predicts
the USD price of 1 kg of a selected commodity.
Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
Upload `WFP Food Price Original Data.csv` when prompted.
Modeling decisions
Only `Unit = KG` rows are used, so predictions represent 1 kg.
Only historical `Data Type = Aggregated` rows are used; source forecast rows are excluded.
Dates are parsed into month and year.
Duplicate, incomplete, non-positive, and extreme commodity-level observations are removed.
SOS and SLS prices are converted to USD using editable exchange-rate assumptions in the sidebar.
Evaluation uses the latest 20% of observations as a chronological holdout.
The best model is selected using the lowest RMSE and then retrained on all cleaned data.
Required CSV columns
`Admin 1`, `Admin 2`, `Market Name`, `Commodity`, `Price Date`, `Price`, `Unit`,
`Currency`, and `Data Type`.
