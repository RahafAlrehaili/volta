# VOLTA – Streamlit App

Predicting tomorrow's electricity price in Spain. The app runs the final model from
`notebooks/final_model.ipynb` (LSTM + `is_weekend`, trained on 2015–2017, tested on 2018).
Every number in the app is computed from the model's predictions.

**Live app:** https://volta-app.streamlit.app/

All forecast hours are shown in Spain time (Europe/Madrid); `(+1)` marks hours on the next day.

## Folder structure

```
streamlit_app/
├─ app.py                      The app: forecast & device optimizer, forecast insights, about
├─ requirements.txt
├─ Dockerfile                  Runs the app in a container
├─ .streamlit/config.toml      Theme (presentation colours)
├─ data/volta_clean_hourly.csv Hourly dataset
├─ models/
│  ├─ volta_lstm_final.keras   Final LSTM
│  └─ volta_scaler_final.pkl   MinMaxScaler fitted on 2015–2017
├─ volta/
│  ├─ forecast.py              Model inputs, checks, predictions, metrics, find_cheapest_hours (no Streamlit)
│  └─ cache.py                 Loads data/model once and caches the predictions
└─ tests/
   ├─ test_forecast_validation.py  Data, scaler, model input/output and feature-order checks
   ├─ test_app_errors.py           Error messages and logging when loading or forecasting fails
   └─ test_cheapest_hours.py       Cheapest-hours recommendation
```

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. The first load takes a few seconds while TensorFlow
loads and the model predicts every day.

## Run with Docker

```
docker build -t volta .
docker run -p 8501:8501 volta
```

The container listens on `$PORT` if it is set (default 8501).

## Tests

```
pip install pytest
python -m pytest tests -v
```

Most tests use a fake model, so they are fast; one test loads the real LSTM.

## Errors

If the data, scaler or model cannot be loaded, or a check in `volta/forecast.py` fails,
the app shows a short message and stops. The full error is written to the logs
(the terminal locally, **Manage app → Logs** on Streamlit Cloud).

## Deploy (Streamlit Community Cloud)

1. Push to GitHub.
2. On share.streamlit.io: **Create app** → repository `Ghadi213/volta`, branch `main`,
   main file `streamlit_app/app.py`.
3. In **Advanced settings** choose Python 3.12, then **Deploy**.

## Updating the model

Save the new model and scaler from the notebook into `streamlit_app/models/` with the
same file names. If the inputs change (window length, extra features), update
`volta/forecast.py` so it builds the inputs exactly like the notebook.
