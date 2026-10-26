"""
The final VOLTA model, run exactly like notebooks/final_model.ipynb (Step 15).

- Input 1: the past 168 hours of [scaled price, is_weekend]
- Input 2: is_weekend for the 24 hours being forecast
- Output : 24 hourly prices (00:00-23:00 UTC), converted back to EUR/MWh
- Scaler : MinMaxScaler fitted on 2015-2017 (models/volta_scaler_final.pkl)

No Streamlit code here, so the same functions can be used from a notebook.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = APP_DIR.parent

TARGET = "price actual"
INPUT_WINDOW = 168      # past hours the model sees
H = 24                  # hours it predicts
TZ = "Europe/Madrid"    # is_weekend and all displayed hours use Spain time
TRAIN_START = pd.Timestamp("2015-01-01", tz="UTC")
TEST_YEAR = 2018        # the final model was trained on 2015-2017 only


def _first_existing(*paths: Path) -> Path:
    """Use the copy inside streamlit_app/ if there is one, otherwise the repo copy."""
    for p in paths:
        if p.exists():
            return p
    return paths[-1]


DATA_FILE = _first_existing(APP_DIR / "data" / "volta_clean_hourly.csv",
                            REPO_DIR / "data" / "processed" / "volta_clean_hourly.csv")
MODEL_FILE = _first_existing(APP_DIR / "models" / "volta_lstm_final.keras",
                             REPO_DIR / "models" / "volta_lstm_final.keras")
SCALER_FILE = _first_existing(APP_DIR / "models" / "volta_scaler_final.pkl",
                              REPO_DIR / "models" / "volta_scaler_final.pkl")


def load_prices(path: Path = DATA_FILE) -> pd.Series:
    """Hourly 'price actual' indexed by UTC time."""
    df = pd.read_csv(path, usecols=["time", TARGET])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").set_index("time")[TARGET]


def load_model():
    """Return (keras model, scaler). TensorFlow is imported here because it is slow."""
    import joblib
    from tensorflow import keras

    return keras.models.load_model(MODEL_FILE), joblib.load(SCALER_FILE)


def weekend_flag(index: pd.DatetimeIndex) -> np.ndarray:
    """1 on Saturday/Sunday in Spain time, else 0 (same as the notebook)."""
    return (index.tz_convert(TZ).dayofweek >= 5).astype(int)


def local_hour_labels(timestamps) -> list[str]:
    """
    "HH:MM" in Spain time (summer and winter time) for a list of UTC timestamps.

    - Hours on a later local day than the first one get " (+1)".
    - On the October clock change one local hour happens twice, so those two
      labels also get their zone ("02:00 CEST", "02:00 CET") to stay unique.
    """
    local = pd.DatetimeIndex(timestamps).tz_convert(TZ)
    first_day = local[0].date()
    labels = pd.Series([t.strftime("%H:%M") + (" (+1)" if t.date() > first_day else "")
                        for t in local])
    repeated = labels.duplicated(keep=False)
    return [f"{label} {t.strftime('%Z')}" if rep else label
            for label, t, rep in zip(labels, local, repeated)]


def forecast_days(price: pd.Series) -> pd.DatetimeIndex:
    """Every day (00:00 UTC) that has 168 past hours and 24 known future hours."""
    first = TRAIN_START + pd.Timedelta(hours=INPUT_WINDOW)
    days = pd.date_range(first, price.index.max().floor("D"), freq="D")
    return days[days + pd.Timedelta(hours=H - 1) <= price.index.max()]


def predict_all(price: pd.Series, model, scaler) -> pd.DataFrame:
    """
    Run the model once for every forecast day.

    Returns one row per forecast hour with the prediction, the real price and
    the three baselines from the notebook (all use only prices known at 00:00 UTC).
    """
    steps = price.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all(), "price series must be hourly with no gaps"

    scaled = scaler.transform(price.values.reshape(-1, 1)).ravel()
    features = np.column_stack([scaled, weekend_flag(price.index)])  # (hours, 2)

    days = forecast_days(price)
    pos = price.index.get_indexer(days)
    X_past = np.stack([features[p - INPUT_WINDOW:p] for p in pos])          # (days, 168, 2)
    X_future = np.stack([features[p:p + H, 1:] for p in pos])               # (days, 24, 1)

    pred_scaled = model.predict([X_past, X_future], batch_size=256, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    ts = days.repeat(H) + pd.to_timedelta(np.tile(np.arange(H), len(days)), unit="h")
    out = pd.DataFrame({
        "day": days.repeat(H),
        "timestamp": ts,
        "local_time": ts.tz_convert(TZ).tz_localize(None),
        "predicted": pred,
        "actual": price.reindex(ts).values,
        "lag_24": price.reindex(ts - pd.Timedelta(hours=24)).values,
        "lag_168": price.reindex(ts - pd.Timedelta(hours=168)).values,
        "last_value": price.reindex(days.repeat(H) - pd.Timedelta(hours=1)).values,
    })
    out["split"] = np.where(out["day"].dt.year == TEST_YEAR, "Test (2018)", "Training (2015-2017)")
    return out


METHODS = {
    "predicted": "Final LSTM",
    "lag_24": "Lag-24 (same hour yesterday)",
    "lag_168": "Lag-168 (same hour last week)",
    "last_value": "Last value (hour before forecast)",
}


def score(df: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE (EUR/MWh) of the model and each baseline, plus the LSTM improvement."""
    rows = {}
    for col, name in METHODS.items():
        err = df["actual"] - df[col]
        rows[name] = {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean())}
    table = pd.DataFrame(rows).T
    lstm_mae = table.loc["Final LSTM", "MAE"]
    table["LSTM improvement (%)"] = (1 - lstm_mae / table["MAE"]) * 100
    table.loc["Final LSTM", "LSTM improvement (%)"] = np.nan
    return table.sort_values("MAE")


def find_cheapest_hours(predictions, power_kw: float, hours: int) -> dict:
    """
    Step 16 of the notebook: the cheapest block of `hours` consecutive hours.
    `start`/`end` are positions inside the 24 forecast hours (end is exclusive).
    """
    predictions = np.asarray(predictions, dtype=float)
    costs = [power_kw * hours * np.mean(predictions[i:i + hours]) / 1000
             for i in range(len(predictions) - hours + 1)]
    start = int(np.argmin(costs))
    return {"start": start, "end": start + hours, "cost": float(costs[start]),
            "worst_cost": float(np.max(costs))}
