"""
The final VOLTA model, run exactly like notebooks/final_model.ipynb (Step 15).

- Input 1: the past 168 hours of [scaled price, is_weekend]
- Input 2: is_weekend for the 24 hours being forecast
- Output : 24 hourly prices (00:00-23:00 UTC), converted back to EUR/MWh
- Scaler : MinMaxScaler fitted on 2015-2017 (models/volta_scaler_final.pkl)
- Checks : bad prices, a scaler that does not match training, or inputs/outputs
           with the wrong shape stop with a clear ValueError (nothing is fixed silently)

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
TRAIN_END = pd.Timestamp("2017-12-31 23:00", tz="UTC")  # last hour the final model and scaler saw
TEST_YEAR = 2018        # the final model was trained on 2015-2017 only
PAST_SHAPE = (INPUT_WINDOW, 2)  # input 1, per day: [scaled price, is_weekend]
FUTURE_SHAPE = (H, 1)           # input 2, per day: [is_weekend]


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

    model, scaler = keras.models.load_model(MODEL_FILE), joblib.load(SCALER_FILE)
    validate_scaler(scaler, load_prices())
    return model, scaler


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


# ---------------------------------------------------------------------------
# Checks. They never fill, drop or change data: they stop with a clear
# ValueError when something would make the forecast wrong. (Unlike `assert`,
# they still run when Python is started with -O.)
# ---------------------------------------------------------------------------
def _first(index: pd.Index, mask) -> str:
    """The first entry of `index` where `mask` is True, for error messages."""
    return str(index[np.flatnonzero(mask)[0]])


def validate_prices(price: pd.Series) -> None:
    """Stop if the hourly price series cannot be given to the model as it is."""
    idx = price.index
    if price.empty:
        raise ValueError("No prices to forecast from: the price series is empty.")
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("Prices must be indexed by timezone-aware times (UTC), as load_prices() returns.")

    if not pd.api.types.is_numeric_dtype(price):
        text = pd.to_numeric(price, errors="coerce").isna() & price.notna()
        where = (f" First: {price.iloc[np.flatnonzero(text)[0]]!r} at {_first(idx, text)}."
                 if text.any() else "")
        raise ValueError(f"'{TARGET}' must contain numbers only.{where}")

    bad = ~np.isfinite(price.to_numpy(dtype=float))
    if bad.any():
        raise ValueError(f"'{TARGET}' has {bad.sum()} missing or infinite value(s), first at "
                         f"{_first(idx, bad)}. Fix the data: nothing is filled automatically.")

    dup = idx.duplicated()
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicated time(s), first: {_first(idx, dup)}.")

    steps = idx[1:] - idx[:-1]
    back = steps <= pd.Timedelta(0)
    if back.any():
        i = np.flatnonzero(back)[0]
        raise ValueError(f"Times are not in increasing order: {idx[i + 1]} comes after {idx[i]}.")
    irregular = steps != pd.Timedelta(hours=1)
    if irregular.any():
        i = np.flatnonzero(irregular)[0]
        raise ValueError(f"Prices must be hourly with no missing hours: {steps[i]} between {idx[i]} "
                         f"and {idx[i + 1]} ({irregular.sum()} irregular step(s) in total).")


def validate_scaler(scaler, price: pd.Series) -> None:
    """
    Stop unless `scaler` matches the one fitted in notebooks/final_model.ipynb (cell 77):
    a MinMaxScaler with default settings, fitted on the price column only, on every
    hour of 2015-2017. The training prices are read from `price` and only compared,
    never used to fit anything.
    """
    from sklearn.preprocessing import MinMaxScaler

    if not isinstance(scaler, MinMaxScaler) or not hasattr(scaler, "data_min_"):
        raise ValueError(f"Expected a fitted MinMaxScaler, got {type(scaler).__name__}.")
    if scaler.n_features_in_ != 1:
        raise ValueError(f"The scaler was fitted on {scaler.n_features_in_} columns; the model uses 1 (price).")
    if tuple(scaler.feature_range) != (0, 1) or scaler.clip:
        raise ValueError(f"Unexpected scaler settings feature_range={scaler.feature_range}, "
                         f"clip={scaler.clip}; training used the defaults (0, 1) and clip=False.")

    train = price.loc[TRAIN_START:TRAIN_END]
    expected_rows = int((TRAIN_END - TRAIN_START) / pd.Timedelta(hours=1)) + 1
    if len(train) != expected_rows:
        raise ValueError(f"Cannot check the scaler: the data has {len(train)} of the {expected_rows} "
                         f"training hours ({TRAIN_START} to {TRAIN_END}).")
    found = (int(np.ravel(scaler.n_samples_seen_)[0]), scaler.data_min_[0], scaler.data_max_[0])
    wanted = (expected_rows, train.min(), train.max())
    if found[0] != wanted[0] or not np.isclose(found[1:], wanted[1:]).all():
        raise ValueError("The scaler does not match the 2015-2017 training prices "
                         f"(rows, min, max): scaler {found}, training data {wanted}.")


def check_model_inputs(model, X_past: np.ndarray, X_future: np.ndarray) -> None:
    """Stop if the inputs do not have the shapes and feature order the model was trained with."""
    n = len(X_past)
    if X_past.shape != (n, *PAST_SHAPE) or X_future.shape != (n, *FUTURE_SHAPE):
        raise ValueError(f"Model inputs have shapes {X_past.shape} and {X_future.shape}; "
                         f"expected (days, {INPUT_WINDOW}, 2) and (days, {H}, 1).")
    # is_weekend is the 2nd past column and the only future column, so it must be 0/1.
    # If the columns were swapped, the scaled price would show up here instead.
    for name, flags in [("past", X_past[:, :, 1]), ("future", X_future[:, :, 0])]:
        if not np.isin(flags, (0, 1)).all():
            raise ValueError(f"The {name} is_weekend column is not 0/1: the feature order is wrong.")
    # A saved Keras model knows the input shapes it was trained with: compare them too
    if hasattr(model, "inputs"):
        model_shapes = [tuple(i.shape[1:]) for i in model.inputs]
        if model_shapes != [PAST_SHAPE, FUTURE_SHAPE]:
            raise ValueError(f"This model expects inputs {model_shapes}, but VOLTA builds "
                             f"{[PAST_SHAPE, FUTURE_SHAPE]}. Was a different model file saved?")


def check_model_output(pred_scaled, n_days: int) -> None:
    """Stop unless the model returned 24 real numbers for every forecast day."""
    pred_scaled = np.asarray(pred_scaled)
    if pred_scaled.shape != (n_days, H):
        raise ValueError(f"The model returned shape {pred_scaled.shape}; expected ({n_days}, {H}): "
                         f"{H} prices per day.")
    bad = ~np.isfinite(pred_scaled)
    if bad.any():
        raise ValueError(f"The model returned {bad.sum()} NaN or infinite value(s).")


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
    validate_prices(price)

    scaled = scaler.transform(price.values.reshape(-1, 1)).ravel()
    features = np.column_stack([scaled, weekend_flag(price.index)])  # (hours, 2)

    days = forecast_days(price)
    pos = price.index.get_indexer(days)
    if len(days) == 0:
        raise ValueError(f"Not enough data to forecast: a day needs {INPUT_WINDOW} past hours and "
                         f"its {H} hours (data covers {price.index.min()} to {price.index.max()}).")
    short = pos < INPUT_WINDOW  # -1 means the day is not in the data at all
    if short.any():
        raise ValueError(f"{short.sum()} forecast day(s) lack the {INPUT_WINDOW} past hours the model "
                         f"needs, first: {days[short][0].date()} (data starts {price.index.min()}).")
    X_past = np.stack([features[p - INPUT_WINDOW:p] for p in pos])          # (days, 168, 2)
    X_future = np.stack([features[p:p + H, 1:] for p in pos])               # (days, 24, 1)
    check_model_inputs(model, X_past, X_future)

    pred_scaled = model.predict([X_past, X_future], batch_size=256, verbose=0)
    check_model_output(pred_scaled, len(days))
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
