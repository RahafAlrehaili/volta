"""
Tests for the checks in volta/forecast.py.

Run from streamlit_app/:   python -m pytest tests -v

Most tests use a fake model (no TensorFlow), so they are fast. The last test
loads the real LSTM once to check the checks accept the real data and model.
"""
import io
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from volta import forecast as fc  # noqa: E402


# ---------------------------------------------------------------- helpers ---
def hourly(start="2015-01-01", hours=24 * 10):
    """A clean hourly price series in UTC (10 days from 2015-01-01 = 3 forecast days)."""
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.Series(50 + 10 * np.sin(np.arange(hours) / 5), index=idx)


class FakeModel:
    """Stands in for the LSTM: returns `value` (scaled) for every hour."""

    def __init__(self, n_out=fc.H, value=0.5, input_shapes=None):
        self.n_out, self.value = n_out, value
        if input_shapes is not None:  # like a Keras model's .inputs
            self.inputs = [SimpleNamespace(shape=(None, *s)) for s in input_shapes]

    def predict(self, inputs, **kwargs):
        self.seen = [x.shape for x in inputs]
        return np.full((len(inputs[0]), self.n_out), self.value, dtype=float)


@pytest.fixture(scope="module")
def scaler():
    return joblib.load(fc.SCALER_FILE)


@pytest.fixture(scope="module")
def real_prices():
    return fc.load_prices()


def train_prices(real_prices):
    return real_prices.loc[fc.TRAIN_START:fc.TRAIN_END].values.reshape(-1, 1)


def ts(text):
    return re.escape(str(pd.Timestamp(text, tz="UTC")))


# ------------------------------------------------------------- good data ---
def test_good_data_gives_24_prices_per_day(scaler):
    model = FakeModel()
    out = fc.predict_all(hourly(), model, scaler)
    assert model.seen == [(3, 168, 2), (3, 24, 1)]
    assert len(out) == 3 * 24
    # 0.5 (scaled) back to EUR/MWh with the saved scaler
    assert np.allclose(out["predicted"], scaler.inverse_transform([[0.5]])[0, 0])


def test_model_with_matching_input_shapes_passes(scaler):
    model = FakeModel(input_shapes=[fc.PAST_SHAPE, fc.FUTURE_SHAPE])
    fc.predict_all(hourly(), model, scaler)


def test_real_csv_passes(real_prices):
    fc.validate_prices(real_prices)


def test_real_scaler_matches_training(scaler, real_prices):
    fc.validate_scaler(scaler, real_prices)


def test_scaler_fitted_the_same_way_as_the_notebook_passes(real_prices):
    # The check must not reject a correct scaler: refitting exactly like cell 77
    # gives the same scaler, so it has to pass too.
    fc.validate_scaler(MinMaxScaler().fit(train_prices(real_prices)), real_prices)


# ------------------------------------------------------------ bad prices ---
def test_missing_price(scaler):
    price = hourly()
    price.iloc[100] = np.nan
    with pytest.raises(ValueError, match="missing or infinite.*" + ts("2015-01-05 04:00")):
        fc.predict_all(price, FakeModel(), scaler)


def test_infinite_price():
    price = hourly()
    price.iloc[5] = np.inf
    with pytest.raises(ValueError, match="missing or infinite"):
        fc.validate_prices(price)


def test_non_numeric_price_from_csv(scaler):
    rows = [f"{t},{p},x" for t, p in hourly().items()]
    rows[20] = rows[20].split(",")[0] + ",abc,x"
    price = fc.load_prices(io.StringIO("time,price actual,other\n" + "\n".join(rows)))
    with pytest.raises(ValueError, match="numbers only.*'abc'.*" + ts("2015-01-01 20:00")):
        fc.predict_all(price, FakeModel(), scaler)


def test_empty_series():
    with pytest.raises(ValueError, match="empty"):
        fc.validate_prices(pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC")))


def test_times_without_timezone():
    price = hourly()
    price.index = price.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        fc.validate_prices(price)


def test_duplicated_time():
    price = hourly()
    price = pd.concat([price, price.iloc[[50]]]).sort_index()
    with pytest.raises(ValueError, match="duplicated.*" + ts("2015-01-03 02:00")):
        fc.validate_prices(price)


def test_times_out_of_order():
    price = hourly()
    price = pd.concat([price.iloc[:5], price.iloc[[6, 5]], price.iloc[7:]])
    with pytest.raises(ValueError, match="increasing order.*" + ts("2015-01-01 05:00")):
        fc.validate_prices(price)


def test_missing_hour(scaler):
    price = hourly().drop(pd.Timestamp("2015-01-02 06:00", tz="UTC"))
    with pytest.raises(ValueError, match="missing hours.*" + ts("2015-01-02 05:00")
                       + " and " + ts("2015-01-02 07:00")):
        fc.predict_all(price, FakeModel(), scaler)


def test_checks_still_run_with_python_O():
    # `python -O` removes every assert; the new checks must still stop bad data.
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "assert False, 'asserts are ON, so -O did not work'\n"
        "import pandas as pd\n"
        "from volta import forecast as fc\n"
        "s = pd.Series(1.0, index=pd.date_range('2015-01-01', periods=5, freq='h', tz='UTC'))\n"
        "fc.validate_prices(s.drop(s.index[2]))\n" % str(APP_DIR)
    )
    run = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert run.returncode != 0
    assert "missing hours" in run.stderr


# ------------------------------------------------------- not enough data ---
def test_too_little_data(scaler):
    with pytest.raises(ValueError, match="Not enough data"):
        fc.predict_all(hourly(hours=100), FakeModel(), scaler)


def test_data_starting_too_late(scaler):
    with pytest.raises(ValueError, match="lack the 168 past hours.*2015-01-08"):
        fc.predict_all(hourly(start="2015-01-05"), FakeModel(), scaler)


# ------------------------------------------------- model inputs/outputs ---
def make_inputs(days=2):
    X_past = np.zeros((days, 168, 2))
    X_past[:, :, 0] = 0.4  # scaled price
    X_future = np.ones((days, 24, 1))
    return X_past, X_future


def test_input_with_wrong_shape():
    X_past, X_future = make_inputs()
    with pytest.raises(ValueError, match="shapes"):
        fc.check_model_inputs(FakeModel(), X_past[:, :100], X_future)


def test_input_with_swapped_feature_order():
    X_past, X_future = make_inputs()
    with pytest.raises(ValueError, match="feature order"):
        fc.check_model_inputs(FakeModel(), X_past[:, :, ::-1], X_future)


def test_model_trained_on_other_inputs(scaler):
    v2_like = FakeModel(input_shapes=[(168, 5), (24, 4)])  # e.g. the 5-feature v2 model
    with pytest.raises(ValueError, match="different model file"):
        fc.predict_all(hourly(), v2_like, scaler)


def test_model_returns_12_values_not_24(scaler):
    with pytest.raises(ValueError, match="24 prices per day"):
        fc.predict_all(hourly(), FakeModel(n_out=12), scaler)


def test_model_returns_nan(scaler):
    with pytest.raises(ValueError, match="NaN or infinite"):
        fc.predict_all(hourly(), FakeModel(value=np.nan), scaler)


# ----------------------------------------------------------------- scaler ---
def test_scaler_of_wrong_type(real_prices):
    with pytest.raises(ValueError, match="fitted MinMaxScaler"):
        fc.validate_scaler(StandardScaler().fit(train_prices(real_prices)), real_prices)


def test_unfitted_scaler(real_prices):
    with pytest.raises(ValueError, match="fitted MinMaxScaler"):
        fc.validate_scaler(MinMaxScaler(), real_prices)


def test_scaler_with_two_columns(real_prices):
    two = np.column_stack([train_prices(real_prices), train_prices(real_prices)])
    with pytest.raises(ValueError, match="2 columns"):
        fc.validate_scaler(MinMaxScaler().fit(two), real_prices)


@pytest.mark.parametrize("settings", [{"feature_range": (-1, 1)}, {"clip": True}])
def test_scaler_with_other_settings(real_prices, settings):
    with pytest.raises(ValueError, match="settings"):
        fc.validate_scaler(MinMaxScaler(**settings).fit(train_prices(real_prices)), real_prices)


def test_scaler_from_the_validation_stage(real_prices):
    # notebook cell 9: fitted on 2015-2016 only (not the final 2015-2017 scaler)
    end_2016 = pd.Timestamp("2016-12-31 23:00", tz="UTC")
    early = real_prices.loc[fc.TRAIN_START:end_2016].values.reshape(-1, 1)
    with pytest.raises(ValueError, match="does not match"):
        fc.validate_scaler(MinMaxScaler().fit(early), real_prices)


def test_scaler_fitted_on_all_years(real_prices):
    # would leak the 2018 test prices into the scaler
    with pytest.raises(ValueError, match="does not match"):
        fc.validate_scaler(MinMaxScaler().fit(real_prices.values.reshape(-1, 1)), real_prices)


def test_scaler_with_same_rows_but_other_prices(real_prices):
    shifted = MinMaxScaler().fit(train_prices(real_prices) + 1.0)
    with pytest.raises(ValueError, match="does not match"):
        fc.validate_scaler(shifted, real_prices)


def test_scaler_cannot_be_checked_without_training_data(scaler, real_prices):
    with pytest.raises(ValueError, match="Cannot check the scaler"):
        fc.validate_scaler(scaler, real_prices.loc["2018"])


# ------------------------------------------------- real model, real data ---
def test_real_model_end_to_end(real_prices):
    model, scaler = fc.load_model()  # also runs validate_scaler
    out = fc.predict_all(real_prices, model, scaler)
    assert len(out) == len(fc.forecast_days(real_prices)) * fc.H
    assert out["predicted"].notna().all()
