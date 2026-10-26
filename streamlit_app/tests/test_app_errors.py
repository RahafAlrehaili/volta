"""
Tests for the error handling in app.py.

Run from streamlit_app/:   python -m pytest tests -v

The app is run headless with Streamlit's AppTest. The model is replaced by a
fake (no TensorFlow), and failures are simulated with mocks or temporary files.
"""
import logging
import pickle
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from volta import cache  # noqa: E402
from volta import forecast as fc  # noqa: E402

APP = str(APP_DIR / "app.py")
DATA_MESSAGE = "Could not load the electricity price data"
MODEL_MESSAGE = "Could not load the forecasting model"
FORECAST_MESSAGE = "Could not create the price forecasts"
UNEXPECTED_MESSAGE = "Something went wrong while preparing the forecast"


class FakeModel:
    """Stands in for the LSTM: returns `value` (scaled) for 24 hours per day."""

    def __init__(self, value=0.5, n_out=fc.H):
        self.value, self.n_out = value, n_out

    def predict(self, inputs, **kwargs):
        return np.full((len(inputs[0]), self.n_out), self.value)


def raise_(error):
    def fail(*args, **kwargs):
        raise error
    return fail


@pytest.fixture(autouse=True)
def fresh_caches():
    """Every test starts without cached data, model or predictions."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def fake_model(monkeypatch):
    """Use the fake model with the real (small) scaler file: no TensorFlow."""
    scaler = joblib.load(fc.SCALER_FILE)
    model = FakeModel()
    monkeypatch.setattr(fc, "load_model", lambda: (model, scaler))
    return model


def run_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def assert_stopped_with(at, message):
    """One clear message, no traceback, and nothing built from the failed step."""
    assert not at.exception, "a traceback was shown"
    assert [e.value for e in at.error] and message in at.error[0].value
    assert len(at.error) == 1
    shown = at.error[0].value
    assert "Traceback" not in shown and "/" not in shown, "technical details shown to the user"
    assert len(at.metric) == 0, "prices were shown after the failure"
    assert len(at.get("plotly_chart")) == 0, "a chart was shown after the failure"
    assert len(at.sidebar.radio) == 0, "the app went on after the failure"


def error_logs(caplog):
    return [r for r in caplog.records if r.name == "volta.app" and r.levelno == logging.ERROR]


# --------------------------------------------------------------- success ---
def test_app_runs_normally_without_errors(fake_model, caplog):
    at = run_app()
    assert not at.exception and not at.error
    assert len(at.metric) == 3
    assert len(at.get("plotly_chart")) == 1
    assert error_logs(caplog) == []


# ------------------------------------------------------------------- data ---
def test_missing_data_file(monkeypatch, tmp_path, caplog):
    missing = tmp_path / "volta_clean_hourly.csv"
    real_load_prices = fc.load_prices
    monkeypatch.setattr(fc, "load_prices", lambda path=missing: real_load_prices(path))
    at = run_app()
    assert_stopped_with(at, DATA_MESSAGE)
    (log,) = error_logs(caplog)
    assert log.getMessage().startswith("Loading the price data failed")
    assert isinstance(log.exc_info[1], FileNotFoundError)  # full error + traceback logged
    assert log.exc_text and "Traceback" in log.exc_text


@pytest.mark.parametrize("content", [
    b"time,price\n2015-01-01 00:00:00+00:00,50\n",  # 'price actual' column missing
    b"time,price actual\nnot-a-date,50\n",           # time cannot be read
    b"\xff\xfe\x00\x81\x8d binary, not a CSV",        # not text at all
])
def test_unreadable_data_file(monkeypatch, tmp_path, caplog, content):
    bad = tmp_path / "volta_clean_hourly.csv"
    bad.write_bytes(content)
    real_load_prices = fc.load_prices
    monkeypatch.setattr(fc, "load_prices", lambda path=bad: real_load_prices(path))
    at = run_app()
    assert_stopped_with(at, DATA_MESSAGE)
    (log,) = error_logs(caplog)
    assert log.getMessage().startswith("Loading the price data failed")
    assert isinstance(log.exc_info[1], ValueError)


# ------------------------------------------------------- model and scaler ---
@pytest.mark.parametrize("error", [
    FileNotFoundError("volta_lstm_final.keras not found"),
    ValueError("The scaler does not match the 2015-2017 training prices"),
    ImportError("No module named 'tensorflow'"),
    EOFError("broken volta_scaler_final.pkl"),
    pickle.UnpicklingError("invalid load key"),
    zipfile.BadZipFile("File is not a zip file"),
])
def test_model_or_scaler_cannot_be_loaded(monkeypatch, caplog, error):
    monkeypatch.setattr(fc, "load_model", raise_(error))
    at = run_app()
    assert_stopped_with(at, MODEL_MESSAGE)
    (log,) = error_logs(caplog)
    assert log.getMessage().startswith("Loading the model and scaler failed")
    assert log.exc_info[1] is error


def test_real_scaler_check_failure_is_shown_as_model_error(monkeypatch, caplog):
    # The real validate_scaler raising inside load_model (keras is not touched)
    from sklearn.preprocessing import MinMaxScaler
    wrong = MinMaxScaler().fit(np.array([[0.0], [500.0]]))

    def load_model_with_wrong_scaler():
        fc.validate_scaler(wrong, fc.load_prices())
        return FakeModel(), wrong

    monkeypatch.setattr(fc, "load_model", load_model_with_wrong_scaler)
    at = run_app()
    assert_stopped_with(at, MODEL_MESSAGE)
    assert "does not match" in str(error_logs(caplog)[0].exc_info[1])


# -------------------------------------------------------------- forecasts ---
@pytest.mark.parametrize("model, reason", [
    (FakeModel(value=np.nan), "NaN or infinite"),  # caught by check_model_output
    (FakeModel(n_out=12), "24 prices per day"),
])
def test_forecast_cannot_be_created(monkeypatch, caplog, model, reason):
    scaler = joblib.load(fc.SCALER_FILE)
    monkeypatch.setattr(fc, "load_model", lambda: (model, scaler))
    at = run_app()
    assert_stopped_with(at, FORECAST_MESSAGE)
    (log,) = error_logs(caplog)
    assert log.getMessage().startswith("Creating the forecasts failed")
    assert reason in str(log.exc_info[1])


# ------------------------------------------------------ unexpected errors ---
def test_unexpected_error_is_not_called_a_data_or_model_error(fake_model, monkeypatch, caplog):
    monkeypatch.setattr(fc, "predict_all", raise_(KeyError("a bug in the code")))
    at = run_app()
    assert_stopped_with(at, UNEXPECTED_MESSAGE)
    for specific in (DATA_MESSAGE, MODEL_MESSAGE, FORECAST_MESSAGE):
        assert specific not in at.error[0].value
    (log,) = error_logs(caplog)
    assert log.getMessage().startswith("Unexpected error while creating the forecasts")
    assert isinstance(log.exc_info[1], KeyError)


def test_unexpected_error_later_in_the_page_hides_the_traceback(fake_model, monkeypatch):
    # After loading, e.g. while building one day's table: Streamlit's own handler
    monkeypatch.setattr(cache, "day_forecast", raise_(RuntimeError("secret /home/path")))
    at = run_app()
    assert len(at.exception) == 1
    assert "redacted" in at.exception[0].message
    assert "secret" not in at.exception[0].message
    assert at.exception[0].stack_trace == []


# ---------------------------------------------- failures are not cached ---
def test_the_app_recovers_once_the_problem_is_fixed(fake_model, monkeypatch):
    state = {"broken": True}
    real_load_prices = fc.load_prices

    def flaky_load_prices(*args, **kwargs):
        if state["broken"]:
            raise FileNotFoundError("data not there yet")
        return real_load_prices(*args, **kwargs)

    monkeypatch.setattr(fc, "load_prices", flaky_load_prices)
    assert_stopped_with(run_app(), DATA_MESSAGE)

    state["broken"] = False  # e.g. the file is back
    at = run_app()
    assert not at.error and not at.exception
    assert len(at.metric) == 3
