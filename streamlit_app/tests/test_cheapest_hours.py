"""
Tests for the cheapest-time recommendation.

fc.find_cheapest_hours (volta/forecast.py) is the only place that computes it;
app.py only adds the Spain-time labels. Run from streamlit_app/:
    python -m pytest tests -v
"""
import datetime
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
from volta import forecast as fc  # noqa: E402

APP = str(APP_DIR / "app.py")


def old_app_window(prices, power_kw, duration_hours):
    """The loop app.py used before it called fc.find_cheapest_hours (the reference)."""
    n = len(prices)
    duration_hours = min(int(duration_hours), n)
    min_cost, best = float("inf"), 0
    for i in range(n - duration_hours + 1):
        cost = np.sum((prices[i:i + duration_hours] / 1000.0) * power_kw)
        if cost < min_cost:
            min_cost, best = cost, i
    return best, min_cost, np.mean(prices[best:best + duration_hours])


# ---------------------------------------------------------- the function ---
def test_finds_the_cheapest_block():
    prices = np.array([60.0] * 24)
    prices[5:8] = [20.0, 21.0, 22.0]
    best = fc.find_cheapest_hours(prices, power_kw=2.0, hours=3)
    assert (best["start"], best["end"]) == (5, 8)
    assert best["cost"] == np.sum(prices[5:8] / 1000.0 * 2.0)  # EUR = sum(EUR/MWh / 1000 * kW)
    assert best["avg_price"] == 21.0


def test_equal_costs_pick_the_earliest_block():
    prices = np.array([50.0] * 24)
    prices[[3, 15]] = 10.0  # two equally cheap hours
    assert fc.find_cheapest_hours(prices, 1.0, 1)["start"] == 3


@pytest.mark.parametrize("hours", [24, 30])
def test_whole_day_and_longer_than_a_day(hours):
    prices = np.linspace(40, 80, 24)
    best = fc.find_cheapest_hours(prices, 2.0, hours)  # longer than 24 is capped at 24
    assert (best["start"], best["end"]) == (0, 24)
    assert best["avg_price"] == pytest.approx(prices.mean())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_same_result_as_the_old_app_loop(dtype):
    # Prices rounded to 2 decimals with many repeats, so equal blocks are common;
    # float32 like the LSTM column, float64 like the baselines.
    rng = np.random.default_rng(0)
    for _ in range(3000):
        prices = rng.choice([35.5, 40.0, 41.25, 50.0, 62.1], size=24).astype(dtype)
        hours, kw = int(rng.integers(1, 25)), float(rng.choice([0.1, 2.0, 3.7, 50.0]))
        best = fc.find_cheapest_hours(prices, kw, hours)
        start, cost, avg = old_app_window(prices, kw, hours)
        assert (best["start"], best["cost"], best["avg_price"]) == (start, cost, avg)


def test_real_tie_from_the_data_keeps_the_earliest_block():
    # 2015-05-10, Lag-24 prices, 5 h at 2 kW: two blocks cost the same. The app shows
    # the earlier one (13:00 UTC = 15:00 in Madrid); the old forecast.py formula
    # (hours * mean) picked the later one.
    price = fc.load_prices()
    start = pd.Timestamp("2015-05-09", tz="UTC")
    lag_24 = price.loc[start:start + pd.Timedelta(hours=23)].round(2).values
    best = fc.find_cheapest_hours(lag_24, 2.0, 5)
    assert (best["start"], best["end"]) == (13, 18)


# ------------------------------------------------------ the app uses it ---
class PatternModel:
    """Fake LSTM: the same 24 scaled prices every day, cheapest at `low` positions."""

    def __init__(self, low):
        self.pattern = np.full(24, 0.6)
        self.pattern[low] = [0.1, 0.12, 0.11][:len(low)]

    def predict(self, inputs, **kwargs):
        return np.tile(self.pattern, (len(inputs[0]), 1))


@pytest.fixture(autouse=True)
def fresh_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def click_recommendation(monkeypatch, low, day, kw=2.0, hours=3):
    scaler = joblib.load(fc.SCALER_FILE)
    monkeypatch.setattr(fc, "load_model", lambda: (PatternModel(low), scaler))
    calls = []
    real = fc.find_cheapest_hours

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(fc, "find_cheapest_hours", spy)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.date_input[0].set_value(day).run()
    at.number_input[0].set_value(kw)
    at.number_input[1].set_value(hours).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    html = " ".join(m.value for m in at.markdown)
    return at, calls, html


def test_app_shows_the_shared_result_in_spain_time(monkeypatch):
    # Summer day: forecast positions 4-6 are 06:00-09:00 in Madrid (UTC+2)
    at, calls, html = click_recommendation(monkeypatch, [4, 5, 6], datetime.date(2018, 7, 15))
    assert len(calls) == 1  # the app asked the shared function, once
    prices, kw, hours = calls[0]
    assert (kw, hours) == (2.0, 3) and len(prices) == 24
    best = fc.find_cheapest_hours(prices, kw, hours)
    assert (best["start"], best["end"]) == (4, 7)
    assert "06:00 – 09:00" in html
    assert f"€{best['cost']:.2f}" in html and f"€{best['avg_price']:.1f} /MWh" in html


def test_app_labels_the_october_clock_change_correctly(monkeypatch):
    # 2018-10-28: positions 0-2 are 02:00 CEST, 02:00 CET, 03:00 -> ends at 04:00
    _, calls, html = click_recommendation(monkeypatch, [0, 1, 2], datetime.date(2018, 10, 28))
    assert len(calls) == 1
    assert "02:00 CEST – 04:00" in html
