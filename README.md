<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/volta_banner_dark.jpg">
    <img src="assets/volta_banner_light.jpg" alt="VOLTA – Electricity Price Forecasting" width="640">
  </picture>
</p>

<h3 align="center">Predicting Tomorrow's Electricity Price</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-LSTM%20%7C%20GRU-D00000?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/pytest-tested-0A9EDC?logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

VOLTA is a time-series forecasting project that predicts Spain's actual electricity price for each of the next 24 hours.

We trained and compared LSTM and GRU models on historical prices, tested them against simple baselines, and built an interactive Streamlit dashboard that shows the 24-hour forecast and suggests the cheapest time to run a device.

<p align="center">
  <a href="https://volta-app.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open the live dashboard in Streamlit"></a>
</p>

## Data

- **Source:** [Kaggle — Energy consumption, generation, prices and weather in Spain](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
- **Period:** 2015–2018, hourly (35,064 hours)
- **Contents:** electricity generation, demand and price for Spain, plus weather for its five largest cities

## Approach

- **Target:** `price actual`, Spain's hourly electricity price (€/MWh)
- **Input:** the previous 168 hours (7 days) of prices, plus `is_weekend`
- **Output:** 24 hourly prices, forecast once a day at 00:00 UTC
- **Split by time:** train 2015–2016, validation 2017, test 2018
- **Model selection:** LSTM vs GRU, feature engineering, then hyperparameter tuning

## Results

The final model is an **LSTM with an `is_weekend` feature**, trained on 2015–2017 and tested on 2018, a year it never saw.

| Model | Test MAE (€/MWh) |
|---|---|
| **Final LSTM** | **4.03** |
| Lag-24 (same hour yesterday) | 5.21 |
| Lag-168 (same hour last week) | 6.39 |
| Last value | 7.44 |

On average, the forecast is about 4 € away from the real price, which is **22.5% more accurate than the best baseline (Lag-24)**.

## Dashboard

The [live dashboard](https://volta-app.streamlit.app/) runs the final LSTM on every day of the data:

- **Forecast & Device Optimizer:** the 24-hour price forecast for a chosen day, and the cheapest block of hours to run a device (based on its power and run time) with the estimated cost.
- **Forecast Insights:** the cheapest and most expensive hours of the day, grouped into low, moderate and high prices.
- **Spain time:** the model forecasts from 00:00 UTC, but every hour is shown in Spain time (Europe/Madrid), including summer/winter clock changes. `(+1)` marks hours on the next day.

### Reliability

- **Input checks:** before forecasting, the app checks the price data (numbers only, hourly, no gaps or duplicates), that the scaler matches the 2015–2017 training data, and that the model inputs and outputs have the right shapes and feature order. If something is wrong it stops with a clear error instead of fixing data silently.
- **Clear error messages:** if the data or model fails to load, users see a short message, while the full details are written to the app logs.
- **One shared calculation:** the cheapest-hours recommendation is computed in one function (`find_cheapest_hours` in `volta/forecast.py`), used by the app and covered by tests.
- **Tests:** `pytest` tests cover the input checks, the error handling and the cheapest-hours recommendation.

## Project Structure

```
volta/
├─ notebooks/                    Run them in this order:
│  ├─ data_quality.ipynb         1. Data quality checks on the raw data
│  ├─ data_preparation.ipynb     2. Cleaning; creates data/processed/volta_clean_hourly.csv
│  ├─ eda.ipynb                  3. Exploratory data analysis
│  ├─ feature_experiments.ipynb  4. LSTM/GRU feature experiments (is_weekend wins)
│  └─ final_model.ipynb          5. Baselines, LSTM vs GRU, tuning, final test on 2018
├─ data/                         Not in the repo (see step 3 below)
│  ├─ raw/                       Kaggle CSV files
│  └─ processed/                 Clean hourly table made by data_preparation.ipynb
├─ models/                       Final model and scaler saved by final_model.ipynb
├─ streamlit_app/                Interactive dashboard (has its own data and model copy)
│  ├─ app.py
│  ├─ volta/                     Model inputs, checks, predictions and caching
│  ├─ tests/                     pytest tests (checks, errors, cheapest hours)
│  ├─ data/
│  ├─ models/
│  ├─ assets/
│  └─ Dockerfile                 Runs the dashboard in a container
├─ assets/                       README images
├─ requirements.txt              Packages for the notebooks
└─ README.md
```

## How to Run

You need **Python 3.12** and **git**.

### 1. Clone the repository

```bash
git clone https://github.com/Ghadi213/volta.git
cd volta
```

### 2. Create a virtual environment and install the packages

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the data (only needed for the notebooks)

1. Create the data folder:

   ```bash
   mkdir -p data/raw
   ```

2. Open the [Kaggle dataset](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) and click **Download**.
3. Unzip it and put the two CSV files in `data/raw/`:

   ```
   data/raw/energy_dataset.csv
   data/raw/weather_features.csv
   ```

### 4. Run the notebooks

```bash
jupyter notebook notebooks/
```

Run them in this order: `data_quality` → `data_preparation` → `eda` → `feature_experiments` → `final_model`.

`data_preparation.ipynb` must run before the modeling notebooks, because it creates `data/processed/volta_clean_hourly.csv`, which they read. `final_model.ipynb` saves the final model and scaler to `models/`.

### 5. Run the dashboard

> **Just want to see it?** The dashboard is live at **[volta-app.streamlit.app](https://volta-app.streamlit.app/)**, no installation needed.

To run it on your own machine: the app does not need the Kaggle download: its data and trained model are already in `streamlit_app/`.

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. The first load takes a few seconds while the model predicts every day. Press `Ctrl + C` in the terminal to stop it.

### 6. Run the dashboard with Docker (optional)

```bash
cd streamlit_app
docker build -t volta .
docker run -p 8501:8501 volta
```

Then open http://localhost:8501.

### 7. Run the tests

```bash
cd streamlit_app
pip install pytest
python -m pytest tests -v
```

## Team

Rahaf Alrehaili · Ghadi Alrashdi · Ayah Qalo · Mohammed Alyami
