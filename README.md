# Weather Forecast & Model Management

An automated system for weather-based fault prediction, designed to be easy to understand and manage.

## Overview

This project is an interactive Streamlit application that collects, processes, and uses weather data to predict fault notifications based on weather conditions. The system trains machine learning models and generates predictions that are displayed in a user-friendly interface.

## Features

- **Data Collection:** Fetches weather forecasts and historical weather data via APIs and loads fault notifications from Excel files.
- **Data Processing:** Cleans the raw data, adds useful features, and aggregates data into daily summaries.
- **Fault Classification:** Uses rules and machine learning to categorize fault notifications.
- **Model Training:** Trains multiple machine learning models to predict faults based on weather and time features.
- **Forecast Generation:** Uses the trained models to predict future faults based on the latest weather forecasts.
- **Results Visualization:** Shows predictions, model performance, and detailed plots in an interactive app.
- **Regular Retraining:** Updates data and retrains models to maintain accuracy over time.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/weather_forecast_app.git
   cd weather_forecast_app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Start the Streamlit app:
```
streamlit run streamlit_app.py
```

The app has three main sections:

1. **Latest Predictions:** Run the forecast pipeline to generate and view the latest predictions.
2. **Models:** Upload new fault notification data and retrain the models.
3. **Explanation:** View detailed explanations about how the system works.

## Directory Structure

- **`data/script/`**
  - `forecast.py`: Fetches weather forecast data.
  - `historical.py`: Fetches historical weather data.
  - `process_daily_weather.py`: Cleans and enriches weather data.
  - `classifie_vks.py`: Classifies fault notifications.
  - `result.py`: Creates visualizations of fault data.

- **`data/`**
  - `update.py`: Updates datasets and triggers retraining.
  - `csv-api/` and `csv-daily/`: Store raw and processed weather data.
  - `notifications/`: Contains fault notification Excel files.

- **`predictions/`**
  - `forecast.py`: Generates fault forecasts using trained models.
  - `cooling.py` and `heating.py`: Train, optimize, evaluate, and save models.
  - `best-model/`: Stores the best models and their metrics.
  - `plots/`: Contains visualizations of model performance.

- **`streamlit/`**
  - `main.py`: The app interface to run the pipeline, retrain models, and view results.

## Deployment

This app can be deployed on Streamlit Cloud:

1. Push the code to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub account and select the repository.
4. Set `streamlit_app.py` as the main file.
5. Click "Deploy".

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- Matplotlib
- Seaborn
- Plotly
- Optuna
- OpenPyXL

## License

[MIT](LICENSE)
