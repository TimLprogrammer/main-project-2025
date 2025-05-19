import pandas as pd
import os
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_weather_data(filepath):
    """
    Load weather data from CSV file
    """
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logging.error(f"Error loading {filepath}: {str(e)}")
        return None

def round_numeric_columns(df):
    """
    Round all numeric columns to 2 decimal places
    """
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        df[col] = df[col].round(2)
    return df

def add_time_features(df):
    """
    Add time-related features
    """
    # Add basic time features
    df['week'] = df['date'].dt.isocalendar().week  # 1-52
    df['day'] = df['date'].dt.dayofweek + 1  # 1-7 (Monday=1, Sunday=7)
    df['month'] = df['date'].dt.month  # 1-12

    # Add weekend indicator (0=weekday, 1=weekend)
    df['is_weekend'] = (df['day'].isin([6, 7])).astype(int)

    # Add season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df['season'] = df['date'].dt.month.map({
        12: 1, 1: 1, 2: 1,  # Winter
        3: 2, 4: 2, 5: 2,   # Spring
        6: 3, 7: 3, 8: 3,   # Summer
        9: 4, 10: 4, 11: 4  # Fall
    })

    # Remove the date column
    df.drop('date', axis=1, inplace=True)

    return df

def add_derived_features(df):
    """
    Add derived features that are relevant for predictions
    """
    # Temperature-related features
    df['temp_range'] = df['temperature_max'] - df['temperature_min']
    df['temp_humidity_interaction'] = df['temperature_avg'] * df['humidity_avg']

    # Rolling means (3-day moving average)
    df['temp_rolling_mean'] = df['temperature_avg'].rolling(window=3, min_periods=1).mean()
    df['humidity_rolling_mean'] = df['humidity_avg'].rolling(window=3, min_periods=1).mean()

    # Extreme weather indicators
    df['high_temp_day'] = (df['temperature_max'] > 25).astype(int)  # Summer day
    df['low_temp_day'] = (df['temperature_min'] < 0).astype(int)   # Frost day
    df['high_humidity_day'] = (df['humidity_avg'] > 85).astype(int)

    # Wind features
    df['wind_force_beaufort'] = df['wind_speed_avg'].apply(lambda x: wind_to_beaufort(x))
    df['strong_wind_hours'] = (df['wind_speed_max'] > 10).astype(int)

    return df

def wind_to_beaufort(windspeed):
    """Convert windspeed (m/s) to Beaufort scale"""
    if windspeed < 0.3: return 0
    elif windspeed < 1.6: return 1
    elif windspeed < 3.4: return 2
    elif windspeed < 5.5: return 3
    elif windspeed < 8.0: return 4
    elif windspeed < 10.8: return 5
    elif windspeed < 13.9: return 6
    elif windspeed < 17.2: return 7
    elif windspeed < 20.8: return 8
    elif windspeed < 24.5: return 9
    elif windspeed < 28.5: return 10
    elif windspeed < 32.7: return 11
    else: return 12

def aggregate_daily_weather(df):
    """
    Aggregate hourly weather data to daily statistics
    """
    try:
        # Basic aggregations
        daily_stats = df.groupby(df['date'].dt.date).agg({
            'temperature_2m': ['min', 'max', 'mean'],
            'relative_humidity_2m': ['min', 'max', 'mean'],
            'dew_point_2m': ['min', 'max', 'mean'],
            'apparent_temperature': ['min', 'max', 'mean'],
            'pressure_msl': ['min', 'max', 'mean'],
            'cloud_cover': ['min', 'max', 'mean'],
            'wind_speed_10m': ['min', 'max', 'mean'],
            'wind_gusts_10m': ['max', 'mean']
        })

        # Extra statistics
        daily_extra = df.groupby(df['date'].dt.date).agg({
            'temperature_2m': lambda x: x.max() - x.min(),
            'pressure_msl': lambda x: x.max() - x.min(),
            'cloud_cover': [
                lambda x: (x < 10).sum(),
                lambda x: (x > 90).sum()
            ],
            'wind_gusts_10m': lambda x: (x > 14).sum(),
            'wind_direction_10m': lambda x: x.mode().iloc[0] if not x.mode().empty else None
        })

        # Combine statistics
        daily_stats = pd.concat([daily_stats, daily_extra], axis=1)

        # Flatten multi-level columns and rename
        daily_stats.columns = [
            'temperature_min', 'temperature_max', 'temperature_avg',
            'humidity_min', 'humidity_max', 'humidity_avg',
            'dew_point_min', 'dew_point_max', 'dew_point_avg',
            'apparent_temp_min', 'apparent_temp_max', 'apparent_temp_avg',
            'pressure_min', 'pressure_max', 'pressure_avg',
            'cloud_cover_min', 'cloud_cover_max', 'cloud_cover_avg',
            'wind_speed_min', 'wind_speed_max', 'wind_speed_avg',
            'wind_gust_max', 'wind_gust_avg',
            'temperature_range', 'pressure_delta',
            'clear_hours', 'cloudy_hours',
            'strong_wind_gust_count', 'dominant_wind_direction'
        ]

        # Reset index and keep the date
        daily_stats = daily_stats.reset_index()
        daily_stats = daily_stats.rename(columns={'index': 'date'})

        # Calculate time features
        daily_stats['week'] = pd.to_datetime(daily_stats['date']).dt.isocalendar().week
        daily_stats['day'] = pd.to_datetime(daily_stats['date']).dt.dayofweek + 1
        daily_stats['month'] = pd.to_datetime(daily_stats['date']).dt.month
        daily_stats['is_weekend'] = (daily_stats['day'].isin([6, 7])).astype(int)
        daily_stats['season'] = daily_stats['month'].map({
            12: 1, 1: 1, 2: 1,  # Winter
            3: 2, 4: 2, 5: 2,   # Spring
            6: 3, 7: 3, 8: 3,   # Summer
            9: 4, 10: 4, 11: 4  # Fall
        })

        # Add derived features
        daily_stats = add_derived_features(daily_stats)

        # Reorder columns with time features first
        column_order = ['date', 'week', 'day', 'month', 'season', 'is_weekend'] + [
            col for col in daily_stats.columns
            if col not in ['date', 'week', 'day', 'month', 'season', 'is_weekend']
        ]

        daily_stats = daily_stats[column_order]

        # Round all numeric columns
        daily_stats = round_numeric_columns(daily_stats)

        return daily_stats

    except Exception as e:
        logging.error(f"Error in aggregate_daily_weather: {str(e)}")
        return None

def main():
    # Define paths
    base_path = "/Users/timlind/Documents/Stage/project/main_project"
    input_path = os.path.join(base_path, "data", "csv-api")
    output_path = os.path.join(base_path, "data", "csv-daily")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process historical data
    logging.info("Processing historical data...")
    historical_file = os.path.join(input_path, "historical.csv")
    if os.path.exists(historical_file):
        df_historical = load_weather_data(historical_file)
        if df_historical is not None:
            daily_historical = aggregate_daily_weather(df_historical)
            if daily_historical is not None:
                output_file = os.path.join(output_path, "historical_daily.csv")
                daily_historical.to_csv(output_file, index=False)
                logging.info(f"Historical daily data saved to {output_file}")
                logging.info(f"Historical daily data shape: {daily_historical.shape}")
                logging.info("Columns in output:")
                for col in daily_historical.columns:
                    if col != 'date' and col != 'dominant_wind_direction':
                        logging.info(f"{col}: {daily_historical[col].dtype}")

    # Process forecast data
    logging.info("Processing forecast data...")
    forecast_file = os.path.join(input_path, "forecast.csv")
    if os.path.exists(forecast_file):
        df_forecast = load_weather_data(forecast_file)
        if df_forecast is not None:
            daily_forecast = aggregate_daily_weather(df_forecast)
            if daily_forecast is not None:
                output_file = os.path.join(output_path, "forecast_daily.csv")
                daily_forecast.to_csv(output_file, index=False)
                logging.info(f"Forecast daily data saved to {output_file}")
                logging.info(f"Forecast daily data shape: {daily_forecast.shape}")

if __name__ == "__main__":
    main()
