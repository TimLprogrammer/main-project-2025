import pandas as pd
import os
import logging

# Logging configuratie
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_weather_data(filepath):
    """
    Laad weer data uit CSV bestand
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
    Rond alle numerieke kolommen af op 2 decimalen
    """
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        df[col] = df[col].round(2)
    return df

def add_time_features(df):
    """
    Voeg tijd-gerelateerde features toe
    """
    # Voeg basis tijd features toe
    df['week'] = df['date'].dt.isocalendar().week  # 1-52
    df['dag'] = df['date'].dt.dayofweek + 1  # 1-7 (Maandag=1, Zondag=7)
    df['maand'] = df['date'].dt.month  # 1-12
    
    # Voeg weekend indicator toe (0=werkdag, 1=weekend)
    df['is_weekend'] = (df['dag'].isin([6, 7])).astype(int)
    
    # Voeg seizoen toe (1=Winter, 2=Lente, 3=Zomer, 4=Herfst)
    df['seizoen'] = df['date'].dt.month.map({
        12: 1, 1: 1, 2: 1,  # Winter
        3: 2, 4: 2, 5: 2,   # Lente
        6: 3, 7: 3, 8: 3,   # Zomer
        9: 4, 10: 4, 11: 4  # Herfst
    })
    
    # Verwijder de date kolom
    df.drop('date', axis=1, inplace=True)
    
    return df

def add_derived_features(df):
    """
    Voeg afgeleide features toe die relevant zijn voor voorspellingen
    """
    # Temperatuur-gerelateerde features
    df['temp_range'] = df['temperatuur_max'] - df['temperatuur_min']
    df['temp_vocht_interactie'] = df['temperatuur_avg'] * df['luchtvochtigheid_avg']
    
    # Rolling means (3-daags voortschrijdend gemiddelde)
    df['temp_rolling_mean'] = df['temperatuur_avg'].rolling(window=3, min_periods=1).mean()
    df['vocht_rolling_mean'] = df['luchtvochtigheid_avg'].rolling(window=3, min_periods=1).mean()
    
    # Extreme weer indicatoren
    df['hoge_temp_dag'] = (df['temperatuur_max'] > 25).astype(int)  # Zomerse dag
    df['lage_temp_dag'] = (df['temperatuur_min'] < 0).astype(int)   # Vorst dag
    df['hoge_vocht_dag'] = (df['luchtvochtigheid_avg'] > 85).astype(int)
    
    # Wind features
    df['windkracht_beaufort'] = df['windsnelheid_avg'].apply(lambda x: wind_to_beaufort(x))
    df['sterke_wind_uren'] = (df['windsnelheid_max'] > 10).astype(int)
    
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
    Aggregeer uurlijkse weerdata naar dagelijkse statistieken
    """
    try:
        # Basis aggregaties
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

        # Extra statistieken
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

        # Combineer statistieken
        daily_stats = pd.concat([daily_stats, daily_extra], axis=1)

        # Flatten multi-level columns en hernoem
        daily_stats.columns = [
            'temperatuur_min', 'temperatuur_max', 'temperatuur_avg',
            'luchtvochtigheid_min', 'luchtvochtigheid_max', 'luchtvochtigheid_avg',
            'dauwpunt_min', 'dauwpunt_max', 'dauwpunt_avg',
            'gevoelstemp_min', 'gevoelstemp_max', 'gevoelstemp_avg',
            'luchtdruk_min', 'luchtdruk_max', 'luchtdruk_avg',
            'bewolking_min', 'bewolking_max', 'bewolking_avg',
            'windsnelheid_min', 'windsnelheid_max', 'windsnelheid_avg',
            'windstoot_max', 'windstoot_avg',
            'temperatuur_range', 'luchtdruk_delta',
            'uren_helder', 'uren_bewolkt',
            'aantal_sterke_windstoten', 'dominante_windrichting'
        ]

        # Reset index en behoud de datum
        daily_stats = daily_stats.reset_index()
        daily_stats = daily_stats.rename(columns={'index': 'date'})

        # Bereken tijd features
        daily_stats['week'] = pd.to_datetime(daily_stats['date']).dt.isocalendar().week
        daily_stats['dag'] = pd.to_datetime(daily_stats['date']).dt.dayofweek + 1
        daily_stats['maand'] = pd.to_datetime(daily_stats['date']).dt.month
        daily_stats['is_weekend'] = (daily_stats['dag'].isin([6, 7])).astype(int)
        daily_stats['seizoen'] = daily_stats['maand'].map({
            12: 1, 1: 1, 2: 1,  # Winter
            3: 2, 4: 2, 5: 2,   # Lente
            6: 3, 7: 3, 8: 3,   # Zomer
            9: 4, 10: 4, 11: 4  # Herfst
        })
        
        # Voeg afgeleide features toe
        daily_stats = add_derived_features(daily_stats)
        
        # Herorden kolommen met tijd-features eerst
        column_order = ['date', 'week', 'dag', 'maand', 'seizoen', 'is_weekend'] + [
            col for col in daily_stats.columns 
            if col not in ['date', 'week', 'dag', 'maand', 'seizoen', 'is_weekend']
        ]
        
        daily_stats = daily_stats[column_order]
        
        # Rond alle numerieke kolommen af
        daily_stats = round_numeric_columns(daily_stats)

        return daily_stats

    except Exception as e:
        logging.error(f"Error in aggregate_daily_weather: {str(e)}")
        return None

def main():
    # Definieer paden
    base_path = "/Users/timlind/Documents/Stage/project/main_project"
    input_path = os.path.join(base_path, "data", "csv-api")
    output_path = os.path.join(base_path, "data", "csv-daily")  # Gewijzigd van csv-dagelijks naar csv-daily
    
    # Maak output directory als deze niet bestaat
    os.makedirs(output_path, exist_ok=True)

    # Verwerk historical data
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
                logging.info("Kolommen in output:")
                for col in daily_historical.columns:
                    if col != 'date' and col != 'dominante_windrichting':
                        logging.info(f"{col}: {daily_historical[col].dtype}")

    # Verwerk forecast data
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
