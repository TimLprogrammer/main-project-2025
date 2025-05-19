#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
forecast.py

Dit script gebruikt de getrainde modellen uit cooling.py en heating.py om voorspellingen
te maken voor nieuwe gegevens. Het past de juiste feature engineering toe voor elk model
en combineert de voorspellingen in één uitvoer.

Het script:
1. Importeert de bestaande trainingsmodellen uit cooling.py en heating.py
2. Leest de dataset van forecast_daily.csv in
3. Maakt twee afgeleide datasets met de juiste feature engineering
4. Past de respectievelijke modellen toe op deze voorbereide datasets
5. Combineert de resultaten in één dataframe
6. Slaat het gecombineerde resultaat op
"""

import pandas as pd
import numpy as np
import os
from os import path
import pickle
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler

# Importeer de feature engineering functies uit cooling.py en heating.py
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from predictions.cooling import preprocess_data as cooling_preprocess_data
from predictions.cooling import add_temporal_features as cooling_add_temporal_features
from predictions.heating import preprocess_data as heating_preprocess_data
from predictions.heating import add_temporal_features as heating_add_temporal_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console only
    ]
)
logger = logging.getLogger(__name__)

# Define base path
BASE_PATH = path.dirname(path.dirname(path.abspath(__file__)))

# Define paths
FORECAST_DATA_PATH = path.join(BASE_PATH, 'data', 'csv-daily', 'forecast_daily.csv')
COOLING_MODEL_PATH = path.join(BASE_PATH, 'predictions', 'best-model', 'cooling', 'best_model_cooling.pkl')
HEATING_MODEL_PATH = path.join(BASE_PATH, 'predictions', 'best-model', 'heating', 'best_model_heating.pkl')
OUTPUT_PATH = path.join(BASE_PATH, 'predictions', 'forecast_predictions.csv')

def load_data(data_path):
    """
    Laad de forecast data uit een CSV bestand

    Parameters:
    data_path (str): Pad naar het forecast data CSV bestand

    Returns:
    pd.DataFrame: Geladen data
    """
    try:
        logger.info(f"Laden van forecast data van: {data_path}")
        # Controleer of het bestand bestaat
        if not os.path.exists(data_path):
            logger.error(f"Bestand bestaat niet: {data_path}")
            return None
        data = pd.read_csv(data_path)
        logger.info(f"Forecast data geladen: {data.shape[0]} rijen, {data.shape[1]} kolommen")
        return data
    except Exception as e:
        logger.error(f"Fout bij het laden van forecast data: {e}")
        return None

def load_model(model_path):
    """
    Laad een getraind model uit een pickle bestand

    Parameters:
    model_path (str): Pad naar het model pickle bestand

    Returns:
    object: Geladen model
    """
    try:
        logger.info(f"Laden van model van: {model_path}")
        # Controleer of het bestand bestaat
        if not os.path.exists(model_path):
            logger.error(f"Model bestand bestaat niet: {model_path}")
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model succesvol geladen")
        return model
    except Exception as e:
        logger.error(f"Fout bij het laden van model: {e}")
        return None

def preprocess_cooling_data(df):
    """
    Pas cooling-specifieke feature engineering toe op de forecast data

    Parameters:
    df (pd.DataFrame): Forecast data

    Returns:
    pd.DataFrame: Verwerkte data klaar voor cooling model voorspelling
    """
    logger.info("Toepassen van cooling-specifieke feature engineering")

    # Gebruik de geïmporteerde functies uit cooling.py
    # Stap 1: Basis preprocessing - gebruik een dummy target_col die niet in de data voorkomt
    processed_df = cooling_preprocess_data(df, target_col='dummy_target_not_in_data')

    # Stap 2: Voeg temporele features toe
    processed_df = cooling_add_temporal_features(processed_df)

    # Drop date column for prediction
    features = processed_df.drop(columns=['date'], errors='ignore')

    logger.info(f"Cooling feature engineering voltooid: {features.shape[1]} features")
    return features

def preprocess_heating_data(df):
    """
    Pas heating-specifieke feature engineering toe op de forecast data

    Parameters:
    df (pd.DataFrame): Forecast data

    Returns:
    pd.DataFrame: Verwerkte data klaar voor heating model voorspelling
    """
    logger.info("Toepassen van heating-specifieke feature engineering")

    # Gebruik de geïmporteerde functies uit heating.py
    # Stap 1: Basis preprocessing - gebruik een dummy target_col die niet in de data voorkomt
    processed_df = heating_preprocess_data(df, target_col='dummy_target_not_in_data')

    # Stap 2: Voeg temporele features toe
    processed_df = heating_add_temporal_features(processed_df)

    # Drop date column for prediction
    features = processed_df.drop(columns=['date'], errors='ignore')

    logger.info(f"Heating feature engineering voltooid: {features.shape[1]} features")
    return features

def ensure_model_features(model, features_df):
    """
    Zorg ervoor dat de features overeenkomen met wat het model verwacht

    Parameters:
    model: Het getrainde model
    features_df (pd.DataFrame): De features voor voorspelling

    Returns:
    pd.DataFrame: Features dataframe met de juiste kolommen voor het model
    """
    # Controleer of het model feature_names_ heeft (sommige modellen hebben dit niet)
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        logger.info(f"Model verwacht {len(expected_features)} features")

        # Controleer welke features ontbreken
        missing_features = [f for f in expected_features if f not in features_df.columns]
        if missing_features:
            logger.warning(f"Ontbrekende features: {missing_features}")
            # Voeg ontbrekende features toe met waarde 0
            for feature in missing_features:
                features_df[feature] = 0

        # Controleer of er extra features zijn die het model niet verwacht
        extra_features = [f for f in features_df.columns if f not in expected_features]
        if extra_features:
            logger.warning(f"Extra features die worden verwijderd: {extra_features}")
            features_df = features_df.drop(columns=extra_features)

        # Zorg ervoor dat de volgorde van de kolommen overeenkomt met wat het model verwacht
        features_df = features_df[expected_features]

    return features_df

def make_predictions():
    """
    Hoofdfunctie om data en modellen te laden, data voor te bereiden, voorspellingen te maken en resultaten op te slaan
    """
    logger.info("Start van forecast voorspellingsproces")

    # Laad forecast data
    forecast_data = load_data(FORECAST_DATA_PATH)
    if forecast_data is None:
        logger.error("Kon forecast data niet laden. Stoppen.")
        return

    # Bewaar een kopie van de datums voor de uiteindelijke output
    dates = forecast_data['date'].copy()

    # Voeg temperatuur trend features toe
    for window in [1, 3, 4]:
        forecast_data[f'temp_avg_{window}d'] = forecast_data['temperatuur_avg'].rolling(window=window, min_periods=1).mean()
        forecast_data[f'temp_trend_{window}'] = forecast_data[f'temp_avg_{window}d'] - forecast_data[f'temp_avg_{window}d'].shift(1)

    # Laad modellen
    cooling_model = load_model(COOLING_MODEL_PATH)
    heating_model = load_model(HEATING_MODEL_PATH)

    if cooling_model is None or heating_model is None:
        logger.error("Kon een of beide modellen niet laden. Stoppen.")
        return

    # Bereid data voor voor cooling model
    cooling_features = preprocess_cooling_data(forecast_data)

    # Schaal cooling features (StandardScaler zoals gebruikt in cooling.py)
    cooling_scaler = StandardScaler()
    cooling_features_scaled = pd.DataFrame(
        cooling_scaler.fit_transform(cooling_features),
        columns=cooling_features.columns
    )

    # Zorg ervoor dat de features overeenkomen met wat het cooling model verwacht
    cooling_features_scaled = ensure_model_features(cooling_model, cooling_features_scaled)

    # Bereid data voor voor heating model
    heating_features = preprocess_heating_data(forecast_data)

    # Schaal heating features (RobustScaler zoals gebruikt in heating.py)
    heating_scaler = RobustScaler()
    heating_features_scaled = pd.DataFrame(
        heating_scaler.fit_transform(heating_features),
        columns=heating_features.columns
    )

    # Zorg ervoor dat de features overeenkomen met wat het heating model verwacht
    heating_features_scaled = ensure_model_features(heating_model, heating_features_scaled)

    # Maak voorspellingen
    logger.info("Maken van cooling fault voorspellingen")
    cooling_predictions = cooling_model.predict(cooling_features_scaled)

    logger.info("Maken van heating fault voorspellingen")
    heating_predictions = heating_model.predict(heating_features_scaled)

    # Rond voorspellingen af naar gehele getallen
    cooling_predictions_rounded = np.round(cooling_predictions).astype(int)
    heating_predictions_rounded = np.round(heating_predictions).astype(int)

    # Maak output dataframe
    output_df = pd.DataFrame({
        'date': dates,
        'cooling_fault_prediction': cooling_predictions_rounded,
        'heating_fault_prediction': heating_predictions_rounded
    })

    # Sla voorspellingen op
    try:
        output_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Voorspellingen opgeslagen in: {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Fout bij het opslaan van voorspellingen: {e}")

    logger.info("Forecast voorspellingsproces voltooid")

    return output_df

if __name__ == "__main__":
    logger.info("Start forecast.py")
    # Print huidige werkdirectory en basis pad voor debugging
    logger.info(f"Huidige werkdirectory: {os.getcwd()}")
    logger.info(f"Basis pad: {BASE_PATH}")
    logger.info(f"Forecast data pad: {FORECAST_DATA_PATH}")
    logger.info(f"Cooling model pad: {COOLING_MODEL_PATH}")
    logger.info(f"Heating model pad: {HEATING_MODEL_PATH}")

    predictions = make_predictions()
    if predictions is not None:
        logger.info(f"Voorspellingen gegenereerd voor {len(predictions)} dagen")
        # Toon eerste paar voorspellingen
        logger.info("\nVoorbeeld voorspellingen:")
        print(predictions.head())
    logger.info("forecast.py voltooid")
