# Weather Forecast & Model Management

Een geautomatiseerd systeem voor weergebaseerde foutvoorspelling, ontworpen om gemakkelijk te begrijpen en te beheren te zijn.

## Overzicht

Dit project is een interactieve Streamlit-applicatie die weergegevens verzamelt, verwerkt en gebruikt om foutmeldingen te voorspellen op basis van weeromstandigheden. Het systeem traint machine learning modellen en genereert voorspellingen die in een gebruiksvriendelijke interface worden weergegeven.

## Functionaliteiten

- **Gegevens verzamelen:** Haalt weervoorspellingen en historische weergegevens op via API's en laadt foutmeldingen uit Excel-bestanden.
- **Gegevens verwerken:** Reinigt de ruwe gegevens, voegt nuttige kenmerken toe en aggregeert gegevens tot dagelijkse samenvattingen.
- **Fouten classificeren:** Gebruikt regels en machine learning om foutmeldingen te categoriseren.
- **Modellen trainen:** Traint meerdere machine learning modellen om fouten te voorspellen op basis van weer- en tijdkenmerken.
- **Voorspellingen genereren:** Gebruikt de getrainde modellen om toekomstige fouten te voorspellen op basis van de nieuwste weervoorspellingen.
- **Resultaten visualiseren:** Toont voorspellingen, modelprestaties en gedetailleerde grafieken in een interactieve app.
- **Regelmatig hertrainen:** Werkt gegevens bij en hertraint modellen om in de loop van de tijd nauwkeurig te blijven.

## Installatie

1. Clone de repository:
   ```
   git clone https://github.com/yourusername/weather_forecast_app.git
   cd weather_forecast_app
   ```

2. Installeer de vereiste packages:
   ```
   pip install -r requirements.txt
   ```

## Gebruik

Start de Streamlit-app:
```
streamlit run streamlit_app.py
```

De app heeft drie hoofdsecties:

1. **Latest Predictions:** Voer de forecast pipeline uit om de nieuwste voorspellingen te genereren en te bekijken.
2. **Models:** Upload nieuwe foutmeldingsgegevens en hertrain de modellen.
3. **Explanation:** Bekijk gedetailleerde uitleg over hoe het systeem werkt.

## Mapstructuur

- **`data/script/`**
  - `forecast.py`: Haalt weervoorspellingsgegevens op.
  - `historical.py`: Haalt historische weergegevens op.
  - `process_daily_weather.py`: Reinigt en verrijkt weergegevens.
  - `classifie_vks.py`: Classificeert foutmeldingen.
  - `result.py`: Maakt visualisaties van foutgegevens.

- **`data/`**
  - `update.py`: Werkt datasets bij en activeert hertraining.
  - `csv-api/` en `csv-daily/`: Slaan ruwe en verwerkte weergegevens op.
  - `notifications/`: Bevat Excel-bestanden met foutmeldingen.

- **`predictions/`**
  - `forecast.py`: Genereert foutvoorspellingen met getrainde modellen.
  - `cooling.py` en `heating.py`: Trainen, optimaliseren, evalueren en slaan modellen op.
  - `best-model/`: Slaat de beste modellen en hun metrics op.
  - `plots/`: Bevat visualisaties van modelprestaties.

- **`streamlit/`**
  - `main.py`: De app-interface om de pipeline uit te voeren, modellen te hertrainen en resultaten te bekijken.

## Deployment

Deze app kan worden gedeployed op Streamlit Cloud:

1. Push de code naar een GitHub repository.
2. Ga naar [Streamlit Cloud](https://streamlit.io/cloud).
3. Koppel je GitHub-account en selecteer de repository.
4. Stel `streamlit_app.py` in als het hoofdbestand.
5. Klik op "Deploy".

## Vereisten

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

## Licentie

[MIT](LICENSE)
