import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pandas as pd


st.title('Rainfall dataset predictions')

# Definir el rango de fechas permitido
min_date = datetime(2008, 1, 20)
max_date = datetime(2017, 6, 24)

# Crear un widget de entrada de fecha
Date = st.date_input('Selecciona una fecha', value=min_date, min_value=min_date, max_value=max_date)

MaxTemp = st.slider('MaxTemp', -8.5, 34.0, 12.2)
MinTemp = st.slider('MinTemp', -5.0, 49.0, 23.2)
Rainfall = st.slider('Rainfall', 0.0, 371.0, 2.4)
Evaporation = st.slider('Evaporation', 0.0, 145.0, 5.5)
Sunshine = st.slider('Sunshine', 0.0, 14.5, 7.6)
WindGustDir = st.selectbox('WindgustDir', ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
WindGustSpeed = st.slider('WindGustSpeed', 6.0, 135.0, 40.0)
WindDir9am = st.selectbox('WindDir9am', ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
WindDir3pm = st.selectbox('WindDir3pm', ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
WindSpeed9am = st.slider('WindSpeed9am', 0.0, 130.0, 14.0)
WindSpeed3pm = st.slider('WindSpeed3pm', 0.0, 87.0, 18.7)
Humidity9am = st.slider('Humidity9am', 0.0, 100.0, 68.9)
Humidity3pm = st.slider('Humidity3pm', 0.0, 100.0, 51.5)
Pressure9am = st.slider('Pressure9am', 970.0, 1041.0, 1017.6)
Pressure3pm = st.slider('Pressure3pm', 970.0, 1041.0, 1015.3)
Cloud9am = st.slider('Cloud9am', 0.0, 9.0, 4.4)
Cloud3pm = st.slider('Cloud3pm', 0.0, 9.0, 4.5)
Temp9am = st.slider('Cloud3pm', -8.0, 40.0, 17.0)
Temp3pm = st.slider('Cloud3pm', -6.0, 47.0, 21.7)
Location = st.selectbox('Location',['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'])
RainToday = st.selectbox('Raintoday', ['Yes', 'No'])

# Crear un DataFrame con los datos ingresados por el usuario
data_para_predecir = pd.DataFrame({
    'Date': [Date],
    'Location': [Location],
    'MinTemp': [MinTemp],
    'MaxTemp': [MaxTemp],
    'Rainfall': [Rainfall],
    'Evaporation': [Evaporation],
    'Sunshine': [Sunshine],
    'WindGustDir': [WindGustDir],
    'WindGustSpeed': [WindGustSpeed],
    'WindDir9am': [WindDir9am],
    'WindDir3pm': [WindDir3pm],
    'WindSpeed9am': [WindSpeed9am],
    'WindSpeed3pm': [WindSpeed3pm],
    'Humidity9am': [Humidity9am],
    'Humidity3pm': [Humidity3pm],
    'Pressure9am': [Pressure9am],
    'Pressure3pm': [Pressure3pm],
    'Cloud9am': [Cloud9am],
    'Cloud3pm': [Cloud3pm],
    'Temp9am': [Temp9am],
    'Temp3pm': [Temp3pm],
    'RainToday': [RainToday]
})


class DataStreamlit:
    def __init__(self, df):
        self.df = df

    def filter_locations(self):
        self.df = self.df.drop(['Location'], axis=1)

    def convertir_a_datetime(self, date_column):
        self.df[date_column] = pd.to_datetime(self.df[date_column])

    def determinar_estacion(self, date):
        mes = date.month
        if 3 <= mes <= 5:
            return "Otoño"
        elif 6 <= mes <= 8:
            return "Invierno"
        elif 9 <= mes <= 11:
            return "Primavera"
        else:
            return "Verano"

    def asignar_estacion(self, date_column):
        self.df['Estacion'] = self.df[date_column].apply(self.determinar_estacion)

    def process(self):
        # Filter locations
        self.filter_locations()

        # Convert date and assign season
        self.convertir_a_datetime('Date')
        self.asignar_estacion('Date')
        
        return self.df


df = data_para_predecir  # Replace with the actual path to your CSV file
processor = DataStreamlit(df)
processed_df = processor.process()
processed_df.info()

columns=['WindGustDir', 'WindDir9am', 'WindDir3pm']        
for var in columns:
    processed_df[f'{var}_agr'] = processed_df[var]
    # processed_df.drop(columns=[var], inplace=True)

processed_df['RainToday2'] = processed_df['RainToday']

# Generar dummies para las columnas categóricas
data_para_predecir_dummies = pd.get_dummies(processed_df, columns=['RainToday', 'WindGustDir_agr', 'WindDir9am_agr', 'WindDir3pm_agr'])

# Obtener todas las columnas dummy esperadas por el modelo
dummy_columns = ['WindDir3pm_agr_N', 'WindDir3pm_agr_S', 'WindDir3pm_agr_W','WindDir9am_agr_N', 'WindDir9am_agr_S', 'WindDir9am_agr_W', 'WindGustDir_agr_N', 'WindGustDir_agr_S', 'WindGustDir_agr_W' ]

data_para_predecir_dummies['RainToday'] = data_para_predecir_dummies['RainToday2']
data_para_predecir_dummies = data_para_predecir_dummies.drop('RainToday2', axis=1)

# Añadir las columnas dummy faltantes con valor 0
for col in dummy_columns:
    if col not in data_para_predecir_dummies.columns:
        data_para_predecir_dummies[col] = 0
    
# data_para_predecir_dummies = data_para_predecir_dummies.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)
data_para_predecir_dummies.info()

orden_columnas = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindDir9am',
                   'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
    'Temp9am', 'Temp3pm', 'RainToday', 'Estacion',
    'WindGustDir_agr_N', 'WindGustDir_agr_S', 'WindGustDir_agr_W', 
    'WindDir9am_agr_N', 'WindDir9am_agr_S', 'WindDir9am_agr_W', 'WindDir3pm_agr_N', 
    'WindDir3pm_agr_S', 'WindDir3pm_agr_W', 'RainToday_Yes']

# # Ordenar las columnas de acuerdo a lo esperado por el modelo
data_para_predecir_dummies = data_para_predecir_dummies[orden_columnas]


# Cargar el pipeline entrenado de clasificación
joblib_file = r'pipeline_clas.joblib'
try:
    pipeline_entrenado_clas = joblib.load(joblib_file)
    print('Pipeline cargado exitosamente')
except FileNotFoundError:
    print(f"El archivo {joblib_file} no existe")
except Exception as e:
    print(f"Ocurrió un error al cargar el papeline: {e}")

# Cargar el pipeline entrenado de regresión
joblib_file2 = r'pipeline_reg.joblib'
try:
    pipeline_entrenado_reg = joblib.load(joblib_file2)
    print('Pipeline cargado exitosamente')
except FileNotFoundError:
    print(f"El archivo {joblib_file2} no existe")
except Exception as e:
    print(f"Ocurrió un error al cargar el papeline: {e}")

# Realizar la predicción si ambos pipelines se cargaron correctamente
if st.button('Predecir') and 'pipeline_entrenado_clas' in locals() and 'pipeline_entrenado_reg' in locals():
    try:
        prediccion_clas = pipeline_entrenado_clas.predict(data_para_predecir_dummies)
        prediccion_reg = pipeline_entrenado_reg.predict(data_para_predecir_dummies)
        st.subheader('Predicción:')
        st.write(f"Predicción clasificación: {'Va a llover mañana' if prediccion_clas[0] == 1 else 'No va a llover mañana'}")
        st.write(f"Predicción regresión: {prediccion_reg[0]}")
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")


st.write(f"{'Predicción clasificación:', prediccion_clas}\n{'Predicción regresión:', prediccion_reg}")