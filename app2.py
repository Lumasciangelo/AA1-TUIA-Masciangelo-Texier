import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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


class AgruparDireccionesViento(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def determinar_viento(self, viento):
        if viento in ["NE", "ENE", "ESE"]:
            return "E"
        elif viento in ["SSE", "SE", "SSW"]:
            return "S"
        elif viento in ["NNE", "NNW", "NW"]:
            return "N"
        else:
            return "W"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for var in self.variables:
            X_copy[f'{var}_agr'] = X_copy[var].apply(self.determinar_viento)
            X_copy.drop(columns=[var], inplace=True)
        return X_copy

viento = AgruparDireccionesViento(variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_viento = viento.fit(processed_df)
df_viento2 = df_viento.transform(processed_df)
df_viento2.info()

class GenerarDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columnas_multiple, columnas_simple):
        self.columnas_multiple = columnas_multiple
        self.columnas_simple = columnas_simple

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Procesar columnas que tienen múltiples variables relacionadas (ej: WindGustDir)
        for col_base in self.columnas_multiple:
            dummies_train = pd.get_dummies(X_copy[col_base], prefix=col_base, dtype=int, drop_first=True)
            X_copy = pd.concat([X_copy, dummies_train], axis=1)
            X_copy.drop(columns=[col_base], inplace=True)

        # Procesar columnas que tienen un único valor categórico binario (ej: RainToday)
        for col in self.columnas_simple:
            dummies_train = pd.get_dummies(X_copy[col], prefix=col, dtype=int, drop_first=True)
            X_copy = pd.concat([X_copy, dummies_train], axis=1)
            X_copy.drop(columns=[col], inplace=True)

        return X_copy

dummies = GenerarDummies(columnas_multiple = ['WindGustDir_agr', 'WindDir9am_agr', 'WindDir3pm_agr'], columnas_simple= ['RainToday'])
df_dummie = dummies.fit(df_viento2)
df_dummie2 = df_viento.transform(df_viento2)
df_dummie2.info()

class CrearDiferenciasYEliminar(BaseEstimator, TransformerMixin):
    def __init__(self, pares_columnas):
        self.pares_columnas = pares_columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col1, col2 in self.pares_columnas:
            diff_col_name = f'{col1}_menos_{col2}'
            X_copy[diff_col_name] = X_copy[col1] - X_copy[col2]
            X_copy.drop(columns=[col1, col2], inplace=True)
        return X_copy


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.drop(columns=self.variables, inplace=True)
        return X_copy

ImputacionMedianaPorEstacion(x_train)

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


# prediccion_reg = pipeline_entrenado_reg.predict(data_para_predecir)

st.write(f"{'Predicción clasificación:', prediccion_clas}\n{'Predicción regresión:', prediccion_reg}")