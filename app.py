import streamlit as st
import numpy as np
import joblib

st.title('Rainfall dataset predictions')

joblib_file = r'C:\AA1\AA1-TUIA-MASCIANGELO-TEXIER\pipeline_clas.joblib'
try:
    pipeline_entrenado_clas = joblib.load(joblib_file)
    print('Pipeline cargado exitosamente')
except FileNotFoundError:
    print(f"El archivo {joblib_file} no existe")
except Exception as e:
    print(f"Ocurrió un error al cargar el papeline: {e}")

joblib_file2 = r'C:\AA1\AA1-TUIA-MASCIANGELO-TEXIER\pipeline_reg.joblib'
try:
    pipeline_entrenado_reg = joblib.load(joblib_file2)
    print('Pipeline cargado exitosamente')
except FileNotFoundError:
    print(f"El archivo {joblib_file2} no existe")
except Exception as e:
    print(f"Ocurrií un error al cargar el papeline: {e}")

Date = st.date_input('Date','2008/12/01','2017/06/24', )
MaxTemp = st.slider('MaxTemp', -8.5, 34.0, 12.2)
MinTemp = st.slider('MinTemp', -5.0, 49.0, 23.2)
Rainfall = st.slider('Rainfall', 0.0, 371.0, 2.4)
Evaporation = st.slider('Evaporation', 0.0, 145.0, 5.5)
Sunshine = st.slider('Sunshine', 0.0, 14.5, 7.6)
WindGustDir = st.multiselect('WindgustDir', ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
WindGustSpeed = st.slider('WindGustSpeed', 6.0, 135.0, 40.0)
WindDir9am = st.multiselect('WindDir9am', ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
WindDir3pm = st.multiselect('WindDir3pm', ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE',
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
Location = st.multiselect('Location',['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'])
RainToday = st.multiselect('Raintoday', ['Yes', 'No'])


data_para_predecir = np.array([[Date, MaxTemp, MinTemp, Rainfall, Evaporation, Sunshine, WindGustDir, 
                                WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm,
                                Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm,
                                Temp9am, Temp3pm, Location, RainToday]])

prediccion_clas = pipeline_entrenado_clas.predict(data_para_predecir)

prediccion_reg = pipeline_entrenado_reg.predict(data_para_predecir)

st.write(f"{'Predicción clasificación:', prediccion_clas}\n{'Predicción regresión:', prediccion_reg}")