from sklearn.pipeline import Pipeline
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
from preprocesamiento import ImputacionMedianaPorEstacion, AgruparDireccionesViento, GenerarDummies, CrearDiferenciasYEliminar
from preprocesamiento import DataProcessor, DropColumns, CrearDiferenciasYEliminar, RedNeuronalClass
import joblib
import tensorflow as tf


# Usage
df = pd.read_csv('weatherAUS.csv')  # Replace with the actual path to your CSV file
processor = DataProcessor(df)
processed_df = processor.process()

        
# Dividir el DataFrame en conjuntos de entrenamiento y prueba
df_train = processed_df.loc[processed_df['Date'] < '2016-01-01']
df_test = processed_df.loc[processed_df['Date'] >= '2016-01-01']

x_train = df_train.drop(columns = ['RainTomorrow', 'RainfallTomorrow'], axis=1)
y_train_regresion = df_train['RainfallTomorrow']
y_train_clasificacion = df_train['RainTomorrow']


pipeline_clasificacion = Pipeline([
        ('imputacion_mediana_por_estacion', ImputacionMedianaPorEstacion(X=x_train, variables=['MinTemp', 'MaxTemp'])),
        ('agrupar_direcciones', AgruparDireccionesViento(X=x_train, variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('dummies', GenerarDummies(X= x_train, columnas_multiple = ['WindGustDir', 'WindDir9am', 'WindDir3pm'], columnas_simple= ['RainToday'])),
        ('diferencias', CrearDiferenciasYEliminar(X= x_train, pares_columnas=[('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp'), ('Temp3pm', 'Temp9am'), ('Humidity9am', 'Humidity3pm')])),
        ('eliminar', DropColumns(X=x_train, variables=['Date', 'Estacion'])),
        ('estandarizar', StandardScaler()),
        ('red_neuronal_clasificacion',RedNeuronalClass())
    ])
    #return pipeline

#modelo clasificación
pipeline_clasificacion.fit(x_train, y_train_clasificacion, epochs=13, batch_size=32)
joblib.dump(pipeline_clasificacion, 'pipeline_clas.joblib')

pipeline_regresion = Pipeline([
        ('imputacion_mediana_por_estacion', ImputacionMedianaPorEstacion(df_train=df_train, variables=['MinTemp', 'MaxTemp'])),
        ('agrupar_direcciones', AgruparDireccionesViento(df_train=df_train, variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('dummies', GenerarDummies(df_train= df_train, columnas_multiple = ['WindGustDir', 'WindDir9am', 'WindDir3pm'], columnas_simple= ['RainToday'])),
        ('diferencias', CrearDiferenciasYEliminar(df_train= df_train, pares_columnas=[('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp'), ('Temp3pm', 'Temp9am'), ('Humidity9am', 'Humidity3pm')])),
        ('eliminar', DropColumns(df_train=df_train, variables=['Date', 'Estacion'])),
        ('estandarizar', StandardScaler()),
        ('random_forest_regresion_optuna', )
    ])
    #return pipeline
    
#modelo regresion
pipeline_regresion.fit(x_train, y_train_regresion)

joblib.dump(pipeline_regresion, 'pipeline_res.joblib')
