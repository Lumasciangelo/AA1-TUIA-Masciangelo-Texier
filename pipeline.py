from sklearn.pipeline import Pipeline
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
from preprocesamiento import ImputacionMedianaPorEstacion, AgruparDireccionesViento, GenerarDummies, CrearDiferenciasYEliminar
from preprocesamiento import DataProcessor, DropColumns, CrearDiferenciasYEliminar, RedNeuronalClass
import joblib
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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
        ('eliminar', DropColumns(variables=['Date'])),
        ('imputacion_mediana_por_estacion', ImputacionMedianaPorEstacion(variables=['MinTemp', 'MaxTemp'])),
        ('agrupar_direcciones', AgruparDireccionesViento(variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('dummies', GenerarDummies(columnas_multiple = ['WindGustDir_agr', 'WindDir9am_agr', 'WindDir3pm_agr'], columnas_simple= ['RainToday'])),
        ('diferencias', CrearDiferenciasYEliminar(pares_columnas=[('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp'), ('Temp3pm', 'Temp9am'), ('Humidity9am', 'Humidity3pm')])),
        # ('estandarizar', StandardScaler())
        # ('red_neuronal_clasificacion',RedNeuronalClass())
    ])
    #return pipeline

#modelo clasificación

# Supongamos que x_train es tu DataFrame de entrenamiento
pipeline_clasificacion.fit(x_train)

# Transformar los datos y verificar
x_train_transformed = pipeline_clasificacion.transform(x_train)

print("DataFrame transformado:")
print(x_train_transformed)
x_train_transformed.info()

pipeline_clasificacion.fit(x_train, y_train_clasificacion)

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

