from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
from preprocesamiento import ImputacionMedianaPorEstacion, AgruparDireccionesViento, GenerarDummies, CrearDiferenciasYEliminar
from preprocesamiento import DataProcessor, DropColumns
import joblib


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


pipeline = Pipeline([
        ('imputacion_mediana_por_estacion', ImputacionMedianaPorEstacion(df_train=df_train, variables=['MinTemp', 'MaxTemp'])),
        ('agrupar_direcciones', AgruparDireccionesViento(df_train=df_train, variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('dummies', GenerarDummies(df_train= df_train, columnas_simple= ['RainToday'], columnas_multiples = ['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('diferencias', CrearDiferenciasYEliminar(pares_columnas=[('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp'), ('Temp3pm', 'Temp9am'), ('Humidity9am', 'Humidity3pm')])),
        ('eliminar', DropColumns(df_train=df_train, variables=['Date', 'Estacion'])),
        ('estandarizar', StandardScaler())
    ])
    #return pipeline
    

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

joblib.dump(pipeline, 'pipeline.joblib')
