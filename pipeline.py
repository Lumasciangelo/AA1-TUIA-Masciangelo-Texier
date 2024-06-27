from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
from preprocesamiento import ImputacionMedianaPorEstacion, AgruparDireccionesViento, GenerarDummies, CrearDiferenciasYEliminar
from preprocesamiento import DataProcessor, DropColumns, CrearDiferenciasYEliminar
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegressor

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
        ('imputacion_mediana_por_estacion', ImputacionMedianaPorEstacion(variables=['MinTemp', 'MaxTemp'])),
        ('agrupar_direcciones', AgruparDireccionesViento(variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('dummies', GenerarDummies(columnas_multiple = ['WindGustDir_agr', 'WindDir9am_agr', 'WindDir3pm_agr'], columnas_simple= ['RainToday'])),
        ('diferencias', CrearDiferenciasYEliminar(pares_columnas=[('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp'), ('Temp3pm', 'Temp9am'), ('Humidity9am', 'Humidity3pm')])),
        ('eliminar', DropColumns(variables=['Date', 'WindGustSpeed'])),
        ('estandarizar', StandardScaler()),
        ('regresion_logistica', LogisticRegression())
    ])
    #return pipeline

#modelo clasificación
pipeline_clasificacion.fit(x_train, y_train_clasificacion)

# Transformar los datos y verificar
#x_train_transformed = pipeline_clasificacion.transform(x_train)

#print("DataFrame transformado:")
#print(x_train_transformed)
#x_train_transformed.info()

joblib.dump(pipeline_clasificacion, 'pipeline_clas.joblib')

pipeline_regresion = Pipeline([
        ('eliminar', DropColumns(variables=['Date', 'WindGustSpeed'])),
        ('imputacion_mediana_por_estacion', ImputacionMedianaPorEstacion(variables=['MinTemp', 'MaxTemp'])),
        ('agrupar_direcciones', AgruparDireccionesViento(variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('dummies', GenerarDummies(columnas_multiple = ['WindGustDir_agr', 'WindDir9am_agr', 'WindDir3pm_agr'], columnas_simple= ['RainToday'])),
        ('diferencias', CrearDiferenciasYEliminar(pares_columnas=[('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp'), ('Temp3pm', 'Temp9am'), ('Humidity9am', 'Humidity3pm')])),
        ('estandarizar', StandardScaler()),
        ('regresionlineal', LinearRegressor())
    ])
    #return pipeline
    
#modelo regresion
pipeline_regresion.fit(x_train, y_train_regresion)

joblib.dump(pipeline_regresion, 'pipeline_reg.joblib')

