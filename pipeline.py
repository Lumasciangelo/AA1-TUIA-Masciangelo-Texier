from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from preprocesamiento import ImputacionMedianaPorEstacion, AgruparDireccionesViento, CrearDummies, CrearDiferenciasYEliminar
from preprocesamiento import ImputacionMedianaPorDia, Data, ImputacionMaxima, ImputacionModasPorDia

def build_pipeline():
    pipeline = Pipeline([
        ('data', Data('weatherAUS.csv')),
        ('imputacion_por_dia', ImputacionMedianaPorDia(variables=['Evaporation', 'Rainfall', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed', 
                       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm',
                       'RainfallTomorrow'])),
        ('imputacion_maxima', ImputacionMaxima(variables = ['WindGustSpeed'])),
        ('imputacion_moda_por_dia', ImputacionModasPorDia(variables = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'])),
        ('imputacion_por_estacion', ImputacionMedianaPorEstacion(variables=['MinTemp', 'MaxTemp'])),
        ('agrupar_direcciones', AgruparDireccionesViento(variables=['WindGustDir', 'WindDir9am', 'WindDir3pm'])),
        ('dummies', CrearDummies(variables=['WindGustDir_agr', 'WindDir9am_agr', 'WindDir3pm_agr', 'RainToday'])),
        ('diferencias', CrearDiferenciasYEliminar(pares_columnas=[('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp')]))
    ])
    return pipeline