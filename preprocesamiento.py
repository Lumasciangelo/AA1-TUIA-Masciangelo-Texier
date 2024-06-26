from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense


class DataProcessor:
    def __init__(self, df):
        self.df = df

    def filter_locations(self):
        locations_to_keep = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport']
        self.df = self.df[self.df['Location'].isin(locations_to_keep)]
        self.df = self.df.drop(['Location', 'Unnamed: 0'], axis=1)

    def imputacion_mediana_por_dia(self, variables):
        medianas_por_dia = {variable: self.df.groupby('Date')[variable].median() for variable in variables}
        for variable in variables:
            self.df[variable] = self.df.apply(lambda row: medianas_por_dia[variable][row['Date']] if pd.isnull(row[variable]) else row[variable], axis=1)

    def imputacion_maxima(self, variables):
        maxima = np.maximum(self.df['WindSpeed3pm'], self.df['WindSpeed9am'])
        for variable in variables:
            self.df[variable] = self.df.apply(lambda row: maxima if pd.isnull(row[variable]) else row[variable], axis=1)

    def imputacion_modas_por_dia(self, variables):
        modas_por_dia = {variable: self.df.groupby('Date')[variable].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan) for variable in variables}
        for variable in variables:
            self.df[variable] = self.df.apply(lambda row: modas_por_dia[variable][row['Date']] if pd.isnull(row[variable]) else row[variable], axis=1)

    def imputacion_winddir9am(self, variable):
        self.df.loc[self.df[variable].isna(), 'WindDir9am'] = self.df.loc[self.df[variable].isna(), 'WindGustDir']

    def imputacion_windgustdir(self, variable):
        self.df[variable].fillna('N', inplace=True)

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
        
        # Imputations
        mediana_variables = ['Evaporation', 'Rainfall', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm',
                       'RainfallTomorrow']
        maxima_variables = ['WindGustSpeed']
        moda_variables = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
        
        self.imputacion_mediana_por_dia(mediana_variables)
        self.imputacion_maxima(maxima_variables)
        self.imputacion_modas_por_dia(moda_variables)
        
        self.imputacion_winddir9am('WindDir9am')
        self.imputacion_windgustdir('WindGustDir')
        
        # Convert date and assign season
        self.convertir_a_datetime('Date')
        self.asignar_estacion('Date')
        
        return self.df



class ImputacionMedianaPorEstacion(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.medianas_por_estacion_train = {}

    def fit(self, X, y=None):
        for variable in self.variables:
            self.medianas_por_estacion_train[variable] = X.groupby('Estacion')[variable].median()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for variable in self.variables:
            X_copy[variable] = X_copy.apply(lambda row: row[variable] if pd.notnull(row[variable]) else self.medianas_por_estacion_train[variable][row['Estacion']], axis=1)
        return X_copy.drop(columns=['Estacion'])


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

    

#MODELOS

#modelo clasificacion
#modelo de red neuronal
# class RedNeuronalClass:
#     def __init__(self):
#         self.model = self.build_model()

#     def build_model(self, random_state = 12):
#         model = Sequential()
#         model.add(Dense(38, activation='relu', input_shape=(38,)))  # Capa oculta con 38 neuronas y función de activación ReLU
#         model.add(Dense(1, activation='sigmoid'))  # Capa de salida con 1 neurona y función de activación sigmoid
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         return model

#     def fit(self, X, y, epochs=13, batch_size=32, validation_data=None):
#         history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
#         return history
    
class RedNeuronalClass:
    def __init__(self, input_dim=20, epochs=13, batch_size=32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Dense(38, activation='relu', input_shape=(self.input_dim,)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)
        return self


