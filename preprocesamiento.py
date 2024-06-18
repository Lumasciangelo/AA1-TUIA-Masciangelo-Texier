from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Data:
    def __init__(self, df):
        self.df = pd.read_csv('df')

    def filter_locations(self):
        locations_to_keep = [' Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport' ]
        df = self.df[self.df['Location'].isin(locations_to_keep)]
        df = df.drop(['Location', 'Unnamed: 0'], axis=1)
        return df
        
class ImputacionMedianaPorDia:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, df):
        self.medianas_por_dia = {variable: df.groupby('Date')[variable].median() for variable in self.variables}
        return self

    def transform(self, df):
        for variable in self.variables:
            df[variable] = df.apply(lambda row: self.medianas_por_dia[variable][row['Date']] if pd.isnull(row[variable]) else row[variable], axis=1)
        return df
    

class ImputacionMaxima:
    def __init__(self, variables):
        self.variables = variables
        
    def fit(self, df):
        self.maxima = {variable: np.maximum(df['WindSpeed3pm'], df['WindSpeed9am']) for variable in self.variables}
        return self

    def transform(self, df):
        for variable in self.variables:
            df[variable] = df.apply(lambda row: self.maxima[variable] if pd.isnull(row[variable]) else row[variable], axis=1)
        return df


class ImputacionModasPorDia:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, df):
        self.modas_por_dia = {variable: df.groupby('Date')[variable].mode() for variable in self.variables}
        return self

    def transform(self, df):
        for variable in self.variables:
            df[variable] = df.apply(lambda row: self.modas_por_dia[variable][row['Date']] if pd.isnull(row[variable]) else row[variable], axis=1)
        return df
    

class ImputacionMedianaPorEstacion:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, df):
        self.medianas_por_estacion = {variable: df.groupby('Estacion')[variable].median() for variable in self.variables}
        return self

    def transform(self, df):
        for variable in self.variables:
            df[variable] = df.apply(lambda row: self.medianas_por_estacion[variable][row['Estacion']] if pd.isnull(row[variable]) else row[variable], axis=1)
        return df

class AgruparDireccionesViento:
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
        for var in self.variables:
            X[f'{var}_agr'] = X[var].apply(lambda x: self.determinar_viento(x))
            X = X.drop(var, axis=1)
        return X

class CrearDummies:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            dummies = pd.get_dummies(X[var], prefix=var, drop_first=True)
            X = X.drop(var, axis=1)
            X = pd.concat([X, dummies], axis=1)
        return X

class CrearDiferenciasYEliminar:
    def __init__(self, pares_columnas):
        self.pares_columnas = pares_columnas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col1, col2 in self.pares_columnas:
            diff_col_name = f'{col1}_menos_{col2}'
            X[diff_col_name] = X[col1] - X[col2]
            X = X.drop([col1, col2], axis=1)
        return X
