from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import numpy as np

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
        mediana_variables = ['Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm']
        maxima_variables = ['WindSpeed9am', 'WindSpeed3pm']
        moda_variables = ['RainToday', 'RainTomorrow']
        
        self.imputacion_mediana_por_dia(mediana_variables)
        self.imputacion_maxima(maxima_variables)
        self.imputacion_modas_por_dia(moda_variables)
        
        self.imputacion_winddir9am('WindDir9am')
        self.imputacion_windgustdir('WindGustDir')
        
        # Convert date and assign season
        self.convertir_a_datetime('Date')
        self.asignar_estacion('Date')
        
        return self.df


class ImputacionMedianaPorEstacion:
    def __init__(self, variables):
        self.variables = variables
        self.medianas_por_estacion_train = {}

    def fit(self, df_train, variables):
        # Calcular la mediana para cada variable por cada grupo en la columna 'Estacion' en el conjunto de entrenamiento
        self.medianas_por_estacion_train = {variable: df_train.groupby(self.variables)[variable].median() for variable in variables}

    def transformar_fila(self, fila, variables):
        for variable in variables:
            if pd.isnull(fila[variable]):
                fila[variable] = self.medianas_por_estacion_train[variable][fila[self.variables]]
        return fila

    def transform(self, df_train, variables):
        # Aplicar la función a cada fila del DataFrame
        df_train = df_train.apply(lambda fila: self.transformar_fila(fila, variables), axis=1)
        return df_train


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

    def fit(self):
        return self #este no se bien que va...

    def transform(self, df_train):
        for var in self.variables:
            df_train[f'{var}_agr'] = df_train[var].apply(lambda x: self.determinar_viento(x))
            df_train = df_train.drop(var, axis=1)
        return df_train

class CrearDummies:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, df_train):
        return self

    def transform(self, df_train):
        for var in self.variables:
            dummies = pd.get_dummies(df_train[var], prefix=var, drop_first=True)
            df_train = df_train.drop(var, axis=1)
            df_train = pd.concat([df_train, dummies], axis=1)
        return df_train

class CrearDiferenciasYEliminar:
    def __init__(self, pares_columnas):
        self.pares_columnas = pares_columnas

    def fit(self, df_train, y=None):
        return self

    def transform(self, df_train):
        for col1, col2 in self.pares_columnas:
            diff_col_name = f'{col1}_menos_{col2}'
            X[diff_col_name] = X[col1] - X[col2]
            X = X.drop([col1, col2], axis=1)
        return X

class DropColumns:
    def __init__(self, variables):
        self.variables = variables 

    def eliminar(self, df_train):

## Eliminar date y estacion 
## estandarizar
