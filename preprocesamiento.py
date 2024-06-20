from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import numpy as np
import tensorflow as tf

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
    def __init__(self, X, variables, y= None):
        self.X = X
        self.variables = variables
        self.medianas_por_estacion_train = self._calcular_medianas()

    def _calcular_medianas(self):
        # Calcular la mediana para cada variable por cada grupo en la columna 'Estacion' en el conjunto de entrenamiento
        medianas_por_estacion_train = {variable: self.X.groupby('Estacion')[variable].median() for variable in self.variables}
        return medianas_por_estacion_train

    def _llenar_faltantes_mediana_por_estacion(self, fila):
        for variable in self.variables:
            if pd.isnull(fila[variable]):
                fila[variable] = self.medianas_por_estacion_train[variable][fila['Estacion']]
        return fila

    def imputar(self):
        # Aplicar la función a cada fila del conjunto de entrenamiento
        self.df_train = self.X.apply(lambda fila: self._llenar_faltantes_mediana_por_estacion(fila), axis=1)
        return self.X


class AgruparDireccionesViento:
    def __init__(self, variables, X, y=None):
        self.variables = variables
        self.X = X

    def determinar_viento(self, viento):
        if viento in ["NE", "ENE", "ESE"]:
            return "E"
        elif viento in ["SSE", "SE", "SSW"]:
            return "S"
        elif viento in ["NNE", "NNW", "NW"]:
            return "N"
        else:
            return "W"

    def transform(self):
        for var in self.variables:
            self.X[f'{var}_agr'] = self.X[var].apply(self.determinar_viento)
            self.X.drop(columns=[var], inplace=True)
        return self.X

class GenerarDummies:
    def __init__(self, X, columnas_multiple, columnas_simple, y= None):
        self.X = X
        self.columnas_multiple = columnas_multiple
        self.columnas_simple = columnas_simple

    def generar_dummies(self):
        # Procesar columnas que tienen múltiples variables relacionadas (ej: WindGustDir)
        for col_base in self.columnas_multiple:
            # Crear dummies para cada variable base en el conjunto de entrenamiento
            dummies_train = pd.get_dummies(self.X[f'{col_base}_dummie'], dtype=int, drop_first=True)
            
            # Renombrar las columnas dummies para agregar el prefijo de la columna base
            dummies_train = dummies_train.rename(columns=lambda x: f'{col_base}_{x}')
            
            # Eliminar las columnas originales y las agregadas por la función agrupar_direcciones_viento
            self.X = self.X.drop([col_base, f'{col_base}_dummie'], axis=1)
            
            # Concatenar las dummies con el DataFrame original
            self.X = pd.concat([self.X, dummies_train], axis=1)

        # Procesar columnas que tienen un único valor categórico binario (ej: RainToday)
        for col in self.columnas_simple:
            dummies_train = pd.get_dummies(self.X[col], dtype=int, drop_first=True)
            
            # Renombrar la columna dummy a simplemente el nombre de la variable original
            dummies_train = dummies_train.rename(columns={'Yes': f'{col}_Yes'})
            
            # Eliminar las columnas originales
            self.df_train = self.X.drop(col, axis=1)
            
            # Concatenar las dummies con el DataFrame original
            self.X = pd.concat([self.X, dummies_train], axis=1)
        
        return self.X

class CrearDiferenciasYEliminar:
    def __init__(self, X, pares_columnas, y=None):
        self.X = X
        self.pares_columnas = pares_columnas

    def crear_diferencias(self):
        for col1, col2 in self.pares_columnas:
            # Crear la nueva columna de diferencia para el conjunto de entrenamiento
            diff_col_name = f'{col1}_menos_{col2}'
            self.X[diff_col_name] = self.X[col1] - self.X[col2]
            
            # Eliminar las columnas originales
            self.X = self.X.drop([col1, col2], axis=1)
        
        return self.X

class DropColumns:
    def __init__(self, variables, X, y=None):
        self.variables = variables
        self.X = X

    def eliminar(self):
        self.X = self.X.drop(self.variables, axis=1)
        return self.X
    

#MODELOS

#modelo clasificacion
#modelo de red neuronal
class RedNeuronalClass:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(38, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


