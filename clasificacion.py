#### Hacer una funcion que impute por la mediana por dia, y luego pasarle todas las variables 
#### Otra que cree la variable estacion, donde adentro pase la variable date a datetime
#### Otra funcion que impute los datos de Windgustspeed entre el mayor de 9am y 3pm 
#### Otra que impute por la moda WindGustDir, WindDir9am, y WindDir3pm, raintoday, raintomorrow
#### Este dia 2008-02-27, no hay valores de winddir9am entonces lo imputamos con la de windgustdir

#### Completamos los valores faltantes de WindGustDir con el valor que más aparece en esa estacion

#### Dividir en train y test en la fecha '2016-01-01'
#### Otra que impute la temperatura minima, y la maxima por la mediana de cada estacion

#### Agrupar las direcciones de viento (las 3) en norte, sur, este y oeste 
#### Generamos dummies para las variables categoricas 
#### Eliminamos la variable estacion 
#### Una funcion que haga la diferencia de temp, presion, etc. Y que elimine todas las variables que no va a usar
#### Estandarizar con standarscaler

## Para clasificación: eliminar rainfalltomorrow, usamos randomforestregressor, 
## Para regresion lineal: eliminar raintomorrow, usamos redes neuronales, 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('weatherAUS.csv')

# Las localidades que queremos evaluar son Adelaide, Canberra, Cobar, Dartmoor, Melbourne, MelbourneAirport, MountGambier, Sydney y SydneyAirport. El resto las eliminamos según el enunciado del tp
categorias_importantes = [' Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport' ]
df_filtrado = df[df['Location'].isin(categorias_importantes)]
df_filtrado = df_filtrado.drop(['Location', 'Unnamed: 0'], axis = 1)


## LLENAR CON LA MEDIANA POR DIA
def imputacion_mediana_por_dia(df, variables):
    # Calcular la mediana para cada variable por cada grupo en la columna 'Date'
    medianas_por_fecha = {variable: df.groupby('Date')[variable].median() for variable in variables}
    
    # Definir una función para aplicar a cada fila
    def llenar_faltantes_mediana_por_dia(fila):
        for variable in variables:
            if pd.isnull(fila[variable]):
                fila[variable] = medianas_por_fecha[variable][fila['Date']]
        return fila
    
    # Aplicar la función a cada fila del DataFrame
    df = df.apply(llenar_faltantes_mediana_por_dia, axis=1)
    return df

# Imputar los valores faltantes utilizando la mediana por fecha para varias variables
medianas_a_imputar = ['Evaporation', 'Rainfall', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed', 
                       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'Cloud9am', 'Cloud3pm',
                       'RainfallTomorrow']
df_filtrado = imputacion_mediana_por_dia(df, medianas_a_imputar)



## PARA WINDGUSTSPEED DESPUES DE LLENAR CON LA MEDIANA 
if df_filtrado['WindGustSpeed'].isnull().any():
    df_filtrado['WindGustSpeed'] = np.maximum(df_filtrado['WindSpeed3pm'], df_filtrado['WindSpeed9am'])



## LLENAR CON LA MODA
def imputacion_moda(df, variables):
    # Calcular la moda por fecha para cada variable
    modas_por_fecha = df.groupby('Date')[variables].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    
    # Definir función para llenar los valores faltantes con la moda
    def llenar_faltantes(fila):
        for variable in variables:
            # Verificar si el valor es NaN
            if pd.isnull(fila[variable]):
                # Obtener la moda para la fecha específica de la fila
                moda_fecha = modas_por_fecha[variable].loc[fila['Date']]
                # Si la moda es None (si todos los valores para esa fecha son NaN), dejar el valor como NaN
                if moda_fecha is not None:
                    fila[variable] = moda_fecha
        return fila
    
    # Aplicar la función a cada fila del DataFrame
    df_filtrado = df.apply(llenar_faltantes, axis=1)
    return df_filtrado

modas_a_imputar = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
df_filtrado = imputacion_moda(df, modas_a_imputar)


## PARA WINDDIR9AM DESPUES DE APLICAR LA MODA
df_filtrado.loc[df_filtrado['WindDir9am'].isna(), 'WindDir9am'] = df_filtrado.loc[df_filtrado['WindDir9am'].isna(), 'WindGustDir']

## PARA WINDGUSTDIR DESPUES DE APLICAR LA MODA
df_filtrado['WindGustDir'].fillna('N', inplace=True)



## PARA DETERMINAR LA ESTACION
# Armamos la funcion para determinar la estacion
def estaciones(df):
    # Primero transformo la columna date a tipo datetime
    df_filtrado['Date'] = pd.to_datetime(df_filtrado['Date'])

    def determinar_estacion(fecha):
        # extraemos el mes
        mes = fecha.month
        # Determinamos las estaciones
        if  3<= mes <=5:
            return "Otoño"
        elif 6<= mes <= 8:
            return "Invierno"
        elif 9 <= mes <= 11:
            return "Primavera"
        else:
            return "Verano"

    # Aplicamos la función determinar_estacion al DataFrame df_filtrado
    df_filtrado['Estacion'] = df_filtrado['Date'].apply(lambda x: determinar_estacion(x))
    return df_filtrado

df_filtrado = estaciones(df_filtrado)


## DIVIDIR EN TRAIN Y EN TEST
df_train = df_filtrado.loc[df_filtrado['Date'] < '2016-01-01']
df_test = df_filtrado.loc[df_filtrado['Date'] >= '2016-01-01']



## LLENAR CON LA MEDIANA POR ESTACION
def imputacion_mediana_por_estacion(df_train, df_test, variables):
    # Calcular la mediana para cada variable por cada grupo en la columna 'Estacion' en el conjunto de entrenamiento
    medianas_por_estacion_train = {variable: df_train.groupby('Estacion')[variable].median() for variable in variables}
    
    # Definir una función para aplicar a cada fila del DataFrame
    def llenar_faltantes_mediana_por_estacion(fila, medianas):
        for variable in variables:
            if pd.isnull(fila[variable]):
                fila[variable] = medianas[variable][fila['Estacion']]
        return fila

    # Aplicar la función a cada fila del conjunto de entrenamiento
    df_train = df_train.apply(lambda fila: llenar_faltantes_mediana_por_estacion(fila, medianas_por_estacion_train), axis=1)
    
    # Aplicar la función a cada fila del conjunto de prueba
    df_test = df_test.apply(lambda fila: llenar_faltantes_mediana_por_estacion(fila, medianas_por_estacion_train), axis=1)
    
    return df_train, df_test

medianas_por_estacion = ['MinTemp', 'MaxTemp']
df_filtrado = imputacion_mediana_por_estacion(df_train, df_test, medianas_por_estacion)


## PARA AGRUPAR SEGUN LA DIRECCION DEL VIENTO
def agrupar_direcciones_viento(df_train, df_test, columnas):
    def determinar_viento(viento):
        # Determinamos las estaciones
        if viento in ["NE", "ENE", "ESE"]:
            return "E"
        elif viento in ["SSE", "SE", "SSW"]:
            return "S"
        elif viento in ["NNE", "NNW", "NW"]:
            return "N"
        else:
            return "W"
    
    # Definir una función interna para aplicar la agrupación a cada valor de una columna
    def aplicar_direccion(fila, columna):
        return determinar_viento(fila[columna])
    
    for columna in columnas:
        columna_agr = columna + '_agr'
        df_train[columna_agr] = df_train[columna].apply(lambda x: aplicar_direccion({'columna': x}, 'columna'))
        df_test[columna_agr] = df_test[columna].apply(lambda x: aplicar_direccion({'columna': x}, 'columna'))
        # Eliminar la columna original
        df_train.drop(columns=[columna], inplace=True)
        df_test.drop(columns=[columna], inplace=True)
    
    return df_train, df_test

vientos_agr = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
df_filtrado = agrupar_direcciones_viento(df_train, df_test, vientos_agr)



## HACER LA DIFERENCIA DE VARIABLES 





## GENERAR DUMMIES 
def generar_dummies_personalizadas(df_train, df_test, columnas_multiple, columnas_simple):
    # Procesar columnas que tienen múltiples variables relacionadas (ej: WindGustDir)
    for col_base in columnas_multiple:
        # Crear dummies para cada variable base en el conjunto de entrenamiento y prueba
        dummies_train = pd.get_dummies(df_train[f'{col_base}_agr'], dtype=int, drop_first=True)
        dummies_test = pd.get_dummies(df_test[f'{col_base}_agr'], dtype=int, drop_first=True)
        
        # Renombrar las columnas dummies para agregar el prefijo de la columna base
        dummies_train = dummies_train.rename(columns=lambda x: f'{col_base}_{x}')
        dummies_test = dummies_test.rename(columns=lambda x: f'{col_base}_{x}')
        
        # Eliminar las columnas originales y las agregadas por la función agrupar_direcciones_viento
        df_train = df_train.drop([col_base, f'{col_base}_agr'], axis=1)
        df_test = df_test.drop([col_base, f'{col_base}_agr'], axis=1)
        
        # Concatenar las dummies con los DataFrames originales
        df_train = pd.concat([df_train, dummies_train], axis=1)
        df_test = pd.concat([df_test, dummies_test], axis=1)

    # Procesar columnas que tienen un único valor categórico binario (ej: RainToday)
    for col in columnas_simple:
        dummies_train = pd.get_dummies(df_train[col], dtype=int, drop_first=True)
        dummies_test = pd.get_dummies(df_test[col], dtype=int, drop_first=True)
        
        # Renombrar la columna dummy a simplemente el nombre de la variable original
        dummies_train = dummies_train.rename(columns={'Yes': col})
        dummies_test = dummies_test.rename(columns={'Yes': col})
        
        # Eliminar las columnas originales
        df_train = df_train.drop(col, axis=1)
        df_test = df_test.drop(col, axis=1)
        
        # Concatenar las dummies con los DataFrames originales
        df_train = pd.concat([df_train, dummies_train], axis=1)
        df_test = pd.concat([df_test, dummies_test], axis=1)
    
    return df_train, df_test

dummies_multiples = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
dummies_simples = ['RainToday'] # RainTomorrow?


## ELIMINAR LAS VARIABLES ESTACION Y DATE
df_train = df_train.drop('Estacion', axis=1)
df_test = df_test.drop('Estacion', axis=1)
df_train = df_train.drop('Date', axis=1)
df_test = df_test.drop('Date', axis=1)


## CREAR DIFERENCIA DE VARIABLES
def crear_diferencias_y_eliminar(df_train, df_test, pares_columnas):
    for col1, col2 in pares_columnas:
        # Crear la nueva columna de diferencia para el conjunto de entrenamiento
        diff_col_name = f'{col1}_menos_{col2}'
        df_train[diff_col_name] = df_train[col1] - df_train[col2]
        
        # Crear la nueva columna de diferencia para el conjunto de prueba
        df_test[diff_col_name] = df_test[col1] - df_test[col2]
        
        # Eliminar las columnas originales
        df_train = df_train.drop([col1, col2], axis=1)
        df_test = df_test.drop([col1, col2], axis=1)
    
    return df_train, df_test

diff_variables = [('Pressure9am', 'Pressure3pm'), ('WindSpeed9am', 'WindSpeed3pm'), ('MaxTemp', 'MinTemp'), 
                  ('Temp3pm', 'Temp9am'), ('Humidity9am', 'Humidity3pm')]


## ESTANDARIZAMOS 
scaler = StandardScaler() # Creamos el objeto scaler
df_scaled_train = scaler.fit_transform(df_train) 
df_scaled_test = scaler.transform(df_test) 

# Los convertimos a DataFrame porque si no son objetos de numpy
df_scaled_train = pd.DataFrame(df_scaled_train, columns=df_train.columns)
df_scaled_test = pd.DataFrame(df_scaled_test, columns=df_test.columns)