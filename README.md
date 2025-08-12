# Predicción de lluvia en Australia 🌧️🇦🇺

Este proyecto es el **Trabajo Práctico Integrador** de la materia **Aprendizaje Automático 1** (Tecnicatura Universitaria en Inteligencia Artificial – UNR).  
El objetivo fue desarrollar un sistema completo de **predicción de lluvia en Australia** a partir del dataset **weatherAUS**, incluyendo **análisis exploratorio, modelado, evaluación y puesta en producción con Streamlit**.

---

## 📌 Objetivos del proyecto
- Familiarizarse con **scikit-learn**, **TensorFlow** y **Streamlit**.
- Preprocesar y analizar datos meteorológicos.
- Entrenar y evaluar modelos de regresión y clasificación para predecir:
  - **RainTomorrow** (lloverá mañana: Sí/No).
  - **RainfallTomorrow** (cantidad de lluvia en mm).
- Implementar explicabilidad de modelos con **SHAP**.
- Comparar diferentes enfoques y optimizar hiperparámetros.
- Desplegar el mejor modelo en una **aplicación web interactiva**.

---

## 🔍 Desarrollo

### 1. Análisis Exploratorio de Datos (EDA)
- Inspección inicial del dataset.
- Análisis de valores faltantes y decisiones de imputación.
- Visualizaciones: histogramas, boxplots, scatterplots.
- Matriz de correlación para selección de variables relevantes.
- Balance de clases en **RainTomorrow**.

### 2. Preprocesamiento
- Filtrado y selección de ubicaciones relevantes (Adelaide, Canberra, Cobar, Dartmoor, Melbourne, MelbourneAirport, MountGambier, Sydney, SydneyAirport).
- Conversión de fechas y extracción de estación del año.
- Agrupación de direcciones de viento.
- Codificación de variables categóricas (One-Hot Encoding).
- Estandarización de variables numéricas.
- Creación de nuevas variables (diferencias de temperatura, presión, humedad).
- Separación Train/Test con validación cruzada.

### 3. Modelado
#### 🔹 Regresión
- **LinearRegression**.
- Métodos con gradiente descendente.
- Regularización: **Lasso, Ridge, Elastic Net**.
- Métricas: R², MAE, RMSE, MAPE.

#### 🔹 Clasificación
- **LogisticRegression**.
- Curvas ROC y elección de umbral.
- Matrices de confusión y análisis de falsos positivos/negativos.
- Métricas: Accuracy, Precision, Recall, F1 Score.

#### 🔹 Modelos base y redes neuronales
- Implementación de modelos baseline.
- Redes neuronales con TensorFlow y comparación de resultados.
- Optimización de hiperparámetros con Grid Search y Random Search.

### 4. Explicabilidad
- **SHAP** para interpretabilidad local y global.
- Identificación de variables más influyentes en cada modelo.

### 5. Comparación y selección de modelos
- Selección del mejor modelo de regresión y de clasificación.
- Justificación de la elección para puesta en producción.

---

## 🌐 Aplicación en Streamlit
Se desarrolló una app en **Streamlit** que:
- Recibe datos meteorológicos desde un formulario interactivo.
- Procesa la entrada replicando el pipeline de entrenamiento.
- Devuelve:
  - Predicción de si lloverá mañana.
  - Predicción de la cantidad estimada de lluvia.

---

## 🚀 Ejecución local
### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/AA1-TUIA-Apellido1-Apellido2.git
cd AA1-TUIA-Apellido1-Apellido2

### 2. Instalar dependencias
```bash
pip install -r requirements.txt

### 3. Ejecutar la aplicación
```bash
streamlit run app.py

---

## 📊 Principales resultados
- Se logró un **modelo de clasificación** con *alta precisión y buen recall*, equilibrando falsos positivos y negativos.
- El **modelo de regresión** seleccionado presentó un error medio bajo y buen ajuste (R² alto).
- La interpretabilidad mostró que variables como **Humidity3pm, Rainfall, Cloud3pm y WindGustSpeed** son determinantes en la predicción.

---

## ✒️ Autores
- Masciangelo, Lucía
- Texier, Julieta
