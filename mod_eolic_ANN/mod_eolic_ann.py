# -----------------------------------------------------------------------------
# SCRIPT PARA MODELO ANN DE PRONÓSTICO EÓLICO (BASADO EN MODELO DE REFERENCIA)
# -----------------------------------------------------------------------------
# Se adapta la lógica de un modelo funcional para predecir la producción
# de un aerogenerador, asegurando la correcta manipulación de las fechas.
# -----------------------------------------------------------------------------

# --- Bloque 1: Importación de Librerías ---
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# --- Bloque 2: Carga y Preparación de Datos ---

print("Iniciando el proceso de predicción eólica con ANN...")
file_name = 'T1.csv'
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Error: El archivo '{file_name}' no se encontró.")

# Carga de datos
df = pd.read_csv(file_name)

# Conversión de fecha y establecimiento como índice
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
df.set_index('Date/Time', inplace=True)

# Limpieza de datos (eliminamos filas con valores nulos si las hubiera)
df.dropna(inplace=True)
print("Datos cargados y limpiados correctamente.")

# Definición de características (features) y objetivo (target)
features = ['Wind Speed (m/s)', 'Wind Direction (°)']
target = 'LV ActivePower (kW)'
X = df[features]
y = df[target]

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Escalado de las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Datos preparados y divididos.")

# --- Bloque 3: Entrenamiento y Predicción del Modelo ANN ---

# Definición del modelo
ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    verbose=False
)

print("Entrenando el modelo ANN y realizando predicciones...")

# Medición del tiempo de entrenamiento y predicción
start_time = time.time()
ann_model.fit(X_train_scaled, y_train)
y_pred = ann_model.predict(X_test_scaled)
end_time = time.time()
processing_time = end_time - start_time

print(f"✅ Modelo entrenado y predicciones realizadas.")
print(f"⏱️ Tiempo total de procesamiento: {processing_time:.2f} segundos.\n")

# --- Bloque 4: Evaluación del Modelo ---

# Cálculo de métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Métricas de Rendimiento del Modelo (ANN) ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} kW")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} kW")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")
print("----------------------------------------------\n")

# --- Bloque 5: Visualización de Resultados ---

print("Generando la gráfica de resultados...")
fig, ax = plt.subplots(figsize=(20, 7))

# Gráfico de la producción real vs. la pronosticada
ax.plot(y_test.index, y_test.values, label='Producción Real', color='#1f77b4', linewidth=1.5)
ax.plot(y_test.index, y_pred, label='Producción Pronosticada (ANN)', color='#ff7f0e', linestyle='--', linewidth=1.5)

# Aplicación de formatos de fuente y etiquetas
ax.set_ylabel('Producción Eólica (kW)', fontsize=20)
ax.set_xlabel('Fecha', fontsize=20)

# *** LÍNEA MODIFICADA ***
# Se elimina la rotación para que las fechas queden horizontales.
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.legend(fontsize=16, loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# --- Bloque 6: Guardado de la Figura ---
output_filename = 'pronostico_ann_produccion_eolica.svg'
plt.savefig(output_filename, format='svg')
print(f"¡Proceso finalizado! La gráfica ha sido guardada como '{output_filename}'.")