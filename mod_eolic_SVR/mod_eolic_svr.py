# -----------------------------------------------------------------------------
# SCRIPT PARA MODELO SVR DE PRONÓSTICO DE PRODUCCIÓN EÓLICA
# -----------------------------------------------------------------------------
# Se utiliza un modelo de Support Vector Regression (SVR) para predecir la
# producción de un aerogenerador, manteniendo la estructura de procesamiento
# de datos y visualización del modelo anterior.
# -----------------------------------------------------------------------------

# --- Bloque 1: Importación de Librerías ---
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR # <--- Importación del modelo SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# --- Bloque 2: Carga y Preparación de Datos ---

print("Iniciando el proceso de predicción eólica con SVR...")
file_name = 'T1.csv'
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Error: El archivo '{file_name}' no se encontró.")

# Carga de datos
df = pd.read_csv(file_name)

# Conversión de fecha y establecimiento como índice
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
df.set_index('Date/Time', inplace=True)

# Limpieza de datos
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

# --- Bloque 3: Entrenamiento y Predicción del Modelo SVR ---

# *** MODELO MODIFICADO ***
# Definición del modelo SVR con hiperparámetros estándar (kernel RBF).
svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')

print("Entrenando el modelo SVR y realizando predicciones...")
# NOTA: El entrenamiento de SVR puede ser más lento que el de la ANN en este dataset.

# Medición del tiempo de entrenamiento y predicción
start_time = time.time()
svr_model.fit(X_train_scaled, y_train)
y_pred = svr_model.predict(X_test_scaled)
end_time = time.time()
processing_time = end_time - start_time

print(f"✅ Modelo entrenado y predicciones realizadas.")
print(f"⏱️ Tiempo total de procesamiento: {processing_time:.2f} segundos.\n")

# --- Bloque 4: Evaluación del Modelo ---

# Cálculo de métricas
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Métricas de Rendimiento del Modelo (SVR) ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} kW")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} kW")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")
print("----------------------------------------------\n")

# --- Bloque 5: Visualización de Resultados ---

print("Generando la gráfica de resultados...")
fig, ax = plt.subplots(figsize=(20, 7))

# Gráfico de la producción real vs. la pronosticada
ax.plot(y_test.index, y_test.values, label='Producción Real', color='#1f77b4', linewidth=1.5)
ax.plot(y_test.index, y_pred, label='Producción Pronosticada (SVR)', color='#ff7f0e', linestyle='--', linewidth=1.5) # Etiqueta actualizada

# Aplicación de formatos
ax.set_ylabel('Producción Eólica (kW)', fontsize=20)
ax.set_xlabel('Fecha', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=16, loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# --- Bloque 6: Guardado de la Figura ---
output_filename = 'pronostico_svr_produccion_eolica.svg' # Nombre de archivo actualizado
plt.savefig(output_filename, format='svg')
print(f"¡Proceso finalizado! La gráfica ha sido guardada como '{output_filename}'.")