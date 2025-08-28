import pandas as pd
import numpy as np
import time # <--- Añadido para medir el tiempo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# --- Bloques 1, 2 y 3 (Carga, Limpieza y Preparación) se mantienen igual ---

print("Iniciando el proceso de predicción con SVR...")

file_path = 'pv_data.xlsx'
if not os.path.exists(file_path):
    print(f"Error: El archivo '{file_path}' no se encontró.")
    exit()

df = pd.read_excel(file_path, skiprows=[1], na_values='n/a')
df.columns = ['Timestamp', 'PV_Production_Wh', 'Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M')
df.set_index('Timestamp', inplace=True)
df_filtered = df[df.index.month <= 5]
df_clean = df_filtered.interpolate(method='linear')
if df_clean.isnull().sum().sum() > 0:
    df_clean.fillna(method='ffill', inplace=True)
    df_clean.fillna(method='bfill', inplace=True)

features = ['Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']
target = 'PV_Production_Wh'
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Entrenamiento del Modelo SVM ---

print("Entrenando el modelo de Máquinas de Vectores de Soporte (SVR)...")
svm_model = SVR(kernel='rbf', C=100, epsilon=0.1)

# --- 5. Realización de Predicciones y Medición de Tiempo ---

start_time = time.time() # <--- INICIA EL CRONÓMETRO

svm_model.fit(X_train_scaled, y_train) # Entrena
y_pred = svm_model.predict(X_test_scaled) # Predice

end_time = time.time() # <--- DETIENE EL CRONÓMETRO

# Calcula e imprime el tiempo transcurrido
processing_time = end_time - start_time
print(f"\n✅ Modelo entrenado y predicciones realizadas.")
print(f"⏱️ Tiempo total de procesamiento (entrenamiento + predicción): {processing_time:.4f} segundos.\n")


# --- Evaluación y Visualización (se mantienen igual) ---

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("--- Métricas de Rendimiento del Modelo (SVR) ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")
print("----------------------------------------------\n")

print("Generando gráfico...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(y_test.index, y_test.values, label='Producción Real', color='dodgerblue', alpha=0.8)
ax.plot(y_test.index, y_pred, label='Producción Predicha (SVR)', color='green', linestyle='--', alpha=0.9)
ax.set_title('Comparación: Real vs. Predicha (SVR)', fontsize=16)
ax.set_xlabel('Fecha y Hora', fontsize=12)
ax.set_ylabel('Producción Fotovoltaica (Wh)', fontsize=12)
ax.legend()
plt.tight_layout()
output_filename = 'comparacion_produccion_fv_svr.svg'
plt.savefig(output_filename, format='svg')
print(f"Gráfico guardado como '{output_filename}'.")