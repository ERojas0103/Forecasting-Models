import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor # <--- Cambio: Importamos el modelo de ensamble
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# --- 1. Carga y Preparación de Datos ---

print("Iniciando el proceso de predicción con Gradient Boosting...")

# Define el nombre del archivo de datos
file_path = 'pv_data.xlsx'

# Comprueba si el archivo existe
if not os.path.exists(file_path):
    print(f"Error: El archivo '{file_path}' no se encontró en el directorio.")
    exit()

print(f"Cargando datos desde '{file_path}'...")
try:
    df = pd.read_excel(file_path, skiprows=[1], na_values='n/a')
except Exception as e:
    print(f"Error al leer el archivo Excel: {e}")
    exit()

# Renombra las columnas para un acceso más fácil
df.columns = [
    'Timestamp',
    'PV_Production_Wh',
    'Irradiation_Wm2',
    'Ambient_Temp_C',
    'Module_Temp_C'
]

# Convierte la columna 'Timestamp' a formato de fecha y hora
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M')

# Establece 'Timestamp' como el índice del DataFrame
df.set_index('Timestamp', inplace=True)

# --- 2. Filtrado y Limpieza de Datos ---

print("Filtrando los datos para los primeros 5 meses...")
df_filtered = df[df.index.month <= 5]

print(f"Datos originales: {df.shape[0]} filas. Datos filtrados (5 meses): {df_filtered.shape[0]} filas.")

# Imputación de datos
print("Realizando imputación de datos nulos...")
df_clean = df_filtered.interpolate(method='linear')

if df_clean.isnull().sum().sum() > 0:
    df_clean.fillna(method='ffill', inplace=True)
    df_clean.fillna(method='bfill', inplace=True)

print("Limpieza de datos completada.")

# --- 3. Preparación para el Modelo de Machine Learning ---

features = ['Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']
target = 'PV_Production_Wh'

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Datos de entrenamiento: {X_train.shape[0]} muestras.")
print(f"Datos de prueba: {X_test.shape[0]} muestras.")

# Aunque los modelos basados en árboles no son tan sensibles a la escala,
# no es mala práctica mantener este paso.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Entrenamiento del Modelo de Ensamble (Gradient Boosting) ---

print("Entrenando el modelo de Ensamble (Gradient Boosting Regressor)...")
# Inicializa el modelo de Gradient Boosting
gbr_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    loss='squared_error'
)

# Entrena el modelo
gbr_model.fit(X_train_scaled, y_train)
print("Modelo entrenado exitosamente.")

# --- 5. Realización de Predicciones y Evaluación ---

print("Realizando predicciones en el conjunto de prueba...")
y_pred = gbr_model.predict(X_test_scaled)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Métricas de Rendimiento del Modelo (Gradient Boosting) ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")
print("----------------------------------------------------------\n")

# --- 6. Visualización de Resultados ---

print("Generando gráfico de comparación...")
plt.style.use('seaborn-v0_8-whitegrid')

# --- CAMBIOS APLICADOS ---
# Establecemos el tamaño de fuente global para la figura
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(figsize=(18, 8))

# Grafica los datos reales
ax.plot(y_test.index, y_test.values, label='Producción Real', color='dodgerblue', alpha=0.8)

# Grafica los datos predichos
ax.plot(y_test.index, y_pred, label='Producción Predicha (Gradient Boosting)', color='purple', linestyle='--', alpha=0.9)

# Configuración del gráfico
# Se eliminó la línea ax.set_title()

# Ya no se necesita especificar fontsize porque se estableció globalmente
ax.set_xlabel('Fecha y Hora')
ax.set_ylabel('Producción Fotovoltaica (Wh)')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Guarda el gráfico en formato SVG con un nuevo nombre
output_filename = 'comparacion_produccion_fv_gbr.svg'
plt.savefig(output_filename, format='svg')

print(f"¡Proceso completado! El gráfico se ha guardado como '{output_filename}'.")