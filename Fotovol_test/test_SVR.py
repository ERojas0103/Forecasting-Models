import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
import time

print("Iniciando el entrenamiento de todos los modelos...")

# --- 1. Carga y Preparación de Datos ---
file_path = 'pv_data.xlsx'
if not os.path.exists(file_path):
    print(f"Error: El archivo de datos '{file_path}' no se encontró.")
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
X_train = df_clean[features]
y_train = df_clean[target]

# --- 2. Escalado de Datos ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler.joblib')
print("Escalador de datos entrenado y guardado en 'scaler.joblib'.")

# --- 3. Entrenamiento y Medición de Tiempo ---

# Modelo 1: Red Neuronal Artificial (ANN)
print("\nEntrenando modelo ANN...")
start_time_ann = time.time()
ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
    max_iter=500, random_state=42, warm_start=True
)
ann_model.fit(X_train_scaled, y_train)
end_time_ann = time.time()
joblib.dump(ann_model, 'ann_model.joblib')
print(f"Modelo ANN guardado en 'ann_model.joblib'.")
print(f"Tiempo de Entrenamiento ANN: {end_time_ann - start_time_ann:.4f} segundos.")

# Modelo 2: Regresión de Vectores de Soporte (SVR)
print("\nEntrenando modelo SVR...")
start_time_svr = time.time()
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
end_time_svr = time.time()
joblib.dump(svr_model, 'svr_model.joblib')
print(f"Modelo SVR guardado en 'svr_model.joblib'.")
print(f"Tiempo de Entrenamiento SVR: {end_time_svr - start_time_svr:.4f} segundos.")

# Modelo 3: Gradient Boosting
print("\nEntrenando modelo Gradient Boosting...")
start_time_gbr = time.time()
gbr_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
)
gbr_model.fit(X_train_scaled, y_train)
end_time_gbr = time.time()
joblib.dump(gbr_model, 'gbr_model.joblib')
print(f" Modelo GBR guardado en 'gbr_model.joblib'.")
print(f"Tiempo de Entrenamiento GBR: {end_time_gbr - start_time_gbr:.4f} segundos.")

print("\n--- Proceso de entrenamiento completado. ---")
