import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import os

print("Iniciando el entrenamiento del modelo inicial...")

# --- 1. Carga y Preparación de Datos ---
file_path = 'pv_data.xlsx'
if not os.path.exists(file_path):
    print(f"Error: El archivo de datos '{file_path}' no se encontró.")
    exit()

print("Cargando y limpiando los datos base...")
df = pd.read_excel(file_path, skiprows=[1], na_values='n/a')
df.columns = ['Timestamp', 'PV_Production_Wh', 'Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M')
df.set_index('Timestamp', inplace=True)

# --- 2. Filtrado y Limpieza ---
df_filtered = df[df.index.month <= 5]
df_clean = df_filtered.interpolate(method='linear')
if df_clean.isnull().sum().sum() > 0:
    df_clean.fillna(method='ffill', inplace=True)
    df_clean.fillna(method='bfill', inplace=True)

print(f"Se usarán {df_clean.shape[0]} registros para el entrenamiento inicial.")

# --- 3. Preparación para el Modelo ---
features = ['Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']
target = 'PV_Production_Wh'

X_train = df_clean[features]
y_train = df_clean[target]

# Escala de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- 4. Entrenamiento del Modelo ANN ---
print("Entrenando el modelo de Red Neuronal Artificial (ANN)...")
# Usamos 'adam' como solver y warm_start=True para permitir el reaprendizaje (partial_fit)
ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,       # Aumentamos iteraciones para un buen entrenamiento inicial
    random_state=42,
    warm_start=True,    # Clave para permitir el reaprendizaje
    verbose=False
)

ann_model.fit(X_train_scaled, y_train)

# --- 5. Guardado del Modelo y el Escalador ---
model_filename = 'ann_model.joblib'
scaler_filename = 'scaler.joblib'

joblib.dump(ann_model, model_filename)
joblib.dump(scaler, scaler_filename)

print("\n✅ ¡Entrenamiento inicial completado!")
print(f"Modelo guardado en: {model_filename}")
print(f"Escalador guardado en: {scaler_filename}")