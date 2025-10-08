import pandas as pd
import joblib
import time
import os

# --- Configuración ---
SCALER_PATH = 'scaler.joblib'
INPUT_CSV_PATH = 'sep_14.csv'
MODELS_TO_EVALUATE = {
    'ANN': 'ann_model.joblib',
    'SVR': 'svr_model.joblib',
    'GBR': 'gbr_model.joblib'
}


def process_input_csv(file_path):
    """Lee y procesa el archivo CSV de entrada, garantizando el orden de las columnas."""
    try:
        df = pd.read_csv(file_path, skiprows=[1])
        column_map = {
            'Fecha y hora': 'Timestamp',
            'Irradiación | Sensor Card / Box (1)': 'Irradiation_Wm2',
            'Temperatura ambiente | Sensor Card / Box (1)': 'Ambient_Temp_C',
            'Temperatura de módulo | Sensor Card / Box (1)': 'Module_Temp_C'
        }
        df.rename(columns=column_map, inplace=True)

        # Se define como una lista para garantizar el orden de las columnas.
        required_features = ['Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']

        # Se convierte a set solo para la validación.
        if not set(required_features).issubset(df.columns):
            print(f"Error: El CSV debe contener las columnas: {', '.join(required_features)}")
            return None

        # Se retorna el DataFrame con las columnas en el orden correcto.
        return df[required_features]

    except Exception as e:
        print(f"Error al procesar el archivo CSV: {e}")
        return None


# --- Inicio del Script de Evaluación ---

print("Iniciando evaluación de velocidad de predicción para todos los modelos...\n")

# Cargar y preparar los datos de entrada una sola vez
if not os.path.exists(SCALER_PATH) or not os.path.exists(INPUT_CSV_PATH):
    print(f"Error: No se encontraron '{SCALER_PATH}' o '{INPUT_CSV_PATH}'.")
    exit()

scaler = joblib.load(SCALER_PATH)
features_df = process_input_csv(INPUT_CSV_PATH)

if features_df is None:
    exit()

X_scaled = scaler.transform(features_df)
print(f"Datos de entrada ({X_scaled.shape[0]} filas) listos para la predicción.\n")

# Iterar sobre cada modelo para evaluarlo
for model_name, model_path in MODELS_TO_EVALUATE.items():
    print(f"--- EVALUANDO MODELO: {model_name} ---")

    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo del modelo '{model_path}'. Saltando evaluación.")
        continue

    # Cargar el modelo
    model = joblib.load(model_path)

    # Medir el tiempo de predicción
    start_time = time.perf_counter()
    model.predict(X_scaled)
    end_time = time.perf_counter()

    prediction_time_ms = (end_time - start_time) * 1000

    print(f"⏱️ Tiempo de Procesamiento (Predicción) {model_name}: {prediction_time_ms:.4f} milisegundos\n")

print("--- Evaluación completada. ---")
