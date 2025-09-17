# -----------------------------------------------------------------------------
# SCRIPT PARA MODELO XGBOOST AVANZADO (CON FEATURE ENGINEERING)
# -----------------------------------------------------------------------------
# Versión corregida para compatibilidad con la librería del usuario.
# El parámetro 'early_stopping_rounds' se mueve al constructor del modelo.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import time
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURACIÓN ---
NOMBRE_ARCHIVO = 'T1.csv'
COLUMNA_OBJETIVO = 'LV ActivePower (kW)'
N_PASOS_PASADOS = 12
PORCENTAJE_ENTRENAMIENTO = 0.8


def cargar_y_preparar_datos(filepath, columna_objetivo):
    """Carga y prepara la serie temporal desde el archivo CSV."""
    print("Cargando y preparando datos...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: El archivo '{filepath}' no se encontró.")

    df = pd.read_csv(filepath)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
    df.set_index('Date/Time', inplace=True)

    df_limpio = df[[columna_objetivo]].copy()
    df_limpio.dropna(inplace=True)
    return df_limpio


def crear_caracteristicas(df, columna_objetivo, n_pasados):
    """Crea un dataset con lags (valores pasados) y características de tiempo."""
    print("Creando características avanzadas (lags + tiempo)...")

    df['hora'] = df.index.hour
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes'] = df.index.day
    df['mes'] = df.index.month

    for i in range(1, n_pasados + 1):
        df[f'lag_{i}'] = df[columna_objetivo].shift(i)

    df.dropna(inplace=True)

    caracteristicas = [col for col in df.columns if col != columna_objetivo]
    X = df[caracteristicas]
    y = df[columna_objetivo]

    return X, y


def main():
    """Flujo principal para entrenar, evaluar y visualizar el modelo."""
    df_datos = cargar_y_preparar_datos(NOMBRE_ARCHIVO, COLUMNA_OBJETIVO)
    X, y = crear_caracteristicas(df_datos, COLUMNA_OBJETIVO, N_PASOS_PASADOS)

    punto_division = int(len(X) * PORCENTAJE_ENTRENAMIENTO)
    X_train, X_test = X.iloc[:punto_division], X.iloc[punto_division:]
    y_train, y_test = y.iloc[:punto_division], y.iloc[punto_division:]
    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

    # *** LÍNEA MODIFICADA ***
    # Se mueve 'early_stopping_rounds' a la definición del modelo.
    modelo = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50  # <--- PARÁMETRO MOVIDO AQUÍ
    )

    print("\nEntrenando el modelo XGBoost y realizando predicciones...")
    start_time = time.time()

    # *** LÍNEA MODIFICADA ***
    # Se elimina 'early_stopping_rounds' de la llamada a .fit().
    modelo.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = modelo.predict(X_test)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"✅ Modelo entrenado y predicciones realizadas.")
    print(f"⏱️ Tiempo total de procesamiento: {processing_time:.2f} segundos.\n")

    # Evaluación
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("--- Métricas de Rendimiento del Modelo (XGBoost Avanzado) ---")
    print(f"Error Absoluto Medio (MAE): {mae:.2f} kW")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} kW")
    print(f"Coeficiente de Determinación (R²): {r2:.2f}")
    print("----------------------------------------------------------\n")

    # Visualización
    print("Generando la gráfica de resultados...")
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.plot(y_test.index, y_test.values, label='Producción Real', color='#1f77b4', linewidth=1.5)
    ax.plot(y_test.index, y_pred, label='Producción Pronosticada (XGBoost)', color='#ff7f0e', linestyle='--',
            linewidth=1.5)
    ax.set_ylabel('Producción Eólica (kW)', fontsize=20)
    ax.set_xlabel('Fecha', fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Guardado
    output_filename = 'pronostico_xgboost_avanzado_eolica.svg'
    plt.savefig(output_filename, format='svg')
    print(f"¡Proceso finalizado! La gráfica ha sido guardada como '{output_filename}'.")


if __name__ == '__main__':
    main()