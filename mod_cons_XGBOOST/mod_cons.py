import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.dates as mdates

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = 'PotenciaActiva.csv'
COLUMNA_OBJETIVO = 'Potencia Total Med'
N_PASOS_PASADOS = 24
PORCENTAJE_ENTRENAMIENTO = 0.8


def cargar_y_preparar_datos(filepath, columna_objetivo):
    """Carga y prepara la serie temporal desde el archivo CSV."""
    print("Cargando y preparando datos...")
    df = pd.read_csv(filepath, sep=';')
    df['datetime'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df_limpio = df[[columna_objetivo]].copy()
    df_limpio[columna_objetivo] = pd.to_numeric(df_limpio[columna_objetivo], errors='coerce')
    df_limpio.dropna(inplace=True)
    return df_limpio


def crear_caracteristicas_avanzadas(df, columna_objetivo, n_pasados):
    """Crea un dataset con lags y características de tiempo."""
    print("Creando características avanzadas (lags + tiempo)...")
    # 1. Crear características de tiempo
    df['hora'] = df.index.hour
    df['dia_semana'] = df.index.dayofweek  # Lunes=0, Domingo=6
    df['dia_mes'] = df.index.day
    df['mes'] = df.index.month

    # 2. Crear características de lags (valores pasados)
    for i in range(1, n_pasados + 1):
        df[f'lag_{i}'] = df[columna_objetivo].shift(i)

    df.dropna(inplace=True)

    # 3. Separar en X (características) y y (objetivo)
    caracteristicas = [col for col in df.columns if col != columna_objetivo]
    X = df[caracteristicas]
    y = df[columna_objetivo]

    return X, y


def main():
    """Flujo principal para entrenar y evaluar el modelo avanzado."""
    # 1. Cargar datos
    df_datos = cargar_y_preparar_datos(NOMBRE_ARCHIVO, COLUMNA_OBJETIVO)

    # 2. Ingeniería de Características
    X, y = crear_caracteristicas_avanzadas(df_datos, COLUMNA_OBJETIVO, N_PASOS_PASADOS)

    # 3. Dividir datos cronológicamente
    punto_division = int(len(X) * PORCENTAJE_ENTRENAMIENTO)
    X_train, X_verification = X.iloc[:punto_division], X.iloc[punto_division:]
    y_train, y_verification = y.iloc[:punto_division], y.iloc[punto_division:]

    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_verification)} para verificación.")

    # 4. Construir y entrenar el modelo XGBoost
    print("\nConstruyendo y entrenando el modelo XGBoost...")
    modelo = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=50  # Se detendrá si el error no mejora en 50 rondas
    )

    modelo.fit(
        X_train, y_train,
        eval_set=[(X_verification, y_verification)],
        verbose=False
    )

    # 5. Realizar predicciones
    print("Realizando predicciones...")
    predicciones = modelo.predict(X_verification)

    # 6. Evaluar el modelo
    mae = mean_absolute_error(y_verification, predicciones)
    rmse = np.sqrt(mean_squared_error(y_verification, predicciones))
    print("\n--- Evaluación del Modelo XGBoost Avanzado ---")
    print(f"Error Absoluto Medio (MAE): {mae:.2f} W")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} W")

    # 7. Visualizar los resultados
    print("Generando gráfico de resultados...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- CAMBIOS APLICADOS ---
    # Establecemos el tamaño de fuente global para la figura
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(15, 8))

    # Se eliminó la línea plt.title()

    fechas_verification = y_verification.index

    plt.scatter(fechas_verification, y_verification, color='blue', label='Valor Real', alpha=0.4, s=30)
    plt.scatter(fechas_verification, predicciones, color='orange', label='Predicción del Modelo', alpha=0.4, s=30)

    # Ya no se necesita especificar fontsize porque se estableció globalmente
    plt.xlabel('Fecha')
    plt.ylabel('Potencia Total Media (W)')
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
    plt.gcf().autofmt_xdate()

    nombre_grafico_salida = 'prediccion_xgboost_avanzado.svg'
    plt.savefig(nombre_grafico_salida, format='svg', bbox_inches='tight')
    print(f"Gráfico guardado como '{nombre_grafico_salida}'")
    plt.show()


if __name__ == '__main__':
    main()