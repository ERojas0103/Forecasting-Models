import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.dates as mdates

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = 'PotenciaActiva.csv'
COLUMNA_OBJETIVO = 'Potencia Total Med'
N_PASOS_PASADOS = 24
PORCENTAJE_ENTRENAMIENTO = 0.8


def cargar_y_preparar_datos(filepath, columna_objetivo):
    """Carga y prepara la serie temporal desde el archivo CSV."""
    print(f"Cargando datos desde '{filepath}'...")
    df = pd.read_csv(filepath, sep=';')

    df['datetime'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)

    serie_temporal = df[[columna_objetivo]].copy()
    serie_temporal[columna_objetivo] = pd.to_numeric(serie_temporal[columna_objetivo], errors='coerce')
    serie_temporal.dropna(inplace=True)

    print(f"Datos cargados. Se encontraron {len(serie_temporal)} registros.")
    return serie_temporal


def crear_dataset_supervisado(datos, n_pasados=1):
    """Transforma una serie temporal en un dataset para aprendizaje supervisado."""
    X, y = [], []
    for i in range(len(datos) - n_pasados):
        X.append(datos[i:(i + n_pasados), 0])
        y.append(datos[i + n_pasados, 0])
    return np.array(X), np.array(y)


def main():
    """Flujo principal para entrenar, evaluar y visualizar el modelo de predicción."""
    # 1. Cargar y preparar datos
    df_completo = cargar_y_preparar_datos(NOMBRE_ARCHIVO, COLUMNA_OBJETIVO)
    datos_completos = df_completo.values

    # 2. Escalar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_escalados = scaler.fit_transform(datos_completos)

    # 3. Crear el dataset supervisado
    X, y = crear_dataset_supervisado(datos_escalados, N_PASOS_PASADOS)

    # --- CAMBIO: Conservar las fechas para el eje X ---
    # El índice de fechas debe alinearse con las etiquetas 'y'
    fechas = df_completo.index[N_PASOS_PASADOS:]

    # 4. Dividir datos y fechas en 80% entrenamiento y 20% verificación
    punto_division = int(len(X) * PORCENTAJE_ENTRENAMIENTO)
    X_train, X_verification = X[:punto_division], X[punto_division:]
    y_train, y_verification = y[:punto_division], y[punto_division:]
    fechas_train, fechas_verification = fechas[:punto_division], fechas[punto_division:]

    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_verification)} para verificación.")

    # 5. Construir y entrenar el modelo
    print("\nConstruyendo y entrenando el modelo con Scikit-learn...")
    modelo = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        verbose=True,
        early_stopping=True
    )

    modelo.fit(X_train, y_train)

    # 6. Realizar predicciones
    print("Realizando predicciones en el conjunto de verificación...")
    predicciones_escaladas = modelo.predict(X_verification).reshape(-1, 1)

    predicciones = scaler.inverse_transform(predicciones_escaladas)
    y_verification_reales = scaler.inverse_transform(y_verification.reshape(-1, 1))

    # 7. Evaluar el modelo
    mae = mean_absolute_error(y_verification_reales, predicciones)
    rmse = np.sqrt(mean_squared_error(y_verification_reales, predicciones))
    print("\n--- Evaluación del Modelo ---")
    print(f"Error Absoluto Medio (MAE): {mae:.2f} W")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} W")

    # 8. Visualizar los resultados con fechas en el eje X
    print("Generando gráfico de resultados con el nuevo estilo...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- CAMBIOS APLICADOS ---
    # Establecemos el tamaño de fuente global para la figura
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(15, 8))

    # Se eliminó la línea plt.title()

    # --- CAMBIOS DE ESTILO AQUÍ ---
    # Puntos azules para los valores reales con opacidad del 40%
    plt.scatter(fechas_verification, y_verification_reales, color='blue', label='Valor Real', alpha=0.4, s=30)
    # Puntos naranjas para las predicciones con opacidad del 40%
    plt.scatter(fechas_verification, predicciones, color='orange', label='Predicción del Modelo', alpha=0.4, s=30)

    # Ya no se necesita especificar fontsize porque se estableció globalmente
    plt.xlabel('Fecha')
    plt.ylabel('Potencia Total Media (W)')
    plt.legend()

    # Formatear el eje de fechas para mejor legibilidad
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
    plt.gcf().autofmt_xdate()  # Rotar fechas automáticamente

    nombre_grafico_salida = 'prediccion_fechas_reales.svg'
    plt.savefig(nombre_grafico_salida, format='svg', bbox_inches='tight')
    print(f"Gráfico guardado como '{nombre_grafico_salida}'")
    plt.show()


if __name__ == '__main__':
    main()