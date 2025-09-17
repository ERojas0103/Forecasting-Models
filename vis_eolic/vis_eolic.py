# -----------------------------------------------------------------------------
# SCRIPT DE VISUALIZACIÓN DE DATOS DE AEROGENERADOR (V3)
# -----------------------------------------------------------------------------
# Este script genera una visualización con tipos de gráficos optimizados
# para cada variable: gráficos de línea para potencia/velocidad y un
# diagrama de dispersión para la dirección del viento.
# -----------------------------------------------------------------------------

# --- Bloque 1: Importación de Librerías ---
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Bloque 2: Carga y Preparación de los Datos ---

# Definición del nombre del archivo de entrada.
file_name = 'T1.csv'

# Verificación de la existencia del archivo.
if not os.path.exists(file_name):
    raise FileNotFoundError(f"El archivo '{file_name}' no se encontró en el directorio actual.")
else:
    # Carga del archivo CSV.
    df = pd.read_csv(file_name)

    # Conversión de la columna 'Date/Time' a formato datetime y establecimiento como índice.
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
    df.set_index('Date/Time', inplace=True)

    # --- Bloque 3: Generación de la Visualización ---

    print("Datos cargados. Iniciando la generación del gráfico optimizado...")

    # Lista de las columnas a ser incluidas en la visualización.
    columnas_a_graficar = [
        'LV ActivePower (kW)',
        'Wind Speed (m/s)',
        'Wind Direction (°)'
    ]

    # Diccionarios para la personalización de cada subplot.
    titulos = {
        'LV ActivePower (kW)': 'Potencia Activa Generada',
        'Wind Speed (m/s)': 'Velocidad del Viento',
        'Wind Direction (°)': 'Distribución de la Dirección del Viento'
    }
    etiquetas_y = {
        'LV ActivePower (kW)': '(kW)',
        'Wind Speed (m/s)': '(m/s)',
        'Wind Direction (°)': 'Grados(°)'
    }

    # Creación de la figura y los ejes con el tamaño especificado.
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 7), sharex=True)

    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Bucle para generar cada uno de los subplots.
    for i, col in enumerate(columnas_a_graficar):
        ax = axes[i]

        # --- Lógica para asignar el tipo de gráfico adecuado ---
        if col == 'Wind Direction (°)':
            # Para la Dirección del Viento: un scatterplot es ideal.
            ax.plot(df.index, df[col], marker='.', markersize=2, linestyle='None', color=colores[i], alpha=0.5)
        else:
            # Para Potencia y Velocidad: un gráfico de línea es el más claro.
            ax.plot(df.index, df[col], color=colores[i], linewidth=0.9)

        # Asignación de títulos y etiquetas personalizados.
        ax.set_title(titulos[col], fontsize=20)
        ax.set_ylabel(etiquetas_y[col], fontsize=20)

        # Ajuste del tamaño de fuente para los números de los ejes.
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Etiquetado del eje X.
    axes[-1].set_xlabel('Fecha', fontsize=20)

    # Ajuste automático del espaciado.
    plt.tight_layout()

    # --- Bloque 4: Exportación del Gráfico ---

    output_filename = 'visualizacion_optimizada.svg'
    plt.savefig(output_filename, format='svg')

    print(f"Proceso finalizado. El gráfico ha sido guardado como '{output_filename}'.")