import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = 'estacionenergetica.csv'
COLUMNA_CONSUMO = 'Demanda_PTS_W'
INTERVALO_REMUESTREO = '2T'  # Remuestreo cada 2 minutos
# --- NUEVO: Fecha de inicio para la gráfica ---
FECHA_INICIO = '2025-08-15'


def graficar_lineplot_filtrado(filepath, columna_objetivo, intervalo, fecha_inicio):
    """
    Carga los datos, filtra a partir de una fecha de inicio y genera un
    lineplot limpio de la serie temporal.
    """
    # --- 1. Carga y Preparación de Datos ---
    print(f"Cargando y preparando datos desde '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filepath}'.")
        return

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)

    if columna_objetivo not in df.columns:
        print(f"Error: La columna '{columna_objetivo}' no se encuentra en el archivo.")
        return

    df[columna_objetivo] = pd.to_numeric(df[columna_objetivo], errors='coerce')
    df.dropna(subset=[columna_objetivo], inplace=True)

    # --- 2. Filtrado de Fechas ---
    print(f"Filtrando datos para mostrar solo a partir del {fecha_inicio}...")
    df_filtrado = df.loc[fecha_inicio:]

    if df_filtrado.empty:
        print(f"No se encontraron datos válidos a partir de la fecha especificada.")
        return

    print(f"Datos filtrados. {len(df_filtrado)} registros válidos encontrados.")

    # --- 3. Remuestreo de la Serie Temporal ---
    serie_remuestreada = df_filtrado[columna_objetivo].resample(intervalo).mean()

    # --- 4. Creación del Gráfico ---
    print("Generando lineplot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 7))

    ax.plot(serie_remuestreada.index, serie_remuestreada.values,
            color='dodgerblue',
            linewidth=2)

    # --- 5. Configuración de Estilo ---
    ax.set_ylabel('Demanda de Potencia Promedio (W)', fontsize=18)
    ax.set_xlabel('Fecha', fontsize=18)
    ax.set_title(f'Historial de Consumo (desde el {pd.to_datetime(fecha_inicio).strftime("%d-%b-%Y")})', fontsize=20,
                 weight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=0)

    # --- 6. Guardado del Archivo ---
    nombre_archivo_salida = 'lineplot_consumo_post_agosto.svg'
    plt.tight_layout()
    plt.savefig(nombre_archivo_salida, format='svg', bbox_inches='tight')
    print(f"Gráfica guardada como '{nombre_archivo_salida}'")
    plt.show()


if __name__ == '__main__':
    graficar_lineplot_filtrado(NOMBRE_ARCHIVO, COLUMNA_CONSUMO, INTERVALO_REMUESTREO, FECHA_INICIO)