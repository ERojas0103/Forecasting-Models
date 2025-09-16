import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = 'PotenciaActiva.csv'


def cargar_y_preparar_datos(filepath):
    """Carga y prepara los datos desde el archivo CSV."""
    print(f"Cargando y preparando datos desde '{filepath}'...")
    df = pd.read_csv(filepath, sep=';')

    df['datetime'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)

    cols_potencia = [col for col in df.columns if 'Potencia' in col]
    for col in cols_potencia:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=cols_potencia, inplace=True)
    print("Datos cargados correctamente.")
    return df


def graficar_perfil_diario_percentiles(df, fecha, nombre_archivo):
    """
    Genera un 'fan chart' con bandas de percentiles estimados para un día específico.
    """

    datos_dia = df[df.index.date == fecha.date()].copy()

    if datos_dia.empty:
        print(f"No se encontraron datos para la fecha {fecha.date()}. No se generará la gráfica.")
        return

    print(f"Generando 'fan chart' por percentiles para el día {fecha.date()}...")

    # --- CAMBIO: Tasa de remuestreo ajustada a 1 minuto ---
    tasa_remuestreo = '1T'
    datos_resampled = datos_dia.resample(tasa_remuestreo).agg({
        'Potencia Total Med': 'mean',
        'Potencia Total Min': 'min',
        'Potencia Total Max': 'max'
    })
    datos_resampled.dropna(inplace=True)

    # Creación de Bandas Estimadas alrededor de la Media
    p_mean = datos_resampled['Potencia Total Med']
    p_min = datos_resampled['Potencia Total Min']
    p_max = datos_resampled['Potencia Total Max']

    # Interpolamos para crear las bandas
    p25 = p_min + 0.5 * (p_mean - p_min)
    p75 = p_mean + 0.5 * (p_max - p_mean)
    p10 = p_min + 0.2 * (p_mean - p_min)
    p90 = p_mean + 0.8 * (p_max - p_mean)

    # --- GRAFICACIÓN POR CAPAS ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 7))

    # --- CAMBIO: Nombres correctos para las bandas ---
    ax.fill_between(datos_resampled.index, p10, p90, color='deepskyblue', alpha=0.2, label='Rango Percentil 10-90')
    ax.fill_between(datos_resampled.index, p25, p75, color='deepskyblue', alpha=0.3, label='Rango Percentil 25-75')
    ax.plot(datos_resampled.index, p_mean, color='dodgerblue', label='Potencia Media', linewidth=2.5)

    # Configuración de estilo
    ax.set_ylabel('Potencia (W)', fontsize=18)
    ax.set_xlabel('Hora del Día', fontsize=18)
    ax.legend(fontsize=18)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(rotation=45, ha='right')

    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(nombre_archivo, format='svg', bbox_inches='tight')
    print(f"Gráfica guardada como '{nombre_archivo}'")
    plt.close()


def main():
    """Flujo principal del análisis."""
    df = cargar_y_preparar_datos(NOMBRE_ARCHIVO)

    consumo_diario = df['Potencia Total Med'].resample('D').sum()
    registros_diarios = df['Potencia Total Med'].resample('D').count()
    consumo_diario_filtrado = consumo_diario[registros_diarios > 60 * 12]

    fecha_mayor_actividad = consumo_diario_filtrado.idxmax()
    fecha_menor_actividad = consumo_diario_filtrado.idxmin()

    print("\n--- Análisis de Actividad ---")
    print(f"Día de MAYOR actividad identificado: {fecha_mayor_actividad.date()}")
    print(f"Día de MENOR actividad identificado: {fecha_menor_actividad.date()}")
    print("-----------------------------\n")

    # Generar la gráfica para el día de mayor actividad
    graficar_perfil_diario_percentiles(
        df=df,
        fecha=fecha_mayor_actividad,
        nombre_archivo='perfil_mayor_actividad_percentiles.svg'
    )

    # Generar la gráfica para el día de menor actividad
    graficar_perfil_diario_percentiles(
        df=df,
        fecha=fecha_menor_actividad,
        nombre_archivo='perfil_menor_actividad_percentiles.svg'
    )


if __name__ == '__main__':
    main()