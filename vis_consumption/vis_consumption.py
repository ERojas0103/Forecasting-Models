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


def graficar_perfil_diario_lineplot(df, fecha, nombre_archivo):
    """
    Genera un gráfico de líneas (lineplot) mostrando la potencia media, mínima y máxima.
    """

    datos_dia = df[df.index.date == fecha.date()].copy()

    if datos_dia.empty:
        print(f"No se encontraron datos para la fecha {fecha.date()}. No se generará la gráfica.")
        return

    print(f"Generando lineplot para el día {fecha.date()}...")

    # Remuestreo a 1 minuto para regularizar el índice de tiempo y evitar artefactos
    tasa_remuestreo = '1T'
    datos_resampled = datos_dia.resample(tasa_remuestreo).agg({
        'Potencia Total Med': 'mean',
        'Potencia Total Min': 'min',
        'Potencia Total Max': 'max'
    })
    datos_resampled.dropna(inplace=True)

    # --- GRAFICACIÓN CON LÍNEAS ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 7))

    # Línea principal para la Potencia Media
    ax.plot(datos_resampled.index, datos_resampled['Potencia Total Med'],
            color='dodgerblue',
            label='Potencia Media',
            linewidth=2.5)

    # Líneas para el rango de variabilidad (Mínima y Máxima)
    ax.plot(datos_resampled.index, datos_resampled['Potencia Total Min'],
            color='skyblue',
            label='Potencia Mínima/Máxima',
            linewidth=1.5,
            linestyle='--')

    ax.plot(datos_resampled.index, datos_resampled['Potencia Total Max'],
            color='skyblue',
            linewidth=1.5,
            linestyle='--')

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
    graficar_perfil_diario_lineplot(
        df=df,
        fecha=fecha_mayor_actividad,
        nombre_archivo='perfil_mayor_actividad_lineplot.svg'
    )

    # Generar la gráfica para el día de menor actividad
    graficar_perfil_diario_lineplot(
        df=df,
        fecha=fecha_menor_actividad,
        nombre_archivo='perfil_menor_actividad_lineplot.svg'
    )


if __name__ == '__main__':
    main()