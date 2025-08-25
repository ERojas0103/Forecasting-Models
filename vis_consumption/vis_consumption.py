import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


def generar_visualizacion_total(nombre_archivo_csv: str):
    """
    Genera y guarda una visualización enfocada únicamente en la Potencia Total
    (mínima, media y máxima) a partir del archivo PotenciaActiva.csv.

    Args:
        nombre_archivo_csv (str): La ruta al archivo CSV.
    """
    print(f"Cargando datos desde '{nombre_archivo_csv}'...")

    try:
        # Cargar los datos especificando el separador
        df = pd.read_csv(nombre_archivo_csv, sep=';')

        # --- PREPROCESAMIENTO DE DATOS ---
        df['datetime'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('datetime', inplace=True)

        cols_potencia = [col for col in df.columns if 'Potencia' in col]
        for col in cols_potencia:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=cols_potencia, inplace=True)

        print("Datos cargados. Generando gráfico de Potencia Total...")

        # --- CREACIÓN DE LA VISUALIZACIÓN ---
        sns.set_theme(style="whitegrid")

        # Crear una figura con un solo gráfico. Ajustamos el tamaño.
        fig, ax = plt.subplots(figsize=(14, 7))

        # Título principal
        fig.suptitle('Análisis de la Potencia Activa Total', fontsize=18, weight='bold')

        # Graficar la línea de Potencia Media
        ax.plot(df.index, df['Potencia Total Med'], color='royalblue', label='Potencia Total Media', linewidth=2)

        # Añadir el área sombreada para el rango Min-Max
        ax.fill_between(
            df.index,
            df['Potencia Total Min'],
            df['Potencia Total Max'],
            color='royalblue',
            alpha=0.2,
            label='Rango Potencia Total (Mín-Máx)'
        )

        # --- AJUSTES DEL GRÁFICO Y GUARDADO ---
        ax.set_ylabel('Potencia (W)', fontsize=12, weight='bold')
        ax.set_xlabel('Fecha', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Mejorar el formato de las fechas en el eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        nombre_archivo_salida = 'grafico_potencia_total.svg'
        plt.savefig(nombre_archivo_salida, format='svg', bbox_inches='tight')

        print(f"¡Éxito! El gráfico ha sido guardado como '{nombre_archivo_salida}'")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo_csv}'.")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")


if __name__ == '__main__':
    archivo_de_datos = 'PotenciaActiva.csv'
    generar_visualizacion_total(archivo_de_datos)