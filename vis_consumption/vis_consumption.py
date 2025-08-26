import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


def generar_scatterplot_media(nombre_archivo_csv: str):
    """
    Genera un scatterplot sin título y con fuentes agrandadas, mostrando
    únicamente la Potencia Total Media del archivo PotenciaActiva.csv.

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

        # Asegurarse de que las columnas de potencia son numéricas
        cols_potencia = [col for col in df.columns if 'Potencia' in col]
        for col in cols_potencia:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=cols_potencia, inplace=True)

        print("Datos cargados. Generando scatterplot de Potencia Total Media...")

        # --- CREACIÓN DE LA VISUALIZACIÓN ---
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(14, 8))

        # --- CAMBIO: Se grafica únicamente la Potencia Total Media ---
        ax.scatter(df.index, df['Potencia Total Med'], label='Potencia Total Media', s=15, alpha=0.7,
                   color='dodgerblue')

        # Aumento del tamaño de las fuentes
        ax.set_ylabel('Potencia (W)', fontsize=18, weight='bold')
        ax.set_xlabel('Fecha', fontsize=18, weight='bold')
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

        # --- AJUSTES DEL GRÁFICO Y GUARDADO ---
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        nombre_archivo_salida = 'grafico_potencia_total.svg'
        plt.savefig(nombre_archivo_salida, format='svg', bbox_inches='tight')

        print(f"¡Éxito! El gráfico ha sido guardado como '{nombre_archivo_salida}'")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo_csv}'.")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")


if __name__ == '__main__':
    archivo_de_datos = 'PotenciaActiva.csv'
    generar_scatterplot_media(archivo_de_datos)