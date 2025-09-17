# --- Bloque 1: Importación de Librerías ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Bloque 2: Carga y Preparación de los Datos ---

# Nombre del archivo de entrada.
file_name = 'T1.csv'

# Verificación de la existencia del archivo.
if not os.path.exists(file_name):
    raise FileNotFoundError(f"El archivo '{file_name}' no se encontró en el directorio actual.")
else:
    # Carga del archivo CSV.
    df = pd.read_csv(file_name)

    # --- Bloque 3: Cálculo y Visualización de la Correlación ---

    print("Datos cargados. Calculando la matriz de correlación...")

    # Seleccionar solo las columnas de interés para el análisis.
    columnas_correlacion = [
        'LV ActivePower (kW)',
        'Wind Speed (m/s)',
        'Wind Direction (°)'
    ]
    df_seleccion = df[columnas_correlacion]

    # Cambiar los nombres para que sean más cortos y legibles en el gráfico.
    df_seleccion.columns = ['Potencia Activa (kW)', 'Velocidad Viento (m/s)', 'Dirección Viento (°)']

    # Calcular la matriz de correlación de Pearson.
    matriz_corr = df_seleccion.corr(method='pearson')

    # Configurar la figura para la visualización.
    plt.figure(figsize=(13, 13))

    # Crear el mapa de calor con seaborn y los ajustes de fuente.
    sns.heatmap(
        matriz_corr,
        annot=True,
        annot_kws={'size': 20},  # Tamaño de fuente para los números
        cmap='RdYlGn',
        fmt='.2f',
        linewidths=.5,
        vmin=-1,
        vmax=1
    )

    # Ajustar el tamaño de fuente para las etiquetas de los ejes X e Y.
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Ajustar para que todo se vea correctamente.
    plt.tight_layout()

    # --- Bloque 4: Guardar el Gráfico ---
    # Se define el nombre del archivo con la nueva extensión.
    output_filename = 'mapa_correlacion_pearson_ajustado.svg'

    # Se guarda la figura especificando el formato SVG.
    plt.savefig(output_filename, format='svg')

    print(f"Proceso finalizado. El mapa de calor ha sido guardado como '{output_filename}'.")

    # Opcional: mostrar el gráfico si se ejecuta interactivamente.
    plt.show()