import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("Iniciando la generación del mapa de correlación...")

# --- Configuración ---
file_path = 'pv_data.xlsx'
output_filename = 'mapa_correlacion_pearson.svg'

# --- 1. Carga y Preparación de Datos ---
if not os.path.exists(file_path):
    print(f"Error: El archivo de datos '{file_path}' no se encontró.")
    exit()

print(f"Cargando y procesando datos de '{file_path}'...")
df = pd.read_excel(file_path, skiprows=[1], na_values='n/a')
df.columns = [
    'Timestamp', 'Producción (Wh)', 'Irradiación (W/m²)',
    'Temp. Ambiente (°C)', 'Temp. Módulo (°C)'
]
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M')
df.set_index('Timestamp', inplace=True)

# --- 2. Filtrado y Limpieza para los primeros 5 meses ---
print("Filtrando datos para los primeros 5 meses...")
df_filtered = df[df.index.month <= 5].copy()

# La correlación no funciona con valores nulos, así que los rellenamos
df_filtered.interpolate(method='linear', inplace=True)
df_filtered.dropna(inplace=True)

# Seleccionamos solo las columnas numéricas para la matriz
numeric_df = df_filtered.select_dtypes(include=['number'])

# --- 3. Cálculo de la Matriz de Correlación de Pearson ---
print("Calculando la matriz de correlación de Pearson...")
correlation_matrix = numeric_df.corr(method='pearson')

# --- 4. Creación del Mapa de Calor (Heatmap) ---
print("Generando el mapa de calor...")
plt.figure(figsize=(12, 10))

# Se crea el mapa de calor con Seaborn
# annot=True: Muestra los valores numéricos en cada celda.
# cmap='RdYlGn': Define la paleta de colores Rojo-Amarillo-Verde.
# vmin=-1, vmax=1: Fija los límites del color a los de la correlación de Pearson.
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f", # Formato de 2 decimales para los números
    cmap='RdYlGn',
    vmin=-1,
    vmax=1,
    linewidths=.5,
    annot_kws={"size": 14} # Tamaño de la fuente de los números
)

# Personalización del gráfico
plt.title("Mapa de Correlación de Pearson (Primeros 5 Meses)", fontsize=20)
plt.xticks(fontsize=18, rotation=45, ha="right")
plt.yticks(fontsize=18, rotation=0)

# Ajusta el layout para que todo sea visible
plt.tight_layout(pad=2.0)

# --- 5. Guardado del Gráfico ---
try:
    plt.savefig(output_filename, format='svg')
    print(f"\n✅ ¡Mapa de calor guardado exitosamente como '{output_filename}'!")
except Exception as e:
    print(f"\nError al guardar el mapa de calor: {e}")