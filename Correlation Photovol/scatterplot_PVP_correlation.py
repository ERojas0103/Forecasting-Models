import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

print("Iniciando la generación del gráfico de correlación...")

# --- Configuración ---
file_path = 'pv_data.xlsx'
output_filename = 'correlacion_irradiacion_pvp.png'
font_size = 20

# --- 1. Carga y Preparación de Datos ---
if not os.path.exists(file_path):
    print(f"Error: El archivo de datos '{file_path}' no se encontró.")
    exit()

print(f"Cargando datos desde '{file_path}'...")
df = pd.read_excel(file_path, skiprows=[1], na_values='n/a')
df.columns = [
    'Timestamp', 'PV_Production_Wh', 'Irradiation_Wm2',
    'Ambient_Temp_C', 'Module_Temp_C'
]

# Elimina filas donde falten los datos clave para la correlación
df.dropna(subset=['Irradiation_Wm2', 'PV_Production_Wh'], inplace=True)
# Filtra valores no físicos (producción negativa o irradiación negativa)
df = df[(df['PV_Production_Wh'] >= 0) & (df['Irradiation_Wm2'] >= 0)]

print("Datos cargados y procesados.")

# --- 2. Cálculo de Correlación ---
# Extrae las dos variables de interés
x = df['Irradiation_Wm2']
y = df['PV_Production_Wh']

# Calcula el coeficiente de correlación de Pearson (r)
corr_coeff, _ = pearsonr(x, y)
print(f"Coeficiente de Correlación de Pearson (r): {corr_coeff:.4f}")

# --- 3. Creación del Gráfico de Dispersión ---
print("Generando el gráfico...")
fig, ax = plt.subplots(figsize=(14, 9))

# Dibuja el scatter plot
ax.scatter(x, y, s=10, alpha=0.4, color='darkcyan')

# --- 4. Personalización del Gráfico ---

ax.set_xlabel("Irradiación Solar (W/m²)", fontsize=font_size)
ax.set_ylabel("Producción Fotovoltaica (Wh)", fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.grid(True, linestyle='--', alpha=0.6)

# Añade el valor de la correlación al gráfico
ax.text(
    0.05, 0.95, # Posición en coordenadas del gráfico (esquina superior izquierda)
    f'Correlación de Pearson (r) = {corr_coeff:.4f}',
    transform=ax.transAxes, # Usa el sistema de coordenadas del gráfico
    fontsize=font_size - 2, # Un poco más pequeño que el resto
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
)

plt.tight_layout()

# --- 5. Guardado del Gráfico en PNG ---
try:
    plt.savefig(output_filename, format='png', dpi=300) # dpi=300 para alta resolución
    print(f"\n✅ ¡Gráfico guardado exitosamente como '{output_filename}'!")
except Exception as e:
    print(f"\nError al guardar el gráfico: {e}")