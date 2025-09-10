import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import locale

print("Iniciando la generación del gráfico de dispersión con detalle diario...")

# --- Configuración ---
file_path = 'pv_data.xlsx'
output_filename = 'produccion_scatterplot_diario_anual.svg'
font_size = 18

# Intenta configurar el idioma a español para los nombres de los meses (ej. 'Ene', 'Feb')
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    print("Advertencia: No se pudo configurar el idioma a español. Se usarán los nombres de meses por defecto.")

# --- 1. Carga y Preparación de Datos ---
if not os.path.exists(file_path):
    print(f"Error: El archivo de datos '{file_path}' no se encontró.")
    exit()

print(f"Cargando datos desde '{file_path}'...")
df = pd.read_excel(file_path, skiprows=[1])
df.columns = [
    'Timestamp', 'PV_Production_Wh', 'Irradiation_Wm2',
    'Ambient_Temp_C', 'Module_Temp_C'
]
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M')
df.dropna(subset=['PV_Production_Wh'], inplace=True)
print("Datos cargados y procesados.")

# --- 2. Creación del Gráfico de Dispersión ---
print("Generando el gráfico...")
fig, ax = plt.subplots(figsize=(20, 8))

# Dibuja el scatter plot
# **CAMBIO CLAVE**: Usamos el Timestamp completo en el eje X para que cada punto
# tenga su propia posición basada en el día y la hora.
ax.scatter(df['Timestamp'], df['PV_Production_Wh'], s=5, alpha=0.5, color='steelblue')

# --- 3. Personalización del Gráfico ---
ax.set_xlabel("Mes", fontsize=font_size)
ax.set_ylabel("PVP (Wh)", fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)

# **CAMBIO CLAVE**: Se configura el eje X para que muestre los meses
# Se establece un marcador principal en el inicio de cada mes.
ax.xaxis.set_major_locator(mdates.MonthLocator())
# Se formatea la etiqueta de ese marcador para que muestre el nombre abreviado del mes (ej. 'Ene').
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Ajusta los límites del eje X para que abarque todo el rango de fechas
ax.set_xlim(df['Timestamp'].min(), df['Timestamp'].max())

ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# --- 4. Guardado del Gráfico ---
try:
    plt.savefig(output_filename, format='svg')
    print(f"\n✅ ¡Gráfico guardado exitosamente como '{output_filename}'!")
except Exception as e:
    print(f"\nError al guardar el gráfico: {e}")