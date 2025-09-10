import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import locale

print("Iniciando la generación del gráfico multivariable...")

# --- Configuración ---
file_path = 'pv_data.xlsx'
output_filename = 'variables_plot_5_meses.svg'
font_size = 22


# Intenta configurar el idioma a español para los nombres de los meses
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    print("Advertencia: No se pudo configurar el idioma a español. Se usarán los nombres de meses por defecto.")

# --- 1. Carga y Preparación de Datos ---
if not os.path.exists(file_path):
    print(f"Error: El archivo de datos '{file_path}' no se encontró.")
    exit()

print(f"Cargando y procesando datos de '{file_path}'...")
df = pd.read_excel(file_path, skiprows=[1], na_values='n/a')
df.columns = [
    'Timestamp', 'PV_Production_Wh', 'Irradiation_Wm2',
    'Ambient_Temp_C', 'Module_Temp_C'
]
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M')
df.set_index('Timestamp', inplace=True) # Establecer Timestamp como índice es útil para filtrar

# --- 2. Filtrado y Limpieza para los primeros 5 meses ---
print("Filtrando datos para los primeros 5 meses...")
df_filtered = df[df.index.month <= 5].copy()

# Usamos interpolación para rellenar huecos en los datos, ideal para gráficos de línea
df_filtered.interpolate(method='linear', inplace=True)
df_filtered.dropna(inplace=True) # Elimina cualquier nulo restante

print(f"Se graficarán {df_filtered.shape[0]} registros.")

# --- 3. Creación de los Gráficos (Subplots) ---
# Se crean 4 subgráficos apilados verticalmente (4 filas, 1 columna)
# sharex=True hace que todos compartan el mismo eje X, lo que es ideal para comparar en el tiempo
fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

# --- Gráfico 1: Producción Fotovoltaica ---
axes[0].plot(df_filtered.index, df_filtered['PV_Production_Wh'], color='steelblue', linewidth=1)
axes[0].set_ylabel("PVP (Wh)", fontsize=font_size)
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- Gráfico 2: Irradiación Solar ---
axes[1].plot(df_filtered.index, df_filtered['Irradiation_Wm2'], color='purple', linewidth=1)
axes[1].set_ylabel("Irradiación (W/m²)", fontsize=font_size)
axes[1].grid(True, linestyle='--', alpha=0.6)

# --- Gráfico 3: Temperatura Ambiente ---
axes[2].plot(df_filtered.index, df_filtered['Ambient_Temp_C'], color='green', linewidth=1)
axes[2].set_ylabel("Temp. Ambiente (°C)", fontsize=font_size)
axes[2].grid(True, linestyle='--', alpha=0.6)

# --- Gráfico 4: Temperatura del Módulo ---
axes[3].plot(df_filtered.index, df_filtered['Module_Temp_C'], color='red', linewidth=1)
axes[3].set_ylabel("Temp. Módulo (°C)", fontsize=font_size)

# --- 4. Personalización General del Gráfico ---
# Solo el último gráfico necesita la etiqueta del eje X
axes[3].set_xlabel("Mes", fontsize=font_size)
axes[3].grid(True, linestyle='--', alpha=0.6)

# Se configura el formato del eje X para todos los gráficos (ya que está compartido)
axes[3].xaxis.set_major_locator(mdates.MonthLocator())
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%b')) # 'Ene', 'Feb', etc.

# Se ajusta el tamaño de la fuente de todos los ticks
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=font_size)

# Ajusta el espaciado para que no se solapen los elementos
plt.tight_layout(pad=1.0)

# --- 5. Guardado del Gráfico ---
try:
    plt.savefig(output_filename, format='svg')
    print(f"\n✅ ¡Gráfico guardado exitosamente como '{output_filename}'!")
except Exception as e:
    print(f"\nError al guardar el gráfico: {e}")