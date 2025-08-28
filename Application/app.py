import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import os
import numpy as np  # <--- Importado para calcular la raíz cuadrada
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, r2_score

# --- Variables Globales ---
MODEL_PATH = 'ann_model.joblib'
SCALER_PATH = 'scaler.joblib'
model = None
scaler = None
last_prediction_data = None
last_prediction_timestamps = None


# --- Funciones de Lógica ---

def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        status_label.config(text="Modelo cargado correctamente.", foreground="green")
        return True
    else:
        messagebox.showerror("Error",
                             "No se encontraron los archivos del modelo. Ejecuta primero 'train_initial_model.py'.")
        root.destroy()
        return False


def process_input_csv(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=[1])
        column_map = {
            'Fecha y hora': 'Timestamp', 'Producción fotovoltaica': 'PV_Production_Wh',
            'Irradiación | Sensor Card / Box (1)': 'Irradiation_Wm2',
            'Temperatura ambiente | Sensor Card / Box (1)': 'Ambient_Temp_C',
            'Temperatura de módulo | Sensor Card / Box (1)': 'Module_Temp_C'
        }
        df.rename(columns=column_map, inplace=True)
        required_features = {'Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C'}
        if not required_features.issubset(df.columns):
            messagebox.showerror("Error de Formato", f"El CSV debe contener: {', '.join(required_features)}")
            return None
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M')
        else:
            df['Timestamp'] = pd.to_datetime(
                pd.date_range(start=pd.to_datetime('today').date(), periods=len(df), freq='5T'))
        return df
    except Exception as e:
        messagebox.showerror("Error de Lectura", f"No se pudo procesar el archivo CSV: {e}")
        return None


def predict_production():
    global last_prediction_data, last_prediction_timestamps
    file_path = filedialog.askopenfilename(
        title="Selecciona el INFORME CSV para la predicción",
        filetypes=(("CSV Files", "*.csv"),)
    )
    if not file_path: return
    processed_df = process_input_csv(file_path)
    if processed_df is None: return
    features_df = processed_df[['Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']]
    timestamps = processed_df['Timestamp']
    X_scaled = scaler.transform(features_df)
    predictions = model.predict(X_scaled)
    last_prediction_data = predictions
    last_prediction_timestamps = timestamps
    update_plot(timestamps, predictions, title="Predicción de Producción Fotovoltaica")
    verify_button.config(state=tk.NORMAL)
    rmse_label_value.config(text="N/A")  # <--- Actualizado para RMSE
    r2_label_value.config(text="N/A")
    messagebox.showinfo("Éxito", f"Se ha generado una predicción para {len(predictions)} intervalos de tiempo.")


def verify_and_retrain():
    global model
    if last_prediction_data is None:
        messagebox.showwarning("Advertencia", "Primero debes generar una predicción.")
        return
    file_path = filedialog.askopenfilename(
        title="Selecciona el INFORME CSV con los datos REALES",
        filetypes=(("CSV Files", "*.csv"),)
    )
    if not file_path: return
    processed_df = process_input_csv(file_path)
    if processed_df is None: return
    if 'PV_Production_Wh' not in processed_df.columns:
        messagebox.showerror("Error de Formato",
                             "El CSV de verificación debe contener la columna 'Producción fotovoltaica'.")
        return
    prediction_len = len(last_prediction_data)
    verification_len = len(processed_df)
    if verification_len < prediction_len:
        num_missing = prediction_len - verification_len
        messagebox.showinfo("Normalización", f"Se añadirán {num_missing} filas con valores cero para completar.")
        padding_data = {col: [0] * num_missing for col in processed_df.columns if col != 'Timestamp'}
        padding_df = pd.DataFrame(padding_data)
        last_timestamp = processed_df['Timestamp'].iloc[-1]
        try:
            freq = pd.infer_freq(processed_df['Timestamp']) or '5T'
        except TypeError:
            freq = '5T'
        new_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(freq), periods=num_missing, freq=freq)
        padding_df['Timestamp'] = new_timestamps
        processed_df = pd.concat([processed_df, padding_df], ignore_index=True)
    elif verification_len > prediction_len:
        messagebox.showwarning("Normalización", "Se recortarán los datos de verificación para que coincidan.")
        processed_df = processed_df.iloc[:prediction_len]

    real_values = processed_df['PV_Production_Wh'].values
    features_df = processed_df[['Irradiation_Wm2', 'Ambient_Temp_C', 'Module_Temp_C']]
    timestamps = processed_df['Timestamp']

    # --- CAMBIO EN LA MÉTRICA: DE MSE A RMSE ---
    mse = mean_squared_error(real_values, last_prediction_data)
    rmse = np.sqrt(mse)  # Se calcula la raíz cuadrada del MSE
    r2 = r2_score(real_values, last_prediction_data)

    # Se actualizan las etiquetas con el nuevo valor
    rmse_label_value.config(text=f"{rmse:.2f} Wh")  # <--- Actualizado para RMSE
    r2_label_value.config(text=f"{r2:.2f}")

    update_plot(timestamps, last_prediction_data, real_values, title="Comparación: Predicción vs. Real")

    print("Iniciando reaprendizaje...")
    X_new_scaled = scaler.transform(features_df)
    model.partial_fit(X_new_scaled, real_values)
    joblib.dump(model, MODEL_PATH)
    print("Modelo actualizado y guardado.")
    messagebox.showinfo("Reaprendizaje Completo", "El modelo ha aprendido de los nuevos datos y ha sido actualizado.")


def update_plot(timestamps, predicted_data, real_data=None, title=""):
    ax.clear()
    ax.plot(timestamps, predicted_data, label='Producción Predicha', color='orangered', linestyle='--', marker='.',
            markersize=2)
    if real_data is not None:
        ax.plot(timestamps, real_data, label='Producción Real', color='dodgerblue', alpha=0.8, marker='.', markersize=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Hora del Día", fontsize=10)
    ax.set_ylabel("Producción (Wh)", fontsize=10)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor("#E6E6E6")
    fig.autofmt_xdate()
    canvas.draw()


# --- Configuración de la Interfaz Gráfica (GUI) ---
BG_COLOR = "#B0E0E6"
BTN_COLOR = "#336699"
BTN_HOVER_COLOR = "#4682B4"
TEXT_COLOR = "white"

root = tk.Tk()
root.title("Software de Predicción Fotovoltaica")
root.geometry("1200x750")
root.configure(bg=BG_COLOR)

style = ttk.Style(root)
style.theme_use('clam')
style.configure("TFrame", background=BG_COLOR)
style.configure("TLabel", background=BG_COLOR, font=("Helvetica", 10))
style.configure("TLabelframe", background=BG_COLOR, bordercolor=BTN_COLOR)
style.configure("TLabelframe.Label", background=BG_COLOR, foreground="black", font=("Helvetica", 11, "bold"))
style.configure("custom.TButton", background=BTN_COLOR, foreground=TEXT_COLOR, font=("Helvetica", 12, "bold"),
                borderwidth=0, padding=10)
style.map("custom.TButton", background=[('active', BTN_HOVER_COLOR), ('!disabled', BTN_COLOR)],
          foreground=[('active', TEXT_COLOR)])

main_frame = ttk.Frame(root, padding="10", style="TFrame")
main_frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(main_frame, text="Software de Predicción de PVP UFPS", font=("Helvetica", 20, "bold"),
                        anchor="center")
title_label.pack(pady=(5, 0))
author_label = ttk.Label(main_frame, text="by: Edward Rojas", font=("Helvetica", 10, "italic"), anchor="center")
author_label.pack(pady=(0, 15))

control_frame = ttk.Frame(main_frame)
control_frame.pack(fill=tk.X, pady=5)
control_frame.columnconfigure((0, 1), weight=1)

predict_button = ttk.Button(control_frame, text="1. Cargar Informe y Predecir", command=predict_production,
                            style="custom.TButton")
predict_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

verify_button = ttk.Button(control_frame, text="2. Verificar y Reaprender", command=verify_and_retrain,
                           state=tk.DISABLED, style="custom.TButton")
verify_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

content_frame = ttk.Frame(main_frame)
content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
content_frame.columnconfigure(0, weight=3)
content_frame.columnconfigure(1, weight=1)
content_frame.rowconfigure(0, weight=1)

plot_frame = ttk.LabelFrame(content_frame, text="Gráfico de Predicción")
plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

fig, ax = plt.subplots()
fig.patch.set_facecolor(BG_COLOR)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

metrics_frame = ttk.LabelFrame(content_frame, text="Métricas de Verificación")
metrics_frame.grid(row=0, column=1, sticky="nsew")

# --- CAMBIO DE ETIQUETA: DE MSE A RMSE ---
rmse_label_title = ttk.Label(metrics_frame, text="Raíz del Error Cuadrático Medio (RMSE):",
                             font=("Helvetica", 12, "bold"), anchor="center")
rmse_label_title.pack(pady=(50, 0), fill=tk.X)
rmse_label_value = ttk.Label(metrics_frame, text="N/A", font=("Courier", 20), foreground=BTN_COLOR, anchor="center")
rmse_label_value.pack(pady=5, fill=tk.X)

r2_label_title = ttk.Label(metrics_frame, text="Coeficiente R²:", font=("Helvetica", 12, "bold"), anchor="center")
r2_label_title.pack(pady=(40, 0), fill=tk.X)
r2_label_value = ttk.Label(metrics_frame, text="N/A", font=("Courier", 20), foreground=BTN_COLOR, anchor="center")
r2_label_value.pack(pady=5, fill=tk.X)

status_frame = ttk.Frame(main_frame, padding=5)
status_frame.pack(fill=tk.X)
status_label = ttk.Label(status_frame, text="Cargando modelo...", foreground="black")
status_label.pack(side=tk.LEFT)


def on_enter(e):
    e.widget.config(style="hover.TButton")


def on_leave(e):
    e.widget.config(style="custom.TButton")


style.configure("hover.TButton", background=BTN_HOVER_COLOR)
predict_button.bind("<Enter>", on_enter)
predict_button.bind("<Leave>", on_leave)
verify_button.bind("<Enter>", on_enter)
verify_button.bind("<Leave>", on_leave)

root.after(100, load_model)
root.mainloop()