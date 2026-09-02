# Renewable Energy Forecasting — UFPS Microgrid

Machine learning models for forecasting **photovoltaic generation, wind generation and electricity consumption** in the microgrid of Universidad Francisco de Paula Santander (Cúcuta, Colombia).

This is the codebase of my undergraduate thesis in Electronics Engineering, *"Predicción de producción de energía renovable y consumo eléctrico para la microrred de la UFPS usando técnicas de aprendizaje automático"* (2025), directed by Ph.D. Sergio Basilio Sepúlveda Mora and co-directed by Mag. Mario Joaquín Illera Bustos.

A derived paper was **accepted at EU PVSEC 2025** (European Photovoltaic Solar Energy Conference and Exhibition), and a second one is under review at *Revista UIS Ingenierías*.

## Results

Six model families were benchmarked with a weighted selection matrix over prediction error and execution time.

| Metric | Best model | Value |
|---|---|---|
| R² (coefficient of determination) | ANN / SVR / GB | **0.982** |
| RMSE | ANN | **13.554 Wh** |
| RMSE | SVR | 13.563 Wh |
| RMSE | Gradient Boosting | 14.325 Wh |

All three models explain **over 98% of the variance** in photovoltaic production, a signal characterised by high variability.

## Repository layout

| Folder | Contents |
|---|---|
| `Application/` | Tkinter desktop app for daily predictions. Models persisted with `joblib` for fast inference (`ann_model.joblib`, `scaler.joblib`). |
| `mod_fotovol_ANN/`, `mod_fotovol_SVM/`, `mod_fotovol_MCH/` | Photovoltaic generation models — ANN, SVR and ensemble (Gradient Boosting). |
| `mod_eolic_ANN/`, `mod_eolic_SVR/`, `mod_eolic_XGBoost/` | Wind generation models. |
| `mod_cons_ANN/`, `mod_cons_SVM/`, `mod_cons_XGBOOST/` | Electricity consumption models. |
| `Correlation Photovol/`, `Correlation Eolic/` | Pearson correlation analysis and scatter plots for feature selection. |
| `vis_fotovol/`, `vis_eolic/`, `vis_consumption/` | Exploratory visualisation of the raw series before imputation. |
| `Fotovol_test/` | Test bench for individual runs. |

## Data

Production data comes from **Fronius SolarWeb**, the platform managing the university's photovoltaic installation. The models expect the CSV layout that SolarWeb generates in its periodic reports — a different file format or column order will fail.

## Stack

Python · TensorFlow · Scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Joblib · Tkinter

## Notes and future work

- Panel temperature turned out to have low impact on forecast quality. Cloud cover and relative humidity are the variables worth adding, but the physical installation had no sensors for them.
- Day-ahead forecasts without cloud cover data carry visible error.
- Visualising the series before imputation was essential to catch erratic behaviour and data gaps.

## Author

**Edward Julian Rojas-Ortega** — Electronics Engineer, UFPS
[LinkedIn](https://www.linkedin.com/in/edward-rojas-ortega/) · edwardrojas0301@gmail.com
