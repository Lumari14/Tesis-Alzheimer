"""
Script: 04b_eda_distribuciones.py

Objetivo:
- Análisis exploratorio (EDA) básico del dataset limpio:
  1) Distribución de muestras por diagnóstico
  2) Distribución de muestras por región cerebral
  3) Tabla cruzada diagnosis x region (opcional, muy útil)

Entradas:
- data/processed/GSE5281_datos_limpios.csv

Salidas:
- figures/eda/distribucion_diagnosis.png
- figures/eda/distribucion_region.png
- results/eda/tabla_diagnosis_region.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) Cargar datos limpios
# =========================
input_path = "data/processed/GSE5281_datos_limpios.csv"
df = pd.read_csv(input_path, index_col=0)

# Crear carpetas de salida
os.makedirs("figures/eda", exist_ok=True)
os.makedirs("results/eda", exist_ok=True)

# =========================
# 2) Distribución por diagnóstico
# =========================
diag_counts = df["diagnosis"].value_counts()

plt.figure()
diag_counts.plot(kind="bar")
plt.title("Distribución de muestras por diagnóstico (dataset limpio)")
plt.xlabel("Diagnóstico")
plt.ylabel("Número de muestras")
plt.xticks(rotation=0)
out_diag = "figures/eda/distribucion_diagnosis.png"
plt.savefig(out_diag, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Guardado: {out_diag}")
print("\nDistribución por diagnóstico:\n", diag_counts)

# =========================
# 3) Distribución por región cerebral
# =========================
region_counts = df["region"].value_counts()

plt.figure(figsize=(10, 5))
region_counts.plot(kind="bar")
plt.title("Distribución de muestras por región cerebral (dataset limpio)")
plt.xlabel("Región cerebral")
plt.ylabel("Número de muestras")
plt.xticks(rotation=45, ha="right")
out_region = "figures/eda/distribucion_region.png"
plt.savefig(out_region, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Guardado: {out_region}")
print("\nDistribución por región:\n", region_counts)

# =========================
# 4) Tabla cruzada diagnosis x region (recomendado)
# =========================
# Esto es MUY útil para tu tesis: muestra cuántos controles y casos hay por región
cross_tab = pd.crosstab(df["region"], df["diagnosis"])
out_tab = "results/eda/tabla_diagnosis_region.csv"
cross_tab.to_csv(out_tab)
print(f"\n✅ Tabla cruzada guardada: {out_tab}")
print("\nTabla diagnosis x region:\n", cross_tab)

print("\n✅ EDA básico completado.")

# =========================
# 5) Gráfico de puntos: diagnóstico vs región cerebral
# =========================
# Este gráfico permite visualizar cómo se distribuyen
# controles y casos de Alzheimer en cada región cerebral.

import numpy as np

# Convertimos la tabla cruzada a formato largo (long format)
cross_tab_long = cross_tab.reset_index().melt(
    id_vars="region",
    var_name="diagnosis",
    value_name="count"
)

plt.figure(figsize=(10, 5))

for diag in cross_tab_long["diagnosis"].unique():
    subset = cross_tab_long[cross_tab_long["diagnosis"] == diag]
    plt.scatter(
        subset["region"],
        subset["count"],
        s=subset["count"] * 10,  # tamaño del punto proporcional al nº de muestras
        alpha=0.7,
        label=diag
    )

plt.xticks(rotation=45, ha="right")
plt.xlabel("Región cerebral")
plt.ylabel("Número de muestras")
plt.title("Distribución de muestras por diagnóstico y región cerebral")
plt.legend(title="Diagnóstico")

out_scatter = "figures/eda/diagnosis_vs_region_scatter.png"
plt.savefig(out_scatter, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Gráfico diagnóstico vs región guardado en: {out_scatter}")
