"""
06b_logreg_coeficientes_top20_display.py

Objetivo:
- Generar barplot de coeficientes de la regresión logística
- Usar nombres legibles: GENE_SYMBOL (probe_id)
- Mostrar Top 20 genes por |coeficiente|

Requisitos:
- models/logistic_regression_gse5281.pkl
- results/top20_biomarcadores_logreg_coef.csv
- results/annotation/probe_to_gene_mapping.csv
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# Config
# =========================
coef_path = "results/top20_biomarcadores_logreg_coef.csv"
model_path = "models/logistic_regression_gse5281.pkl"
mapping_path = "results/annotation/probe_to_gene_mapping.csv"

out_dir = "figures/linear_model"
os.makedirs(out_dir, exist_ok=True)

# =========================
# 1) Cargar datos
# =========================
print("=== CARGANDO COEFICIENTES Y MAPPING ===")
coef_df = pd.read_csv(coef_path)

mapping = pd.read_csv(mapping_path)
probe_to_symbol = dict(zip(mapping["probe_id"], mapping["gene_symbol"]))

def gene_display(probe: str) -> str:
    sym = probe_to_symbol.get(probe, "")
    sym = "" if pd.isna(sym) else str(sym).strip()
    probe = str(probe).strip()
    return f"{sym} ({probe})" if sym else probe

coef_df["gene_display"] = coef_df["gene"].apply(gene_display)

# Ordenar para visualización
coef_df = coef_df.sort_values("coeficiente")

print("Top genes a visualizar:")
print(coef_df[["gene_display", "coeficiente"]])

# =========================
# 2) Crear barplot
# =========================
plt.figure(figsize=(8, 6))

colors = coef_df["coeficiente"].apply(
    lambda x: "red" if x > 0 else "blue"
)

plt.barh(
    coef_df["gene_display"],
    coef_df["coeficiente"],
    color=colors
)

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Coeficiente del modelo (log-odds)")
plt.ylabel("Gen")
plt.title("Genes más influyentes según regresión logística")

legend_elements = [
    Patch(facecolor="red", label="Asociado a Alzheimer"),
    Patch(facecolor="blue", label="Asociado a Control")
]
plt.legend(handles=legend_elements)

out_path = os.path.join(out_dir, "logreg_coeficientes_top20_display.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Figura guardada en: {out_path}")
print("Listo para usar en la tesis.")
