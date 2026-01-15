"""
07b_shap_rf_top20_summary_plots.py

Objetivo:
- Re-crear SOLO 2 figuras SHAP para Random Forest usando nombres legibles:
  1) SHAP summary beeswarm (Top 20)
  2) SHAP summary bar (Top 20, mean(|SHAP|))

Requisitos:
- models/random_forest_gse5281.pkl
- data/processed/model_input/X_train_scaled.npy
- data/processed/model_input/y_train.npy
- data/processed/GSE5281_datos_limpios.csv (para reconstruir SelectKBest coherentemente)
- results/annotation/probe_to_gene_mapping.csv (probe -> gene_symbol)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import shap
from sklearn.feature_selection import SelectKBest, f_classif

# =========================
# Config
# =========================
K_FEATURES = 2000
TOP_N = 20

X_train_path = "data/processed/model_input/X_train_scaled.npy"
y_train_path = "data/processed/model_input/y_train.npy"
clean_df_path = "data/processed/GSE5281_datos_limpios.csv"
rf_model_path = "models/random_forest_gse5281.pkl"
mapping_path = "results/annotation/probe_to_gene_mapping.csv"

out_fig_dir = "figures/shap"
out_res_dir = "results/shap"
os.makedirs(out_fig_dir, exist_ok=True)
os.makedirs(out_res_dir, exist_ok=True)

# =========================
# 1) Cargar datos
# =========================
print("=== CARGANDO DATOS Y MODELO RF ===")
X_train_scaled = np.load(X_train_path)
y_train = np.load(y_train_path, allow_pickle=True)

rf = joblib.load(rf_model_path)

print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"y_train: {y_train.shape}")
print(f"Clases RF: {list(rf.classes_)}")

# =========================
# 2) Reconstruir SelectKBest (ANOVA, k=2000) para obtener:
#    - X_train_sel
#    - selected_genes (probes)
# =========================
print("\n=== RECONSTRUYENDO SELECTKBEST (ANOVA, k=2000) ===")
df_clean = pd.read_csv(clean_df_path, index_col=0)

meta_cols = ["diagnosis", "region", "age", "sex"]
gene_cols = [c for c in df_clean.columns if c not in meta_cols]

# X completo (muestras x genes) debe corresponder al orden de gene_cols
X_full = df_clean[gene_cols].values

# IMPORTANTE:
# X_train_scaled ya es la versión escalada y dividida de X_full en tu pipeline.
# Pero aquí necesitamos SelectKBest con las MISMAS features en el MISMO orden.
# Usaremos X_train_scaled y gene_cols para reconstruir el selector:
# Ajustamos SelectKBest con X_train_scaled y y_train (igual que antes).
selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)

selected_mask = selector.get_support()
selected_genes = np.array(gene_cols)[selected_mask]

print(f"X_train_sel: {X_train_sel.shape}")
print(f"Genes seleccionados: {len(selected_genes)} (esperado {K_FEATURES})")

# =========================
# 3) Crear nombres "GENE_SYMBOL (probe_id)" usando mapping
# =========================
print("\n=== CARGANDO MAPPING PROBE -> GENE SYMBOL (GPL570) ===")
mapping = pd.read_csv(mapping_path)
probe_to_symbol = dict(zip(mapping["probe_id"], mapping["gene_symbol"]))

def gene_display(probe: str) -> str:
    sym = probe_to_symbol.get(probe, "")
    sym = "" if pd.isna(sym) else str(sym).strip()
    probe = str(probe).strip()
    return f"{sym} ({probe})" if sym else probe

gene_display_names = np.array([gene_display(p) for p in selected_genes])

# =========================
# 4) Calcular SHAP values (TreeExplainer)
# =========================
print("\n=== CALCULANDO SHAP VALUES ===")
explainer = shap.TreeExplainer(rf)

# Para clasificación, shap puede devolver array (n_samples, n_features, n_classes)
shap_values_all = explainer.shap_values(X_train_sel)

# Determinar clase positiva: en tu caso Alzheimer’s Disease suele ser la clase positiva
classes = list(rf.classes_)
positive_class = "Alzheimer's Disease" if "Alzheimer's Disease" in classes else classes[0]
pos_idx = classes.index(positive_class)
print(f"Clase positiva elegida: {positive_class} (index={pos_idx})")

# Normalizar forma a (n_samples, n_features)
shap_values_all = np.array(shap_values_all)
# shap_values_all puede venir como:
# - lista (n_classes) de arrays (n_samples, n_features)
# - o array (n_samples, n_features, n_classes)
if shap_values_all.ndim == 3:
    # (n_samples, n_features, n_classes)
    shap_values_pos = shap_values_all[:, :, pos_idx]
elif shap_values_all.ndim == 2:
    # (n_samples, n_features) ya
    shap_values_pos = shap_values_all
else:
    # lista convertida: (n_classes, n_samples, n_features)
    # si quedó como (n_classes, n_samples, n_features)
    if shap_values_all.ndim == 3 and shap_values_all.shape[0] == len(classes):
        shap_values_pos = shap_values_all[pos_idx, :, :]
    else:
        raise ValueError(f"Forma inesperada de SHAP values: {shap_values_all.shape}")

print(f"shap_values_pos: {shap_values_pos.shape} (esperado n_samples x n_features)")

# =========================
# 5) Calcular importancia global (mean abs SHAP)
# =========================
print("\n=== IMPORTANCIA GLOBAL SHAP (mean(|SHAP|)) ===")
mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)

shap_importance_df = pd.DataFrame({
    "probe_id": selected_genes,
    "gene_symbol": [probe_to_symbol.get(p, "") for p in selected_genes],
    "gene_display": gene_display_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

out_all = os.path.join(out_res_dir, "shap_importancia_rf_todos_genes_display.csv")
shap_importance_df.to_csv(out_all, index=False)
print(f"✅ Guardado: {out_all}")

top20_df = shap_importance_df.head(TOP_N).copy()
out_top20 = os.path.join(out_res_dir, "top20_genes_shap_rf_display.csv")
top20_df.to_csv(out_top20, index=False)
print(f"✅ Guardado: {out_top20}")

# Índices (en el espacio de features seleccionadas) de los Top 20
top20_indices = top20_df.index.values
# Pero ojo: top20_df index viene de shap_importance_df tras sort (index original no consecutivo).
# Necesitamos el índice real dentro del array (0..n_features-1). Lo extraemos así:
top20_real_idx = shap_importance_df.reset_index().head(TOP_N)["index"].values

X_train_top20 = X_train_sel[:, top20_real_idx]
shap_top20 = shap_values_pos[:, top20_real_idx]
names_top20 = gene_display_names[top20_real_idx]

# =========================
# 6) FIGURA 1: Summary beeswarm (Top 20)
# =========================
print("\n=== CREANDO FIGURAS SHAP TOP 20 ===")

plt.figure()
shap.summary_plot(
    shap_top20,
    X_train_top20,
    feature_names=names_top20,
    show=False
)
out_beeswarm = os.path.join(out_fig_dir, "shap_summary_beeswarm_rf_top20_display.png")
plt.savefig(out_beeswarm, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Guardado: {out_beeswarm}")

# =========================
# 7) FIGURA 2: Summary bar (Top 20)
# =========================
plt.figure()
shap.summary_plot(
    shap_top20,
    X_train_top20,
    feature_names=names_top20,
    plot_type="bar",
    show=False
)
out_bar = os.path.join(out_fig_dir, "shap_summary_bar_rf_top20_display.png")
plt.savefig(out_bar, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Guardado: {out_bar}")

print("\n✅ Listo. Generadas SOLO las 2 figuras SHAP TOP 20 con gene_symbol (probe_id).")
print("Revisa:")
print(f" - {out_beeswarm}")
print(f" - {out_bar}")
