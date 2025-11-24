"""
Script: 07_shap_random_forest.py

Objetivo
--------
Aplicar SHAP al modelo Random Forest entrenado sobre GSE5281 para:

1. Calcular los valores SHAP (importancia de cada gen por muestra).
2. Obtener una medida global de importancia (media del valor absoluto SHAP por gen).
3. Generar gráficos:
      - summary beeswarm plot (global)
      - summary bar plot (importancia media)
      - dependence plots para los genes más importantes.
4. Guardar:
      - Tabla CSV con importancia SHAP global por gen.
      - Tabla CSV con el top 20 de genes más importantes.
      - Figuras en 'figures/shap/'.

Este análisis permite interpretar el modelo Random Forest de forma no lineal
y complementar la identificación de posibles biomarcadores transcriptómicos.
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shap

from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# 1. Rutas y carga de datos + modelo
# ============================================================

input_dir   = "data/processed/model_input"
models_dir  = "models"
results_dir = "results"
figures_dir = "figures"
shap_dir    = os.path.join(figures_dir, "shap")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(shap_dir, exist_ok=True)

print("=== CARGANDO DATOS Y MODELO RANDOM FOREST ===")

X_train_path = os.path.join(input_dir, "X_train_scaled.npy")
X_test_path  = os.path.join(input_dir, "X_test_scaled.npy")
y_train_path = os.path.join(input_dir, "y_train.npy")
y_test_path  = os.path.join(input_dir, "y_test.npy")
genes_path   = os.path.join(input_dir, "gene_columns.txt")

# Datos escalados
X_train_scaled = np.load(X_train_path)
X_test_scaled  = np.load(X_test_path)

# Etiquetas (strings)
y_train = np.load(y_train_path, allow_pickle=True)
y_test  = np.load(y_test_path, allow_pickle=True)

# Nombres de todos los genes
with open(genes_path, "r") as f:
    gene_names = [line.strip() for line in f.readlines()]

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled  shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")
print(f"Número total de genes: {len(gene_names)}")
print()

# Cargar modelo Random Forest entrenado
rf_path = os.path.join(models_dir, "random_forest_gse5281.pkl")
with open(rf_path, "rb") as f:
    rf = pickle.load(f)

print(f"✅ Modelo Random Forest cargado desde: {rf_path}")
print("Clases del modelo:", rf.classes_)
print()


# ============================================================
# 2. Reconstruir SelectKBest (mismas 2000 features que en el entrenamiento)
# ============================================================

K_FEATURES = 2000
print("=== RECONSTRUYENDO SELECTKBEST (ANOVA, k=2000) ===")

selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel  = selector.transform(X_test_scaled)

print("Forma de X_train_sel:", X_train_sel.shape)
print("Forma de X_test_sel: ", X_test_sel.shape)
print()

selected_indices = selector.get_support(indices=True)
selected_genes   = [gene_names[i] for i in selected_indices]

print(f"Número de genes seleccionados: {len(selected_genes)} (debería ser {K_FEATURES})")
print("Ejemplos de genes seleccionados:", selected_genes[:10])
print()


# ============================================================
# 3. Cálculo de valores SHAP con TreeExplainer
# ============================================================

print("=== CALCULANDO VALORES SHAP PARA RANDOM FOREST ===")

n_samples, n_features = X_train_sel.shape
classes = list(rf.classes_)
positive_class = "Alzheimer's Disease"
pos_index = classes.index(positive_class)

print("Clases del modelo:", classes)
print("Índice de la clase positiva:", pos_index)
print()

explainer = shap.TreeExplainer(rf)
shap_values_all = explainer.shap_values(X_train_sel)

# --- Normalizar forma de shap_values para obtener siempre (n_samples, n_features) ---

def get_shap_values_for_positive_class(shap_values_all, pos_index, n_samples, n_features, n_classes):
    """
    Devuelve un array de shape (n_samples, n_features) con los valores SHAP
    correspondientes a la clase positiva, independientemente de cómo SHAP
    devuelva las dimensiones.
    """
    # Caso 1: lista de arrays (uno por clase)
    if isinstance(shap_values_all, list):
        # shap_values_all[pos_index]: (n_samples, n_features)
        return np.array(shap_values_all[pos_index])

    # Caso 2: array de numpy
    sh = np.array(shap_values_all)
    print("Forma bruta de shap_values_all:", sh.shape)

    if sh.ndim == 3:
        # Intentamos detectar qué eje corresponde a qué
        s0, s1, s2 = sh.shape

        # Posibles formas:
        # (n_samples, n_classes, n_features)
        if s0 == n_samples and s1 == n_classes and s2 == n_features:
            return sh[:, pos_index, :]
        # (n_classes, n_samples, n_features)
        if s0 == n_classes and s1 == n_samples and s2 == n_features:
            return sh[pos_index, :, :]
        # (n_samples, n_features, n_classes)
        if s0 == n_samples and s1 == n_features and s2 == n_classes:
            return sh[:, :, pos_index]

        raise ValueError(
            f"No se reconoce la forma de shap_values_all: {sh.shape} "
            f"(esperado algo como (samples, classes, features), etc.)"
        )

    elif sh.ndim == 2:
        # Ya está en forma (n_samples, n_features)
        return sh

    else:
        raise ValueError(
            f"Dimensionalidad inesperada de shap_values_all: {sh.ndim}D con forma {sh.shape}"
        )

n_classes = len(classes)
shap_values_pos = get_shap_values_for_positive_class(
    shap_values_all,
    pos_index,
    n_samples,
    n_features,
    n_classes
)

print("Forma normalizada de shap_values_pos (esperado: n_samples x n_features):",
      shap_values_pos.shape)
print()


# ============================================================
# 4. Importancia global basada en SHAP (media del valor absoluto)
# ============================================================

print("=== CALCULANDO IMPORTANCIA GLOBAL SHAP POR GEN ===")

# Media del valor absoluto SHAP por gen -> vector de longitud n_features
mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)  # (n_features,)

shap_importance_df = pd.DataFrame({
    "gene": selected_genes,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

shap_importance_path = os.path.join(results_dir, "shap_importancia_rf_todos_genes.csv")
shap_importance_df.to_csv(shap_importance_path, index=False)
print(f"✅ Importancia SHAP global guardada en: {shap_importance_path}")

TOP_N = 20
top_shap_df = shap_importance_df.head(TOP_N)
top_shap_path = os.path.join(results_dir, f"top{TOP_N}_genes_shap_rf.csv")
top_shap_df.to_csv(top_shap_path, index=False)
print(f"✅ Top {TOP_N} genes según SHAP guardado en: {top_shap_path}")
print()


# ============================================================
# 5. Gráficos SHAP: summary (beeswarm) y bar plot
# ============================================================

print("=== GENERANDO GRÁFICOS SHAP GLOBALES (summary plots) ===")

top_genes_list = top_shap_df["gene"].tolist()

# DataFrame para nombres de columnas
X_train_sel_df = pd.DataFrame(X_train_sel, columns=selected_genes)

# Indices de los top genes en el orden de selected_genes
top_indices = [selected_genes.index(g) for g in top_genes_list]

X_train_top = X_train_sel_df[top_genes_list].values                 # (n_samples, TOP_N)
shap_values_top = shap_values_pos[:, top_indices]                   # (n_samples, TOP_N)

# 5.1 Beeswarm plot
plt.figure()
shap.summary_plot(
    shap_values_top,
    X_train_top,
    feature_names=top_genes_list,
    show=False
)
plt.tight_layout()
summary_beeswarm_path = os.path.join(shap_dir, f"shap_summary_beeswarm_rf_top{TOP_N}.png")
plt.savefig(summary_beeswarm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Beeswarm summary plot guardado en: {summary_beeswarm_path}")

# 5.2 Bar plot
plt.figure()
shap.summary_plot(
    shap_values_top,
    X_train_top,
    feature_names=top_genes_list,
    plot_type="bar",
    show=False
)
plt.tight_layout()
summary_bar_path = os.path.join(shap_dir, f"shap_summary_bar_rf_top{TOP_N}.png")
plt.savefig(summary_bar_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Bar summary plot guardado en: {summary_bar_path}")
print()


# ============================================================
# 6. Gráficos SHAP: dependence plots para los top genes
# ============================================================

print(f"=== GENERANDO SHAP DEPENDENCE PLOTS PARA TOP {TOP_N} GENES ===")

for gene in top_genes_list:
    plt.figure()
    shap.dependence_plot(
        gene,
        shap_values_pos,
        X_train_sel_df,
        feature_names=selected_genes,
        show=False
    )
    plt.tight_layout()
    dep_path = os.path.join(shap_dir, f"shap_dependence_rf_{gene}.png")
    plt.savefig(dep_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Dependence plot guardado para {gene}: {dep_path}")

print()
print("✅ Análisis SHAP para Random Forest completado.")
print("Revisa en:")
print(f"   - {results_dir}  -> tablas CSV de importancia SHAP.")
print(f"   - {shap_dir}     -> figuras SHAP (summary y dependence plots).")
