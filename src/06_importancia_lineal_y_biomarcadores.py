"""
Script: 06_importancia_lineal_y_biomarcadores.py

Objetivo
--------
A partir del modelo de regresión logística previamente entrenado y los datos ya
procesados, este script:

1. Reconstruye la selección de características (SelectKBest con ANOVA, k=2000).
2. Obtiene la importancia de cada gen según:
      - Coeficientes del modelo (y sus odds ratios).
      - Permutation importance (importancia por permutación).
3. Genera tablas con:
      - Todos los genes ordenados por importancia.
      - Top 20 genes más relevantes como posibles biomarcadores.
4. Genera gráficos de barras para los 20 genes más importantes según:
      - |coeficiente|
      - permutation importance

El objetivo es identificar posibles biomarcadores transcriptómicos
interpretables para la enfermedad de Alzheimer.
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance


# ============================================================
# 1. Definir rutas y cargar datos + modelo
# ============================================================

input_dir   = "data/processed/model_input"
models_dir  = "models"
results_dir = "results"
figures_dir = "figures"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=== CARGANDO DATOS Y MODELO ===")

X_train_path = os.path.join(input_dir, "X_train_scaled.npy")
X_test_path  = os.path.join(input_dir, "X_test_scaled.npy")
y_train_path = os.path.join(input_dir, "y_train.npy")
y_test_path  = os.path.join(input_dir, "y_test.npy")
genes_path   = os.path.join(input_dir, "gene_columns.txt")

# Datos escalados (todas las features: ~54.675 genes)
X_train_scaled = np.load(X_train_path)
X_test_scaled  = np.load(X_test_path)

# Etiquetas (strings: "normal", "Alzheimer's Disease")
y_train = np.load(y_train_path, allow_pickle=True)
y_test  = np.load(y_test_path, allow_pickle=True)

# Lista de genes en el orden original
with open(genes_path, "r") as f:
    gene_names = [line.strip() for line in f.readlines()]

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled  shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")
print(f"Número total de genes (features): {len(gene_names)}")
print()

# Cargar el modelo de regresión logística entrenado
logreg_path = os.path.join(models_dir, "logistic_regression_gse5281.pkl")
with open(logreg_path, "rb") as f:
    logreg = pickle.load(f)

print(f"✅ Modelo de regresión logística cargado desde: {logreg_path}")
print()


# ============================================================
# 2. Reconstruir la selección de características (SelectKBest)
# ============================================================

"""
En el script 05 se utilizó SelectKBest con ANOVA F-test y k = 2000, y el
modelo de regresión logística se entrenó con esas 2000 features.

Para poder mapear correctamente los coeficientes del modelo a los nombres
de los genes, necesitamos reproducir exactamente la misma selección:

    - mismo K_FEATURES
    - mismo score_func (f_classif)
    - mismo orden de features

Esto se consigue volviendo a aplicar SelectKBest sobre X_train_scaled, y_train.
"""

K_FEATURES = 2000

print("=== RECONSTRUYENDO SELECTKBEST (ANOVA, k=2000) ===")
selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel  = selector.transform(X_test_scaled)

print("Forma de X_train_sel:", X_train_sel.shape)
print("Forma de X_test_sel: ", X_test_sel.shape)
print("Número de coeficientes en el modelo:", len(logreg.coef_[0]))
print()

# Comprobación: el número de coeficientes debe coincidir con k
if len(logreg.coef_[0]) != K_FEATURES:
    print("⚠️ Advertencia: el número de coeficientes del modelo no coincide con K_FEATURES.")
    print("   Esto podría indicar un cambio entre scripts.")
else:
    print("✅ El número de coeficientes coincide con K_FEATURES.")
print()

# Índices y nombres de genes seleccionados (en el MISMO orden que las columnas de X_train_sel)
selected_indices = selector.get_support(indices=True)
selected_genes   = [gene_names[i] for i in selected_indices]


# ============================================================
# 3. Importancia por coeficientes (y odds ratios)
# ============================================================

"""
Para un modelo de regresión logística binaria:

    - Cada coeficiente indica el efecto de incrementar la expresión de un gen
      sobre el logit de la probabilidad de Alzheimer.
    - exp(coef) = odds ratio (OR):
        OR > 1  -> mayor expresión se asocia a mayor probabilidad de Alzheimer
        OR < 1  -> mayor expresión se asocia a menor probabilidad de Alzheimer

Para medir "importancia", usamos el valor absoluto del coeficiente:
    - |coef| grande -> mayor impacto en la predicción.
"""

print("=== IMPORTANCIA BASADA EN COEFICIENTES ===")

coefs = logreg.coef_[0]

coef_df = pd.DataFrame({
    "gene": selected_genes,
    "coeficiente": coefs,
    "odds_ratio": np.exp(coefs)
})

coef_df["abs_coef"] = coef_df["coeficiente"].abs()

coef_df_sorted = coef_df.sort_values("abs_coef", ascending=False)

# Guardamos todos los genes ordenados por |coef|
coef_all_path = os.path.join(results_dir, "coef_importancia_logreg_todos_genes.csv")
coef_df_sorted.to_csv(coef_all_path, index=False)
print(f"✅ Tabla completa de coeficientes guardada en: {coef_all_path}")

# Top 20 posibles biomarcadores
top_n = 20
top_coef_df = coef_df_sorted.head(top_n)

top_coef_path = os.path.join(results_dir, f"top{top_n}_biomarcadores_logreg_coef.csv")
top_coef_df.to_csv(top_coef_path, index=False)
print(f"✅ Top {top_n} biomarcadores (por |coef|) guardado en: {top_coef_path}")
print()


# ============================================================
# 4. Importancia por permutación (Permutation Importance)
# ============================================================

"""
Permutation importance mide cuánto empeora el rendimiento del modelo cuando se
permuta aleatoriamente una feature (gen) en el conjunto de test.

Idea:
    - Si al permutar un gen la AUC cae mucho, ese gen es importante.
    - Si la AUC apenas cambia, ese gen aporta poca información única.

Esto es un método "modelo-agnóstico", ya clásico en ML, y complementa la visión
de los coeficientes.
"""

print("=== IMPORTANCIA POR PERMUTACIÓN (Permutation Importance) ===")

# Usamos como métrica de scoring la AUC, coherente con tu evaluación anterior.
result = permutation_importance(
    logreg,
    X_test_sel,
    y_test,
    n_repeats=30,        # número de permutaciones por gen (puedes ajustar)
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "gene": selected_genes,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

perm_all_path = os.path.join(results_dir, "permutation_importance_logreg_todos_genes.csv")
perm_df.to_csv(perm_all_path, index=False)
print(f"✅ Tabla completa de permutation importance guardada en: {perm_all_path}")

top_perm_df = perm_df.head(top_n)
top_perm_path = os.path.join(results_dir, f"top{top_n}_biomarcadores_logreg_permutation.csv")
top_perm_df.to_csv(top_perm_path, index=False)
print(f"✅ Top {top_n} biomarcadores (por permutation importance) guardado en: {top_perm_path}")
print()


# ============================================================
# 5. Opcional: combinar rankings (coeficientes + permutación)
# ============================================================

"""
Para tener una visión más robusta, combinamos ambos enfoques:

    - rank_abs_coef: ranking según |coeficiente|
    - rank_perm:     ranking según importance_mean

Luego definimos un ranking combinado (promedio de ambos) y
ordenamos los genes según este valor.

Los genes que quedan arriba en ambas métricas son candidatos
fuertes a biomarcadores.
"""

print("=== COMBINANDO RANKINGS (coeficientes + permutación) ===")

coef_rank = coef_df_sorted[["gene", "abs_coef"]].copy()
coef_rank["rank_abs_coef"] = np.arange(1, len(coef_rank) + 1)

perm_rank = perm_df[["gene", "importance_mean"]].copy()
perm_rank["rank_perm"] = np.arange(1, len(perm_rank) + 1)

combined = pd.merge(coef_rank, perm_rank, on="gene", how="inner")
combined["rank_combined"] = (combined["rank_abs_coef"] + combined["rank_perm"]) / 2.0

combined_sorted = combined.sort_values("rank_combined", ascending=True)

combined_all_path = os.path.join(results_dir, "ranking_combinado_logreg_coef_permutation.csv")
combined_sorted.to_csv(combined_all_path, index=False)
print(f"✅ Ranking combinado guardado en: {combined_all_path}")

top_combined_df = combined_sorted.head(top_n)
top_combined_path = os.path.join(results_dir, f"top{top_n}_biomarcadores_logreg_ranking_combinado.csv")
top_combined_df.to_csv(top_combined_path, index=False)
print(f"✅ Top {top_n} biomarcadores (ranking combinado) guardado en: {top_combined_path}")
print()


# ============================================================
# 6. Gráficos: top 20 por coeficientes y por permutation importance
# ============================================================

print("=== GENERANDO GRÁFICOS DE BARRAS PARA TOP 20 GENES ===")

# 6.1. Gráfico top 20 por |coeficiente|
plt.figure(figsize=(10, 6))
plt.barh(top_coef_df["gene"][::-1], top_coef_df["coeficiente"][::-1])
plt.xlabel("Coeficiente (log-odds)")
plt.title("Top 20 genes por |coeficiente| - Regresión logística")
plt.tight_layout()

coef_fig_path = os.path.join(figures_dir, "top20_genes_logreg_coeficientes.png")
plt.savefig(coef_fig_path, dpi=300)
plt.close()
print(f"✅ Figura guardada: {coef_fig_path}")

# 6.2. Gráfico top 20 por permutation importance
plt.figure(figsize=(10, 6))
plt.barh(top_perm_df["gene"][::-1], top_perm_df["importance_mean"][::-1])
plt.xlabel("Permutation importance (ΔAUC medio)")
plt.title("Top 20 genes por permutation importance - Regresión logística")
plt.tight_layout()

perm_fig_path = os.path.join(figures_dir, "top20_genes_logreg_permutation_importance.png")
plt.savefig(perm_fig_path, dpi=300)
plt.close()
print(f"✅ Figura guardada: {perm_fig_path}")

print()
print("✅ Análisis de importancia lineal y biomarcadores completado.")
print("Revisa en la carpeta 'results' las tablas CSV generadas, y en 'figures' los gráficos.")

