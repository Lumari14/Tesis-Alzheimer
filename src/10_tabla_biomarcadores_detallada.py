"""
Script: 10_tabla_biomarcadores_detallada.py

Objetivo
--------
Construir una tabla detallada con los genes que aparecen:

    - En la intersección de los top 50
    - En la intersección de los top 100

entre:

    1) Modelo lineal (regresión logística) con ranking combinado
       -> ranking_combinado_logreg_coef_permutation.csv

    2) Modelo no lineal (Random Forest) interpretado con SHAP
       -> shap_importancia_rf_todos_genes.csv

y enriquecer esa tabla con:

    - coeficiente (signed)
    - abs_coef
    - odds_ratio
    - rank_combined (ranking lineal)
    - mean_abs_shap (importancia SHAP)
    - rank_shap (ranking SHAP)
    - flags indicando si el gen pertenece a la intersección top50 y/o top100

Salida:
    - results/biomarcadores_detallados_top50_top100.csv
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. Definir rutas y cargar datos base
# ============================================================

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

print("=== CARGANDO ARCHIVOS BASE ===")

# Ranking combinado de regresión logística (todos los genes)
logreg_ranking_path = os.path.join(
    results_dir, "ranking_combinado_logreg_coef_permutation.csv"
)

# Coeficientes completos de la regresión logística (incluye coeficiente y OR)
logreg_coef_path = os.path.join(
    results_dir, "coef_importancia_logreg_todos_genes.csv"
)

# Importancia SHAP global (todos los genes)
shap_importance_path = os.path.join(
    results_dir, "shap_importancia_rf_todos_genes.csv"
)

# Intersecciones top50 y top100
inter_top50_path = os.path.join(
    results_dir, "biomarcadores_interseccion_logreg_shap_top50.csv"
)
inter_top100_path = os.path.join(
    results_dir, "biomarcadores_interseccion_logreg_shap_top100.csv"
)

# Cargar dataframes
logreg_ranking_df = pd.read_csv(logreg_ranking_path)
logreg_coef_df    = pd.read_csv(logreg_coef_path)
shap_importance_df = pd.read_csv(shap_importance_path)

# Intersecciones (pueden estar vacías, pero en tu caso top50 tiene 1 y top100 tiene 7)
inter_top50_df  = pd.read_csv(inter_top50_path)
inter_top100_df = pd.read_csv(inter_top100_path)

print(f"Ranking lineal cargado desde: {logreg_ranking_path}")
print(f"Coeficientes logreg cargados desde: {logreg_coef_path}")
print(f"Importancia SHAP cargada desde: {shap_importance_path}")
print(f"Intersección top50 cargada desde: {inter_top50_path}")
print(f"Intersección top100 cargada desde: {inter_top100_path}")
print()

print("Forma ranking logreg:", logreg_ranking_df.shape)
print("Forma coeficientes logreg:", logreg_coef_df.shape)
print("Forma shap_importance:", shap_importance_df.shape)
print("Forma inter_top50:", inter_top50_df.shape)
print("Forma inter_top100:", inter_top100_df.shape)
print()


# ============================================================
# 2. Construir conjunto de genes de interés (top50 ∪ top100)
# ============================================================

"""
Queremos una tabla que contenga:

    - El gen común de la intersección top50 (si existe).
    - Los genes de la intersección top100 (7 en tu caso).

Es decir, la unión de ambos conjuntos.
"""

print("=== CONSTRUYENDO LISTA DE GENES DE INTERÉS ===")

genes_top50 = set(inter_top50_df["gene"].tolist()) if not inter_top50_df.empty else set()
genes_top100 = set(inter_top100_df["gene"].tolist()) if not inter_top100_df.empty else set()

genes_union = sorted(genes_top50.union(genes_top100))

print(f"Número de genes en intersección top50:  {len(genes_top50)}")
print(f"Número de genes en intersección top100: {len(genes_top100)}")
print(f"Número total de genes únicos (union):   {len(genes_union)}")
print("Genes de interés:", genes_union)
print()


# ============================================================
# 3. Preparar rankings completos: añadir rank_shap
# ============================================================

"""
En logreg_ranking_df ya tienes:

    - gene
    - abs_coef
    - rank_abs_coef
    - importance_mean (permutation importance)
    - rank_perm
    - rank_combined

En logreg_coef_df tienes:

    - gene
    - coeficiente (signed)
    - odds_ratio
    - abs_coef

En shap_importance_df tienes:

    - gene
    - mean_abs_shap

Ahora queremos añadir un ranking SHAP explícito:

    - rank_shap: 1 = gen con mayor mean_abs_shap
"""

print("=== PREPARANDO RANKINGS COMPLETOS (añadiendo rank_shap) ===")

# Ordenar por mean_abs_shap descendente y asignar ranking
shap_importance_df = shap_importance_df.sort_values(
    "mean_abs_shap", ascending=False
).reset_index(drop=True)

shap_importance_df["rank_shap"] = np.arange(1, shap_importance_df.shape[0] + 1)

print("Primeras filas shap_importance_df con rank_shap:")
print(shap_importance_df.head())
print()


# ============================================================
# 4. Filtrar solo los genes de interés y fusionar información
# ============================================================

"""
Vamos a construir una tabla final que contenga, para cada gen de interés:

    - gene
    - coeficiente (logreg)
    - abs_coef (logreg)
    - odds_ratio (logreg)
    - importance_mean (permutation importance de logreg)
    - rank_combined (ranking lineal global)
    - mean_abs_shap (RF + SHAP)
    - rank_shap (ranking SHAP global)
    - in_top50_intersection (True/False)
    - in_top100_intersection (True/False)
"""

print("=== FUSIONANDO INFORMACIÓN PARA GENES DE INTERÉS ===")

# Logreg: ranking + coeficientes (unimos por gene)
logreg_merged = pd.merge(
    logreg_ranking_df,
    logreg_coef_df[["gene", "coeficiente", "odds_ratio"]],
    on="gene",
    how="left"
)

# SHAP: ya tiene gene, mean_abs_shap, rank_shap
shap_merged = shap_importance_df[["gene", "mean_abs_shap", "rank_shap"]].copy()

# Filtrar solo genes de interés en cada tabla
logreg_subset = logreg_merged[logreg_merged["gene"].isin(genes_union)].copy()
shap_subset   = shap_merged[shap_merged["gene"].isin(genes_union)].copy()

# Mezclar logreg_subset y shap_subset por gene
final_df = pd.merge(
    logreg_subset,
    shap_subset,
    on="gene",
    how="outer"  # outer por seguridad, pero en principio debería ser inner
)

# Añadir flags de pertenencia a intersecciones
final_df["in_top50_intersection"]  = final_df["gene"].isin(genes_top50)
final_df["in_top100_intersection"] = final_df["gene"].isin(genes_top100)

# Ordenar la tabla final por rank_combined (lineal) y luego por rank_shap
final_df = final_df.sort_values(
    ["in_top50_intersection", "in_top100_intersection", "rank_combined", "rank_shap"],
    ascending=[False, False, True, True]
)

print("Tabla final (primeras filas):")
print(final_df.head())
print()


# ============================================================
# 5. Guardar tabla final
# ============================================================

output_path = os.path.join(
    results_dir, "biomarcadores_detallados_top50_top100.csv"
)

final_df.to_csv(output_path, index=False)
print(f"✅ Tabla detallada de biomarcadores guardada en: {output_path}")

print("\nColumnas de la tabla final:")
print(final_df.columns.tolist())

print("\n✅ Proceso completado. Ahora puedes abrir el CSV en VS Code o Excel\n"
      "   para ver, para cada gen candidato:\n"
      "   - coeficiente y odds_ratio (modelo lineal)\n"
      "   - importancia por permutation y ranking combinado\n"
      "   - importancia SHAP y ranking SHAP\n"
      "   - flags de pertenencia a intersección top50/top100.")

