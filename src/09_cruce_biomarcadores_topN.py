"""
Script: 09_cruce_biomarcadores_topN.py

Objetivo
--------
Repetir el cruce de biomarcadores entre:

1) Modelo lineal (regresión logística) con ranking combinado:
      - Archivo: results/ranking_combinado_logreg_coef_permutation.csv
      - Columnas esperadas: gene, abs_coef, importance_mean, rank_abs_coef, rank_perm, rank_combined

2) Modelo no lineal (Random Forest) interpretado con SHAP:
      - Archivo: results/shap_importancia_rf_todos_genes.csv
      - Columnas esperadas: gene, mean_abs_shap

Para diferentes tamaños de top N (por ejemplo N=50, N=100), el script:

    - Extrae el top N de cada ranking.
    - Calcula:
         * Genes comunes (intersección).
         * Genes exclusivos de logreg.
         * Genes exclusivos de SHAP.
    - Guarda tablas CSV con estos grupos.
    - Genera un gráfico comparando importancia lineal vs SHAP para los
      genes comunes (si existen).

Esto permite estudiar si, ampliando el rango a top 50/top 100,
aparecen biomarcadores robustos compartidos entre ambos enfoques.
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Rutas y carga de rankings completos
# ============================================================

results_dir = "results"
figures_dir = "figures"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=== CARGANDO RANKINGS COMPLETOS ===")

# Ranking combinado completo de regresión logística
logreg_full_path = os.path.join(
    results_dir, "ranking_combinado_logreg_coef_permutation.csv"
)

# Ranking completo de importancia SHAP (Random Forest)
shap_full_path = os.path.join(
    results_dir, "shap_importancia_rf_todos_genes.csv"
)

logreg_full = pd.read_csv(logreg_full_path)
shap_full   = pd.read_csv(shap_full_path)

print(f"Ranking completo (logreg) cargado desde: {logreg_full_path}")
print(f"Ranking completo (SHAP RF) cargado desde: {shap_full_path}")
print()
print("Columnas en ranking logreg:", logreg_full.columns.tolist())
print("Columnas en ranking SHAP:  ", shap_full.columns.tolist())
print()

# Asegurarnos de que están ordenados por ranking de importancia
# Para logreg: rank_combined ya refleja el ranking global (1 = más importante)
logreg_full_sorted = logreg_full.sort_values("rank_combined", ascending=True)

# Para SHAP: mean_abs_shap mayor = más importante
shap_full_sorted = shap_full.sort_values("mean_abs_shap", ascending=False)


# ============================================================
# 2. Función auxiliar para cruzar top N
# ============================================================

def cruce_para_topN(N: int):
    """
    Para un valor N dado:

    - Toma los top N genes de:
         * ranking combinado de logreg
         * ranking SHAP de RF
    - Calcula genes comunes y exclusivos.
    - Guarda tablas CSV.
    - Genera un gráfico comparativo de importancia para genes comunes.
    """

    print("=" * 60)
    print(f"=== CRUCE PARA TOP {N} GENES ===")

    # 2.1. Extraer top N
    logreg_topN = logreg_full_sorted.head(N).copy()
    shap_topN   = shap_full_sorted.head(N).copy()

    print(f"Top {N} logreg (primeras filas):")
    print(logreg_topN.head())
    print()
    print(f"Top {N} SHAP (primeras filas):")
    print(shap_topN.head())
    print()

    # 2.2. Intersección de genes
    intersection_df = pd.merge(
        logreg_topN,
        shap_topN,
        on="gene",
        how="inner",
        suffixes=("_logreg", "_shap")
    )

    print(f"Número de genes en top {N} logreg: {logreg_topN.shape[0]}")
    print(f"Número de genes en top {N} SHAP:   {shap_topN.shape[0]}")
    print(f"Número de genes comunes (top {N}): {intersection_df.shape[0]}")
    print()

    # 2.3. Guardar tablas

    # Intersección
    inter_path = os.path.join(
        results_dir, f"biomarcadores_interseccion_logreg_shap_top{N}.csv"
    )
    intersection_df.to_csv(inter_path, index=False)
    print(f"✅ Tabla de genes comunes (top{N}) guardada en: {inter_path}")

    # Exclusivos -> primero marcamos origen
    logreg_topN_copy = logreg_topN.copy()
    shap_topN_copy   = shap_topN.copy()
    logreg_topN_copy["source"] = "logreg"
    shap_topN_copy["source"]   = "shap"

    union_df = pd.merge(
        logreg_topN_copy[["gene", "source"]],
        shap_topN_copy[["gene", "source"]],
        on="gene",
        how="outer",
        suffixes=("_logreg", "_shap")
    )

    union_df["in_logreg"] = ~union_df["source_logreg"].isna()
    union_df["in_shap"]   = ~union_df["source_shap"].isna()

    solo_logreg_df = union_df[
        (union_df["in_logreg"]) & (~union_df["in_shap"])
    ].copy()

    solo_shap_df = union_df[
        (~union_df["in_logreg"]) & (union_df["in_shap"])
    ].copy()

    solo_logreg_path = os.path.join(
        results_dir, f"biomarcadores_solo_logreg_top{N}.csv"
    )
    solo_shap_path = os.path.join(
        results_dir, f"biomarcadores_solo_shap_top{N}.csv"
    )

    solo_logreg_df.to_csv(solo_logreg_path, index=False)
    solo_shap_df.to_csv(solo_shap_path, index=False)

    print(f"✅ Genes exclusivos de logreg (top{N}) guardados en: {solo_logreg_path}")
    print(f"✅ Genes exclusivos de SHAP   (top{N}) guardados en: {solo_shap_path}")
    print()

    # 2.4. Gráfico comparativo para genes comunes (si los hay)
    if intersection_df.shape[0] > 0:
        print(f"=== GENERANDO GRÁFICO COMPARATIVO PARA GENES COMUNES (top{N}) ===")

        # Queremos comparar:
        #  - Importancia lineal: abs_coef (logreg)
        #  - Importancia SHAP: mean_abs_shap (RF)
        # Ya las tenemos en intersection_df gracias al merge.

        plot_df = intersection_df.sort_values(
            "mean_abs_shap", ascending=True
        )

        plt.figure(figsize=(10, 6))

        y_pos = range(plot_df.shape[0])

        # Barras horizontales: abs_coef
        plt.barh(
            y_pos,
            plot_df["abs_coef"],
            alpha=0.7,
            label="Importancia lineal (|coef|)"
        )

        # Superponemos importance SHAP (podría estar en distinta escala,
        # pero lo indicamos en la etiqueta; si hiciera falta, se puede
        # normalizar en un paso posterior).
        plt.barh(
            y_pos,
            plot_df["mean_abs_shap"],
            alpha=0.7,
            label="Importancia SHAP (media |SHAP|)"
        )

        plt.yticks(y_pos, plot_df["gene"])
        plt.xlabel("Valor de importancia (escalas no normalizadas)")
        plt.title(
            f"Genes comunes en top{N} (logreg + SHAP)\n"
            "Comparación de importancia lineal vs no lineal"
        )
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(
            figures_dir,
            f"comparacion_importancia_genes_comunes_logreg_shap_top{N}.png"
        )
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"✅ Figura comparativa (top{N}) guardada en: {fig_path}")
        print()
    else:
        print(f"⚠️ No hay genes comunes para top{N}. No se genera gráfico comparativo.")
        print()


# ============================================================
# 3. Ejecutar cruce para N = 50 y N = 100
# ============================================================

if __name__ == "__main__":
    for N in [50, 100]:
        cruce_para_topN(N)

    print("✅ Cruce de biomarcadores para top50 y top100 completado.")
    print("Revisa la carpeta 'results' para las tablas y 'figures' para los gráficos.")
