"""
Script: 08_cruce_biomarcadores.py

Objetivo
--------
Cruzar los genes más importantes identificados por:

1) Modelo lineal (regresión logística) con ranking combinado:
      - Archivo: results/top20_biomarcadores_logreg_ranking_combinado.csv
      - Contiene columnas como: gene, abs_coef, importance_mean, rank_abs_coef, rank_perm, rank_combined

2) Modelo no lineal (Random Forest) interpretado con SHAP:
      - Archivo: results/top20_genes_shap_rf.csv
      - Contiene columnas: gene, mean_abs_shap

Este script:
    - Identifica genes que aparecen en ambos rankings (intersección).
    - Identifica genes exclusivos de cada método.
    - Guarda tablas resumen en la carpeta 'results'.
    - Genera un gráfico comparando la importancia lineal vs SHAP
      para los genes comunes (potenciales biomarcadores "fuertes").
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Definir rutas y cargar archivos
# ============================================================

results_dir = "results"
figures_dir = "figures"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=== CARGANDO RANKINGS DE BIOMARCADORES ===")

# Archivo con top 20 biomarcadores según ranking combinado (logística)
logreg_combined_path = os.path.join(
    results_dir, "top20_biomarcadores_logreg_ranking_combinado.csv"
)

# Archivo con top 20 genes más importantes según SHAP (Random Forest)
shap_top_path = os.path.join(
    results_dir, "top20_genes_shap_rf.csv"
)

# Cargar dataframes
logreg_df = pd.read_csv(logreg_combined_path)
shap_df   = pd.read_csv(shap_top_path)

print(f"Cargado ranking combinado de regresión logística desde: {logreg_combined_path}")
print(f"Cargado ranking SHAP de Random Forest desde:           {shap_top_path}")
print()

print("Primeras filas - ranking combinado (logística):")
print(logreg_df.head())
print()
print("Primeras filas - ranking SHAP (Random Forest):")
print(shap_df.head())
print()


# ============================================================
# 2. Intersección de genes (presentes en ambos enfoques)
# ============================================================

"""
La intersección nos da genes que:

- Son importantes según el modelo lineal (coeficientes + permutation).
- Son importantes según el modelo no lineal (Random Forest + SHAP).

Estos genes son candidatos a biomarcadores "robustos" porque aparecen
destacados desde dos aproximaciones distintas.
"""

print("=== CALCULANDO INTERSECCIÓN DE GENES ===")

# Unir por columna 'gene'
intersection_df = pd.merge(
    logreg_df,
    shap_df,
    on="gene",
    how="inner",
    suffixes=("_logreg", "_shap")
)

print(f"Número de genes en ranking combinado (logística): {logreg_df.shape[0]}")
print(f"Número de genes en ranking SHAP (RF):            {shap_df.shape[0]}")
print(f"Número de genes comunes a ambos rankings:        {intersection_df.shape[0]}")
print()

# Guardar la tabla de intersección
intersection_path = os.path.join(
    results_dir, "biomarcadores_interseccion_logreg_shap.csv"
)
intersection_df.to_csv(intersection_path, index=False)
print(f"✅ Tabla de genes comunes guardada en: {intersection_path}")
print()


# ============================================================
# 3. Genes exclusivos de cada enfoque
# ============================================================

"""
También es útil ver los genes que:

- Solo aparecen en el top 20 de la regresión logística.
- Solo aparecen en el top 20 de SHAP/Random Forest.

Esto muestra posibles diferencias entre lo que captan modelos lineales
y no lineales.
"""

print("=== CALCULANDO GENES EXCLUSIVOS DE CADA MÉTODO ===")

# Añadimos una marca de fuente a cada tabla para facilitar el merge
logreg_df["source"] = "logreg"
shap_df["source"]   = "shap"

# Merge externo para ver todo el universo de genes (unión)
union_df = pd.merge(
    logreg_df[["gene", "source"]],
    shap_df[["gene", "source"]],
    on="gene",
    how="outer",
    suffixes=("_logreg", "_shap")
)

# Crear columnas booleanas para indicar presencia en cada ranking
union_df["in_logreg_top20"] = ~union_df["source_logreg"].isna()
union_df["in_shap_top20"]   = ~union_df["source_shap"].isna()

# Genes solo en logreg (top20)
solo_logreg_df = union_df[
    (union_df["in_logreg_top20"]) & (~union_df["in_shap_top20"])
].copy()

# Genes solo en shap (top20)
solo_shap_df = union_df[
    (~union_df["in_logreg_top20"]) & (union_df["in_shap_top20"])
].copy()

# Guardar
solo_logreg_path = os.path.join(
    results_dir, "biomarcadores_solo_logreg_top20.csv"
)
solo_shap_path = os.path.join(
    results_dir, "biomarcadores_solo_shap_top20.csv"
)

solo_logreg_df.to_csv(solo_logreg_path, index=False)
solo_shap_df.to_csv(solo_shap_path, index=False)

print(f"✅ Genes exclusivos de logreg (top20) guardados en: {solo_logreg_path}")
print(f"✅ Genes exclusivos de SHAP (top20) guardados en:   {solo_shap_path}")
print()


# ============================================================
# 4. Gráfico de comparación para genes comunes
# ============================================================

"""
Para los genes que aparecen en ambos rankings, generamos un gráfico de barras
comparando:

- Importancia lineal (abs_coef)
- Importancia SHAP (mean_abs_shap)

Esto te da una figura muy útil para el TFM, porque muestra cómo ambos enfoques
coinciden (o no) en la magnitud de importancia de cada gen.
"""

if intersection_df.shape[0] > 0:
    print("=== GENERANDO GRÁFICO DE BARRAS PARA GENES COMUNES ===")

    # Ordenamos por importancia SHAP (por ejemplo) para un gráfico más bonito
    plot_df = intersection_df.sort_values(
        "mean_abs_shap", ascending=True
    )  # ascending=True para que el más importante quede arriba en el barh

    # Crear una figura con dos barras por gen (abs_coef vs mean_abs_shap)
    plt.figure(figsize=(10, 6))

    # Posiciones para las barras
    y_pos = range(plot_df.shape[0])

    # Escalamos opcionalmente las métricas si estuvieran en escalas muy distintas
    # Por simplicidad, aquí las dejamos tal cual y lo indicamos en el eje X.
    plt.barh(
        y_pos,
        plot_df["abs_coef"],
        alpha=0.7,
        label="Importancia lineal (|coef|)"
    )
    plt.barh(
        y_pos,
        plot_df["mean_abs_shap"],
        alpha=0.7,
        label="Importancia SHAP (media |SHAP|)"
    )

    plt.yticks(y_pos, plot_df["gene"])
    plt.xlabel("Valor de importancia (escala no normalizada)")
    plt.title("Genes comunes en top20 (logreg + SHAP)\nComparación de importancia lineal vs no lineal")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(
        figures_dir, "comparacion_importancia_genes_comunes_logreg_shap.png"
    )
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"✅ Figura comparativa guardada en: {fig_path}")
    print()
else:
    print("⚠️ No hay genes comunes entre los top 20 de logreg y SHAP.")
    print("   No se generará gráfico comparativo.")
    print()


# ============================================================
# 5. Mensaje final
# ============================================================

print("✅ Cruce de biomarcadores completado.")
print("Archivos generados en 'results':")
print("   - biomarcadores_interseccion_logreg_shap.csv")
print("   - biomarcadores_solo_logreg_top20.csv")
print("   - biomarcadores_solo_shap_top20.csv")
print("Y una figura comparativa (si hay intersección) en 'figures':")
print("   - comparacion_importancia_genes_comunes_logreg_shap.png")
