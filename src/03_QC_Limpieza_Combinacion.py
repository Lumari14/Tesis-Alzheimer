"""
Script de Control de Calidad (QC) para el dataset GSE5281

Objetivo:
---------
Verificar que los pasos de:
  1) combinación de datos (metadatos + expresión génica)
  2) limpieza de datos (filtro por diagnosis/region y varianza de genes)

se han realizado correctamente y entender:
  - cuántas muestras se pierden en cada paso
  - cuántos genes se eliminan por el filtro de varianza
  - si las clases de diagnosis siguen bien representadas

Este script NO modifica archivos, solo los lee y muestra información.
"""

import os
import pandas as pd

# ============================================================
# 1. Definir rutas de los archivos
# ============================================================

exp_path      = "data/raw/GSE5281_expression_matrix.csv"
meta_path     = "data/raw/GSE5281_sample_metadata.csv"
combined_path = "data/processed/GSE5281_datos_combinados.csv"
clean_path    = "data/processed/GSE5281_datos_limpios.csv"

print("=== RUTAS DE ARCHIVOS ===")
print("Expresión cruda:   ", exp_path)
print("Metadatos crudos:  ", meta_path)
print("Combinado:         ", combined_path)
print("Limpio:            ", clean_path)
print()

# ============================================================
# 2. Cargar datos
#    - df_exp: matriz de expresión
#    - df_meta: metadatos originales
#    - df_comb: tabla combinada (metadatos + expresión)
#    - df_clean: tabla limpia (tras tu script de limpieza)
# ============================================================

df_exp   = pd.read_csv(exp_path, index_col=0)
df_meta  = pd.read_csv(meta_path, index_col=0)
df_comb  = pd.read_csv(combined_path, index_col=0)
df_clean = pd.read_csv(clean_path, index_col=0)

print("=== FORMAS DE LOS DATAFRAMES ===")
print(f"Expresión (df_exp):          {df_exp.shape}  (genes x muestras o viceversa)")
print(f"Metadatos (df_meta):         {df_meta.shape} (muestras x variables)")
print(f"Combinado (df_comb):         {df_comb.shape} (muestras x (metadatos + genes))")
print(f"Limpio (df_clean):           {df_clean.shape} (muestras x (metadatos + genes filtrados))")
print()

# ============================================================
# 3. Comprobar coherencia de IDs entre expresión y metadatos
#    - En df_exp, los IDs de muestra están como columnas
#    - En df_meta, los IDs de muestra están como índice
#    Queremos ver si coinciden.
# ============================================================

meta_ids = set(df_meta.index)
exp_ids  = set(df_exp.columns)

print("=== COINCIDENCIA DE IDS DE MUESTRAS (metadatos vs expresión cruda) ===")
print("Total muestras en metadatos: ", len(meta_ids))
print("Total muestras en expresión: ", len(exp_ids))
print("Muestras solo en metadatos:  ", len(meta_ids - exp_ids))
print("Muestras solo en expresión:  ", len(exp_ids - meta_ids))
print("Muestras en la intersección: ", len(meta_ids & exp_ids))

if meta_ids - exp_ids:
    print("\nEjemplos de IDs solo en METADATOS (máx 5):")
    print(list(meta_ids - exp_ids)[:5])

if exp_ids - meta_ids:
    print("\nEjemplos de IDs solo en EXPRESIÓN (máx 5):")
    print(list(exp_ids - meta_ids)[:5])

print()

# ============================================================
# 4. Revisar la estructura de la tabla combinada
#    Recordatorio: tú creaste df_comb con:
#    df_combined = df_meta[["diagnosis", "region", "age", "sex"]].join(df_exp.transpose())
#    Queremos comprobar:
#       - que esas columnas están presentes
#       - cuántos genes añadiste
# ============================================================

meta_cols = ["diagnosis", "region", "age", "sex"]

print("=== COLUMNAS EN LA TABLA COMBINADA (df_comb) ===")
print("Primeras 10 columnas:", list(df_comb.columns[:10]))
print("¿Están diagnosis/region/age/sex?:",
      all(col in df_comb.columns for col in meta_cols))

n_meta_cols = len(meta_cols)
n_total_cols_comb = df_comb.shape[1]
n_genes_comb = n_total_cols_comb - n_meta_cols

print(f"Número de columnas totales en combinado: {n_total_cols_comb}")
print(f"Número de genes aproximados en combinado: {n_genes_comb}")
print(f"Número de genes en df_exp (filas):       {df_exp.shape[0]}")
print()

# ============================================================
# 5. Reconstruir lógicamente la limpieza desde df_comb
#    Esto es solo para medir impacto, no para guardar nada.
#
#    Tus pasos fueron:
#      (1) dropna en diagnosis y region
#      (2) filtro de varianza < 0.01 en genes
#
#    Vamos a repetir esos pasos sobre df_comb y contar:
#      - cuántas muestras se pierden en (1)
#      - cuántos genes se pierden en (2)
# ============================================================

df_comb_copy = df_comb.copy()

print("=== NAs EN METADATOS (ANTES DE LIMPIEZA) ===")
print(df_comb_copy[meta_cols].isna().sum())
print()

# -------- 5.1. Impacto de dropna en diagnosis y region --------
n_before_samples = df_comb_copy.shape[0]
df_no_na = df_comb_copy.dropna(subset=["diagnosis", "region"])
n_after_samples = df_no_na.shape[0]

print("=== IMPACTO DE DROPNA EN diagnosis/region ===")
print(f"Muestras antes de dropna:        {n_before_samples}")
print(f"Muestras después de dropna:      {n_after_samples}")
print(f"Muestras eliminadas por NAs:     {n_before_samples - n_after_samples}")

if n_before_samples - n_after_samples > 0:
    lost_samples = df_comb_copy.index.difference(df_no_na.index)
    print("Ejemplos de IDs de muestras eliminadas (máx 5):")
    print(list(lost_samples[:5]))
print()

# -------- 5.2. Impacto del filtro de varianza en genes --------
# Solo consideramos como "genes" todas las columnas que no son metadatos.
gene_cols = [c for c in df_no_na.columns if c not in meta_cols]

varianzas = df_no_na[gene_cols].var()
genes_validos    = varianzas[varianzas >= 0.01].index
genes_eliminados = varianzas[varianzas < 0.01].index

print("=== IMPACTO DEL FILTRO DE VARIANZA (umbral = 0.01) ===")
print("Genes totales antes del filtro:   ", len(gene_cols))
print("Genes conservados (var >= 0.01):  ", len(genes_validos))
print("Genes eliminados (var < 0.01):    ", len(genes_eliminados))
if len(gene_cols) > 0:
    print("Porcentaje genes eliminados:      {:.2f}%".format(
        100 * len(genes_eliminados) / len(gene_cols)
    ))
print()

print("Resumen de varianzas de genes CONSERVADOS:")
print(varianzas[genes_validos].describe())
print()

print("Resumen de varianzas de genes ELIMINADOS:")
print(varianzas[genes_eliminados].describe())
print()

# ============================================================
# 6. Comparar directamente el combinado (df_comb) vs el limpio (df_clean)
#    para asegurarnos de que:
#      - el número de muestras coincide con lo esperado
#      - el número de genes es coherente con el filtro de varianza
# ============================================================

print("=== COMPARACIÓN COMBINADO (df_comb) vs LIMPIO (df_clean) ===")
print(f"Muestras en combinado: {df_comb.shape[0]}")
print(f"Muestras en limpio:    {df_clean.shape[0]}")
print()

# Volvemos a definir metadatos y genes, pero ahora en cada tabla:
genes_comb = [c for c in df_comb.columns  if c not in meta_cols]
genes_clean = [c for c in df_clean.columns if c not in meta_cols]

print(f"Genes en combinado: {len(genes_comb)}")
print(f"Genes en limpio:    {len(genes_clean)}")

genes_perdidos = set(genes_comb) - set(genes_clean)
print(f"Genes eliminados según archivo limpio: {len(genes_perdidos)}")
if len(genes_comb) > 0:
    print("Porcentaje genes eliminados (según limpio): {:.2f}%".format(
        100 * len(genes_perdidos) / len(genes_comb)
    ))

# Opcional: mostrar algunos nombres de genes que se eliminaron
if genes_perdidos:
    print("\nEjemplos de genes eliminados (máx 10):")
    print(list(genes_perdidos)[:10])
print()

# ============================================================
# 7. Revisar la distribución de diagnosis en el archivo limpio
#    Esto es crucial para modelos: comprobar que no te has quedado
#    con una clase muy desbalanceada o vacía.
# ============================================================

if "diagnosis" in df_clean.columns:
    print("=== DISTRIBUCIÓN DE DIAGNOSIS EN ARCHIVO LIMPIO (df_clean) ===")
    print(df_clean["diagnosis"].value_counts(dropna=False))
else:
    print("Advertencia: 'diagnosis' no se encontró en df_clean.columns")

print("\n✅ QC completado. Revisa los números impresos arriba para decidir si:")
print("   - el umbral de varianza (0.01) es razonable")
print("   - se han eliminado pocas o muchas muestras por NAs en diagnosis/region")
print("   - la distribución de diagnosis sigue siendo adecuada para ML.")


