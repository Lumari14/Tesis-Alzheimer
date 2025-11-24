"""
Script: 11_anotacion_biomarcadores.py

Objetivo
--------
Anotar los biomarcadores robustos (sondas Affymetrix) con información biológica:

    - Símbolo de gen (symbol)
    - Nombre del gen (name)
    - ID Entrez (entrezgene)
    - ID(s) Ensembl (ensembl.gene)
    - Resumen biológico (summary)

Fuente de datos:
    - Entrada:  results/biomarcadores_detallados_top50_top100.csv
    - Salida:   results/biomarcadores_detallados_top50_top100_annotated.csv

La anotación se hace usando la API de MyGene.info, que permite mapear
probesets de Affymetrix (IDs tipo '223272_s_at') a genes humanos.

REQUISITOS:
    - Tener instalada la librería 'mygene'  -> pip install mygene
    - Tener conexión a internet
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import numpy as np
import pandas as pd

from mygene import MyGeneInfo


# ============================================================
# 1. Cargar tabla de biomarcadores detallados
# ============================================================

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

input_path = os.path.join(
    results_dir, "biomarcadores_detallados_top50_top100.csv"
)

print("=== CARGANDO TABLA DE BIOMARCADORES DETALLADOS ===")
print(f"Leyendo archivo: {input_path}")

biomarkers_df = pd.read_csv(input_path)

print("Forma de la tabla:", biomarkers_df.shape)
print("Columnas disponibles:")
print(biomarkers_df.columns.tolist())
print()

# Lista de sondas (probes Affymetrix) a anotar
probe_ids = sorted(biomarkers_df["gene"].unique())
print(f"Número de sondas únicas a anotar: {len(probe_ids)}")
print("Probes:", probe_ids)
print()


# ============================================================
# 2. Consultar MyGene.info para anotar las sondas
# ============================================================

"""
MyGene.info permite buscar por:

    - 'reporter'  -> Affymetrix probeset id (ej. 223272_s_at)

Vamos a usar:
    scopes = 'reporter'
    fields = symbol, name, entrezgene, ensembl.gene, summary
    species = 'human'
"""

print("=== CONSULTANDO MyGene.info PARA ANOTAR PROBES ===")

mg = MyGeneInfo()

# Hacemos la consulta masiva (querymany)
# Nota: si hubiera muchas sondas, se podría hacer por bloques; aquí son pocas.
annot_results = mg.querymany(
    probe_ids,
    scopes="reporter",  # ID de probeset de Affymetrix
    fields="symbol,name,entrezgene,ensembl.gene,summary",
    species="human"
)

print(f"Número de resultados devueltos por MyGene.info: {len(annot_results)}")
print("Ejemplo de resultado (primer elemento):")
if len(annot_results) > 0:
    print(annot_results[0])
print()

# Convertimos la lista de diccionarios a DataFrame
annot_df = pd.DataFrame(annot_results)

print("Forma de annot_df original:", annot_df.shape)
print("Columnas de annot_df:")
print(annot_df.columns.tolist())
print()

# 'query' contiene el ID original usado (la sonda/probe)
# Puede haber filas duplicadas si un probe mapea a varios genes.
# Nos quedamos con la primera anotación por cada 'query' por simplicidad.
annot_df = annot_df.drop_duplicates(subset="query", keep="first")

print("Forma de annot_df tras eliminar duplicados por 'query':", annot_df.shape)
print()


# ============================================================
# 3. Limpieza/transformación de algunas columnas de anotación
# ============================================================

"""
La columna 'ensembl' puede ser:
    - un dict con la key 'gene'
    - una lista de dicts
Lo vamos a transformar en una cadena con uno o varios IDs Ensembl.

También renombraremos 'query' a 'gene' para poder fusionar fácilmente.
"""

def extraer_ensembl_gene(value):
    """Extrae IDs Ensembl de la estructura devuelta por MyGene.

    Posibles formatos:
        - dict: {'gene': 'ENSG000001...'}
        - list de dicts: [{'gene': 'ENSG...'}, {'gene': 'ENSG...'}, ...]
        - NaN / None

    Devuelve:
        - string con IDs separados por ';' si hay varios
        - None si no hay información
    """
    if isinstance(value, dict):
        # Un solo diccionario
        gene_id = value.get("gene")
        return gene_id
    elif isinstance(value, list):
        # Lista de diccionarios
        ids = []
        for item in value:
            if isinstance(item, dict) and "gene" in item:
                ids.append(item["gene"])
        if ids:
            return ";".join(ids)
        else:
            return None
    else:
        # Nada útil
        return None

# Si existe la columna 'ensembl', la transformamos
if "ensembl" in annot_df.columns:
    annot_df["ensembl_gene_ids"] = annot_df["ensembl"].apply(extraer_ensembl_gene)
else:
    annot_df["ensembl_gene_ids"] = None

# Renombramos 'query' -> 'gene' para poder hacer merge
annot_df = annot_df.rename(columns={"query": "gene"})

# Nos quedamos solo con las columnas que nos interesan
annot_df = annot_df[
    ["gene", "symbol", "name", "entrezgene", "ensembl_gene_ids", "summary", "notfound"]
    if "notfound" in annot_df.columns
    else ["gene", "symbol", "name", "entrezgene", "ensembl_gene_ids", "summary"]
]

print("annot_df (después de limpieza) - primeras filas:")
print(annot_df.head())
print()


# ============================================================
# 4. Fusionar anotación con la tabla de biomarcadores
# ============================================================

"""
Fusionamos 'biomarkers_df' (toda la info de modelos) con 'annot_df' (info biológica)
usando la columna 'gene' (ID de sonda Affymetrix).

La tabla final contendrá:

    - Medidas de importancia lineal: coeficiente, odds_ratio, rank_combined, etc.
    - Medidas de importancia no lineal: mean_abs_shap, rank_shap.
    - Flags: in_top50_intersection, in_top100_intersection.
    - Anotación: symbol, name, entrezgene, ensembl_gene_ids, summary.
"""

print("=== FUSIONANDO ANOTACIÓN CON BIOMARCADORES ===")

final_annotated_df = pd.merge(
    biomarkers_df,
    annot_df,
    on="gene",
    how="left"   # left: mantenemos todos los genes del set de biomarcadores
)

print("Forma de la tabla final anotada:", final_annotated_df.shape)
print("Columnas de la tabla final anotada:")
print(final_annotated_df.columns.tolist())
print()

print("Primeras filas de la tabla final anotada:")
print(final_annotated_df.head())
print()


# ============================================================
# 5. Guardar tabla final
# ============================================================

output_path = os.path.join(
    results_dir, "biomarcadores_detallados_top50_top100_annotated.csv"
)

final_annotated_df.to_csv(output_path, index=False)

print(f"✅ Tabla anotada guardada en: {output_path}")
print("\nAhora puedes abrir este CSV en VS Code o Excel para ver, para cada sonda:")
print("   - información del modelo lineal (coeficiente, OR, ranking)")
print("   - información SHAP (importancia no lineal)")
print("   - símbolo de gen, nombre, IDs Entrez/Ensembl y resumen biológico.")
