"""
12_anotar_probes_a_genes.py

Objetivo:
- Mapear IDs de probes (ej. 238558_at) a símbolos de genes (ej. MAPT, APP, etc.)
  usando la anotación del microarray desde GEO (GPL).
- Generar un archivo de mapeo: results/annotation/probe_to_gene_mapping.csv
- Crear versiones anotadas de tus resultados (CSVs) que tengan columna gene_symbol
  y una columna gene_label (symbol si existe, si no el probe).
- Preparar lista de nombres para re-generar plots (SHAP/coef) con nombres legibles.

Requisitos:
- GEOparse instalado (ya lo usaste para GSE)
- pandas, numpy

Nota:
- Un probe puede mapear a varios genes o ninguno. En esos casos:
  - varios genes -> se concatenan con '|'
  - ninguno -> se deja vacío y gene_label vuelve al probe
"""

import os
import re
import glob
import pandas as pd
import GEOparse

# =========================
# Config
# =========================
GSE_ID = "GSE5281"
OUT_DIR = "results/annotation"
os.makedirs(OUT_DIR, exist_ok=True)

mapping_out = os.path.join(OUT_DIR, "probe_to_gene_mapping.csv")

# Archivos de resultados donde típicamente hay columna 'gene'
RESULT_FILES_GLOB = [
    "results/*genes*.csv",
    "results/*biomarcadores*.csv",
    "results/*ranking*.csv",
    "results/*shap*.csv",
    "results/*permutation*.csv",
    "results/*coef*.csv",
]

# =========================
# 1) Detectar plataforma (GPL) desde GSE
# =========================
print("=== CARGANDO GSE PARA DETECTAR GPL ===")
gse = GEOparse.get_GEO(geo=GSE_ID, destdir="data/raw", how="full")

# GEOparse deja el id de plataforma(s) en gse.gpls
gpl_ids = list(gse.gpls.keys())
if len(gpl_ids) == 0:
    raise RuntimeError("No se pudo detectar GPL desde el GSE.")
print(f"Plataformas detectadas: {gpl_ids}")

# En GSE5281 típicamente es una sola (ej. GPL570)
GPL_ID = gpl_ids[0]
print(f"Usando GPL: {GPL_ID}")

# =========================
# 2) Descargar/leer anotación de GPL
# =========================
print("\n=== CARGANDO ANOTACIÓN GPL ===")
gpl = GEOparse.get_GEO(geo=GPL_ID, destdir="data/raw", how="full")

# Tabla de anotación
gpl_table = gpl.table.copy()
print(f"Anotación GPL filas x cols: {gpl_table.shape}")

# Normalizar nombres de columnas (a veces varían)
cols_lower = {c.lower(): c for c in gpl_table.columns}

# Intentar encontrar columna de símbolo de gen
candidate_cols = [
    "gene symbol", "gene_symbol", "genesymbol", "symbol",
    "gene symbol ", "gene symbol(s)", "gene_symbol(s)"
]
symbol_col = None
for cand in candidate_cols:
    if cand in cols_lower:
        symbol_col = cols_lower[cand]
        break

# Alternativa para GPL570: suele tener "Gene Symbol"
if symbol_col is None:
    for c in gpl_table.columns:
        if "gene" in c.lower() and "symbol" in c.lower():
            symbol_col = c
            break

if symbol_col is None:
    raise RuntimeError(
        "No encontré la columna de símbolo de gen en la anotación GPL. "
        f"Columnas disponibles: {list(gpl_table.columns)[:20]} ..."
    )

print(f"Columna de símbolo de gen detectada: {symbol_col}")

# Columna del ID de probe suele ser "ID"
id_col = "ID" if "ID" in gpl_table.columns else gpl_table.columns[0]
print(f"Columna ID probe detectada: {id_col}")

# =========================
# 3) Construir mapping probe -> gene_symbol(s)
# =========================
print("\n=== CREANDO MAPPING PROBE -> GENE SYMBOL ===")

def clean_symbol(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Normalizar separadores frecuentes
    s = s.replace("///", "|").replace(";", "|").replace(",", "|")
    # Quitar dobles espacios
    s = re.sub(r"\s+", " ", s)
    # Si hay "NA" textual
    if s.upper() in {"NA", "N/A", "---", ""}:
        return ""
    # Quitar espacios alrededor de '|'
    s = "|".join([p.strip() for p in s.split("|") if p.strip()])
    return s

mapping_df = gpl_table[[id_col, symbol_col]].copy()
mapping_df.columns = ["probe_id", "gene_symbol_raw"]
mapping_df["gene_symbol"] = mapping_df["gene_symbol_raw"].apply(clean_symbol)

# Si hay probes repetidos, quedarse con el primero no vacío (o el primero)
mapping_df = mapping_df.sort_values(by=["probe_id", "gene_symbol"], ascending=[True, False])
mapping_df = mapping_df.drop_duplicates(subset=["probe_id"], keep="first")

# Guardar mapping
mapping_df[["probe_id", "gene_symbol"]].to_csv(mapping_out, index=False)
print(f"✅ Mapping guardado en: {mapping_out}")
print(mapping_df.head())

# Diccionario rápido
probe_to_symbol = dict(zip(mapping_df["probe_id"], mapping_df["gene_symbol"]))

# =========================
# 4) Anotar todos los CSVs en results/
# =========================
print("\n=== ANOTANDO ARCHIVOS DE RESULTS ===")

# Reunir archivos únicos
files = []
for pattern in RESULT_FILES_GLOB:
    files.extend(glob.glob(pattern))
files = sorted(set(files))

if not files:
    print("⚠️ No encontré CSVs en results/ con los patrones esperados.")
else:
    print(f"Archivos encontrados: {len(files)}")

os.makedirs("results/annotated", exist_ok=True)

def annotate_gene_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Busca una columna 'gene' (o similar) con probe IDs y añade:
    - gene_symbol
    - gene_label (symbol si existe, si no probe)
    """
    # posibles nombres de columna con probe id
    possible_gene_cols = ["gene", "probe", "probe_id", "feature", "id"]
    gene_col = None
    for c in df.columns:
        if c.lower() in possible_gene_cols:
            gene_col = c
            break
    if gene_col is None:
        # si no, pero hay una columna que parece probes (terminan en _at)
        for c in df.columns:
            if c.lower() == "genes":
                gene_col = c
                break

    if gene_col is None or gene_col not in df.columns:
        return df  # no se puede anotar

    # Mapear
    df = df.copy()
    df["gene_symbol"] = df[gene_col].map(probe_to_symbol).fillna("")
    df["gene_label"] = df["gene_symbol"].where(df["gene_symbol"] != "", df[gene_col])

    # Reordenar columnas si existe gene_col
    # Poner gene_col, gene_symbol, gene_label al inicio si aplica
    cols = list(df.columns)
    for colname in ["gene_label", "gene_symbol"]:
        if colname in cols:
            cols.remove(colname)
    # mover gene_col también al inicio
    cols.remove(gene_col)
    new_cols = [gene_col, "gene_symbol", "gene_label"] + cols
    df = df[new_cols]
    return df

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"⚠️ No pude leer {f}: {e}")
        continue

    df_annot = annotate_gene_column(df)

    base = os.path.basename(f).replace(".csv", "")
    out_f = os.path.join("results/annotated", f"{base}_annotated.csv")
    df_annot.to_csv(out_f, index=False)
    print(f"✅ Anotado: {out_f}")

print("\n✅ Proceso de anotación completado.")
print("Siguiente paso: regenerar plots (SHAP y coeficientes) usando gene_label.")
