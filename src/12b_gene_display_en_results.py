import os
import glob
import pandas as pd

MAPPING_PATH = "results/annotation/probe_to_gene_mapping.csv"
IN_DIR = "results/annotated"
OUT_DIR = "results/final_display"
os.makedirs(OUT_DIR, exist_ok=True)

mapping = pd.read_csv(MAPPING_PATH)
probe_to_symbol = dict(zip(mapping["probe_id"], mapping["gene_symbol"]))

def make_gene_display(probe: str) -> str:
    sym = probe_to_symbol.get(probe, "")
    sym = "" if pd.isna(sym) else str(sym).strip()
    probe = str(probe).strip()
    return f"{sym} ({probe})" if sym else probe

files = sorted(glob.glob(os.path.join(IN_DIR, "*_annotated.csv")))
if not files:
    raise FileNotFoundError(f"No encontré archivos en {IN_DIR}. ¿Existe y tiene *_annotated.csv?")

print(f"Archivos a procesar: {len(files)}")

for f in files:
    df = pd.read_csv(f)

    # Detectar columna de probe en tus resultados (normalmente 'gene')
    probe_col = None
    for c in df.columns:
        if c.lower() in {"gene", "probe", "probe_id", "feature", "id"}:
            probe_col = c
            break

    if probe_col is None:
        # si no hay columna de probe, copiamos tal cual
        out = os.path.join(OUT_DIR, os.path.basename(f).replace("_annotated", "_display"))
        df.to_csv(out, index=False)
        print(f"⚠️ Sin columna probe detectada en {os.path.basename(f)}. Copiado sin cambios.")
        continue

    df["gene_display"] = df[probe_col].apply(make_gene_display)

    # Reordenar columnas para que sea cómodo en tesis
    cols = list(df.columns)
    cols.remove("gene_display")
    # deja gene_display justo después del probe_col
    probe_idx = cols.index(probe_col)
    cols = cols[:probe_idx+1] + ["gene_display"] + cols[probe_idx+1:]
    df = df[cols]

    out = os.path.join(OUT_DIR, os.path.basename(f).replace("_annotated", "_display"))
    df.to_csv(out, index=False)
    print(f"✅ Guardado: {out}")

print("\n✅ Listo. Usa results/final_display/*_display.csv para tablas y plots.")
