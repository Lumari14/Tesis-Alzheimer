import pandas as pd

# Cargar anotación GPL570
annot = pd.read_csv(
    "data/raw/GPL570-55999.txt",
    sep="\t",
    comment="#",
    low_memory=False
)

# Sondas identificadas como biomarcadores
probes = [
    "223272_s_at",
    "215789_s_at",
    "223460_at",
    "227582_at",
    "228027_at",
    "218328_at",
    "224689_at"
]

# Filtrar las sondas de interés
df = annot[annot["ID"].isin(probes)].copy()

# Mostrar nombres completos de los genes
print(df[["ID", "Gene Symbol", "Gene Title"]])


