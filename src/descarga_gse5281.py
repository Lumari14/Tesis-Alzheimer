import os
import GEOparse

# Crear carpeta 'data/raw' si no existe
os.makedirs("data/raw", exist_ok=True)

# Descargar el dataset GSE5281 desde GEO
print("Descargando GSE5281...")
gse = GEOparse.get_GEO(geo="GSE5281", destdir="data/raw", annotate_gpl=True)

# Guardar resumen de las muestras
print("Guardando metadatos de muestras...")
gse.phenotype_data.to_csv("data/raw/GSE5281_sample_metadata.csv")

# Guardar tabla de expresión génica
print("Guardando tabla de expresión génica...")
gse.pivot_samples('VALUE').to_csv("data/raw/GSE5281_expression_matrix.csv")

print("Descarga y guardado completados.")
