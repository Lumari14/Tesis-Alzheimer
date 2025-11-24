import pandas as pd
import numpy as np
import os

# Cargar datos combinados
df = pd.read_csv("data/processed/GSE5281_datos_combinados.csv", index_col=0)
print(f"Forma original: {df.shape}")

# Paso 1: Filtrar muestras que tengan diagnosis y region
df = df.dropna(subset=["diagnosis", "region"])
print(f"Después de filtrar por diagnosis y region: {df.shape}")

# Convertir age a numérico (si existe)
if "age" in df.columns:
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Separar metadatos y datos de expresión
meta_cols = ["diagnosis", "region", "age", "sex"]
gene_cols = [col for col in df.columns if col not in meta_cols]

# Paso 2: Filtrar genes con varianza baja (< 0.01)
varianzas = df[gene_cols].var()
genes_validos = varianzas[varianzas >= 0.01].index
df_filtrado = pd.concat([df[meta_cols], df[genes_validos]], axis=1)

print(f"Después de filtrar genes con varianza baja: {df_filtrado.shape}")

# Guardar archivo limpio
os.makedirs("data/processed", exist_ok=True)
df_filtrado.to_csv("data/processed/GSE5281_datos_limpios.csv")

print("✅ Archivo guardado: data/processed/GSE5281_datos_limpios.csv")
