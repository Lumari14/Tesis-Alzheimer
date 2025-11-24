import pandas as pd

import os

# Rutas a los archivos
exp_path = "data/raw/GSE5281_expression_matrix.csv"
meta_path = "data/raw/GSE5281_sample_metadata.csv"

# Leer archivos
df_exp = pd.read_csv(exp_path, index_col=0)
df_meta = pd.read_csv(meta_path, index_col=0)

# Extraer columnas clave de metadatos
df_meta["diagnosis"] = df_meta["characteristics_ch1.8.Disease State"].str.replace("Disease State: ", "", regex=False)
df_meta["region"] = df_meta["characteristics_ch1.4.Organ Region"].str.replace("Organ Region: ", "", regex=False)
df_meta["age"] = df_meta["characteristics_ch1.11.age"].str.replace("age: ", "", regex=False)
df_meta["sex"] = df_meta["characteristics_ch1.9.sex"].str.replace("sex: ", "", regex=False)

# Combinar metadatos con expresión génica (trasponer df_exp para que las muestras estén como filas)
df_combined = df_meta[["diagnosis", "region", "age", "sex"]].join(df_exp.transpose())

# Crear carpeta para resultados si no existe
os.makedirs("data/processed", exist_ok=True)

# Guardar tabla combinada
df_combined.to_csv("data/processed/GSE5281_datos_combinados.csv")

print("✅ Tabla combinada guardada en: data/processed/GSE5281_datos_combinados.csv")
