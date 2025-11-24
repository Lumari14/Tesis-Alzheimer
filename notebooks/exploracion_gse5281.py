import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta de resultados si no existe
os.makedirs("results", exist_ok=True)

# Cargar metadatos de las muestras
metadata_path = "data/raw/GSE5281_sample_metadata.csv"
df_meta = pd.read_csv(metadata_path, index_col=0)

# Verificar nombres de columnas si es necesario
# print(df_meta.columns.tolist())

# Extraer diagnóstico y región cerebral si existen las columnas esperadas
if "characteristics_ch1.8.Disease State" in df_meta.columns:
    df_meta["diagnosis"] = df_meta["characteristics_ch1.8.Disease State"].str.replace("Disease State: ", "")

if "characteristics_ch1.4.Organ Region" in df_meta.columns:
    df_meta["region"] = df_meta["characteristics_ch1.4.Organ Region"].str.replace("Organ Region: ", "")

# Verifica que se extrajeron correctamente
print("Diagnósticos únicos:", df_meta["diagnosis"].unique())
print("Regiones únicas:", df_meta["region"].unique())

# Graficar diagnóstico
plt.figure(figsize=(6, 4))
sns.countplot(x="diagnosis", data=df_meta)
plt.title("Distribución de muestras por diagnóstico")
plt.xlabel("Grupo")
plt.ylabel("Número de muestras")
plt.tight_layout()
plt.savefig("results/distribucion_diagnostico.png")
plt.show()

# Graficar regiones
plt.figure(figsize=(10, 5))
sns.countplot(y="region", data=df_meta, order=df_meta["region"].value_counts().index)
plt.title("Distribución de muestras por región cerebral")
plt.xlabel("Número de muestras")
plt.ylabel("Región cerebral")
plt.tight_layout()
plt.savefig("results/distribucion_regiones.png")
plt.show()
