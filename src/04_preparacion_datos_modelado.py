"""
Script: 04_preparacion_datos_modelado.py

Objetivo
--------
A partir del archivo limpio 'GSE5281_datos_limpios.csv', este script:

1. Carga el dataset depurado (sin muestras con diagnosis/region vacíos).
2. Define qué columnas son metadatos (diagnosis, region, age, sex)
   y cuáles corresponden a expresión génica (features).
3. Separa las variables:
      - X: matriz de expresión génica (features de entrada)
      - y: diagnóstico (variable objetivo para modelos supervisados)
4. Estandariza los datos de expresión (media=0, desviación estándar=1),
   paso importante para muchos algoritmos de ML.
5. Realiza una partición train/test estratificada según el diagnóstico,
   para poder evaluar el rendimiento de los modelos en datos no vistos.
6. Guarda los objetos resultantes (X_train, X_test, y_train, y_test, scaler)
   en la carpeta 'data/processed/model_input/' para usarlos en scripts posteriores.

Este script NO entrena modelos todavía, solo deja los datos listos para modelar.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt


# ============================================================
# 1. Definir rutas y cargar el archivo limpio
# ============================================================

# Ruta al archivo limpio generado en el paso de limpieza/QC
clean_path = "data/processed/GSE5281_datos_limpios.csv"

print("=== CARGA DE DATOS LIMPIOS ===")
print(f"Leyendo archivo: {clean_path}")

# index_col=0 porque el índice (fila) suele ser el ID de la muestra (GSM...)
df = pd.read_csv(clean_path, index_col=0)

print("Forma del dataframe limpio (muestras x columnas):", df.shape)
print("Primeras filas:")
print(df.head())
print()


# ============================================================
# 2. Identificar columnas de metadatos y columnas de genes
# ============================================================

"""
En este dataset, sabemos por pasos anteriores que:
- 'diagnosis'  -> estado (normal vs Alzheimer's Disease)
- 'region'     -> región cerebral
- 'age'        -> edad del donante
- 'sex'        -> sexo del donante

Estas columnas son metadatos clínicos.
El resto de columnas corresponden a sondas/genes de expresión génica.
"""

meta_cols = ["diagnosis", "region", "age", "sex"]

# Comprobamos que todas las columnas de metadatos están en el dataframe
print("=== COMPROBACIÓN DE METADATOS ===")
print("Columnas disponibles en el dataframe:")
print(df.columns[:10])  # mostramos solo las primeras 10 para abreviar
print()

missing_meta = [col for col in meta_cols if col not in df.columns]
if missing_meta:
    print("⚠️ Advertencia: faltan las siguientes columnas de metadatos:", missing_meta)
else:
    print("✅ Todas las columnas de metadatos están presentes:", meta_cols)
print()

# Definimos las columnas de genes como "todas menos las de metadatos"
gene_cols = [col for col in df.columns if col not in meta_cols]

print("=== RESUMEN DE VARIABLES ===")
print(f"Número de columnas de metadatos: {len(meta_cols)}")
print(f"Número de columnas de genes (features): {len(gene_cols)}")
print(f"Número total de muestras: {df.shape[0]}")
print()

# Opcional: comprobar distribución de diagnosis (ya la vimos en el QC)
print("Distribución de diagnosis en el archivo limpio:")
print(df["diagnosis"].value_counts())
print()


# ============================================================
# 3. Definir X (features) e y (variable objetivo)
# ============================================================

"""
Para el modelado inicial vamos a usar solo la expresión génica como features.
Más adelante podrías experimentar con incluir también 'age', 'sex' o 'region'
como variables adicionales, pero es habitual empezar con genes únicamente.

- X: matriz de expresión [n_muestras x n_genes]
- y: diagnóstico (normal vs Alzheimer's Disease)
"""

X = df[gene_cols].values   # convertimos a matriz NumPy para sklearn
y = df["diagnosis"].values # por ahora dejamos y como etiquetas categóricas (strings)

print("=== DEFINICIÓN DE X e y ===")
print("Forma de X (muestras x genes):", X.shape)
print("Forma de y (muestras,):", y.shape)
print("Ejemplos de etiquetas en y:", np.unique(y))
print()


# ============================================================
# 4. Estandarización de los datos de expresión (X)
# ============================================================

"""
Muchos modelos de ML (regresión logística, SVM, redes neuronales, PCA, etc.)
funcionan mejor si cada feature tiene media 0 y desviación estándar 1.

Aquí usamos StandardScaler:
- ajustamos (fit) sobre TODAS las muestras
- transformamos X a X_scaled

Nota: en un pipeline real de producción, el fit debería hacerse solo con
X_train y luego aplicar transform a X_train y X_test. Para mayor claridad
en el TFM y porque aún no entrenamos, vamos a hacer el escalado después de
separar train/test (ver siguiente sección).
"""


# ============================================================
# 5. División train/test estratificada
# ============================================================

"""
Dividimos el dataset en:
- Conjunto de entrenamiento (train): ~75% de las muestras
- Conjunto de prueba (test):        ~25% de las muestras

Usamos 'stratify=y' para que la proporción normal vs Alzheimer se mantenga
similar en ambos conjuntos. Esto es importante para evaluar los modelos de
forma justa.
"""

print("=== DIVISIÓN TRAIN/TEST ===")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,       # 25% de las muestras para test
    random_state=42,      # semilla reproducible
    stratify=y            # mantiene la proporción de clases
)

print("Tamaño X_train:", X_train.shape)
print("Tamaño X_test: ", X_test.shape)
print("Tamaño y_train:", y_train.shape)
print("Tamaño y_test: ", y_test.shape)
print()

# Comprobamos distribución de clases en train y test
print("Distribución de diagnosis en y_train:")
print(pd.Series(y_train).value_counts())
print()

print("Distribución de diagnosis en y_test:")
print(pd.Series(y_test).value_counts())
print()


# ============================================================
# 6. Estandarizar SOLO con datos de entrenamiento
#    y aplicar la transformación a train y test
# ============================================================

"""
Ahora sí aplicamos el escalado correcto para un pipeline de ML:

1) Ajustar (fit) el StandardScaler solo con X_train.
2) Transformar X_train y X_test con ese scaler.
   De esta forma, la información del conjunto de test no se "filtra"
   hacia el entrenamiento (evitamos data leakage).
"""

print("=== ESTANDARIZACIÓN DE FEATURES ===")
scaler = StandardScaler()

# Ajustamos el scaler SOLO con X_train
X_train_scaled = scaler.fit_transform(X_train)

# Aplicamos la misma transformación a X_test
X_test_scaled = scaler.transform(X_test)

print("Escalado completado.")
print("Media de X_train_scaled (aprox 0):", np.mean(X_train_scaled))
print("Desviación estándar de X_train_scaled (aprox 1):", np.std(X_train_scaled))
print()

# ============================================================
# 7. GRÁFICOS: ANTES vs DESPUÉS DEL ESCALADO
# ============================================================
""" Objetivo:
# - Visualizar cómo cambian las distribuciones al aplicar StandardScaler (Z-score)
# - Generar 3 imágenes:
#     1) Histograma ANTES del escalado
#     2) Histograma DESPUÉS del escalado
#     3) Imagen combinada (antes + después)
#
# Nota:
# - Para que sea interpretable, tomamos solo un subconjunto de genes (p.ej. 50)
# - Aplanamos (flatten) los valores para ver la distribución global en train
# - Usamos únicamente X_train y X_train_scaled (evita data leakage y es estándar)
"""

# Carpeta de salida
os.makedirs("figures/normalization", exist_ok=True)

# ---- 1) Seleccionar un subconjunto de genes para visualizar ----
# Recomendación: 50 genes es suficiente y no sobrecarga la figura
N_GENES_PLOT = 50

# Si tienes gene_cols (lista con los nombres de genes), genial para referencia.
# Pero para el histograma solo necesitamos valores numéricos.
# Tomaremos las primeras N columnas (genes) de X_train y X_train_scaled.
# (También podrías elegir al azar si prefieres.)
subset_idx = np.arange(min(N_GENES_PLOT, X_train.shape[1]))

# Valores "antes" (sin estandarizar) y "después" (Z-score)
vals_before = X_train[:, subset_idx].ravel()
vals_after  = X_train_scaled[:, subset_idx].ravel()

# ---- 2) Histograma ANTES del escalado ----
plt.figure()
plt.hist(vals_before, bins=60)
plt.title(f"Gene expression distribution (TRAIN) - BEFORE scaling (first {len(subset_idx)} genes)")
plt.xlabel("Expression value")
plt.ylabel("Frequency")
out_before = "figures/normalization/hist_before_scaling.png"
plt.savefig(out_before, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Guardado: {out_before}")

# ---- 3) Histograma DESPUÉS del escalado ----
plt.figure()
plt.hist(vals_after, bins=60)
plt.title(f"Gene expression distribution (TRAIN) - AFTER StandardScaler (first {len(subset_idx)} genes)")
plt.xlabel("Z-score value")
plt.ylabel("Frequency")
out_after = "figures/normalization/hist_after_scaling.png"
plt.savefig(out_after, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Guardado: {out_after}")

# ---- 4) Imagen combinada (antes + después) ----
# En una sola figura con dos paneles para compararlo visualmente
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(vals_before, bins=60)
axes[0].set_title("BEFORE scaling")
axes[0].set_xlabel("Expression value")
axes[0].set_ylabel("Frequency")

axes[1].hist(vals_after, bins=60)
axes[1].set_title("AFTER StandardScaler (Z-score)")
axes[1].set_xlabel("Z-score value")
axes[1].set_ylabel("Frequency")

fig.suptitle(f"Distribution comparison (TRAIN, first {len(subset_idx)} genes)", fontsize=12)

out_combined = "figures/normalization/hist_before_vs_after_scaling.png"
plt.savefig(out_combined, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✅ Guardado: {out_combined}")


# ============================================================
# 7. Guardar los datos procesados y el scaler
# ============================================================

"""
Guardamos los arrays y el scaler en disco para reutilizarlos en el siguiente
script (por ejemplo, '05_modelado_clasificacion.py').

Guardaremos:
- X_train_scaled, X_test_scaled
- y_train, y_test
- scaler (objeto de sklearn)
- lista de nombres de genes (por si queremos inspeccionar coeficientes después)
"""

output_dir = "data/processed/model_input"
os.makedirs(output_dir, exist_ok=True)

print("=== GUARDANDO DATOS PROCESADOS ===")

# Guardamos los conjuntos de datos en formato NumPy
np.save(os.path.join(output_dir, "X_train_scaled.npy"), X_train_scaled)
np.save(os.path.join(output_dir, "X_test_scaled.npy"),  X_test_scaled)
np.save(os.path.join(output_dir, "y_train.npy"),        y_train)
np.save(os.path.join(output_dir, "y_test.npy"),         y_test)

# Guardamos el scaler con pickle
with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Guardamos también la lista de genes por si luego queremos mapear coeficientes
genes_path = os.path.join(output_dir, "gene_columns.txt")
with open(genes_path, "w") as f:
    for g in gene_cols:
        f.write(g + "\n")

print(f"Archivos guardados en: {output_dir}")
print("- X_train_scaled.npy")
print("- X_test_scaled.npy")
print("- y_train.npy")
print("- y_test.npy")
print("- scaler.pkl")
print("- gene_columns.txt")
print()

print("✅ Preparación de datos para modelado completada.")
print("Ahora puedes pasar al siguiente paso: entrenar y evaluar modelos de clasificación.")


