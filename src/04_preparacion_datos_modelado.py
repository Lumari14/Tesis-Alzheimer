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
