"""
Script: 05b_validacion_overfitting.py

Objetivo
--------
Evaluar posibles signos de sobreajuste (overfitting) en los modelos entrenados
(Regresión Logística y Random Forest) mediante:

  1. Validación cruzada estratificada (5-fold) en el conjunto de entrenamiento.
  2. Comparación de AUC en train vs test.

Este script asume que ya se han generado los archivos:

  - data/processed/model_input/X_train_scaled.npy
  - data/processed/model_input/X_test_scaled.npy
  - data/processed/model_input/y_train.npy
  - data/processed/model_input/y_test.npy

y que en el modelado se utilizó SelectKBest (ANOVA) con k=2000.
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


# ============================================================
# 1. Definir rutas y cargar datos
# ============================================================

print("=== CARGANDO DATOS PROCESADOS ===")

base_dir = "data/processed/model_input"

X_train_path = os.path.join(base_dir, "X_train_scaled.npy")
X_test_path  = os.path.join(base_dir, "X_test_scaled.npy")
y_train_path = os.path.join(base_dir, "y_train.npy")
y_test_path  = os.path.join(base_dir, "y_test.npy")

# Cargamos matrices de features (float)
X_train_scaled = np.load(X_train_path)
X_test_scaled  = np.load(X_test_path)

# Cargamos etiquetas (guardadas como object -> allow_pickle=True)
y_train = np.load(y_train_path, allow_pickle=True)
y_test  = np.load(y_test_path,  allow_pickle=True)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled  shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")

# Mostrar distribución de clases
print("\nDistribución de clases en y_train:")
print(pd.Series(y_train).value_counts())
print("\nDistribución de clases en y_test:")
print(pd.Series(y_test).value_counts())
print()

# Número total de genes (features) antes de selección
n_features_total = X_train_scaled.shape[1]
print(f"Número total de genes (features): {n_features_total}")
print()


# ============================================================
# 2. Selección de características: SelectKBest (ANOVA)
# ============================================================

"""
Usamos el mismo enfoque que en el script de modelado:

    - SelectKBest con f_classif (ANOVA)
    - k = 2000 características más informativas

IMPORTANTE:
    Se ajusta SIEMPRE solo sobre X_train, y_train.
    Luego se transforma X_test con el transformador ya ajustado.
"""

K_FEATURES = 2000

print("=== RECONSTRUYENDO SELECTKBEST (ANOVA, k=2000) ===")

selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel  = selector.transform(X_test_scaled)

print(f"Forma de X_train_sel: {X_train_sel.shape}")
print(f"Forma de X_test_sel:  {X_test_sel.shape}")
print(f"Número de genes seleccionados: {X_train_sel.shape[1]} (debería ser {K_FEATURES})")
print()

# Para referencia, podemos recuperar los nombres de genes si existen
gene_cols_path = os.path.join(base_dir, "gene_columns.txt")
selected_genes = None
if os.path.exists(gene_cols_path):
    with open(gene_cols_path, "r") as f:
        all_genes = [line.strip() for line in f.readlines()]
    # selector.get_support() indica qué columnas se han seleccionado
    mask = selector.get_support()
    selected_genes = np.array(all_genes)[mask]
    print("Ejemplos de genes seleccionados:", selected_genes[:10])
    print()


# ============================================================
# 3. Definir modelos
# ============================================================

"""
Definimos los mismos tipos de modelos usados en el script de modelado.
Si en el script 05_modelado_clasificacion_basica.py usaste otros hiperparámetros,
ajusta estos para que coincidan.

- Regresión Logística: modelo lineal con regularización L2.
- Random Forest: modelo no lineal basado en árboles de decisión.
"""

print("=== DEFINIENDO MODELOS ===")

logreg = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=5000,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

models = {
    "logistic_regression": logreg,
    "random_forest": rf
}

print("Modelos definidos: ", list(models.keys()))
print()


# ============================================================
# 4. Validación cruzada estratificada (5-fold) en TRAIN
# ============================================================

"""
Aquí queremos ver:

  - Cómo se comporta el modelo al hacer validación cruzada
    solo en X_train_sel, y_train.
  - Usamos StratifiedKFold para mantener la proporción de clases.
  - Métrica: AUC ROC (valores cerca de 1 son muy buenos).

Si el AUC medio de CV es razonable y cercano al AUC en test,
es una buena señal de que no hay sobreajuste severo.
"""

print("=== VALIDACIÓN CRUZADA (5-fold) EN TRAIN ===")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n--- Modelo: {name} ---")
    cv_scores = cross_val_score(
        model,
        X_train_sel,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )
    print("AUC en cada fold:", np.round(cv_scores, 4))
    print("AUC CV media:", np.round(cv_scores.mean(), 4))
    print("Desviación estándar AUC:", np.round(cv_scores.std(), 4))

print()


# ============================================================
# 5. Comparación AUC TRAIN vs TEST
# ============================================================

"""
Ahora entrenamos cada modelo en TODO el conjunto de entrenamiento (X_train_sel, y_train)
y comparamos:

  - AUC en TRAIN
  - AUC en TEST

Para ello:

  1. Entrenamos el modelo.
  2. Obtenemos probabilidades de pertenecer a la clase positiva:
       "Alzheimer's Disease".
  3. Binarizamos y_train / y_test (0 = otra clase, 1 = Alzheimer).
  4. Calculamos AUC con roc_auc_score.

Si AUC_train >> AUC_test, hay indicios fuertes de sobreajuste.
Si AUC_train ~ AUC_test, el modelo generaliza razonablemente bien.
"""

print("=== COMPARACIÓN AUC TRAIN vs TEST ===")

# Binarizamos etiquetas (0/1), con 1 = "Alzheimer's Disease"
positive_class = "Alzheimer's Disease"

# y_bin = 1 si es Alzheimer, 0 si no
y_train_bin = (y_train == positive_class).astype(int)
y_test_bin  = (y_test == positive_class).astype(int)


for name, model in models.items():
    print(f"\n--- Modelo: {name} ---")

    # Entrenar en TODO el train
    model.fit(X_train_sel, y_train)

    # Probabilidades para la clase positiva
    classes = list(model.classes_)
    if positive_class not in classes:
        raise ValueError(
            f"La clase positiva '{positive_class}' no está entre las clases del modelo: {classes}"
        )
    pos_idx = classes.index(positive_class)

    y_train_prob = model.predict_proba(X_train_sel)[:, pos_idx]
    y_test_prob  = model.predict_proba(X_test_sel)[:, pos_idx]

    # AUC
    auc_train = roc_auc_score(y_train_bin, y_train_prob)
    auc_test  = roc_auc_score(y_test_bin,  y_test_prob)

    print(f"AUC TRAIN: {auc_train:.4f}")
    print(f"AUC TEST : {auc_test:.4f}")
    print("Diferencia AUC (train - test):", np.round(auc_train - auc_test, 4))

print("\n✅ Evaluación de sobreajuste completada.")
print("Revisa los AUC de validación cruzada y las diferencias entre TRAIN y TEST para cada modelo.")
