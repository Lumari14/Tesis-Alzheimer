"""
Script: 05_modelado_clasificacion_basica.py

Objetivo
--------
Utilizar los datos ya preparados en el paso 4 (X_train_scaled, X_test_scaled, y_train, y_test)
para entrenar y evaluar dos modelos supervisados de clasificación binaria:

    1) Regresión logística
    2) Random Forest

Pasos principales:
------------------
1. Carga de los datos procesados (arrays .npy y lista de genes).
2. Selección de características con SelectKBest (ANOVA F-test) para reducir dimensionalidad.
3. Entrenamiento de:
       - Regresión logística (modelo lineal)
       - Random Forest (modelo basado en árboles)
4. Evaluación de los modelos:
       - Classification report (precision, recall, F1-score)
       - Matriz de confusión
       - Curva ROC y AUC
5. Guardado de:
       - Modelos entrenados (.pkl)
       - Lista de genes seleccionados
       - Métricas en archivos .txt
       - Gráficos de ROC y matrices de confusión en .png

Este script asume que previamente ejecutaste:
    04_preparacion_datos_modelado.py
y que los archivos se encuentran en: data/processed/model_input/
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)


# ============================================================
# 1. Definir rutas y cargar los datos procesados
# ============================================================

input_dir   = "data/processed/model_input"
models_dir  = "models"
results_dir = "results"
figures_dir = "figures"

# Crear carpetas si no existen
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=== CARGANDO DATOS PROCESADOS ===")

X_train_path = os.path.join(input_dir, "X_train_scaled.npy")
X_test_path  = os.path.join(input_dir, "X_test_scaled.npy")
y_train_path = os.path.join(input_dir, "y_train.npy")
y_test_path  = os.path.join(input_dir, "y_test.npy")
genes_path   = os.path.join(input_dir, "gene_columns.txt")

X_train_scaled = np.load(X_train_path)
X_test_scaled  = np.load(X_test_path)

# y_train y y_test contienen strings ("normal", "Alzheimer's Disease"),
# por eso necesitamos allow_pickle=True.
y_train = np.load(y_train_path, allow_pickle=True)
y_test  = np.load(y_test_path, allow_pickle=True)

# Cargar nombres de genes
with open(genes_path, "r") as f:
    gene_names = [line.strip() for line in f.readlines()]

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled  shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")
print(f"Número de genes (features): {len(gene_names)}")
print()

print("Distribución de clases en y_train:")
print(pd.Series(y_train).value_counts())
print()
print("Distribución de clases en y_test:")
print(pd.Series(y_test).value_counts())
print()


# ============================================================
# 2. Selección de características con SelectKBest (ANOVA F-test)
# ============================================================

"""
Tenemos ~54.675 genes y solo 78 muestras en entrenamiento.
Para muchos modelos es útil reducir dimensionalidad y quedarnos con los genes
más asociados al diagnóstico.

Utilizamos SelectKBest con:
    - score_func = f_classif (ANOVA F-test)
    - k = número de genes a conservar

Esto selecciona los k genes con mayor F-score.
"""

K_FEATURES = 2000  # número de genes a conservar (puedes experimentar con otros valores)

print("=== SELECCIÓN DE CARACTERÍSTICAS (SelectKBest) ===")
print(f"Reduciendo de {X_train_scaled.shape[1]} genes a {K_FEATURES} genes...")

selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel  = selector.transform(X_test_scaled)

print("Forma de X_train_sel:", X_train_sel.shape)
print("Forma de X_test_sel: ", X_test_sel.shape)
print()

# Índices y nombres de genes seleccionados
selected_indices = selector.get_support(indices=True)
selected_genes   = [gene_names[i] for i in selected_indices]

# Scores y p-values de todos los genes, filtramos solo los seleccionados
scores_all  = selector.scores_
pvalues_all = selector.pvalues_

selected_scores  = scores_all[selected_indices]
selected_pvalues = pvalues_all[selected_indices]

genes_selected_df = pd.DataFrame({
    "gene": selected_genes,
    "F_score": selected_scores,
    "p_value": selected_pvalues
}).sort_values("F_score", ascending=False)

genes_selected_path = os.path.join(results_dir, "genes_seleccionados_SelectKBest.csv")
genes_selected_df.to_csv(genes_selected_path, index=False)
print(f"✅ Lista de genes seleccionados guardada en: {genes_selected_path}")
print()


# ============================================================
# 3. Función auxiliar para evaluar modelos
# ============================================================

def evaluar_modelo(nombre_modelo, modelo, X_test, y_test, positive_class="Alzheimer's Disease"):
    """
    Evalúa un modelo binario con varias métricas:
        - classification_report
        - confusion_matrix
        - AUC
        - Curva ROC

    Guarda resultados en:
        - results/metricas_{nombre_modelo}.txt
        - figures/confusion_matrix_{nombre_modelo}.png
        - figures/roc_curve_{nombre_modelo}.png (si aplica)

    Parámetros
    ----------
    nombre_modelo : str
        Nombre del modelo (por ejemplo "logistic_regression" o "random_forest")
    modelo : objeto sklearn ya entrenado
    X_test : ndarray
        Features del conjunto de prueba
    y_test : array-like
        Etiquetas verdaderas del conjunto de prueba (strings)
    positive_class : str
        Nombre de la clase que consideramos positiva (para AUC y ROC)
    """

    print(f"=== EVALUANDO MODELO: {nombre_modelo} ===")

    # Predicción de etiquetas
    y_pred = modelo.predict(X_test)

    # Probabilidades de la clase positiva (si el modelo lo soporta)
    y_prob = None
    if hasattr(modelo, "predict_proba"):
        classes = list(modelo.classes_)
        pos_index = classes.index(positive_class)
        y_prob = modelo.predict_proba(X_test)[:, pos_index]

    # Classification report usando las etiquetas originales
    report = classification_report(y_test, y_pred, digits=4)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

    # Convertir y_test a binario (0/1) para AUC y ROC:
    # 1 = positive_class, 0 = resto
    y_true_bin = (np.array(y_test) == positive_class).astype(int)

    if y_prob is not None:
        auc = roc_auc_score(y_true_bin, y_prob)
    else:
        auc = None

    # Mostrar por pantalla
    print("Classification report:")
    print(report)
    print("Matriz de confusión (filas = verdad, columnas = predicción):")
    print(pd.DataFrame(cm, index=modelo.classes_, columns=modelo.classes_))
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print()

    # Guardar métricas en archivo .txt
    metrics_path = os.path.join(results_dir, f"metricas_{nombre_modelo}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Modelo: {nombre_modelo}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n\n")
        f.write("Matriz de confusión:\n")
        f.write(pd.DataFrame(cm, index=modelo.classes_, columns=modelo.classes_).to_string())
        f.write("\n\n")
        if auc is not None:
            f.write(f"AUC: {auc:.4f}\n")
    print(f"✅ Métricas guardadas en: {metrics_path}")

    # Graficar y guardar matriz de confusión
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.set_title(f"Matriz de confusión - {nombre_modelo}")
    ax_cm.set_xticks(np.arange(len(modelo.classes_)))
    ax_cm.set_yticks(np.arange(len(modelo.classes_)))
    ax_cm.set_xticklabels(modelo.classes_, rotation=45, ha="right")
    ax_cm.set_yticklabels(modelo.classes_)
    plt.colorbar(im, ax=ax_cm)
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Real")

    # Añadir números en las celdas
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig_cm.tight_layout()
    fig_cm_path = os.path.join(figures_dir, f"confusion_matrix_{nombre_modelo}.png")
    plt.savefig(fig_cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"✅ Figura de matriz de confusión guardada en: {fig_cm_path}")

    # Curva ROC utilizando y_true_bin (0/1) y y_prob
    if (y_prob is not None) and (len(np.unique(y_true_bin)) == 2):
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(
            y_true_bin,
            y_prob,
            name=nombre_modelo,
            ax=ax_roc
        )
        ax_roc.set_title(f"Curva ROC - {nombre_modelo}")
        fig_roc.tight_layout()
        fig_roc_path = os.path.join(figures_dir, f"roc_curve_{nombre_modelo}.png")
        plt.savefig(fig_roc_path, dpi=300)
        plt.close(fig_roc)
        print(f"✅ Curva ROC guardada en: {fig_roc_path}")

    print()  # Separador visual en consola


# ============================================================
# 4. Modelo 1: Regresión Logística
# ============================================================

"""
Usamos una regresión logística con regularización L2 (ridge).
Con muchas features y pocas muestras, aumentamos max_iter para asegurar convergencia.

solver='liblinear' funciona bien para problemas binarios.
"""

print("=== ENTRENANDO MODELO: REGRESIÓN LOGÍSTICA ===")

logreg = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    max_iter=5000
)

logreg.fit(X_train_sel, y_train)

# Guardar el modelo
logreg_path = os.path.join(models_dir, "logistic_regression_gse5281.pkl")
with open(logreg_path, "wb") as f:
    pickle.dump(logreg, f)

print(f"✅ Modelo de regresión logística guardado en: {logreg_path}")
print()

# Evaluar el modelo
evaluar_modelo("logistic_regression", logreg, X_test_sel, y_test)


# ============================================================
# 5. Modelo 2: Random Forest
# ============================================================

"""
Random Forest es un modelo basado en muchos árboles de decisión.

Parámetros usados:
    - n_estimators = 300 árboles
    - class_weight='balanced' para compensar posibles ligeros desbalances
    - random_state para reproducibilidad
"""

print("=== ENTRENANDO MODELO: RANDOM FOREST ===")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train_sel, y_train)

# Guardar el modelo
rf_path = os.path.join(models_dir, "random_forest_gse5281.pkl")
with open(rf_path, "wb") as f:
    pickle.dump(rf, f)

print(f"✅ Modelo Random Forest guardado en: {rf_path}")
print()

# Evaluar el modelo
evaluar_modelo("random_forest", rf, X_test_sel, y_test)


# ============================================================
# 6. Mensaje final
# ============================================================

print("✅ Modelado básico completado.")
print("Se han entrenado y evaluado:")
print("   - Regresión logística")
print("   - Random Forest")
print("Revisa las carpetas:")
print(f"   - {models_dir}   -> modelos .pkl")
print(f"   - {results_dir}  -> métricas y lista de genes seleccionados")
print(f"   - {figures_dir}  -> curvas ROC y matrices de confusión")



