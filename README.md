# 🧬 Transcriptomic Analysis & Machine Learning for Alzheimer’s Biomarker Discovery  
**Master’s Thesis in Bioinformatics (UNIR)**  
**Author:** Laura Marín Ríos  
**Year:** 2025  

---

## 📌 Project Summary

This repository contains the full pipeline developed for my Master’s Thesis, which aims to identify **robust transcriptomic biomarkers** of Alzheimer’s Disease (AD) using a combination of:

- Differential expression–style feature selection  
- Machine Learning models (linear + nonlinear)  
- Explainable AI (XAI) techniques such as SHAP  
- Automated gene annotation  
- Reproducible and modular Python workflows  

All analyses are performed on the **GSE5281** dataset from the Gene Expression Omnibus (GEO), which contains microarray data from several brain regions of both Alzheimer’s patients and healthy controls.

The project demonstrates how machine learning and biological interpretation can be combined to uncover new candidate biomarkers and better understand disease-related gene expression patterns.

---

## 🧱 Project Structure
Tesis-Alzheimer/
│
├── src/ # Full analysis pipeline (Python scripts)
│ ├── descarga_gse5281.py
│ ├── 01_limpieza_gse5281.py
│ ├── 02_combinacion_gse5281.py
│ ├── 03_QC_Limpieza_Combinacion.py
│ ├── 04_perapacion_datos_modelado.py
│ ├── 05_modelado_clasificacion_basica.py
│ ├── 06_Importancia_lineal_y_biomarcadores.py
│ ├── 07_shap_random_forest.py
│ ├── 08_cruce_biomarkers.py
│ ├── 09_cruce_biomarcadores_topN.py
│ ├──10_tabla_biomarcadores_detallada.py
│ ├── 11_anotacion_biomarcadores_topN.py
│
├── results/ # CSV files: metrics, rankings, biomarkers
├── figures/ # ROC, confusion matrices, SHAP plots, etc.
├── models/ # Trained ML models (.pkl)
├── notebooks/ # Additional Jupyter notebooks
├── data/ # (Structure only; raw data excluded from repo)
│
└── README.md


---

## 🔬 Dataset: GSE5281 (GEO)

The dataset GSE5281 includes Affymetrix Human Genome U133 Plus 2.0 microarray data collected from multiple brain regions across:

- **Alzheimer’s Disease patients**
- **Healthy controls**

After cleaning and filtering:

- **105 samples**  
- **54,675 genes**  
- **Two classes:** *normal* vs *Alzheimer’s Disease*

---

## 🚀 Full Analysis Pipeline

### **1. Data Acquisition & Cleaning**
- Download directly from GEO  
- Extract and standardize metadata  
- Remove samples with missing annotations  
- Filter genes with low variance  
- Validate sample IDs  
- Generate QC reports  

### **2. Machine Learning Preparation**
- Stratified train/test split  
- Standard scaling  
- Feature selection using **SelectKBest (ANOVA)** → 2,000 most variable genes  
- Store processed matrices and metadata  

### **3. Classification Models**
Two main models were trained:

- **Logistic Regression** (linear)
- **Random Forest** (nonlinear)

Performance:

- Logistic Regression → **AUC ≈ 0.99**  
- Random Forest → **AUC ≈ 0.95**

### **4. Explainability & Feature Importance**
**Linear explainability**
- Coefficients  
- Odds ratios  
- Permutation importance  
- Combined ranking

**Nonlinear explainability**
- SHAP values (Random Forest)
- Global SHAP ranking
- Summary plots (beeswarm + bar)
- SHAP dependence plots for top genes

### **5. Biomarker Integration**
Cross-model comparison identified:

- **1 shared gene in the top 50** of both models  
- **7 shared genes in the top 100**

These represent the **most robust multi-model biomarker candidates** for Alzheimer’s Disease.

### **6. Gene Annotation**
Automatic annotation with:
- Official gene symbols  
- Gene descriptions  
- Biological functions  
- Known involvement in AD or neurodegeneration

---

## 🧬 Robust Biomarkers Identified

| Probe ID       | Gene | Main Function | Evidence Level |
|----------------|------|---------------|----------------|
| 223272_s_at    | NTPCR | NTP-related metabolism | Moderate |
| 215789_s_at    | AJAP1 | Neuronal adhesion | Moderate |
| 223460_at      | CAMKK1 | Ca2+ signaling / synaptic plasticity | Strong (pathway) |
| 227582_at      | KLHDC9 | Protein–protein interaction | Novel |
| 228027_at      | GPRASP2 | GPCR signaling | Moderate |
| 218328_at      | COQ4 | Mitochondrial CoQ10 biosynthesis | Moderate |
| 224689_at      | MANBAL | Glycoprotein / lysosomal processing | Moderate |

Three of these genes have **little or no prior connection to AD**, making them promising **novel candidates** for future research.

---

## 📊 Key Figures (stored in `/figures/`)

- ROC curves for all models  
- Confusion matrices  
- SHAP global summary plots  
- SHAP dependence plots  
- Linear vs SHAP comparison plots  
- Feature selection and ranking plots  

---

## 🧠 Technologies

- Python 3.11+  
- NumPy, Pandas  
- Scikit-Learn  
- SHAP  
- Matplotlib / Seaborn  
- GEOparse  
- MyGene  

---

## ♻️ Reusability

This pipeline is fully modular and can be reused for:

- Any GEO microarray dataset  
- RNA-seq datasets (TPM/FPKM/log-TPM)  
- Disease classification tasks  
- Biomarker discovery studies  
- Explainable AI research  

---

## 💬 Citation

If you use this repository or pipeline:

Marín Ríos, L. (2025). Transcriptomic analysis and machine learning for biomarker identification in Alzheimer’s Disease. Master’s Thesis in Bioinformatics, UNIR.


---

## 📬 Contact

**Laura Marín Ríos**  
Bioinformatics & Data Science  
GitHub: https://github.com/Lumari14




