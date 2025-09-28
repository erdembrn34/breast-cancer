# Breast Cancer Analysis & Patient Report Automation

## Project Overview
This project demonstrates a full machine learning and natural language processing (NLP) pipeline for analyzing breast cancer data, generating patient summaries, and extracting insights from clinical reports. It combines:

- **Machine Learning**: RandomForest for cancer classification.
- **Explainable AI**: SHAP for feature importance and model explainability.
- **Tabular Deep Learning**: TabNet for advanced tabular modeling.
- **Class Imbalance Handling**: SMOTE for oversampling minority classes.
- **Natural Language Processing (LLM)**: Summarization, question-answering, and sentiment analysis for patient feedback using Hugging Face Transformers.
- **Visualization**: Matplotlib and Seaborn for charts, ROC curves, and feature importance plots.

---

## Dataset
The project uses the **Breast Cancer Wisconsin dataset**:

- **Columns**: 32 numerical features derived from cell nucleus measurements, plus `diagnosis` (M = malignant, B = benign).  
- **Rows**: 569 patient samples.

---

## Project Structure

- `notebook.ipynb` – Main Jupyter notebook with full workflow.
- `data.csv` – Input dataset.
- `README.md` – Project documentation.

---

## Key Features

1. **Data Preprocessing**
   - Removal of unnecessary columns (`id`, `Unnamed: 32`)
   - Scaling of features using `StandardScaler`
   - Train/Test split

2. **Machine Learning**
   - RandomForest with hyperparameter tuning
   - Model evaluation: accuracy, classification report, confusion matrix, ROC/PR curves

3. **Explainable AI**
   - Feature importance using RandomForest
   - SHAP TreeExplainer visualization

4. **Tabular Deep Learning**
   - TabNetClassifier for advanced tabular predictions

5. **Class Imbalance Handling**
   - SMOTE to oversample minority class (malignant cases)

6. **Natural Language Processing (LLM)**
   - Summarization and question-answering with Hugging Face Transformers
   - Patient feedback sentiment analysis

7. **Visualization**
   - Count plots, ROC & PR curves
   - Top feature importance charts

---

## Installation

1. Clone this repository:
```bash
git clone (https://github.com/erdembrn34/breast-cancer/)

