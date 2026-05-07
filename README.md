# 🍷 Wine Quality Prediction

A supervised machine learning project to classify wine quality based on physicochemical properties.

## Overview

The goal is to predict wine quality (rated 1–10) as a multiclass classification problem, mapping scores into three categories:

| Label  | Quality score |
|--------|--------------|
| Low    | < 5          |
| Medium | 5 – 7        |
| High   | > 7          |

The dataset contains both **red** and **white** wines (6,497 samples, 12 features) sourced from [Kaggle](https://www.kaggle.com/datasets/rajyellow46/wine-quality).

---

## Features

- `type` — red or white
- `fixed acidity`, `volatile acidity`, `citric acid`
- `residual sugar`, `chlorides`
- `free sulfur dioxide`, `total sulfur dioxide`
- `density`, `pH`, `sulphates`, `alcohol`

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis and correlation heatmaps
- Missing value detection and handling (SimpleImputer / KNNImputer)
- Class imbalance analysis

### 2. Preprocessing
- Encoding of categorical feature (`type`)
- Feature scaling with `StandardScaler` / `MinMaxScaler`
- Class imbalance handling via **SMOTE** and **RandomOverSampler**
- Dimensionality reduction explored with **PCA**

### 3. Feature Selection
- Sequential Feature Selector (`mlxtend`) to identify the most informative features

### 4. Models Compared

| Model               | Notes                          |
|---------------------|--------------------------------|
| Logistic Regression | Baseline linear model          |
| K-Nearest Neighbors | Distance-based classifier      |
| Decision Tree       | Interpretable tree model       |
| Random Forest       | Ensemble, best generalization  |
| AdaBoost            | Boosting on weak learners      |
| SVM (RBF kernel)    | Effective on high-dim data     |
| XGBoost             | Gradient boosting, top performer |

### 5. Evaluation
- **Stratified K-Fold** cross-validation
- Metrics: F1-score (macro), Precision, Recall, ROC-AUC
- Confusion matrix visualization
- Learning and validation curves for overfitting analysis

### 6. Hyperparameter Tuning
- `GridSearchCV` and `RandomizedSearchCV` on best candidates

---

## Results

> XGBoost and Random Forest achieved the best performance, with macro F1-score consistently above the baseline across all folds.

*(See notebook for detailed metrics and plots)*

---

## Project Structure

```
wine-quality-prediction/
│
├── Notebook_1.ipynb       # Main analysis notebook
├── winequalityN.csv       # Dataset
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key libraries: `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `mlxtend`

---

## How to Run

```bash
git clone https://github.com/<your-username>/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
jupyter notebook Notebook_1.ipynb
```

---

## Author

**[Your Name]**  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
