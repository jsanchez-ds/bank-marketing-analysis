# 🏦 Bank Marketing Campaign Analysis

Analyzing direct marketing campaigns of a Portuguese banking institution to predict whether a client will subscribe to a term deposit. This project combines **exploratory data analysis**, **statistical testing**, **customer segmentation**, and **predictive modeling** to derive actionable business insights.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## 🎯 Objective

> **Business Question:** Which client profiles are most likely to subscribe to a term deposit, and how can the bank optimize its campaign strategy to increase conversion rates?

This project addresses the question from multiple angles:
- **Descriptive**: Who are the clients that subscribe? What campaign patterns drive conversion?
- **Inferential**: Are conversion rate differences across segments statistically significant?
- **Predictive**: Can we build a model to identify high-probability subscribers before calling them?
- **Prescriptive**: What actionable recommendations can improve campaign ROI?

---

## 📊 Dataset

| Feature | Detail |
|---------|--------|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Reference** | Moro et al. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing* |
| **Records** | 45,211 |
| **Features** | 16 input + 1 target |
| **Target** | `y` — Has the client subscribed a term deposit? (yes/no) |
| **Class Balance** | ~88% No / ~12% Yes (imbalanced) |

**Key variables:** age, job, marital status, education, balance, housing loan, contact type, campaign duration, number of contacts, days since previous contact, previous campaign outcome.

---

## 🔬 Methodology

```
01_eda.ipynb                    02_statistical_testing.ipynb
  ├── Data cleaning               ├── Chi-square tests
  ├── Univariate analysis          ├── T-tests / Mann-Whitney
  ├── Bivariate analysis           └── Conversion rate comparison
  └── Feature engineering              by segment

03_segmentation.ipynb           04_predictive_modeling.ipynb
  ├── K-Means clustering           ├── Logistic Regression
  ├── Segment profiling            ├── Random Forest
  └── Segment-level conversion     ├── XGBoost
                                   ├── Cross-validation
                                   ├── Hyperparameter tuning
                                   └── SHAP explainability
```

---

## 📈 Key Results

### Model Performance

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

> *Results will be populated after running the notebooks.*

### Main Findings

1. **[Finding 1]** — To be completed after analysis
2. **[Finding 2]** — To be completed after analysis
3. **[Finding 3]** — To be completed after analysis

---

## 🗂️ Project Structure

```
bank-marketing-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_statistical_testing.ipynb
│   ├── 03_segmentation.ipynb
│   └── 04_predictive_modeling.ipynb
├── src/
│   └── utils.py
├── data/
│   └── README.md
└── figures/
    └── (exported plots)
```

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Statsmodels` `SciPy` `Matplotlib` `Seaborn` `SHAP`

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/Jonathan742001/bank-marketing-analysis.git
cd bank-marketing-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Follow instructions in data/README.md

# 4. Run the notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

---

## 👤 Author

**Jonathan Sánchez**
- GitHub: [@Jonathan742001](https://github.com/Jonathan742001)
- Universidad de Chile — Industrial Engineering
