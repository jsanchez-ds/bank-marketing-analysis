# 🏦 Bank Marketing Campaign Analysis

Analyzing direct marketing campaigns of a Portuguese banking institution to predict whether a client will subscribe to a term deposit. This project combines **exploratory data analysis**, **statistical testing**, **customer segmentation**, and **predictive modeling** to derive actionable business insights.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5-E25A1C?logo=apachespark&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-green)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io)

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
| **Class Balance** | 88.7% No / 11.3% Yes (imbalanced) |

**Key variables:** age, job, marital status, education, balance, housing loan, contact type, campaign duration, number of contacts, days since previous contact, previous campaign outcome.

> ⚠️ **Note on `duration`:** Per the UCI documentation and Moro et al. (2014), the `duration` field is only known *after* a call ends and is therefore excluded from all predictive models to avoid data leakage.

---

## 🔬 Methodology

The analysis is structured as a Databricks pipeline with three notebooks:

```
notebook_1_EDA.ipynb              notebook_2_Limpieza_Preprocesamiento.ipynb
  ├── Schema & quality checks       ├── Encoding (Label / OHE)
  ├── Univariate analysis           ├── Feature engineering
  ├── Bivariate analysis            ├── Stratified train/test split
  ├── Conversion-rate segments      └── Persisted as parquet
  └── Statistical testing

notebook_3_MachineLearning.ipynb
  ├── Decision Tree (baseline)
  ├── Random Forest + GridSearch
  ├── XGBoost + GridSearch
  ├── Stratified K-Fold CV
  └── ROC-AUC evaluation on hold-out test set
```

**Stack:** PySpark for data processing, scikit-learn + XGBoost for modeling, matplotlib/seaborn for visualization.

---

## 📈 Key Results

### Model Performance (Hold-out Test Set, ROC-AUC)

| Model | CV AUC | Test AUC | Notes |
|---|---|---|---|
| Decision Tree (max_depth=5) | — | **0.7745** | Interpretable baseline |
| XGBoost (default) | — | **0.7819** | Strong out-of-the-box |
| XGBoost (GridSearch) | 0.9611 | **0.7647** | ⚠️ CV-test gap reveals leakage |
| **Random Forest (best)** | — | **0.7959** ★ | Best generalization |

> **Critical finding:** The XGBoost grid-search CV reported AUC ≈ 0.96 but only achieved 0.76 on the held-out test set. Root cause: SMOTE was applied to the full training set *before* CV, leaking synthetic samples across folds. The fix — applying SMOTE inside an `imblearn.Pipeline` so each fold resamples independently — is implemented in the [v2 iteration](#-v2-iteration). This is exactly the kind of methodology bug a real production pipeline must catch.

### Main Findings

1. **Severe class imbalance (88.7% / 11.3%)** — accuracy is misleading; ROC-AUC and PR-AUC are the relevant metrics, and resampling must be CV-aware.
2. **Previous contact is the strongest signal** — clients previously contacted convert at **63.8%** vs **9.3%** for new clients (≈ **7× lift**). Re-engagement campaigns should be prioritized.
3. **High-converting segments**: students (**31%**) and retirees (**25%**) convert at roughly **3× the base rate** (11.3%). Both segments are systematically under-targeted in the current campaign mix.
4. **Channel inefficiency in May**: 13,769 calls placed but only **6.4%** converted — the worst ROI of the year. Volume-based scheduling outperformed by quality-based targeting.
5. **Top predictive features** (Random Forest importance): `poutcome_success`, `pdays`, `previous`, `month`, `contact_type`, `balance`.

---

## 🗂️ Project Structure

```
bank-marketing-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook_1_EDA.ipynb
├── notebook_2_Limpieza_Preprocesamiento.ipynb
└── notebook_3_MachineLearning.ipynb
```

The `databricks-version` branch contains the original Databricks `.py` source format for direct import into a workspace.

---

## 🔁 v2 Iteration

A second version of the modeling pipeline lives on the [`v2-imblearn-pipeline`](https://github.com/jsanchez-ds/bank-marketing-analysis/tree/v2-imblearn-pipeline) branch. It addresses the SMOTE-in-CV leakage diagnosed above by wrapping resampling and the classifier in an `imblearn.pipeline.Pipeline`, plus adds:

- Threshold tuning optimized for F1 / business cost
- PR-AUC alongside ROC-AUC (more honest under heavy imbalance)
- `joblib` model + encoder persistence
- Pinned `requirements.txt` for reproducibility

---

## 🛠️ Tech Stack

`Python` `PySpark` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `imbalanced-learn` `Matplotlib` `Seaborn`

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/jsanchez-ds/bank-marketing-analysis.git
cd bank-marketing-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from UCI
#    https://archive.ics.uci.edu/dataset/222/bank+marketing

# 4. Run the notebooks in order
jupyter notebook notebook_1_EDA.ipynb
```

To run on Databricks instead, import the `.py` files from the `databricks-version` branch.

---

---

## Live Demo (Streamlit)

An interactive Streamlit app wraps the Random Forest classifier and exposes it as
a scoring tool with three tabs:

- **Predict** — score a hypothetical client with a tunable decision threshold and
  CALL/SKIP recommendation.
- **Insights** — headline conversion-rate findings (warm leads ~7x lift, students
  ~3x base rate, May campaign ROI).
- **Model Card** — held-out ROC-AUC / PR-AUC, ROC curve, confusion matrix, top 15
  features, and the `duration` leakage caveat.

Run locally:

```bash
pip install -r app/requirements.txt
python app/train_model.py            # one-time, ~30s
streamlit run app/streamlit_app.py
```

See [`app/README.md`](app/README.md) for the full guide and Streamlit Community
Cloud deployment steps.

---

## 👤 Author

**Jonathan Sánchez**
- GitHub: [@jsanchez-ds](https://github.com/jsanchez-ds)
- Universidad de Chile — Industrial Engineering
