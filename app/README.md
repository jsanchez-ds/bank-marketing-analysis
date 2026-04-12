# Streamlit demo · Bank Marketing Predictor

Interactive web app that wraps the Random Forest classifier from
`notebooks/notebook_3_MachineLearning.ipynb` and exposes it as a scoring tool.

## Tabs

| Tab | What it shows |
|---|---|
| **Predict** | Form to score a hypothetical client; tunable decision threshold; lift over base rate; CALL/SKIP recommendation. |
| **Insights** | Headline conversion-rate findings (warm leads ~7x lift, students ~3x, May campaign ROI). |
| **Model Card** | Hold-out ROC-AUC / PR-AUC, ROC curve, confusion matrix, top 15 features, leakage caveat. |

## Run locally

```bash
# from the repository root
pip install -r app/requirements.txt

# 1. Train the model once (downloads UCI bank-full.csv via OpenML fallback)
python app/train_model.py

# 2. Launch the app
streamlit run app/streamlit_app.py
```

The training step writes `app/artifacts/model.joblib` (~12 MB) and
`app/artifacts/metrics.json`. Both are committed so the cloud deployment does
not need to retrain.

## Deploy to Streamlit Community Cloud

1. Push this folder to GitHub (already done if you are reading this on the
   `main` branch).
2. Sign in at <https://share.streamlit.io> with the GitHub account that owns
   the repo.
3. Click **New app** and point it at:
   - **Repository:** `jsanchez-ds/bank-marketing-analysis`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Under **Advanced settings**, pin Python to `3.11`.
5. Click **Deploy**. The first build takes ~3 minutes; subsequent pushes redeploy
   in seconds.

## Re-training

Re-run `python app/train_model.py` and commit the updated `model.joblib` /
`metrics.json`. The app reads metadata (ROC curve points, confusion matrix,
feature importances) directly from the artifact, so the **Model Card** tab
stays in sync automatically.

## Why no `duration` feature?

Per Moro et al. (2014), `duration` is only observed *after* a call ends, so any
model that uses it leaks the target. The training pipeline drops it explicitly.
