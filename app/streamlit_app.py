"""
Streamlit demo for the Bank Marketing classifier.

The app loads a pre-trained Random Forest pipeline (see ``train_model.py``) and
exposes three tabs:

* **Predict** — interactive form returning a subscription probability and a
  business recommendation derived from a configurable decision threshold.
* **Insights** — the headline findings from the original analysis (segments,
  channel inefficiencies, lifts).
* **Model Card** — held-out metrics, ROC curve, confusion matrix, top features,
  and the data-leakage caveat around ``duration``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"

st.set_page_config(
    page_title="Bank Marketing — Term Deposit Predictor",
    page_icon=None,
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model artifact...")
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        st.error(
            f"Model artifact not found at `{ARTIFACT_PATH}`.\n\n"
            "Run `python app/train_model.py` once before launching the app."
        )
        st.stop()
    return joblib.load(ARTIFACT_PATH)


artifact = load_artifact()
pipeline = artifact["pipeline"]
meta = artifact["metadata"]
schema = meta["feature_schema"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Bank Marketing — Term Deposit Predictor")
st.caption(
    "Random Forest trained on the UCI Bank Marketing dataset (45,211 calls). "
    "The `duration` field is excluded to prevent target leakage."
)

tab_predict, tab_insights, tab_model = st.tabs(
    ["Predict", "Insights", "Model Card"]
)

# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Score a hypothetical client")
    st.write(
        "Adjust the inputs below to estimate the probability that a client "
        "subscribes to a term deposit. The recommendation reflects the chosen "
        "decision threshold."
    )

    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.30,
        step=0.05,
        help=(
            "The base subscription rate is ~11%. Lowering the threshold favors "
            "recall (catch more subscribers); raising it favors precision "
            "(fewer wasted calls)."
        ),
    )

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        age = st.slider("Age", 18, 95, 41)
        balance = st.number_input("Average yearly balance (EUR)", -8000, 105000, 1500, step=100)
        job = st.selectbox("Job", schema["categorical"]["job"], index=0)
        marital = st.selectbox("Marital status", schema["categorical"]["marital"])
        education = st.selectbox("Education", schema["categorical"]["education"])

    with col_mid:
        default = st.selectbox("Has credit in default?", schema["categorical"]["default"])
        housing = st.selectbox("Has housing loan?", schema["categorical"]["housing"], index=0)
        loan = st.selectbox("Has personal loan?", schema["categorical"]["loan"])
        contact = st.selectbox("Contact channel", schema["categorical"]["contact"])
        month = st.selectbox(
            "Last contact month",
            schema["categorical"]["month"],
            index=schema["categorical"]["month"].index("may")
            if "may" in schema["categorical"]["month"]
            else 0,
        )

    with col_right:
        day = st.slider("Day of month (last contact)", 1, 31, 15)
        campaign = st.slider("Contacts during this campaign", 1, 50, 2)
        previous = st.slider("Contacts before this campaign", 0, 50, 0)
        pdays = st.slider(
            "Days since last contact (-1 = never contacted before)",
            -1,
            900,
            -1,
        )
        poutcome = st.selectbox(
            "Outcome of previous campaign", schema["categorical"]["poutcome"]
        )

    row = pd.DataFrame(
        [
            {
                "age": age,
                "balance": balance,
                "day": day,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "month": month,
                "poutcome": poutcome,
            }
        ]
    )

    proba = float(pipeline.predict_proba(row)[0, 1])
    base_rate = meta["positive_rate"]
    lift = proba / base_rate if base_rate > 0 else float("nan")

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Subscription probability", f"{proba:.1%}")
    m2.metric("Lift over base rate", f"{lift:.2f}x", help=f"Base rate: {base_rate:.1%}")
    m3.metric("Threshold", f"{threshold:.0%}")

    if proba >= threshold:
        st.success(
            f"**CALL** — predicted probability ({proba:.1%}) is at or above "
            f"the {threshold:.0%} threshold."
        )
    else:
        st.warning(
            f"**SKIP** — predicted probability ({proba:.1%}) is below the "
            f"{threshold:.0%} threshold. Better to allocate this contact "
            f"slot to a higher-scoring lead."
        )

    with st.expander("Why this prediction?"):
        st.write(
            "The model relies most heavily on whether the prospect was a "
            "**previous-campaign success** (`poutcome=success`), how recently "
            "they were last contacted (`pdays`), and their **contact channel** "
            "— cellphones convert dramatically better than `unknown` (mostly "
            "landlines). The full feature ranking is on the **Model Card** tab."
        )

# ---------------------------------------------------------------------------
# Insights tab
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Headline findings from the analysis")

    st.markdown(
        """
        These are the conversion-rate insights surfaced by the EDA notebook
        (`notebooks/notebook_1_EDA.ipynb`) on the full 45,211-row dataset.
        """
    )

    seg = pd.DataFrame(
        {
            "Segment": [
                "Previously contacted (warm)",
                "Students",
                "Retirees",
                "New leads (cold)",
                "May campaign (volume push)",
                "Overall base rate",
            ],
            "Conversion": [0.638, 0.31, 0.25, 0.093, 0.064, 0.113],
            "Notes": [
                "~7x lift vs cold leads — re-engagement is the strongest signal",
                "~3x base rate, systematically under-targeted",
                "~2x base rate, under-targeted",
                "Default cold-call performance",
                "13,769 calls placed but worst ROI of the year",
                "11.3% across the full campaign",
            ],
        }
    )
    st.dataframe(
        seg.style.format({"Conversion": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Recommendations")
    st.markdown(
        """
        1. **Prioritize re-engagement.** Warm leads convert at 63.8% versus 9.3% cold —
           a previously contacted prospect is worth ~7 cold calls.
        2. **Re-weight the segment mix.** Students and retirees together are
           ~3x the base rate, but the May campaign optimized for volume rather
           than segment quality.
        3. **Switch May from broadcast to targeted.** 13,769 May calls produced
           a 6.4% conversion — model-scored targeting at the same volume would
           materially improve campaign ROI.
        4. **Skip `unknown` contact channel when possible.** Landlines perform
           dramatically worse than cellphones in feature importance.
        """
    )

# ---------------------------------------------------------------------------
# Model card tab
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Model card")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test ROC-AUC", f"{meta['test_roc_auc']:.4f}")
    c2.metric("Test PR-AUC", f"{meta['test_pr_auc']:.4f}")
    c3.metric("Train rows", f"{meta['n_train']:,}")
    c4.metric("Test rows", f"{meta['n_test']:,}")

    st.markdown(
        """
        **Architecture.** `RandomForestClassifier(n_estimators=300, max_depth=12,
        min_samples_leaf=20, class_weight="balanced")` over a one-hot encoded
        feature matrix. No `duration` feature — see leakage note below.

        **Validation.** Stratified 80/20 split, fixed `random_state=42`. The
        original notebook (`notebook_3_MachineLearning.ipynb`) reports a Random
        Forest test AUC of ~0.7959 against the same dataset.
        """
    )

    st.markdown("### ROC curve")
    roc = meta["roc_curve"]
    roc_df = pd.DataFrame({"FPR": roc["fpr"], "TPR": roc["tpr"]})
    st.line_chart(roc_df.set_index("FPR"), height=300)

    st.markdown("### Confusion matrix (threshold = 0.50)")
    cm = np.array(meta["confusion_matrix"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: no", "Actual: yes"],
        columns=["Predicted: no", "Predicted: yes"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Top 15 features")
    feats = pd.DataFrame(meta["top_features"], columns=["Feature", "Importance"])
    st.bar_chart(feats.set_index("Feature"))

    st.warning(
        "**Leakage note.** Per Moro et al. (2014), the `duration` field is only "
        "known *after* a call ends and is therefore excluded from every "
        "predictive feature set. The original notebook also documents an "
        "earlier SMOTE-in-CV leakage bug; the fix (resampling inside an "
        "`imblearn.Pipeline`) lives on the `v2-imblearn-pipeline` branch of "
        "the parent repository."
    )
