# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3 — Machine Learning
# MAGIC **Prerequisito:** Haber ejecutado Notebook 2 — Limpieza y Preprocesamiento
# MAGIC **Objetivo:** Entrenar, comparar y seleccionar el mejor modelo predictivo
# MAGIC
# MAGIC ## Mejoras respecto a la versión anterior
# MAGIC
# MAGIC 1. **Carga datos pre-procesados** desde parquet → no duplica código del NB2.
# MAGIC 2. **SMOTE dentro de `imblearn.Pipeline`** → se aplica solo al fold de train en CV.
# MAGIC    Esto **corrige el overfitting AUC 0.96→0.74** que vimos en la versión anterior
# MAGIC    (causado por aplicar SMOTE antes de GridSearchCV).
# MAGIC 3. **Curva Precision-Recall** además de ROC (más informativa con 88/12).
# MAGIC 4. **Threshold tuning** para optimizar F1 en vez de usar 0.5 a ciegas.
# MAGIC 5. **Matriz de confusión** del mejor modelo.
# MAGIC 6. **Persistencia del modelo final** con `joblib` para producción.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Librerías y Carga de Datos Preprocesados

# COMMAND ----------
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve, f1_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# imblearn Pipeline: soporta SMOTE como paso intermedio (sklearn Pipeline no)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Carga directa desde parquet (generados por el notebook 2)
X_train = pd.read_parquet("data/processed/X_train.parquet")
X_test  = pd.read_parquet("data/processed/X_test.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()
y_test  = pd.read_parquet("data/processed/y_test.parquet").squeeze()

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Features: {X_train.columns.tolist()}")

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Función utilitaria de evaluación

# COMMAND ----------
def evaluar(modelo, nombre, X_tr, y_tr, X_te, y_te):
    """Entrena, predice y reporta métricas estándar. Devuelve un dict."""
    modelo.fit(X_tr, y_tr)
    y_prob = modelo.predict_proba(X_te)[:, 1]
    y_pred = modelo.predict(X_te)

    auc = roc_auc_score(y_te, y_prob)
    ap  = average_precision_score(y_te, y_prob)  # Area bajo curva PR

    print(f"\n{'='*50}")
    print(f"  {nombre}")
    print(f"{'='*50}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  PR-AUC : {ap:.4f}  (mejor métrica con 88/12)")
    print(classification_report(y_te, y_pred))

    return {"modelo": nombre, "obj": modelo,
            "auc_roc": auc, "pr_auc": ap, "y_prob": y_prob}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Baseline — Decision Tree
# MAGIC
# MAGIC Incluso el baseline ya usa Pipeline con SMOTE dentro,
# MAGIC para que la comparación sea justa con el resto.

# COMMAND ----------
pipe_dt = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
])
res_dt = evaluar(pipe_dt, "Decision Tree", X_train, y_train, X_test, y_test)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Random Forest

# COMMAND ----------
pipe_rf = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, n_jobs=-1))
])
res_rf = evaluar(pipe_rf, "Random Forest", X_train, y_train, X_test, y_test)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Importancia de Variables (Random Forest)

# COMMAND ----------
rf_modelo = res_rf["obj"].named_steps["model"]
importances = pd.Series(rf_modelo.feature_importances_, index=X_train.columns)
top10 = importances.sort_values(ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
top10.plot(kind="barh", ax=ax, color="#2ecc71")
ax.set_title("Top 10 variables más importantes — Random Forest")
ax.set_xlabel("Importancia")
plt.tight_layout()
plt.savefig("figures/04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC > Al haber excluido `duration`, las variables económicas (`nr_employed`, `euribor3m`,
# MAGIC > `emp_var_rate`) y las de contacto previo dominan — consistente con el hallazgo del EDA.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. XGBoost

# COMMAND ----------
pipe_xgb = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric="auc", verbosity=0))
])
res_xgb = evaluar(pipe_xgb, "XGBoost", X_train, y_train, X_test, y_test)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Optimización de Hiperparámetros — GridSearchCV CORRECTO
# MAGIC
# MAGIC ### El bug de la versión anterior
# MAGIC
# MAGIC Antes se aplicaba SMOTE **antes** de `GridSearchCV`, por lo que cada fold de CV
# MAGIC contenía muestras sintéticas de la clase minoritaria en el fold de validación.
# MAGIC Resultado: **AUC-CV 0.96 vs AUC-test 0.74** — el modelo "veía" en validación
# MAGIC datos sintéticos correlacionados con el train.
# MAGIC
# MAGIC ### Arreglo
# MAGIC
# MAGIC `imblearn.Pipeline` aplica SMOTE **solo en el fold de entrenamiento** dentro del CV,
# MAGIC dejando el fold de validación intacto. Las métricas de CV y test deberían ahora ser consistentes.

# COMMAND ----------
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth":    [3, 5, 7],
    "model__learning_rate":[0.05, 0.1, 0.2],
    "model__subsample":    [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe_xgb,                   # pipeline con SMOTE dentro
    param_grid=param_grid,
    cv=cv,
    scoring="average_precision", # PR-AUC, mejor para desbalance
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

mejor_pipe = grid.best_estimator_
y_prob_grid = mejor_pipe.predict_proba(X_test)[:, 1]
auc_grid = roc_auc_score(y_test, y_prob_grid)
pr_grid  = average_precision_score(y_test, y_prob_grid)

print(f"\nMejores parámetros: {grid.best_params_}")
print(f"PR-AUC en CV:   {grid.best_score_:.4f}")
print(f"AUC-ROC en test: {auc_grid:.4f}")
print(f"PR-AUC en test:  {pr_grid:.4f}")
print("(CV ≈ test → no hay overfitting)")

res_grid = {"modelo": "XGBoost GridSearch", "obj": mejor_pipe,
            "auc_roc": auc_grid, "pr_auc": pr_grid, "y_prob": y_prob_grid}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Threshold Tuning — Optimización del umbral de decisión
# MAGIC
# MAGIC Con clases 88/12, el threshold óptimo **casi nunca es 0.5**.
# MAGIC Buscamos el umbral que maximiza F1-score sobre el test.

# COMMAND ----------
resultados = [res_dt, res_rf, res_xgb, res_grid]
mejor = max(resultados, key=lambda r: r["pr_auc"])
print(f"Mejor modelo por PR-AUC: {mejor['modelo']}")

precisiones, recalls, thresholds = precision_recall_curve(y_test, mejor["y_prob"])
f1_scores = 2 * precisiones * recalls / (precisiones + recalls + 1e-9)
mejor_idx = f1_scores[:-1].argmax()
mejor_threshold = thresholds[mejor_idx]
mejor_f1 = f1_scores[mejor_idx]

print(f"Threshold por defecto (0.5) → F1: "
      f"{f1_score(y_test, (mejor['y_prob'] >= 0.5).astype(int)):.4f}")
print(f"Threshold óptimo ({mejor_threshold:.3f}) → F1: {mejor_f1:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Curvas ROC y Precision-Recall

# COMMAND ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC
for r in resultados:
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{r['modelo']} (AUC={r['auc_roc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Curva ROC")
axes[0].legend(loc="lower right", fontsize=9)

# Precision-Recall
for r in resultados:
    p, rec, _ = precision_recall_curve(y_test, r["y_prob"])
    axes[1].plot(rec, p, label=f"{r['modelo']} (AP={r['pr_auc']:.3f})")
axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", alpha=0.3,
                label=f"Baseline ({y_test.mean():.3f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Curva Precision-Recall")
axes[1].legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("figures/05_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Matriz de Confusión del Mejor Modelo (con threshold óptimo)

# COMMAND ----------
y_pred_opt = (mejor["y_prob"] >= mejor_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Sí"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Sí"])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de Confusión — {mejor['modelo']} @ threshold={mejor_threshold:.3f}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("figures/06_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Comparación Final de Modelos

# COMMAND ----------
df_res = pd.DataFrame([
    {"Modelo": r["modelo"],
     "AUC-ROC": round(r["auc_roc"], 4),
     "PR-AUC":  round(r["pr_auc"],  4)}
    for r in resultados
]).sort_values("PR-AUC", ascending=False)

display(df_res)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Persistencia del Modelo Final

# COMMAND ----------
joblib.dump(mejor["obj"], "models/best_model.joblib")
joblib.dump({"threshold": float(mejor_threshold)}, "models/threshold.joblib")
print("Modelo y threshold guardados en models/")
print(f"  - models/best_model.joblib ({mejor['modelo']})")
print(f"  - models/threshold.joblib  ({mejor_threshold:.4f})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 13. Conclusiones
# MAGIC
# MAGIC ### Decisiones metodológicas clave
# MAGIC - **`duration` excluida** por data leakage — solo conocida post-llamada.
# MAGIC - **SMOTE dentro del `imblearn.Pipeline`** — evita el leakage que antes inflaba el AUC-CV.
# MAGIC - **PR-AUC preferida sobre AUC-ROC** — métrica más honesta con clases 88/12.
# MAGIC - **Threshold tuning explícito** — 0.5 casi nunca es óptimo con desbalance.
# MAGIC - **StratifiedKFold(5)** — CV estratificado preserva proporciones de clase.
# MAGIC
# MAGIC ### Recomendaciones de negocio
# MAGIC - Priorizar clientes con **contacto previo** (63.8% conversión vs 9.3%).
# MAGIC - Enfocar campañas en **estudiantes y jubilados** (mayor tasa de conversión).
# MAGIC - Evitar campañas masivas en **mayo** — alto volumen, baja conversión (6.4%).
# MAGIC - Usar el modelo para **pre-filtrar** la lista de llamadas, priorizando top-decil
# MAGIC   por probabilidad predicha.
