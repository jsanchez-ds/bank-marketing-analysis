# Bank Marketing — Predicción de Suscripción a Depósito a Plazo

Análisis completo (EDA, preprocesamiento, ML) sobre el dataset **UCI Bank Marketing**
de campañas telefónicas de un banco portugués. Predice si un cliente contratará
un depósito a plazo.

**Stack:** PySpark · pandas · scikit-learn · imbalanced-learn · XGBoost · matplotlib
**Entorno:** Databricks (portable a local)

---

## Resultados

| Modelo | AUC-ROC | PR-AUC | Observación |
|---|---|---|---|
| Decision Tree | 0.73 | — | Baseline |
| Random Forest | **0.77** | — | Mejor modelo previo |
| XGBoost sin tuning | 0.75 | — | — |
| XGBoost GridSearch (con SMOTE en pipeline) | *pendiente* | *pendiente* | Overfitting corregido |

> Métricas a repoblar tras ejecutar la versión corregida end-to-end.

### Hallazgos de negocio

1. **Contacto previo es la señal más fuerte** — clientes ya contactados convierten
   al 63.8% vs 9.3% de clientes nuevos (7x más probabilidad).
2. **Segmentos top** — estudiantes (31%) y jubilados (25%), casi 3x el promedio global.
3. **Peor mes vs mejor mes** — mayo concentra 13k llamadas con solo 6.4% de conversión;
   marzo/diciembre/septiembre llegan a ~45-50% pero con volumen bajo.
4. **Variable descartada** — `duration` genera data leakage (solo conocida post-llamada).

---

## Estructura del repositorio

```
bank-marketing-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md              # Instrucciones de descarga
│   └── processed/             # Parquet generados por NB2 (no versionado)
├── notebooks/
│   ├── notebook_1_EDA.py                       # Databricks source format
│   ├── notebook_2_Limpieza_Preprocesamiento.py
│   └── notebook_3_MachineLearning.py
├── figures/                   # PNGs generados por los notebooks
└── models/                    # joblib del mejor modelo (no versionado)
```

---

## Cómo reproducirlo

### En Databricks
1. Importa los `.py` al workspace — Databricks los reconoce como notebooks.
2. Sube el CSV como tabla `bank_additional_full` (Unity Catalog o Hive metastore).
3. Ejecuta en orden: NB1 → NB2 → NB3.

### Localmente
```bash
pip install -r requirements.txt
# Descarga el CSV siguiendo data/README.md
jupytext --to notebook notebooks/*.py   # convierte a .ipynb
jupyter notebook notebooks/
```

---

## Decisiones técnicas clave

- **SMOTE dentro de `imblearn.Pipeline`** — se aplica solo al fold de train en CV,
  evitando leakage de muestras sintéticas al fold de validación.
- **PR-AUC sobre AUC-ROC** como métrica de selección — más honesta con clases 88/12.
- **Threshold tuning** — buscar el umbral que maximiza F1 en test, no asumir 0.5.
- **Feature engineering antes del split** — `fue_contactado`, `contacto_intensivo`,
  `economia_favorable`, coherentes para train y test.
- **Persistencia intermedia** — NB2 guarda parquet, NB3 no re-preprocesa.

---

## Autor

**Jonathan Sánchez Pesantes**
- GitHub: [@Jonathan742001](https://github.com/Jonathan742001)
