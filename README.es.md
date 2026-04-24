🌐 [English](README.md) · **Español**

# 🏦 Análisis de Campañas de Marketing Bancario

[![Streamlit Demo](https://img.shields.io/badge/Demo_Vivo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://bank-marketing-analysis-jsanchez.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RF-F7931E?logo=scikitlearn&logoColor=white)

> **[→ Prueba el demo interactivo](https://bank-marketing-analysis-jsanchez.streamlit.app/)** — scorea cualquier perfil de cliente, explora segmentos y revisa el desempeño del modelo.

Análisis de campañas de marketing directo de una entidad bancaria portuguesa para predecir si un cliente se va a suscribir a un depósito a plazo. El proyecto combina **análisis exploratorio de datos**, **tests estadísticos**, **segmentación de clientes** y **modelamiento predictivo** para derivar insights accionables de negocio.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5-E25A1C?logo=apachespark&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## 🎯 Objetivo

> **Pregunta de negocio:** ¿Qué perfiles de cliente tienen más probabilidad de suscribir un depósito a plazo, y cómo puede el banco optimizar la estrategia de campaña para aumentar la tasa de conversión?

El proyecto aborda la pregunta desde múltiples ángulos:
- **Descriptivo**: ¿Quiénes son los clientes que suscriben? ¿Qué patrones de campaña impulsan la conversión?
- **Inferencial**: ¿Son las diferencias de tasa de conversión entre segmentos estadísticamente significativas?
- **Predictivo**: ¿Podemos construir un modelo para identificar suscriptores de alta probabilidad antes de llamarlos?
- **Prescriptivo**: ¿Qué recomendaciones accionables mejoran el ROI de la campaña?

---

## 📊 Dataset

| Feature | Detalle |
|---------|---------|
| **Fuente** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Referencia** | Moro et al. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing* |
| **Registros** | 45,211 |
| **Features** | 16 input + 1 target |
| **Target** | `y` — ¿Suscribió el cliente a un depósito a plazo? (yes/no) |
| **Balance de clases** | 88.7% No / 11.3% Yes (desbalanceado) |

**Variables clave:** age, job, marital status, education, balance, housing loan, contact type, campaign duration, number of contacts, days since previous contact, previous campaign outcome.

> ⚠️ **Nota sobre `duration`:** Según la documentación de UCI y Moro et al. (2014), el campo `duration` solo se conoce *después* de que termina la llamada, por lo que se excluye de todos los modelos predictivos para evitar data leakage.

---

## 🔬 Metodología

El análisis está estructurado como un pipeline Databricks con tres notebooks:

```
notebook_1_EDA.ipynb              notebook_2_Limpieza_Preprocesamiento.ipynb
  ├── Schema & quality checks       ├── Encoding (Label / OHE)
  ├── Análisis univariado           ├── Feature engineering
  ├── Análisis bivariado            ├── Split train/test estratificado
  ├── Segmentos por conv. rate      └── Persistido como parquet
  └── Tests estadísticos

notebook_3_MachineLearning.ipynb
  ├── Decision Tree (baseline)
  ├── Random Forest + GridSearch
  ├── XGBoost + GridSearch
  ├── Stratified K-Fold CV
  └── Evaluación ROC-AUC en test hold-out
```

**Stack:** PySpark para procesamiento de datos, scikit-learn + XGBoost para modelamiento, matplotlib/seaborn para visualización.

---

## 📈 Resultados principales

### Desempeño del modelo (Test hold-out, ROC-AUC)

| Modelo | CV AUC | Test AUC | Notas |
|---|---|---|---|
| Decision Tree (max_depth=5) | — | **0.7745** | Baseline interpretable |
| XGBoost (default) | — | **0.7819** | Fuerte out-of-the-box |
| XGBoost (GridSearch) | 0.9611 | **0.7647** | ⚠️ Gap CV-test revela leakage |
| **Random Forest (best)** | — | **0.7959** ★ | Mejor generalización |

> **Hallazgo crítico:** El grid search de XGBoost reportó AUC ≈ 0.96 en CV pero solo logró 0.76 en el test hold-out. Causa raíz: SMOTE se aplicó sobre todo el set de training *antes* del CV, filtrando muestras sintéticas entre folds. El fix — aplicar SMOTE dentro de un `imblearn.Pipeline` para que cada fold resamplee independientemente — está implementado en la [iteración v2](#-iteración-v2). Esto es exactamente el tipo de bug de metodología que un pipeline productivo debe detectar.

### Hallazgos principales

1. **Desbalance severo de clases (88.7% / 11.3%)** — accuracy engaña; ROC-AUC y PR-AUC son las métricas relevantes, y el resampling tiene que respetar el CV.
2. **El contacto previo es la señal más fuerte** — clientes contactados previamente convierten al **63.8%** vs **9.3%** de clientes nuevos (≈ **7× lift**). Las campañas de re-engagement deberían ser prioridad.
3. **Segmentos de alta conversión**: estudiantes (**31%**) y jubilados (**25%**) convierten aproximadamente **3× la tasa base** (11.3%). Ambos segmentos están sistemáticamente sub-targeteados en el mix actual de campaña.
4. **Ineficiencia del canal en mayo**: 13,769 llamadas realizadas pero solo **6.4%** convirtió — el peor ROI del año. El scheduling basado en volumen fue superado por el targeting basado en calidad.
5. **Top features predictivas** (importancia en Random Forest): `poutcome_success`, `pdays`, `previous`, `month`, `contact_type`, `balance`.

---

## 🗂️ Estructura del proyecto

```
bank-marketing-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook_1_EDA.ipynb
├── notebook_2_Limpieza_Preprocesamiento.ipynb
└── notebook_3_MachineLearning.ipynb
```

La rama `databricks-version` contiene el formato `.py` original de Databricks para import directo a un workspace.

---

## 🔁 Iteración v2

Una segunda versión del pipeline de modelamiento vive en la rama [`v2-imblearn-pipeline`](https://github.com/jsanchez-ds/bank-marketing-analysis/tree/v2-imblearn-pipeline). Resuelve el leakage de SMOTE-en-CV diagnosticado arriba envolviendo el resampling y el clasificador en un `imblearn.pipeline.Pipeline`, y además agrega:

- Ajuste de threshold optimizado para F1 / costo de negocio
- PR-AUC junto con ROC-AUC (más honesto bajo desbalance fuerte)
- Persistencia de modelo + encoders vía `joblib`
- `requirements.txt` pinneado para reproducibilidad

---

## 🛠️ Stack técnico

`Python` `PySpark` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `imbalanced-learn` `Matplotlib` `Seaborn`

---

## 🚀 Cómo reproducir

```bash
# 1. Clonar el repo
git clone https://github.com/jsanchez-ds/bank-marketing-analysis.git
cd bank-marketing-analysis

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el dataset desde UCI
#    https://archive.ics.uci.edu/dataset/222/bank+marketing

# 4. Correr los notebooks en orden
jupyter notebook notebook_1_EDA.ipynb
```

Para correr en Databricks, importa los archivos `.py` desde la rama `databricks-version`.

---

## Demo en vivo (Streamlit)

**[→ Pruébalo en vivo](https://bank-marketing-analysis-jsanchez.streamlit.app/)**

Una app Streamlit interactiva envuelve el clasificador Random Forest y lo expone como herramienta de scoring con tres pestañas:

- **Predict** — scorea un cliente hipotético con threshold ajustable y recomendación CALL/SKIP.
- **Insights** — hallazgos principales de tasa de conversión (warm leads ~7× lift, estudiantes ~3× tasa base, ROI de campaña en mayo).
- **Model Card** — ROC-AUC / PR-AUC hold-out, curva ROC, matriz de confusión, top 15 features, y la caveat sobre el leakage de `duration`.

Correr localmente:

```bash
pip install -r app/requirements.txt
python app/train_model.py            # una vez, ~30s
streamlit run app/streamlit_app.py
```

Ver [`app/README.md`](app/README.md) para la guía completa y pasos de deploy a Streamlit Community Cloud.

---

## 👤 Autor

**Jonathan Sánchez**
- GitHub: [@jsanchez-ds](https://github.com/jsanchez-ds)
- Universidad de Chile — Ingeniería Civil Industrial
