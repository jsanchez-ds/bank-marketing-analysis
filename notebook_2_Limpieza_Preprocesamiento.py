# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2 — Limpieza y Preprocesamiento
# MAGIC **Prerequisito:** Haber ejecutado Notebook 1 — EDA
# MAGIC **Objetivo:** Preparar los datos para modelamiento y **guardarlos en disco**
# MAGIC para que el notebook 3 no tenga que repetir todo el preprocesamiento.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Librerías e Ingesta

# COMMAND ----------
import os
import pickle
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

spark = SparkSession.builder.getOrCreate()

try:
    df_pd = spark.table("bank_additional_full").toPandas()
    print("Cargado desde tabla Databricks.")
except Exception:
    df_pd = pd.read_csv("data/bank-additional-full.csv", sep=";")
    print("Cargado desde CSV local.")

print(f"Shape original: {df_pd.shape}")

# Crear carpeta de salida si no existe
os.makedirs("data/processed", exist_ok=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Limpieza de Nombres de Columnas

# COMMAND ----------
# Los puntos en nombres de columnas generan conflictos en Spark y pandas
df_pd.columns = [c.replace(".", "_") for c in df_pd.columns]
print("Columnas limpias:")
print(df_pd.columns.tolist())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Tratamiento de Valores Especiales

# COMMAND ----------
# pdays = 999 es un valor codificado que significa "nunca contactado"
# Se reemplaza por -1 para que los modelos lo traten como categoría especial
# (no usamos NaN porque SMOTE no acepta valores nulos)
df_pd["pdays"] = df_pd["pdays"].replace(999, -1)

print(f"Valores únicos en pdays (top 10): {sorted(df_pd['pdays'].unique())[:10]}")
print(f"Clientes nunca contactados: {(df_pd['pdays'] == -1).sum()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Feature Engineering (ANTES del split y SMOTE)
# MAGIC
# MAGIC Creamos las variables derivadas ahora, sobre el dataset completo, para que:
# MAGIC - SMOTE interpole sobre un espacio de features consistente.
# MAGIC - Train y test queden con exactamente las mismas columnas.

# COMMAND ----------
df_pd["fue_contactado"]      = (df_pd["pdays"] != -1).astype(int)
df_pd["contacto_intensivo"]  = (df_pd["campaign"] > 3).astype(int)
df_pd["economia_favorable"]  = (df_pd["emp_var_rate"] < 0).astype(int)

print("Features derivadas creadas:")
print(df_pd[["fue_contactado", "contacto_intensivo", "economia_favorable"]].describe())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Encoding de Variables Categóricas
# MAGIC
# MAGIC **Decisión de diseño:** usamos `LabelEncoder` porque los modelos finales serán
# MAGIC basados en árboles (Random Forest, XGBoost), que no asumen orden entre categorías.
# MAGIC Para modelos lineales (Logistic Regression) la alternativa correcta sería `OneHotEncoder`.
# MAGIC
# MAGIC **Mejora vs versión anterior:** guardamos un **diccionario de encoders**, uno por columna,
# MAGIC para poder transformar datos nuevos de forma consistente (antes se reutilizaba un único
# MAGIC `LabelEncoder` en un loop, lo que impedía reaplicar el encoding a datos nuevos).

# COMMAND ----------
cat_cols = ["job", "marital", "education", "contact",
            "month", "day_of_week", "poutcome",
            "default", "housing", "loan"]

encoders = {}
for c in cat_cols:
    encoder = LabelEncoder()
    df_pd[c] = encoder.fit_transform(df_pd[c].astype(str))
    encoders[c] = encoder

print("Encoding completado. Tipos de datos:")
print(df_pd.dtypes)

# Guardar encoders para producción
with open("data/processed/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("\nEncoders guardados en data/processed/encoders.pkl")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Encoding de Variable Objetivo

# COMMAND ----------
df_pd["y"] = (df_pd["y"] == "yes").astype(int)
print(f"Distribución variable objetivo:\n{df_pd['y'].value_counts()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Eliminación de Variable con Data Leakage
# MAGIC
# MAGIC `duration` (duración de la llamada) solo se conoce **después** de la llamada,
# MAGIC es decir, después de que el cliente ya tomó su decisión.
# MAGIC Usarla para predecir sería trampa (data leakage) — el modelo no sería válido en producción.
# MAGIC Referencia: documentación oficial del dataset UCI (Moro et al., 2014).

# COMMAND ----------
X = df_pd.drop(["y", "duration"], axis=1)
y = df_pd["y"]

print(f"Features utilizadas ({len(X.columns)}):")
print(X.columns.tolist())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. División Train / Test (estratificada)

# COMMAND ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} registros")
print(f"Test:  {X_test.shape[0]} registros")
print(f"\nDistribución en train:\n{y_train.value_counts(normalize=True).round(4)}")
print(f"\nDistribución en test:\n{y_test.value_counts(normalize=True).round(4)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Persistencia de datasets preprocesados
# MAGIC
# MAGIC **Mejora importante:** guardamos los datos listos para modelar, así el Notebook 3
# MAGIC no necesita repetir todo el preprocesamiento (antes lo hacía, duplicando ~40 líneas de código).
# MAGIC
# MAGIC **Nota crítica:** NO aplicamos SMOTE aquí. SMOTE se aplica **dentro del pipeline de CV
# MAGIC en el Notebook 3** para evitar leakage de muestras sintéticas hacia los folds de validación.
# MAGIC Guardar SMOTE pre-aplicado fue el bug que generó el overfitting AUC 0.96→0.74 en la versión anterior.

# COMMAND ----------
X_train.to_parquet("data/processed/X_train.parquet")
X_test.to_parquet("data/processed/X_test.parquet")
y_train.to_frame().to_parquet("data/processed/y_train.parquet")
y_test.to_frame().to_parquet("data/processed/y_test.parquet")

print("Datasets guardados en data/processed/")
for f in sorted(os.listdir("data/processed")):
    size_kb = os.path.getsize(f"data/processed/{f}") / 1024
    print(f"  {f}  ({size_kb:.1f} KB)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Resumen del Preprocesamiento
# MAGIC
# MAGIC | Paso | Acción | Motivo |
# MAGIC |---|---|---|
# MAGIC | Nombres columnas | Reemplazar puntos por `_` | Compatibilidad Spark/pandas |
# MAGIC | pdays = 999 | Reemplazar por -1 | Valor codificado, no numérico real |
# MAGIC | Feature engineering | 3 variables nuevas ANTES del split | Evitar inconsistencias con SMOTE |
# MAGIC | Categóricas | LabelEncoder **con dict persistente** | Permite transformar datos nuevos |
# MAGIC | duration | Eliminada | Data leakage |
# MAGIC | Split | 80% train / 20% test, estratificado | Preservar proporción de clases |
# MAGIC | SMOTE | **NO aquí — dentro del Pipeline en NB3** | Evitar leakage en CV |
# MAGIC | Persistencia | Parquet en `data/processed/` | Evitar reejecutar el preprocessing |
