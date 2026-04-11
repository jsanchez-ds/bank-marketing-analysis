# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1 — Análisis Exploratorio de Datos (EDA)
# MAGIC **Dataset:** Bank Marketing — UCI Machine Learning Repository
# MAGIC **Objetivo:** Predecir si un cliente suscribirá un depósito a plazo (variable `y`)
# MAGIC **Autor:** Jonathan Sánchez Pesantes

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Configuración e Ingesta
# MAGIC
# MAGIC El notebook funciona tanto en Databricks (lee desde Unity Catalog / Hive metastore)
# MAGIC como localmente (lee un CSV). Esto mejora la portabilidad y hace el código reproducible.

# COMMAND ----------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum, count, avg, round as spark_round

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "8")

# Intento de carga: primero tabla Databricks, fallback a CSV local
try:
    df = spark.table("bank_additional_full")
    print("Cargado desde tabla Databricks.")
except Exception:
    df = spark.read.csv(
        "data/bank-additional-full.csv",
        header=True, inferSchema=True, sep=";"
    )
    print("Cargado desde CSV local.")

# Limpiar nombres de columnas (los puntos generan conflictos en Spark)
for col_name in df.columns:
    if "." in col_name:
        df = df.withColumnRenamed(col_name, col_name.replace(".", "_"))

print(f"Filas: {df.count()} | Columnas: {len(df.columns)}")
df.printSchema()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Vista General del Dataset

# COMMAND ----------
display(df.describe())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Variable Objetivo — Desbalance de Clases

# COMMAND ----------
target_dist = (
    df.groupBy("y")
    .count()
    .withColumn("porcentaje_%", spark_round(col("count") / df.count() * 100, 2))
    .orderBy("count", ascending=False)
)
display(target_dist)

# Gráfico portable (funciona fuera de Databricks)
target_pd = target_dist.toPandas()
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(target_pd["y"], target_pd["porcentaje_%"],
              color=["#e74c3c", "#2ecc71"])
ax.set_ylabel("Porcentaje (%)")
ax.set_title("Distribución de la variable objetivo")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, pct in zip(bars, target_pd["porcentaje_%"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{pct}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC > **Hallazgo:** Dataset fuertemente desbalanceado — 88.7% no suscribe vs 11.3% que sí.
# MAGIC > Esto debe tratarse antes de modelar para evitar que el modelo aprenda a predecir siempre "no".

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Tasa de Conversión por Segmento

# COMMAND ----------
# Tasa de conversión por tipo de trabajo
job_rate = (
    df.groupBy("job")
    .agg(
        count("*").alias("total"),
        spark_sum(when(col("y") == "yes", 1).otherwise(0)).alias("suscritos"),
        spark_round(avg(when(col("y") == "yes", 1).otherwise(0)) * 100, 2).alias("tasa_%")
    )
    .orderBy("tasa_%", ascending=False)
)
display(job_rate)

# Visualización
job_pd = job_rate.toPandas()
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(job_pd["job"], job_pd["tasa_%"], color="#3498db")
ax.axvline(x=11.3, color="red", linestyle="--", label="Promedio global (11.3%)")
ax.set_xlabel("Tasa de conversión (%)")
ax.set_title("Tasa de conversión por profesión")
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.savefig("figures/02_conversion_by_job.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC > **Hallazgo:** Estudiantes (31%) y jubilados (25%) tienen la mayor tasa de conversión,
# MAGIC > casi 3x el promedio. Blue-collar tiene la menor (6.9%).

# COMMAND ----------
# Tasa de conversión por mes
month_rate = (
    df.groupBy("month")
    .agg(
        count("*").alias("total"),
        spark_round(avg(when(col("y") == "yes", 1).otherwise(0)) * 100, 2).alias("tasa_%")
    )
    .orderBy("tasa_%", ascending=False)
)
display(month_rate)

# COMMAND ----------
# MAGIC %md
# MAGIC > **Hallazgo:** Marzo, diciembre y septiembre tienen tasas ~45-50%, pero volúmenes bajos.
# MAGIC > Mayo concentra el mayor volumen (13.769 llamadas) pero solo 6.4% de conversión.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Impacto del Contacto Previo — El hallazgo más importante

# COMMAND ----------
# pdays = 999 significa "nunca contactado"
df_pdays = df.withColumn(
    "nunca_contactado",
    when(col("pdays") == 999, "Sí").otherwise("No")
)

contact_rate = (
    df_pdays.groupBy("nunca_contactado")
    .agg(
        count("*").alias("total"),
        spark_round(avg(when(col("y") == "yes", 1).otherwise(0)) * 100, 2).alias("tasa_%")
    )
    .orderBy("nunca_contactado")
)
display(contact_rate)

# Gráfico comparativo
contact_pd = contact_rate.toPandas()
fig, ax = plt.subplots(figsize=(7, 5))
colors_bar = ["#2ecc71", "#e74c3c"]
bars = ax.bar(contact_pd["nunca_contactado"], contact_pd["tasa_%"], color=colors_bar)
ax.set_ylabel("Tasa de conversión (%)")
ax.set_title("Impacto del contacto previo en la conversión")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, pct in zip(bars, contact_pd["tasa_%"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{pct}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/03_contact_impact.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC > **Hallazgo más importante del EDA:**
# MAGIC > Clientes contactados previamente convierten al **63.8%** vs **9.3%** los nuevos.
# MAGIC > Un cliente con contacto previo tiene **7 veces más probabilidad** de suscribir.
# MAGIC > **Recomendación de negocio:** agotar lista de clientes con contacto previo antes de llamar nuevos.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Resumen de Hallazgos EDA
# MAGIC
# MAGIC | Hallazgo | Detalle |
# MAGIC |---|---|
# MAGIC | Desbalance de clases | 88.7% no / 11.3% sí — requiere tratamiento (SMOTE) |
# MAGIC | Segmento con mayor conversión | Estudiantes (31%) y jubilados (25%) |
# MAGIC | Mejor mes | Marzo, diciembre, septiembre (~45-50%) |
# MAGIC | Variable más discriminante | Contacto previo: 63.8% vs 9.3% |
# MAGIC | Variable con data leakage | `duration` — solo conocida post-llamada, **no usar en modelo** |
