"""
advanced_analysis_pyspark.py
----------------------------
PySpark version of advanced_analysis.py (Ola Mobility project).

  1. Area x Hour demand heatmap
  2. K-Means segmentation of pickup areas into business tiers (Spark MLlib)
  3. Statistical test: is "peak hour" a real effect or just noise? (chi-square)
  4. Cancellation-risk model (Spark MLlib Random Forest) with honest validation

Run:  spark-submit advanced_analysis_pyspark.py      (or: python advanced_analysis_pyspark.py)
Outputs: heatmap_area_hour.png, area_clusters.png, feature_importance.png, area_segments.csv
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2 as chi2_dist

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

from ola_spark_common import get_spark, load_data

spark = get_spark("OlaAdvancedAnalysis")
spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------ LOAD
df = load_data(spark, "ola_bengaluru.csv").cache()
n_rows = df.count()
dmin, dmax = df.agg(F.min("Date"), F.max("Date")).first()
print(f"Loaded {n_rows:,} bookings from {dmin} to {dmax}")

# ------------------------------------------------- 1. AREA x HOUR HEATMAP
pivot = (df.groupBy("Pickup_Location")
           .pivot("Hour", list(range(24)))
           .count()
           .fillna(0)
           .toPandas()
           .set_index("Pickup_Location"))
pivot.columns = [int(c) for c in pivot.columns]
order = pivot.sum(axis=1).sort_values(ascending=False).index

plt.figure(figsize=(14, 10))
sns.heatmap(pivot.loc[order], cmap="YlOrRd", cbar_kws={"label": "Bookings"})
plt.title("Demand Heatmap: Pickup Area vs Hour of Day")
plt.tight_layout()
plt.savefig("heatmap_area_hour.png", dpi=150)
plt.close()
print("Saved heatmap_area_hour.png")

# ------------------------------------- 2. K-MEANS SEGMENTATION OF AREAS
area = (df.groupBy("Pickup_Location")
          .agg(F.count("Booking_ID").alias("Total_Bookings"),
               F.avg("Booking_Value").alias("Avg_Fare"),
               F.avg("Cancelled").alias("Cancel_Rate"),
               F.avg("Success").alias("Success_Rate"),
               F.avg("Ride_Distance").alias("Avg_Distance"),
               F.avg("Driver_Ratings").alias("Avg_Rating"))
          .fillna(0))

features = ["Total_Bookings", "Avg_Fare", "Cancel_Rate", "Avg_Distance", "Avg_Rating"]

assembler = VectorAssembler(inputCols=features, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
area_vec = assembler.transform(area)
area_scaled = scaler.fit(area_vec).transform(area_vec).cache()

# choose k with the silhouette method rather than guessing
evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                                metricName="silhouette", distanceMeasure="squaredEuclidean")
scores = {}
for k in range(2, 7):
    preds = KMeans(k=k, seed=42, maxIter=100, featuresCol="features").fit(area_scaled) \
        .transform(area_scaled)
    scores[k] = evaluator.evaluate(preds)
best_k = max(scores, key=scores.get)
print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

km_model = KMeans(k=best_k, seed=42, maxIter=100, featuresCol="features").fit(area_scaled)
clustered = km_model.transform(area_scaled).withColumnRenamed("prediction", "Cluster")

# PCA to 2D for plotting
pca_model = PCA(k=2, inputCol="features", outputCol="pca").fit(clustered)
clustered = pca_model.transform(clustered)

# only 50 rows now -> safe to bring to pandas for naming / plotting
area_pd = clustered.select("Pickup_Location", *[c for c in features],
                           "Success_Rate", "Cluster", "pca").toPandas()
area_pd["pc1"] = area_pd["pca"].apply(lambda v: float(v[0]))
area_pd["pc2"] = area_pd["pca"].apply(lambda v: float(v[1]))
area_pd = area_pd.drop(columns="pca")

# human-readable tier names, ranked by a simple value score (demand + fare - cancellations)
mean_all = area_pd[["Total_Bookings", "Avg_Fare", "Cancel_Rate"]].mean()
value = {}
for c, g in area_pd.groupby("Cluster"):
    value[c] = (g["Total_Bookings"].mean() / mean_all["Total_Bookings"]
                + g["Avg_Fare"].mean() / mean_all["Avg_Fare"]
                - g["Cancel_Rate"].mean() / mean_all["Cancel_Rate"])
ranked = sorted(value, key=value.get, reverse=True)
labels = ["Premium Hotspot", "Steady Performer", "Watch List", "Low Priority",
          "Underperformer", "Weak Zone"]
name_map = {c: labels[i] for i, c in enumerate(ranked)}
area_pd["Segment"] = area_pd["Cluster"].map(name_map)

(area_pd[["Pickup_Location", "Total_Bookings", "Avg_Fare", "Cancel_Rate", "Success_Rate",
          "Avg_Distance", "Avg_Rating", "Cluster", "Segment"]]
    .rename(columns={"Pickup_Location": "Pickup Location"})
    .round(3).sort_values("Pickup Location").to_csv("area_segments.csv", index=False))
print(area_pd.groupby("Segment")[features].mean().round(2))

plt.figure(figsize=(9, 6))
sns.scatterplot(x=area_pd["pc1"], y=area_pd["pc2"], hue=area_pd["Segment"], s=110, palette="Set2")
for _, r in area_pd.iterrows():
    plt.annotate(r["Pickup_Location"].replace("Area-", ""), (r["pc1"], r["pc2"]),
                 fontsize=7, xytext=(3, 3), textcoords="offset points")
plt.title(f"K-Means Segmentation of Pickup Areas (k={best_k}, PCA view)")
plt.xlabel("Principal component 1"); plt.ylabel("Principal component 2")
plt.tight_layout(); plt.savefig("area_clusters.png", dpi=150); plt.close()
print("Saved area_clusters.png and area_segments.csv")

# ------------------------- 3. IS THE "PEAK HOUR" STATISTICALLY REAL?
hourly = df.groupBy("Hour").count().orderBy("Hour").toPandas().set_index("Hour")["count"]
expected = hourly.mean()                               # H0: demand is equal in every hour
chi2 = float(((hourly - expected) ** 2 / expected).sum())
p = float(chi2_dist.sf(chi2, df=len(hourly) - 1))
spread = (hourly.max() - hourly.min()) / hourly.mean() * 100
print(f"\nChi-square test of uniform hourly demand: chi2={chi2:.1f}, p={p:.4f}")
print(f"Busiest hour: {hourly.idxmax()}:00 ({hourly.max()}), quietest: "
      f"{hourly.idxmin()}:00 ({hourly.min()}); spread = {spread:.1f}% of the mean")
print("-> Demand is "
      + ("NOT uniform (peaks are statistically significant)" if p < 0.05
         else "statistically flat: peak hours are real but small / close to random variation"))

# ------------------------ 4. CANCELLATION-RISK MODEL (Random Forest)
vehicle_types = [r[0] for r in df.select("Vehicle_Type").distinct().orderBy("Vehicle_Type").collect()]

model_df = (df.select("Booking_ID", "Hour", "DayOfWeek", "IsWeekend", "Vehicle_Type",
                      "Pickup_Location", F.col("Cancelled").alias("label"))
              .withColumn("Area_ID", F.regexp_replace("Pickup_Location", "Area-", "").cast("int")))

# one-hot encode vehicle type (same as pd.get_dummies)
dummy_cols = []
for v in vehicle_types:
    cname = f"Vehicle_Type_{v}"
    model_df = model_df.withColumn(cname, (F.col("Vehicle_Type") == v).cast("int"))
    dummy_cols.append(cname)

feature_cols = ["Hour", "DayOfWeek", "IsWeekend", "Area_ID"] + dummy_cols

# stratified 75 / 25 split
train = model_df.sampleBy("label", fractions={0: 0.75, 1: 0.75}, seed=42)
test = model_df.join(train.select("Booking_ID"), on="Booking_ID", how="left_anti")

# class_weight="balanced" equivalent: n / (n_classes * n_class_samples)
n_train = train.count()
cnt = {r["label"]: r["count"] for r in train.groupBy("label").count().collect()}
w = {k: n_train / (2.0 * v) for k, v in cnt.items()}
train = train.withColumn("weight", F.when(F.col("label") == 1, w.get(1, 1.0)).otherwise(w.get(0, 1.0)))

rf = RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="weight",
                            numTrees=200, maxDepth=8, seed=42)
pipe = Pipeline(stages=[VectorAssembler(inputCols=feature_cols, outputCol="features"), rf])
model = pipe.fit(train)
pred = model.transform(test).cache()

auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                    metricName="areaUnderROC").evaluate(pred)
print(f"\nCancellation model ROC-AUC on held-out data: {auc:.3f}  (0.5 = random guessing)")

# classification report (precision / recall / F1 per class)
cm = {(r["label"], r["prediction"]): r["count"]
      for r in pred.groupBy("label", "prediction").count().collect()}
print(f"{'':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
for cls in (0, 1):
    tp = cm.get((cls, float(cls)), 0)
    fp = sum(v for (l, pr), v in cm.items() if pr == float(cls) and l != cls)
    fn = sum(v for (l, pr), v in cm.items() if l == cls and pr != float(cls))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"{cls:>10}{prec:>11.3f}{rec:>9.3f}{f1:>9.3f}{tp + fn:>9}")

imp = pd.Series(model.stages[-1].featureImportances.toArray(), index=feature_cols).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#FFD100", edgecolor="black")
plt.title(f"Cancellation Model - Feature Importance (AUC = {auc:.2f})")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()
print("Saved feature_importance.png")
print("Interpretation: an AUC near 0.5 means time, area and vehicle type alone do not "
      "predict cancellations - they are driven by factors not in the dataset "
      "(driver availability, surge pricing, weather, traffic).")

spark.stop()
