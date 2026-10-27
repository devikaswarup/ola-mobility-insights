# 🚕 Ola Mobility Insights (PySpark)

Demand, cancellation and driver-dispatch analytics for Ola bookings in Bengaluru, built with **PySpark** and **Streamlit**. All filtering and aggregation run in Spark; only the small aggregated results are converted to pandas for charting.

## Features

**Interactive dashboard (`app_pyspark.py`)**
- **Overview**: total bookings, completion and cancellation rates, revenue, average fare, distance, driver rating and pickup ETA, plus daily bookings, revenue by vehicle type, booking outcomes, payment methods and a chi-square test of whether the hourly peak is real
- **Area → Peak Time**: pick a pickup area and see its busiest hour
- **Time → Peak Area**: pick an hour and see the busiest pickup areas
- **Heatmap**: area × hour demand heatmap and weekday vs weekend pattern
- **Cancellations**: reasons drivers and customers cancel, highest and lowest cancellation-rate areas, cancellation rate by hour
- **Dispatch Planner**: recommends where to stage idle drivers at a given hour (score = 50% demand + 30% average fare + 20% reliability)

**Advanced analysis (`advanced_analysis_pyspark.py`)**
1. Area × hour demand heatmap
2. K-Means segmentation of pickup areas into business tiers (Spark MLlib, k chosen by silhouette score)
3. Chi-square test for uniform hourly demand
4. Cancellation-risk model (Spark MLlib Random Forest) with held-out ROC-AUC and feature importance

## Project structure

```
├── app_pyspark.py                 # Streamlit dashboard
├── advanced_analysis_pyspark.py   # Segmentation, statistical test, ML model
├── ola_spark_common.py            # Spark session, data loading/cleaning, dispatch scoring
├── ola_bengaluru.csv              # Dataset (place in the project root)
├── requirements.txt
└── README.md
```

## Setup

**Prerequisites:** Python 3.9+ and Java 8, 11 or 17 (required by Spark).

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the dashboard:

```bash
streamlit run app_pyspark.py
```

Run the advanced analysis:

```bash
python advanced_analysis_pyspark.py
# or
spark-submit advanced_analysis_pyspark.py
```

Outputs written to the project folder:

| File | Description |
|---|---|
| `area_segments.csv` | Each pickup area with its cluster and segment name |
| `heatmap_area_hour.png` | Pickup area vs hour demand heatmap |
| `area_clusters.png` | K-Means segments (PCA view) |
| `feature_importance.png` | Cancellation model feature importance |

## Tech stack

PySpark (DataFrames, MLlib) · Streamlit · Plotly · pandas · SciPy · Matplotlib · Seaborn

## Notes

- Spark's `StandardScaler` uses sample standard deviation, so cluster assignments can differ slightly from a scikit-learn version.
- The train/test split is stratified but approximate (75/25).
- A ROC-AUC close to 0.5 means time, area and vehicle type alone do not predict cancellations. Those are likely driven by factors not in the dataset, such as driver availability, surge pricing, weather and traffic.

## Deploying on Streamlit Community Cloud

Add a `packages.txt` file containing one line, `default-jre`, so Java is installed alongside the Python dependencies.

## Data source

Bengaluru Ola Cabs dataset · Urban Mobility Optimization Project
