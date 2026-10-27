"""
app_pyspark.py
--------------
Ola Mobility Insights dashboard, with all data crunching done in PySpark.
Spark does the filtering / aggregation; only the small aggregated results are
converted to pandas for Plotly / Streamlit charts.

Run:  streamlit run app_pyspark.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chisquare
from pyspark.sql import functions as F

from ola_spark_common import get_spark, load_data, dispatch_scores

st.set_page_config(page_title="Ola Mobility Insights", page_icon="🚕", layout="wide")

# ---------- OLA-STYLE THEME (CSS) ----------
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #F5F5F5; }
    .ola-header { background-color: #FFD100; padding: 22px 20px; border-radius: 10px;
        margin-bottom: 25px; display: flex; align-items: center; justify-content: center; }
    .ola-header h1 { color: #111111; font-family: 'Trebuchet MS', sans-serif; font-weight: 800;
        margin: 0; font-size: 32px; letter-spacing: 1px; }
    .ola-header p { color: #222222; margin: 4px 0 0 0; font-size: 14px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1E1E1E; border-radius: 8px 8px 0 0;
        padding: 10px 20px; color: #F5F5F5; }
    .stTabs [aria-selected="true"] { background-color: #FFD100 !important;
        color: #111111 !important; font-weight: 700; }
    label, .stMarkdown p { color: #F5F5F5 !important; }
    .result-card { background-color: #1E1E1E; border-left: 6px solid #FFD100;
        padding: 18px 20px; border-radius: 8px; margin: 15px 0; }
    .result-card h3 { color: #FFD100; margin: 0 0 6px 0; font-size: 16px; }
    .result-card p { color: #FFFFFF !important; font-size: 22px; font-weight: 700; margin: 0; }
    .result-card span { color: #BBBBBB; font-size: 13px; }
    .ola-footer { text-align: center; color: #777777; font-size: 12px; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ola-header"><div>
    <h1>OLA MOBILITY INSIGHTS</h1>
    <p style="text-align:center;">Demand, cancellations & driver dispatch analytics — Bengaluru (PySpark)</p>
</div></div>
""", unsafe_allow_html=True)


# ---------- SPARK SESSION + DATA (cached across reruns) ----------
@st.cache_resource
def get_data():
    spark = get_spark("OlaDashboard")
    spark.sparkContext.setLogLevel("WARN")
    return load_data(spark, "ola_bengaluru.csv").cache()      # Spark DF can't be pickled -> cache_resource


full_df = get_data()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header(" Filters")
vehicles = sorted(r[0] for r in full_df.select("Vehicle_Type").distinct().collect() if r[0])
sel_vehicles = st.sidebar.multiselect("Vehicle type", vehicles, default=vehicles)
dmin, dmax = full_df.agg(F.min("Date"), F.max("Date")).first()
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(date_range) != 2:
    date_range = (dmin, dmax)

df = full_df.filter(F.col("Vehicle_Type").isin(sel_vehicles)
                    & (F.col("Date") >= F.lit(date_range[0]))
                    & (F.col("Date") <= F.lit(date_range[1])))
n_view = df.count()
if n_view == 0:
    st.warning("No data for the selected filters.")
    st.stop()
st.sidebar.caption(f"{n_view:,} bookings in view")

counts = df.groupBy("Pickup_Location", "Hour").agg(F.count("*").alias("Bookings")).toPandas()

PLOT = dict(template="plotly_dark", paper_bgcolor="#121212", plot_bgcolor="#1E1E1E")
YELLOW = "#FFD100"

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "OVERVIEW", " AREA → PEAK TIME", " TIME → PEAK AREA",
    "  HEATMAP", " CANCELLATIONS", " DISPATCH PLANNER"])

# ============================ OVERVIEW ============================
with tab0:
    completed = df.filter(F.col("Success") == 1)
    k = df.agg(
        F.avg("Success").alias("succ"),
        F.avg("Cancelled").alias("canc"),
        F.sum(F.when(F.col("Success") == 1, F.col("Booking_Value"))).alias("rev"),
        F.avg(F.when(F.col("Success") == 1, F.col("Booking_Value"))).alias("fare"),
        F.avg(F.when(F.col("Success") == 1, F.col("Ride_Distance"))).alias("dist"),
        F.avg(F.when(F.col("Success") == 1, F.col("Driver_Ratings"))).alias("rating"),
        F.avg(F.when(F.col("Success") == 1, F.col("Avg_VTAT"))).alias("vtat"),
    ).first()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bookings", f"{n_view:,}")
    c2.metric("Completion rate", f"{k['succ']*100:.1f}%")
    c3.metric("Cancellation rate", f"{k['canc']*100:.1f}%")
    c4.metric("Total revenue", f"₹{(k['rev'] or 0)/1e6:.1f}M")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg fare", f"₹{(k['fare'] or 0):,.0f}")
    c6.metric("Avg distance", f"{(k['dist'] or 0):.1f} km")
    c7.metric("Avg driver rating", f"{(k['rating'] or 0):.2f}")
    c8.metric("Avg pickup ETA", f"{(k['vtat'] or 0):.1f} min")

    colA, colB = st.columns(2)
    with colA:
        daily = df.groupBy("Date").agg(F.count("*").alias("Bookings")).orderBy("Date").toPandas()
        fig = px.line(daily, x="Date", y="Bookings", title="Daily bookings", markers=True,
                      color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        rev = (completed.groupBy("Vehicle_Type").agg(F.sum("Booking_Value").alias("Booking Value"))
                        .toPandas())
        fig = px.bar(rev, x="Vehicle_Type", y="Booking Value", title="Revenue by vehicle type",
                     color_discrete_sequence=[YELLOW], labels={"Vehicle_Type": "Vehicle Type"})
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    colC, colD = st.columns(2)
    with colC:
        status = df.groupBy("Booking_Status").count().toPandas()
        fig = px.pie(status, names="Booking_Status", values="count", hole=0.5,
                     title="Booking outcomes",
                     color_discrete_sequence=["#FFD100", "#FF6B35", "#4ECDC4", "#888888"])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with colD:
        pay = completed.groupBy("Payment_Method").count().toPandas()
        fig = px.pie(pay, names="Payment_Method", values="count", title="Payment methods",
                     color_discrete_sequence=px.colors.sequential.YlOrBr_r)
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    # statistical honesty check
    hourly = df.groupBy("Hour").count().orderBy("Hour").toPandas().set_index("Hour")["count"]
    _, p = chisquare(hourly.values)
    spread = (hourly.max() - hourly.min()) / hourly.mean() * 100
    verdict = ("statistically significant" if p < 0.05
               else "NOT statistically significant (demand is close to uniform across hours)")
    st.info(f"**Statistical check:** busiest hour {hourly.idxmax()}:00, quietest "
            f"{hourly.idxmin()}:00 — a {spread:.1f}% spread. Chi-square test for uniform "
            f"demand gives p = {p:.3f}, so the hourly peak is {verdict}.")

# ============================ AREA -> PEAK HOUR ============================
with tab1:
    areas = sorted(counts["Pickup_Location"].unique())
    selected_area = st.selectbox("Select an area", areas)
    area_data = counts[counts["Pickup_Location"] == selected_area].sort_values("Hour")
    if not area_data.empty:
        peak_row = area_data.loc[area_data["Bookings"].idxmax()]
        st.markdown(f"""
        <div class="result-card">
            <h3>PEAK HOUR</h3>
            <p>{int(peak_row['Hour'])}:00</p>
            <span>{int(peak_row['Bookings'])} bookings recorded at this hour in {selected_area}</span>
        </div>""", unsafe_allow_html=True)
        st.bar_chart(area_data.set_index("Hour")["Bookings"], color="#FFD100")
    else:
        st.warning("No data found for this area.")

# ============================ HOUR -> PEAK AREA ============================
with tab2:
    hours = sorted(counts["Hour"].unique())
    selected_hour = st.selectbox("Select an hour (24h format)", hours)
    hour_data = counts[counts["Hour"] == selected_hour]
    if not hour_data.empty:
        peak_row = hour_data.loc[hour_data["Bookings"].idxmax()]
        st.markdown(f"""
        <div class="result-card">
            <h3>BUSIEST AREA</h3>
            <p>{peak_row['Pickup_Location']}</p>
            <span>{int(peak_row['Bookings'])} bookings recorded at {selected_hour}:00</span>
        </div>""", unsafe_allow_html=True)
        top10 = hour_data.sort_values("Bookings", ascending=False).head(10)
        st.bar_chart(top10.set_index("Pickup_Location")["Bookings"], color="#FFD100")
    else:
        st.warning("No data found for this hour.")

# ============================ HEATMAP ============================
with tab3:
    st.subheader("Demand heatmap: area × hour")
    top_n = st.slider("Number of busiest areas to show", 5, 50, 20)
    pivot = (df.groupBy("Pickup_Location").pivot("Hour", list(range(24))).count()
               .fillna(0).toPandas().set_index("Pickup_Location"))
    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:top_n]]
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Hour", y="Area", color="Bookings"))
    fig.update_layout(**PLOT, height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday vs weekend pattern")
    wk = df.withColumn("Type", F.when(F.col("IsWeekend") == 1, "Weekend").otherwise("Weekday"))
    days_per_type = wk.groupBy("Type").agg(F.countDistinct("Date").alias("Days"))
    hr = (wk.groupBy("Type", "Hour").agg(F.count("*").alias("Bookings"))
            .join(days_per_type, on="Type")
            .withColumn("Bookings per day", F.col("Bookings") / F.col("Days"))
            .orderBy("Type", "Hour").toPandas())
    fig = px.line(hr, x="Hour", y="Bookings per day", color="Type", markers=True,
                  color_discrete_sequence=[YELLOW, "#4ECDC4"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

# ============================ CANCELLATIONS ============================
with tab4:
    st.subheader("Cancellation analysis")
    col1, col2 = st.columns(2)
    with col1:
        r = (df.filter(F.col("Booking_Status") == "Cancelled by Driver")
               .groupBy("Reason_for_Cancelling_by_Driver").count()
               .orderBy(F.col("count").desc()).toPandas())
        fig = px.bar(r, y="Reason_for_Cancelling_by_Driver", x="count", orientation="h",
                     title="Why drivers cancel", color_discrete_sequence=[YELLOW],
                     labels={"Reason_for_Cancelling_by_Driver": "Reason for Cancelling by Driver"})
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        r = (df.filter(F.col("Booking_Status") == "Cancelled by Customer")
               .groupBy("Reason_for_Cancelling_by_Customer").count()
               .orderBy(F.col("count").desc()).toPandas())
        fig = px.bar(r, y="Reason_for_Cancelling_by_Customer", x="count", orientation="h",
                     title="Why customers cancel", color_discrete_sequence=["#FF6B35"],
                     labels={"Reason_for_Cancelling_by_Customer": "Reason for Cancelling by Customer"})
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)

    by_area = (df.groupBy("Pickup_Location")
                 .agg(F.count("Booking_ID").alias("Bookings"),
                      F.round(F.avg("Cancelled") * 100, 1).alias("Cancel rate %"))
                 .orderBy(F.col("Cancel rate %").desc()).toPandas())
    colX, colY = st.columns(2)
    with colX:
        st.markdown("** Highest cancellation-rate areas**")
        st.dataframe(by_area.head(8), hide_index=True, use_container_width=True)
    with colY:
        st.markdown("** Most reliable areas**")
        st.dataframe(by_area.tail(8).iloc[::-1], hide_index=True, use_container_width=True)

    hc = (df.groupBy("Hour").agg((F.avg("Cancelled") * 100).alias("Cancel rate %"))
            .orderBy("Hour").toPandas())
    fig = px.line(hc, x="Hour", y="Cancel rate %", markers=True,
                  title="Cancellation rate by hour", color_discrete_sequence=["#FF6B35"])
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, use_container_width=True)

# ============================ DISPATCH PLANNER ============================
with tab5:
    st.subheader(" Driver dispatch planner")
    st.caption("Where should idle drivers go at a given hour? Areas are scored on demand, "
               "fare value and reliability (low cancellation rate).")
    plan_hour = st.select_slider("Hour of day", options=list(range(24)), value=18)

    top5 = dispatch_scores(df, plan_hour).limit(5).toPandas()
    if top5.empty:
        st.warning("No bookings at this hour for the selected filters.")
    else:
        best = top5.iloc[0]
        st.markdown(f"""
        <div class="result-card">
            <h3>RECOMMENDED STAGING AREA AT {plan_hour}:00</h3>
            <p>{best['Pickup_Location']}</p>
            <span>Score {best['Score']:.0f}/100 · {int(best['Demand'])} bookings ·
            avg fare ₹{best['Avg_Fare']:,.0f} · {best['Cancel_Rate']*100:.1f}% cancellations</span>
        </div>""", unsafe_allow_html=True)

        fig = px.bar(top5, x="Pickup_Location", y="Score", title="Top 5 areas to position drivers",
                     hover_data=["Demand", "Avg_Fare", "Cancel_Rate"], color_discrete_sequence=[YELLOW])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Score = 50% demand + 30% average fare + 20% reliability, each scaled 0–1 "
                   "across areas for the selected hour.")

st.markdown('<div class="ola-footer">Data source: Bengaluru Ola Cabs dataset · '
            'Urban Mobility Optimization Project · Powered by PySpark</div>', unsafe_allow_html=True)
