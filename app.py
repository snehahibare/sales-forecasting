import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached so the dashboard stays fast)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    yearly_sales = pd.read_csv("data/yearly_sales.csv")
    monthly_sales = pd.read_csv("data/monthly_sales.csv", parse_dates=["Month"])
    region_category = pd.read_csv("data/region_category_sales.csv")
    segment_history = pd.read_csv("data/segment_history_forecast.csv", parse_dates=["Month"])
    with open("data/segment_metrics.json") as f:
        segment_metrics = json.load(f)
    weekly_anomalies = pd.read_csv("data/weekly_anomalies.csv", parse_dates=["Week"])
    product_clusters = pd.read_csv("data/product_clusters.csv")
    return (yearly_sales, monthly_sales, region_category, segment_history,
            segment_metrics, weekly_anomalies, product_clusters)

(yearly_sales, monthly_sales, region_category, segment_history,
 segment_metrics, weekly_anomalies, product_clusters) = load_data()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Sales Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
)
st.sidebar.markdown("---")
st.sidebar.caption("End-to-End Sales Forecasting & Demand Intelligence System — Superstore Sales Dataset")

# ===========================================================================
# PAGE 1 — Sales Overview
# ===========================================================================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Sales by Year")
        fig = px.bar(yearly_sales, x="Year", y="Sales", text_auto=".2s",
                     color_discrete_sequence=["#4C72B0"])
        fig.update_layout(yaxis_title="Sales ($)")
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Monthly Sales Trend")
        fig = px.line(monthly_sales, x="Month", y="Sales", markers=True,
                      color_discrete_sequence=["#DD8452"])
        fig.update_layout(yaxis_title="Sales ($)")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Sales by Region & Category")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_regions = st.multiselect(
            "Filter by Region", options=sorted(region_category["Region"].unique()),
            default=sorted(region_category["Region"].unique())
        )
    with filter_col2:
        selected_categories = st.multiselect(
            "Filter by Category", options=sorted(region_category["Category"].unique()),
            default=sorted(region_category["Category"].unique())
        )

    filtered = region_category[
        region_category["Region"].isin(selected_regions) &
        region_category["Category"].isin(selected_categories)
    ]
    fig = px.bar(filtered, x="Region", y="Sales", color="Category", barmode="group")
    fig.update_layout(yaxis_title="Sales ($)")
    st.plotly_chart(fig, width='stretch')

# ===========================================================================
# PAGE 2 — Forecast Explorer
# ===========================================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts generated with XGBoost — the best-performing model from Task 3's comparison.")

    segment_options = sorted(segment_history["Segment"].unique())
    default_idx = segment_options.index("Overall") if "Overall" in segment_options else 0
    selected_segment = st.selectbox("Select Category or Region", segment_options, index=default_idx)

    horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    seg_data = segment_history[segment_history["Segment"] == selected_segment].sort_values("Month")
    actual = seg_data[seg_data["Type"] == "Actual"]
    forecast = seg_data[seg_data["Type"] == "Forecast"].iloc[:horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual["Month"], y=actual["Sales"], mode="lines",
                              name="Actual", line=dict(color="#4C72B0")))
    fig.add_trace(go.Scatter(x=forecast["Month"], y=forecast["Sales"], mode="lines+markers",
                              name="Forecast", line=dict(color="#DD8452", dash="dash")))
    fig.update_layout(title=f"{selected_segment} — Sales Forecast ({horizon} month{'s' if horizon > 1 else ''})",
                       yaxis_title="Sales ($)", xaxis_title="Month")
    st.plotly_chart(fig, width='stretch')

    metrics = segment_metrics.get(selected_segment, {})
    m1, m2 = st.columns(2)
    m1.metric("MAE (holdout test)", f"${metrics.get('MAE', 0):,.1f}")
    m2.metric("RMSE (holdout test)", f"${metrics.get('RMSE', 0):,.1f}")
    st.caption("MAE/RMSE computed on a 3-month holdout for this specific segment.")

# ===========================================================================
# PAGE 3 — Anomaly Report
# ===========================================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Weekly sales, flagged by Isolation Forest and Z-Score (>2 std dev from rolling mean).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_anomalies["Week"], y=weekly_anomalies["Sales"],
                              mode="lines", name="Weekly Sales", line=dict(color="#4C72B0", width=1)))

    iso_pts = weekly_anomalies[weekly_anomalies["iso_anomaly"]]
    fig.add_trace(go.Scatter(x=iso_pts["Week"], y=iso_pts["Sales"], mode="markers",
                              name="Isolation Forest Anomaly",
                              marker=dict(color="#C44E52", size=10)))

    z_pts = weekly_anomalies[weekly_anomalies["zscore_anomaly"]]
    fig.add_trace(go.Scatter(x=z_pts["Week"], y=z_pts["Sales"], mode="markers",
                              name="Z-Score Anomaly",
                              marker=dict(color="#DD8452", size=10, symbol="diamond")))

    fig.update_layout(title="Weekly Sales — Detected Anomalies", yaxis_title="Sales ($)")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Detected Anomaly Weeks")
    anomaly_table = weekly_anomalies[weekly_anomalies["any_anomaly"]][
        ["Week", "Sales", "iso_anomaly", "zscore_anomaly"]
    ].sort_values("Sales", ascending=False).reset_index(drop=True)
    anomaly_table.columns = ["Week", "Sales ($)", "Flagged by Isolation Forest", "Flagged by Z-Score"]
    st.dataframe(anomaly_table, width='stretch')

# ===========================================================================
# PAGE 4 — Product Demand Segments
# ===========================================================================
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("K-Means clustering (K=4) on sub-category demand features, visualized via PCA.")

    fig = px.scatter(
        product_clusters, x="PCA1", y="PCA2", color="Cluster_Label",
        text="Sub-Category", size="Total_Sales_Volume", size_max=40,
        hover_data=["Total_Sales_Volume", "YoY_Growth_Rate_%", "Sales_Volatility", "Avg_Order_Value"]
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(title="Product Sub-Category Clusters (PCA-reduced)")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Sub-Categories by Demand Cluster")
    display_cols = ["Sub-Category", "Cluster_Label", "Total_Sales_Volume", "YoY_Growth_Rate_%",
                     "Sales_Volatility", "Avg_Order_Value"]
    st.dataframe(
        product_clusters[display_cols].sort_values("Cluster_Label").reset_index(drop=True),
        width='stretch'
    )

    st.subheader("Recommended Stocking Strategy")
    strategy_map = {
        "High Volume, Stable Demand": "Keep consistently high stock with standard reorder points — stockouts here are the costliest mistake.",
        "Moderate Volume, Stable Demand": "Maintain steady inventory levels with periodic reorders; low risk of sudden demand swings.",
        "Moderate Volume, Volatile Demand": "Stock conservatively and monitor closely — order value is high but demand swings unpredictably.",
        "Low Volume, High Volatility": "Reorder in small, frequent batches; avoid tying up capital in unpredictable, slow-moving items.",
        "Growing Demand": "Increase stock ahead of demand and monitor closely — under-stocking risks losing sales as demand builds.",
    }
    for label in product_clusters["Cluster_Label"].unique():
        st.markdown(f"**{label}:** {strategy_map.get(label, 'Monitor and adjust as more data becomes available.')}")
