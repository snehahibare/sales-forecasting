"""
Precomputes all data the Streamlit dashboard needs and saves it to /data as CSV/JSON.
Run this once (or whenever train.csv changes) - app.py just reads these files.
"""
import pandas as pd
import numpy as np
import json
import os
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

os.makedirs('data', exist_ok=True)

# ---------- Load & feature engineer ----------
df = pd.read_csv('train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)

monthly_ts = df.set_index('Order Date')['Sales'].resample('MS').sum()
monthly_ts.index.freq = 'MS'
weekly_ts = df.set_index('Order Date')['Sales'].resample('W')['Sales'].sum() if False else df.set_index('Order Date')['Sales'].resample('W').sum()

# ---------- Overview page data ----------
yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()
yearly_sales.to_csv('data/yearly_sales.csv', index=False)

monthly_sales_df = monthly_ts.reset_index()
monthly_sales_df.columns = ['Month', 'Sales']
monthly_sales_df.to_csv('data/monthly_sales.csv', index=False)

region_category = df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
region_category.to_csv('data/region_category_sales.csv', index=False)

# ---------- Forecast Explorer: XGBoost per segment ----------
def build_segment_monthly_series(mask):
    seg_df = df[mask]
    seg_monthly = seg_df.set_index('Order Date')['Sales'].resample('MS').sum()
    seg_monthly = seg_monthly.reindex(monthly_ts.index, fill_value=0)
    return seg_monthly

def make_features(seg_series):
    d = seg_series.reset_index()
    d.columns = ['Month', 'Sales']
    d['lag1'] = d['Sales'].shift(1)
    d['lag2'] = d['Sales'].shift(2)
    d['lag3'] = d['Sales'].shift(3)
    d['rolling_mean_3'] = d['Sales'].shift(1).rolling(window=3).mean()
    d['month_num'] = d['Month'].dt.month
    d['quarter'] = d['Month'].dt.quarter
    d['season'] = d['Month'].dt.month.apply(get_season)
    dummies = pd.get_dummies(d['season'], prefix='season')
    d = pd.concat([d, dummies], axis=1)
    feature_cols = ['lag1', 'lag2', 'lag3', 'rolling_mean_3', 'month_num', 'quarter'] + list(dummies.columns)
    return d, feature_cols, dummies.columns

def eval_metrics(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, rmse

def fit_and_forecast(seg_series, n_future=3, test_months=3):
    d, feature_cols, season_cols = make_features(seg_series)
    model_df = d.dropna().reset_index(drop=True)

    # Holdout evaluation (last `test_months`)
    train_part = model_df.iloc[:-test_months]
    test_part = model_df.iloc[-test_months:]
    X_train, y_train = train_part[feature_cols], train_part['Sales']
    X_test, y_test = test_part[feature_cols], test_part['Sales']

    eval_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    eval_model.fit(X_train, y_train)
    test_preds = eval_model.predict(X_test)
    mae, rmse = eval_metrics(y_test, test_preds)

    # Full-data model for actual future forecast
    X_full, y_full = model_df[feature_cols], model_df['Sales']
    full_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    full_model.fit(X_full, y_full)

    history = d[['Month', 'Sales']].copy()
    future_dates = pd.date_range(history['Month'].max() + pd.DateOffset(months=1), periods=n_future, freq='MS')
    preds = []
    for step_date in future_dates:
        lag1 = history['Sales'].iloc[-1]
        lag2 = history['Sales'].iloc[-2]
        lag3 = history['Sales'].iloc[-3]
        roll_mean3 = history['Sales'].iloc[-3:].mean()
        row = {'lag1': lag1, 'lag2': lag2, 'lag3': lag3, 'rolling_mean_3': roll_mean3,
               'month_num': step_date.month, 'quarter': step_date.quarter}
        season = get_season(step_date.month)
        for col in season_cols:
            row[col] = 1 if col == f'season_{season}' else 0
        X_step = pd.DataFrame([row])[feature_cols]
        pred = full_model.predict(X_step)[0]
        preds.append(pred)
        history = pd.concat([history, pd.DataFrame({'Month': [step_date], 'Sales': [pred]})], ignore_index=True)

    forecast_series = pd.Series(preds, index=future_dates)
    return forecast_series, mae, rmse

segments = {
    'Overall': pd.Series(True, index=df.index),
    'Furniture': df['Category'] == 'Furniture',
    'Technology': df['Category'] == 'Technology',
    'Office Supplies': df['Category'] == 'Office Supplies',
    'West': df['Region'] == 'West',
    'East': df['Region'] == 'East',
    'Central': df['Region'] == 'Central',
    'South': df['Region'] == 'South',
}

forecast_records = []
history_records = []
metrics = {}

for name, mask in segments.items():
    seg_series = build_segment_monthly_series(mask)
    forecast_series, mae, rmse = fit_and_forecast(seg_series, n_future=3, test_months=3)
    metrics[name] = {'MAE': round(float(mae), 1), 'RMSE': round(float(rmse), 1)}
    for date, val in seg_series.items():
        history_records.append({'Segment': name, 'Month': date, 'Sales': val, 'Type': 'Actual'})
    for date, val in forecast_series.items():
        history_records.append({'Segment': name, 'Month': date, 'Sales': val, 'Type': 'Forecast'})

history_df = pd.DataFrame(history_records)
history_df.to_csv('data/segment_history_forecast.csv', index=False)

with open('data/segment_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Forecast data done for segments:", list(segments.keys()))

# ---------- Anomaly Report ----------
anomaly_df = weekly_ts.reset_index()
anomaly_df.columns = ['Week', 'Sales']
anomaly_df['rolling_mean_4'] = anomaly_df['Sales'].rolling(window=4, min_periods=1, center=True).mean()
anomaly_df['rolling_std_4'] = anomaly_df['Sales'].rolling(window=4, min_periods=1, center=True).std().fillna(0)
anomaly_df['pct_change'] = anomaly_df['Sales'].pct_change().fillna(0)

iso_features = anomaly_df[['Sales', 'rolling_mean_4', 'rolling_std_4', 'pct_change']]
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_df['iso_anomaly'] = iso_forest.fit_predict(iso_features) == -1

ROLL_WINDOW = 8
anomaly_df['rolling_mean_z'] = anomaly_df['Sales'].rolling(window=ROLL_WINDOW, min_periods=3).mean()
anomaly_df['rolling_std_z'] = anomaly_df['Sales'].rolling(window=ROLL_WINDOW, min_periods=3).std()
anomaly_df['z_score'] = (anomaly_df['Sales'] - anomaly_df['rolling_mean_z']) / anomaly_df['rolling_std_z']
anomaly_df['zscore_anomaly'] = anomaly_df['z_score'].abs() > 2
anomaly_df['any_anomaly'] = anomaly_df['iso_anomaly'] | anomaly_df['zscore_anomaly']

anomaly_df.to_csv('data/weekly_anomalies.csv', index=False)
print(f"Anomaly data done: {anomaly_df['iso_anomaly'].sum()} IsoForest, {anomaly_df['zscore_anomaly'].sum()} Z-score")

# ---------- Product Demand Segments (clustering) ----------
subcat_yearly = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack('Year')
subcat_growth = subcat_yearly.pct_change(axis=1).mean(axis=1) * 100
subcat_monthly = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='MS')])['Sales'].sum().unstack('Sub-Category')
subcat_volatility = subcat_monthly.std()

subcat_features = pd.DataFrame({
    'Total_Sales_Volume': df.groupby('Sub-Category')['Sales'].sum(),
    'YoY_Growth_Rate_%': subcat_growth,
    'Sales_Volatility': subcat_volatility,
    'Avg_Order_Value': df.groupby('Sub-Category')['Sales'].mean(),
}).dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(subcat_features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
subcat_features['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = subcat_features.groupby('Cluster')[['Total_Sales_Volume', 'YoY_Growth_Rate_%', 'Sales_Volatility', 'Avg_Order_Value']].mean()
cluster_summary['CV'] = cluster_summary['Sales_Volatility'] / cluster_summary['Total_Sales_Volume']

growth_rank = cluster_summary['YoY_Growth_Rate_%'].rank(ascending=False)
volume_rank = cluster_summary['Total_Sales_Volume'].rank(ascending=False)
cv_rank = cluster_summary['CV'].rank(ascending=True)

def label_cluster(cluster_id):
    g, v, c = growth_rank[cluster_id], volume_rank[cluster_id], cv_rank[cluster_id]
    if g == 1:
        return 'Growing Demand'
    if v == 1 and c <= 2:
        return 'High Volume, Stable Demand'
    if v == cluster_summary.shape[0] or c == cluster_summary.shape[0]:
        return 'Low Volume, High Volatility'
    if c > cluster_summary.shape[0] / 2:
        return 'Moderate Volume, Volatile Demand'
    return 'Moderate Volume, Stable Demand'

cluster_labels = {cid: label_cluster(cid) for cid in cluster_summary.index}
subcat_features['Cluster_Label'] = subcat_features['Cluster'].map(cluster_labels)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
subcat_features['PCA1'] = coords[:, 0]
subcat_features['PCA2'] = coords[:, 1]

subcat_features.reset_index().to_csv('data/product_clusters.csv', index=False)
print("Clustering data done. Clusters:")
for cid, label in cluster_labels.items():
    print(f"  {cid}: {label}")

print("\nAll dashboard data precomputed successfully.")
