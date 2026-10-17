import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lifelines import KaplanMeierFitter
import shap

df = pd.read_csv('transactions_dataset.csv')

df=df.dropna()
df["OrderDate"]=pd.to_datetime(df["OrderDate"], format="mixed")

invalid= df[
 (df["OrderDate"] > "2025-12-31") |
 (df["OrderDate"] < "2022-01-01")

]

df = df[
    (df["OrderDate"] >= "2022-01-01") &
    (df["OrderDate"] <= "2025-12-31")
].copy()

print(df["OrderID"].nunique())
print(df["CustomerID"].nunique())
print(df["ProductID"].nunique())
print(df["Quantity"].min())
print(df["Quantity"].max())
print(df["UnitPrice"].min())
print(df["UnitPrice"].max())

negq= df[df["Quantity"]<0]
negp= df[df["UnitPrice"]<0]

negq.head(20)
print(negq["Product Name"].value_counts().head(20))
print(negq["UnitPrice"].describe())

negp.head(20)
print(negp["Product Name"].value_counts().head(20))

df["UnitPrice"]=df["UnitPrice"].abs()

zeroq=df[df["Quantity"]==0]
zerop=df[df["UnitPrice"]==0]

df.drop(index=zeroq.index, inplace=True)
df.drop(index=zerop.index, inplace=True)

print((df["Quantity"]==0).sum())
print((df["UnitPrice"]==0).sum())

print(df["UnitPrice"].quantile(0.90))
print(df["UnitPrice"].quantile(0.95))
print(df["UnitPrice"].quantile(0.99))
print(df["UnitPrice"].quantile(0.995))
print(df["UnitPrice"].quantile(0.999))

extreme=df[df["UnitPrice"]>1292.04]
df.drop(index=extreme.index, inplace=True)

df["CustomerID"]=df["CustomerID"].str.strip().str.title()
df["OrderID"]=df["OrderID"].str.strip().str.title()
df["ProductID"]=df["ProductID"].str.strip().str.title()
df["Product Name"]=df["Product Name"].str.strip().str.title()
df["ProductCategory"]=df["ProductCategory"].str.strip().str.title()
df["Country"]=df["Country"].str.strip().str.title()

country_mapping = {
    "Uk": "United Kingdom",
    "U.K.": "United Kingdom",
    "Us": "United States",
    "U.S.A.": "United States",
    "Usa": "United States",
    "Uae": "United Arab Emirates",
    "U.A.E.": "United Arab Emirates"
}

df["Country"]=df["Country"].replace(country_mapping)
print(df["Country"].value_counts())

df["Revenue"]= df["Quantity"]*df["UnitPrice"]

print("Negative Revenue:", (df["Revenue"] < 0).sum())
print("Zero Revenue:", (df["Revenue"] == 0).sum())

customer_orders= df.groupby("CustomerID")["OrderID"].nunique()
print(customer_orders.value_counts().sort_index().head(20))

one_order=(customer_orders==1).sum()
repeat_order=(customer_orders>1).sum()
total_customers = customer_orders.shape[0]


print("Total Customer: ", total_customers)
print("One Order Customers: ", one_order)
print("Repeat Customers: ", repeat_order)

per_repeat_order = round(repeat_order/total_customers*100,2)
print("Percentage of Repeat Customers: ", per_repeat_order, "%")

customer_dates = (
    df.groupby("CustomerID")["OrderDate"]
      .apply(lambda x: x.sort_values().unique())
)

purchase_intervals = []

for dates in customer_dates:
    if len(dates) > 1:
        intervals = pd.Series(dates).diff().dt.days.dropna()
        purchase_intervals.extend(intervals)

purchase_intervals = pd.Series(purchase_intervals)

plt.figure(figsize=(10, 6))

plt.hist(purchase_intervals, bins=50)

plt.xlabel("Days Between Purchases")
plt.ylabel("Number of Purchase Intervals")
plt.title("Distribution of Customer Purchase Intervals")

orders= (
    df.groupby(["CustomerID", "OrderID"]).agg(
    OrderDate=("OrderDate", "min"),
    OrderRevenue=("Revenue", "sum")
).reset_index()
)

customer_orders = (
    orders.sort_values(["CustomerID", "OrderDate"]).groupby("CustomerID")["OrderDate"]
)
print(orders.head())

calibration_orders= orders[orders["OrderDate"]<="2024-12-31"].copy()
holdout_orders= orders[orders["OrderDate"]>="2025-01-01"].copy()

frequency = calibration_orders.groupby("CustomerID")["OrderID"].nunique()-1

first_purchase = calibration_orders.groupby("CustomerID")["OrderDate"].min()
last_purchase = calibration_orders.groupby("CustomerID")["OrderDate"].max()
recency = (last_purchase - first_purchase).dt.days

calibration_end = calibration_orders["OrderDate"].max()
tenure = (calibration_end-first_purchase).dt.days

cust_summary = pd.concat(
    [frequency, recency, tenure], axis=1
).reset_index()
cust_summary.columns = [
    "CustomerID",
    "Frequency",
    "Recency",
    "Tenure"
]

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(
    cust_summary["Frequency"],
    cust_summary["Recency"],
    cust_summary["Tenure"]
)

cust_summary["P_active"] = bgf.conditional_probability_alive(
    cust_summary["Frequency"],
    cust_summary["Recency"],
    cust_summary["Tenure"]
    )

holdout_ends = holdout_orders["OrderDate"].max()
holdout_start = holdout_orders["OrderDate"].min()
holdout_days = (holdout_ends-holdout_start).days

cust_summary["2025_Predictions"] = bgf.predict(
    holdout_days,
    cust_summary["Frequency"],
    cust_summary["Recency"],
    cust_summary["Tenure"]

)

actual_2025 = holdout_orders.groupby("CustomerID")["OrderID"].nunique().rename("Actual_2025")
cust_summary = cust_summary.merge(
    actual_2025,
    on="CustomerID",
    how="left"
)

cust_summary["2025_Predictions"]= cust_summary["2025_Predictions"].fillna(0).astype(int)
cust_summary["Actual_2025"]= cust_summary["Actual_2025"].fillna(0).astype(int)

bgf_validation = cust_summary.dropna(subset=["2025_Predictions"]).copy()

mae = mean_absolute_error(
    bgf_validation["Actual_2025"],
    bgf_validation["2025_Predictions"]
)

rmse = mean_squared_error(
    bgf_validation["Actual_2025"],
    bgf_validation["2025_Predictions"]
)

positive_orders = calibration_orders[calibration_orders["OrderRevenue"]>0].copy()
customer_monetary = (
    positive_orders
    .groupby("CustomerID")
    .agg(
        Frequency=("OrderID", "nunique"),
        Actual_Monetary_Value=("OrderRevenue", "mean")
    )
    .reset_index()
)

gamma_gamma_data = customer_monetary.copy()

gamma_gamma_data["Frequency"] = (
    gamma_gamma_data["Frequency"] - 1
)

gamma_gamma_data = gamma_gamma_data[
    gamma_gamma_data["Frequency"] > 0
].copy()

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(
    gamma_gamma_data["Frequency"],
    gamma_gamma_data["Actual_Monetary_Value"]
)

gamma_gamma_data["Expected_Monetary_Value"] = ggf.conditional_expected_average_profit(
    gamma_gamma_data["Frequency"],
    gamma_gamma_data["Actual_Monetary_Value"]
)

cust_summary = cust_summary.merge(
    gamma_gamma_data[
        ["CustomerID", "Expected_Monetary_Value", "Actual_Monetary_Value"]
    ],
    on="CustomerID",
    how="left"
)

actual_2025_revenue = (
    holdout_orders
    .groupby("CustomerID")["OrderRevenue"]
    .sum()
    .rename("Actual_2025_Revenue")
)


cust_summary = cust_summary.merge(
    actual_2025_revenue,
    on="CustomerID",
    how="left"
)


cust_summary["Actual_2025_Revenue"] = (cust_summary["Actual_2025_Revenue"].fillna(0))

cust_summary["Predicted_2025_Revenue"] = (
    cust_summary["2025_Predictions"]
    * cust_summary["Expected_Monetary_Value"]
)

print(mae)
print(rmse)
buyers = cust_summary[cust_summary["Actual_2025"] > 0].copy()


customer_features = calibration_orders.groupby("CustomerID").agg(
    TotalOrders=("OrderID", "nunique"),
    FirstPurchase=("OrderDate", "min"),
    LastPurchase=("OrderDate", "max"),
    TotalRevenue=("OrderRevenue","sum"),
    AverageOrderValue=("OrderRevenue","mean"),
    RevenueStd=("OrderRevenue", "std")
).reset_index()

customer_features["Recency"] = (
    customer_features["LastPurchase"] -
    customer_features["FirstPurchase"]
).dt.days

customer_features["Tenure"] = (
    calibration_end -
    customer_features["FirstPurchase"]
).dt.days

customer_features["Frequency"] = (
    customer_features["TotalOrders"]
)

sorted_orders = calibration_orders.sort_values(["CustomerID", "OrderDate"]).copy()
sorted_orders["PurchaseInterval"] = (
    sorted_orders.groupby("CustomerID")["OrderDate"]
                .diff()
                .dt.days
)

interval_features = sorted_orders.groupby("CustomerID").agg(
    AverageInterval = ("PurchaseInterval", "mean"),
    IntervalStd = ("PurchaseInterval","std")
).reset_index()
customer_features = customer_features.merge(
    interval_features,
    on="CustomerID",
    how="left"
)

calibration_transactions = df[
    df["OrderDate"] <= "2024-12-31"
].copy()

product_diversity = calibration_transactions.groupby("CustomerID").agg(
    UniqueProducts = ("ProductID", "nunique"),
    UniqueCategory = ("ProductCategory", "nunique")
).reset_index()
customer_features = customer_features.merge(
    product_diversity,
    on="CustomerID",
    how="left"
)

return_orders = calibration_orders[calibration_orders["OrderRevenue"] < 0].copy()
return_features = return_orders.groupby("CustomerID").agg(
    ReturnCount = ("OrderID", "nunique"),
    TotalReturnValue = ("OrderRevenue" , lambda x: x.abs().sum())
).reset_index()

customer_features = customer_features.merge(
    return_features,
    on = "CustomerID",
    how = "left"
)

customer_features["ReturnCount"] = customer_features["ReturnCount"].fillna(0).astype(int)
customer_features["TotalReturnValue"] = customer_features["TotalReturnValue"].fillna(0)
customer_features["ReturnRate"] = customer_features["ReturnCount"]/customer_features["TotalOrders"]

start_90 = calibration_end - pd.Timedelta(days=90)
start_180 = calibration_end - pd.Timedelta(days=180)

orders_90 = calibration_orders[
    calibration_orders["OrderDate"] > start_90
].copy()

orders_180 = calibration_orders[
    calibration_orders["OrderDate"] > start_180
].copy()

recent_90 = (
    orders_90
    .groupby("CustomerID")
    .agg(
        RecentOrders90=("OrderID", "nunique"),
        RecentRevenue90=("OrderRevenue", "sum")
    )
    .reset_index()
)

recent_180 = (
    orders_180
    .groupby("CustomerID")
    .agg(
        RecentOrders180=("OrderID", "nunique"),
        RecentRevenue180=("OrderRevenue", "sum")
    )
    .reset_index()
)

customer_features = customer_features.merge(
    recent_90,
    on="CustomerID",
    how="left"
)

customer_features = customer_features.merge(
    recent_180,
    on="CustomerID",
    how="left"
)

recent_columns = [
    "RecentOrders90",
    "RecentRevenue90",
    "RecentOrders180",
    "RecentRevenue180"
]

customer_features[recent_columns] = (
    customer_features[recent_columns].fillna(0)
)

customer_features = customer_features.merge(
    actual_2025_revenue,
    on="CustomerID",
    how="left"
)

customer_features["Actual_2025_Revenue"] = (
    customer_features["Actual_2025_Revenue"]
    .fillna(0)
)

probabilistic_features = cust_summary[
    [
        "CustomerID",
        "2025_Predictions",
        "P_active",
        "Expected_Monetary_Value",
        "Predicted_2025_Revenue"
    ]
].copy()

customer_features = customer_features.merge(
    probabilistic_features,
    on="CustomerID",
    how="left"
)

behavioural_features = [
    "TotalOrders",
    "TotalRevenue",
    "AverageOrderValue",
    "RevenueStd",
    "Recency",
    "Tenure",
    "Frequency",
    "AverageInterval",
    "IntervalStd",
    "UniqueProducts",
    "UniqueCategory",
    "ReturnCount",
    "TotalReturnValue",
    "ReturnRate",
    "RecentOrders90",
    "RecentRevenue90",
    "RecentOrders180",
    "RecentRevenue180"
]


probabilistic_features = [
    "2025_Predictions",
    "P_active",
    "Expected_Monetary_Value",
    "Predicted_2025_Revenue"
]

target = "Actual_2025_Revenue"

missing_features = [
    "RevenueStd",
    "AverageInterval",
    "IntervalStd",
    "2025_Predictions",
    "Expected_Monetary_Value",
    "Predicted_2025_Revenue"
]

for feature in missing_features:
    customer_features[f"{feature}_Missing"] = (
        customer_features[feature].isna().astype(int)
    )

customer_features[
    ["RevenueStd", "AverageInterval", "IntervalStd"]
] = customer_features[
    ["RevenueStd", "AverageInterval", "IntervalStd"]
].fillna(0)

probabilistic_imputation = [
    "2025_Predictions",
    "Expected_Monetary_Value",
    "Predicted_2025_Revenue"
]

for feature in probabilistic_imputation:
    median_value = customer_features[feature].median()
    customer_features[feature] = (
        customer_features[feature].fillna(median_value)
    )

bf_with_missing = behavioural_features + [
    "RevenueStd_Missing",
    "AverageInterval_Missing",
    "IntervalStd_Missing"
]

pf_with_missing = probabilistic_features + [
    "2025_Predictions_Missing",
    "Expected_Monetary_Value_Missing",
    "Predicted_2025_Revenue_Missing"
]

X_behavioural = customer_features[bf_with_missing].copy()
X_hybrid = customer_features[bf_with_missing + probabilistic_features].copy()
y = customer_features[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_behavioural,
    y,
    test_size=0.20,
    random_state=42
)

behavioural_xgb = XGBRegressor(
    objective = "reg:squarederror",
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
    n_jobs = -1
)

behavioural_xgb.fit(X_train, y_train)
behavioural_predictions = behavioural_xgb.predict(X_test)

behavioural_mae = mean_absolute_error(y_test, behavioural_predictions)
behavioural_rmse = mean_squared_error(y_test, behavioural_predictions)**0.5
behavioural_r2 = r2_score(y_test, behavioural_predictions)
actual_total = y_test.sum()
predicted_total = behavioural_predictions.sum()

print("Behavioural XGBoost Results")
print("---------------------------")
print(f"MAE: {behavioural_mae:.2f}")
print(f"RMSE: {behavioural_rmse:.2f}")
print(f"R²: {behavioural_r2:.4f}")
print()
print(f"Actual total revenue:    {actual_total:,.2f}")
print(f"Predicted total revenue: {predicted_total:,.2f}")

X_hybrid_train, X_hybrid_test, y_hybrid_train, y_hybrid_test = train_test_split(
    X_hybrid,
    y,
    test_size=0.20,
    random_state=42
)

hybrid_xgb = XGBRegressor(
    objective = "reg:squarederror",
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
    n_jobs = -1
)

hybrid_xgb.fit(X_hybrid_train, y_hybrid_train)
hybrid_predictions = hybrid_xgb.predict(X_hybrid_test)

hybrid_mae = mean_absolute_error(y_hybrid_test, hybrid_predictions)
hybrid_rmse = mean_squared_error(y_hybrid_test, hybrid_predictions)**0.5
hybrid_r2 = r2_score(y_hybrid_test, hybrid_predictions)
actual_total = y_hybrid_test.sum()
predicted_total = hybrid_predictions.sum()

print("Hybrid XGBoost Results")
print("--------------------------")
print(f"MAE: {hybrid_mae:.2f}")
print(f"RMSE: {hybrid_rmse:.2f}")
print(f"R²: {hybrid_r2:.4f}")
print()
print(f"Actual total revenue:    {actual_total:,.2f}")
print(f"Predicted total revenue: {predicted_total:,.2f}")

model_comparison = pd.DataFrame({
    "Model": [
        "BG/NBD + Gamma-Gamma",
        "Behavioural XGBoost",
        "Hybrid XGBoost"
    ],
    "Prediction Target": [
        "Purchase count",
        "Customer revenue",
        "Customer revenue"
    ],
    "MAE": [
        1.11,
        817.58,
        811.90
    ],
    "RMSE": [
        2.55,
        1440.29,
        1436.69
    ],
    "R²": [
        "N/A",
        0.60,
        0.60
    ]
})

print(model_comparison)

customer_features["Hybrid_Predicted_Revenue"] = hybrid_xgb.predict(X_hybrid)
customer_features["Customer_Percentile"] = customer_features["Hybrid_Predicted_Revenue"].rank(
    method="first", ascending=False, pct=True
    )
conditions = [
    customer_features["Customer_Percentile"] <=0.10,
    customer_features["Customer_Percentile"] <= 0.30,
    customer_features["Customer_Percentile"] <= 0.5,
    ]

customer_features["Customer_Segment"] = np.select(conditions,["VIP","Loyal", "Occasional"], default="Low-Value")

print(customer_features["Customer_Segment"].value_counts())
segment_summary = customer_features.groupby("Customer_Segment").agg(
    CustomerCount = ("CustomerID", "nunique"),
    TotalRevenue = ("Actual_2025_Revenue", "sum"),
    AverageRevenue = ("Actual_2025_Revenue", "mean"),
    Frequency = ("TotalOrders", "mean"),
    AverageTenure = ("Tenure", "mean"),
    LastPurchase = ("Recency", "mean")
)
print(segment_summary)

cutoff_date = df["OrderDate"].max()

all_orders = (
    orders[orders["OrderDate"] <= cutoff_date]
    .drop_duplicates(["CustomerID", "OrderID"])
    .sort_values(["CustomerID", "OrderDate"])
    .copy()
)

first_dates = all_orders.groupby("CustomerID")["OrderDate"].min()

cohort_ids = first_dates[
    (first_dates >= pd.Timestamp("2022-01-01")) &
    (first_dates < pd.Timestamp("2025-01-01"))
].index

cohort_orders = all_orders[
    all_orders["CustomerID"].isin(cohort_ids)
].copy()

survival_data = cohort_orders.groupby("CustomerID").agg(
    First_Purchase=("OrderDate", "min"),
    Total_Orders=("OrderID", "nunique")
).reset_index()

second_purchase = (
    cohort_orders.groupby("CustomerID")["OrderDate"]
    .apply(lambda dates: dates.iloc[1] if len(dates) > 1 else pd.NaT)
)

survival_data["NextPurchase"] = survival_data["CustomerID"].map(
    second_purchase
)

survival_data["Event"] = (
    survival_data["NextPurchase"].notna().astype(int)
)

survival_data["Duration"] = (
    survival_data["NextPurchase"]
    .fillna(cutoff_date)
    - survival_data["First_Purchase"]
).dt.days

kmf = KaplanMeierFitter()

kmf.fit(
    durations=survival_data["Duration"],
    event_observed=survival_data["Event"],
    label="Probability of No Second Purchase"
)

plt.close("all")

fig, ax = plt.subplots(figsize=(10, 5))

kmf.plot_survival_function(ci_show=False, ax=ax)

ax.set_title("Time Until Second Purchase")
ax.set_xlabel("Days Since First Purchase")
ax.set_ylabel("Probability of Not Yet Making a Second Purchase")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

plt.tight_layout()
#plt.show()

print("Observation cutoff:", cutoff_date)
print("Customers in cohort:", len(survival_data))
print("Second-purchase events:", survival_data["Event"].sum())
print("Censored customers:", (survival_data["Event"] == 0).sum())

print(survival_data.head())

last_purchase_dates = df.groupby("CustomerID")["OrderDate"].max()
days_since_last = (cutoff_date - last_purchase_dates).dt.days

churned_customer = days_since_last[days_since_last > 400].index.tolist()
winback_customer = days_since_last[(days_since_last >= 300) & (days_since_last <= 400)].index.tolist()
active_customer = days_since_last[days_since_last < 300].index.tolist()

print("\n--- CUSTOMER CHURN SEGMENTATION REPORT ---")
print(f"Observation Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Active Customers (<300 days): {len(active_customer)}")
print(f"Win-Back Targets (300-400 days): {len(winback_customer)}")
print(f"Officially Churned (>400 days): {len(churned_customer)}")

customer_features["DaysSinceLastPurchase"] = customer_features["CustomerID"].map(days_since_last)

conditions_time = [
    customer_features["DaysSinceLastPurchase"] > 400,
    (customer_features["DaysSinceLastPurchase"] >= 300) & (customer_features["DaysSinceLastPurchase"] <= 400)
]
choices_time = ["Churned", "Win-Back"]
customer_features["Churn_Status"] = np.select(conditions_time, choices_time, default="Active")

matrix_report = pd.crosstab(
    customer_features["Customer_Segment"], 
    customer_features["Churn_Status"],
    margins=True
)
print("\n---THE CONNECTED MATRIX REPORT ---")
print(matrix_report)

matrix_data = matrix_report.iloc[:-1, :-1]

ax = matrix_data.plot(kind="bar", figsize=(10, 6), width=0.8)

plt.title("Customer Segments by Churn Status Gaps")
plt.xlabel("Customer Value Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Churn Status")

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    fontsize=9, xytext=(0, 3), 
                    textcoords='offset points')

plt.tight_layout()
#plt.show()

all_winback = customer_features[customer_features["Churn_Status"] == "Win-Back"]

total_customers = len(customer_features)
total_winback_customers = len(all_winback)

target_count_10 = int(round(total_winback_customers * 0.10))
total_winback_pool_value = all_winback["Predicted_2025_Revenue"].sum()
recovered_revenue_10 = total_winback_pool_value * 0.10

baseline_active_rev = customer_features[customer_features["Churn_Status"] == "Active"]["Predicted_2025_Revenue"].sum()
percentage_revenue_increase = (recovered_revenue_10 / baseline_active_rev) * 100

pct_of_total_customers_targeted = (target_count_10 / total_customers) * 100

print("\n---10% WIN-BACK SIMULATION REPORT ---")
print(f"Total Customers:                      {total_customers}")
print(f"Total Win-Back Customers Available:       {total_winback_customers}")
print(f"Simulation Targeting Strategy:            10.00% of Win-Back Pool")
print(f"Number of Customers Targeted:             {target_count_10}")
print(f"Percentage of Total Customer Targeted:   {pct_of_total_customers_targeted:.2f}%")
print(f"Total Value in Win-Back Pool:            ${total_winback_pool_value:,.2f}")
print(f"Projected Revenue Recovered (10% Win):    ${recovered_revenue_10:,.2f}")
print(f"Percentage Revenue Increase:              {percentage_revenue_increase:.1f}%")

explainer = shap.TreeExplainer(hybrid_xgb)
shap_values = explainer.shap_values(X_hybrid_test)
base_value = float(np.ravel(explainer.expected_value)[0])

print("------------")
print("SHAP values shape:", shap_values.shape)
print("X_hybrid_test shape:", X_hybrid_test.shape)
print(f"Baseline (average prediction): {base_value:,.2f}")

plt.close("all")
plt.figure(figsize = (10, 8))
shap.summary_plot(shap_values, X_hybrid_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Hybrid XGBoost)")
plt.tight_layout()
#plt.show()

plt.close("all")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_hybrid_test, show=False)
plt.title("SHAP Summary Plot (Hybrid XGBoost)")
plt.tight_layout()
#plt.show()

plt.close("all")
plt.figure(figsize=(9, 6))
shap.dependence_plot("AverageOrderValue", shap_values, X_hybrid_test, show=False)
plt.title("SHAP Dependence: AverageOrderValue")
plt.tight_layout()
#plt.show()

plt.close("all")
i = int(np.argmax(hybrid_predictions))
explanation = shap.Explanation(
    values=shap_values[i],
    base_values=base_value,
    data=X_hybrid_test.iloc[i].values,
    feature_names=X_hybrid_test.columns.tolist()
)

shap.plots.waterfall(explanation, max_display=12, show=False)
plt.tight_layout()
#plt.show()

print(f"Baseline (average prediction): {base_value:,.2f}")
print(f"Prediction for this customer:  {hybrid_predictions[i]:,.2f}")

plt.close("all")
j = int(np.argmin(hybrid_predictions))
explanation_low = shap.Explanation(
    values=shap_values[j],
    base_values=base_value,
    data=X_hybrid_test.iloc[j].values,
    feature_names=X_hybrid_test.columns.tolist()
)

shap.plots.waterfall(explanation_low, max_display=12, show=False)
plt.tight_layout()
#plt.show()

print(f"Baseline (average prediction): {base_value:,.2f}")
print(f"Prediction for this customer:  {hybrid_predictions[j]:,.2f}")

print("\n---THE CONNECTED MATRIX REPORT ---")
print(matrix_report)
