import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

print(customer_features.head())