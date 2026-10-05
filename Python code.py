import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss
)

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

print()
print("2025_revenue_distribution")
print("--------------------------")
print("Total Customers:", len(y_test))
print("Zero Revenue Customers: ",  (y_test==0).sum())
print("Positive Revenue Customers: ",  (y_test>0).sum())
print("Neagtive Revenue Customers: ",  (y_test<0).sum())

print()
print("Percentages")
print("Zero:", f"{(y_test==0).mean()*100:.2f}%")
print("Positive:", f"{(y_test>0).mean()*100:.2f}%")
print("Negative:", f"{(y_test<0).mean()*100:.2f}%")

print()
print("Revenue Stats:")
print(y_test.describe())

print()
print("Positive Revenue Stats:")
print(y_test[y_test>0].describe())

y_train_class = (y_train > 0).astype(int)
y_test_class = (y_test > 0).astype(int)

print("Training classification target:")
print(y_train_class.value_counts())

print("\nTraining percentages:")
print(
    (y_train_class.value_counts(normalize=True) * 100)
    .round(2)
)

print("\nTesting classification target:")
print(y_test_class.value_counts())

print("\nTesting percentages:")
print(
    (y_test_class.value_counts(normalize=True) * 100)
    .round(2)
)

purchase_xgb = XGBClassifier(
    objective = "binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

purchase_xgb.fit(X_train, y_train_class)
purchase_probability = purchase_xgb.predict_proba(X_test)[:, 1]

print("First 10 purchase probabilities:")
print(purchase_probability[:10])

roc_auc = roc_auc_score(
    y_test_class,
    purchase_probability
)

logloss = log_loss(
    y_test_class,
    purchase_probability
)

brier = brier_score_loss(
    y_test_class,
    purchase_probability
)

print("Stage 1 Classification Performance")
print("-----------------------------------")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"Log Loss:  {logloss:.4f}")
print(f"Brier Score: {brier:.4f}")

positive_train_mask = y_train > 0

X_value_train = X_train.loc[positive_train_mask].copy()
y_value_train = y_train.loc[positive_train_mask].copy()

print("Stage 2 training customers:", len(y_value_train))
print()
print("Positive revenue statistics:")
print(y_value_train.describe())

y_value_train_log = np.log1p(y_value_train)

print("Original revenue:")
print(y_value_train.head())

print("\nLog-transformed revenue:")
print(y_value_train_log.head())

print("\nLog-transformed statistics:")
print(y_value_train_log.describe())

value_xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

value_xgb.fit(
    X_value_train,
    y_value_train_log
)

predicted_log_revenue = value_xgb.predict(X_test)

conditional_revenue_prediction = np.expm1(
    predicted_log_revenue
)

print("First 10 conditional revenue predictions:")
print(conditional_revenue_prediction[:10])

print("\nConditional revenue statistics:")
print(
    pd.Series(conditional_revenue_prediction).describe()
)

two_stage_predictions = (
    purchase_probability *
    conditional_revenue_prediction
)

print("First 10 two-stage revenue predictions:")
print(two_stage_predictions[:10])

print("\nTwo-stage prediction statistics:")
print(
    pd.Series(two_stage_predictions).describe()
)

two_stage_mae = mean_absolute_error(
    y_test,
    two_stage_predictions
)

two_stage_rmse = mean_squared_error(
    y_test,
    two_stage_predictions
) ** 0.5

two_stage_r2 = r2_score(
    y_test,
    two_stage_predictions
)

print("Two-Stage XGBoost Performance")
print("-----------------------------")
print(f"MAE:  {two_stage_mae:.2f}")
print(f"RMSE: {two_stage_rmse:.2f}")
print(f"R²:   {two_stage_r2:.4f}")

print(mae)
print(rmse)