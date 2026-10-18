# Customer Value Forecasting System

**Modeling Purchase Lifecycles to Maximize Retention**

By **Huzaifa Saqib**

## Objective

An end-to-end customer analytics project combining probabilistic modeling, XGBoost, survival analysis, and SHAP to forecast customer revenue, segment customers by value, identify inactivity, and develop data-driven retention strategies.

## Tech Stack

Python · Pandas · NumPy · XGBoost · Scikit-learn · Lifetines · Lifelines · SHAP · Matplotlib

## Methodology and Flow

1. **Clean** the data: fix dates, handle negative and zero values, cap price outliers, standardize text and country names.
2. **Split by time:** orders up to 2024 build the features, 2025 is the holdout target.
3. **Engineer features:** 18 behavioural features (spend, frequency, recency, timing, product variety, returns, recent activity) plus 3 missing-value flags.
4. **Probabilistic models:** BG/NBD estimates the chance a customer is still active and their expected purchases; Gamma-Gamma estimates expected order value.
5. **Forecast 2025 revenue** with two XGBoost models: behavioural (21 inputs) and hybrid (25 inputs, adding the probabilistic outputs). Both are scored against a naive baseline that predicts each customer's 2025 revenue as their 2024 revenue.
6. **Segment** customers by predicted revenue: VIP (top 10%), Loyal (next 20%), Occasional (next 20%), Low-Value (bottom 50%).
7. **Model the lifecycle** with a Kaplan-Meier curve of time to second purchase.
8. **Classify churn** by days since last purchase: Active (under 300), Win-Back (300 to 400), Churned (over 400). The cutoffs are checked against the distribution of purchase gaps.
9. **Simulate win-back** revenue for the Win-Back pool.
10. **Explain** the hybrid model with SHAP.

## Results

### Model comparison

11,459 customers: 9,167 for training and 2,292 for testing.

| Model | Target | MAE | RMSE | R² |
|---|---|---|---|---|
| Naive baseline (2024 revenue) | Customer revenue | 2,543.02 | 4,589.57 | -3.04 |
| BG/NBD + Gamma-Gamma | Purchase count | 1.11 | 1.60 | n/a |
| Behavioural XGBoost | Customer revenue | 817.58 | 1,440.29 | 0.60 |
| Hybrid XGBoost | Customer revenue | 811.90 | 1,436.69 | 0.60 |

Both XGBoost models clearly beat the naive baseline. The hybrid model's MAE is about 68% lower (811.90 against 2,543.02) and its RMSE is about 69% lower. The baseline's R² of -3.04 means repeating last year's revenue is a worse forecast than predicting the average for every customer.

Between the two XGBoost models the gap is small. The hybrid model improves MAE by under 1%.
### Purchase gaps and churn thresholds

Days between consecutive purchases, across all customers. Most gaps are short, with a thin tail of long ones.

| Percentile of purchase gaps | Days |
|---|---|
| 90th | 210 |
| 95th | 298 |
| 97th | 380 |
| 98th | 449 |
| 99th | 593 |

The Win-Back line (300 days) sits at the 95th percentile, and the Churned line (400 days) falls between the 97th and 98th. The cutoffs are business choices, but the data places them in the far tail of normal buying behaviour.

### Segmentation and churn status

| Segment | Rule (predicted 2025 revenue) | Customers | Active | Win-Back | Churned |
|---|---|---|---|---|---|
| VIP | Top 10% | 1,145 | 1,076 (94.0%) | 68 (5.9%) | 1 (0.1%) |
| Loyal | Next 20% | 2,292 | 2,043 (89.1%) | 243 (10.6%) | 6 (0.3%) |
| Occasional | Next 20% | 2,292 | 2,010 (87.7%) | 278 (12.1%) | 4 (0.2%) |
| Low-Value | Bottom 50% | 5,730 | 1,435 (25.0%) | 825 (14.4%) | 3,470 (60.6%) |
| **Total** | | **11,459** | **6,564 (57.3%)** | **1,414 (12.3%)** | **3,481 (30.4%)** |

Churn is almost entirely a Low-Value problem: 99.7% of churned customers are in that segment.

### Kaplan-Meier: time to second purchase (approximate)

| Days since first purchase | Not yet made 2nd purchase | Made 2nd purchase |
|---|---|---|
| About 110 | About 50% | About 50% |
| 200 | About 29% | About 71% |
| 400 | About 18% | About 82% |
| 1,000 and beyond | About 10% | About 90% |

### Win-back simulation

| Item | Result |
|---|---|
| Total customers | 11,459 |
| Win-Back customers available (300 to 400 days) | 1,414 |
| Targeted (10% of the pool) | 141 customers (1.23% of all customers) |
| Total predicted value in the pool | $3,073,387.46 |
| Projected revenue recovered (10% win) | $307,338.75 |
| Increase vs. predicted revenue of Active customers | 1.2% |

### SHAP: what drives the forecast

| Rank | Feature | Importance (approx.) | Reading |
|---|---|---|---|
| 1 | AverageOrderValue | 950 | Higher order value raises predicted revenue |
| 2 | TotalRevenue | 560 | Past spend predicts future spend |
| 3 | Expected_Monetary_Value | 165 | Smoothed order value (Gamma-Gamma) |
| 4 | Predicted_2025_Revenue | 75 | Probabilistic revenue estimate |
| 5 | IntervalStd | 65 | Variation in time between orders |
| Bottom | Frequency, RecentOrders90, ReturnRate | Under 10 | Little influence |

| Key finding | Detail |
|---|---|
| Spending dominates | AverageOrderValue and TotalRevenue make up about two-thirds of total importance |
| Frequency barely matters | How much customers spend per order matters far more than how often they buy |
| Probabilistic features add little | The four BG/NBD and Gamma-Gamma features hold about 13% of importance, which explains the small hybrid gain |
| S-shaped order value effect | Flat below about 500, rises steeply to about 4,000, then levels off above about 5,000 |

## Recommendations

| When | Segment / area | What to do | Success measure |
|---|---|---|---|
| Now | Win-Back: VIP and Loyal (311) | Personal outreach with a tailored offer before these customers pass the 400 day line | Reactivation rate; revenue vs. the $307k scenario |
| Now | Win-Back: Occasional (278) | Automated reminder sequence with a time-limited incentive | Reactivation rate; cost per reactivation |
| Now | Low-Value (5,730; 3,470 churned) | Low-cost automated email only, avoid discounts and paid media | Cost per customer kept low |
| Next | New customers | Trigger a follow-up around day 60 to 100 without a second purchase, since about half of new customers have made a second purchase by day 110 | Second-purchase rate within 200 days |
| Next | VIP (1,145) | Account management and an alert when a VIP nears the 300 day Win-Back line | VIP Active rate (now 94%) |
| Next | Loyal to VIP | Grow order value through bundles, upsell and basket thresholds; the model shows order value, not frequency, drives revenue | Average order value; share moving to VIP |
| Next | Returns | Review returns policy and capture return reasons; track net-negative customers | Return rate; net revenue per customer |

## Limitations

- The win-back result assumes a 10% recovery rate and ignores campaign cost.
- The purchase-gap data only includes customers who came back, so it supports the 300 and 400 day churn thresholds without proving that customers silent for longer are gone for good.

## Conclusion

This project integrates customer revenue forecasting, probabilistic modeling, lifecycle analysis, segmentation, and explainable AI into a practical customer analytics framework. It demonstrates how predictive insights can inform retention priorities while acknowledging uncertainty, validation limitations, and business assumptions.
