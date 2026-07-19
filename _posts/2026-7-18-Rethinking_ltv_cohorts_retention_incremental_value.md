# Rethinking LTV: Cohorts, Retention, and Incremental Value

Lifetime value is often presented as a standalone model. In real marketing analytics work, it is more useful to treat LTV as a decision framework.

The practical question is not only:

> How much value will this customer generate?

It is also:

> How much value is incremental because of a marketing action?

This notebook builds a synthetic customer panel to demonstrate four connected ideas:

1. Cohort LTV: how value accumulates over customer age.
2. Retention curves: how many customers remain active over time.
3. Survival-adjusted LTV: how future value changes when we account for churn.
4. Incremental LTV: why observed treated-vs-untreated value can be biased.

The data is generated inside the notebook. It is designed to look like a customer transaction panel, but the treatment, retention, and causal effects are simulated so the true structure is known.

## Step 0 — Create a semi-synthetic customer panel

The goal is not to perfectly reproduce a real business. The goal is to create a realistic enough environment where the main LTV issues are visible.

We simulate customers with different levels of historical engagement and monetary value. Then we create monthly records, treatment exposure, churn, conversion, and value.

The important design choice is that treatment is not randomly assigned. Higher-value customers are more likely to be targeted. This creates selection bias, which is exactly the problem that often makes observed LTV misleading.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

np.random.seed(42)

# -----------------------------
# Simulation settings
# -----------------------------
N_CUSTOMERS = 50_000
N_MONTHS = 12
DISCOUNT_RATE_MONTHLY = 0.01

# -----------------------------
# Customer-level heterogeneity
# -----------------------------
customers = pd.DataFrame({
    "customer_id": np.arange(N_CUSTOMERS),
})

# Cohort month is the first active month. This lets us compare customers by age since acquisition.
customers["cohort_month"] = np.random.choice(np.arange(1, 7), size=N_CUSTOMERS, p=[0.18, 0.17, 0.17, 0.16, 0.16, 0.16])

# Latent customer quality drives spend, conversion, retention, and targeting probability.
customers["customer_quality"] = np.random.normal(0, 1, N_CUSTOMERS)

# Public-dataset-like behavioral features. These are not real public data; they mimic common transaction features.
customers["historical_frequency"] = np.random.poisson(lam=np.exp(1.1 + 0.35 * customers["customer_quality"]))
customers["historical_monetary"] = np.random.lognormal(mean=3.2 + 0.35 * customers["customer_quality"], sigma=0.55)
customers["recency"] = np.clip(np.random.normal(45 - 8 * customers["customer_quality"], 18, N_CUSTOMERS), 1, 120)

# Simple customer segments based on historical monetary value.
customers["customer_segment"] = pd.qcut(
    customers["historical_monetary"],
    q=3,
    labels=["low_value", "mid_value", "high_value"]
)

# Baseline conversion propensity: what would happen without treatment.
logit_base = (
    -3.4
    + 0.35 * customers["customer_quality"]
    + 0.025 * customers["historical_frequency"]
    - 0.006 * customers["recency"]
)
customers["baseline_propensity"] = 1 / (1 + np.exp(-logit_base))

# Baseline monthly value conditional on conversion.
customers["baseline_order_value"] = 20 + 0.55 * customers["historical_monetary"] + np.random.normal(0, 8, N_CUSTOMERS)
customers["baseline_order_value"] = customers["baseline_order_value"].clip(lower=5)

customers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>cohort_month</th>
      <th>customer_quality</th>
      <th>historical_frequency</th>
      <th>historical_monetary</th>
      <th>recency</th>
      <th>customer_segment</th>
      <th>baseline_propensity</th>
      <th>baseline_order_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>-0.019063</td>
      <td>3</td>
      <td>20.796782</td>
      <td>48.554004</td>
      <td>mid_value</td>
      <td>0.026008</td>
      <td>21.000071</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>1.207288</td>
      <td>6</td>
      <td>13.759468</td>
      <td>46.345481</td>
      <td>low_value</td>
      <td>0.042880</td>
      <td>17.735016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>0.625292</td>
      <td>2</td>
      <td>38.360114</td>
      <td>43.901992</td>
      <td>high_value</td>
      <td>0.032466</td>
      <td>48.544132</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>-0.804689</td>
      <td>1</td>
      <td>41.829695</td>
      <td>53.988580</td>
      <td>high_value</td>
      <td>0.018333</td>
      <td>45.811020</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>1.787113</td>
      <td>7</td>
      <td>67.892000</td>
      <td>33.152118</td>
      <td>high_value</td>
      <td>0.057409</td>
      <td>43.059870</td>
    </tr>
  </tbody>
</table>
</div>



The customer table creates realistic variation before any marketing action happens. Some customers have higher frequency, higher monetary value, lower recency, and higher baseline propensity.

This matters because targeting systems often prefer these customers. If we later compare treated and untreated customers directly, treated customers may look better even if the treatment itself caused only a modest lift.


```python
# -----------------------------
# Expand to customer-month panel
# -----------------------------
panel = customers.loc[customers.index.repeat(N_MONTHS)].copy()
panel["calendar_month"] = np.tile(np.arange(1, N_MONTHS + 1), N_CUSTOMERS)
panel["months_since_cohort"] = panel["calendar_month"] - panel["cohort_month"] + 1

# Keep only months where the customer has already entered the population.
panel = panel[panel["months_since_cohort"] >= 1].copy()

# Treatment assignment is intentionally biased.
# Higher-quality and higher-value customers are more likely to receive the offer.
treat_logit = (
    -1.5
    + 0.75 * panel["customer_quality"]
    + 0.012 * panel["historical_frequency"]
    + 0.007 * panel["historical_monetary"]
    - 0.004 * panel["recency"]
)
panel["treatment_probability"] = 1 / (1 + np.exp(-treat_logit))
panel["treatment"] = np.random.binomial(1, panel["treatment_probability"])

# True treatment effects are known because this is simulated.
# Treatment modestly increases conversion and slightly improves retention.
panel["true_conversion_lift"] = 0.025
panel["true_retention_lift"] = 0.010

# Base churn probability decreases for better customers.
base_churn = (
    0.18
    - 0.025 * panel["customer_quality"]
    - 0.0007 * panel["historical_monetary"]
    + 0.005 * panel["months_since_cohort"]
)

panel["monthly_churn_probability"] = (
    base_churn
    - panel["true_retention_lift"] * panel["treatment"]
).clip(0.02, 0.45)

# Simulate active status. Once a customer churns, they stay inactive.
panel = panel.sort_values(["customer_id", "calendar_month"]).reset_index(drop=True)
panel["active_flag"] = 0
panel["churn_flag"] = 0

for customer_id, idx in panel.groupby("customer_id").groups.items():
    active = True
    for i in idx:
        if active:
            panel.at[i, "active_flag"] = 1
            churn = np.random.binomial(1, panel.at[i, "monthly_churn_probability"])
            panel.at[i, "churn_flag"] = churn
            if churn == 1:
                active = False
        else:
            panel.at[i, "active_flag"] = 0
            panel.at[i, "churn_flag"] = 0

# Conversion probability is active-dependent and treatment-dependent.
panel["conversion_probability"] = (
    panel["baseline_propensity"]
    + panel["true_conversion_lift"] * panel["treatment"]
).clip(0, 0.80)

panel["conversion"] = np.random.binomial(1, panel["conversion_probability"] * panel["active_flag"])

# Monthly value is realized only when active and converted.
noise = np.random.normal(0, 10, len(panel))
panel["monthly_value"] = (
    panel["conversion"]
    * (panel["baseline_order_value"] + 5 * panel["treatment"] + noise)
).clip(lower=0)

panel["discount_factor"] = 1 / ((1 + DISCOUNT_RATE_MONTHLY) ** (panel["months_since_cohort"] - 1))
panel["discounted_monthly_value"] = panel["monthly_value"] * panel["discount_factor"]

panel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>cohort_month</th>
      <th>customer_quality</th>
      <th>historical_frequency</th>
      <th>historical_monetary</th>
      <th>recency</th>
      <th>customer_segment</th>
      <th>baseline_propensity</th>
      <th>baseline_order_value</th>
      <th>calendar_month</th>
      <th>...</th>
      <th>true_conversion_lift</th>
      <th>true_retention_lift</th>
      <th>monthly_churn_probability</th>
      <th>active_flag</th>
      <th>churn_flag</th>
      <th>conversion_probability</th>
      <th>conversion</th>
      <th>monthly_value</th>
      <th>discount_factor</th>
      <th>discounted_monthly_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>-0.019063</td>
      <td>3</td>
      <td>20.796782</td>
      <td>48.554004</td>
      <td>mid_value</td>
      <td>0.026008</td>
      <td>21.000071</td>
      <td>3</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.01</td>
      <td>0.170919</td>
      <td>1</td>
      <td>0</td>
      <td>0.026008</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>-0.019063</td>
      <td>3</td>
      <td>20.796782</td>
      <td>48.554004</td>
      <td>mid_value</td>
      <td>0.026008</td>
      <td>21.000071</td>
      <td>4</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.01</td>
      <td>0.175919</td>
      <td>1</td>
      <td>0</td>
      <td>0.026008</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.990099</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>-0.019063</td>
      <td>3</td>
      <td>20.796782</td>
      <td>48.554004</td>
      <td>mid_value</td>
      <td>0.026008</td>
      <td>21.000071</td>
      <td>5</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.01</td>
      <td>0.170919</td>
      <td>1</td>
      <td>0</td>
      <td>0.051008</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.980296</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>-0.019063</td>
      <td>3</td>
      <td>20.796782</td>
      <td>48.554004</td>
      <td>mid_value</td>
      <td>0.026008</td>
      <td>21.000071</td>
      <td>6</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.01</td>
      <td>0.175919</td>
      <td>1</td>
      <td>0</td>
      <td>0.051008</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.970590</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>-0.019063</td>
      <td>3</td>
      <td>20.796782</td>
      <td>48.554004</td>
      <td>mid_value</td>
      <td>0.026008</td>
      <td>21.000071</td>
      <td>7</td>
      <td>...</td>
      <td>0.025</td>
      <td>0.01</td>
      <td>0.180919</td>
      <td>1</td>
      <td>0</td>
      <td>0.051008</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.960980</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



At this point the panel contains both observed outcomes and known simulated truth.

The observed outcomes are what an analyst would normally see: treatment, active flag, conversion, and monthly value.

The known truth is what lets us demonstrate the key lesson: the treatment has a real incremental effect, but the observed treated group also has better baseline customers. That combination makes naive LTV comparisons too optimistic.

## Step 1 — Cohort LTV

Cohort analysis groups customers by when they entered the population and then tracks value by customer age.

This avoids comparing a new customer in month 1 with an older customer in month 10. For LTV, customer age matters because value has had different amounts of time to accumulate.


```python
cohort_ltv = (
    panel.groupby(["cohort_month", "months_since_cohort"], as_index=False)
    .agg(
        customers=("customer_id", "nunique"),
        monthly_value=("monthly_value", "mean"),
        discounted_monthly_value=("discounted_monthly_value", "mean")
    )
    .sort_values(["cohort_month", "months_since_cohort"])
)

cohort_ltv["cumulative_ltv"] = cohort_ltv.groupby("cohort_month")["monthly_value"].cumsum()
cohort_ltv["cumulative_discounted_ltv"] = cohort_ltv.groupby("cohort_month")["discounted_monthly_value"].cumsum()

cohort_ltv.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cohort_month</th>
      <th>months_since_cohort</th>
      <th>customers</th>
      <th>monthly_value</th>
      <th>discounted_monthly_value</th>
      <th>cumulative_ltv</th>
      <th>cumulative_discounted_ltv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>9010</td>
      <td>1.428473</td>
      <td>1.428473</td>
      <td>1.428473</td>
      <td>1.428473</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>9010</td>
      <td>1.306431</td>
      <td>1.293496</td>
      <td>2.734904</td>
      <td>2.721969</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>9010</td>
      <td>1.124744</td>
      <td>1.102582</td>
      <td>3.859648</td>
      <td>3.824551</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>9010</td>
      <td>0.986077</td>
      <td>0.957076</td>
      <td>4.845725</td>
      <td>4.781627</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>9010</td>
      <td>0.787939</td>
      <td>0.757194</td>
      <td>5.633663</td>
      <td>5.538821</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>6</td>
      <td>9010</td>
      <td>0.672617</td>
      <td>0.639972</td>
      <td>6.306280</td>
      <td>6.178793</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>7</td>
      <td>9010</td>
      <td>0.610392</td>
      <td>0.575017</td>
      <td>6.916672</td>
      <td>6.753809</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>8</td>
      <td>9010</td>
      <td>0.434869</td>
      <td>0.405610</td>
      <td>7.351541</td>
      <td>7.159419</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>9</td>
      <td>9010</td>
      <td>0.449860</td>
      <td>0.415438</td>
      <td>7.801401</td>
      <td>7.574857</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>10</td>
      <td>9010</td>
      <td>0.416923</td>
      <td>0.381209</td>
      <td>8.218323</td>
      <td>7.956067</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>11</td>
      <td>9010</td>
      <td>0.248660</td>
      <td>0.225109</td>
      <td>8.466983</td>
      <td>8.181175</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>12</td>
      <td>9010</td>
      <td>0.211322</td>
      <td>0.189413</td>
      <td>8.678306</td>
      <td>8.370588</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(9, 5))
for cohort, d in cohort_ltv.groupby("cohort_month"):
    plt.plot(d["months_since_cohort"], d["cumulative_ltv"], marker="o", label=f"Cohort {cohort}")

plt.title("Cumulative LTV by Cohort")
plt.xlabel("Months Since Cohort Start")
plt.ylabel("Average Cumulative Value per Customer")
plt.legend(title="Cohort Month")
plt.grid(True, alpha=0.3)
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_8_0_ltv.png">
  <figcaption style="text-align:center;">Fig1.Average Cumulative Value per Customer overtime </figcaption>
</figure>
    


The cumulative LTV curves show how value builds over time. Earlier cohorts have longer observation windows, so they naturally have more opportunity to accumulate value.

This is why raw total value can be misleading. A cohort with lower total value may simply be younger, not worse.

For practical marketing analytics, cohort LTV is often the first diagnostic before any predictive or causal modeling. It answers a basic question: how does value accumulate as customers age?

## Step 2 — Retention curves

Retention explains whether customers continue to remain active long enough to generate future value.

A customer with high initial conversion value may still have low lifetime value if they churn quickly. A customer with moderate short-term value can become more valuable if they remains active for many periods.


```python
retention = (
    panel.groupby(["cohort_month", "months_since_cohort"], as_index=False)
    .agg(retention_rate=("active_flag", "mean"))
)

retention.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cohort_month</th>
      <th>months_since_cohort</th>
      <th>retention_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.838180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0.704440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0.586459</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0.487125</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>6</td>
      <td>0.396892</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>7</td>
      <td>0.331188</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>8</td>
      <td>0.270255</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>9</td>
      <td>0.222863</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>10</td>
      <td>0.180688</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>11</td>
      <td>0.144950</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>12</td>
      <td>0.113430</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(9, 5))
for cohort, d in retention.groupby("cohort_month"):
    plt.plot(d["months_since_cohort"], d["retention_rate"], marker="o", label=f"Cohort {cohort}")

plt.title("Retention Curves by Cohort")
plt.xlabel("Months Since Cohort Start")
plt.ylabel("Share of Customers Active")
plt.ylim(0, 1.05)
plt.legend(title="Cohort Month")
plt.grid(True, alpha=0.3)
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_12_0_ltv.png">
  <figcaption style="text-align:center;">Fig2.Retention Curves by Cohort </figcaption>
</figure>
    
    


The retention curve shows the percentage of each cohort that remains active at each customer age.

This is the bridge between conversion modeling and lifetime value. Conversion tells us whether value happens in a given period. Retention tells us how many future periods remain available for value to happen.

In an LTV system, short-term propensity and long-term retention should not be treated as separate business questions. They are two parts of the same expected value problem.

## Step 3 — Survival-adjusted LTV

A survival-style view does not assume that every customer remains active for the full horizon. Instead, each future month is weighted by the probability that the customer is still active.

A compact way to write the idea is:

LTVᵢ = Σₜ P(activeᵢ,ₜ) × E(valueᵢ,ₜ) × discountₜ




This is enough for a practical marketing blog post. The goal is not to turn the analysis into a survival modeling tutorial. The goal is to show that future value should be adjusted for the probability of remaining active.


```python
# Estimate empirical survival by customer segment and customer age.
segment_survival = (
    panel.groupby(["customer_segment", "months_since_cohort"], as_index=False, observed=True)
    .agg(empirical_survival=("active_flag", "mean"))
)

segment_value = (
    panel[panel["active_flag"] == 1]
    .groupby(["customer_segment", "months_since_cohort"], as_index=False, observed=True)
    .agg(avg_value_if_active=("monthly_value", "mean"))
)

survival_ltv = segment_survival.merge(
    segment_value,
    on=["customer_segment", "months_since_cohort"],
    how="left"
)

survival_ltv["avg_value_if_active"] = survival_ltv["avg_value_if_active"].fillna(0)
survival_ltv["discount_factor"] = 1 / ((1 + DISCOUNT_RATE_MONTHLY) ** (survival_ltv["months_since_cohort"] - 1))
survival_ltv["survival_adjusted_value"] = (
    survival_ltv["empirical_survival"]
    * survival_ltv["avg_value_if_active"]
    * survival_ltv["discount_factor"]
)

survival_ltv["survival_adjusted_ltv"] = (
    survival_ltv.groupby("customer_segment", observed=True)["survival_adjusted_value"].cumsum()
)

survival_ltv.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_segment</th>
      <th>months_since_cohort</th>
      <th>empirical_survival</th>
      <th>avg_value_if_active</th>
      <th>discount_factor</th>
      <th>survival_adjusted_value</th>
      <th>survival_adjusted_ltv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>low_value</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0.720686</td>
      <td>1.000000</td>
      <td>0.720686</td>
      <td>0.720686</td>
    </tr>
    <tr>
      <th>1</th>
      <td>low_value</td>
      <td>2</td>
      <td>0.814664</td>
      <td>0.673658</td>
      <td>0.990099</td>
      <td>0.543371</td>
      <td>1.264057</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low_value</td>
      <td>3</td>
      <td>0.659987</td>
      <td>0.754161</td>
      <td>0.980296</td>
      <td>0.487929</td>
      <td>1.751985</td>
    </tr>
    <tr>
      <th>3</th>
      <td>low_value</td>
      <td>4</td>
      <td>0.530629</td>
      <td>0.868627</td>
      <td>0.970590</td>
      <td>0.447364</td>
      <td>2.199349</td>
    </tr>
    <tr>
      <th>4</th>
      <td>low_value</td>
      <td>5</td>
      <td>0.426591</td>
      <td>0.732034</td>
      <td>0.960980</td>
      <td>0.300094</td>
      <td>2.499443</td>
    </tr>
    <tr>
      <th>5</th>
      <td>low_value</td>
      <td>6</td>
      <td>0.336113</td>
      <td>0.854610</td>
      <td>0.951466</td>
      <td>0.273304</td>
      <td>2.772747</td>
    </tr>
    <tr>
      <th>6</th>
      <td>low_value</td>
      <td>7</td>
      <td>0.265075</td>
      <td>0.840886</td>
      <td>0.942045</td>
      <td>0.209980</td>
      <td>2.982727</td>
    </tr>
    <tr>
      <th>7</th>
      <td>low_value</td>
      <td>8</td>
      <td>0.211907</td>
      <td>0.848681</td>
      <td>0.932718</td>
      <td>0.167741</td>
      <td>3.150468</td>
    </tr>
    <tr>
      <th>8</th>
      <td>low_value</td>
      <td>9</td>
      <td>0.164871</td>
      <td>0.687027</td>
      <td>0.923483</td>
      <td>0.104604</td>
      <td>3.255072</td>
    </tr>
    <tr>
      <th>9</th>
      <td>low_value</td>
      <td>10</td>
      <td>0.128887</td>
      <td>0.809115</td>
      <td>0.914340</td>
      <td>0.095352</td>
      <td>3.350424</td>
    </tr>
    <tr>
      <th>10</th>
      <td>low_value</td>
      <td>11</td>
      <td>0.099020</td>
      <td>1.121873</td>
      <td>0.905287</td>
      <td>0.100567</td>
      <td>3.450990</td>
    </tr>
    <tr>
      <th>11</th>
      <td>low_value</td>
      <td>12</td>
      <td>0.080727</td>
      <td>0.857025</td>
      <td>0.896324</td>
      <td>0.062012</td>
      <td>3.513002</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(9, 5))
for segment, d in survival_ltv.groupby("customer_segment", observed=True):
    plt.plot(d["months_since_cohort"], d["survival_adjusted_ltv"], marker="o", label=str(segment))

plt.title("Survival-Adjusted LTV by Customer Segment")
plt.xlabel("Months Since Cohort Start")
plt.ylabel("Average Survival-Adjusted LTV")
plt.legend(title="Segment")
plt.grid(True, alpha=0.3)
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_16_0_ltv.png">
  <figcaption style="text-align:center;">Fig3.Survival-Adjusted LTV by Customer Segment </figcaption>
</figure>
    
    


Survival-adjusted LTV separates two mechanisms:

1. How much value active customers generate.
2. How likely customers are to remain active long enough to generate that value.

This is useful because high-value segments can look attractive for two different reasons. They may spend more when active, or they may remain active longer. For marketing decisions, those are different levers.

A retention-based extension is the main additional technical layer beyond standard propensity-and-value ranking.

## Step 4 — Causal / incremental LTV

Observed LTV is not the same as incremental LTV.

If treatment is targeted toward customers who already have high baseline value, then treated customers will have higher LTV even without any treatment effect. A naive comparison will attribute too much value to the marketing action.

This section compares three estimates:

1. Naive observed difference: average treated LTV minus average untreated LTV.
2. Model-adjusted difference: regression adjustment using pre-treatment customer features.
3. True simulated incremental LTV: known because we generated the treatment effect.


```python
# Customer-level LTV over the observed horizon.
customer_ltv = (
    panel.groupby("customer_id", as_index=False)
    .agg(
        observed_ltv=("discounted_monthly_value", "sum"),
        ever_treated=("treatment", "max"),
        avg_treatment_probability=("treatment_probability", "mean"),
        historical_frequency=("historical_frequency", "first"),
        historical_monetary=("historical_monetary", "first"),
        recency=("recency", "first"),
        customer_quality=("customer_quality", "first"),
        baseline_propensity=("baseline_propensity", "first"),
        customer_segment=("customer_segment", "first")
    )
)

naive_treated_ltv = customer_ltv.loc[customer_ltv["ever_treated"] == 1, "observed_ltv"].mean()
naive_untreated_ltv = customer_ltv.loc[customer_ltv["ever_treated"] == 0, "observed_ltv"].mean()
naive_difference = naive_treated_ltv - naive_untreated_ltv

naive_treated_ltv, naive_untreated_ltv, naive_difference
```




    (np.float64(8.894354703021222),
     np.float64(2.717358585046762),
     np.float64(6.1769961179744595))



The naive comparison is easy to compute, but it is not a causal estimate.

Because targeting is biased toward higher-quality customers, the treated group has higher expected value before treatment. The naive difference mixes true treatment impact with pre-existing customer differences.


```python
# Regression adjustment using pre-treatment features.
features = ["ever_treated", "historical_frequency", "historical_monetary", "recency", "baseline_propensity", "customer_segment"]
X = customer_ltv[features]
y = customer_ltv["observed_ltv"]

preprocess = ColumnTransformer(
    transformers=[
        ("segment", OneHotEncoder(drop="first"), ["customer_segment"]),
    ],
    remainder="passthrough"
)

regression_adjustment = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

regression_adjustment.fit(X, y)

# Estimate adjusted treatment effect by predicting each customer twice:
# once as treated and once as untreated, while holding other features fixed.
X_treated = X.copy()
X_untreated = X.copy()
X_treated["ever_treated"] = 1
X_untreated["ever_treated"] = 0

pred_treated = regression_adjustment.predict(X_treated)
pred_untreated = regression_adjustment.predict(X_untreated)

adjusted_difference = np.mean(pred_treated - pred_untreated)
adjusted_difference
```




    np.float64(-0.22865059552685008)




```python
# Approximate the true simulated incremental LTV by replaying the known treatment effects.
# This uses the known conversion and retention lift used in the simulation.
# It is not something available in real observational data; it is included here as a benchmark.

true_incremental_by_month = panel.copy()

# Expected incremental conversion value among active treated observations.
true_incremental_by_month["incremental_conversion_value"] = (
    true_incremental_by_month["active_flag"]
    * true_incremental_by_month["treatment"]
    * true_incremental_by_month["true_conversion_lift"]
    * true_incremental_by_month["baseline_order_value"]
    * true_incremental_by_month["discount_factor"]
)

# Approximate incremental retention value:
# if treatment lowers churn, it preserves some future customer value.
# This simple approximation values the retained month using expected baseline value.
true_incremental_by_month["incremental_retention_value"] = (
    true_incremental_by_month["treatment"]
    * true_incremental_by_month["true_retention_lift"]
    * true_incremental_by_month["baseline_propensity"]
    * true_incremental_by_month["baseline_order_value"]
    * true_incremental_by_month["discount_factor"]
)

true_incremental_total = (
    true_incremental_by_month["incremental_conversion_value"].sum()
    + true_incremental_by_month["incremental_retention_value"].sum()
)

true_incremental_per_treated_customer = true_incremental_total / customer_ltv["ever_treated"].sum()

comparison = pd.DataFrame({
    "estimate": ["Naive observed difference", "Model-adjusted difference", "True simulated incremental LTV"],
    "ltv_difference": [naive_difference, adjusted_difference, true_incremental_per_treated_customer]
})

comparison
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimate</th>
      <th>ltv_difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Naive observed difference</td>
      <td>6.176996</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Model-adjusted difference</td>
      <td>-0.228651</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True simulated incremental LTV</td>
      <td>1.609285</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8, 5))
plt.bar(comparison["estimate"], comparison["ltv_difference"])
plt.title("Observed vs Adjusted vs True Incremental LTV")
plt.ylabel("Average LTV Difference")
plt.xticks(rotation=20, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_23_0_ltv.png">
  <figcaption style="text-align:center;">Fig4.Observed vs Adjusted vs True Incremental LTV </figcaption>
</figure>
    
    


This is the central lesson.

The naive observed difference is usually too high because the treated group was not randomly selected. Treated customers were already more valuable before the treatment.

The model-adjusted estimate moves closer to the simulated truth because it controls for baseline differences. In real business settings, this can be improved further with randomized experiments, uplift models, doubly robust estimation, or other causal methods.

The key point is that LTV is not only a prediction problem. It is also a measurement problem. If the goal is to decide whether a marketing action creates value, the target should be incremental LTV, not observed LTV.

## Practical takeaway

A useful LTV framework combines four pieces:

1. **Cohorts** to understand how value accumulates by customer age.
2. **Retention** to measure whether customers remain active long enough to generate future value.
3. **Survival-adjusted value** to avoid assuming every customer stays active for the full horizon.
4. **Causal validation** to separate observed value from incremental value.

In production, LTV does not have to be a single specialized model. It can be built as a composition of familiar components:

Expected Value = Propensity × Value × Retention




For decision-making, the stronger version is:

Incremental LTV = Incremental Propensity × Value × Retention

That distinction matters. A high observed LTV customer is not automatically a high incremental-value customer. Marketing decisions should prioritize value that the action actually creates.
