---
layout: post
title: "Double Machine Learning Finds Segments. Bayesian Decides Which Ones to Trust"
date: 2026-04-17 12:00:00 -0500
---

## From uplift ranking to uncertainty-aware decisions on the Criteo dataset

### Abstract

This post compares Bayesian uplift modeling with Double Machine Learning (DML) on the Criteo dataset, focusing on how each approach supports decision-making under uncertainty. While DML provides strong uplift estimation and ranking capabilities, Bayesian methods offer explicit uncertainty quantification that becomes critical in sparse or noisy segments. The analysis highlights when each approach is sufficient and when uncertainty-aware modeling provides a meaningful advantage.

---

This post builds on the earlier Bayesian campaign decisioning analysis on the Hillstrom dataset, extending the discussion to compare Bayesian uplift workflows with modern causal ML approaches.

The central question is not just how to estimate uplift, but how to make reliable targeting decisions when segment-level effects are noisy and uncertain. In practice, many segments that appear highly valuable based on point estimates are also the least stable, making uncertainty an essential part of the decision process.

A natural next question is:

> How does a Bayesian uplift workflow compare to a modern causal ML approach like Double Machine Learning (DML), particularly when decisions must be made under uncertainty?

---

This analysis answers that question using the **Criteo uplift dataset**.

The goal is not to declare a single winner, but to understand where each approach is reliable and what it enables from a decision-making perspective.

---

## What this analysis demonstrates

### From the DML side

- debiased treatment-effect estimation with flexible nuisance models  
- practical individualized uplift estimation at scale  
- strong default choice when the primary objective is causal estimation  

### From the Bayesian side

- explicit uncertainty quantification  
- hierarchical shrinkage for noisy personalization segments  
- posterior predictive validation  
- more natural decision support under uncertainty  

---

## Core business question

> If I want to personalize treatment decisions, when is DML sufficient, and when does a Bayesian uplift model provide a meaningful advantage?

---

## Practical framing

These methods are not treated as ideological alternatives.

A more relevant question is:

- When is robust causal estimation sufficient?  
- When are stable, uncertainty-aware targeting decisions necessary?  

That is the focus of this analysis.



```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# EconML is used for the DML section.
# If it is not installed in your environment, uncomment the next line:
# !pip install econml

from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

```

## 1. Load the Criteo uplift dataset

This notebook assumes you already have the Criteo uplift dataset available locally.

> download data from URL: http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
Set the file path below.

### Expected structure

The notebook expects:
- one binary treatment column
- one binary outcome column
- all remaining columns to be pre-treatment covariates

Because local copies of the dataset can differ slightly in naming, I add a small helper block to standardize column names.

If your file uses different names, adjust the mapping in the next cell.




```python
# ------------------------------------------------------------------
# Set your local path here
# ------------------------------------------------------------------
DATA_PATH = "data/criteo-uplift-v2.1.csv.gz"


df = pd.read_csv(DATA_PATH,compression='gzip', nrows=500000)

# Normalize column names for easier handling
df.columns = [str(c).strip().lower() for c in df.columns]

print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:20], "...")
df.head()

```

    Shape: (13979592, 16)
    Columns: ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'treatment', 'conversion', 'visit', 'exposure'] ...
    




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
      <th>f0</th>
      <th>f1</th>
      <th>f2</th>
      <th>f3</th>
      <th>f4</th>
      <th>f5</th>
      <th>f6</th>
      <th>f7</th>
      <th>f8</th>
      <th>f9</th>
      <th>f10</th>
      <th>f11</th>
      <th>treatment</th>
      <th>conversion</th>
      <th>visit</th>
      <th>exposure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.616365</td>
      <td>10.059654</td>
      <td>8.976429</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>0.294443</td>
      <td>4.833815</td>
      <td>3.955396</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.616365</td>
      <td>10.059654</td>
      <td>9.002689</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>0.294443</td>
      <td>4.833815</td>
      <td>3.955396</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.616365</td>
      <td>10.059654</td>
      <td>8.964775</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>0.294443</td>
      <td>4.833815</td>
      <td>3.955396</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.616365</td>
      <td>10.059654</td>
      <td>9.002801</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>0.294443</td>
      <td>4.833815</td>
      <td>3.955396</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.616365</td>
      <td>10.059654</td>
      <td>9.037999</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>0.294443</td>
      <td>4.833815</td>
      <td>3.955396</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df["treatment"].value_counts(normalize=True))
print(df["conversion"].value_counts(normalize=True))
```

    treatment
    1    0.85
    0    0.15
    Name: proportion, dtype: float64
    conversion
    0    0.997083
    1    0.002917
    Name: proportion, dtype: float64
    

## 2. Standardize treatment / outcome column names

The exact Criteo file format may vary depending on source or preprocessing.
This cell tries to map common naming conventions into:

- `treatment`
- `conversion`

If your dataset already uses these names, no change is needed.
If not, edit the mapping list manually.



```python
# Common alternative names seen across uplift datasets / user preprocessed copies.
treatment_candidates = ["treatment", "exposure", "t", "group", "visit"]
outcome_candidates = ["conversion", "outcome", "y", "label", "response"]

def first_existing(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

t_col = first_existing(treatment_candidates, df.columns)
y_col = first_existing(outcome_candidates, df.columns)

if t_col is None or y_col is None:
    raise ValueError(
        "Could not automatically identify treatment/outcome columns. "
        "Please set t_col and y_col manually."
    )

work = df.copy()
work = work.rename(columns={t_col: "treatment", y_col: "conversion"})

work["treatment"] = work["treatment"].astype(int)
work["conversion"] = work["conversion"].astype(int)

print("Treatment column used:", t_col)
print("Outcome column used:", y_col)
work[["treatment", "conversion"]].head()

```

    Treatment column used: treatment
    Outcome column used: conversion
    




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
      <th>treatment</th>
      <th>conversion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Basic data audit

Before comparing methods, I want a basic feel for the treatment and outcome balance.

This is not the final answer.
It is just a quick reality check.



```python
print("Rows:", len(work))
print("Treatment rate:", round(work["treatment"].mean(), 5))
print("Conversion rate:", round(work["conversion"].mean(), 5))

raw_summary = work.groupby("treatment")["conversion"].agg(["mean", "sum", "count"])
raw_summary["conversion_rate_pct"] = 100 * raw_summary["mean"]
raw_summary

```

    Rows: 13979592
    Treatment rate: 0.85
    Conversion rate: 0.00292
    




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
      <th>mean</th>
      <th>sum</th>
      <th>count</th>
      <th>conversion_rate_pct</th>
    </tr>
    <tr>
      <th>treatment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001938</td>
      <td>4063</td>
      <td>2096937</td>
      <td>0.193759</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.003089</td>
      <td>36711</td>
      <td>11882655</td>
      <td>0.308946</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**

This quick view tells me:
- whether treatment assignment is roughly balanced
- whether conversion is a rare event
- whether raw treatment-control differences are directionally sensible

But raw differences are still not enough for a serious personalization system.

I still need to know:
- how much uplift varies by user or segment
- how noisy those estimates are
- whether the model supports confident action


## 4. Split features from treatment and outcome

Everything except treatment and outcome is treated as pre-treatment covariate input.

This notebook assumes the remaining columns are all valid pre-treatment features.
If your local file contains IDs or post-treatment leakage variables, remove them here.



```python
#sampling to get a faster run by compromising on performance slightly
work=work.sample(150000,random_state=42).reset_index(drop=True)

exclude_cols = ["treatment", "conversion"]
feature_cols = [c for c in work.columns if c not in exclude_cols]

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

print("Numeric feature count:", len(num_cols))
print("Categorical feature count:", len(cat_cols))

X = work[feature_cols].copy()
T = work["treatment"].astype(int).values
Y = work["conversion"].astype(float).values
```

    Numeric feature count: 14
    Categorical feature count: 0
    

## 5. Common preprocessing pipeline

To compare methods fairly, I want both workflows to use the same cleaned feature matrix.

This is a practical pipeline:
- median imputation + scaling for numeric features
- mode imputation + one-hot encoding for categorical features



```python
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ]
)

X_proc = preprocess.fit_transform(X)
X_proc = np.asarray(X_proc.todense()) if hasattr(X_proc, "todense") else np.asarray(X_proc)

print("Processed feature matrix shape:", X_proc.shape)

```

    Processed feature matrix shape: (150000, 14)
    

## 6. Double Machine Learning baseline

Double Machine Learning is attractive because it separates the problem into:

1. Model the outcome using covariates
2. Model the treatment assignment using covariates
3. Residualize / orthogonalize
4. Estimate treatment effects on the debiased residual structure

That makes it a strong choice when:
- feature relationships are nonlinear
- confounding structure is rich
- the main goal is treatment-effect estimation rather than posterior uncertainty



```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

dml = LinearDML(
model_y=RandomForestRegressor( #econML treats outcome nuisance model as conditional expectation E[Y | X,T]] not a probability classification
n_estimators=100,
min_samples_leaf=100,
random_state=RANDOM_STATE,
n_jobs=-1,
),
model_t=RandomForestClassifier(
n_estimators=100,
min_samples_leaf=100,
random_state=RANDOM_STATE,
n_jobs=-1,
),
discrete_treatment=True, # <-- this is the key fix
cv=3,    
random_state=RANDOM_STATE,
)

dml.fit(Y, T, X=X_proc)

dml_ate = dml.ate(X_proc)
print("DML estimated ATE:", dml_ate)

```

    DML estimated ATE: 0.017506617518227914
    

**Interpretation**

The DML ATE gives me a useful first read on whether treatment is helping on average.

That is a good starting point, but it is still only a population-level answer.

For personalization, the real question is not just:

> Is the treatment helpful overall?

It is:

> Where is the uplift concentrated, and how much should I trust the ranking?

That is why the rest of the notebook moves from a single DML average effect to DML deciles and then to a Bayesian validation layer.


## 7. Individualized DML uplift estimates

To make DML more relevant to personalization, I compute user-level conditional treatment effect estimates.

These are not posterior distributions.
They are point estimates of individualized uplift.



```python
work["dml_cate"] = dml.effect(X_proc)

work["dml_decile"] = pd.qcut(
    work["dml_cate"].rank(method="first"),
    10,
    labels=False
)

work[["dml_cate", "dml_decile"]].head()

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
      <th>dml_cate</th>
      <th>dml_decile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.002424</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.000142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.000153</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.000073</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000033</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
dml_decile_summary = (
    work.groupby("dml_decile")
    .agg(
        mean_dml_cate=("dml_cate", "mean"),
        conversion=("conversion", "mean"),
        treatment_rate=("treatment", "mean"),
        n=("conversion", "size"),
    )
    .reset_index()
    .sort_values("mean_dml_cate", ascending=False)
)

dml_decile_summary

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
      <th>dml_decile</th>
      <th>mean_dml_cate</th>
      <th>conversion</th>
      <th>treatment_rate</th>
      <th>n</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0.177048</td>
      <td>0.022667</td>
      <td>0.894333</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.001615</td>
      <td>0.001067</td>
      <td>0.843400</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0.000763</td>
      <td>0.000867</td>
      <td>0.847067</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.000279</td>
      <td>0.000067</td>
      <td>0.843933</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-0.000015</td>
      <td>0.000133</td>
      <td>0.850000</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.000167</td>
      <td>0.000067</td>
      <td>0.842333</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.000296</td>
      <td>0.000000</td>
      <td>0.846267</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-0.000488</td>
      <td>0.000200</td>
      <td>0.847733</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.000702</td>
      <td>0.000267</td>
      <td>0.846067</td>
      <td>15000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.002971</td>
      <td>0.006000</td>
      <td>0.835333</td>
      <td>15000</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**

At this point DML has done the part it is best at: it has turned the raw feature space into an individualized uplift score and a practical ranking.

That ranking is useful operationally because it tells me who appears most promising according to the causal ML model.

But it still leaves an uncomfortable gap:

- the ranking is a point-estimate ranking
- it does not tell me which deciles are genuinely robust
- it does not tell me how much noise may be sitting inside the top or bottom buckets

That is exactly the gap the Bayesian layer is meant to address.


## 8. Build a Bayesian comparison dataset

For the Bayesian model, I intentionally do **not** try to fit a massive full-feature model over every processed covariate.

That would make the notebook heavier and distract from the actual comparison.

Instead, I use a practical compromise:
- DML already summarized heterogeneous uplift into user ranking
- I use DML uplift deciles as the grouping structure
- Bayesian hierarchy then asks whether those ranked segments still look good after shrinkage and uncertainty modeling

This makes the comparison easier to interpret:
- DML gives a ranking
- Bayesian tests and stabilizes that ranking



```python
bayes_df = work[["treatment", "conversion", "dml_decile"]].copy()
bayes_df["group"] = bayes_df["dml_decile"].astype(int)
bayes_df["group_idx"] = bayes_df["group"]
#ensuring only 60K max rows are input in to bayes model
MAX_N = 60000
if len(bayes_df) > MAX_N:
    bayes_df = bayes_df.sample(MAX_N, random_state=RANDOM_STATE).reset_index(drop=True)

group_names = sorted(bayes_df["group"].unique().tolist())
coords = {
    "obs_id": np.arange(len(bayes_df)),
    "group": group_names,
}

print(bayes_df.shape)
bayes_df.head()

```

    (60000, 5)
    




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
      <th>treatment</th>
      <th>conversion</th>
      <th>dml_decile</th>
      <th>group</th>
      <th>group_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 9. Prior predictive check


Before fitting the Bayesian model, I want to know:

> If my priors were true, would they generate plausible baseline conversion behavior?

If the priors imply absurd conversion behavior, the model is already in trouble before the data has had any chance to update it.



```python
with pm.Model(coords=coords) as prior_model:

    treatment = pm.Data("treatment", bayes_df["treatment"].values, dims="obs_id")
    group_idx = pm.Data("group_idx", bayes_df["group_idx"].values, dims="obs_id")

    alpha = pm.Normal("alpha", mu=-4.0, sigma=1.0)

    mu_t = pm.Normal("mu_t", mu=0.0, sigma=0.5)
    sigma_t = pm.HalfNormal("sigma_t", sigma=0.5)

    z_t = pm.Normal("z_t", mu=0.0, sigma=1.0, dims="group")
    beta_t_group = pm.Deterministic(
        "beta_t_group",
        mu_t + z_t * sigma_t,
        dims="group"
    )

    logit_p = alpha + beta_t_group[group_idx] * treatment
    p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims="obs_id")

    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=RANDOM_STATE)

```

    Sampling: [alpha, mu_t, sigma_t, z_t]
    


```python
prior_mean_conv = prior_pred.prior["p"].mean(dim="obs_id").values.flatten()

plt.figure(figsize=(8, 4))
plt.hist(prior_mean_conv, bins=40)
plt.title("Prior predictive distribution of average conversion")
plt.xlabel("Implied average conversion rate")
plt.ylabel("Count")
plt.show()

```

<figure>
  <img src="{{ site.baseurl }}/images/output_25_0_C.png">
  <figcaption style="text-align:center;">Fig1. Prior predictive distribution of average conversion. </figcaption>
</figure> 
    



**Interpretation**

This prior check is just making sure the Bayesian layer starts from a sane place.

Given how low conversion is in this dataset, I want the prior to imply low baseline conversion and moderate room for treatment heterogeneity — not wild conversion rates that only look good on paper.

So this step is less about excitement and more about discipline:

if the prior already implies unrealistic behavior, any later posterior story becomes much harder to trust.


## 10. Bayesian hierarchical uplift model

Now I fit the Bayesian model.

Key design choice:

- treatment effect varies by DML uplift decile
- those decile-level treatment effects are partially pooled through a shared hierarchy

This lets me test whether the DML ranking survives a more uncertainty-aware, shrinkage-based framework.



```python
with pm.Model(coords=coords) as bayes_model:

    treatment = pm.Data("treatment", bayes_df["treatment"].values, dims="obs_id")
    group_idx = pm.Data("group_idx", bayes_df["group_idx"].values, dims="obs_id")
    y_obs = pm.Data("y_obs", bayes_df["conversion"].values, dims="obs_id")

    alpha = pm.Normal("alpha", mu=-4.0, sigma=1.0)

    mu_t = pm.Normal("mu_t", mu=0.0, sigma=0.5)
    sigma_t = pm.HalfNormal("sigma_t", sigma=0.5)

    z_t = pm.Normal("z_t", mu=0.0, sigma=1.0, dims="group")
    beta_t_group = pm.Deterministic(
        "beta_t_group",
        mu_t + z_t * sigma_t,
        dims="group"
    )

    logit_p = alpha + beta_t_group[group_idx] * treatment
    p = pm.Deterministic("p", pm.math.sigmoid(logit_p), dims="obs_id")

    outcome = pm.Bernoulli("outcome", p=p, observed=y_obs, dims="obs_id")

    idata = pm.sample(
        draws=1000,
        tune=1500,
        chains=4,
        target_accept=0.95,
        random_seed=RANDOM_STATE,
        return_inferencedata=True,
    )

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [alpha, mu_t, sigma_t, z_t]
    


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 1_500 tune and 1_000 draw iterations (6_000 + 4_000 draws total) took 2400 seconds.
    

## 11. Bayesian diagnostic validation

Before interpreting anything, I want to confirm that the posterior fit is trustworthy.



```python
diag_summary = az.summary(idata, var_names=["alpha", "mu_t", "sigma_t", "beta_t_group"], round_to=3)
diag_summary

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
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha</th>
      <td>-6.544</td>
      <td>0.257</td>
      <td>-7.028</td>
      <td>-6.065</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>3334.553</td>
      <td>2405.224</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>mu_t</th>
      <td>-0.480</td>
      <td>0.375</td>
      <td>-1.155</td>
      <td>0.235</td>
      <td>0.008</td>
      <td>0.005</td>
      <td>2110.148</td>
      <td>2628.621</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>sigma_t</th>
      <td>1.451</td>
      <td>0.242</td>
      <td>1.046</td>
      <td>1.925</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1816.531</td>
      <td>2525.441</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>beta_t_group[0]</th>
      <td>1.460</td>
      <td>0.310</td>
      <td>0.904</td>
      <td>2.059</td>
      <td>0.005</td>
      <td>0.005</td>
      <td>3797.359</td>
      <td>2902.477</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>beta_t_group[1]</th>
      <td>-1.747</td>
      <td>0.781</td>
      <td>-3.270</td>
      <td>-0.394</td>
      <td>0.011</td>
      <td>0.012</td>
      <td>4832.212</td>
      <td>3234.974</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>beta_t_group[2]</th>
      <td>-1.732</td>
      <td>0.771</td>
      <td>-3.119</td>
      <td>-0.260</td>
      <td>0.013</td>
      <td>0.013</td>
      <td>3986.051</td>
      <td>2344.377</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>beta_t_group[3]</th>
      <td>-2.450</td>
      <td>1.001</td>
      <td>-4.282</td>
      <td>-0.604</td>
      <td>0.017</td>
      <td>0.018</td>
      <td>4025.831</td>
      <td>2886.824</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>beta_t_group[4]</th>
      <td>-2.422</td>
      <td>0.992</td>
      <td>-4.357</td>
      <td>-0.765</td>
      <td>0.017</td>
      <td>0.017</td>
      <td>3571.068</td>
      <td>3002.445</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>beta_t_group[5]</th>
      <td>-1.303</td>
      <td>0.675</td>
      <td>-2.608</td>
      <td>-0.084</td>
      <td>0.011</td>
      <td>0.011</td>
      <td>3677.438</td>
      <td>2901.654</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>beta_t_group[6]</th>
      <td>-2.412</td>
      <td>0.970</td>
      <td>-4.351</td>
      <td>-0.759</td>
      <td>0.015</td>
      <td>0.017</td>
      <td>4477.024</td>
      <td>2565.410</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>beta_t_group[7]</th>
      <td>-0.480</td>
      <td>0.507</td>
      <td>-1.454</td>
      <td>0.479</td>
      <td>0.008</td>
      <td>0.007</td>
      <td>4605.365</td>
      <td>3588.117</td>
      <td>1.001</td>
    </tr>
    <tr>
      <th>beta_t_group[8]</th>
      <td>-0.951</td>
      <td>0.615</td>
      <td>-2.129</td>
      <td>0.124</td>
      <td>0.009</td>
      <td>0.010</td>
      <td>4611.971</td>
      <td>3007.763</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>beta_t_group[9]</th>
      <td>2.878</td>
      <td>0.270</td>
      <td>2.368</td>
      <td>3.370</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>3523.564</td>
      <td>2858.652</td>
      <td>1.001</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**

Before trusting any Bayesian segment result, I need the diagnostics to look clean.

What I want here is straightforward:

- no divergences
- R-hat essentially at 1
- effective sample size comfortably large

Since those conditions hold, I can treat the posterior segment summaries as real signal rather than sampling noise created by a poorly behaving chain.


## 12. Posterior predictive check

The next step is not just to inspect coefficients.
I also want to know whether the fitted model can reproduce aggregate conversion behavior in the observed data.



```python
with bayes_model:
    ppc = pm.sample_posterior_predictive(idata, var_names=["outcome", "p"], random_seed=RANDOM_STATE)

idata.extend(ppc)

```

    Sampling: [outcome]
    


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




```python
pp_y = idata.posterior_predictive["outcome"].values
pp_rate = pp_y.mean(axis=(0, 2))
observed_rate = bayes_df["conversion"].mean()

plt.figure(figsize=(8, 4))
plt.hist(pp_rate, bins=40)
plt.axvline(observed_rate, linestyle="--")
plt.title("Posterior predictive check: average conversion rate")
plt.xlabel("Posterior predictive conversion rate")
plt.ylabel("Count")
plt.show()

print("Observed average conversion:", round(observed_rate, 6))

```

<figure>
  <img src="{{ site.baseurl }}/images/output_34_0_C.png">
  <figcaption style="text-align:center;">Fig2. Posterior predictive check: average conversion rate. </figcaption>
</figure> 
    

    Observed average conversion: 0.003217
    

**Interpretation**

This posterior predictive check is deliberately modest.

It is not claiming the Bayesian layer reproduces every aspect of the Criteo data. It is checking something simpler and still important:

> Can the fitted model reproduce the overall conversion level in the sample it was trained on?

If the observed rate sits inside the posterior predictive distribution, that is a good aggregate-level calibration sign. It does not prove the model is perfect, but it does tell me the Bayesian layer is not obviously disconnected from the basic conversion behavior in the data.


## 13. Convert Bayesian treatment effects into business-space uplift

The Bayesian model estimates treatment effects on the **log-odds** scale.

That is fine statistically, but not what a business audience wants.

For decisioning, I want:
- predicted conversion under control
- predicted conversion under treatment
- expected uplift in conversion probability

coefficient scale is model-space;
uplift is business-space.



```python
posterior = idata.posterior

alpha_s = posterior["alpha"].values.reshape(-1)
beta_t_group_s = posterior["beta_t_group"].stack(sample=("chain", "draw")).values

rows = []

segment_profiles = (
    bayes_df.groupby("group")
    .agg(
        n=("conversion", "size"),
        observed_conversion=("conversion", "mean"),
        treated_rate=("treatment", "mean"),
    )
    .reset_index()
)

for g_idx, row in segment_profiles.iterrows():
    eta_control = alpha_s
    eta_treat = alpha_s + beta_t_group_s[g_idx, :]

    p_control = 1 / (1 + np.exp(-eta_control))
    p_treat = 1 / (1 + np.exp(-eta_treat))
    uplift = p_treat - p_control

    rows.append({
        "group": row["group"],
        "n": row["n"],
        "observed_conversion": row["observed_conversion"],
        "post_control_mean": p_control.mean(),
        "post_treat_mean": p_treat.mean(),
        "post_uplift_mean": uplift.mean(),
        "post_uplift_p10": np.quantile(uplift, 0.10),
        "post_uplift_p90": np.quantile(uplift, 0.90),
        "prob_uplift_positive": (uplift > 0).mean(),
    })

uplift_df = pd.DataFrame(rows).sort_values("post_uplift_mean", ascending=False)
uplift_df

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
      <th>group</th>
      <th>n</th>
      <th>observed_conversion</th>
      <th>post_control_mean</th>
      <th>post_treat_mean</th>
      <th>post_uplift_mean</th>
      <th>post_uplift_p10</th>
      <th>post_uplift_p90</th>
      <th>prob_uplift_positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>6040.0</td>
      <td>0.022682</td>
      <td>0.001484</td>
      <td>0.025020</td>
      <td>0.023536</td>
      <td>0.020897</td>
      <td>0.026238</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>5917.0</td>
      <td>0.007267</td>
      <td>0.001484</td>
      <td>0.006257</td>
      <td>0.004773</td>
      <td>0.003332</td>
      <td>0.006328</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>6019.0</td>
      <td>0.000997</td>
      <td>0.001484</td>
      <td>0.000977</td>
      <td>-0.000507</td>
      <td>-0.001178</td>
      <td>0.000199</td>
      <td>0.17075</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>5994.0</td>
      <td>0.000501</td>
      <td>0.001484</td>
      <td>0.000642</td>
      <td>-0.000841</td>
      <td>-0.001477</td>
      <td>-0.000219</td>
      <td>0.04575</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>6072.0</td>
      <td>0.000329</td>
      <td>0.001484</td>
      <td>0.000469</td>
      <td>-0.001015</td>
      <td>-0.001586</td>
      <td>-0.000446</td>
      <td>0.01600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>5957.0</td>
      <td>0.000168</td>
      <td>0.001484</td>
      <td>0.000323</td>
      <td>-0.001160</td>
      <td>-0.001711</td>
      <td>-0.000623</td>
      <td>0.00375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>6054.0</td>
      <td>0.000165</td>
      <td>0.001484</td>
      <td>0.000323</td>
      <td>-0.001161</td>
      <td>-0.001723</td>
      <td>-0.000643</td>
      <td>0.00550</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>5910.0</td>
      <td>0.000000</td>
      <td>0.001484</td>
      <td>0.000190</td>
      <td>-0.001294</td>
      <td>-0.001827</td>
      <td>-0.000804</td>
      <td>0.00150</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>6059.0</td>
      <td>0.000000</td>
      <td>0.001484</td>
      <td>0.000189</td>
      <td>-0.001295</td>
      <td>-0.001827</td>
      <td>-0.000800</td>
      <td>0.00150</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>5978.0</td>
      <td>0.000000</td>
      <td>0.001484</td>
      <td>0.000185</td>
      <td>-0.001299</td>
      <td>-0.001825</td>
      <td>-0.000812</td>
      <td>0.00125</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**

This table is where the Bayesian results become decision-friendly.

A few things stand out immediately from these outputs:

- **Decile 9** is the clearest win. Its posterior expected uplift is about **0.0235**, with a **100% posterior probability** that uplift is positive. That is a very strong confirmation that the top DML bucket contains real treatment signal.
- **Decile 0** is the big surprise. DML gave it a strongly negative mean CATE, but the Bayesian layer estimates a **positive expected uplift of about 0.0048** with **100% probability uplift is positive**.
- Most of the remaining deciles are not just uncertain — they are actually **negative on posterior mean uplift**, and their posterior probability of positive uplift is extremely low.
- **Decile 7** is the only mild gray area: it is close to zero and still has only about **17% probability** of positive uplift, which is far too weak to act on confidently.

So the Bayesian layer is **not** broadly validating the full DML ranking. It is saying something more specific:

> the very top DML decile looks real, the very bottom decile deserves a second look, and most of the middle ranking is not convincing once uncertainty is taken seriously.

One technical note matters here: because this Bayesian layer uses treatment and DML decile only — not the full raw feature set — the `post_control_mean` is common across deciles by construction. So the main objects to interpret are `post_uplift_mean` and `prob_uplift_positive`, not the identical control means.


## 14. Visualize Bayesian expected uplift by segment

This plot is one of the clearest decisioning views in the notebook.

It shows:
- point estimate of expected uplift
- posterior uncertainty interval
- which DML-defined deciles still look good after Bayesian shrinkage



```python
plot_df = uplift_df.sort_values("post_uplift_mean", ascending=True).copy()

plt.figure(figsize=(10, 6))
plt.errorbar(
    x=plot_df["post_uplift_mean"],
    y=plot_df["group"].astype(str),
    xerr=[
        plot_df["post_uplift_mean"] - plot_df["post_uplift_p10"],
        plot_df["post_uplift_p90"] - plot_df["post_uplift_mean"]
    ],
    fmt="o"
)

for i, row in plot_df.reset_index(drop=True).iterrows():
    plt.text(
        row["post_uplift_mean"] + 0.00002,
        i,
        f"{row['post_uplift_mean']:.4f}",
        va="center"
    )

plt.axvline(0, linestyle="--")
plt.title("Bayesian posterior expected uplift by DML decile")
plt.xlabel("Expected uplift in conversion probability")
plt.ylabel("DML uplift decile")
plt.show()

```

<figure>
  <img src="{{ site.baseurl }}/images/output_40_0_C.png">
  <figcaption style="text-align:center;">Fig3. Bayesian posterior expected uplift by DML decile. </figcaption>
</figure> 

## 15. Visualize Bayesian confidence that uplift is positive

This plot answers a different question.

The previous uplift plot was about **magnitude**.

This plot is about **confidence**:

> How likely is it that treatment helps at all in this decile?



```python
plot_df = uplift_df.sort_values("prob_uplift_positive", ascending=True).copy()

plt.figure(figsize=(10, 6))
plt.barh(plot_df["group"].astype(str), plot_df["prob_uplift_positive"])
plt.axvline(0.8, linestyle="--")
plt.xlim(max(0.0, plot_df["prob_uplift_positive"].min() - 0.02), 1.0)

for i, v in enumerate(plot_df["prob_uplift_positive"]):
    plt.text(v + 0.002, i, f"{v:.3f}", va="center")

plt.title("Bayesian posterior probability uplift is positive")
plt.xlabel("P(uplift > 0)")
plt.ylabel("DML uplift decile")
plt.show()

```

<figure>
  <img src="{{ site.baseurl }}/images/output_42_0_C.png">
  <figcaption style="text-align:center;">Fig4. Bayesian posterior probability uplift is positive. </figcaption>
</figure> 
    

**Interpretation**

These two Bayesian plots make the story much clearer than the raw DML ranking alone.

The uplift plot shows that only **decile 9** is far to the right with a clearly positive interval. **Decile 0** is also positive, but much smaller. Everything else is clustered near zero or below it.

The probability plot sharpens that further:

- **Decile 9:** posterior probability uplift is positive = **1.000**
- **Decile 0:** posterior probability uplift is positive = **1.000**
- **Decile 7:** only about **0.171**
- all remaining deciles: effectively near zero confidence

That means the Bayesian layer is doing exactly what I wanted it to do. It is forcing me to separate:

- segments that look strong enough to act on
- segments that only looked interesting as point estimates
- segments that likely contain little real treatment signal

Put differently, DML gave me a broad ranking; Bayesian turned that into a much narrower set of segments I would actually be comfortable discussing in a deployment conversation.


## 16. Compare DML ranking and Bayesian decisioning

This is the key practical section.

DML gives:
- a personalized uplift ranking

Bayesian gives:
- stabilized expected uplift
- uncertainty intervals
- probability uplift is positive

That means the two methods are not really competing on the exact same output.
They are serving different parts of the decision workflow.



```python
comparison_df = (
    dml_decile_summary[["dml_decile", "mean_dml_cate", "n"]]
    .rename(columns={"dml_decile": "group"})
    .merge(
        uplift_df[["group", "post_uplift_mean", "prob_uplift_positive", "post_control_mean", "post_treat_mean"]],
        on="group",
        how="left",
    )
    .sort_values("mean_dml_cate", ascending=False)
)

comparison_df

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
      <th>group</th>
      <th>mean_dml_cate</th>
      <th>n</th>
      <th>post_uplift_mean</th>
      <th>prob_uplift_positive</th>
      <th>post_control_mean</th>
      <th>post_treat_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>0.177048</td>
      <td>15000</td>
      <td>0.023536</td>
      <td>1.00000</td>
      <td>0.001484</td>
      <td>0.025020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>0.001615</td>
      <td>15000</td>
      <td>-0.000841</td>
      <td>0.04575</td>
      <td>0.001484</td>
      <td>0.000642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0.000763</td>
      <td>15000</td>
      <td>-0.000507</td>
      <td>0.17075</td>
      <td>0.001484</td>
      <td>0.000977</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>0.000279</td>
      <td>15000</td>
      <td>-0.001294</td>
      <td>0.00150</td>
      <td>0.001484</td>
      <td>0.000190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-0.000015</td>
      <td>15000</td>
      <td>-0.001015</td>
      <td>0.01600</td>
      <td>0.001484</td>
      <td>0.000469</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>-0.000167</td>
      <td>15000</td>
      <td>-0.001295</td>
      <td>0.00150</td>
      <td>0.001484</td>
      <td>0.000189</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>-0.000296</td>
      <td>15000</td>
      <td>-0.001299</td>
      <td>0.00125</td>
      <td>0.001484</td>
      <td>0.000185</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-0.000488</td>
      <td>15000</td>
      <td>-0.001160</td>
      <td>0.00375</td>
      <td>0.001484</td>
      <td>0.000323</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-0.000702</td>
      <td>15000</td>
      <td>-0.001161</td>
      <td>0.00550</td>
      <td>0.001484</td>
      <td>0.000323</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>-0.002971</td>
      <td>15000</td>
      <td>0.004773</td>
      <td>1.00000</td>
      <td>0.001484</td>
      <td>0.006257</td>
    </tr>
  </tbody>
</table>
</div>



**Interpretation**

This comparison table answers the main practical question of the notebook:

> Does the DML ranking hold up once I impose shrinkage and uncertainty-aware validation?

The answer here is: **only partially**.

What is clearly validated:
- **Decile 9** remains the strongest segment by a wide margin on both DML ranking and Bayesian uplift. This is the cleanest case where the two methods agree.

What is not validated:
- **Decile 8** ranked second by DML, but the Bayesian layer gives it **negative posterior mean uplift** and only about **4.6% probability** of positive uplift.
- **Decile 7** also ranked highly in DML, but Bayesian confidence is still weak at roughly **17%**.
- Several lower and middle deciles look actively unattractive after Bayesian validation.

What is unexpected:
- **Decile 0** flips the story. DML treats it as the worst bucket, but the Bayesian layer sees a meaningfully positive uplift with very high confidence in the sampled data.

That is exactly the kind of result that justifies this hybrid workflow. If I had stopped at DML, I would likely have treated the full ranking as trustworthy. The Bayesian layer shows that this would have been too optimistic.

The practical takeaway is not that DML is wrong. It is that:

> DML ranking by itself is not enough to decide deployment confidence.


## 17. Optional decision rule

The Bayesian results naturally lead to a three-bucket decision framework.

### 1. Target confidently
Use this when:
- expected uplift is clearly positive
- the posterior interval stays on the positive side (p10>0 & p90>0)
- `P(uplift > 0)` is very high

In these results, **decile 9** clearly belongs here.  
Both DML and Bayesian modeling agree that it is the strongest segment, and the Bayesian layer removes any real ambiguity about whether the uplift is positive.

### 2. Investigate before acting
Use this when:
- the Bayesian result disagrees materially with the DML ranking
- the segment could matter, but the disagreement is too large to ignore

In these results, **decile 0** belongs here.

DML ranked decile 0 as the worst segment, but the Bayesian layer estimates positive uplift with very high confidence. That is not the kind of disagreement I would brush aside.

I would not immediately deploy against decile 0 at scale, but I also would not discard it. Instead, I would treat it as a high-priority diagnostic segment:
- inspect raw treatment vs control conversion inside the bucket
- check whether the segment contains substructure that DML compressed too aggressively
- validate with a focused follow-up test if the business stakes justify it

### 3. Deprioritize
Use this when:
- posterior mean uplift is near zero or negative
- posterior probability of positive uplift is weak

That is what I would do with most of the remaining deciles in this notebook. Even if some looked interesting in DML, the Bayesian layer does not give me enough confidence to act on them aggressively.

This is where the hybrid workflow becomes useful:

- DML helps surface candidate segments
- Bayesian modeling helps decide which of those candidates deserve real attention



```python
decision_df = comparison_df.copy()

decision_df["recommended_action"] = np.select(
    [
        (decision_df["prob_uplift_positive"] >= 0.90) & (decision_df["post_uplift_mean"] > 0),
        (decision_df["prob_uplift_positive"] >= 0.70) & (decision_df["post_uplift_mean"] > 0),
    ],
    [
        "Target confidently",
        "Promising but monitor / test",
    ],
    default="Weak evidence / low priority",
)

decision_df.sort_values(["prob_uplift_positive", "post_uplift_mean"], ascending=False)

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
      <th>group</th>
      <th>mean_dml_cate</th>
      <th>n</th>
      <th>post_uplift_mean</th>
      <th>prob_uplift_positive</th>
      <th>post_control_mean</th>
      <th>post_treat_mean</th>
      <th>recommended_action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>0.177048</td>
      <td>15000</td>
      <td>0.023536</td>
      <td>1.00000</td>
      <td>0.001484</td>
      <td>0.025020</td>
      <td>Target confidently</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>-0.002971</td>
      <td>15000</td>
      <td>0.004773</td>
      <td>1.00000</td>
      <td>0.001484</td>
      <td>0.006257</td>
      <td>Target confidently</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0.000763</td>
      <td>15000</td>
      <td>-0.000507</td>
      <td>0.17075</td>
      <td>0.001484</td>
      <td>0.000977</td>
      <td>Weak evidence / low priority</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>0.001615</td>
      <td>15000</td>
      <td>-0.000841</td>
      <td>0.04575</td>
      <td>0.001484</td>
      <td>0.000642</td>
      <td>Weak evidence / low priority</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-0.000015</td>
      <td>15000</td>
      <td>-0.001015</td>
      <td>0.01600</td>
      <td>0.001484</td>
      <td>0.000469</td>
      <td>Weak evidence / low priority</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-0.000702</td>
      <td>15000</td>
      <td>-0.001161</td>
      <td>0.00550</td>
      <td>0.001484</td>
      <td>0.000323</td>
      <td>Weak evidence / low priority</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-0.000488</td>
      <td>15000</td>
      <td>-0.001160</td>
      <td>0.00375</td>
      <td>0.001484</td>
      <td>0.000323</td>
      <td>Weak evidence / low priority</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>0.000279</td>
      <td>15000</td>
      <td>-0.001294</td>
      <td>0.00150</td>
      <td>0.001484</td>
      <td>0.000190</td>
      <td>Weak evidence / low priority</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>-0.000167</td>
      <td>15000</td>
      <td>-0.001295</td>
      <td>0.00150</td>
      <td>0.001484</td>
      <td>0.000189</td>
      <td>Weak evidence / low priority</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>-0.000296</td>
      <td>15000</td>
      <td>-0.001299</td>
      <td>0.00125</td>
      <td>0.001484</td>
      <td>0.000185</td>
      <td>Weak evidence / low priority</td>
    </tr>
  </tbody>
</table>
</div>



## 18. Final practical comparison

### What DML did well in this notebook
DML did the heavy lifting in the original feature space. It turned the raw Criteo covariates into an individualized uplift score and a first-pass ranking that is actually usable.

That matters. Without DML, I would not have a practical way to sort a large feature space into candidate treatment segments this efficiently.

### What the Bayesian layer added
The Bayesian layer answered the question DML alone could not answer:

> Which parts of that ranking still look credible once uncertainty and partial pooling are taken seriously?

That extra layer changed the story in a meaningful way.

- It strongly confirmed **decile 9**
- It rejected most of the rest of the ranking as weak, negative, or too uncertain
- It surfaced a major disagreement in **decile 0**, where DML and Bayesian modeling point in opposite directions

That last point is especially important. If I had stopped at the DML ranking, I would likely have treated decile 0 as safely ignorable. The Bayesian layer says that would have been too casual.

### How I would use the two methods together
If I were operationalizing this workflow, I would use the methods in sequence rather than forcing them into a winner-take-all comparison.

- **DML** is the discovery engine: it searches the rich covariate space and produces candidate uplift structure
- **Bayesian modeling** is the credibility filter: it tells me which parts of that structure survive uncertainty-aware validation

In this notebook, that leads to a simple action map:

- **Decile 9** → target confidently
- **Decile 0** → investigate because the methods disagree
- **Most other deciles** → deprioritize


## 19. Final takeaways

This analysis leads to a more useful conclusion than a simple “Bayesian vs DML” comparison.

The two approaches address different parts of the same decision problem.

### Interpreting the results

- **Decile 9** is the clearest success case: DML ranked it highest, and the Bayesian layer confirms strong positive uplift with high confidence.  
- The rest of the DML ranking becomes less compelling once uncertainty is introduced.  
- **Decile 8 and decile 7** appear attractive under DML, but Bayesian validation does not support confident action.  
- **Decile 0** is the most interesting segment: DML ranked it worst, while Bayesian modeling estimates positive uplift with near certainty in the sampled data.  

This shows that the Bayesian layer is not just “adding uncertainty” in the abstract—it materially changes how decisions should be made.

Rather than treating the DML ranking as a final answer, this leads to a more actionable segmentation:

1. **Deploy with confidence** → decile 9  
2. **Investigate disagreements** → decile 0  
3. **Deprioritize for now** → remaining deciles  

---

## Practical conclusion

The raw DML ranking should not be deployed as-is.

A more robust approach is:

- **DML** to generate candidate uplift segments  
- **Bayesian validation** to determine which segments are credible, which are weak, and which require deeper investigation  

The key takeaway is:

> Point estimates are useful for discovery, but uncertainty is what makes a ranking deployable.
