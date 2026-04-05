# From Uplift Modeling to Counterfactual Explanations on a Public Experimental Dataset

This notebook recreates the **main ideas and code patterns from Chapter 10 of *Causal Inference and Machine Learning* by Aleksander Molak**, but on a **different public dataset**.

Instead of the Hillstrom email dataset used in the book, we will use the **LaLonde job-training experiment**, a classic randomized dataset that is publicly available. The goal is the same:

- verify randomization,
- fit several CATE / uplift estimators,
- compare compute cost,
- evaluate models with uplift-by-decile and expected response,
- extract confidence intervals,
- finish with a small section on **counterfactual explanations**.

## Why this notebook is slightly adapted
The original chapter uses:
- a **multi-treatment** marketing dataset,
- and some code patterns tailored to that setup.

The LaLonde data is:
- a **binary treatment** experiment (`treat` vs `control`),
- with a **continuous outcome** (`re78`, earnings in 1978).

So the notebook mirrors the chapter's workflow, but simplifies a few formulas to the binary-treatment case.

> **Important:** the final counterfactual section explains **model behavior**, not ground-truth causality. That is also the key message in Molak's chapter.


```python

# If you are running this notebook in a fresh environment, uncomment the next line.
!pip install -q econml dowhy dice-ml lightgbm rdatasets
```


```python

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier

# EconML estimators
from econml.metalearners import SLearner, TLearner, XLearner
from econml.dr import DRLearner
from econml.dml import LinearDML, CausalForestDML

np.random.seed(42)
pd.set_option("display.max_columns", 200)
```

## 1. Load a different public dataset: LaLonde

The LaLonde dataset is a randomized job-training study and is widely used in causal inference.
We will download a public CSV mirror from the `Rdatasets` repository.


```python

url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv"
df = pd.read_csv(url)

# Keep only the useful columns
df = df.drop(columns=["rownames"], errors="ignore")

# Treatment and outcome naming consistent with the chapter
df = df.rename(columns={"treat": "treatment", "re78": "outcome"})

df.head()
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
      <th>treatment</th>
      <th>age</th>
      <th>educ</th>
      <th>race</th>
      <th>married</th>
      <th>nodegree</th>
      <th>re74</th>
      <th>re75</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>37</td>
      <td>11</td>
      <td>black</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9930.0460</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>9</td>
      <td>hispan</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3595.8940</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>30</td>
      <td>12</td>
      <td>black</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24909.4500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>27</td>
      <td>11</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7506.1460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33</td>
      <td>8</td>
      <td>black</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>289.7899</td>
    </tr>
  </tbody>
</table>
</div>



### Data dictionary (quick version)

- `treatment`: whether the person received job training
- `outcome`: earnings in 1978
- `age`, `educ`: age and education
- `black`, `hispan`, `married`, `nodegree`: demographic indicators
- `re74`, `re75`: earnings in prior years

The experiment is randomized, which means treatment should **not** be predictable from covariates if randomization worked well.


```python

df.describe(include="all").T
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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>treatment</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.301303</td>
      <td>0.459198</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>age</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27.363192</td>
      <td>9.881187</td>
      <td>16.0</td>
      <td>20.0</td>
      <td>25.0</td>
      <td>32.0</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>educ</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.26873</td>
      <td>2.628325</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>race</th>
      <td>614</td>
      <td>3</td>
      <td>white</td>
      <td>299</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>married</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.415309</td>
      <td>0.493177</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>nodegree</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.630293</td>
      <td>0.483119</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>re74</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4557.546569</td>
      <td>6477.964479</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1042.33</td>
      <td>7888.49825</td>
      <td>35040.07</td>
    </tr>
    <tr>
      <th>re75</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2184.938207</td>
      <td>3295.679043</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>601.5484</td>
      <td>3248.9875</td>
      <td>25142.24</td>
    </tr>
    <tr>
      <th>outcome</th>
      <td>614.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6792.834483</td>
      <td>7470.730792</td>
      <td>0.0</td>
      <td>238.283425</td>
      <td>4759.0185</td>
      <td>10893.5925</td>
      <td>60307.93</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Build feature, treatment, and outcome matrices


```python

X = df.drop(columns=["treatment", "outcome"])
X = pd.get_dummies(X,drop_first=True)
T = df["treatment"].astype(int)
Y = df["outcome"].astype(float)

print("Rows:", len(df))
print("Treatment rate:", T.mean().round(4))
print("Average outcome:", Y.mean().round(2))
X.head()
```

    Rows: 614
    Treatment rate: 0.3013
    Average outcome: 6792.83
    




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
      <th>age</th>
      <th>educ</th>
      <th>married</th>
      <th>nodegree</th>
      <th>re74</th>
      <th>re75</th>
      <th>race_hispan</th>
      <th>race_white</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Randomization sanity check

Just like in the chapter, we first ask:

> Can observed covariates predict treatment?

If treatment assignment is really random, a model should not do much better than naive guessing.


```python

# Check marginal treatment distribution
treatment_dist = T.value_counts(normalize=True).sort_index()
treatment_dist
```




    treatment
    0    0.698697
    1    0.301303
    Name: proportion, dtype: float64




```python

X_train_eda, X_test_eda, T_train_eda, T_test_eda = train_test_split(
    X, T, test_size=0.5, random_state=42, stratify=T
)

clf_eda = LGBMClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    verbosity=-1,
    random_state=42
)
clf_eda.fit(X_train_eda, T_train_eda)

T_pred_eda = clf_eda.predict(X_test_eda)
eda_accuracy = accuracy_score(T_test_eda, T_pred_eda)
eda_accuracy
```




$\displaystyle 0.820846905537459$



For a binary treatment, the naive benchmark is roughly the larger class probability.
Now we simulate what a **random classifier** would achieve if it only respected the treatment marginal.


```python

p1 = T.mean()
n_test = len(T_test_eda)

random_scores = []
for _ in range(10000):
    random_pred = np.random.binomial(1, p1, size=n_test)
    random_scores.append((random_pred == T_test_eda.to_numpy()).mean())

ci_low, ci_high = np.quantile(random_scores, [0.025, 0.975])

print("Observed treatment-prediction accuracy :", round(eda_accuracy, 4))
print("95% empirical random-accuracy interval :", (round(ci_low, 4), round(ci_high, 4)))
```

    Observed treatment-prediction accuracy : 0.8208
    95% empirical random-accuracy interval : (np.float64(0.5277), np.float64(0.6319))
    


```python

plt.figure(figsize=(8, 4.5))
plt.hist(random_scores, bins=40, alpha=0.8)
plt.axvline(eda_accuracy, linestyle="--", linewidth=2, label="Model accuracy")
plt.title("Randomization check: model accuracy vs random baseline")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.legend()
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_14_0_2.png">
  <figcaption style="text-align:center;">Fig1.Randomization check: model accuracy vs random baseline. </figcaption>
</figure>
    
    


If the model accuracy falls near the random baseline, that is good news:
the observed covariates do not strongly predict treatment, which is what we hope to see in a randomized experiment.

## 4. Train / test split for uplift modeling

Following the chapter, we create a separate train and test split.
Because experimental datasets can still be small in terms of **effective signal**, it is useful to keep a large test set.


```python

X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
    X, Y, T, test_size=0.5, random_state=42, stratify=T
)

print("Train rows:", len(X_train))
print("Test rows :", len(X_test))
print("Treatment rate train:", T_train.mean().round(4))
print("Treatment rate test :", T_test.mean().round(4))
```

    Train rows: 307
    Test rows : 307
    Treatment rate train: 0.3029
    Treatment rate test : 0.2997
    

## 5. Helper functions and model definitions

We now recreate the same family of estimators used in the chapter:

- S-Learner
- T-Learner
- X-Learner
- DR-Learner
- Linear DML
- Causal Forest DML

For consistency, we use LightGBM as the main base learner, just like the book often does.


```python

def create_regressor():
    return LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=-1,
        random_state=42
    )

def create_classifier():
    return LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=-1,
        random_state=42
    )

s_learner = SLearner(overall_model=create_regressor())

t_learner = TLearner(models=[create_regressor(), create_regressor()])

x_learner = XLearner(
    models=[create_regressor(), create_regressor()],
    cate_models=[create_regressor(), create_regressor()],
)

dr_learner = DRLearner(
    model_propensity=LogisticRegression(max_iter=2000),
    model_regression=create_regressor(),
    model_final=create_regressor(),
    cv=5,
)

linear_dml = LinearDML(
    model_y=create_regressor(),
    model_t=create_classifier(),
    discrete_treatment=True,
    cv=5,
    random_state=42,
)

causal_forest = CausalForestDML(
    model_y=create_regressor(),
    model_t=create_classifier(),
    discrete_treatment=True,
    cv=5,
    random_state=42,
)

models = {
    "SLearner": s_learner,
    "TLearner": t_learner,
    "XLearner": x_learner,
    "DRLearner": dr_learner,
    "LinearDML": linear_dml,
    "CausalForestDML": causal_forest,
}
```

## 6. Fit all models and compare training time

This mirrors the timing comparison in the chapter. Exact times will vary by machine, but relative ordering is the main idea.


```python

fit_times = {}

for model_name, model in models.items():
    start = time.time()
    if model_name in {"LinearDML", "CausalForestDML"}:
        model.fit(Y=y_train, T=T_train, X=X_train)
    else:
        model.fit(Y=y_train, T=T_train, X=X_train)
    stop = time.time()
    fit_times[model_name] = stop - start
    print(f"{model_name:<16} fitted in {fit_times[model_name]:.3f} seconds")
```

    SLearner         fitted in 0.052 seconds
    TLearner         fitted in 0.088 seconds
    XLearner         fitted in 0.129 seconds
    DRLearner        fitted in 2.457 seconds
    LinearDML        fitted in 0.371 seconds
    CausalForestDML  fitted in 0.583 seconds
    


```python

timing_df = pd.DataFrame({
    "Model": list(fit_times.keys()),
    "TimeSeconds": list(fit_times.values())
}).sort_values("TimeSeconds")

baseline = timing_df["TimeSeconds"].min()
timing_df["RelativeToFastest"] = (timing_df["TimeSeconds"] / baseline).round(1)

timing_df
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
      <th>Model</th>
      <th>TimeSeconds</th>
      <th>RelativeToFastest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SLearner</td>
      <td>0.051677</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TLearner</td>
      <td>0.087885</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XLearner</td>
      <td>0.129461</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LinearDML</td>
      <td>0.370959</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CausalForestDML</td>
      <td>0.582778</td>
      <td>11.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DRLearner</td>
      <td>2.456517</td>
      <td>47.5</td>
    </tr>
  </tbody>
</table>
</div>




```python

plt.figure(figsize=(9, 4.5))
plt.bar(timing_df["Model"], timing_df["RelativeToFastest"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Relative training time")
plt.title("Relative compute cost across CATE estimators")
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_23_0_2.png">
  <figcaption style="text-align:center;">Fig2.Relative compute cost across CATE estimators. </figcaption>
</figure>
    
    


## 7. Get CATE / uplift predictions

For a binary treatment experiment, the predicted uplift is simply the estimated effect of going from control (`T0=0`) to treatment (`T1=1`).


```python

def cate_predict(model, X_data):
    return model.effect(X_data)

cate_train = {name: cate_predict(model, X_train) for name, model in models.items()}
cate_test  = {name: cate_predict(model, X_test)  for name, model in models.items()}

pd.DataFrame({k: np.asarray(v).ravel()[:5] for k, v in cate_test.items()})
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
      <th>SLearner</th>
      <th>TLearner</th>
      <th>XLearner</th>
      <th>DRLearner</th>
      <th>LinearDML</th>
      <th>CausalForestDML</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>66.838847</td>
      <td>6255.915004</td>
      <td>3166.608390</td>
      <td>3546.645036</td>
      <td>5768.924299</td>
      <td>3262.998457</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-137.971741</td>
      <td>-4166.883389</td>
      <td>-616.743227</td>
      <td>-4107.529289</td>
      <td>8828.882399</td>
      <td>884.823406</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1596.118279</td>
      <td>2900.016716</td>
      <td>474.007860</td>
      <td>-5306.252895</td>
      <td>3452.218217</td>
      <td>1884.682510</td>
    </tr>
    <tr>
      <th>3</th>
      <td>539.519164</td>
      <td>294.686420</td>
      <td>1326.560882</td>
      <td>101.825536</td>
      <td>1065.778789</td>
      <td>950.812086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>494.817338</td>
      <td>4926.474853</td>
      <td>2585.430688</td>
      <td>6747.790039</td>
      <td>-353.910100</td>
      <td>207.545285</td>
    </tr>
  </tbody>
</table>
</div>



## 8. Uplift by decile

This is one of the chapter's main ideas.

### Intuition
1. Score each unit by predicted uplift.
2. Sort from highest predicted uplift to lowest.
3. Split into 10 bins (deciles).
4. Inside each decile, estimate the **observed uplift**:
Observed uplift = E[Y | T = 1] − E[Y | T = 0]
5. A good model should show **higher observed uplift in top deciles** than in lower deciles.


```python

def uplift_by_decile(y, t, uplift, n_bins=10):
    temp = pd.DataFrame({
        "y": np.asarray(y),
        "t": np.asarray(t).astype(int),
        "uplift": np.asarray(uplift).ravel()
    }).sort_values("uplift", ascending=False).reset_index(drop=True)

    temp["decile"] = pd.qcut(
        np.arange(len(temp)),
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    rows = []
    for decile, group in temp.groupby("decile"):
        treated = group.loc[group["t"] == 1, "y"]
        control = group.loc[group["t"] == 0, "y"]
        uplift_obs = treated.mean() - control.mean() if len(treated) and len(control) else np.nan
        rows.append({
            "decile": int(decile),
            "n": len(group),
            "treated_n": len(treated),
            "control_n": len(control),
            "observed_uplift": uplift_obs
        })
    return pd.DataFrame(rows)
```


```python

def plot_uplift_by_decile(decile_df, title):
    plt.figure(figsize=(8, 4.5))
    plt.bar(decile_df["decile"], decile_df["observed_uplift"])
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Decile (0 = highest predicted uplift)")
    plt.ylabel("Observed uplift")
    plt.show()
```


```python

# Example: one model on train and test
example_train = uplift_by_decile(y_train, T_train, cate_train["XLearner"])
example_test  = uplift_by_decile(y_test,  T_test,  cate_test["XLearner"])

plot_uplift_by_decile(example_train, "XLearner — uplift by decile (train)")
plot_uplift_by_decile(example_test, "XLearner — uplift by decile (test)")
```

<figure>
  <img src="{{ site.baseurl }}/images/output_29_0_2.png">
  <figcaption style="text-align:center;">Fig3a.XLearner — uplift by decile (train). </figcaption>
</figure>

<figure>
  <img src="{{ site.baseurl }}/images/output_29_1_2.png">
  <figcaption style="text-align:center;">Fig3b.XLearner — uplift by decile (test). </figcaption>
</figure>

    


### Compare all models


```python

fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.4 * len(models)))

for row_idx, (name, _) in enumerate(models.items()):
    train_df = uplift_by_decile(y_train, T_train, cate_train[name])
    test_df  = uplift_by_decile(y_test,  T_test,  cate_test[name])

    axes[row_idx, 0].bar(train_df["decile"], train_df["observed_uplift"])
    axes[row_idx, 0].axhline(0, linewidth=1)
    axes[row_idx, 0].set_title(f"{name} — Train")
    axes[row_idx, 0].set_xlabel("Decile")
    axes[row_idx, 0].set_ylabel("Obs uplift")

    axes[row_idx, 1].bar(test_df["decile"], test_df["observed_uplift"])
    axes[row_idx, 1].axhline(0, linewidth=1)
    axes[row_idx, 1].set_title(f"{name} — Test")
    axes[row_idx, 1].set_xlabel("Decile")
    axes[row_idx, 1].set_ylabel("Obs uplift")

plt.tight_layout()
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_31_0_2.png">
  <figcaption style="text-align:center;">Fig4.Compare — uplift by decile. </figcaption>
</figure>
        

## 9. Expected response (binary-treatment version)

In the chapter, expected response is used as a more decision-oriented metric.

### Intuition
For each person:
- the model recommends treatment if predicted uplift is positive,
- otherwise the model recommends control.

Because we only observe the outcome under the **actually assigned treatment**, we estimate how good the policy is with an **inverse-propensity-weighted policy value**:

Expected Response = E[ Y × I(T = π(X)) / P(T | X) ]

In a randomized experiment with roughly constant assignment probability, this becomes straightforward to compute.


```python

def expected_response_binary(y, t, uplift_scores, p_treat=None):
    y = np.asarray(y)
    t = np.asarray(t).astype(int)
    uplift_scores = np.asarray(uplift_scores).ravel()

    if p_treat is None:
        p_treat = t.mean()

    p_control = 1 - p_treat
    recommended_treatment = (uplift_scores > 0).astype(int)

    weight = np.where(recommended_treatment == 1, 1 / p_treat, 1 / p_control)
    observed_if_followed = (recommended_treatment == t).astype(int)

    return np.mean(y * observed_if_followed * weight)
```


```python

metric_rows = []
for name in models:
    train_er = expected_response_binary(y_train, T_train, cate_train[name], p_treat=T_train.mean())
    test_er  = expected_response_binary(y_test,  T_test,  cate_test[name],  p_treat=T_test.mean())
    metric_rows.append({
        "Model": name,
        "ExpectedResponse_Train": train_er,
        "ExpectedResponse_Test": test_er
    })

expected_response_df = pd.DataFrame(metric_rows).sort_values("ExpectedResponse_Test", ascending=False)
expected_response_df
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
      <th>Model</th>
      <th>ExpectedResponse_Train</th>
      <th>ExpectedResponse_Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SLearner</td>
      <td>10084.517560</td>
      <td>9177.923448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TLearner</td>
      <td>10047.760508</td>
      <td>8225.655137</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XLearner</td>
      <td>9822.502760</td>
      <td>8089.190804</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CausalForestDML</td>
      <td>8044.869124</td>
      <td>7214.912195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DRLearner</td>
      <td>8838.831954</td>
      <td>5989.903949</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LinearDML</td>
      <td>5440.521197</td>
      <td>4828.437861</td>
    </tr>
  </tbody>
</table>
</div>




```python

plt.figure(figsize=(9, 4.5))
plt.bar(expected_response_df["Model"], expected_response_df["ExpectedResponse_Test"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Expected response (test)")
plt.title("Policy value comparison on the test set")
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_35_0_2.png">
  <figcaption style="text-align:center;">Fig5.Policy value comparison on the test set. </figcaption>
</figure>
    
    


## 10. Confidence intervals with Linear DML

One nice feature highlighted in the chapter is that `LinearDML` can provide confidence intervals directly.


```python

lb, ub = models["LinearDML"].effect_interval(X_test, T0=0, T1=1, alpha=0.05)

print("Lower bounds (first 5):", lb[:5])
print("Upper bounds (first 5):", ub[:5])
```

    Lower bounds (first 5): [ -288.91205216  2458.50208897 -2216.77656149 -4170.36233339
     -3868.06794743]
    Upper bounds (first 5): [11826.76064953 15199.26270957  9121.21299571  6301.91991069
      3160.24774699]
    


```python

intervals = np.column_stack([lb, ub])
contains_zero = np.mean(np.sign(intervals[:, 0]) != np.sign(intervals[:, 1]))

print("Fraction of test observations whose 95% CI contains 0:", round(float(contains_zero), 4))
```

    Fraction of test observations whose 95% CI contains 0: 0.8534
    


```python

plt.figure(figsize=(8, 4.5))
sample_idx = np.arange(min(50, len(lb)))
plt.errorbar(
    sample_idx,
    cate_test["LinearDML"][:len(sample_idx)],
    yerr=[
        cate_test["LinearDML"][:len(sample_idx)] - lb[:len(sample_idx)],
        ub[:len(sample_idx)] - cate_test["LinearDML"][:len(sample_idx)]
    ],
    fmt='o'
)
plt.axhline(0, linewidth=1)
plt.title("LinearDML: point estimates with 95% confidence intervals")
plt.xlabel("Sample index")
plt.ylabel("Estimated treatment effect")
plt.show()
```

<figure>
  <img src="{{ site.baseurl }}/images/output_39_0_2.png">
  <figcaption style="text-align:center;">Fig5.LinearDML: point estimates with 95% confidence intervals. </figcaption>
</figure>
    

### Optional targeting rule
A very practical rule is:

- only target people whose estimated uplift is positive **and**
- whose interval does **not** include zero.

That gives you a more conservative treatment policy.


```python

conservative_treat = (
    (np.asarray(cate_test["LinearDML"]).ravel() > 0) &
    (lb > 0)
).astype(int)

pd.Series(conservative_treat).value_counts(normalize=True).rename("share")
```




    0    0.879479
    1    0.120521
    Name: share, dtype: float64



## 11. A compact comparison table

This is not the exact table from the book, but it serves the same purpose:
bring together practical aspects you might care about.


```python

comparison = expected_response_df.merge(
    timing_df[["Model", "RelativeToFastest"]],
    on="Model",
    how="left"
).rename(columns={"RelativeToFastest": "RelativeComputeCost"})

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
      <th>Model</th>
      <th>ExpectedResponse_Train</th>
      <th>ExpectedResponse_Test</th>
      <th>RelativeComputeCost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SLearner</td>
      <td>10084.517560</td>
      <td>9177.923448</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TLearner</td>
      <td>10047.760508</td>
      <td>8225.655137</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XLearner</td>
      <td>9822.502760</td>
      <td>8089.190804</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CausalForestDML</td>
      <td>8044.869124</td>
      <td>7214.912195</td>
      <td>11.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DRLearner</td>
      <td>8838.831954</td>
      <td>5989.903949</td>
      <td>47.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LinearDML</td>
      <td>5440.521197</td>
      <td>4828.437861</td>
      <td>7.2</td>
    </tr>
  </tbody>
</table>
</div>



## 12. Extra: counterfactual explanations

The chapter ends with a short section on **counterfactual explanations**.

### Key idea
This is **not** about estimating the true causal effect in the world.
It is about asking:

> *What small change to the input would flip the model's decision?*

To demonstrate this in a way that is close to the chapter, we will:
1. take the estimated uplift from `LinearDML`,
2. convert it into a simple recommendation label:
   - `1` if the model recommends treatment,
   - `0` if the model recommends control,
3. use **DiCE** to find feature changes that would flip the recommendation.

This explains the **decision policy**, not the true data-generating mechanism.


```python

# Create a recommendation label from the LinearDML uplift estimates
train_recommend = (np.asarray(cate_train["LinearDML"]).ravel() > 0).astype(int)
test_recommend  = (np.asarray(cate_test["LinearDML"]).ravel() > 0).astype(int)

recommend_train_df = X_train.copy()
recommend_train_df["recommend_treatment"] = train_recommend

recommend_test_df = X_test.copy()
recommend_test_df["recommend_treatment"] = test_recommend

recommend_train_df.head()
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
      <th>age</th>
      <th>educ</th>
      <th>married</th>
      <th>nodegree</th>
      <th>re74</th>
      <th>re75</th>
      <th>race_hispan</th>
      <th>race_white</th>
      <th>recommend_treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>26</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>18</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>296</th>
      <td>28</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>5260.631</td>
      <td>3790.113</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>23</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>369</th>
      <td>18</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0.000</td>
      <td>1491.339</td>
      <td>False</td>
      <td>True</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Train a small interpretable recommendation model for DiCE
recommendation_model = LogisticRegression(max_iter=5000)
recommendation_model.fit(X_train, train_recommend)

print("Share recommended for treatment in train:", train_recommend.mean().round(3))
print("Share recommended for treatment in test :", test_recommend.mean().round(3))
```

    Share recommended for treatment in train: 0.638
    Share recommended for treatment in test : 0.642
    


```python

# DiCE setup
import dice_ml
from dice_ml import Dice

dice_data = dice_ml.Data(
    dataframe=recommend_train_df,
    continuous_features=["age", "educ", "re74", "re75"],
    outcome_name="recommend_treatment",
)

dice_model = dice_ml.Model(model=recommendation_model, backend="sklearn", model_type="classifier")
dice = Dice(dice_data, dice_model, method="random")
```


```python

# Pick one person the policy currently does NOT recommend for treatment
candidate_pool = recommend_test_df.copy()
candidate_pool["pred"] = recommendation_model.predict(X_test)

query_idx = candidate_pool.index[candidate_pool["pred"] == 0][0]
query_instance = X_test.loc[[query_idx]]

query_instance
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
      <th>age</th>
      <th>educ</th>
      <th>married</th>
      <th>nodegree</th>
      <th>re74</th>
      <th>re75</th>
      <th>race_hispan</th>
      <th>race_white</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>372</th>
      <td>17</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1453.742</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python

cf = dice.generate_counterfactuals(
    query_instance,
    total_CFs=3,
    desired_class="opposite",
    features_to_vary=["age", "educ", "re74", "re75", "married", "nodegree"]
)

cf.visualize_as_dataframe(show_only_changes=True)
```

    100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.55it/s]

    Query instance (original outcome : 0)
    

    
    


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
      <th>age</th>
      <th>educ</th>
      <th>married</th>
      <th>nodegree</th>
      <th>re74</th>
      <th>re75</th>
      <th>race_hispan</th>
      <th>race_white</th>
      <th>recommend_treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1453.741943</td>
      <td>True</td>
      <td>True</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    
    Diverse Counterfactual set (new outcome: 1)
    


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
      <th>age</th>
      <th>educ</th>
      <th>married</th>
      <th>nodegree</th>
      <th>re74</th>
      <th>re75</th>
      <th>race_hispan</th>
      <th>race_white</th>
      <th>recommend_treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>2584.0</td>
      <td>-</td>
      <td>-</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>17788.4</td>
      <td>-</td>
      <td>-</td>
      <td>False</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### How to read the counterfactuals
Each row says something like:

> “If this person's features changed in these small ways, the **recommendation model** would flip from 'do not treat' to 'treat'.”

Again, this is an explanation of the **policy model**, not a guarantee about the real world.
That is exactly the spirit of the final section in Molak's chapter.

## 13. Practical takeaways

### What this notebook reproduced from the chapter
- randomized-experiment sanity checks,
- S / T / X / DR / LinearDML / CausalForestDML estimators,
- fit-time comparison,
- uplift-by-decile evaluation,
- expected response / policy value,
- confidence intervals,
- counterfactual explanations.

### What changed from the chapter
- We used a **different public dataset** (LaLonde, not Hillstrom).
- The setup is **binary treatment** rather than multi-treatment.
- The final counterfactual section explains the **recommendation model**, which is the cleanest way to mirror the chapter on this dataset.




```python

```
