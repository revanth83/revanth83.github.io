# Chapter 4 — Linear Regression for Causal Inference  
## Bank Marketing Case Study

This notebook illustrates **Chapter 4 of Matheus Facure’s _Causal Inference in Python_**
using the **Bank Marketing** dataset.

**Causal question:**  
Does contacting customers by **cellular** instead of **telephone** increase the probability
of subscribing to a term deposit?

- Treatment `T`: contact = cellular (1) vs telephone (0)  
- Outcome `Y`: subscription = yes (1) vs no (0)


## Dataset (links + download options)

We use the **Bank Marketing** dataset (Portuguese bank direct marketing campaigns).

### Official source (UCI Machine Learning Repository)
- Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing  
  (Direct downloadable files are linked on that page.)

### Kaggle mirror (CSV download; requires Kaggle login)
- https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

### Which file should you use?
- Kaggle typically provides `bank-full.csv` / `bank.csv`
- The UCI dataset provides multiple formats; this notebook loads from UCI via `ucimlrepo`


# 📘 Causal Linear Regression — Blog Interpretation Layer

## What problem are we solving?
We are trying to estimate the **causal effect of a treatment variable** on an outcome variable using linear regression under causal assumptions.

## Predictive vs Causal Regression
Predictive regression answers:
> If X changes, how does Y move in historical data?

Causal regression answers:
> If we *intervene* and change Treatment, how does Y change?

Required assumptions:
- No unobserved confounding (Conditional Ignorability)
- Correct functional form (or good approximation)
- No post-treatment leakage
- Sufficient overlap between treated and control populations

## Business Interpretation
In marketing / fintech:
- Treatment coefficient ≈ Incremental lift
- Positive → treatment helps
- Negative → treatment hurts
- Near zero → no incremental value



## ⚠️ Key Causal Assumptions Being Used Here

### 1️⃣ Conditional Ignorability
After controlling for covariates X:
Treatment ⟂ Potential Outcomes

If violated → biased effect estimate

### 2️⃣ Overlap (Positivity)
Every user has some probability of treatment and control

If violated:
- Extrapolation risk
- Unstable coefficients

### 3️⃣ No Post-Treatment Controls
Do NOT include variables influenced by treatment
This creates collider bias or blocks part of the treatment effect



## 🧪 Diagnostics — Causal Meaning

Residual Diagnostics:
- Random residuals → Model specification reasonable
- Patterned residuals → Possible nonlinearity or missing confounder

Coefficient Stability:
- Large swings across specs → weak identification or collinearity

Overlap Checks:
If treated and control covariate distributions differ heavily:
→ Model extrapolates → causal estimate fragile



```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os

np.random.seed(42)
FIG_DIR = "figures_ch4_bank_marketing"
os.makedirs(FIG_DIR, exist_ok=True)

```


```python
!pip install ucimlrepo
```

    Requirement already satisfied: ucimlrepo in c:\users\revan\minicondanew\lib\site-packages (0.0.7)
    Requirement already satisfied: pandas>=1.0.0 in c:\users\revan\minicondanew\lib\site-packages (from ucimlrepo) (2.3.3)
    Requirement already satisfied: certifi>=2020.12.5 in c:\users\revan\minicondanew\lib\site-packages (from ucimlrepo) (2025.11.12)
    Requirement already satisfied: numpy>=1.26.0 in c:\users\revan\minicondanew\lib\site-packages (from pandas>=1.0.0->ucimlrepo) (2.3.5)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\revan\minicondanew\lib\site-packages (from pandas>=1.0.0->ucimlrepo) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\revan\minicondanew\lib\site-packages (from pandas>=1.0.0->ucimlrepo) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\revan\minicondanew\lib\site-packages (from pandas>=1.0.0->ucimlrepo) (2025.2)
    Requirement already satisfied: six>=1.5 in c:\users\revan\minicondanew\lib\site-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->ucimlrepo) (1.17.0)
    


## Load public Bank Marketing data (UCI)



```python

from ucimlrepo import fetch_ucirepo

bank = fetch_ucirepo(id=222)
df = pd.concat([bank.data.features, bank.data.targets], axis=1)

df["Y"] = (df["y"].astype(str).str.lower() == "yes").astype(int)
df["contact"] = df["contact"].astype(str).str.lower()
df = df[df["contact"].isin(["cellular","telephone"])].copy()
df["T"] = (df["contact"]=="cellular").astype(int)

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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
      <th>Y</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12657</th>
      <td>27</td>
      <td>management</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>35</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>4</td>
      <td>jul</td>
      <td>255</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12658</th>
      <td>54</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>466</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>4</td>
      <td>jul</td>
      <td>297</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12659</th>
      <td>43</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>105</td>
      <td>no</td>
      <td>yes</td>
      <td>cellular</td>
      <td>4</td>
      <td>jul</td>
      <td>668</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12660</th>
      <td>31</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>19</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>4</td>
      <td>jul</td>
      <td>65</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12661</th>
      <td>27</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>126</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>4</td>
      <td>jul</td>
      <td>436</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>NaN</td>
      <td>no</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




## Naive regression (difference in means)



## 📊 How to Interpret the Treatment Coefficient

Treatment Coefficient ≈ Average Treatment Effect (ATE) **if assumptions hold**

Example:
Coefficient = 0.12

Interpretation:
If treatment is applied, outcome increases by ~0.12 units on average,
holding confounders constant.

Business Translation:
Expected incremental lift per treated user ≈ coefficient value



```python

m_naive = smf.ols("Y ~ T", data=df).fit()
m_naive.summary().tables[1]

```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.1342</td> <td>    0.007</td> <td>   20.384</td> <td> 0.000</td> <td>    0.121</td> <td>    0.147</td>
</tr>
<tr>
  <th>T</th>         <td>    0.0150</td> <td>    0.007</td> <td>    2.171</td> <td> 0.030</td> <td>    0.001</td> <td>    0.029</td>
</tr>
</table>




## Adjusted regression with month fixed effects



## 📊 How to Interpret the Treatment Coefficient

Treatment Coefficient ≈ Average Treatment Effect (ATE) **if assumptions hold**

Example:
Coefficient = 0.12

Interpretation:
If treatment is applied, outcome increases by ~0.12 units on average,
holding confounders constant.

Business Translation:
Expected incremental lift per treated user ≈ coefficient value



```python

num_controls = ["age","balance","campaign","pdays","previous","day"]
num_controls = [c for c in num_controls if c in df.columns]

cat_controls = ["job","marital","education","housing","loan","month","poutcome"]
cat_controls = [c for c in cat_controls if c in df.columns]

formula = "Y ~ T"
for c in num_controls:
    formula += f" + {c}"
for c in cat_controls:
    formula += f" + C({c})"

m_adj = smf.ols(formula, data=df).fit()
m_adj.summary().tables[1]

```




<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                 <td>    0.0799</td> <td>    0.040</td> <td>    2.003</td> <td> 0.045</td> <td>    0.002</td> <td>    0.158</td>
</tr>
<tr>
  <th>C(job)[T.blue-collar]</th>     <td>   -0.0197</td> <td>    0.015</td> <td>   -1.313</td> <td> 0.189</td> <td>   -0.049</td> <td>    0.010</td>
</tr>
<tr>
  <th>C(job)[T.entrepreneur]</th>    <td>   -0.0317</td> <td>    0.027</td> <td>   -1.169</td> <td> 0.243</td> <td>   -0.085</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(job)[T.housemaid]</th>       <td>   -0.0374</td> <td>    0.032</td> <td>   -1.178</td> <td> 0.239</td> <td>   -0.100</td> <td>    0.025</td>
</tr>
<tr>
  <th>C(job)[T.management]</th>      <td>    0.0140</td> <td>    0.016</td> <td>    0.865</td> <td> 0.387</td> <td>   -0.018</td> <td>    0.046</td>
</tr>
<tr>
  <th>C(job)[T.retired]</th>         <td>    0.0218</td> <td>    0.024</td> <td>    0.928</td> <td> 0.354</td> <td>   -0.024</td> <td>    0.068</td>
</tr>
<tr>
  <th>C(job)[T.self-employed]</th>   <td>    0.0060</td> <td>    0.025</td> <td>    0.239</td> <td> 0.811</td> <td>   -0.043</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(job)[T.services]</th>        <td>    0.0029</td> <td>    0.017</td> <td>    0.167</td> <td> 0.867</td> <td>   -0.031</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(job)[T.student]</th>         <td>    0.0855</td> <td>    0.027</td> <td>    3.213</td> <td> 0.001</td> <td>    0.033</td> <td>    0.138</td>
</tr>
<tr>
  <th>C(job)[T.technician]</th>      <td>   -0.0068</td> <td>    0.015</td> <td>   -0.458</td> <td> 0.647</td> <td>   -0.036</td> <td>    0.022</td>
</tr>
<tr>
  <th>C(job)[T.unemployed]</th>      <td>    0.0746</td> <td>    0.027</td> <td>    2.773</td> <td> 0.006</td> <td>    0.022</td> <td>    0.127</td>
</tr>
<tr>
  <th>C(marital)[T.married]</th>     <td>    0.0178</td> <td>    0.013</td> <td>    1.355</td> <td> 0.175</td> <td>   -0.008</td> <td>    0.043</td>
</tr>
<tr>
  <th>C(marital)[T.single]</th>      <td>    0.0261</td> <td>    0.015</td> <td>    1.744</td> <td> 0.081</td> <td>   -0.003</td> <td>    0.055</td>
</tr>
<tr>
  <th>C(education)[T.secondary]</th> <td>    0.0108</td> <td>    0.014</td> <td>    0.798</td> <td> 0.425</td> <td>   -0.016</td> <td>    0.037</td>
</tr>
<tr>
  <th>C(education)[T.tertiary]</th>  <td>    0.0287</td> <td>    0.017</td> <td>    1.727</td> <td> 0.084</td> <td>   -0.004</td> <td>    0.061</td>
</tr>
<tr>
  <th>C(housing)[T.yes]</th>         <td>   -0.1011</td> <td>    0.010</td> <td>  -10.009</td> <td> 0.000</td> <td>   -0.121</td> <td>   -0.081</td>
</tr>
<tr>
  <th>C(loan)[T.yes]</th>            <td>   -0.0382</td> <td>    0.012</td> <td>   -3.237</td> <td> 0.001</td> <td>   -0.061</td> <td>   -0.015</td>
</tr>
<tr>
  <th>C(month)[T.aug]</th>           <td>    0.1050</td> <td>    0.020</td> <td>    5.240</td> <td> 0.000</td> <td>    0.066</td> <td>    0.144</td>
</tr>
<tr>
  <th>C(month)[T.dec]</th>           <td>    0.1243</td> <td>    0.036</td> <td>    3.495</td> <td> 0.000</td> <td>    0.055</td> <td>    0.194</td>
</tr>
<tr>
  <th>C(month)[T.feb]</th>           <td>   -0.0059</td> <td>    0.016</td> <td>   -0.359</td> <td> 0.720</td> <td>   -0.038</td> <td>    0.026</td>
</tr>
<tr>
  <th>C(month)[T.jan]</th>           <td>   -0.0727</td> <td>    0.020</td> <td>   -3.674</td> <td> 0.000</td> <td>   -0.112</td> <td>   -0.034</td>
</tr>
<tr>
  <th>C(month)[T.jul]</th>           <td>    0.1828</td> <td>    0.027</td> <td>    6.877</td> <td> 0.000</td> <td>    0.131</td> <td>    0.235</td>
</tr>
<tr>
  <th>C(month)[T.jun]</th>           <td>    0.1547</td> <td>    0.024</td> <td>    6.543</td> <td> 0.000</td> <td>    0.108</td> <td>    0.201</td>
</tr>
<tr>
  <th>C(month)[T.mar]</th>           <td>    0.2091</td> <td>    0.030</td> <td>    6.869</td> <td> 0.000</td> <td>    0.149</td> <td>    0.269</td>
</tr>
<tr>
  <th>C(month)[T.may]</th>           <td>   -0.0257</td> <td>    0.013</td> <td>   -1.969</td> <td> 0.049</td> <td>   -0.051</td> <td>   -0.000</td>
</tr>
<tr>
  <th>C(month)[T.nov]</th>           <td>   -0.0323</td> <td>    0.016</td> <td>   -2.044</td> <td> 0.041</td> <td>   -0.063</td> <td>   -0.001</td>
</tr>
<tr>
  <th>C(month)[T.oct]</th>           <td>    0.1450</td> <td>    0.023</td> <td>    6.182</td> <td> 0.000</td> <td>    0.099</td> <td>    0.191</td>
</tr>
<tr>
  <th>C(month)[T.sep]</th>           <td>    0.2023</td> <td>    0.024</td> <td>    8.306</td> <td> 0.000</td> <td>    0.155</td> <td>    0.250</td>
</tr>
<tr>
  <th>C(poutcome)[T.other]</th>      <td>    0.0334</td> <td>    0.010</td> <td>    3.318</td> <td> 0.001</td> <td>    0.014</td> <td>    0.053</td>
</tr>
<tr>
  <th>C(poutcome)[T.success]</th>    <td>    0.4063</td> <td>    0.012</td> <td>   34.528</td> <td> 0.000</td> <td>    0.383</td> <td>    0.429</td>
</tr>
<tr>
  <th>T</th>                         <td>    0.0414</td> <td>    0.016</td> <td>    2.639</td> <td> 0.008</td> <td>    0.011</td> <td>    0.072</td>
</tr>
<tr>
  <th>age</th>                       <td>    0.0008</td> <td>    0.000</td> <td>    1.569</td> <td> 0.117</td> <td>   -0.000</td> <td>    0.002</td>
</tr>
<tr>
  <th>balance</th>                   <td> 3.072e-06</td> <td> 1.32e-06</td> <td>    2.320</td> <td> 0.020</td> <td> 4.76e-07</td> <td> 5.67e-06</td>
</tr>
<tr>
  <th>campaign</th>                  <td>   -0.0144</td> <td>    0.003</td> <td>   -5.478</td> <td> 0.000</td> <td>   -0.020</td> <td>   -0.009</td>
</tr>
<tr>
  <th>pdays</th>                     <td>    0.0001</td> <td> 4.29e-05</td> <td>    3.052</td> <td> 0.002</td> <td> 4.69e-05</td> <td>    0.000</td>
</tr>
<tr>
  <th>previous</th>                  <td>    0.0018</td> <td>    0.001</td> <td>    2.059</td> <td> 0.040</td> <td> 8.62e-05</td> <td>    0.004</td>
</tr>
</table>




## Frisch–Waugh–Lovell (FWL) theorem



## 📊 How to Interpret the Treatment Coefficient

Treatment Coefficient ≈ Average Treatment Effect (ATE) **if assumptions hold**

Example:
Coefficient = 0.12

Interpretation:
If treatment is applied, outcome increases by ~0.12 units on average,
holding confounders constant.

Business Translation:
Expected incremental lift per treated user ≈ coefficient value



## 🧪 Diagnostics — Causal Meaning

Residual Diagnostics:
- Random residuals → Model specification reasonable
- Patterned residuals → Possible nonlinearity or missing confounder

Coefficient Stability:
- Large swings across specs → weak identification or collinearity

Overlap Checks:
If treated and control covariate distributions differ heavily:
→ Model extrapolates → causal estimate fragile



```python

f_T = "T ~ age + balance + C(month)"
f_Y = "Y ~ age + balance + C(month)"

mT = smf.ols(f_T, data=df).fit()
mY = smf.ols(f_Y, data=df).fit()

df["T_res"] = mT.resid
df["Y_res"] = mY.resid

m_fwl = smf.ols("Y_res ~ T_res", data=df).fit()
m_fwl.summary().tables[1]

```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td> 7.752e-15</td> <td>    0.002</td> <td> 4.09e-12</td> <td> 1.000</td> <td>   -0.004</td> <td>    0.004</td>
</tr>
<tr>
  <th>T_res</th>     <td>    0.0348</td> <td>    0.007</td> <td>    5.130</td> <td> 0.000</td> <td>    0.022</td> <td>    0.048</td>
</tr>
</table>




## Heterogeneous effects (interaction with month)



## 📊 How to Interpret the Treatment Coefficient

Treatment Coefficient ≈ Average Treatment Effect (ATE) **if assumptions hold**

Example:
Coefficient = 0.12

Interpretation:
If treatment is applied, outcome increases by ~0.12 units on average,
holding confounders constant.

Business Translation:
Expected incremental lift per treated user ≈ coefficient value



```python

m_inter = smf.ols("Y ~ T*C(month) + age + balance", data=df).fit()
m_inter.summary().tables[1]

```




<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>    0.1503</td> <td>    0.026</td> <td>    5.840</td> <td> 0.000</td> <td>    0.100</td> <td>    0.201</td>
</tr>
<tr>
  <th>C(month)[T.aug]</th>   <td>   -0.0609</td> <td>    0.032</td> <td>   -1.880</td> <td> 0.060</td> <td>   -0.124</td> <td>    0.003</td>
</tr>
<tr>
  <th>C(month)[T.dec]</th>   <td>    0.1768</td> <td>    0.061</td> <td>    2.907</td> <td> 0.004</td> <td>    0.058</td> <td>    0.296</td>
</tr>
<tr>
  <th>C(month)[T.feb]</th>   <td>   -0.0500</td> <td>    0.032</td> <td>   -1.579</td> <td> 0.114</td> <td>   -0.112</td> <td>    0.012</td>
</tr>
<tr>
  <th>C(month)[T.jan]</th>   <td>   -0.0935</td> <td>    0.038</td> <td>   -2.433</td> <td> 0.015</td> <td>   -0.169</td> <td>   -0.018</td>
</tr>
<tr>
  <th>C(month)[T.jul]</th>   <td>   -0.1160</td> <td>    0.027</td> <td>   -4.333</td> <td> 0.000</td> <td>   -0.168</td> <td>   -0.064</td>
</tr>
<tr>
  <th>C(month)[T.jun]</th>   <td>    0.0530</td> <td>    0.045</td> <td>    1.179</td> <td> 0.239</td> <td>   -0.035</td> <td>    0.141</td>
</tr>
<tr>
  <th>C(month)[T.mar]</th>   <td>    0.2151</td> <td>    0.053</td> <td>    4.096</td> <td> 0.000</td> <td>    0.112</td> <td>    0.318</td>
</tr>
<tr>
  <th>C(month)[T.may]</th>   <td>   -0.1171</td> <td>    0.029</td> <td>   -4.057</td> <td> 0.000</td> <td>   -0.174</td> <td>   -0.061</td>
</tr>
<tr>
  <th>C(month)[T.nov]</th>   <td>   -0.0911</td> <td>    0.030</td> <td>   -3.063</td> <td> 0.002</td> <td>   -0.149</td> <td>   -0.033</td>
</tr>
<tr>
  <th>C(month)[T.oct]</th>   <td>    0.2386</td> <td>    0.038</td> <td>    6.242</td> <td> 0.000</td> <td>    0.164</td> <td>    0.314</td>
</tr>
<tr>
  <th>C(month)[T.sep]</th>   <td>    0.1794</td> <td>    0.048</td> <td>    3.713</td> <td> 0.000</td> <td>    0.085</td> <td>    0.274</td>
</tr>
<tr>
  <th>T</th>                 <td>    0.0127</td> <td>    0.025</td> <td>    0.508</td> <td> 0.612</td> <td>   -0.036</td> <td>    0.062</td>
</tr>
<tr>
  <th>T:C(month)[T.aug]</th> <td>   -0.0275</td> <td>    0.033</td> <td>   -0.826</td> <td> 0.409</td> <td>   -0.093</td> <td>    0.038</td>
</tr>
<tr>
  <th>T:C(month)[T.dec]</th> <td>    0.1171</td> <td>    0.066</td> <td>    1.764</td> <td> 0.078</td> <td>   -0.013</td> <td>    0.247</td>
</tr>
<tr>
  <th>T:C(month)[T.feb]</th> <td>    0.0232</td> <td>    0.033</td> <td>    0.703</td> <td> 0.482</td> <td>   -0.042</td> <td>    0.088</td>
</tr>
<tr>
  <th>T:C(month)[T.jan]</th> <td>    0.0015</td> <td>    0.040</td> <td>    0.036</td> <td> 0.971</td> <td>   -0.077</td> <td>    0.080</td>
</tr>
<tr>
  <th>T:C(month)[T.jul]</th> <td>    0.0183</td> <td>    0.028</td> <td>    0.656</td> <td> 0.512</td> <td>   -0.036</td> <td>    0.073</td>
</tr>
<tr>
  <th>T:C(month)[T.jun]</th> <td>    0.1884</td> <td>    0.047</td> <td>    3.995</td> <td> 0.000</td> <td>    0.096</td> <td>    0.281</td>
</tr>
<tr>
  <th>T:C(month)[T.mar]</th> <td>    0.1182</td> <td>    0.055</td> <td>    2.130</td> <td> 0.033</td> <td>    0.009</td> <td>    0.227</td>
</tr>
<tr>
  <th>T:C(month)[T.may]</th> <td>    0.0429</td> <td>    0.030</td> <td>    1.431</td> <td> 0.152</td> <td>   -0.016</td> <td>    0.102</td>
</tr>
<tr>
  <th>T:C(month)[T.nov]</th> <td>   -0.0110</td> <td>    0.031</td> <td>   -0.355</td> <td> 0.723</td> <td>   -0.072</td> <td>    0.050</td>
</tr>
<tr>
  <th>T:C(month)[T.oct]</th> <td>    0.0058</td> <td>    0.041</td> <td>    0.141</td> <td> 0.888</td> <td>   -0.075</td> <td>    0.087</td>
</tr>
<tr>
  <th>T:C(month)[T.sep]</th> <td>    0.1375</td> <td>    0.051</td> <td>    2.684</td> <td> 0.007</td> <td>    0.037</td> <td>    0.238</td>
</tr>
<tr>
  <th>age</th>               <td>    0.0007</td> <td>    0.000</td> <td>    3.828</td> <td> 0.000</td> <td>    0.000</td> <td>    0.001</td>
</tr>
<tr>
  <th>balance</th>           <td> 4.007e-06</td> <td> 6.04e-07</td> <td>    6.631</td> <td> 0.000</td> <td> 2.82e-06</td> <td> 5.19e-06</td>
</tr>
</table>




## Key takeaways

- Regression = adjusted comparison  
- Month fixed effects remove seasonality bias  
- FWL explains why controls work  
- Interactions show when marketing works better  



## 🧠 When Linear Regression Works Well for Causal Inference

✅ Large sample size  
✅ Good overlap  
✅ Strong confounder coverage  
✅ Approximately linear effect  

## 🚫 When It Struggles

❌ Strong nonlinear HTE  
❌ Hidden confounders  
❌ Extreme treatment imbalance  
❌ Post-treatment variable leakage  

## 🔄 Bridge to Meta-Learners and Forests
If linear model struggles:
→ S-Learner / T-Learner  
→ X-Learner  
→ Causal Forests  

