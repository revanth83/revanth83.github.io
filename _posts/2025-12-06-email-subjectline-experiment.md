---
layout: post
title: "Email Subject Line Experiment"
subtitle: "A/B Testing, Statistical Inference, and Power Analysis in Practice"
date: 2025-12-06
author: "Revanth Garlapati"
tags: [A/B Testing, Experimentation, Statistics, Marketing Analytics]
categories: [Data Science, Experimentation]
description: "A practical walkthrough of analyzing an email subject line A/B test using difference in means, confidence intervals, hypothesis testing, and power analysis."
---
## Randomized Experiments and Statistical Review

---

## Abstract

This analysis builds practical intuition for randomized experiments and how to correctly interpret treatment effects in real-world settings.

Using a realistic email campaign example, we analyze how different subject lines impact customer response rates. Beyond computing differences, the focus is on understanding **uncertainty, statistical significance, and whether observed effects are actually meaningful**.

The goal is to answer a practical question:

**When we see a lift in an A/B test, how do we know if it is real, reliable, and worth acting on?**

---

## Introduction

Randomized experiments are the gold standard for measuring causal effects. When implemented correctly, they allow us to isolate the impact of an intervention from noise and confounding factors.

In practice, however, many mistakes arise not from running experiments, but from **incorrect interpretation of results** — overreacting to noise, misreading statistical significance, or ignoring practical impact.

This tutorial focuses on **building intuition**, not just formulas.

We use a simple but realistic marketing example — an email campaign — to walk through how to:

- Compare treatment vs control groups  
- Quantify uncertainty using standard errors and confidence intervals  
- Evaluate statistical significance  
- Interpret results in a business context  

---

## Data Source

The dataset used in this analysis is publicly available on Kaggle:

https://www.kaggle.com/datasets/aristotelisch/playground-mock-email-campaign

It simulates a real-world email marketing campaign with randomized subject lines and customer responses.

---

## Dataset

The analysis uses three datasets:

- **sent_emails.csv** — when and to whom each email was sent  
  *(includes Customer_ID, SubjectLine_ID)*  

- **responded.csv** — which customers responded  
  *(includes Customer_ID)*  

- **userbase.csv** — customer-level attributes  
  *(not required for this analysis, but available)*  

---

## Experiment Setup

We construct a randomized experiment with one control and two treatments:

- **Control:** SubjectLine_ID = 1  
- **Treatment A:** SubjectLine_ID = 2 vs control  
- **Treatment B:** SubjectLine_ID = 3 vs control  

The outcome variable is:

- **responded = 1** if the customer appears in responded.csv, else **0**

---

## What We Will Do

For each treatment vs control comparison, we will compute:

- Difference in mean response rates (**treatment effect**)  
- Standard Error (**SE**)  
- 95% Confidence Interval  
- t-statistic and p-value  

We will also:

- Apply a simple **sample size rule-of-thumb**  
- Interpret results in **business terms** (not just statistical terms)  

---

## Layout of the Analysis

The notebook is structured as follows:

1. Load and merge datasets  
2. Construct experiment groups  
3. Compute treatment effects  
4. Estimate uncertainty (SE and confidence intervals)  
5. Perform hypothesis testing  
6. Evaluate statistical vs practical significance  
7. Summarize key takeaways  

---

## Reference

This analysis builds on standard concepts in randomized experiments from:

- Matheus Facure — *Causal Inference for the Brave and True* (Chapter 2: Randomized Experiments)

However, all examples, datasets, and interpretations are **original and tailored to practical marketing use cases**.

---


## 1. Load and Merge the Datasets


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

pd.set_option("display.precision", 4)

# Load the three CSVs
sent = pd.read_csv("./chapter2_data/sent_emails.csv")
resp = pd.read_csv("./chapter2_data/responded.csv")
users = pd.read_csv("./chapter2_data/userbase.csv")

# Mark everyone in 'responded' as responded = 1
resp['responded'] = 1

# Collapse in case some customers responded multiple times
resp_flag = resp.groupby('Customer_ID', as_index=False)['responded'].max()

# Merge sent + response flags (left join keeps all sent emails)
df = sent.merge(resp_flag, on='Customer_ID', how='left')

# Customers not in responded.csv get responded = 0
df['responded'] = df['responded'].fillna(0).astype(int)

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
      <th>Sent_Date</th>
      <th>Customer_ID</th>
      <th>SubjectLine_ID</th>
      <th>responded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-28</td>
      <td>1413</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>83889</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-09</td>
      <td>457832</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-20</td>
      <td>127772</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-02-03</td>
      <td>192123</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Quick sanity checks


```python
print("Rows in sent_emails:", len(sent))
print("Unique customers in sent_emails:", sent['Customer_ID'].nunique())
print("Rows in responded:", len(resp))
print("Unique customers in responded:", resp['Customer_ID'].nunique())

print("\nResponse rate overall:")
print(df['responded'].mean())

print("\nDistribution of SubjectLine_ID:")
print(df['SubjectLine_ID'].value_counts())
```

    Rows in sent_emails: 2476354
    Unique customers in sent_emails: 496518
    Rows in responded: 378208
    Unique customers in responded: 264859
    
    Response rate overall:
    0.5998322533854207
    
    Distribution of SubjectLine_ID:
    SubjectLine_ID
    1    826717
    2    824837
    3    824800
    Name: count, dtype: int64
    

## 2. Define Control and Treatment Groups

We follow the below structure:

- Control group: customers who received **SubjectLine_ID = 1**  
- Treatment A: customers who received **SubjectLine_ID = 2**  
- Treatment B: customers who received **SubjectLine_ID = 3**  

We will treat each comparison separately:

1. SubjectLine 2 vs SubjectLine 1  
2. SubjectLine 3 vs SubjectLine 1  

This is equivalent to running two two-arm experiments that share the same control group
(very similar to *no_email vs short* and *no_email vs long* in Facure's cross-sell example).


```python
control = df[df['SubjectLine_ID'] == 1].copy()
treat2  = df[df['SubjectLine_ID'] == 2].copy()
treat3  = df[df['SubjectLine_ID'] == 3].copy()

print("Sample sizes:")
print("Control (1): ", len(control))
print("Treat 2:     ", len(treat2))
print("Treat 3:     ", len(treat3))

print("\nResponse rates by group:")
group_rates = df.groupby('SubjectLine_ID')['responded'].mean()
print(group_rates)
```

    Sample sizes:
    Control (1):  826717
    Treat 2:      824837
    Treat 3:      824800
    
    Response rates by group:
    SubjectLine_ID
    1    0.6021
    2    0.6025
    3    0.5949
    Name: responded, dtype: float64
    

## 3. Difference in Means, SE, CI, t-Statistic, p-Value

We now create a small helper function that, given a treatment group and a control group,
computes:

- Mean response in treatment and control  
- Difference in means (delta)  
- Standard error of the difference  
- 95% confidence interval for delta  
- t-statistic under H0: delta = 0  
- Two-sided p-value  

This mirrors the logic in standard literature: a simple difference in means test between treatment
and control in a randomized experiment.


```python
def diff_in_means_analysis(treat, control, outcome_col='responded'):
    """Compute difference in means, SE, 95% CI, t-statistic and p-value."""
    mu_t = treat[outcome_col].mean()
    mu_c = control[outcome_col].mean()
    delta = mu_t - mu_c

    # Standard error for difference in independent means
    se = np.sqrt(
        treat[outcome_col].var(ddof=1) / len(treat) +
        control[outcome_col].var(ddof=1) / len(control)
    )

    ci_low  = delta - 1.96 * se
    ci_high = delta + 1.96 * se

    # t-statistic for H0: delta = 0
    t_stat = delta / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if se > 0 else np.nan

    return {
        'mu_t': mu_t,
        'mu_c': mu_c,
        'delta': delta,
        'se': se,
        'ci_low': ci_low,
        'ci_high': ci_high,
        't_stat': t_stat,
        'p_value': p_value
    }

res_2_vs_1 = diff_in_means_analysis(treat2, control)
res_3_vs_1 = diff_in_means_analysis(treat3, control)

res_2_vs_1, res_3_vs_1
```




    ({'mu_t': np.float64(0.6025457150928002),
      'mu_c': np.float64(0.602054874884634),
      'delta': np.float64(0.0004908402081661434),
      'se': np.float64(0.000761672396676801),
      'ci_low': np.float64(-0.0010020376893203867),
      'ci_high': np.float64(0.0019837181056526734),
      't_stat': np.float64(0.6444243093325865),
      'p_value': np.float64(0.5193003251701691)},
     {'mu_t': np.float64(0.5948908826382153),
      'mu_c': np.float64(0.602054874884634),
      'delta': np.float64(-0.007163992246418727),
      'se': np.float64(0.0007628828501613492),
      'ci_low': np.float64(-0.008659242632734971),
      'ci_high': np.float64(-0.005668741860102482),
      't_stat': np.float64(-9.390684618095095),
      'p_value': np.float64(0.0)})



### 3.1. Summary table of numerical results


```python
summary = pd.DataFrame.from_dict({
    '2_vs_1': res_2_vs_1,
    '3_vs_1': res_3_vs_1
}, orient='index')

summary
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
      <th>mu_t</th>
      <th>mu_c</th>
      <th>delta</th>
      <th>se</th>
      <th>ci_low</th>
      <th>ci_high</th>
      <th>t_stat</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2_vs_1</th>
      <td>0.6025</td>
      <td>0.6021</td>
      <td>0.0005</td>
      <td>0.0008</td>
      <td>-0.0010</td>
      <td>0.0020</td>
      <td>0.6444</td>
      <td>0.5193</td>
    </tr>
    <tr>
      <th>3_vs_1</th>
      <td>0.5949</td>
      <td>0.6021</td>
      <td>-0.0072</td>
      <td>0.0008</td>
      <td>-0.0087</td>
      <td>-0.0057</td>
      <td>-9.3907</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Interpretation: SubjectLine 2 vs 1


```python
r = res_2_vs_1
print("SubjectLine 2 vs 1:\n")
print(f"Control mean (1):        {r['mu_c']:.4f}")
print(f"Treatment mean (2):      {r['mu_t']:.4f}")
print(f"Difference (delta):      {r['delta']:.4f}")
print(f"Standard error (SE):     {r['se']:.6f}")
print(f"95% CI for delta:        [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
print(f"t-statistic:             {r['t_stat']:.3f}")
print(f"p-value:                 {r['p_value']:.4f}")
```

    SubjectLine 2 vs 1:
    
    Control mean (1):        0.6021
    Treatment mean (2):      0.6025
    Difference (delta):      0.0005
    Standard error (SE):     0.000762
    95% CI for delta:        [-0.0010, 0.0020]
    t-statistic:             0.644
    p-value:                 0.5193
    

### How to read this (2 vs 1)

- **Difference (delta)** tells you how much higher the response rate is for SubjectLine 2
  compared to SubjectLine 1 in absolute terms (for example, a value of `0.012` means a 1.2
  percentage point uplift).  
- The **95% confidence interval** shows the range of plausible values for the true treatment
  effect, given this sample.  
- If the CI **includes 0**, then we *cannot* rule out the possibility of no true effect.  
- The **t-statistic** measures how many standard errors away from 0 the observed delta is.  
- The **p-value** is the probability of seeing a difference at least as extreme as this one,
  if the true delta were actually 0 (no effect).  

**Practical rule of thumb (following standard A/B testing interpretation):**

- If `p_value < 0.05`, we say the effect is *statistically significant* at the 5% level.  
- If `p_value >= 0.05`, we say “we do not have enough evidence to reject no effect.”  

You should now look at your numbers above and ask:

- Is the uplift for SubjectLine 2 vs 1 both **statistically significant** (p < 0.05)
  and **practically meaningful** (large enough in percentage terms to matter for the business)?

## 5. Interpretation: SubjectLine 3 vs 1


```python
r = res_3_vs_1
print("SubjectLine 3 vs 1:\n")
print(f"Control mean (1):        {r['mu_c']:.4f}")
print(f"Treatment mean (3):      {r['mu_t']:.4f}")
print(f"Difference (delta):      {r['delta']:.4f}")
print(f"Standard error (SE):     {r['se']:.6f}")
print(f"95% CI for delta:        [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
print(f"t-statistic:             {r['t_stat']:.3f}")
print(f"p-value:                 {r['p_value']:.4f}")
```

    SubjectLine 3 vs 1:
    
    Control mean (1):        0.6021
    Treatment mean (3):      0.5949
    Difference (delta):      -0.0072
    Standard error (SE):     0.000763
    95% CI for delta:        [-0.0087, -0.0057]
    t-statistic:             -9.391
    p-value:                 0.0000
    

### How to read this (3 vs 1)

The same logic applies here as for SubjectLine 2 vs 1:

- Check the **sign and magnitude** of `delta` to see whether SubjectLine 3 is doing better
  or worse than SubjectLine 1, and by how many percentage points.  
- Look at the **95% CI** to see the range of plausible true effects.  
- Check the **p-value** to see whether the improvement (or drop) is statistically significant.  

From a marketing standpoint, you would typically pick the subject line that is:

1. Statistically significantly better than control (if any), **and**  
2. Practically large enough to justify rollout (for example, +0.5–1.0 percentage point uplift
   might be interesting, depending on scale and economics).

## 6. Power and Sample Size (Facure’s Rule of Thumb)

A commonly used approximation for planning experiment is:

If we want:

- 95% significance (alpha = 5%)  
- 80% power (1 - beta = 80%)  

Then the **minimum detectable effect** delta (in absolute terms) needs to satisfy:

> delta ≈ 2.8 * SE

And if we open up SE and solve for n (per group), we get the rule of thumb:

> n ≈ 16 * sigma^2 / delta^2  

where:

- `sigma^2` is the variance of the outcome (here: response 0/1),  
- `delta` is the smallest effect you care to detect (for example, a 1% uplift = 0.01).


```python
# Use the control group variance as our estimate of sigma^2
sigma2 = control['responded'].var(ddof=1)

# Suppose we care about detecting a 1 percentage point difference (~0.01)
delta_target = 0.01

n_required = 16 * sigma2 / (delta_target ** 2)

print(f"Estimated sigma^2 from control: {sigma2:.6f}")
print(f"Target detectable effect (delta): {delta_target:.4f}")
print(f"Required sample size per group (approx): {n_required:.1f}")
```

    Estimated sigma^2 from control: 0.239585
    Target detectable effect (delta): 0.0100
    Required sample size per group (approx): 38333.6
    

### 6.1. Compare with actual sample sizes


```python
n_ctrl = len(control)
n_2 = len(treat2)
n_3 = len(treat3)

print(f"Required n per group (approx): {n_required:.1f}")
print(f"Actual n (control, 1):   {n_ctrl}")
print(f"Actual n (treatment, 2): {n_2}")
print(f"Actual n (treatment, 3): {n_3}")
```

    Required n per group (approx): 38333.6
    Actual n (control, 1):   826717
    Actual n (treatment, 2): 824837
    Actual n (treatment, 3): 824800
    

### Interpretation of power / sample size

- If your actual group sizes (for control and each treatment) are **larger** than the
  `n_required`, then your experiment is roughly **properly powered** to detect a 1 percentage
  point effect.  
- If your actual group sizes are **smaller** than `n_required`, the experiment might be
  **underpowered**. In this case:
  - Failing to find a statistically significant effect does **not** mean there is no effect.  
  - It may simply mean the sample size is too small to reliably detect the effect you care about.  

This is exactly the warning in the standard literature that **“absence of evidence is not evidence of absence.”**

## 7. Final Summary (for Marketing / Product Stakeholders)

Using the email subject line experiment, we evaluated two treatments against a control:

- SubjectLine 2 vs SubjectLine 1  
- SubjectLine 3 vs SubjectLine 1  

### What the data shows

- **SubjectLine 2 vs 1**
  - Uplift: ~ +0.05 percentage points  
  - 95% CI includes 0  
  - p-value ≈ 0.52  
  → **No statistically significant difference**

- **SubjectLine 3 vs 1**
  - Uplift: ~ –0.72 percentage points  
  - 95% CI entirely below 0  
  - p-value ≈ 0.00  
  → **Statistically significant decrease in performance**

---

### Business interpretation

- There is **no evidence that SubjectLine 2 improves performance** over the control.  
- There is **strong evidence that SubjectLine 3 performs worse** than the control.  

**Decision:**
- Keep SubjectLine 1 (baseline)
- SubjectLine 2 is neutral → no reason to roll out
- SubjectLine 3 should be **rejected**

---

### Was the experiment reliable?

- Required sample size (for detecting ~1pp effect): ~38K per group  
- Actual sample size: ~824K per group  

→ The experiment is **well-powered**, meaning:
- Lack of significance for SubjectLine 2 is **not due to low sample size**
- The null result is **credible**

---

### Key takeaways (Causal + Experimentation)

1. **Randomization enables causal interpretation**  
   Differences observed here can be interpreted as true treatment effects.

2. **Statistical significance matters — but direction matters too**  
   Not all treatments fail equally:
   - Some show *no effect* (SubjectLine 2)
   - Some show *negative impact* (SubjectLine 3)

3. **Confidence intervals are more informative than p-values alone**  
   They show both magnitude and uncertainty of the effect.

4. **Underpowered vs well-powered experiments must be distinguished**  
   Here, large sample sizes ensure conclusions are reliable.

5. **Absence of lift ≠ missed opportunity**  
   It often means the variant truly does not improve performance.

---

### Practical takeaway

In real marketing experiments:

- Focus on **both statistical significance and business impact**
- Avoid rolling out variants with:
  - Insignificant uplift
  - Negative impact (even if small)
- Use **power calculations upfront** to ensure experiments are conclusive

---

This framework can be directly applied to:

- Subject line testing  
- Creative optimization  
- Pricing experiments  
- Personalization strategies  

Any setting where decisions depend on small but meaningful differences in response rates.

