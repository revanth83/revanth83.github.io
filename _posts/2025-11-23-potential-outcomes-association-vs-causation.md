# A Minimal Demo in Python

**Author:** Revanth  
**Context:** Inspired by Chapter 1 of Matheus Facure’s _Causal Inference in Python_.  
**Goal:** Show, with code, why **association is not causation** and how **potential outcomes** help us reason about causal effects.

---

## 1. What this notebook demonstrates

We’ll build a tiny synthetic example that illustrates all the core ideas:

1. **Potential outcomes**: each unit has two outcomes –
**Potential Outcomes**

- `Y0` → outcome if NOT treated (control)
- `Y1` → outcome if treated


2. **Realized outcome as a switch**:  
If Tᵢ = 0  →  Yᵢ = Y₀ᵢ          (control outcome)
If Tᵢ = 1  →  Yᵢ = Y₁ᵢ          (treated outcome)

General form (switching equation):
Yᵢ = (1 − Tᵢ) × Y₀ᵢ  +  Tᵢ × Y₁ᵢ

3. **Selection bias**: when treated and untreated units differ for reasons beyond treatment,  
 E[Y₀ | T = 1]  ≠  E[Y₀ | T = 0],  
   so simple comparisons are biased.
4. **Randomization / ignorability**: when treatment is independent of potential outcomes,  
  (Y₀, Y₁)  independent of  T,  
   then True causal effect:
E[Y₁ − Y₀]  =  E[Y | T = 1] − E[Y | T = 0].
5. **SUTVA** (Stable Unit Treatment Value Assumption):
   - One unit’s treatment does not affect another’s outcome.
   - There is only one “version” of treatment.
6. **Assumptions are essential**: causal inference always relies on assumptions to connect causal quantities to estimators.



```python
#!pip install pandas
```

```python
import numpy as np
import pandas as pd

np.random.seed(42)

pd.set_option("display.precision", 4)
```
## 2. Simulating potential outcomes

We start by simulating **potential outcomes** for each unit.

Think of the two variables as:

- **Y0** → conversion probability *if the customer does **not** receive an email*
- **Y1** → conversion probability *if the customer **does** receive an email*

Only **one** of these outcomes is observed for each customer — depending on whether they were treated.

---

### True Average Treatment Effect (ATE)

The **true causal effect** is the average difference between treated and untreated potential outcomes:

**ATE = mean(Y1 − Y0)**

Or in full words:

> The ATE tells us how much conversion would increase *on average* if all customers received the email.



```python
# number of units (e.g., customers)
N = 2000

# potential outcomes: Y0 = no treatment, Y1 = treatment
# here: think of them as probabilities of conversion
Y0 = np.random.normal(loc=0.50, scale=0.08, size=N)  # baseline
Y1 = np.random.normal(loc=0.65, scale=0.08, size=N)  # better with treatment

df = pd.DataFrame({"Y0": Y0, "Y1": Y1})

true_ate = (df["Y1"] - df["Y0"]).mean()
true_ate
```




    np.float64(0.15348287061625443)



The number printed above is the **ground-truth causal effect** in this simulated world.

We never get Y0 and Y1 for the same individual.
One of them always remains unobserved (the counterfactual).


## 3. Biased treatment assignment (selection bias)

Now let’s simulate a **biased** treatment assignment rule.

Suppose the marketing team tends to send the email to customers who would
already have a higher chance of converting. In our toy world, we let treatment depend on Y1:

- Customers with larger Y1 are **more likely to get treated**.

This creates a dependence between treatment and potential outcomes, so  
(Y0, Y1) are NOT independent of T.


```python
# biased treatment: higher Y1 -> more likely to be treated
T = (df["Y1"] > np.quantile(df["Y1"], 0.5)).astype(int)
df["T"] = T

# realized outcome using the switch function
df["Y"] = (1 - df["T"]) * df["Y0"] + df["T"] * df["Y1"]

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
      <th>Y0</th>
      <th>Y1</th>
      <th>T</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.4309</td>
      <td>0.5609</td>
      <td>0</td>
      <td>0.4309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4975</td>
      <td>0.5995</td>
      <td>0</td>
      <td>0.4975</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.5014</td>
      <td>0.5746</td>
      <td>0</td>
      <td>0.5014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5378</td>
      <td>0.6062</td>
      <td>0</td>
      <td>0.5378</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.3907</td>
      <td>0.6329</td>
      <td>0</td>
      <td>0.3907</td>
    </tr>
  </tbody>
</table>
</div>



The line

\```python
df["Y"] = (1 - df["T"]) * df["Y0"] + df["T"] * df["Y1"]
\```

implements exactly the **switch function**:

Yᵢ = (1 − Tᵢ) × Y₀ᵢ  +  Tᵢ × Y₁ᵢ

## 4. Naive association vs true causal effect

Let’s compare:

- The **naive difference in observed outcomes** between treated and untreated:  
average(Y | T = 1)  −  average(Y | T = 0)


- The **true ATE** that we know from the simulated potential outcomes:  
 average(Y1 − Y0)


```python
treated_mean = df.loc[df["T"] == 1, "Y"].mean()
control_mean = df.loc[df["T"] == 0, "Y"].mean()
naive_diff = treated_mean - control_mean

print(f"True ATE:              {true_ate: .4f}")
print(f"Naive treated - control: {naive_diff: .4f}")
```

    True ATE:               0.1535
    Naive treated - control:  0.2224
    

Because treatment was **not randomized**, treated customers have different potential outcomes than controls.
Formally,average(Y0 | T = 1)  !=  average(Y0 | T = 0).

So the naive comparison average(Y | T = 1)  −  average(Y | T = 0)
is a **biased estimator** of the true causal effect.

## 5. Randomized experiment (association ~= causation)

Now we simulate a **proper randomized experiment**.

We keep the same potential outcomes (Y0, Y1), but assign treatment at random:

Random treatment assignment:
T_rand ~ Bernoulli(p = 0.5)


This makes treatment **independent** of potential outcomes:

(Y0, Y1) are independent of T_rand


Under this condition, the treated and control groups are comparable, and

average(Y1 − Y0)  =  average(Y | T_rand = 1)  −  average(Y | T_rand = 0)



```python
# randomized treatment
df["T_rand"] = np.random.binomial(1, 0.5, size=N)

# realized outcome under randomized assignment
df["Y_rand"] = (1 - df["T_rand"]) * df["Y0"] + df["T_rand"] * df["Y1"]

treated_mean_rand = df.loc[df["T_rand"] == 1, "Y_rand"].mean()

control_mean_rand = df.loc[df["T_rand"] == 0, "Y_rand"].mean()

rand_diff = treated_mean_rand - control_mean_rand

print(f"True ATE:                    {true_ate: .4f}")
print(f"Randomized treated - control: {rand_diff: .4f}")
```

    True ATE:                     0.1535
    Randomized treated - control:  0.1563
    

Now the **difference in average outcomes** between treated and control units
is very close to the **true ATE**. Randomization made the groups comparable, so
**association now equals causation** in this setup.

## 6. SUTVA and other key assumptions

To make sense of this framework, we quietly relied on some important assumptions:

1. **SUTVA (Stable Unit Treatment Value Assumption)**
   - No interference: one unit’s treatment does not affect another unit’s outcome.  
     E.g., sending an email to Customer A does not change what happens to Customer B.
   - No hidden versions of treatment: “treatment” is well-defined and consistent.

2. **Ignorability / Independence**
   - In the randomized case, treatment is independent of potential outcomes:  
     (Y0, Y1) are independent of T.  
     This is what makes simple differences in averages unbiased.

3. **Consistency**
   - If a unit receives treatment level t, the observed outcome equals the corresponding potential outcome:  
     If Tᵢ = t, then Yᵢ = Yᵢ(t).

These assumptions are what allow us to go from the **causal quantity we care about**  
(e.g., average(Y1 − Y0)) to a **statistical estimator** based on observed data  
(e.g., the average difference between treated and control).

## 7. Recap

In this small synthetic example, we:

- Defined **potential outcomes** Y0 and Y1 for each unit.
- Used the **switch function** Yᵢ = (1 − Tᵢ) × Y0ᵢ  +  Tᵢ × Y1ᵢ to build observed outcomes.
- Saw how **biased treatment assignment** (non-random) leads to a **biased estimate** of the treatment effect.
- Showed that with a **randomized experiment**, the simple difference in means recovers the **true ATE**.
- Made explicit the role of **assumptions** (SUTVA, independence, consistency) in causal inference.

This is minimal demo of the core ideas from the first chapter of Facure’s book, and as the conceptual
foundation for more applied causal modeling in marketing (uplift models, propensity scores, etc.).
