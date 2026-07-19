# When Attribution Models Disagree: How Different Methods Lead to Different Strategies

## A practical study of customer journeys showing how attribution methods influence budget allocation.

---

## Business Question

> Which channels should get credit for conversions?

## Dataset

This notebook uses the Kaggle **Multi-Touch Attribution** dataset.
>https://www.kaggle.com/datasets/vivekparasharr/multi-touch-attribution

The dataset contains user-level marketing interactions with timestamp, channel, campaign, conversion, revenue, and other fields.

## Abstract

Multi-touch attribution is useful for understanding observed customer journeys, but different attribution methods answer different questions.

This notebook compares:

1. First-touch attribution
2. Last-touch attribution
3. Linear attribution
4. Simple model-based attribution

The goal is to show that attribution is assumption-driven.

## 1. Setup and Data Loading

The raw data is touchpoint-level. Each row is one user interaction with a marketing channel.

To perform attribution, I first convert the raw data into **customer journeys**: ordered sequences of touchpoints for each user.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

DATA_PATH = "data/multi_touch_attribution_data.csv"
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find 'multi_touch_attribution_data.csv'. "
        "Place the CSV in the same folder as this notebook or update DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())
```

    Shape: (10000, 5)
    


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
      <th>User ID</th>
      <th>Timestamp</th>
      <th>Channel</th>
      <th>Campaign</th>
      <th>Conversion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>83281</td>
      <td>2025-02-10 07:58:51</td>
      <td>Email</td>
      <td>New Product Launch</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>68071</td>
      <td>2025-02-10 23:38:48</td>
      <td>Search Ads</td>
      <td>Winter Sale</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90131</td>
      <td>2025-02-11 10:41:07</td>
      <td>Social Media</td>
      <td>Brand Awareness</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>71026</td>
      <td>2025-02-10 08:19:44</td>
      <td>Direct Traffic</td>
      <td>-</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94486</td>
      <td>2025-02-10 15:15:46</td>
      <td>Email</td>
      <td>Retargeting</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>


## 2. Data Setup

The required fields are:

- User ID
- Timestamp
- Channel
- Campaign
- Conversion

The timestamp is critical because first-touch and last-touch attribution depend entirely on event order.


```python
user_col = "User ID"
time_col = "Timestamp"
channel_col = "Channel"
campaign_col = "Campaign"
conversion_col = "Conversion"

required_cols = [user_col, time_col, channel_col, campaign_col, conversion_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df[time_col] = pd.to_datetime(df[time_col])
df["converted_touch"] = df[conversion_col].astype(str).str.lower().eq("yes").astype(int)
df = df.sort_values([user_col, time_col]).reset_index(drop=True)

print("Rows:", len(df))
print("Unique users:", df[user_col].nunique())
print("Date range:", df[time_col].min(), "to", df[time_col].max())

print("\nChannel distribution:")
display(df[channel_col].value_counts().to_frame("touchpoints"))

print("\nTouchpoint-level conversion flag:")
display(df[conversion_col].value_counts().to_frame("count"))
```

    Rows: 10000
    Unique users: 2847
    Date range: 2025-02-10 00:00:22 to 2025-02-11 23:59:58
    
    Channel distribution:
    


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
      <th>touchpoints</th>
    </tr>
    <tr>
      <th>Channel</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Direct Traffic</th>
      <td>1721</td>
    </tr>
    <tr>
      <th>Referral</th>
      <td>1685</td>
    </tr>
    <tr>
      <th>Display Ads</th>
      <td>1669</td>
    </tr>
    <tr>
      <th>Social Media</th>
      <td>1662</td>
    </tr>
    <tr>
      <th>Email</th>
      <td>1654</td>
    </tr>
    <tr>
      <th>Search Ads</th>
      <td>1609</td>
    </tr>
  </tbody>
</table>
</div>


    
    Touchpoint-level conversion flag:
    


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
    </tr>
    <tr>
      <th>Conversion</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>5056</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>4944</td>
    </tr>
  </tbody>
</table>
</div>


## 3. Build Customer Journeys

I aggregate touchpoints into one row per user.

For each user, I keep:

- ordered channel path,
- ordered campaign path,
- whether the user converted at least once,
- number of touches,
- first and last interaction time.

This creates the base table for attribution.


```python
journeys = (
    df.groupby(user_col)
      .agg(
          path=(channel_col, list),
          campaigns=(campaign_col, list),
          converted=("converted_touch", "max"),
          first_time=(time_col, "min"),
          last_time=(time_col, "max"),
          n_touches=(channel_col, "size"),
      )
      .reset_index()
)

journeys["path_str"] = journeys["path"].apply(lambda x: " > ".join(x))

print("Journeys:", len(journeys))
print("Journey-level conversion rate:", journeys["converted"].mean())
display(journeys[["path_str", "converted", "n_touches"]].head(10))
display(journeys["n_touches"].describe().to_frame("n_touches"))
```

    Journeys: 2847
    Journey-level conversion rate: 0.8363189322093432
    


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
      <th>path_str</th>
      <th>converted</th>
      <th>n_touches</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Search Ads &gt; Display Ads</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Search Ads &gt; Display Ads</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Social Media &gt; Direct Traffic &gt; Email</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Search Ads &gt; Social Media &gt; Social Media &gt; Sea...</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Display Ads &gt; Email &gt; Referral &gt; Display Ads &gt;...</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Referral &gt; Email &gt; Referral &gt; Direct Traffic &gt;...</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Social Media &gt; Display Ads &gt; Display Ads</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Email &gt; Referral</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Display Ads &gt; Direct Traffic &gt; Display Ads &gt; S...</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Social Media &gt; Referral &gt; Direct Traffic &gt; Soc...</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>n_touches</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2,847.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.5125</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.7707</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>12.0000</td>
    </tr>
  </tbody>
</table>
</div>


### Journey Interpretation

If most users had only one touchpoint, MTA would not be very interesting because first-touch and last-touch would mostly match.

The more multi-touch journeys exist, the more attribution becomes a real credit-assignment problem.

## 4. Most Common Customer Paths

Before assigning credit, I inspect common journeys.

This gives the analysis a practical foundation. Attribution results should be interpreted in the context of actual observed paths.


```python
top_paths = journeys["path_str"].value_counts().head(15).reset_index()
top_paths.columns = ["path", "number_of_users"]

display(top_paths)

plt.figure(figsize=(10, 6))
plt.barh(top_paths["path"][::-1], top_paths["number_of_users"][::-1])
plt.title("Most Common Customer Journey Paths")
plt.xlabel("Number of users")
plt.ylabel("Path")
plt.tight_layout()
plt.show()
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
      <th>path</th>
      <th>number_of_users</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Search Ads</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Direct Traffic</td>
      <td>58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Email</td>
      <td>54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Social Media</td>
      <td>54</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Referral</td>
      <td>52</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Referral &gt; Social Media</td>
      <td>26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Display Ads &gt; Direct Traffic</td>
      <td>24</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Referral &gt; Search Ads</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Referral &gt; Email</td>
      <td>21</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Social Media &gt; Referral</td>
      <td>20</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Search Ads &gt; Email</td>
      <td>20</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Direct Traffic &gt; Direct Traffic</td>
      <td>19</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Email &gt; Social Media</td>
      <td>19</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Referral &gt; Direct Traffic</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_9_1.png)
    


## 5. First-Touch Attribution

First-touch gives 100% of conversion credit to the first channel in a converting user’s path.

Business interpretation:

> This method rewards channels that start journeys.

It is useful when the business cares about demand creation, but it can under-credit channels that help close conversions later.


```python
converted_journeys = journeys[journeys["converted"] == 1].copy()
channels = sorted(df[channel_col].unique())

first_counter = Counter()

for path in converted_journeys["path"]:
    first_counter[path[0]] += 1

first_touch = pd.DataFrame({
    "channel": channels,
    "first_touch_credit": [first_counter[c] for c in channels],
})

first_touch["first_touch_share"] = first_touch["first_touch_credit"] / first_touch["first_touch_credit"].sum()

display(first_touch.sort_values("first_touch_share", ascending=False))
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
      <th>channel</th>
      <th>first_touch_credit</th>
      <th>first_touch_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>428</td>
      <td>0.1798</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Direct Traffic</td>
      <td>411</td>
      <td>0.1726</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Referral</td>
      <td>408</td>
      <td>0.1714</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Social Media</td>
      <td>389</td>
      <td>0.1634</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Email</td>
      <td>374</td>
      <td>0.1571</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Search Ads</td>
      <td>371</td>
      <td>0.1558</td>
    </tr>
  </tbody>
</table>
</div>


## 6. Last-Touch Attribution

Last-touch gives 100% of conversion credit to the final channel in a converting user’s path.

Business interpretation:

> This method rewards channels closest to conversion.

It is easy to explain, but it can over-credit bottom-of-funnel channels.


```python
last_counter = Counter()

for path in converted_journeys["path"]:
    last_counter[path[-1]] += 1

last_touch = pd.DataFrame({
    "channel": channels,
    "last_touch_credit": [last_counter[c] for c in channels],
})

last_touch["last_touch_share"] = last_touch["last_touch_credit"] / last_touch["last_touch_credit"].sum()

display(last_touch.sort_values("last_touch_share", ascending=False))
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
      <th>channel</th>
      <th>last_touch_credit</th>
      <th>last_touch_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Direct Traffic</td>
      <td>425</td>
      <td>0.1785</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>401</td>
      <td>0.1684</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Search Ads</td>
      <td>395</td>
      <td>0.1659</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Email</td>
      <td>393</td>
      <td>0.1651</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Referral</td>
      <td>384</td>
      <td>0.1613</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Social Media</td>
      <td>383</td>
      <td>0.1609</td>
    </tr>
  </tbody>
</table>
</div>


## 7. Linear Attribution

Linear attribution divides credit equally across all channels in the path.

Business interpretation:

> This method assumes every touchpoint contributed equally.

It is simple and balanced, but it ignores position, timing, and true influence.


```python
linear_counter = Counter()

for path in converted_journeys["path"]:
    credit = 1 / len(path)
    for ch in path:
        linear_counter[ch] += credit

linear = pd.DataFrame({
    "channel": channels,
    "linear_credit": [linear_counter[c] for c in channels],
})

linear["linear_share"] = linear["linear_credit"] / linear["linear_credit"].sum()

display(linear.sort_values("linear_share", ascending=False))
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
      <th>channel</th>
      <th>linear_credit</th>
      <th>linear_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Direct Traffic</td>
      <td>408.2766</td>
      <td>0.1715</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>407.1562</td>
      <td>0.1710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Referral</td>
      <td>398.7049</td>
      <td>0.1675</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Social Media</td>
      <td>397.6037</td>
      <td>0.1670</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Email</td>
      <td>387.7168</td>
      <td>0.1628</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Search Ads</td>
      <td>381.5418</td>
      <td>0.1602</td>
    </tr>
  </tbody>
</table>
</div>


## 8. Compare Rule-Based Attribution Methods

This is the main tension of the post.

The same customer journeys can produce different answers depending on the attribution rule.


```python
attrib = (
    first_touch[["channel", "first_touch_share"]]
    .merge(last_touch[["channel", "last_touch_share"]], on="channel")
    .merge(linear[["channel", "linear_share"]], on="channel")
)

display(attrib.sort_values("linear_share", ascending=False))

plot_df = attrib.set_index("channel")[["first_touch_share", "last_touch_share", "linear_share"]]

plt.figure(figsize=(11, 6))
x = np.arange(len(plot_df.index))
w = 0.25

plt.bar(x - w, plot_df["first_touch_share"], w, label="First touch")
plt.bar(x, plot_df["last_touch_share"], w, label="Last touch")
plt.bar(x + w, plot_df["linear_share"], w, label="Linear")

plt.xticks(x, plot_df.index, rotation=30, ha="right")
plt.ylabel("Attribution share")
plt.title("Attribution Share by Method")
plt.legend()
plt.tight_layout()
plt.show()
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
      <th>channel</th>
      <th>first_touch_share</th>
      <th>last_touch_share</th>
      <th>linear_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Direct Traffic</td>
      <td>0.1726</td>
      <td>0.1785</td>
      <td>0.1715</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>0.1798</td>
      <td>0.1684</td>
      <td>0.1710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Referral</td>
      <td>0.1714</td>
      <td>0.1613</td>
      <td>0.1675</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Social Media</td>
      <td>0.1634</td>
      <td>0.1609</td>
      <td>0.1670</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Email</td>
      <td>0.1571</td>
      <td>0.1651</td>
      <td>0.1628</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Search Ads</td>
      <td>0.1558</td>
      <td>0.1659</td>
      <td>0.1602</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_17_1.png)
    


## Three Models, Three Different Answers

Applying:

- First-touch attribution
- Last-touch attribution
- Linear attribution

can produce different channel rankings.

> Same data → different conclusions.

Each rule values a different part of the funnel:

- First-touch favors introducers.
- Last-touch favors closers.
- Linear favors broad participation.

This is why attribution is not a single objective truth. It is a structured credit allocation rule.

## Three Models, Three Different Budget Decisions

If I were to act on each model:

- First-touch would push me to invest more in awareness or acquisition-starting channels.
- Last-touch would push me to invest more in closing channels.
- Linear attribution would suggest a more balanced allocation.

These are not small differences. They can lead to fundamentally different marketing strategies.

> Attribution choice = strategy choice.

## Attribution Is a Choice, Not a Discovery

Attribution does not discover the true contribution of each channel.

It encodes assumptions:

- First-touch assumes the first interaction is most important.
- Last-touch assumes the final interaction is most important.
- Linear assumes all interactions are equally important.

> Attribution is assumption-driven.

That does not make it useless. It means the method should match the business question.

## 9. Raw Conversion Difference by Channel Presence

Now I ask a simple diagnostic question:

> Do users who saw a channel convert more often than users who did not?

This is not causal. It is still useful because it shows which channels are associated with conversion at the journey level.


```python
channel_counts = pd.DataFrame([
    {ch: path.count(ch) for ch in channels}
    for path in journeys["path"]
]).fillna(0)

channel_presence = (channel_counts > 0).astype(int)
y = journeys["converted"].values

rows = []

for ch in channels:
    saw = channel_presence[ch].astype(bool).values

    conv_seen = y[saw].mean() if saw.any() else np.nan
    conv_not = y[~saw].mean() if (~saw).any() else np.nan

    rows.append({
        "channel": ch,
        "reach_share": saw.mean(),
        "conversion_rate_if_seen": conv_seen,
        "conversion_rate_if_not_seen": conv_not,
        "raw_conversion_lift": conv_seen - conv_not,
    })

presence = pd.DataFrame(rows).sort_values("raw_conversion_lift", ascending=False)

display(presence)

plt.figure(figsize=(9, 5))
plt.bar(presence["channel"], presence["raw_conversion_lift"])
plt.title("Raw Conversion Rate Difference: Saw Channel vs Did Not See Channel")
plt.ylabel("Conversion rate difference")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
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
      <th>channel</th>
      <th>reach_share</th>
      <th>conversion_rate_if_seen</th>
      <th>conversion_rate_if_not_seen</th>
      <th>raw_conversion_lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>0.4405</td>
      <td>0.9075</td>
      <td>0.7803</td>
      <td>0.1272</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Referral</td>
      <td>0.4535</td>
      <td>0.8900</td>
      <td>0.7918</td>
      <td>0.0982</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Social Media</td>
      <td>0.4514</td>
      <td>0.8864</td>
      <td>0.7951</td>
      <td>0.0912</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Direct Traffic</td>
      <td>0.4521</td>
      <td>0.8858</td>
      <td>0.7955</td>
      <td>0.0903</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Email</td>
      <td>0.4440</td>
      <td>0.8861</td>
      <td>0.7966</td>
      <td>0.0895</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Search Ads</td>
      <td>0.4422</td>
      <td>0.8761</td>
      <td>0.8048</td>
      <td>0.0713</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_22_1.png)
    


### Interpretation

A higher conversion rate among users who saw a channel does not prove that the channel caused conversion.

Users who saw that channel may have:

- higher intent,
- more total touchpoints,
- different campaigns,
- different funnel stage.

This is where attribution and incrementality diverge.

## 10. Model-Based Attribution

Rule-based attribution uses fixed rules. Model-based attribution uses journey features to predict conversion.

Here I build a simple logistic regression model using:

- channel count features,
- channel presence features,
- total number of touches.

This is still not causal, but it gives a richer predictive view than fixed credit rules.


```python
X_counts = channel_counts.add_prefix("count_")
X_presence = channel_presence.add_prefix("saw_")

X = pd.concat([X_counts, X_presence], axis=1)
X["n_touches"] = journeys["n_touches"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

logit = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, C=1.0)),
])

logit.fit(X_train, y_train)
pred = logit.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred)

print(f"Test AUC: {auc:,.3f}")

coefs = pd.Series(logit.named_steps["model"].coef_[0], index=X.columns)

display(
    coefs.sort_values(key=np.abs, ascending=False)
         .head(20)
         .to_frame("coefficient")
)
```

    Test AUC: 0.754
    


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
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>n_touches</th>
      <td>0.6005</td>
    </tr>
    <tr>
      <th>count_Referral</th>
      <td>0.4721</td>
    </tr>
    <tr>
      <th>saw_Display Ads</th>
      <td>0.4604</td>
    </tr>
    <tr>
      <th>count_Direct Traffic</th>
      <td>0.3821</td>
    </tr>
    <tr>
      <th>saw_Email</th>
      <td>0.3272</td>
    </tr>
    <tr>
      <th>count_Social Media</th>
      <td>0.2308</td>
    </tr>
    <tr>
      <th>saw_Social Media</th>
      <td>0.1750</td>
    </tr>
    <tr>
      <th>count_Search Ads</th>
      <td>0.1663</td>
    </tr>
    <tr>
      <th>count_Email</th>
      <td>0.0888</td>
    </tr>
    <tr>
      <th>saw_Search Ads</th>
      <td>0.0867</td>
    </tr>
    <tr>
      <th>count_Display Ads</th>
      <td>0.0536</td>
    </tr>
    <tr>
      <th>saw_Referral</th>
      <td>-0.0436</td>
    </tr>
    <tr>
      <th>saw_Direct Traffic</th>
      <td>0.0381</td>
    </tr>
  </tbody>
</table>
</div>


### Model-Based Interpretation

The model-based view asks:

> Which journey features help predict conversion?

This can be more flexible than first-touch or last-touch, but it still estimates association.

If `n_touches` is highly predictive, that tells us something important: users with more interactions are more likely to convert. That can make channels appearing in longer journeys look more valuable.

## Why Longer Journeys Bias Attribution

Users with more touches often convert at higher rates.

So:

- channels in long journeys receive more credit,
- even if they are not the true drivers.

> Attribution rewards presence in converting journeys, not necessarily causal impact.

This is one of the most important limitations to understand before using attribution for budget allocation.

## 11. Model-Based Channel Signals

To compare with rule-based attribution, I extract channel presence and count coefficients.


```python
model_based = pd.DataFrame({
    "channel": channels,
    "presence_coefficient": [coefs.get(f"saw_{ch}", 0) for ch in channels],
    "count_coefficient": [coefs.get(f"count_{ch}", 0) for ch in channels],
})

model_based["abs_presence_coefficient"] = model_based["presence_coefficient"].abs()
model_based = model_based.sort_values("abs_presence_coefficient", ascending=False)

display(model_based)

plt.figure(figsize=(9, 5))
plt.bar(model_based["channel"], model_based["presence_coefficient"])
plt.title("Model-Based Channel Presence Coefficients")
plt.ylabel("Standardized logistic coefficient")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
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
      <th>channel</th>
      <th>presence_coefficient</th>
      <th>count_coefficient</th>
      <th>abs_presence_coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>0.4604</td>
      <td>0.0536</td>
      <td>0.4604</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Email</td>
      <td>0.3272</td>
      <td>0.0888</td>
      <td>0.3272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Social Media</td>
      <td>0.1750</td>
      <td>0.2308</td>
      <td>0.1750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Search Ads</td>
      <td>0.0867</td>
      <td>0.1663</td>
      <td>0.0867</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Referral</td>
      <td>-0.0436</td>
      <td>0.4721</td>
      <td>0.0436</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Direct Traffic</td>
      <td>0.0381</td>
      <td>0.3821</td>
      <td>0.0381</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](output_29_1.png)
    


### Model-Based Channel Interpretation

A positive model coefficient means the channel feature is associated with higher conversion probability after accounting for other model features.

It does not mean the channel caused the conversion.

The value of this section is comparison: if the model-based ranking differs from first-touch or last-touch, it shows that different attribution frameworks produce different business narratives.

## 12. Final Comparison Table

This table puts all attribution views together.

The most important columns are not just the shares themselves, but the rankings implied by each method.


```python
final = (
    attrib
    .merge(presence[["channel", "raw_conversion_lift"]], on="channel", how="left")
    .merge(model_based[["channel", "presence_coefficient", "count_coefficient"]], on="channel", how="left")
)

for col in [
    "first_touch_share",
    "last_touch_share",
    "linear_share",
    "raw_conversion_lift",
    "presence_coefficient",
]:
    final[f"rank_{col}"] = final[col].rank(ascending=False, method="first").astype(int)

display(final.sort_values("linear_share", ascending=False))
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
      <th>channel</th>
      <th>first_touch_share</th>
      <th>last_touch_share</th>
      <th>linear_share</th>
      <th>raw_conversion_lift</th>
      <th>presence_coefficient</th>
      <th>count_coefficient</th>
      <th>rank_first_touch_share</th>
      <th>rank_last_touch_share</th>
      <th>rank_linear_share</th>
      <th>rank_raw_conversion_lift</th>
      <th>rank_presence_coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Direct Traffic</td>
      <td>0.1726</td>
      <td>0.1785</td>
      <td>0.1715</td>
      <td>0.0903</td>
      <td>0.0381</td>
      <td>0.3821</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Display Ads</td>
      <td>0.1798</td>
      <td>0.1684</td>
      <td>0.1710</td>
      <td>0.1272</td>
      <td>0.4604</td>
      <td>0.0536</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Referral</td>
      <td>0.1714</td>
      <td>0.1613</td>
      <td>0.1675</td>
      <td>0.0982</td>
      <td>-0.0436</td>
      <td>0.4721</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Social Media</td>
      <td>0.1634</td>
      <td>0.1609</td>
      <td>0.1670</td>
      <td>0.0912</td>
      <td>0.1750</td>
      <td>0.2308</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Email</td>
      <td>0.1571</td>
      <td>0.1651</td>
      <td>0.1628</td>
      <td>0.0895</td>
      <td>0.3272</td>
      <td>0.0888</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Search Ads</td>
      <td>0.1558</td>
      <td>0.1659</td>
      <td>0.1602</td>
      <td>0.0713</td>
      <td>0.0867</td>
      <td>0.1663</td>
      <td>6</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## 13. What This Means for Decision-Making

If I use first-touch, I may invest more in channels that start journeys.

If I use last-touch, I may invest more in channels near conversion.

If I use linear attribution, I spread credit across the funnel.

If I use model-based attribution, I emphasize channels that help predict conversion.

These are different decision frameworks. A practitioner should choose the attribution method based on the business question, not because one method is universally correct.

## 14. Where Attribution Can Mislead

Attribution can mislead when it is interpreted as causal impact.

Common issues:

1. **High-intent users may receive more touches.**  
   Channels in those paths get more credit, even if users were already likely to convert.

2. **Bottom-of-funnel channels may be over-credited.**  
   Last-touch can reward the final interaction even when earlier channels created demand.

3. **Rule-based methods encode assumptions.**  
   First-touch, last-touch, and linear attribution are not discovered truths.

4. **Predictive models still estimate association.**  
   Model-based attribution can be richer, but it still does not answer what would have happened without the channel.

## Final Takeaway

> Attribution explains association, not causation.

It is useful for:

- understanding customer journeys,
- comparing credit-allocation assumptions,
- generating hypotheses,
- guiding further testing.

It is not sufficient for final budget decisions in isolation.

The strongest use of MTA is diagnostic: understand the journey first, then validate major investment decisions with incrementality evidence.
