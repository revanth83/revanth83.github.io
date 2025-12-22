
## Dataset (links + download options)

**UCI (official):** https://archive.ics.uci.edu/dataset/222/bank+marketing  
**Kaggle (mirror):** https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

### Recommended
- For **no-login** reproducibility: use UCI (or load via `ucimlrepo`)
- For a quick CSV: download `bank-full.csv` from Kaggle


This post explains **Chapter 4 of Matheus Facure’s _Causal Inference in Python_**
using the **Bank Marketing** dataset.

**Question:** Does contacting customers by **cellular** instead of **telephone**
increase term-deposit subscription?

### Concepts illustrated
- Naive vs adjusted regression
- Month fixed effects
- Frisch–Waugh–Lovell theorem
- Heterogeneous effects
- Omitted variable bias

See the accompanying Jupyter notebook for full code and plots.
