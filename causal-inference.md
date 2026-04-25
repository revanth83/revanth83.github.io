---
layout: page
title: Causal Inference in Practice
permalink: /causal-inference/
---
## Overview
This page organizes my work on causal inference — from foundational concepts to applied modeling in marketing, experimentation, and decisioning.

The focus is on **estimating causal effects, understanding heterogeneity, and making better decisions under uncertainty**.

---

## Foundations: From Association to Causation

Core concepts behind causal reasoning and why correlation is not enough.

- [Association vs Causation: A Minimal Potential Outcomes Demo]({% post_url 2025-11-23-potential-outcomes-association-vs-causation-aligned %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/potential_outcomes_code_frozen_aligned.ipynb))

- [Causal Graphs, Confounding, Colliders, and Selection Bias]({% post_url 2025-12-8-causal_graphs_full_tutorial_networkx %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/causal_graphs_networkx_code_frozen_aligned.ipynb))

---

## Randomized Experiments

Understanding causal effects through controlled experiments.

- [Email Subject Line Experiment: Randomized Experiments and Statistical Review]({% post_url 2025-12-06-email-subjectline-experiment %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/Randomized_email_experiment_code_frozen_aligned.ipynb))

---

## Observational Causal Inference

Estimating causal effects when randomization is not available.

- [Propensity Scores in Practice: IPW and Doubly Robust Estimation]({% post_url 2025-12-25-inverse_propensity_weighting_&_doubly_robust_estimation %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/propensity_scores_practice_aligned.ipynb))

- [Bank Marketing — Causal Linear Regression]({% post_url 2025-12-22-bank_marketing_causal_regression %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/bank_marketing_causal_regression_code_frozen_aligned.ipynb))

---

## Heterogeneous Treatment Effects (CATE & Uplift)

Understanding **who responds**, not just whether something works.

- [Finding Who Actually Responds: CATE-Based Targeting]({% post_url 2025-12-27-CATE_Marketing_Credit_Case_Study %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/CATE_Based-targeting_Credit_CodeFrozen_Aligned.ipynb))

- [Meta-Learners for Heterogeneous Treatment Effects]({% post_url 2026-2-1-Meta-learners_for-Heterogenous_Treatment_Effect_Estimation %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/meta_learners_randhie_code_frozen_blog_aligned.ipynb))

- [Advanced CATE Estimation Methods]({% post_url 2026-04-04-Advanced_CATE_Estimation_approaches %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/advanced_CATE_public_dataset_blog_notebook_code_frozen_aligned.ipynb))

---

## Bayesian & Decision-Focused Causal Modeling

Incorporating uncertainty into decision-making.

- [Bayesian Models for Campaign Decisioning: Handling Uncertainty in Real Data]({% post_url 2026-04-12-Bayesian_campaign_decisioning_hillstrom_data %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/Bayesian_campaign_decisioning_hillstrom_data_aligned.ipynb))

- [Double Machine Learning Finds Segments, Bayesian Decides Which Ones to Trust]({% post_url 2026-04-17-dml-vs-bayesian-uplift-criteo %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/dml-vs-bayesian-uplift-criteo_blog_aligned.ipynb))

---

## Structural Causal Models

Moving from treatment-effect estimation toward mechanism-level reasoning and counterfactual analysis.

- [Building Structural Causal Models: An End-to-End Workflow with DoWhy, EconML, and Refutation Tests]({% post_url 2026-1-31-Building_Structural-Causal_Models_with_dowhy_&_econml_gcm %})  
  ([Code](https://github.com/revanth83/causal_inference_code_notebooks/blob/main/end_to_end_dowhy_econml_gcm_blog_aligned.ipynb))

**Key ideas:**
- Moving beyond average effect estimation  
- Making assumptions explicit with causal graphs  
- Using refutation and counterfactual reasoning  
- Connecting estimation to intervention and policy analysis  

---

## How to Navigate

If you're new to causal inference:

1. Start with **Association vs Causation**
2. Move to **Experiments**
3. Then explore **Observational methods**
4. Finally, go into **CATE and Bayesian decisioning**

---

## Why This Matters

In real-world systems — marketing, product, and credit — decisions are not made on predictions alone.

They require understanding:

- What **causes** outcomes  
- What works **for whom**  
- How **uncertain** those effects are  

This collection reflects that journey from **theory → estimation → decisioning**.
