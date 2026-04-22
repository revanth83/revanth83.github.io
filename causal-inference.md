---
layout: page
title: Causal Inference
permalink: /causal-inference/
---

## Causal Inference in Practice

This page organizes my work on causal inference — from foundational concepts to applied modeling in marketing, experimentation, and decisioning.

The focus is on **estimating causal effects, understanding heterogeneity, and making better decisions under uncertainty**.

---

## Foundations: From Association to Causation

Core concepts behind causal reasoning and why correlation is not enough.

- [Association vs Causation: A Minimal Potential Outcomes Demo]({% post_url 2025-11-23-potential-outcomes-association-vs-causation %})
- [Causal Graphs, Confounding, Colliders, and Selection Bias]({% post_url 2025-12-8-causal_graphs_full_tutorial_networkx %})

---

## Randomized Experiments

Understanding causal effects through controlled experiments.

- [Email Subject Line Experiment: Randomized Experiments and Statistical Review]({% post_url 2025-12-06-email-subjectline-experiment %})

---

## Observational Causal Inference

Estimating causal effects when randomization is not available.

- [Propensity Scores in Practice: IPW and Doubly Robust Estimation]({% post_url 2025-12-25-inverse_propensity_weighting_&_doubly_robust_estimation %})
- [Bank Marketing — Causal Linear Regression]({% post_url 2025-12-22-bank_marketing_causal_regression %})

---

## Heterogeneous Treatment Effects (CATE & Uplift)

Understanding **who responds**, not just whether something works.

- [Finding Who Actually Responds: CATE-Based Targeting]({% post_url 2025-12-27-CATE_Marketing_Credit_Case_Study %})
- [Meta-Learners for Heterogeneous Treatment Effects]({% post_url 2026-2-1-Meta-learners_for-Heterogenous_Treatment_Effect_Estimation %})
- [Advanced CATE Estimation Methods]({% post_url 2026-04-04-Advanced_CATE_Estimation_approaches %})

---

## Bayesian & Decision-Focused Causal Modeling

Incorporating uncertainty into decision-making.

- [Bayesian Models for Campaign Decisioning: Handling Uncertainty in Real Data]({% post_url 2026-04-12-Bayesian_campaign_decisioning_hillstrom_data %})
- [Double Machine Learning Finds Segments, Bayesian Decides Which Ones to Trust]({% post_url 2026-04-17-dml-vs-bayesian-uplift-criteo %})

---
---

## Structural Causal Models

Moving from treatment-effect estimation toward mechanism-level reasoning and counterfactual analysis.

- [Building Structural Causal Models: An End-to-End Workflow with DoWhy, EconML, and Refutation Tests]({% post_url 2026-1-31-Building_Structural-Causal_Models_with_dowhy_&_ecoml_gcm %})

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
