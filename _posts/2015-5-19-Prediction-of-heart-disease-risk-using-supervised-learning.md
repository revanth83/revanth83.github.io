---
layout: post
Title: Processing the MTA turnstile data!
---

Analysis of the various risk factors associated with heart disease helps health care professionals to identify subjects with high risk of having heart disease. 
The objective of this project is to develop an Intelligent Heart Disease Risk Prediction System that uses the patient's data to perform heart disease risk prediction.

The dataset I looked at is publicly available from the University of California, Irvine machine learning repository; in particular, 4 databases coming from the Hungarian Institute of Cardiology in Budapest, the University Hospitals of Zurich and Basel in Switzerland, as well as the V.A. Medical Center in Long Beach and the Cleveland Clinic Foundation in the USA.

Risk factors associated with heart disease are age, blood pressure, smoking habit, total cholesterol, diabetes, family history of heart disease, etc. I considered the attributes in the following file for this study(http://archive.ics.uci.edu/ml/datasets/Heart+Disease).

To build my prediction model, I used supervised machine learning classifiers such as Logistic Regression, K Nearest Neighbour, Decision Trees, Random Forests, various Naive Baye's implementations as well as Support Vector Machines.

The metrics of the model that I wanted to optimize are Precision and Recall. The Precision is the ratio of people that actually develop heart disease out of those the model says will. A precision of 25% means only quarter of those the model says will develop heart disease actually develop it. We need a high Precision in order to avoid predicting heart disease for healthy people!

The recall is the ratio of those the model says will get heart disease out of those who actually will develop it. It essentially means how successful are we at picking out those who will develop heart disease from the population. We need high Recall in order not to miss any diseased person!

I was able to achieve 80% + Precision and 80%+ Recall with various classification models.

<figure>
   <img src="{{ site.baseurl }}/images/mcnulty_roc_curves_for_various_models.png">
  <figcaption>Fig1. - Receiver Operating Characteristic curves for various models.</figcaption>
</figure>

This project also gave me an opportunity to work with remote relational databases and to visualize my data using D3.js an interactive tool for flexible and attractive presentations of data.

My simplified version of interactive intelligent heart disease risk predictor based on d3 can provide heart disease risk prediction based on a person's age, family history, sex, smoking habits etc. The D3 visualization is available at this link (<iframe src="{{ site.baseurl }}/images/indextest4.html" marginwidth="0" marginheight="0" scrolling="no"></iframe>).

The subject may click on the relevant radio button in the legend to get their heart disease risk profile with respect to age. They can also visualise the comparison between their risk profile and a different subject's profile by selecting relevant radio buttons in the legend.  

<figure>
   <img src="{{ site.baseurl }}/images/static_heart_disease_predictor.png">
  <figcaption>Fig2. - Static version of Heart Disease Risk Predictor</figcaption>
</figure>






