# Starbucks Captone Machine Learning

Introduction
This is the final project for the Udacity Machine Learning Engineer Nanodegree Program using AWS SageMaker

The the project contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. The goal of the project is to analyze this "historical" data in order to develop an algorithm that finds the most suiting offer type for each customer.

The data is contained in three files:

- portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
- profile.json - demographic data for each customer
- transcript.json - records for transactions, offers received, offers viewed, and offers completed

## Solution Statement

This is a classification problem, my approach is going to be: create a Machine learning model to
predict the best offer for the user, for BOGO or discount (we are going to leave out the
informational offers, which have no real “conversion”)
There are the steps to follow:

- Fetching data
- Data Cleaning
- Data Preparation
- Data visualization and analysis
- Train Model
- Evaluate the model

## Evaluation Metrics

Since this is a classification problem we can use the following metrics to evaluate the model:

- Precision The proportion of positive cases that were correctly identified.
- Recall The proportion of actual positive cases which are correctly identified.
- F1-score, that combines the two previous measures.
- Roc_auc_score Area Under the ROC Curve (AUC), a measure that calculates the area
under the Receiving Operating Characteristic Curve. This particular curve accounts that
higher probabilities are associated with true positives and vice-versa.

## Algoritms

We explored two algoritms `Linear Learner` and `XGBoost` to find the best models for for each offer, a Discount, a Buy One Get One (BOGO)
or Informational.

For more information you can refer to the `proposal.pdf` file included in this repository