# phishing-detection-generalization
Evaluating the generalizability of machine learning-based phishing email detection across multiple email sources.

# Phishing Email Generalization

## Project Overview

This project is being completed for IS 665: Cybersecurity Analytics at California State University, Long Beach.

The goal of this project is to evaluate how well machine learning-based phishing email detection models generalize to email data from sources that were not included during training. Rather than only evaluating models using a traditional random train/test split, we will examine how model performance changes when duplicate leakage and source-specific patterns are controlled.

## Research Question

**How well do machine-learning phishing email detection models generalize to emails from unseen data sources when source-specific patterns and duplicate leakage are controlled?**

## Dataset

The primary dataset is the Phishing Email Dataset by Naser Abdullah Alam on Kaggle. It combines email data from multiple sources, including:

- Enron
- Ling
- CEAS
- Nazario
- Nigerian Fraud
- SpamAssassin

Before modeling, we will review the labels and documentation for each source to determine which records can appropriately be treated as phishing, spam, or legitimate.

## Project Plan

The project will include:

- Dataset and label audit
- Exploratory data analysis
- Exact and near-duplicate analysis
- TF-IDF text representation
- Baseline machine learning models
- Traditional random-split evaluation
- Source-disjoint evaluation
- Precision, recall, F1, and PR-AUC evaluation
- False-positive and false-negative analysis
- External dataset evaluation, if feasible

## Team

**Team 8**

- Katherine Hernandez
- Jennifer Lopez

## Course

IS 665 – Cybersecurity Analytics  
California State University, Long Beach  
Fall 2026
Thank you!