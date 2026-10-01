# Research Report Outline

## Hide Personal Data and Still Use It

### An Empirical Research Investigation on Data Anonymization Techniques: Evaluating the Privacy-Utility Trade-off in ML Models

---

## 1. Abstract

Briefly describe:

- The problem of using personal data while protecting privacy.
- The purpose of data anonymization.
- The anonymization techniques investigated.
- The machine learning models used.
- The privacy-utility trade-off being evaluated.
- The major findings of the study.

**To be completed after experiments.**

---

## 2. Introduction

### 2.1 Background

Explain the importance of data in modern machine learning and the privacy concerns associated with personal information.

### 2.2 Problem Statement

Organizations need useful data for analysis and machine learning while reducing the risk of exposing personal information.

### 2.3 Motivation

Explain why anonymization techniques are important and why their effect on machine learning performance needs to be measured empirically.

### 2.4 Research Question

> How does increasing the level of data anonymization affect the privacy protection and machine learning utility of a dataset?

---

## 3. Objectives

1. Apply anonymization techniques to a publicly available dataset.
2. Understand and implement k-anonymity.
3. Establish baseline machine learning performance.
4. Apply Low, Medium and High anonymization levels.
5. Compare model performance across anonymization levels.
6. Analyze the privacy-utility trade-off.
7. Discuss re-identification risks and limitations of anonymization.
8. Develop a Streamlit-based anonymization tool.

---

## 4. Literature Review

Review existing research related to:

### 4.1 Data Privacy

- Personal data protection
- Privacy risks in data analysis
- Responsible data practices

### 4.2 Data Anonymization

- Generalization
- Suppression
- Masking
- Data perturbation

### 4.3 K-Anonymity

Explain the concept, purpose, strengths and limitations of k-anonymity.

### 4.4 Re-identification

Discuss how supposedly anonymized datasets can sometimes be linked with external information.

### 4.5 Previous Research

Review relevant academic studies and documented re-identification cases.

---

## 5. Dataset

### 5.1 Dataset Description

Document:

- Dataset name
- Dataset source
- Number of records
- Number of features
- Target variable
- Data types

### 5.2 Dataset Selection Criteria

Explain why the dataset was selected.

### 5.3 Sensitive Attributes

Identify attributes that may contain or represent sensitive information.

### 5.4 Quasi-identifiers

Identify attributes that could potentially contribute to re-identification when combined.

### 5.5 Data Preparation

Document:

- Missing-value handling
- Duplicate handling
- Encoding
- Feature selection
- Train-test split
- Other preprocessing steps

---

## 6. Methodology

### 6.1 Experimental Design

Describe the overall experimental workflow.

```text
Original Dataset
       ↓
EDA and Preprocessing
       ↓
Baseline Models
       ↓
Anonymization
       ↓
Level 1
       ↓
Level 2
       ↓
Level 3
       ↓
Retrain Models
       ↓
Compare Results