









# Customer Churn Survival Analysis

- Estimate customer retention over time
- Compare survival across contract types
- Identify churn risk factors
- Analyze proportional hazard assumptions

---

# Dataset

IBM Telco Customer Churn Dataset

Features include:

- customer demographics
- contract information
- internet service
- payment methods
- billing behavior
- churn outcomes

---

# Methods

## Kaplan–Meier Estimator

Used to estimate customer survival probabilities over time.

## Cox Proportional Hazards Model

Used to evaluate the effect of customer characteristics on churn risk.

---

## Dataset & Kaggle Notebook

This work was first developed and tested in a Kaggle environment.

👉 **Kaggle project:** https://www.kaggle.com/code/chocosof/customer-churn-survival-analysis

---

# Results

## 12-Month Survival Probability

84.32%

## Major Churn Drivers

Higher churn risk:
- higher monthly charges
- fiber optic internet
- electronic check payments

Lower churn risk:
- one-year contracts
- two-year contracts

---

## Repository Structure

```text
  
.
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   └── analysis.py
│
├── results/
│   ├── overall_survival_curve.png
│   ├── contract_survival_curves.png
│   ├── cox_summary.txt
│   ├── significant_variables.txt
│   ├── cox_assumptions.txt
│   └── survival_outputs.txt
│
├─── reports/
│   └── survival_analysis_report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md