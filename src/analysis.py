# ==========================================================
# CUSTOMER CHURN SURVIVAL ANALYSIS
# Kaplan-Meier + Cox Proportional Hazards
# ==========================================================

# Objective:
# Analyze customer retention using survival analysis.

# Main techniques:
# - Kaplan-Meier estimator
# - Cox proportional hazards model

# Business questions:
# 1. What is the probability customers remain subscribed?
# 2. What is the typical customer lifetime?
# 3. Which factors increase churn risk?

# Survival analysis is especially useful because:
# - it models time-to-event data
# - it handles censored observations
# - it estimates retention probabilities over time

#################################################################################
# ===============================================================================
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import contextlib
import os

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 2. PROJECT PATHS

# This section centralizes all project paths.
# If another user wants to run the project on another machine,
# they only need to modify the paths below.

# ------------------------------------------------------------------------------
# Base project directory

# os.getcwd() returns the current working directory.
# This makes the project more portable.

BASE_DIR = os.getcwd()

# ------------------------------------------------------------------------------
# Results folder

# All plots and exported outputs will be saved here.

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create results directory if it does not exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Dataset path

# Change ONLY this path if the dataset location changes.

DATA_PATH = (
    "/kaggle/input/datasets/blastchar/"
    "telco-customer-churn/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 3. LOAD DATA

# Load the customer churn dataset
# Each row corresponds to one customer

# The dataset contains:
# - customer demographics
# - subscription information
# - churn status
# - subscription duration (tenure)

data = pd.read_csv(DATA_PATH)

# Display first rows
print(data.head())

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 4. DATA CLEANING

# Remove leading/trailing spaces from column names
# This prevents errors when selecting columns later

data.columns = data.columns.str.strip()

# Convert TotalCharges into numeric format

# Some rows contain empty strings instead of numbers.
# errors="coerce" replaces invalid values with NaN
# instead of crashing the script.

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove rows containing missing values

# Survival models require clean numerical inputs.
# Missing values may prevent the model from fitting correctly.

data.dropna(inplace=True)

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 5. CREATE EVENT VARIABLE

# Survival analysis requires:
# 1. a duration variable
# 2. an event indicator

# Here:
# event = customer churn

# event = 1:
# The customer left the company
# (the event occurred)

# event = 0:
# The customer was still subscribed
# at the end of observation
# This is called a "censored observation"

# Censoring means:
# We do not know when churn will happen,
# only that it had not happened yet.

data["event"] = data["Churn"].map({
    "Yes": 1,                               # event happened
    "No": 0                                 # censored observation
})

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 6. DEFINE SURVIVAL VARIABLES

# Duration variable:
# Number of months the customer stayed subscribed

duration = data["tenure"]

# Event variable:
# Indicates whether churn occurred

event_observed = data["event"]

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 7. KAPLAN-MEIER ESTIMATOR

# Kaplan-Meier estimator:
# Estimates the probability that a customer
# remains subscribed over time.

# Unlike simple averages,
# Kaplan-Meier correctly handles censored customers
# (customers who had not churned yet).

kmf = KaplanMeierFitter()

# Fit the survival model
# durations: observed customer lifetimes
# event_observed: indicates whether churn happened

kmf.fit(
    durations=duration,
    event_observed=event_observed,
    label="Customer Survival"
)

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 8. PLOT SURVIVAL CURVE

# Plot the estimated survival function

# The survival curve shows:
# Probability a customer is still active
# after a given number of months.

# ci_show=True displays confidence intervals,
# which represent uncertainty in the estimate.

plt.figure(figsize=(9, 6))

kmf.plot_survival_function(ci_show=True)

plt.title("Kaplan-Meier Survival Curve")
plt.xlabel("Months")
plt.ylabel("Probability of Remaining Customer")

plt.grid(alpha=0.3)

plt.tight_layout()

# Save plot into results folder

plt.savefig(
    os.path.join(RESULTS_DIR, "overall_survival_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Interpretation:
# A steep decline in the curve indicates
# many customers are churning during that period.

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 9. SURVIVAL AFTER 12 MONTHS

# Estimate the probability that a customer
# remains subscribed after 12 months

# This is directly obtained from
# the Kaplan-Meier survival function.

survival_12 = kmf.survival_function_at_times(12).values[0]

print("\nProbability customer survives after 12 months:")
print(f"{survival_12:.2%}")

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 10. MEDIAN CUSTOMER LIFETIME

# Median survival time:
# The time at which the survival probability falls below 50%.

# In business terms:
# The typical customer lifetime.

median_survival = kmf.median_survival_time_

print("\nMedian customer lifetime:")
print(median_survival, "months")

# Save main survival outputs

with open(os.path.join(RESULTS_DIR, "survival_outputs.txt"), "w") as f:

    f.write("CUSTOMER CHURN SURVIVAL ANALYSIS\n")
    f.write("====================================\n\n")

    f.write(
        f"Probability customer survives after 12 months: "
        f"{survival_12:.2%}\n"
    )

    f.write(
        f"Median customer lifetime: "
        f"{median_survival} months\n"
    )

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 11. STRATIFIED SURVIVAL CURVES

# Compare survival curves across contract types

# This helps determine whether
# some customer groups retain better than others.

# Each curve represents a different contract category.

plt.figure(figsize=(9, 6))

for contract in data["Contract"].unique():

    subset = data[data["Contract"] == contract]

    kmf.fit(
        subset["tenure"],
        subset["event"],
        label=contract
    )

    kmf.plot_survival_function(ci_show=False)

plt.title("Survival Curves by Contract Type")
plt.xlabel("Months")
plt.ylabel("Probability of Remaining Customer")

plt.grid(alpha=0.3)

plt.tight_layout()

# Save contract survival curves

plt.savefig(
    os.path.join(RESULTS_DIR, "contract_survival_curves.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Example interpretation:
# If one curve declines faster,
# customers in that group churn more quickly.

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 12. PREPARE DATA FOR COX MODEL

# Select variables potentially associated with churn

# These variables include:
# - demographics
# - billing information
# - subscription characteristics

selected_columns = [
    "SeniorCitizen",
    "MonthlyCharges",
    "TotalCharges",
    "tenure",
    "event",
    "Contract",
    "InternetService",
    "PaymentMethod"
]

cox_data = data[selected_columns].copy()

# Convert categorical variables into numerical format

# Machine learning and statistical models
# cannot directly process text categories.

# pd.get_dummies creates binary indicator variables.

cox_data = pd.get_dummies(
    cox_data,
    drop_first=True
)

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 13. COX PROPORTIONAL HAZARDS

# The Cox model estimates how different variables
# influence churn risk over time.

# It does NOT predict exact churn dates.

# Instead, it estimates relative risk
# through hazard ratios.

cph = CoxPHFitter()

# duration_col: observed customer lifetime
# event_col: whether churn occurred

cph.fit(
    cox_data,
    duration_col="tenure",
    event_col="event"
)

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 14. DISPLAY RESULTS

# exp(coef) = Hazard Ratio (HR)

# HR > 1: Higher churn risk
# HR < 1: Lower churn risk

# Example:
# HR = 1.20
# -> 20% higher churn hazard
# HR = 0.80
# -> 20% lower churn hazard

cph.print_summary()

# Save full Cox regression summary

with open(os.path.join(RESULTS_DIR, "cox_summary.txt"), "w") as f:
    f.write(cph.summary.to_string())

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 15. SIGNIFICANT VARIABLES

# Keep only statistically significant variables

# p < 0.05 suggests the association
# is unlikely due to random chance.

significant = cph.summary[
    cph.summary["p"] < 0.05
]

print("\nSignificant variables:\n")

print(
    significant[
        [
            "exp(coef)",
            "exp(coef) lower 95%",
            "exp(coef) upper 95%",
            "p"
        ]
    ]
)

# Save statistically significant variables

with open(
    os.path.join(RESULTS_DIR, "significant_variables.txt"),
    "w"
) as f:

    f.write(
        significant[
            [
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "p"
            ]
        ].to_string()
    )

# ===============================================================================
#################################################################################

#################################################################################
# ===============================================================================
# 16. CHECK MODEL ASSUMPTIONS

# The Cox model assumes:
# The effect of predictors remains proportional over time.

# This diagnostic checks whether that assumption is violated.

# Example:
# If month-to-month contracts increase churn risk,
# the model assumes this effect remains proportional
# throughout the observation period.

# Violating this assumption may reduce
# model reliability.

with open(
    os.path.join(RESULTS_DIR, "cox_assumptions.txt"),
    "w"
) as f:

    with contextlib.redirect_stdout(f):

        cph.check_assumptions(
            cox_data,
            p_value_threshold=0.05,
            show_plots=False
        )

print("\nCox assumption diagnostics saved.")

# ===============================================================================
#################################################################################
