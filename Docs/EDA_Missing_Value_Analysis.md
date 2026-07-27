# Exploratory Data Analysis (EDA) Report
## Missing Value Analysis

**Project:** Delinquency Prediction Model  
**Analysis Stage:** Data Quality Assessment

---

# 1. Objective

The objective of this analysis is to identify missing values present in the delinquency prediction dataset, understand their distribution, analyze possible patterns behind missing information, and define appropriate preprocessing strategies before model development.

Missing value analysis is important in credit risk modelling because missing financial information may itself represent customer behavior or risk patterns.

---

# 2. Dataset Missing Value Summary

The dataset contains the following missing values:

| Feature | Missing Count | Status |
|---|---:|---|
| Customer_ID | 0 | Complete |
| Age | 0 | Complete |
| Income | 39 | Requires Imputation |
| Credit_Score | 2 | Requires Imputation |
| Credit_Utilization | 0 | Complete |
| Missed_Payments | 0 | Complete |
| Delinquent_Account | 0 | Complete |
| Loan_Balance | 29 | Requires Imputation |
| Debt_to_Income_Ratio | 0 | Complete |
| Employment_Status | 0 | Complete |
| Account_Tenure | 0 | Complete |
| Credit_Card_Type | 0 | Complete |
| Location | 0 | Complete |
| Month_1 | 0 | Complete |
| Month_2 | 0 | Complete |
| Month_3 | 0 | Complete |
| Month_4 | 0 | Complete |
| Month_5 | 0 | Complete |
| Month_6 | 0 | Complete |

---

# 3. Missing Value Distribution

The dataset contains missing values only in three numerical features:

- **Income:** 39 missing records
- **Loan_Balance:** 29 missing records
- **Credit_Score:** 2 missing records

No missing values were found in:
- Customer identification
- Demographic information
- Payment history variables
- Target variable

---

# 4. Income Missing Value Analysis

## Overview

Income contains the highest number of missing values: 39


A missing income indicator feature was created:

This allows the machine learning model to learn whether missing income itself is associated with delinquency risk.

---

# 5. Customers With Missing Income

The missing income records belong to different customer profiles with varying:

- Age
- Credit Score
- Payment behavior
- Employment status
- Credit card type

Example observations:

| Customer_ID | Age | Credit Score | Month_4 | Month_5 | Month_6 |
|---|---:|---:|---|---|---|
| CUST0041 | 61 | 372 | On-time | Late | Late |
| CUST0043 | 69 | 792 | Late | Missed | Missed |
| CUST0060 | 19 | 538 | Late | Missed | On-time |
| CUST0291 | 36 | 836 | Late | Late | Late |
| CUST0495 | 32 | 811 | On-time | On-time | On-time |

The missing income group contains customers with both:

- Healthy payment behavior
- High-risk payment behavior

Therefore, missing income should not automatically be considered a negative indicator.

---

# 6. Missing Income by Employment Status

Analysis of missing income distribution across employment categories:

| Employment Status | Missing Income Rate |
|---|---:|
| EMP | 6.17% |
| Employed | 8.54% |
| Self-employed | 3.75% |
| Unemployed | 9.68% |
| employed | 11.69% |
| retired | 6.90% |

## Observation

Employment status contains inconsistent category naming:

Examples:
EMP
Employed
employed


These categories represent the same group and require standardization before modelling.

Recommended transformation:
EMP → Employed
employed → Employed


---

# 7. Missing Income by Credit Card Type

| Credit Card Type | Missing Income Rate |
|---|---:|
| Business | 7.41% |
| Gold | 5.08% |
| Platinum | 10.53% |
| Standard | 9.30% |
| Student | 8.04% |

## Observation

Missing income occurs across all credit card categories.

Platinum and Standard card holders show slightly higher missing income rates, but the difference is not large enough to indicate a strong relationship.

---

# 8. Remaining Missing Income After Group Imputation

After applying employment-based median imputation: 5


These remaining values occur because some employment groups may not contain enough valid income values for calculating a reliable median.

Fallback strategy: Filling remaining missing values using overall Income median


---

# 9. Missing Credit Score Analysis

Missing values: 2


Since only two records are missing, the impact is minimal.

Recommended approach:

- Create missing indicator: Credit_Score_missing()


- Fill using median credit score.

Reason:

Median preserves the distribution and avoids influence from extreme credit scores.

---

# 10. Missing Loan Balance Analysis

Missing values: 29


Loan balance is an important financial feature.

Recommended approach:

Create indicator: Loan_Balance_missing()

Then apply: Median Imputation


Further investigation should check whether missing loan balance is associated with:

- Delinquency status
- Credit utilization
- Debt-to-income ratio

---

# 11. Missing Value Handling Strategy

| Feature | Strategy |
|---|---|
| Income | Employment-based median imputation |
| Remaining Income Missing | Overall median fallback |
| Credit_Score | Median imputation |
| Loan_Balance | Median imputation |
| Missing Information | Preserve using indicator variables |

---

# 12. Key Findings

1. The dataset has limited missing values and is suitable for machine learning modelling.

2. Income contains the highest missing percentage and requires careful handling.

3. Missing income appears across different employment and credit card categories.

4. Missing values should not simply be removed because they may contain predictive information.

5. Missing indicators will be retained as additional model features.

6. Employment categories require cleaning before encoding.

---

# 13. Next Steps

Following missing value treatment:

1. Perform categorical data cleaning.
2. Analyze target variable distribution.
3. Conduct univariate and bivariate analysis.
4. Perform feature engineering.
5. Encode categorical variables.
6. Train baseline classification models.
7. Optimize recall for delinquent customers.
8. Apply SHAP explainability.

---

## Conclusion

The missing value analysis confirms that the dataset quality is acceptable. Proper imputation combined with missing-value indicators will preserve important customer behavior signals and improve the reliability of the delinquency prediction model.








