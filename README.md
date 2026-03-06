# Hospital Readmission Analysis

A three-part machine learning analysis of hospital readmission patterns using survival analysis, unsupervised clustering, and logistic regression classification.

---

## Overview

Hospital readmissions — patients returning shortly after discharge — signal gaps in care and carry significant costs for healthcare systems. This project analyzes 25,000 patient records to understand *why* certain patients are readmitted, identify high-risk groups, and build a predictive model.

### Three-Part Analysis

| Part | Method | Goal |
|------|--------|------|
| 1 | **Kaplan-Meier Survival Analysis** | Understand length-of-stay patterns by readmission status |
| 2 | **K-Means Clustering** | Discover distinct patient types and their readmission risk |
| 3 | **Logistic Regression** | Predict whether a patient will be readmitted |

---

## Dataset

25,000 patient records, one row per hospital visit, with 17 features:

| Feature | Description |
|---------|-------------|
| `age` | Age group (e.g. `[70-80)`) |
| `time_in_hospital` | Length of stay in days |
| `n_lab_procedures` | Number of lab tests performed |
| `n_procedures` | Number of procedures during visit |
| `n_medications` | Number of medications administered |
| `n_outpatient` | Prior outpatient visits |
| `n_inpatient` | Prior inpatient visits |
| `n_emergency` | Prior emergency visits |
| `medical_specialty` | Treating department |
| `diag_1/2/3` | Primary and secondary diagnosis categories |
| `glucose_test` | Glucose test result |
| `A1Ctest` | HbA1c test result |
| `change` | Whether medication was changed |
| `diabetes_med` | Whether diabetes medication was prescribed |
| `readmitted` | Target variable — `yes` / `no` |

**Class balance:** ~47% readmitted, ~53% not readmitted — well-balanced for classification.

---

## Part 1 — Survival Analysis: Length of Stay

Kaplan-Meier curves were used to visualize how long patients stay in hospital, comparing readmitted vs. non-readmitted groups.

**Key findings:**
- Most patients are discharged within 1–5 days regardless of readmission outcome
- The KM curves for both groups are nearly identical — length of stay alone is a weak predictor of readmission
- Other features (prior visits, diagnosis) are likely more informative

---

## Part 2 — K-Means Clustering: Patient Types

K-means (k=4, chosen via elbow method) was applied to 7 numeric features to discover natural patient groupings — without using the readmission label.

**Cluster Summary:**

| Cluster | Label | Size | Readmission Rate | Key Characteristics |
|---------|-------|------|-----------------|---------------------|
| 0 | Typical Patients | 10,815 | 46% | Average stay, moderate labs, few prior visits |
| 1 | **Frequent Visitors** | 1,873 | **76%** | Many prior inpatient, outpatient & emergency visits |
| 2 | Quick Procedure Patients | 7,185 | 41% | Short stays, more procedures, minimal prior contact |
| 3 | Complex / Long-Stay | 5,127 | ~50% | Longest stays, most lab work, most medications |

**Key insight:** The "Frequent Visitors" cluster has a 76% readmission rate — nearly double that of other groups — driven by high prior hospital utilization.

---

## Part 3 — Logistic Regression: Readmission Prediction

A logistic regression model was trained on an 70/30 train-test split using all available features (one-hot encoded categoricals included).

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 61.2% |
| ROC-AUC | 0.65 |

**Confusion Matrix (test set, n=7,500):**

|  | Predicted: No | Predicted: Yes |
|--|--------------|----------------|
| **Actual: No** | 3,178 (TN) | 822 (FP) |
| **Actual: Yes** | 2,088 (FN) | 1,412 (TP) |

### Top Predictors

**Increase readmission risk:**
- `n_inpatient` (+0.40) — strongest predictor; more prior inpatient stays = higher risk
- `age [80-90)` (+0.23) — elderly patients at higher risk
- `n_emergency` (+0.22) — more prior emergency visits = higher risk
- `diag_1_Diabetes` (+0.12) — diabetes as primary diagnosis

**Decrease readmission risk:**
- `medical_specialty_Surgery` (−0.27) — surgery patients less likely to be readmitted
- `diag_1_Injury` (−0.15) — injury patients lower risk
- `diag_1_Musculoskeletal` (−0.11)

---

## Overall Conclusions

All three methods converge on the same finding: **prior hospital utilization is the strongest predictor of readmission.**

- Survival analysis showed length of stay is not a meaningful differentiator
- Clustering identified a high-risk "Frequent Visitor" group with 76% readmission rate
- Logistic regression confirmed prior inpatient visits as the top feature

**Clinical recommendation:** Patients with frequent prior hospitalizations should be flagged for intensive discharge planning and follow-up care, as they represent the highest-risk group for readmission.

---

## Tech Stack

- **Python** — pandas, scikit-learn, matplotlib
- **lifelines** — Kaplan-Meier survival analysis
- **scikit-learn** — K-Means clustering, Logistic Regression, model evaluation
