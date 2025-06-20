# Insurance-risk-score-data-science-project

# Project: Insurance Risk Scoring for Claim Prediction
# Objective
To build a risk scoring system to assist insurance underwriters in predicting the likelihood that a client will file a claim. This model aims to:

Identify high-risk clients,

Inform premium and policy decisions,

Minimize financial exposure from high-risk policyholders.

The solution follows the CRISP-DM methodology.

# CRISP-DM Process Breakdown
## 1. Business Understanding
Objective: Enable underwriters to assess and manage client risk using data-driven methods.

Success Metric: High precision and recall on claim prediction, especially F1-score to balance false positives/negatives.

## 2. Data Understanding
A synthetic dataset with 10,100 rows and 40+ features was generated to simulate real-world insurance data.

Features include:

Numerical: age, income, credit score, savings, etc.

Categorical: marital_status, policy_type, vehicle_type, etc.

Ordinal/Binary: health_status, home_ownership, criminal record, etc.

Target variable: claim_status (1 = Claim Filed, 0 = No Claim)

## 3. Data Preprocessing
Handled missing values (~1%) using imputation.

Encoded categorical variables using:

OrdinalEncoder for ordinal fields

OneHotEncoder for nominal fields

Scaled numerical features with MinMaxScaler.

Removed duplicates and outliers.

## 4. Feature Engineering
Created 40+ original and derived features, including engineered behavior metrics.

Applied RFECV (Recursive Feature Elimination with Cross-Validation) using RandomForestClassifier to select the most relevant features.

Optimal Features Chosen: dynamically selected (based on maximizing F1-score).

## Modeling Phase
Models Compared:
Logistic Regression

Random Forest Classifier

XGBoost Classifier

Decision Tree Classifier

## Hyperparameter Tuning:
Conducted using Optuna for efficient search.

Custom scoring with f1_weighted and roc_auc.

## Evaluation Metrics:
Accuracy

Precision, Recall

F1 Score

ROC-AUC Score

Confusion Matrix

## Final Model Performance
Model	Accuracy	F1 Score	ROC-AUC
Random Forest (Tuned)	~88.3%	0.88	~0.90
XGBoost	~87.9%	0.87	~0.89
Logistic Regression	~85.2%	0.85	~0.86
Decision Tree	~83.7%	0.83	~0.84

### The Random Forest model had the best overall balance of accuracy and F1 score.

# Visualization Summary:
Plots of:

Class distribution and imbalance

Feature correlation heatmap

RFECV feature selection curve

Confusion matrices and ROC curves per model

# Summary and Next Steps
Key Insights:
Credit Score, Driving Record, and Health Status were strong predictors of claim risk.

Feature selection with RFECV improved model robustness by reducing overfitting.

Ensemble models performed better than linear models.

# Deployment Consideration:
This model can be deployed via API or embedded into underwriting dashboards.

It should be retrained periodically as new data accumulates.
