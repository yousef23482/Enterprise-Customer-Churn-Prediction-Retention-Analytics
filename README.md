# Enterprise Customer Churn Prediction & Retention Analytics

## 📌 Project Overview
Customer retention is one of the highest-leverage drivers of recurring revenue in SaaS and subscription-based businesses. This end-to-end data analytics and machine learning project simulates a real-world enterprise environment—from relational database architecture and SQL feature extraction to predictive modeling and business threshold optimization.

The goal of this project is to identify at-risk customers *before* they cancel their subscriptions and uncover the primary behavioral drivers behind customer churn.

---

## 🎯 Business Problem
Acquiring a new customer is significantly more expensive than retaining an existing one. However, traditional reporting often looks backward at customers who have already left. 

To solve this proactively, this pipeline answers two key executive questions:
1. **Who** is at high risk of churning in the near future?
2. **Why** are they leaving, and where should customer success teams focus their retention efforts?

---

## 🗄️ Data Architecture & SQL Engineering
Instead of working with a single flat CSV file, this project models a normalized relational database (`enterprise_churn.db`) built in **SQLite3** containing four core tables linked via `customer_id` foreign keys:
* `customers`: Account demographics, subscription tiers, monthly charges, signup dates, and churn status.
* `transactions`: Historical billing and payment records.
* `support_tickets`: Customer service interactions and resolution metrics.
* `usage_logs`: Product engagement and active session minutes.

### Solving the SQL Fan-Out Problem
Joining multiple one-to-many tables directly creates Cartesian product ("fan-out") errors that inflate counts and sums. To extract a clean, single-row-per-customer feature matrix, the SQL pipeline utilizes **Common Table Expressions (CTEs)** to pre-aggregate behavioral metrics independently before joining, alongside `COALESCE` to safely handle missing values (`NULL` to `0`) for inactive accounts.

---

## ⚙️ Data Wrangling & Feature Engineering
Using **Python (`pandas`, `numpy`)**, raw database outputs were transformed into a model-ready dataset:
* **Customer Tenure:** Calculated exact account age in days from `signup_date` to measure longevity.
* **Spend Efficiency:** Engineered `avg_spend_per_transaction` while guarding against division-by-zero edge cases for zero-transaction users.
* **Categorical Encoding:** Applied **Ordinal Encoding** (`Basic` → `1`, `Pro` → `2`, `Enterprise` → `3`) to preserve tier hierarchy, and **One-Hot Encoding** (`drop_first=True`) to transform `country` categories without introducing multicollinearity.
* **Data Hygiene:** Stripped non-predictive unique identifiers (`customer_id`) to prevent model overfitting.

---

## 🤖 Machine Learning & Precision-Recall Optimization
A **Random Forest Classifier** (`scikit-learn`) was trained on an 80/20 train-test split to predict customer churn (`1` = Churned, `0` = Active).

### Addressing Class Imbalance & Threshold Tuning
Because churned customers represent the minority class (25% of the test set), a standard model using the default **50% probability threshold** achieved 73% overall accuracy but caught only **20% of actual churners** (Recall = `0.20`).

From a business standpoint, missing an at-risk customer (False Negative) results in permanent recurring revenue loss, which is far costlier than offering a retention incentive to a safe customer (False Positive). By extracting raw predicted probabilities (`predict_proba`) and lowering the decision threshold to **30% (`0.30`)**, the model's performance shifted to align with business priorities:

| Metric (Churned Class `1`) | Default Threshold (50%) | Tuned Threshold (30%) | Business Impact |
| :--- | :---: | :---: | :--- |
| **Recall** | 0.20 (20%) | **0.64 (64%)** | **+44% increase** in catching at-risk customers |
| **Precision** | 0.42 (42%) | **0.41 (41%)** | Maintained steady signal quality while casting a wider net |
| **F1-Score** | 0.27 | **0.50** | Nearly doubled overall minority-class detection balance |

---

## 📊 Key Business Insights (Feature Importance)
Extracting feature importances from the trained Random Forest revealed clear behavioral patterns:

1. **Engagement & Spend Dominate (~14%–18% importance each):** `total_amount_spent` (`18.3%`), `avg_active_minutes` (`17.7%`), `avg_spend_per_transaction` (`17.2%`), `monthly_charges` (`15.6%`), and `customer_tenure_days` (`14.4%`) are the top predictors of churn.
2. **Geography & Tier Are Minimal Drivers (< 4.5% each):** `subscription_tier` (`4.2%`) and customer location (`country_UK`, `country_UAE`, `country_USA`, `country_Germany` at ~1.3%–1.5% each) have negligible impact on churn likelihood.

### Strategic Recommendations
* **Trigger Automated Usage Alerts:** Monitor rolling `avg_active_minutes` rather than segmenting retention campaigns by country. Users whose activity drops below baseline thresholds should be flagged immediately for customer success outreach.
* **Early-Tenure Onboarding:** Because tenure and cumulative spend heavily influence retention, focus onboarding incentives on driving early product adoption during a customer's first 60–90 days.

---

## 🛠️ Tech Stack & Repository Structure
* **Languages & Libraries:** Python, SQL, Pandas, NumPy, Scikit-Learn, SQLite3
* **Environment:** Google Colab / Jupyter Notebook

```text
├── churn_analysis.ipynb      # End-to-end SQL, ETL, Feature Engineering & ML pipeline
├── enterprise_churn.db       # Normalized relational SQLite database
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation and executive summary
