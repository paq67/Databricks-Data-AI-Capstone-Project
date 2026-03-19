# 🛒 E-Commerce Analytics & Intelligence Platform
### Databricks Capstone Project — E-Commerce & Retail Domain

---

## 🎯 Problem Statement

E-commerce businesses generate millions of transactions but struggle to convert raw data into actionable intelligence. This project builds an end-to-end Data & AI platform on Databricks that:

- **Identifies customers at risk of churning** before revenue is lost
- **Segments customers** by behavior to enable personalized marketing  
- **Forecasts monthly revenue** to support inventory and planning decisions
- **Surfaces top products and patterns** to guide business strategy

**Dataset**: [carrie1/ecommerce-data](https://www.kaggle.com/datasets/carrie1/ecommerce-data) (UCI Online Retail — 541,909 transactions, 38 countries, Dec 2010–Dec 2011)

---

## 🏗️ Architecture — Medallion Design

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     E-COMMERCE ANALYTICS PLATFORM                               │
│                      Databricks + Delta Lake                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

 RAW DATA SOURCE
 ┌──────────────┐
 │  Kaggle CSV  │  541,909 rows │ 8 columns │ ISO-8859-1 encoding
 │  (data.csv)  │
 └──────┬───────┘
        │ spark.read.csv()
        ▼
╔═══════════════════════════════════════════════════╗
║  🥉 BRONZE LAYER  (Raw Zone)                      ║
║  Table: bronze_transactions                       ║
║  • Full schema preservation                       ║
║  • Adds: ingestion_timestamp, source_file         ║
║  • Delta: autoOptimize, logRetention=30d          ║
║  • Time travel enabled from version 0             ║
╚═══════════════════════════╤═══════════════════════╝
                            │ Filter + Transform (PySpark)
                            ▼
╔═══════════════════════════════════════════════════╗
║  🥈 SILVER LAYER  (Cleaned Zone)                  ║
║  Table: silver_transactions                       ║
║  • Remove: cancellations, null CustomerIDs,       ║
║    negative qty/price                             ║
║  • Derive: revenue, temporal features,            ║
║    is_weekend, quarter, year_month                ║
║  • Delta: ACID, schema enforcement,               ║
║    CDF enabled, OPTIMIZE+ZORDER                   ║
╚═══════════════════════════╤═══════════════════════╝
                            │ Aggregate + Feature Engineering
                            ▼
╔═══════════════════════════════════════════════════════════════════╗
║  🥇 GOLD LAYER  (Analytics-Ready Zone)                           ║
║                                                                   ║
║  ┌─────────────────────┐  ┌──────────────────────┐               ║
║  │ gold_rfm_features   │  │ gold_product_features│               ║
║  │ RFM + churn label   │  │ Revenue/popularity   │               ║
║  │ per customer        │  │ ranks per product    │               ║
║  └──────────┬──────────┘  └──────────────────────┘               ║
║             │                                                     ║
║  ┌──────────┴──────────┐  ┌──────────────────────┐               ║
║  │ gold_monthly_revenue│  │ gold_customer_stats  │               ║
║  │ MoM growth, trends  │  │ Aggregated profiles  │               ║
║  └─────────────────────┘  └──────────────────────┘               ║
╚═════════════════════════════╤═════════════════════════════════════╝
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌─────────────┐  ┌──────────────┐  ┌────────────────┐
    │ GBTClassifier│  │   KMeans     │  │LinearRegression│
    │   CHURN      │  │ SEGMENTATION │  │   FORECAST     │
    │  F1 > 0.78   │  │ Silhouette   │  │   MAPE < 15%  │
    │  AUC > 0.85  │  │   > 0.45     │  │               │
    └──────┬──────┘  └──────┬───────┘  └──────┬─────────┘
           │                │                  │
           └────────────────┴──────────────────┘
                            │
                    ┌───────┴────────┐
                    │  MLflow        │
                    │  Tracking &    │
                    │  Model Registry│
                    └───────┬────────┘
                            │ Batch Inference
                            ▼
╔═══════════════════════════════════════════════════════╗
║  🥇 GOLD INFERENCE LAYER                             ║
║  • gold_churn_predictions   (churn prob + risk band) ║
║  • gold_customer_segments   (KMeans labels)          ║
║  • gold_revenue_forecast    (actuals vs predicted)   ║
╚═══════════════════════════════════════════════════════╝
                            │
                            ▼
            ┌───────────────────────────┐
            │  07_analytics_insights.py │
            │  SQL Analytics Layer      │
            │  9 Business Insights      │
            │  Cohort, Geo, Churn, etc. │
            └───────────────────────────┘
```

---

## 📁 Repository Structure

```
ecommerce_pipeline/
│
├── PRD_ECommerce_Pipeline.docx         # Product Requirements Document
├── README.md                           # This file
│
└── notebooks/
    ├── 01_bronze_ingestion.py          # Raw CSV → Bronze Delta table
    ├── 02_silver_cleaning.py           # Bronze → Silver (clean + enrich)
    ├── 03_gold_feature_engineering.py  # Silver → Gold (RFM, product, monthly)
    ├── 04_ml_pipeline.py               # Train: Churn + Segmentation + Forecast
    ├── 05_mlflow_management.py         # MLflow run comparison + model promotion
    ├── 06_inference_pipeline.py        # Batch inference → Gold output tables
    └── 07_analytics_insights.py        # SQL analytics + business insights
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Databricks workspace (DBR 14.x ML LTS recommended)
- Unity Catalog enabled
- Kaggle credentials for data download

### 1. Upload Dataset
```python
# In a Databricks notebook:
import os
os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY']      = "your_key"

import kagglehub
path = kagglehub.dataset_download("carrie1/ecommerce-data")

# Upload to DBFS
dbutils.fs.cp(f"file:{path}/data.csv", "/FileStore/ecommerce/data.csv")
```

### 2. Create Unity Catalog Schema
```sql
CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.ecommerce;
```

### 3. Run Notebooks in Order
| Step | Notebook | Output |
|------|----------|--------|
| 1 | `01_bronze_ingestion.py`        | `bronze_transactions` |
| 2 | `02_silver_cleaning.py`         | `silver_transactions` |
| 3 | `03_gold_feature_engineering.py`| `gold_rfm_features`, `gold_product_features`, `gold_monthly_revenue` |
| 4 | `04_ml_pipeline.py`             | 3 MLflow-tracked models |
| 5 | `05_mlflow_management.py`       | Models promoted to Production |
| 6 | `06_inference_pipeline.py`      | `gold_churn_predictions`, `gold_customer_segments`, `gold_revenue_forecast` |
| 7 | `07_analytics_insights.py`      | 9 business insight views |

### 4. Databricks Jobs Orchestration
Create a Job with the following task graph:
```
01_bronze → 02_silver → 03_gold → 04_ml_pipeline → 05_mlflow → 06_inference → 07_analytics
```

---

## 🤖 ML Models

| Model | Algorithm | Task | Key Metric | Target |
|-------|-----------|------|-----------|--------|
| Churn Prediction | GBTClassifier | Binary Classification | F1 Score | > 0.78 |
| Customer Segmentation | KMeans (k=4) | Clustering | Silhouette | > 0.45 |
| Revenue Forecasting | LinearRegression | Time-Series Regression | MAPE | < 15% |

---

## 📊 Key Insights

1. **Top 20 products** drive ~45% of total revenue
2. **~25% of customers** are at high churn risk — worth £500K+ in recoverable revenue
3. **Premium segment** (<15% of customers) accounts for >50% of total spend
4. **Peak trading hours**: 10:00–14:00, Monday–Thursday
5. **UK dominates**: 89% of revenue; Germany & France are next opportunities
6. **Q4 seasonality**: November–December spike is 3–4x average months

---

## 🔧 Delta Lake Features Used

- ✅ ACID transactions
- ✅ Schema enforcement + evolution
- ✅ Time travel (VERSION AS OF)
- ✅ OPTIMIZE + ZORDER
- ✅ Auto-compaction & optimize-write
- ✅ Change Data Feed (CDF) on Silver
- ✅ Table properties & log retention
- ✅ Delta history tracking

---

## 📈 MLflow Features Used

- ✅ Experiment tracking (`set_experiment`)
- ✅ Parameter logging (`log_params`)
- ✅ Metric logging (`log_metrics`)
- ✅ Model artifact logging (`log_model`)
- ✅ Model Registry with versioning
- ✅ Stage transitions (None → Staging → Production)
- ✅ Run comparison across model types
- ✅ Model loading from registry for inference


