# FMCG Demand Forecasting & Inventory Intelligence Platform

An end-to-end machine learning platform for **retail demand forecasting and inventory decision support**, built using Python, LightGBM, FastAPI, Streamlit, Docker, MLflow, and a medallion-style data architecture.

The project demonstrates how raw retail transaction data can be transformed into production-ready features, used to train forecasting models, exposed through an API, and consumed by an interactive inventory planning dashboard.

---

## Project Overview

Retailers need accurate demand forecasts to balance two competing risks:

- **Stockouts** — insufficient inventory to satisfy customer demand
- **Overstocking** — excessive inventory that increases holding costs and waste

This project builds an end-to-end forecasting system that predicts future product demand at the **store-product-family level** and converts those forecasts into inventory planning insights.

The system covers the complete ML lifecycle:

```text
Raw Data
   ↓
Bronze Layer
   ↓
Silver Layer
   ↓
Gold Layer
   ↓
Feature Engineering
   ↓
Forecasting Models
   ↓
Model Evaluation
   ↓
Forecast Service
   ↓
FastAPI
   ↓
Streamlit Dashboard
   ↓
Inventory Decisions
```

The application is containerized with Docker Compose so the API and dashboard can run as separate services.

---

## Dataset

The project uses the **Corporación Favorita Grocery Sales Forecasting** dataset from Kaggle's Store Sales — Time Series Forecasting competition.

The dataset contains historical information including:

- Daily store sales
- Product families
- Store information
- Promotions
- Oil prices
- Transactions
- Holiday/event information

The prediction target is:

```text
sales
```

for each combination of:

```text
date × store_nbr × family
```

Raw Kaggle datasets are intentionally excluded from this repository.

---

## Medallion Data Architecture

The data pipeline follows a simplified **Bronze → Silver → Gold** architecture.

### Bronze Layer

Raw CSV files are ingested and converted into Parquet files.

```text
data/raw/
     ↓
data/bronze/
```

Benefits include:

- smaller storage footprint
- faster analytical reads
- columnar storage
- preservation of source data

---

### Silver Layer

The Silver layer performs data cleaning and integration.

Transformations include:

- date parsing
- data type normalization
- missing-value handling
- store metadata joins
- oil-price integration
- transaction integration
- basic data validation

Output:

```text
data/silver/sales_clean.parquet
```

---

### Gold Layer

The Gold layer creates model-ready forecasting features.

Examples include:

**Calendar features**

```text
year
month
day
day_of_week
week_of_year
quarter
is_weekend
```

**Lag features**

```text
sales_lag_1
sales_lag_7
sales_lag_14
sales_lag_28
```

**Rolling statistics**

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
rolling_std_7
```

Rolling features are shifted before calculation to reduce target leakage.

Output:

```text
data/gold/forecast_features.parquet
```

---

## Forecasting Models

The project currently includes:

### Seasonal Naive Baseline

A simple weekly seasonal forecast:

```text
forecast(t) = sales(t - 7)
```

This establishes a baseline that machine learning models should outperform.

### LightGBM

The primary forecasting model is a LightGBM gradient-boosted decision tree model.

Features include:

- store
- product family
- promotions
- oil price
- store metadata
- calendar features
- sales lags
- rolling sales statistics

The trained model is stored as:

```text
models/lightgbm_forecaster.joblib
```

---

## Evaluation Metrics

Forecasting performance is evaluated using:

### RMSLE

Root Mean Squared Logarithmic Error.

Useful when forecast errors should be evaluated proportionally across products with different sales scales.

### MAE

Mean Absolute Error.

Measures the average absolute forecast error.

### RMSE

Root Mean Squared Error.

Penalizes large forecasting errors more strongly.

### WMAPE

Weighted Mean Absolute Percentage Error.

Provides an interpretable portfolio-level measure of forecast error relative to total demand.

---

## Recursive Forecasting

The application supports multi-day recursive forecasting.

For example:

```text
Historical sales
      ↓
Predict Day 1
      ↓
Day 1 prediction becomes lag input
      ↓
Predict Day 2
      ↓
Day 2 prediction becomes lag input
      ↓
...
Predict Day N
```

This allows the API to generate forecasts beyond a single day without requiring actual future sales observations.

A stricter recursive historical backtesting framework is planned to ensure evaluation more closely matches real multi-step production inference.

---

## FastAPI Model Service

The trained forecasting model is exposed through a REST API built with FastAPI.

Main endpoints include:

```text
GET  /health
GET  /stores
GET  /families
POST /forecast
POST /forecast/multi
```

Interactive API documentation is available locally after starting the application:

```text
http://localhost:8000/docs
```

---

## Streamlit Dashboard

The Streamlit dashboard provides an interface for generating forecasts and translating them into inventory insights.

Users can select:

- Store
- Product family
- Forecast horizon
- Promotion assumptions
- Current inventory
- Safety stock percentage

The dashboard displays:

- Daily demand forecasts
- Total forecast demand
- Average daily demand
- Estimated safety stock
- Required inventory
- Expected shortage
- Recommended reorder quantity
- Projected ending inventory

---

## Application Architecture

```text
                         ┌─────────────────────┐
                         │     Kaggle Data     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Bronze Layer     │
                         │      Parquet        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Silver Layer     │
                         │ Cleaning + Joining  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Gold Layer      │
                         │ Forecast Features   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      LightGBM       │
                         │ Forecasting Model   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Forecast Service   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FastAPI       │
                         │       :8000         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Streamlit      │
                         │       :8501         │
                         └─────────────────────┘
```

---

## Docker Architecture

The application runs as two Docker services:

```text
┌──────────────────────┐
│ Streamlit Container  │
│       :8501          │
└──────────┬───────────┘
           │
           │ HTTP
           ▼
┌──────────────────────┐
│   FastAPI Container  │
│       :8000          │
└──────────┬───────────┘
           │
           ▼
       LightGBM
```

Docker Compose provides networking and lifecycle management for both services.

---

## Technology Stack

### Data Engineering

- Python
- pandas
- NumPy
- PyArrow
- Parquet
- DuckDB

### Machine Learning

- scikit-learn
- LightGBM
- XGBoost
- joblib

### Experiment Tracking

- MLflow

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- Streamlit

### DevOps

- Docker
- Docker Compose
- Git
- GitHub
- GitHub Actions
- pytest

---

## Project Structure

```text
fmcg-demand-platform/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── models/
│
├── monitoring/
│
├── notebooks/
│
├── sql/
│
├── src/
│   ├── ingestion/
│   ├── transformation/
│   ├── features/
│   └── models/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd fmcg-demand-platform
```

### 2. Create a Python environment

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the data pipeline

Place the Kaggle source files inside:

```text
data/raw/
```

Then run:

```bash
python src/ingestion/ingest.py
python src/transformation/build_silver.py
python src/features/build_gold.py
```

### 4. Train the model

Run the forecasting training pipeline before starting the API if no model artifact is available.

For example:

```bash
python src/models/train_lightgbm.py
```

### 5. Run with Docker

Build the containers:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

Or run in the background:

```bash
docker compose up -d
```

The applications will be available locally at:

```text
FastAPI:   http://localhost:8000
Swagger:   http://localhost:8000/docs
Streamlit: http://localhost:8501
```

Stop the services with:

```bash
docker compose down
```

---

## Testing

Run the test suite with:

```bash
pytest -v
```

The project includes tests for forecasting metrics and data-quality checks.

GitHub Actions can be used to automatically execute tests when code is pushed or a pull request is created.

---

## Current Limitations

This project is under active development.

Current limitations include:

- Multi-step historical evaluation still needs strict recursive backtesting.
- Future oil-price assumptions currently require a production strategy.
- Holiday/event features require store-aware integration.
- Inventory safety stock currently uses a simplified percentage-based approach.
- Model and large generated data artifacts should eventually be stored using an artifact registry or object storage rather than Git.
- Additional monitoring and automated retraining infrastructure is planned.

These limitations are intentionally documented to distinguish the current implementation from future production enhancements.

---

## Future Improvements

Planned improvements include:

- Strict recursive multi-step backtesting
- Time-series cross-validation
- LightGBM vs XGBoost comparison
- Hyperparameter optimization
- MLflow model registry
- Forecast-error-based safety stock
- Reorder point calculations
- Lead-time modeling
- Service-level optimization
- Model drift monitoring
- Data drift monitoring
- Automated model retraining
- Cloud object storage
- Cloud deployment
- CI/CD deployment pipeline
- Production observability

---

## Business Value

The platform demonstrates how forecasting can support FMCG inventory decisions by connecting:

```text
Demand Forecast
       ↓
Expected Demand
       ↓
Safety Stock
       ↓
Required Inventory
       ↓
Shortage Detection
       ↓
Reorder Recommendation
```

Instead of treating forecasting as an isolated modeling problem, the project connects machine learning predictions to an operational business decision.

---

## Author

**Joshua Bloodymier Salvino**

Physics graduate focused on data science, machine learning engineering, data engineering, and quantitative modeling.

---

## Disclaimer

This project is intended for educational and portfolio purposes. Forecasts and inventory recommendations should not be treated as production business decisions without additional validation, monitoring, and domain-specific controls.