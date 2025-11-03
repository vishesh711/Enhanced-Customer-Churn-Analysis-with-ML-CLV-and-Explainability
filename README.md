# Enhanced Customer Churn Analysis with ML, CLV, and Explainability

A comprehensive, production-ready machine learning pipeline for customer churn prediction, featuring advanced analytics, Customer Lifetime Value (CLV) estimation, customer segmentation, and model explainability. This project provides end-to-end capabilities from data loading and exploration to model training, evaluation, and deployment through an interactive dashboard.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Interactive Dashboard](#interactive-dashboard)
- [Key Components](#key-components)
- [Model Training & Evaluation](#model-training--evaluation)
- [Customer Segmentation & CLV](#customer-segmentation--clv)
- [Model Explainability](#model-explainability)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning pipeline for predicting customer churn with a focus on business value. It combines traditional ML approaches with advanced techniques like:

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Customer Lifetime Value (CLV)**: Statistical models for CLV estimation and prediction
- **Customer Segmentation**: K-means clustering for customer grouping and analysis
- **Model Explainability**: SHAP and LIME for global and local explanations
- **Business Metrics**: Cost-sensitive threshold optimization and ROI analysis
- **Interactive Dashboard**: Streamlit-based web application for analytics and visualization

## ✨ Features

### Core ML Pipeline
- **Data Loading & Processing**: Support for Telco and Olist datasets via Kaggle API
- **Feature Engineering**: Automated feature creation, selection, and transformation
- **Model Training**: Multiple baseline and ensemble models with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, Precision-Recall, Brier Score
- **Probability Calibration**: Isotonic and sigmoid calibration for better probability estimates
- **Threshold Optimization**: Business-aware threshold selection based on cost-benefit analysis

### Advanced Analytics
- **Customer Segmentation**: Unsupervised clustering with optimal cluster selection
- **CLV Calculation**: Beta-Geometric/NBD and Gamma-Gamma models for CLV prediction
- **Feature Importance**: Global and local feature contribution analysis
- **Model Explainability**: SHAP values and LIME explanations for interpretability
- **Risk Scoring**: Customer-level churn risk with actionable insights

### Business Intelligence
- **Cost-Sensitive Evaluation**: Profit curves and expected value calculations
- **ROI Analysis**: Campaign targeting and return on investment metrics
- **Segment Profiling**: Comprehensive characteristics for each customer segment
- **Lift Analysis**: Gains charts and lift tables for marketing optimization

### Interactive Dashboard
- **Overview Dashboard**: KPI metrics, model performance comparison, lift charts
- **Customer Risk Analysis**: Sortable high-risk customer table with filtering
- **Customer Details**: Individual customer profiles with feature contributions
- **Segment Analysis**: Interactive segmentation with CLV visualizations
- **Data Management**: Upload, process, and quality reporting

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │Telco Data│  │Olist Data│  │Custom CSV│                │
│  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Preparation                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │Data Loading  │→ │Data Cleaning │→ │Feature Eng. │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   EDA Module  │  │Train Module  │  │Segment Module│
│                                                      │
│  - Statistical Tests│  │  - Model Training│  │  - Clustering  │
│  - Visualizations  │  │  - Hyperparameter│  │  - CLV Models  │
│  - Data Quality    │  │    Tuning       │  │  - Profiling    │
└──────────────┘  └──────────────┘  └──────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Evaluate Module│  │Explain Module│  │   Model      │
│                                                      │
│  - Metrics   │  │  - SHAP       │  │   Registry    │
│  - Threshold │  │  - LIME       │  │               │
│  - Calibration│  │  - Feature    │  │               │
│               │  │    Importance│  │               │
└──────────────┘  └──────────────┘  └──────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Interactive Dashboard (Streamlit)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Overview │  │   Risk   │  │Customer  │  │ Segment │  │
│  │          │  │ Analysis │  │  Details │  │ Analysis │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Option 1: Minimal Installation (Recommended for EDA)

Install only core dependencies for exploratory data analysis:

```bash
pip install -r requirements-minimal.txt
```

This includes:
- pandas, numpy, scipy (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (basic ML utilities)
- kagglehub (data loading)
- jupyter (notebook support)

### Option 2: Full Installation

Install all dependencies for the complete ML pipeline:

```bash
pip install -r requirements.txt
```

This includes everything from minimal plus:
- Advanced ML libraries (XGBoost, LightGBM, CatBoost)
- Model interpretation tools (SHAP, LIME)
- Hyperparameter optimization (Optuna)
- Web dashboard tools (Streamlit, Plotly)
- CLV modeling (lifetimes)
- Data profiling and validation tools

### Option 3: Development Installation

For contributors and developers:

```bash
pip install -e .[dev]
```

This installs the package in editable mode with development dependencies including testing frameworks and code quality tools.

### Virtual Environment Setup (Recommended)

It's recommended to use a virtual environment:

#### Using venv

```bash
# Create virtual environment
python -m venv churn_env

# Activate virtual environment
# On macOS/Linux:
source churn_env/bin/activate
# On Windows:
churn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

#### Using conda

```bash
# Create conda environment
conda create -n churn_env python=3.10

# Activate environment
conda activate churn_env

# Install dependencies
pip install -r requirements.txt
```

### Kaggle API Setup (Optional)

For downloading datasets from Kaggle, configure your API credentials:

```bash
# Option 1: Create ~/.kaggle/kaggle.json with your credentials
{
  "username": "your_username",
  "key": "your_api_key"
}

# Option 2: Set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

## 🚀 Quick Start

### 1. Load and Prepare Data

```python
from src.data_prep import DataLoader, DataCleaner
from config import config

# Initialize data loader
loader = DataLoader()

# Download Telco dataset
dataset_path = loader.download_telco_data()

# Load and validate data
telco_data = loader.load_telco_data(dataset_path)

# Clean data
cleaner = DataCleaner()
cleaned_data = cleaner.clean_telco_data(telco_data)
```

### 2. Exploratory Data Analysis

```python
from src.eda import EDAAnalyzer, EDAVisualizer

# Initialize EDA components
analyzer = EDAAnalyzer(cleaned_data)
visualizer = EDAVisualizer(cleaned_data)

# Generate statistical summary
summary = analyzer.get_summary_statistics()

# Create visualizations
visualizer.plot_target_distribution()
visualizer.plot_correlation_matrix()
visualizer.plot_feature_distributions()
```

### 3. Feature Engineering

```python
from src.features import FeatureEngineer

# Initialize feature engineer
feature_engineer = FeatureEngineer()

# Prepare features
X, y = feature_engineer.prepare_features(cleaned_data)
X_train, X_test, y_train, y_test = feature_engineer.train_test_split(X, y)
```

### 4. Train Models

```python
from src.train import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train baseline models
baseline_models = trainer.train_baseline_models(X_train, y_train)

# Train ensemble models
ensemble_models = trainer.train_ensemble_models(X_train, y_train)

# Compare models
comparison_df = trainer.get_model_comparison()
print(comparison_df)

# Get best model
best_name, best_model = trainer.get_best_model(metric='roc_auc')
```

### 5. Evaluate Models

```python
from src.evaluate import ModelEvaluator
from src.train import ThresholdOptimizer

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate on test set
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
results = evaluator.evaluate_model(y_test, y_pred_proba)

# Business-aware threshold optimization
optimizer = ThresholdOptimizer()
optimal_threshold, metrics = optimizer.optimize_business_threshold(
    y_test, y_pred_proba, model_name='best_model'
)
```

### 6. Model Explainability

```python
from src.explain import GlobalExplainer, LocalExplainer

# Global explanations
global_explainer = GlobalExplainer(best_model, X_train.columns, 'best_model')
global_explainer.fit_explainer(X_train.sample(100))
explanation = global_explainer.explain_model(X_test.sample(1000))

# Local explanations
local_explainer = LocalExplainer(best_model, X_train.columns)
local_explanation = local_explainer.explain_instance(X_test.iloc[0])
```

### 7. Customer Segmentation & CLV

```python
from src.segment import CustomerSegmenter, CLVAnalyzer

# Customer segmentation
segmenter = CustomerSegmenter(random_state=42)
optimal_clusters = segmenter.find_optimal_clusters(X_train)
segments = segmenter.create_segments(X_train, optimal_clusters)
profiles = segmenter.profile_segments(X_train, y_train, segments)

# CLV Analysis
clv_analyzer = CLVAnalyzer()
clv_predictions = clv_analyzer.predict_clv(cleaned_data)
```

### 8. Launch Interactive Dashboard

```bash
# Option 1: Using the launcher script
python dashboards/run_dashboard.py

# Option 2: Direct Streamlit command
streamlit run dashboards/app.py

# Option 3: Using the project startup script
./start_dashboard.sh
```

The dashboard will be available at `http://localhost:8501`

## 📁 Project Structure

```
enhanced-customer-churn-analysis/
│
├── README.md                  # Project documentation (this file)
├── INSTALL.md                 # Detailed installation guide
├── TROUBLESHOOTING.md         # Troubleshooting guide
├── requirements.txt           # Full dependencies
├── requirements-minimal.txt   # Minimal dependencies
├── setup.py                   # Package setup configuration
├── config.py                  # Configuration settings
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── data_prep.py          # Data loading and cleaning
│   ├── eda.py                # Exploratory data analysis
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training and calibration
│   ├── evaluate.py           # Model evaluation
│   ├── explain.py            # Model explainability (SHAP/LIME)
│   └── segment.py            # Customer segmentation and CLV
│
├── dashboards/                # Interactive dashboard
│   ├── app.py                # Main Streamlit application
│   ├── run_dashboard.py      # Dashboard launcher
│   ├── demo_mode.py          # Demo mode for testing
│   └── requirements.txt      # Dashboard dependencies
│
├── notebooks/                 # Jupyter notebooks
│   └── 01_eda.ipynb          # EDA examples
│
├── tests/                     # Unit tests
│   ├── test_data_prep.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_explain.py
│   ├── test_features.py
│   └── test_segment.py
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│   └── interim/               # Intermediate data files
│
├── models/                    # Saved models
│   ├── *_model.joblib        # Trained models
│   └── *_metadata.json       # Model metadata
│
└── reports/                    # Generated reports
    ├── figures/               # Visualization outputs
    └── tables/                # Table outputs
```

## 💡 Usage Examples

### Complete Pipeline Example

```python
from src.data_prep import DataLoader, DataCleaner
from src.features import FeatureEngineer
from src.train import ModelTrainer, ProbabilityCalibrator
from src.evaluate import ModelEvaluator, ThresholdOptimizer
from config import config

# Step 1: Load data
loader = DataLoader()
dataset_path = loader.download_telco_data()
raw_data = loader.load_telco_data(dataset_path)

# Step 2: Clean data
cleaner = DataCleaner()
cleaned_data = cleaner.clean_telco_data(raw_data)

# Step 3: Feature engineering
fe = FeatureEngineer()
X, y = fe.prepare_features(cleaned_data)
X_train, X_test, y_train, y_test = fe.train_test_split(X, y)

# Step 4: Train models
trainer = ModelTrainer(random_state=config.RANDOM_SEED)
ensemble_models = trainer.train_ensemble_models(X_train, y_train)
best_name, best_model = trainer.get_best_model()

# Step 5: Calibrate probabilities
calibrator = ProbabilityCalibrator(method='isotonic')
calibrated_model = calibrator.calibrate_model(
    best_model, X_train, y_train, best_name
)

# Step 6: Evaluate
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(y_test, y_pred_proba)

# Step 7: Optimize threshold
optimizer = ThresholdOptimizer()
threshold, metrics = optimizer.optimize_business_threshold(
    y_test, y_pred_proba, best_name
)

# Step 8: Save models
trainer.save_models(config.MODEL_PATH)
```

### Customer Segmentation Example

```python
from src.segment import CustomerSegmenter

# Initialize segmenter
segmenter = CustomerSegmenter(random_state=42)

# Find optimal number of clusters
optimal_n = segmenter.find_optimal_clusters(
    X_train, max_clusters=10, method='silhouette'
)

# Create segments
segments = segmenter.create_segments(X_train, optimal_n)

# Profile segments
profiles = segmenter.profile_segments(X_train, y_train, segments)

# Get segment characteristics
for profile in profiles:
    print(f"Segment {profile.segment_id}:")
    print(f"  Size: {profile.size}")
    print(f"  Churn Rate: {profile.churn_rate:.2%}")
    print(f"  Avg CLV: ${profile.avg_clv:.2f}")
```

### CLV Prediction Example

```python
from src.segment import CLVAnalyzer

# Initialize CLV analyzer
clv_analyzer = CLVAnalyzer()

# Predict CLV for customers
clv_predictions = clv_analyzer.predict_clv(
    transaction_data,
    time_horizon=12,  # months
    discount_rate=0.01
)

# Get top-valued customers
top_customers = clv_predictions.nlargest(10, 'predicted_clv')
```

## 🎨 Interactive Dashboard

The project includes a comprehensive Streamlit dashboard for exploring results and making predictions.

### Dashboard Features

1. **Overview Dashboard**
   - Key performance indicators (KPIs)
   - Model performance comparison
   - Interactive lift charts and gains visualizations
   - Business impact analysis with ROI calculations

2. **Customer Risk Analysis**
   - Sortable high-risk customer table with search functionality
   - Adjustable risk thresholds
   - Batch scoring interface for new data
   - CSV download capabilities

3. **Customer Details**
   - Individual customer profile display
   - Feature contribution analysis (SHAP-like explanations)
   - Recommended action templates based on risk level
   - Customer Lifetime Value (CLV) integration

4. **Segment Analysis & CLV**
   - Interactive customer segmentation with K-means clustering
   - CLV distribution and ranking visualizations
   - Campaign targeting and ROI analysis tools
   - Segment profiling with descriptive statistics

### Launching the Dashboard

```bash
# Quick start with demo data
python dashboards/demo_mode.py  # Create demo models first
streamlit run dashboards/app.py

# Or use the launcher script
python dashboards/run_dashboard.py
```

For detailed dashboard documentation, see [dashboards/README.md](dashboards/README.md)

## 🔧 Key Components

### Data Preparation (`src/data_prep.py`)
- **DataLoader**: Handles data loading from Kaggle or local files
- **DataCleaner**: Performs data cleaning and preprocessing
- **DataQualityReport**: Generates data quality assessments

### Feature Engineering (`src/features.py`)
- **FeatureEngineer**: Creates features, handles missing values, encodes categorical variables
- Automated feature selection and transformation
- Support for custom feature engineering pipelines

### Model Training (`src/train.py`)
- **ModelTrainer**: Trains multiple models (baseline and ensemble)
- **ProbabilityCalibrator**: Calibrates model probabilities
- **ThresholdOptimizer**: Business-aware threshold optimization
- **ModelRegistry**: Model versioning and metadata tracking

### Model Evaluation (`src/evaluate.py`)
- **ModelEvaluator**: Comprehensive model evaluation metrics
- Cost-sensitive evaluation
- ROC and Precision-Recall curve analysis
- Business impact calculations

### Model Explainability (`src/explain.py`)
- **GlobalExplainer**: Global model explanations using SHAP
- **LocalExplainer**: Instance-level explanations using LIME
- Feature importance analysis
- Partial dependence plots

### Customer Segmentation (`src/segment.py`)
- **CustomerSegmenter**: K-means clustering with optimal cluster selection
- **CLVAnalyzer**: Customer Lifetime Value prediction
- Segment profiling and characterization
- Campaign targeting recommendations

## 📊 Model Training & Evaluation

### Supported Models

- **Baseline Models**: Majority Class, Stratified, Logistic Regression
- **Ensemble Models**: Random Forest, XGBoost, LightGBM, CatBoost
- All models support hyperparameter tuning via RandomizedSearchCV

### Evaluation Metrics

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Probability Metrics**: ROC-AUC, PR-AUC, Brier Score, Log Loss
- **Business Metrics**: Expected Profit, ROI, Lift, Gains
- **Calibration Metrics**: ECE (Expected Calibration Error), MCE (Maximum Calibration Error)

### Model Selection

Models are evaluated using stratified cross-validation and compared across multiple metrics. The best model is selected based on ROC-AUC by default, but you can specify any metric.

### Probability Calibration

Models can be calibrated using:
- **Isotonic Regression**: Non-parametric calibration
- **Sigmoid/Platt Scaling**: Parametric calibration

### Threshold Optimization

The `ThresholdOptimizer` class finds optimal probability thresholds based on business parameters:
- Retention value per customer
- Cost of contacting a customer
- Cost of losing a customer
- Expected profit maximization

## 👥 Customer Segmentation & CLV

### Segmentation

Customer segmentation uses K-means clustering with optimal cluster selection based on:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

Each segment is profiled with:
- Size and churn rate
- Average CLV
- Characteristic features
- Recommended actions

### CLV Calculation

Customer Lifetime Value is calculated using:
- **Beta-Geometric/NBD Model**: Predicts purchase frequency and customer lifetime
- **Gamma-Gamma Model**: Predicts monetary value per transaction

Features:
- CLV prediction for future periods
- Confidence intervals
- Discounted CLV calculations
- Customer ranking by value

## 🔍 Model Explainability

### Global Explanations

- **SHAP Summary Plots**: Feature importance across all instances
- **Feature Importance Rankings**: Traditional feature importance
- **Partial Dependence Plots**: Feature effect on predictions
- **Feature Interactions**: Interaction strength analysis

### Local Explanations

- **SHAP Values**: Instance-specific feature contributions
- **LIME Explanations**: Local interpretable model explanations
- **Feature Contributions**: Quantified impact of each feature
- **Top Contributing Features**: Most important features for each prediction

## ⚙️ Configuration

Project configuration is managed through `config.py`. Key settings include:

### Business Parameters
```python
RETENTION_VALUE = 1000.0  # Value of retaining a customer
CONTACT_COST = 50.0       # Cost of contacting a customer
CHURN_COST = 500.0        # Cost of losing a customer
```

### Model Parameters
```python
CV_FOLDS = 5              # Cross-validation folds
HYPERPARAMETER_TRIALS = 100  # Trials for hyperparameter tuning
CALIBRATION_METHOD = "isotonic"  # Probability calibration method
```

### Data Parameters
```python
TEST_SIZE = 0.2           # Test set size
VALIDATION_SIZE = 0.2     # Validation set size
RANDOM_SEED = 42          # Random seed for reproducibility
```

Configuration can be overridden via environment variables:
```bash
export RETENTION_VALUE=1500.0
export CONTACT_COST=75.0
export RANDOM_SEED=123
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_train.py -v

# Run with markers
pytest tests/ -m "not slow"
```

Test files are located in the `tests/` directory and cover:
- Data preparation and cleaning
- Feature engineering
- Model training
- Model evaluation
- Model explainability
- Customer segmentation

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment
   - Check Python version (3.8+)

2. **Kaggle API Issues**
   - Configure Kaggle credentials in `~/.kaggle/kaggle.json`
   - Or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables

3. **Memory Issues**
   - Use data sampling for large datasets
   - Consider reducing `SHAP_SAMPLE_SIZE` in config
   - Use a machine with at least 8GB RAM

4. **Model Training Slow**
   - Reduce `HYPERPARAMETER_TRIALS` in config
   - Use fewer CV folds
   - Sample data for hyperparameter tuning

5. **Dashboard Not Loading**
   - Ensure Streamlit is installed: `pip install streamlit`
   - Check that models exist in `models/` directory
   - Run demo mode: `python dashboards/demo_mode.py`

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation as needed
- Use type hints for function signatures
- Run tests before submitting PR

### Code Quality

The project uses:
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run code quality checks:
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Datasets: Telco Customer Churn (Kaggle), Olist Brazilian E-commerce (Kaggle)
- Libraries: scikit-learn, XGBoost, LightGBM, SHAP, LIME, Streamlit
- Inspiration: Various customer churn analysis projects and best practices

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the documentation in `INSTALL.md` and `TROUBLESHOOTING.md`
- Review the dashboard README at `dashboards/README.md`

## 📚 Additional Resources

- [Installation Guide](INSTALL.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Dashboard Documentation](dashboards/README.md)

---

**Made with ❤️ for data scientists and ML engineers**
