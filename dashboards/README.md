# Customer Churn Analytics Dashboard

Interactive Streamlit dashboard for customer churn prediction insights and analysis.

## Features

### 📊 Overview Dashboard
- Key performance indicators (KPIs)
- Model performance comparison
- Interactive lift charts and gains visualizations
- Business impact analysis with ROI calculations

### ⚠️ Customer Risk Analysis
- Sortable high-risk customer table with search functionality
- Adjustable risk thresholds
- Batch scoring interface for new data
- CSV download capabilities

### 👤 Customer Details
- Individual customer profile display
- Feature contribution analysis (simplified SHAP-like explanations)
- Recommended action templates based on risk level
- Customer Lifetime Value (CLV) integration

### 🎯 Segment Analysis & CLV
- Interactive customer segmentation with K-means clustering
- CLV distribution and ranking visualizations
- Campaign targeting and ROI analysis tools
- Segment profiling with descriptive statistics

### 📁 Data Management
- Data upload and processing capabilities
- Support for Telco and Olist datasets via Kaggle
- Data quality reporting
- Model loading and management

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the main project dependencies are installed:
```bash
pip install -r ../requirements.txt
```

## Usage

### Quick Start (Demo Mode)
If you don't have pre-trained models, create demo models first:
```bash
python dashboards/demo_mode.py
```

### Launch Dashboard

#### Option 1: Using the launcher script
```bash
python run_dashboard.py
```

#### Option 2: Direct Streamlit command
```bash
streamlit run app.py
```

#### Option 3: Using the project startup script
```bash
./start_dashboard.sh
```

The dashboard will be available at `http://localhost:8501`

## Data Sources

The dashboard supports multiple data sources:

1. **Upload CSV File**: Upload your own customer data in CSV format
2. **Telco Dataset**: Automatically download and process the Telco customer churn dataset from Kaggle
3. **Olist Dataset**: Automatically download and process the Brazilian e-commerce dataset from Kaggle

## Model Integration

The dashboard can load pre-trained models from the `models/` directory. Supported model formats:
- Scikit-learn models (saved with joblib)
- Model metadata (JSON format)
- Cross-validation results

## Key Components

### Navigation
- Sidebar navigation with page selection
- Data upload and processing controls
- Model management interface

### Interactive Features
- Real-time filtering and search
- Adjustable thresholds and parameters
- Download capabilities for results
- Responsive visualizations with Plotly

### Business Metrics
- Lift tables and gains charts
- Expected savings calculations
- ROI analysis for retention campaigns
- Cost-sensitive evaluation metrics

## Configuration

The dashboard uses the main project configuration from `config.py`. Key parameters:
- Business parameters (retention value, contact cost)
- Model paths and settings
- Data processing options

## Troubleshooting

### Common Issues

1. **"No model selected" Error**: 
   - Load models using the sidebar Model Management section
   - Or create demo models: `python dashboards/demo_mode.py`

2. **Import Errors**: 
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that the `src/` directory is accessible

3. **Data Loading Errors**: 
   - Check internet connection for Kaggle dataset downloads
   - Try uploading a CSV file instead

4. **Model Loading Errors**: 
   - Verify model files exist in the specified directory
   - Use demo mode to create sample models

5. **"TypeError: object of type 'NoneType'" Errors**:
   - This occurs when trying to view analytics without data/models loaded
   - Load data and models first using the sidebar

6. **Memory Issues**: 
   - For large datasets, consider sampling or increasing system memory
   - Use the data preview feature to check dataset size

### Performance Tips

1. Use data sampling for large datasets (>100k rows)
2. Cache expensive computations using Streamlit's caching
3. Limit the number of customers displayed in tables
4. Use appropriate visualization limits for better performance

### Getting Started Checklist

✅ **Step 1**: Install dependencies
```bash
pip install -r dashboards/requirements.txt
```

✅ **Step 2**: Create demo models (if needed)
```bash
python dashboards/demo_mode.py
```

✅ **Step 3**: Launch dashboard
```bash
streamlit run dashboards/app.py
```

✅ **Step 4**: In the dashboard:
1. Load data (sidebar → Data Upload → Load Telco Dataset)
2. Load models (sidebar → Model Management → Load Models)
3. Explore the analytics!

## Development

To extend the dashboard:

1. Add new pages by creating methods in the `ChurnDashboard` class
2. Update the navigation in `render_sidebar()`
3. Follow the existing pattern for session state management
4. Use Plotly for interactive visualizations
5. Implement proper error handling and user feedback

## Dependencies

- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning utilities
- **joblib**: Model serialization

## License

This dashboard is part of the Customer Churn ML Pipeline project.