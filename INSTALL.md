# Installation Guide

This guide will help you set up the Customer Churn ML Pipeline environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Options

### Option 1: Minimal Installation (Recommended for EDA)

Install only the core dependencies needed for the EDA framework:

```bash
pip install -r requirements-minimal.txt
```

This includes:
- pandas, numpy, scipy (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (basic ML utilities)
- kagglehub (data loading)
- jupyter (notebook support)
- pytest (testing)

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
- Data profiling and validation tools
- Development and code quality tools

### Option 3: Development Installation

For contributors and developers:

```bash
pip install -e .[dev]
```

This installs the package in editable mode with development dependencies.

## Virtual Environment Setup (Recommended)

It's recommended to use a virtual environment to avoid conflicts:

### Using venv (Python built-in)

```bash
# Create virtual environment
python -m venv churn_env

# Activate virtual environment
# On macOS/Linux:
source churn_env/bin/activate
# On Windows:
churn_env\Scripts\activate

# Install dependencies
pip install -r requirements-minimal.txt

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n churn_env python=3.10

# Activate environment
conda activate churn_env

# Install dependencies
pip install -r requirements-minimal.txt

# Deactivate when done
conda deactivate
```

## Verification

After installation, verify everything works:

```bash
# Test Python imports
python -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
print('✅ All core libraries imported successfully')
"

# Test project modules
python -c "
import sys
sys.path.append('src')
from eda import EDAAnalyzer, StatisticalTester, EDAVisualizer
from data_prep import DataLoader, DataCleaner
print('✅ Project modules imported successfully')
"

# Run tests
pytest tests/ -v
```

## Jupyter Notebook Setup

To use the EDA notebook:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Navigate to notebooks/01_eda.ipynb
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the correct directory and have activated your virtual environment.

2. **Kaggle API issues**: You may need to set up Kaggle API credentials:
   ```bash
   # Create ~/.kaggle/kaggle.json with your API credentials
   # Or set environment variables:
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

3. **Memory issues**: Some operations may require significant memory. Consider using a machine with at least 8GB RAM.

4. **Plotting issues on headless systems**: If running on a server without display:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Use non-interactive backend
   ```

### Platform-Specific Notes

#### macOS
- You may need to install Xcode command line tools: `xcode-select --install`
- For M1/M2 Macs, some packages may need specific versions

#### Windows
- Consider using Anaconda for easier package management
- Some packages may require Visual Studio Build Tools

#### Linux
- Install system dependencies: `sudo apt-get install python3-dev build-essential`

## Next Steps

After successful installation:

1. **Explore the EDA notebook**: `notebooks/01_eda.ipynb`
2. **Run the test suite**: `pytest tests/`
3. **Check the documentation**: Review the spec files in `.kiro/specs/`
4. **Start with sample data**: The framework can generate synthetic data for testing

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure you're using a supported Python version (3.8+)
3. Try creating a fresh virtual environment
4. Check that all dependencies are properly installed

## Development Setup

For development work:

```bash
# Clone the repository
git clone <repository-url>
cd customer-churn-ml-pipeline

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks (optional)
pre-commit install

# Run tests
pytest tests/ -v --cov=src
```