#!/bin/bash

# Customer Churn ML Pipeline - Environment Setup Script
# This script sets up the Python environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up Customer Churn ML Pipeline Environment"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "churn_env" ]; then
    python3 -m venv churn_env
    echo "✅ Virtual environment created: churn_env"
else
    echo "✅ Virtual environment already exists: churn_env"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source churn_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
read -p "Install full dependencies? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing full requirements..."
    pip install -r requirements.txt
else
    echo "Installing minimal requirements..."
    pip install -r requirements-minimal.txt
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/interim
mkdir -p models
mkdir -p reports/figures reports/tables
mkdir -p logs

echo "✅ Directories created"

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
print('✅ Core libraries imported successfully')
"

# Test project modules
python3 -c "
import sys
sys.path.append('src')
try:
    from eda import EDAAnalyzer, StatisticalTester, EDAVisualizer
    from data_prep import DataLoader, DataCleaner
    print('✅ Project modules imported successfully')
except Exception as e:
    print(f'⚠️  Project modules import warning: {e}')
    print('This is normal if some dependencies are missing')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source churn_env/bin/activate"
echo ""
echo "To start Jupyter notebook:"
echo "  jupyter notebook"
echo ""
echo "To run the EDA notebook:"
echo "  jupyter notebook notebooks/01_eda.ipynb"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""
echo "Happy analyzing! 📊"