@echo off
REM Customer Churn ML Pipeline - Environment Setup Script (Windows)
REM This script sets up the Python environment and installs dependencies

echo 🚀 Setting up Customer Churn ML Pipeline Environment
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python installation found

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "churn_env" (
    python -m venv churn_env
    echo ✅ Virtual environment created: churn_env
) else (
    echo ✅ Virtual environment already exists: churn_env
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call churn_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
set /p choice="Install full dependencies? (y/N): "
if /i "%choice%"=="y" (
    echo Installing full requirements...
    pip install -r requirements.txt
) else (
    echo Installing minimal requirements...
    pip install -r requirements-minimal.txt
)

REM Create necessary directories
echo 📁 Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\interim" mkdir data\interim
if not exist "models" mkdir models
if not exist "reports\figures" mkdir reports\figures
if not exist "reports\tables" mkdir reports\tables
if not exist "logs" mkdir logs

echo ✅ Directories created

REM Test installation
echo 🧪 Testing installation...
python -c "import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; from scipy import stats; print('✅ Core libraries imported successfully')"

echo.
echo 🎉 Setup completed successfully!
echo.
echo To activate the environment in the future, run:
echo   churn_env\Scripts\activate.bat
echo.
echo To start Jupyter notebook:
echo   jupyter notebook
echo.
echo To run the EDA notebook:
echo   jupyter notebook notebooks/01_eda.ipynb
echo.
echo To deactivate the environment:
echo   deactivate
echo.
echo Happy analyzing! 📊
pause