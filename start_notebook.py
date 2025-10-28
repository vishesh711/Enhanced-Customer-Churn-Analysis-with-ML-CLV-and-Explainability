#!/usr/bin/env python3
"""
Startup script to verify environment and launch Jupyter notebook
"""

import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if the environment is set up correctly"""
    print("🔍 Checking environment...")
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
    else:
        print("⚠️  Virtual environment not detected. Run: source churn_env/bin/activate")
        return False
    
    # Check Python version
    if sys.version_info >= (3, 8):
        print(f"✅ Python version: {sys.version.split()[0]}")
    else:
        print(f"❌ Python version too old: {sys.version.split()[0]}")
        return False
    
    # Test imports
    try:
        # Set up paths
        project_root = Path.cwd()
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / 'src'))
        
        # Test core imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Core data science libraries available")
        
        # Test project imports
        from config import config
        from data_prep import DataLoader, DataCleaner, DataSplitter
        from eda import EDAAnalyzer, StatisticalTester, EDAVisualizer
        print("✅ Project modules available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Customer Churn ML Pipeline - Notebook Launcher")
    print("=" * 50)
    
    if not check_environment():
        print("\n❌ Environment check failed!")
        print("\nTo fix:")
        print("1. Activate virtual environment: source churn_env/bin/activate")
        print("2. Install dependencies: pip install -r requirements-minimal.txt")
        print("3. Run this script again")
        return False
    
    print("\n✅ Environment check passed!")
    print("\n📝 Instructions for using the notebook:")
    print("1. The notebook will open in your browser")
    print("2. Navigate to notebooks/01_eda.ipynb")
    print("3. Run cells in order (Shift+Enter)")
    print("4. If you get import errors, restart the kernel and run all cells")
    
    # Ask user if they want to start Jupyter
    try:
        response = input("\n🚀 Start Jupyter notebook? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print("\n🎯 Starting Jupyter notebook...")
            subprocess.run(['jupyter', 'notebook'], check=True)
        else:
            print("\n📋 To start manually, run: jupyter notebook")
            print("Then open: notebooks/01_eda.ipynb")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except subprocess.CalledProcessError:
        print("\n❌ Failed to start Jupyter. Try running manually: jupyter notebook")
    
    return True

if __name__ == "__main__":
    main()