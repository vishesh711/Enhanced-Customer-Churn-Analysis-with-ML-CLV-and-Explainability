"""
Demo mode for the Customer Churn Analytics Dashboard
Creates mock models for demonstration purposes when no trained models are available
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from data_prep import DataLoader, DataCleaner

def create_demo_models():
    """Create demo models for dashboard testing"""
    print("🎯 Creating demo models for dashboard...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load sample data
    print("📊 Loading Telco dataset...")
    loader = DataLoader()
    df = loader.load_telco_data()
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean_telco_data(df)
    
    # Prepare features
    feature_cols = [col for col in df_clean.columns if col not in ['Churn', 'customerID']]
    X = df_clean[feature_cols]
    y = df_clean['Churn']
    
    # Handle categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    print("🤖 Training demo models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        model_path = models_dir / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        
        # Calculate basic metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"    Train accuracy: {train_score:.3f}")
        print(f"    Test accuracy: {test_score:.3f}")
        print(f"    Saved to: {model_path}")
    
    print("✅ Demo models created successfully!")
    print(f"📁 Models saved in: {models_dir.absolute()}")
    print("\n🚀 You can now run the dashboard:")
    print("   streamlit run dashboards/app.py")
    print("\n💡 In the dashboard:")
    print("   1. Load Telco dataset using the sidebar")
    print("   2. Load models from the 'models/' directory")
    print("   3. Explore the interactive analytics!")

if __name__ == "__main__":
    create_demo_models()