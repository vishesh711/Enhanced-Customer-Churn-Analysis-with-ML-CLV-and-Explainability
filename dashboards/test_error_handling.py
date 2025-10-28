#!/usr/bin/env python3
"""
Test script for dashboard error handling
Verifies that all methods handle None cases gracefully
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock

# Add src and root to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

def test_error_handling():
    """Test dashboard error handling without Streamlit runtime"""
    
    print("🧪 Testing dashboard error handling...")
    
    # Mock streamlit session state
    class MockSessionState:
        def __init__(self):
            self.data_loaded = False
            self.models_loaded = False
            self.current_data = None
            self.trained_models = {}
            self.model_metadata = {}
            self.predictions = None
            self.business_metrics = None
            self.segments = None
            self.clv_data = None
    
    # Import dashboard after mocking
    try:
        from app import ChurnDashboard
        print("✅ Dashboard class imported successfully")
    except Exception as e:
        print(f"❌ Failed to import dashboard: {e}")
        return False
    
    # Create dashboard instance
    dashboard = ChurnDashboard()
    
    # Mock streamlit session state
    import streamlit as st
    st.session_state = MockSessionState()
    
    # Test methods that should handle None gracefully
    test_cases = [
        ("get_high_risk_customers", lambda: dashboard.get_high_risk_customers(0.5, 100)),
        ("display_kpi_metrics", lambda: dashboard.display_kpi_metrics()),
        ("display_model_comparison", lambda: dashboard.display_model_comparison()),
        ("display_lift_chart", lambda: dashboard.display_lift_chart()),
        ("display_gains_chart", lambda: dashboard.display_gains_chart()),
        ("display_business_impact", lambda: dashboard.display_business_impact()),
    ]
    
    all_passed = True
    
    for test_name, test_func in test_cases:
        try:
            result = test_func()
            print(f"✅ {test_name}: Handled None case gracefully")
        except Exception as e:
            print(f"❌ {test_name}: Failed with error: {e}")
            all_passed = False
    
    # Test with sample data
    print("\n🧪 Testing with sample data...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(10)],
        'tenure': np.random.randint(1, 72, 10),
        'MonthlyCharges': np.random.uniform(20, 120, 10),
        'Churn': np.random.choice([0, 1], 10)
    })
    
    st.session_state.current_data = sample_data
    st.session_state.data_loaded = True
    
    # Test data-dependent methods
    try:
        dashboard.perform_customer_segmentation()
        print("✅ Customer segmentation: Works with sample data")
    except Exception as e:
        print(f"❌ Customer segmentation: Failed with error: {e}")
        all_passed = False
    
    try:
        dashboard.calculate_clv_estimates()
        print("✅ CLV calculation: Works with sample data")
    except Exception as e:
        print(f"❌ CLV calculation: Failed with error: {e}")
        all_passed = False
    
    if all_passed:
        print("\n🎉 All error handling tests passed!")
        print("✅ Dashboard is robust and handles edge cases gracefully")
    else:
        print("\n❌ Some tests failed - dashboard needs additional error handling")
    
    return all_passed

if __name__ == "__main__":
    success = test_error_handling()
    sys.exit(0 if success else 1)