#!/bin/bash

# Customer Churn Analytics Dashboard Startup Script

echo "🚀 Starting Customer Churn Analytics Dashboard..."

# Check if virtual environment exists
if [ ! -d "churn_env" ]; then
    echo "❌ Virtual environment 'churn_env' not found!"
    echo "Please create and activate the virtual environment first:"
    echo "  python -m venv churn_env"
    echo "  source churn_env/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "  pip install -r dashboards/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source churn_env/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found! Installing dashboard dependencies..."
    pip install -r dashboards/requirements.txt
fi

# Start the dashboard
echo "📊 Launching dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run dashboards/app.py --server.port 8501 --server.address localhost