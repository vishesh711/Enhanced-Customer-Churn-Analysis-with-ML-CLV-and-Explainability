#!/usr/bin/env python3
"""
Dashboard launcher script
Run this script to start the Streamlit dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "app.py"
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()