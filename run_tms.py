"""
Run the Cargo TMS (Transportation Management System) Dashboard
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "frontend/cargo_tms_app.py",
        "--server.port", "8503"
    ])
