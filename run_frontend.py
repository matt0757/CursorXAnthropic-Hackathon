"""
Quick start script for the Streamlit frontend.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"])

