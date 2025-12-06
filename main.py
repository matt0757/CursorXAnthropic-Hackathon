"""
Main entry point - Runs both the FastAPI backend and Cargo TMS Dashboard.
"""
import subprocess
import sys
import time

if __name__ == "__main__":
    print("🚀 Starting Cargo Capacity Forecaster...")
    print("=" * 50)
    
    # Start the FastAPI backend
    print("📡 Starting Backend API (port 8000)...")
    backend_process = subprocess.Popen([
        sys.executable, "run_backend.py"
    ])
    
    # Give backend time to start
    time.sleep(3)
    
    # Start the Cargo TMS Dashboard
    print("📊 Starting Cargo TMS Dashboard (port 8503)...")
    tms_process = subprocess.Popen([
        sys.executable, "run_tms.py"
    ])
    
    print("=" * 50)
    print("✅ All services started!")
    print("   - Backend API: http://localhost:8000")
    print("   - TMS Dashboard: http://localhost:8503")
    print("=" * 50)
    print("Press Ctrl+C to stop all services...")
    
    try:
        # Wait for both processes
        backend_process.wait()
        tms_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        backend_process.terminate()
        tms_process.terminate()
        print("✅ All services stopped.")
