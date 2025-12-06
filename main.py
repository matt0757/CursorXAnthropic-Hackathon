"""
FastAPI Entrypoint - Root level main.py for deployment compatibility.
This file imports the app from backend.main to satisfy deployment requirements.
"""
from backend.main import app

# Export the app instance for ASGI servers
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
