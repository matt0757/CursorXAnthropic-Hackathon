# Quick Start Guide

## 🚀 Fast Setup (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python backend/train_model.py
```

This will:
- Find all CSV files in `data/` folder automatically
- Train the ML models
- Save to `models/forecaster.pkl`

**Expected output:** You should see training progress and evaluation metrics.

### Step 3: Run the Application

**Terminal 1 - Start Backend:**
```bash
python run_backend.py
```
OR
```bash
uvicorn backend.main:app --reload
```

Backend will be at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

**Terminal 2 - Start Frontend:**
```bash
python run_frontend.py
```
OR
```bash
streamlit run frontend/streamlit_app.py
```

Frontend will open automatically in your browser.

## ✅ Verify Installation

1. Check backend health:
   - Open `http://localhost:8000/health`
   - Should return `{"status": "healthy", "model_loaded": true}`

2. Check frontend:
   - Navigate to "Forecast" page
   - Click "Get Forecast"
   - Should see predictions

## 🐛 Troubleshooting

### "Model not found" error
- Make sure you ran `python backend/train_model.py` first
- Check that `models/forecaster.pkl` exists

### "No CSV files found"
- Verify CSV files are in `data/` folder
- Check file names contain `.csv` extension

### Import errors
- Make sure you installed all dependencies: `pip install -r requirements.txt`
- Use Python 3.8 or higher

### Port already in use
- Change port in `run_backend.py` or use: `uvicorn backend.main:app --port 8001`

## 📚 Next Steps

1. Explore the **Forecast** page to see predictions
2. Try the **What-If Simulator** with different scenarios
3. Generate slots in the **Marketplace**
4. Check out API docs at `http://localhost:8000/docs`

Happy coding! 🎉

