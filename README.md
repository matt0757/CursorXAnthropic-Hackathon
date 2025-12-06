# Dynamic Cargo Capacity Forecaster + Optimizer

A comprehensive solution for predicting cargo capacity, optimizing cargo allocation, and managing dynamic pricing.

## 🚀 Key Features

### 1. **Cargo Demand Forecasting (ML)**
   - Predicts future **baggage weight** (function of passenger count)
   - Predicts future **cargo demand** (weight & volume)
   - Ensemble ML models with 95% confidence intervals
   - Feature importance for explainability

### 2. **Cargo Allocation Optimizer** ⭐ NEW
   - 4 optimization strategies:
     - **Balanced** (recommended): Multi-objective optimization
     - **Revenue Max**: Maximize profit
     - **Utilization Max**: Fill all available space
     - **Priority First**: VIP customers get priority
   - Handles dual constraints (weight AND volume)
   - Real-time allocation results

### 3. **Dynamic Pricing** ⭐ NEW
   - Supply/demand-based pricing suggestions
   - 4 pricing tiers: Discount → Normal → Premium → Surge
   - Confidence-adjusted pricing

### 4. **Frontend (Streamlit)**
   - **Forecast Page**: Get capacity predictions with adjustable parameters
   - **Cargo Optimizer** ⭐ NEW: Optimize allocation with visualizations
   - **Marketplace**: Generate and reserve slots

## 📁 Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── forecaster.py        # ML model loader and predictor
│   ├── simulator.py         # What-If simulator logic
│   ├── marketplace.py       # Marketplace slot generation
│   └── train_model.py       # ML training script
├── frontend/
│   └── streamlit_app.py     # Streamlit UI
├── models/
│   └── forecaster.pkl       # Trained model (generated after training)
├── data/
│   └── *.csv                # Dataset files
├── requirements.txt
└── README.md
```

## 🛠️ Quick Start (3 Steps)

### Prerequisites
- Python 3.8+
- Virtual environment activated

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Train Models (REQUIRED - includes new cargo demand prediction)
```powershell
python backend/train_model.py
```

**What it does:**
- Automatically detects CSV files in `data/` folder
- Trains ensemble models for baggage, cargo demand, cargo volume
- Saves to `models/forecaster.pkl`
- **Time:** ~30-60 seconds

**Expected output:**
```
Training base models for baggage...
Training base models for cargo_demand...
Training base models for cargo_volume...
✓ Ensemble models saved to models/forecaster.pkl
```

### Step 3: Run the Application

**Terminal 1 - Backend:**
```powershell
python run_backend.py
```
Backend at: `http://localhost:8000` | API docs: `http://localhost:8000/docs`

**Terminal 2 - Frontend:**
```powershell
python run_frontend.py
```
Opens automatically in your browser at `http://localhost:8501`
## 🎯 How to Use New Features

### Cargo Demand Forecasting
1. Go to **Forecast** page
2. Enter flight parameters
3. Click "Get Forecast"
4. **NEW:** See "Predicted Cargo Demand" and "Predicted Cargo Volume"

### Cargo Allocation Optimizer
1. Go to **Cargo Optimizer** page
2. Set available capacity (e.g., 1000kg, 10m³)
3. Add cargo requests (weight, volume, priority, revenue, customer type)
4. Select optimization strategy (Balanced recommended)
5. Click "🚀 Run Optimization"
6. View allocation results, revenue, and utilization

### Dynamic Pricing
1. On **Cargo Optimizer** page → "💵 Pricing Suggestion"
## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API and model status |
| `/predict` | POST | Predict cargo capacity (now includes cargo demand) |
| `/simulate` | POST | Run what-if scenarios |
| `/marketplace/generate-slots` | POST | Generate cargo slots |
| `/marketplace/optimize` ⭐ | POST | **Optimize cargo allocation** |
| `/marketplace/pricing-suggestion` ⭐ | POST | **Get dynamic pricing** |
| `/marketplace/reserve/{slot_id}` | POST | Reserve a slot |
| `/feature-importance` | GET | Get feature importance |

**Explore API:** `http://localhost:8000/docs`
  ```

### Marketplace
- **POST** `/marketplace/generate-slots` - Generate cargo slots
  ```json
  {
    "predicted_cargo": 500.0,
    "confidence": 0.8,
    "slot_size_kg": 20.0
  }
  ```

- **POST** `/marketplace/reserve/{slot_id}` - Reserve a slot
- **GET** `/marketplace/reservations/{slot_id}` - Get reservation details

### Explainability
- **GET** `/feature-importance` - Get top contributing features

## 🎯 Usage Examples

### Training a Model

```bash
python backend/train_model.py
```

### Making Predictions via API

```python
import requests

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| "Model not found" | Run `python backend/train_model.py` first |
| "API is not running" | Start backend: `python run_backend.py` |
| "No module named..." | Install dependencies: `pip install -r requirements.txt` |
| Import errors | Restart backend server |

## 📊 What's New vs Original?

| Feature | Before | Now |
|---------|--------|-----|
| Predicts cargo demand? | ❌ No | ✅ Yes (weight & volume) |
| Optimizes allocation? | ❌ Equal slots only | ✅ 4 strategies |
| Dynamic pricing? | ❌ Risk-based only | ✅ Supply/demand based |
| Multi-objective? | ❌ No | ✅ Revenue + Priority + Utilization |- Navigate between Forecast, What-If Simulator, and Marketplace pages
   - Adjust parameters and see real-time predictions

## 📝 Notes

- The model uses LightGBM for regression
- Confidence intervals are computed using bootstrap approximation
- Marketplace reservations are stored in-memory (reset on server restart)
- Feature importance is computed as the average of baggage and remaining cargo models
- Mock features (group_travel_ratio, holiday_flag, etc.) are generated during training if not present in dataset

## 🚧 Future Enhancements

- Database integration for persistent reservations
- Real-time data streaming
- Advanced explainability (SHAP values)
- Multi-model ensemble
- Automated retraining pipeline
- Authentication and authorization
- Historical data visualization

## 📄 License

This project is developed for the CursorXAnthropic Hackathon.

## 👥 Contributors

Built for the CursorXAnthropic Hackathon MVP.
## 📁 Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   ├── forecaster.py        # ML predictor with cargo demand
│   ├── optimizer.py         # ⭐ NEW: Cargo allocation optimizer
│   ├── marketplace.py       # Marketplace + optimization integration
│   └── train_model.py       # ML training (ensemble models)
├── frontend/
│   └── streamlit_app.py     # UI (3 pages: Forecast, Cargo Optimizer, Marketplace)
├── models/
│   └── forecaster.pkl       # Trained models (generated after training)
├── data/
│   └── *.csv                # Dataset files
└── requirements.txt
```

## 🎓 Technical Details

- **ML Stack**: Ensemble of LightGBM, Random Forest, Gradient Boosting, XGBoost, Ridge
- **Optimization**: Greedy knapsack for dual-constraint problem (O(n log n))
- **API**: FastAPI with Pydantic validation
- **Frontend**: Streamlit with Plotly visualizations
- **Confidence Intervals**: Bootstrap approximation (95% CI)

For detailed technical implementation, see `TECHNICAL_DETAILS.md`

## 📄 License

Developed for the CursorXAnthropic Hackathon.