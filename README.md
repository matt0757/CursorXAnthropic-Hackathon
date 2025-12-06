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

### 5. **Cargo TMS Dashboard** ⭐ NEW
   - Professional Transportation Management System for airport cargo personnel
   - **Home Dashboard**: Real-time overview of flights, cargo requests, and KPIs
   - **Mesh Flights**: Network view with flight details, cargo status, and ULD management
   - **Flight Services**: Service tracking with AI-powered capacity forecasting
   - **Cargo Planning**: Booking management, optimization, and dynamic pricing
   - **Analytics**: Performance metrics, route analysis, and reporting

## 📁 Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & API endpoints
│   ├── forecaster.py        # ML ensemble model loader and predictor
│   ├── optimizer.py         # Cargo allocation optimization algorithms
│   ├── marketplace.py       # Marketplace slot generation & management
│   └── train_model.py       # ML ensemble training script
├── frontend/
│   ├── streamlit_app.py     # Streamlit UI (3 pages: Forecast, Optimizer, Marketplace)
│   └── cargo_tms_app.py     # ⭐ NEW: TMS Dashboard for cargo personnel
├── models/
│   └── forecaster.pkl       # Trained ensemble model (generated after training)
├── data/
│   ├── *sample*.csv         # Flight data files (2023, 2024)
│   └── *aircraft tail*.csv  # Aircraft capacity reference
├── requirements.txt         # Python dependencies
├── run_backend.py          # Backend launcher script
├── run_frontend.py         # Frontend launcher script
├── run_tms.py              # ⭐ NEW: TMS Dashboard launcher
├── README.md               # This file
├── TECHNICAL_DETAILS.md    # Implementation details
└── ENSEMBLE_MODEL_INFO.md  # Model architecture documentation
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

**Terminal 3 - TMS Dashboard (Optional):**
```powershell
python run_tms.py
```
Opens the Cargo TMS Dashboard at `http://localhost:8502`

## 🎯 How to Use

### 1. Cargo Capacity Forecasting
1. Go to **Forecast** page
2. Enter flight parameters:
   - Passenger count (required)
   - Month, day of week, temporal features
   - Optional: aircraft type, tail number, origin, destination
   - Scenario parameters: group travel ratio, holiday flag, delay probability, weather index
3. Click "🚀 Get Forecast"
4. View predictions with 95% confidence intervals:
   - **Predicted Baggage** (function of passenger count)
   - **Predicted Cargo Demand** (future cargo bookings)
   - **Predicted Cargo Volume** (future cargo volume)
   - **Remaining Cargo Capacity** (what can be sold)

### 2. Cargo Allocation Optimizer
1. Go to **Cargo Optimizer** page
2. Set available capacity (e.g., 1000kg, 10m³)
3. Add cargo requests:
   - Click "➕ Add Cargo Request"
   - Enter: weight (kg), volume (m³), priority (1-5), revenue per kg ($), customer type
   - Add multiple requests
4. Select optimization strategy:
   - **Balanced** (recommended): Multi-objective optimization
   - **Revenue Max**: Maximize total profit
   - **Utilization Max**: Fill all available space
   - **Priority First**: VIP customers get priority
5. Click "🚀 Run Optimization"
6. View results:
   - Total revenue and utilization percentages
   - Allocated vs rejected requests
   - Visual capacity utilization charts

### 3. Dynamic Pricing
1. On **Cargo Optimizer** page, scroll to "💵 Pricing Suggestion"
2. Enter predicted demand and confidence level
3. Get pricing recommendation based on supply/demand ratio

### 4. Marketplace (Slot Generation)
1. Go to **Marketplace** page
2. Enter predicted remaining cargo capacity
3. Set confidence level and slot size
4. Click "🛒 Generate Slots"
5. View available slots with risk scores and prices
6. Reserve individual slots

### 5. Cargo TMS Dashboard ⭐ NEW
A professional Transportation Management System designed for airport cargo personnel.

**Launch:** `python run_tms.py` (runs on port 8502)

**Features:**
- **🏠 Home**: Real-time dashboard with flight status, cargo requests, and daily KPIs
- **📋 Mesh Flights**: Network overview with flight table, detailed flight info, ULD container tracking
- **🔄 Flight Services**: Service status tracking, AI-powered cargo forecasting with confidence intervals
- **📦 Cargo Planning**: Editable booking requests, optimization with 4 strategies, dynamic pricing
- **📊 Analytics**: Utilization trends, revenue by customer, route performance metrics

**Key Capabilities:**
- View active flights with capacity utilization bars
- Track pending cargo booking requests by priority
- Run AI forecasting to predict baggage and remaining capacity
- Optimize cargo allocation across multiple booking requests
- Get dynamic pricing suggestions based on supply/demand
- Analyze route and customer performance
## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with available endpoints list |
| `/health` | GET | Check API and model status |
| `/predict` | POST | Predict cargo capacity (includes baggage, cargo demand, volume, remaining capacity) |
| `/marketplace/generate-slots` | POST | Generate cargo slots from predicted capacity |
| `/marketplace/reserve/{slot_id}` | POST | Reserve a specific cargo slot |
| `/marketplace/reservations/{slot_id}` | GET | Get reservation details |
| `/marketplace/optimize` | POST | **Optimize cargo allocation with multiple strategies** |
| `/marketplace/pricing-suggestion` | POST | **Get dynamic pricing based on supply/demand** |
| `/feature-importance` | GET | Get top feature importance for explainability |

**Interactive API Documentation:** `http://localhost:8000/docs`

### Example API Request - Predict
```python
import requests

flight_data = {
    "passenger_count": 150,
    "year": 2024,
    "month": 6,
    "day_of_week": 2,
    "day_of_month": 15,
    "is_weekend": 0,
    "group_travel_ratio": 0.2,
    "holiday_flag": 0,
    "delay_probability": 0.1,
    "weather_index": 0.5,
    "fuel_weight_kg": 50000,
    "fuel_price_per_kg": 0.8,
    "cargo_price_per_kg": 2.0,
    "aircraft_type": "A330-300",  # optional
    "tail_number": "9M-XXX",  # optional
    "existing_cargo_weight_kg": 0.0,  # optional
    "existing_cargo_volume_m3": 0.0  # optional
}

response = requests.post("http://localhost:8000/predict", json=flight_data)
result = response.json()

print(f"Predicted Baggage: {result['predicted_baggage']:.0f} kg")
print(f"Predicted Cargo Demand: {result['predicted_cargo_demand']:.0f} kg")
print(f"Remaining Capacity: {result['remaining_cargo']:.0f} kg")
print(f"Confidence: {result['confidence']:.1%}")
```

### Example API Request - Optimize Allocation
```python

cargo_requests = [
    {"request_id": "REQ001", "weight": 300, "volume": 1.5, "priority": 5, 
     "revenue_per_kg": 3.0, "customer_type": "premium"},
    {"request_id": "REQ002", "weight": 400, "volume": 2.0, "priority": 3, 
     "revenue_per_kg": 2.0, "customer_type": "standard"},
    {"request_id": "REQ003", "weight": 500, "volume": 2.5, "priority": 2, 
     "revenue_per_kg": 1.5, "customer_type": "spot"}
]

response = requests.post(
    "http://localhost:8000/marketplace/optimize",
    json={
        "available_weight": 1000.0,
        "available_volume": 10.0,
        "cargo_requests": cargo_requests,
        "strategy": "balanced"  # or "revenue_max", "utilization_max", "priority_first"
    }
)

result = response.json()
print(f"Total Revenue: ${result['statistics']['total_revenue']:.2f}")
print(f"Weight Utilization: {result['statistics']['weight_utilization']:.1%}")
print(f"Allocated: {result['statistics']['allocated_count']} requests")
```

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| "Model not found" | Run `python backend/train_model.py` first |
| "API is not running" | Start backend: `python run_backend.py` |
| "No module named..." | Install dependencies: `pip install -r requirements.txt` |
| Port already in use | Change port in run_backend.py or run_frontend.py |
| Import errors | Restart backend server after making changes |
| XGBoost warning | Optional - install with `pip install xgboost` for better performance |

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
│   ├── streamlit_app.py     # UI (3 pages: Forecast, Cargo Optimizer, Marketplace)
│   └── cargo_tms_app.py     # ⭐ TMS Dashboard for airport cargo personnel
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