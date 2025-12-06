# Dynamic Cargo Capacity Forecaster + What-If Simulator + Cargo Marketplace

A comprehensive MVP solution for predicting cargo capacity, simulating what-if scenarios, and managing a cargo marketplace.

## 🚀 Features

1. **Dynamic Cargo Capacity Forecaster (ML)**
   - Automatically detects and loads CSV datasets
   - Trains LightGBM regression models for baggage weight and remaining cargo prediction
   - Provides predictions with confidence intervals
   - Feature importance analysis for explainability

2. **What-If Simulator**
   - Interactive scenario simulation
   - Real-time prediction updates based on parameter changes
   - Explanation of top contributing factors

3. **Cargo Marketplace**
   - Converts predicted cargo into sellable slots
   - Dynamic pricing based on risk/confidence
   - In-memory reservation system

4. **Frontend (Streamlit)**
   - Forecast Page: Upload parameters and get predictions
   - What-If Simulator: Interactive sliders for scenario testing
   - Marketplace: Generate and reserve cargo slots

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

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Installation Steps

1. **Clone/Download the repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the ML model:**
   ```bash
   python backend/train_model.py
   ```
   
   This will:
   - Automatically detect CSV files in the `data/` folder
   - Clean and preprocess the data
   - Train LightGBM models
   - Save the model to `models/forecaster.pkl`

4. **Start the FastAPI backend:**
   ```bash
   uvicorn backend.main:app --reload
   ```
   
   The API will be available at `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

5. **Start the Streamlit frontend (in a new terminal):**
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
   
   The UI will open in your browser automatically.

## 📊 Dataset Format

The training script automatically detects CSV files in the `data/` folder. Expected columns:

- `flight_number`: Flight identifier
- `flight_date`: Date in YYYY-MM-DD format
- `origin`, `destination`: Airport codes
- `tail_number`: Aircraft tail number
- `aircraft_type`: Aircraft model
- `passenger_count`: Number of passengers
- `baggage_weight_kg`: Total baggage weight (target variable)
- `gross_weight_cargo_kg`: Current cargo weight
- `fuel_weight_kg`: Fuel weight
- `fuel_price_per_kg`, `cargo_price_per_kg`: Pricing data

The script will also load aircraft capacity data from files matching `*aircraft tail*.csv` pattern.

## 🔌 API Endpoints

### Health Check
- **GET** `/health` - Check API and model status

### Predictions
- **POST** `/predict` - Predict cargo capacity for a flight
  ```json
  {
    "passenger_count": 150,
    "month": 6,
    "day_of_week": 2,
    ...
  }
  ```

### What-If Simulator
- **POST** `/simulate` - Run scenario simulation
  ```json
  {
    "changes": {
      "expected_passengers": 200,
      "group_travel_ratio": 0.3,
      "holiday_flag": 1,
      "delay_probability": 0.4
    }
  }
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

response = requests.post("http://localhost:8000/predict", json={
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
    "origin_encoded": 0,
    "destination_encoded": 0,
    "tail_number_encoded": 0,
    "aircraft_type_encoded": 0,
    "fuel_weight_kg": 5000,
    "fuel_price_per_kg": 0.85,
    "cargo_price_per_kg": 1.5
})

print(response.json())
```

### What-If Simulation

```python
response = requests.post("http://localhost:8000/simulate", json={
    "changes": {
        "expected_passengers": 200,
        "group_travel_ratio": 0.3,
        "holiday_flag": 1,
        "delay_probability": 0.4
    }
})

print(response.json())
```

## 🧪 Testing

1. **Test API health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **View interactive API docs:**
   Open `http://localhost:8000/docs` in your browser

3. **Use the Streamlit UI:**
   - Navigate between Forecast, What-If Simulator, and Marketplace pages
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
