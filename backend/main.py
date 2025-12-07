"""
FastAPI Backend - Main application entry point.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

from .forecaster import CargoForecaster
from .marketplace import CargoMarketplace

app = FastAPI(
    title="Cargo Capacity Forecaster API",
    description="Dynamic Cargo Capacity Forecaster + Cargo Optimizer + Marketplace",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
forecaster = None
marketplace = CargoMarketplace()

def get_forecaster():
    """Lazy load forecaster (loads model on first use)."""
    global forecaster
    if forecaster is None:
        try:
            forecaster = CargoForecaster()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded. Please train the model first: {str(e)}"
            )
    return forecaster

# Pydantic schemas
class MarketplaceGenerateRequest(BaseModel):
    predicted_cargo: float
    confidence: float = 0.8
    slot_size_kg: float = 20.0

class OptimizationRequest(BaseModel):
    available_weight: float
    available_volume: float
    cargo_requests: List[Dict]
    strategy: str = 'balanced'

class PricingSuggestionRequest(BaseModel):
    available_capacity: float
    predicted_demand: float
    confidence: float

class ReservationRequest(BaseModel):
    customer_info: Optional[Dict] = None

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cargo Capacity Forecaster API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "optimize": "/marketplace/optimize",
            "pricing": "/marketplace/pricing-suggestion",
            "generate_slots": "/marketplace/generate-slots",
            "reserve": "/marketplace/reserve/{slot_id}",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        get_forecaster()
        return {"status": "healthy", "model_loaded": True}
    except:
        return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
async def predict(flight_data: Dict):
    """
    Predict cargo capacity for a flight.
    
    Features:
    - Predicts baggage weight (function of passenger_count)
    - Predicts cargo demand (future cargo bookings)
    - Predicts cargo volume
    - Calculates remaining capacity: min(max_weight, max_volume) - (baggage + predicted_cargo)
    
    Expected flight_data fields:
    - passenger_count (required): Number of passengers (determines baggage)
    - tail_number (optional): Aircraft tail number for capacity lookup
    - existing_cargo_weight_kg (optional): Current cargo weight already on flight (default: 0)
    - existing_cargo_volume_m3 (optional): Current cargo volume already on flight (default: 0)
    - year, month, day_of_week, etc.
    """
    forecaster = get_forecaster()
    try:
        # Extract optional existing cargo parameters
        existing_cargo_weight = flight_data.get('existing_cargo_weight_kg', 0.0)
        existing_cargo_volume = flight_data.get('existing_cargo_volume_m3', 0.0)
        
        result = forecaster.predict(
            flight_data,
            existing_cargo_weight_kg=existing_cargo_weight,
            existing_cargo_volume_m3=existing_cargo_volume
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/marketplace/generate-slots")
async def generate_slots(request: MarketplaceGenerateRequest):
    """
    Generate sellable cargo slots from predicted cargo.
    """
    try:
        slots = marketplace.generate_slots(
            predicted_cargo=request.predicted_cargo,
            confidence=request.confidence,
            slot_size_kg=request.slot_size_kg
        )
        return {"slots": slots, "total_slots": len(slots)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/marketplace/reserve/{slot_id}")
async def reserve_slot(slot_id: str, request: ReservationRequest = ReservationRequest()):
    """
    Reserve a cargo slot.
    """
    try:
        reservation = marketplace.reserve_slot(
            slot_id=slot_id,
            customer_info=request.customer_info
        )
        return {
            "success": True,
            "reservation": reservation,
            "message": f"Slot {slot_id} reserved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/marketplace/reservations/{slot_id}")
async def get_reservation(slot_id: str):
    """Get reservation details."""
    reservation = marketplace.get_reservation(slot_id)
    if reservation is None:
        raise HTTPException(status_code=404, detail="Reservation not found")
    return reservation

@app.get("/feature-importance")
async def feature_importance():
    """Get feature importance for explainability."""
    forecaster = get_forecaster()
    try:
        top_features = forecaster.get_feature_importance(top_n=10)
        return {"top_features": top_features}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/marketplace/optimize")
async def optimize_allocation(request: OptimizationRequest):
    """
    Optimize cargo allocation for multiple requests.
    
    Strategies:
    - revenue_max: Maximize total revenue
    - utilization_max: Maximize capacity utilization
    - priority_first: Prioritize by customer priority
    - balanced: Balance revenue, utilization, and priority
    
    Returns allocation results and statistics.
    """
    try:
        result = marketplace.optimize_allocation(
            available_weight=request.available_weight,
            available_volume=request.available_volume,
            cargo_requests=request.cargo_requests,
            strategy=request.strategy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/marketplace/pricing-suggestion")
async def get_pricing_suggestion(request: PricingSuggestionRequest):
    """
    Get dynamic pricing suggestions based on supply/demand.
    
    Returns recommended pricing strategy and multipliers.
    """
    try:
        result = marketplace.suggest_pricing(
            available_capacity=request.available_capacity,
            predicted_demand=request.predicted_demand,
            confidence=request.confidence
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)