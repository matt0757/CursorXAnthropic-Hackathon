"""
FastAPI Backend - Main application entry point.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import uvicorn

from .forecaster import CargoForecaster
from .simulator import WhatIfSimulator
from .marketplace import CargoMarketplace

app = FastAPI(
    title="Cargo Capacity Forecaster API",
    description="Dynamic Cargo Capacity Forecaster + What-If Simulator + Cargo Marketplace",
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
simulator = None
marketplace = CargoMarketplace()

def get_forecaster():
    """Lazy load forecaster (loads model on first use)."""
    global forecaster, simulator
    if forecaster is None:
        try:
            forecaster = CargoForecaster()
            simulator = WhatIfSimulator(forecaster)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded. Please train the model first: {str(e)}"
            )
    return forecaster, simulator

# Pydantic schemas
class SimulationRequest(BaseModel):
    changes: Dict
    base_template: Optional[Dict] = None

class MarketplaceGenerateRequest(BaseModel):
    predicted_cargo: float
    confidence: float = 0.8
    slot_size_kg: float = 20.0

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
            "simulate": "/simulate",
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
    
    Formula: remaining_cargo = min(max_weight, max_volume) - (baggage + cargo)
    - Baggage is a function of passenger_count
    - Remaining cargo is what can be sold (e.g., to Shopee)
    - Total capacity varies by aircraft type
    
    Expected flight_data fields:
    - passenger_count (required): Number of passengers (determines baggage)
    - tail_number (optional): Aircraft tail number for capacity lookup
    - existing_cargo_weight_kg (optional): Current cargo weight already on flight (default: 0)
    - existing_cargo_volume_m3 (optional): Current cargo volume already on flight (default: 0)
    - year, month, day_of_week, etc.
    """
    forecaster, _ = get_forecaster()
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

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    """
    What-If Simulator endpoint.
    
    Apply scenario changes and get updated predictions.
    """
    _, simulator = get_forecaster()
    try:
        result = simulator.simulate(request.changes, request.base_template)
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
    forecaster, _ = get_forecaster()
    try:
        top_features = forecaster.get_feature_importance(top_n=10)
        return {"top_features": top_features}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

