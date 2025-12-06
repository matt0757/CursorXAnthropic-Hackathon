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
    description="Dynamic Cargo Capacity Forecaster + Multi-Flight Cargo Optimizer + Marketplace",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    cargo_requests: List[Dict]
    origin: Optional[str] = None
    destination: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    commit: bool = True


class PricingSuggestionRequest(BaseModel):
    available_capacity: float
    predicted_demand: float
    confidence: float


class ReservationRequest(BaseModel):
    customer_info: Optional[Dict] = None


class FlightFilterRequest(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None


class LoadFlightsRequest(BaseModel):
    filepath: str


class AddFlightsRequest(BaseModel):
    flights: List[Dict]


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cargo Capacity Forecaster API v2.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "flights": "/flights",
            "flights/utilization": "/flights/utilization",
            "routes": "/routes",
            "optimize": "/marketplace/optimize",
            "pricing": "/marketplace/pricing-suggestion",
            "generate_slots": "/marketplace/generate-slots",
            "reserve": "/marketplace/reserve/{slot_id}",
            "reset": "/marketplace/reset",
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
    """Predict cargo capacity for a flight."""
    forecaster = get_forecaster()
    try:
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


@app.get("/flights")
async def get_flights(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """
    Get available flights with cargo capacity.
    
    Query params:
    - origin: Filter by origin airport code
    - destination: Filter by destination airport code
    - from_date: Start date (YYYY-MM-DD)
    - to_date: End date (YYYY-MM-DD)
    """
    try:
        flights = marketplace.get_available_flights(
            origin=origin,
            destination=destination,
            from_date=from_date,
            to_date=to_date
        )
        return {
            "flights": flights,
            "total": len(flights),
            "flights_loaded": marketplace.is_flights_loaded()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/flights/load")
async def load_flights(request: LoadFlightsRequest):
    """
    Load flights from a CSV file.
    
    Expected CSV columns (matching sample dataset):
    - flight_number, flight_date, origin, destination
    - tail_number, aircraft_type
    - gross_weight_cargo_kg (optional, defaults to 0)
    - gross_volume_cargo_m3 (optional, defaults to 0)
    - passenger_count, baggage_weight_kg, fuel_weight_kg, etc.
    """
    try:
        count = marketplace.load_flights_from_csv(request.filepath)
        return {
            "success": True,
            "message": f"Loaded {count} flights from {request.filepath}",
            "flight_count": count
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/flights/add")
async def add_flights(request: AddFlightsRequest):
    """
    Add flights manually.
    
    Each flight should have:
    - flight_number, flight_date, origin, destination
    - tail_number, aircraft_type
    - gross_weight_cargo_kg (optional, defaults to 0)
    - gross_volume_cargo_m3 (optional, defaults to 0)
    """
    try:
        count = marketplace.add_flights(request.flights)
        return {
            "success": True,
            "message": f"Added {count} flights",
            "flight_count": marketplace.get_flight_count()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/flights/status")
async def get_flights_status():
    """Check if flights are loaded and get count."""
    return {
        "flights_loaded": marketplace.is_flights_loaded(),
        "flight_count": marketplace.get_flight_count()
    }


@app.get("/flights/utilization")
async def get_flight_utilization(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """Get flight utilization summary."""
    try:
        utilization = marketplace.get_flight_utilization(
            origin=origin,
            destination=destination,
            from_date=from_date,
            to_date=to_date
        )
        return {
            "utilization": utilization,
            "total": len(utilization)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/routes")
async def get_routes():
    """Get available routes (origin-destination pairs)."""
    try:
        routes = marketplace.get_routes()
        return {"routes": routes}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/marketplace/generate-slots")
async def generate_slots(request: MarketplaceGenerateRequest):
    """Generate sellable cargo slots from predicted cargo."""
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
    """Reserve a cargo slot."""
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
    Optimize cargo allocation across multiple flights.
    
    Features:
    - Allocates cargo to earliest available flights
    - Higher priority cargo gets earlier flights
    - Low priority + bulky cargo moves to later flights when flight is near full
    - Updates database with new cargo allocations
    
    Request body:
    - cargo_requests: List of cargo requests with weight, volume, priority (1-5), customer_type
    - origin: Filter flights by origin (optional)
    - destination: Filter flights by destination (optional)
    - from_date: Start date YYYY-MM-DD (optional)
    - to_date: End date YYYY-MM-DD (optional)
    - commit: Whether to save allocations to database (default: true)
    """
    try:
        result = marketplace.optimize_allocation(
            cargo_requests=request.cargo_requests,
            origin=request.origin,
            destination=request.destination,
            from_date=request.from_date,
            to_date=request.to_date,
            commit=request.commit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/marketplace/pricing-suggestion")
async def get_pricing_suggestion(request: PricingSuggestionRequest):
    """Get dynamic pricing suggestions based on supply/demand."""
    try:
        result = marketplace.suggest_pricing(
            available_capacity=request.available_capacity,
            predicted_demand=request.predicted_demand,
            confidence=request.confidence
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/marketplace/reset")
async def reset_allocations():
    """Reset all cargo allocations (reload original data)."""
    try:
        marketplace.reset_allocations()
        return {"success": True, "message": "All allocations reset"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
