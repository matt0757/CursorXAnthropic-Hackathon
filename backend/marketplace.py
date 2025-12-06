"""
Cargo Marketplace - Generate slots and manage reservations with multi-flight optimization.
"""
from typing import List, Dict, Optional
import uuid
from datetime import datetime
from .optimizer import CargoOptimizer
from .flight_database import FlightDatabase


class CargoMarketplace:
    """Marketplace for selling cargo capacity slots with multi-flight optimization."""
    
    def __init__(self):
        """Initialize marketplace with empty flight database (no auto-load)."""
        self.reservations: Dict[str, Dict] = {}
        self.base_price_per_kg = 2.0
        self.flight_db = FlightDatabase(auto_load=False)  # Don't auto-load historical data
        self.optimizer = CargoOptimizer(self.flight_db)
    
    def load_flights_from_csv(self, filepath: str) -> int:
        """
        Load flights from a CSV file.
        
        Args:
            filepath: Path to CSV file with future flights data
            
        Returns:
            Number of flights loaded
        """
        return self.flight_db.load_flights_from_csv(filepath)
    
    def add_flights(self, flights: List[Dict]) -> int:
        """
        Add flights manually.
        
        Args:
            flights: List of flight dictionaries
            
        Returns:
            Number of flights added
        """
        return self.flight_db.add_flights_batch(flights)
    
    def get_flight_count(self) -> int:
        """Get number of flights in database."""
        return self.flight_db.get_flight_count()
    
    def is_flights_loaded(self) -> bool:
        """Check if any flights are loaded."""
        return not self.flight_db.is_empty()
    
    def get_available_flights(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """Get available flights with capacity information."""
        return self.optimizer.get_available_flights(
            origin=origin,
            destination=destination,
            from_date=from_date,
            to_date=to_date
        )
    
    def get_flight_utilization(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """Get flight utilization summary."""
        return self.optimizer.get_flight_utilization_summary(
            origin=origin,
            destination=destination,
            from_date=from_date,
            to_date=to_date
        )
    
    def get_routes(self) -> List[Dict]:
        """Get available routes."""
        return self.flight_db.get_routes()
    
    def generate_slots(
        self, 
        predicted_cargo: float, 
        confidence: float = 0.8,
        slot_size_kg: float = 20.0
    ) -> List[Dict]:
        """
        Generate sellable cargo slots from predicted remaining cargo.
        """
        if predicted_cargo <= 0:
            return []
        
        num_slots = int(predicted_cargo / slot_size_kg)
        if num_slots == 0:
            num_slots = 1
            slot_size_kg = predicted_cargo
        
        slots = []
        
        for i in range(num_slots):
            risk_factor = 1.0 + (1.0 - confidence) * 0.5
            price_per_kg = self.base_price_per_kg * risk_factor
            total_price = slot_size_kg * price_per_kg
            risk_score = 1.0 - confidence
            
            slot = {
                'slot_id': f"slot_{str(uuid.uuid4())[:8]}",
                'weight': round(slot_size_kg, 2),
                'price': round(total_price, 2),
                'price_per_kg': round(price_per_kg, 2),
                'risk_score': round(risk_score, 3),
                'risk_factor': round(risk_factor, 2),
                'confidence': round(confidence, 3),
                'status': 'available'
            }
            
            slots.append(slot)
        
        if num_slots > 1:
            total_slots_weight = slot_size_kg * (num_slots - 1)
            remaining = predicted_cargo - total_slots_weight
            if remaining > 0:
                slots[-1]['weight'] = round(remaining, 2)
                slots[-1]['price'] = round(remaining * slots[-1]['price_per_kg'], 2)
        
        return slots
    
    def reserve_slot(self, slot_id: str, customer_info: Optional[Dict] = None) -> Dict:
        """Reserve a cargo slot."""
        reservation = {
            'slot_id': slot_id,
            'reserved_at': datetime.now().isoformat(),
            'customer_info': customer_info or {},
            'status': 'reserved'
        }
        
        self.reservations[slot_id] = reservation
        return reservation
    
    def get_reservation(self, slot_id: str) -> Optional[Dict]:
        """Get reservation details by slot ID."""
        return self.reservations.get(slot_id)
    
    def is_reserved(self, slot_id: str) -> bool:
        """Check if a slot is reserved."""
        return slot_id in self.reservations
    
    def optimize_allocation(
        self,
        cargo_requests: List[Dict],
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        commit: bool = True
    ) -> Dict:
        """
        Optimize cargo allocation across multiple flights.
        
        Args:
            cargo_requests: List of cargo request dictionaries with:
                - request_id: Unique identifier
                - weight: Weight in kg
                - volume: Volume in m³
                - priority: 1-5 (higher = earlier flights)
                - customer_type: 'premium', 'standard', 'spot'
            origin: Filter by origin airport
            destination: Filter by destination airport
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            commit: Whether to commit allocations to database
            
        Returns:
            Dictionary with allocations, statistics, and flight updates
        """
        return self.optimizer.optimize_allocation(
            cargo_requests=cargo_requests,
            origin=origin,
            destination=destination,
            from_date=from_date,
            to_date=to_date,
            commit=commit
        )
    
    def suggest_pricing(
        self,
        available_capacity: float,
        predicted_demand: float,
        confidence: float
    ) -> Dict:
        """Get dynamic pricing suggestions based on supply/demand."""
        return self.optimizer.suggest_pricing(
            available_capacity=available_capacity,
            predicted_demand=predicted_demand,
            confidence=confidence,
            base_price=self.base_price_per_kg
        )
    
    def reset_allocations(self):
        """Reset all allocations (reload original data)."""
        self.optimizer.reset_allocations()
