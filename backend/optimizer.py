"""
Cargo Allocation Optimizer - Multi-flight cargo allocation with priority-based scheduling.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
from .flight_database import FlightDatabase


@dataclass
class CargoRequest:
    """Cargo booking request."""
    request_id: str
    weight: float  # kg
    volume: float  # m³
    priority: int  # 1-5, higher = more priority (higher gets earlier flights)
    customer_type: str  # 'premium', 'standard', 'spot'
    origin: Optional[str] = None  # Origin airport (optional filter)
    destination: Optional[str] = None  # Destination airport (optional filter)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AllocationResult:
    """Result of cargo allocation."""
    request_id: str
    allocated: bool
    weight: float
    volume: float
    flight_number: Optional[str] = None
    flight_date: Optional[str] = None
    reason: Optional[str] = None  # Reason if not allocated
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CargoOptimizer:
    """
    Optimizer for efficient cargo allocation across multiple flights.
    
    Key features:
    - Allocates to earliest available flight first
    - Higher priority cargo gets earlier flights
    - Lower priority + high weight cargo moves to later flights when flight is full
    - Updates flight database when cargo is allocated
    """
    
    # Threshold for considering a flight "near full"
    NEAR_FULL_THRESHOLD = 0.85  # 85% utilization
    
    def __init__(self, flight_db: FlightDatabase = None):
        """Initialize optimizer with flight database."""
        self.flight_db = flight_db or FlightDatabase()
    
    def get_available_flights(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """Get available flights sorted by date (earliest first)."""
        return self.flight_db.get_available_flights(
            origin=origin,
            destination=destination,
            from_date=from_date,
            to_date=to_date
        )
    
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
        
        Algorithm:
        1. Sort cargo requests by priority (highest first)
        2. For each request, try to allocate to earliest available flight
        3. If flight is near full and cargo is low priority + bulky, use later flight
        4. Update database if commit=True
        
        Args:
            cargo_requests: List of cargo request dictionaries
            origin: Filter flights by origin
            destination: Filter flights by destination
            from_date: Start date for available flights
            to_date: End date for available flights
            commit: Whether to commit allocations to database
            
        Returns:
            Dictionary with allocations and statistics
        """
        # Convert to CargoRequest objects
        requests = []
        for req in cargo_requests:
            requests.append(CargoRequest(
                request_id=req.get('request_id', f"REQ{len(requests)+1:03d}"),
                weight=req.get('weight', 0),
                volume=req.get('volume', 0),
                priority=req.get('priority', 3),
                customer_type=req.get('customer_type', 'standard'),
                origin=req.get('origin', origin),
                destination=req.get('destination', destination)
            ))
        
        # Get available flights (sorted by date)
        available_flights = self.get_available_flights(
            origin=origin,
            destination=destination,
            from_date=from_date,
            to_date=to_date
        )
        
        if not available_flights:
            return {
                'allocations': [
                    AllocationResult(
                        request_id=r.request_id,
                        allocated=False,
                        weight=r.weight,
                        volume=r.volume,
                        reason="No flights available for the route/dates"
                    ).to_dict() for r in requests
                ],
                'statistics': {
                    'total_allocated_weight': 0,
                    'total_allocated_volume': 0,
                    'allocated_count': 0,
                    'rejected_count': len(requests),
                    'flights_used': 0
                },
                'flight_updates': []
            }
        
        # Sort requests by priority (descending) - higher priority first
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
        
        # Track flight capacities (copy to not affect original during planning)
        flight_capacity = {
            (f['flight_number'], f['flight_date']): {
                'available_weight': f['available_weight_kg'],
                'available_volume': f['available_volume_m3'],
                'max_weight': f['max_cargo_weight_kg'],
                'max_volume': f['max_cargo_volume_m3'],
                'flight_data': f
            }
            for f in available_flights
        }
        
        # Allocate cargo
        allocations = []
        flight_updates = {}  # Track updates per flight
        
        for req in sorted_requests:
            allocated = False
            
            # Determine if this is a "bulky" cargo (large weight or volume)
            avg_weight = np.mean([r.weight for r in requests]) if requests else 100
            avg_volume = np.mean([r.volume for r in requests]) if requests else 1
            is_bulky = req.weight > avg_weight * 1.5 or req.volume > avg_volume * 1.5
            is_low_priority = req.priority <= 2
            
            # Try flights in order (earliest first)
            for flight_key in sorted(flight_capacity.keys(), key=lambda k: k[1]):  # Sort by date
                flight = flight_capacity[flight_key]
                
                # Check capacity
                if req.weight > flight['available_weight'] or req.volume > flight['available_volume']:
                    continue
                
                # Calculate current utilization
                used_weight = flight['max_weight'] - flight['available_weight']
                weight_utilization = used_weight / flight['max_weight'] if flight['max_weight'] > 0 else 0
                
                used_volume = flight['max_volume'] - flight['available_volume']
                volume_utilization = used_volume / flight['max_volume'] if flight['max_volume'] > 0 else 0
                
                is_near_full = weight_utilization >= self.NEAR_FULL_THRESHOLD or volume_utilization >= self.NEAR_FULL_THRESHOLD
                
                # If flight is near full and cargo is low priority + bulky, try later flight
                if is_near_full and is_low_priority and is_bulky:
                    # Check if there are later flights with more space
                    later_flights = [
                        k for k in flight_capacity.keys()
                        if k[1] > flight_key[1]  # Later date
                        and flight_capacity[k]['available_weight'] >= req.weight
                        and flight_capacity[k]['available_volume'] >= req.volume
                    ]
                    if later_flights:
                        continue  # Skip this flight, try next
                
                # Allocate to this flight
                flight['available_weight'] -= req.weight
                flight['available_volume'] -= req.volume
                
                flight_number, flight_date = flight_key
                
                # Track update for this flight
                if flight_key not in flight_updates:
                    flight_updates[flight_key] = {
                        'flight_number': flight_number,
                        'flight_date': flight_date,
                        'weight_added': 0,
                        'volume_added': 0,
                        'requests': []
                    }
                
                flight_updates[flight_key]['weight_added'] += req.weight
                flight_updates[flight_key]['volume_added'] += req.volume
                flight_updates[flight_key]['requests'].append(req.request_id)
                
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=True,
                    weight=req.weight,
                    volume=req.volume,
                    flight_number=flight_number,
                    flight_date=flight_date
                ))
                
                allocated = True
                break
            
            if not allocated:
                allocations.append(AllocationResult(
                    request_id=req.request_id,
                    allocated=False,
                    weight=req.weight,
                    volume=req.volume,
                    reason="No flight with sufficient capacity"
                ))
        
        # Commit to database if requested
        if commit:
            for flight_key, update in flight_updates.items():
                success, msg = self.flight_db.update_flight_cargo(
                    flight_number=update['flight_number'],
                    flight_date=update['flight_date'],
                    cargo_weight_to_add=update['weight_added'],
                    cargo_volume_to_add=update['volume_added']
                )
                update['committed'] = success
                update['message'] = msg
        
        # Calculate statistics
        allocated_results = [a for a in allocations if a.allocated]
        rejected_results = [a for a in allocations if not a.allocated]
        
        statistics = {
            'total_allocated_weight': sum(a.weight for a in allocated_results),
            'total_allocated_volume': sum(a.volume for a in allocated_results),
            'allocated_count': len(allocated_results),
            'rejected_count': len(rejected_results),
            'flights_used': len(flight_updates),
            'total_requests': len(requests)
        }
        
        return {
            'allocations': [a.to_dict() for a in allocations],
            'statistics': statistics,
            'flight_updates': list(flight_updates.values())
        }
    
    def suggest_pricing(
        self,
        available_capacity: float,
        predicted_demand: float,
        confidence: float,
        base_price: float = 2.0
    ) -> Dict:
        """
        Suggest dynamic pricing based on supply/demand.
        
        Args:
            available_capacity: Available cargo capacity (kg)
            predicted_demand: Predicted cargo demand (kg)
            confidence: Prediction confidence (0-1)
            base_price: Base price per kg
            
        Returns:
            Dictionary with pricing suggestions
        """
        # Demand-to-supply ratio
        demand_ratio = predicted_demand / available_capacity if available_capacity > 0 else 0
        
        # Price multiplier based on demand
        if demand_ratio < 0.5:
            price_multiplier = 0.8
            pricing_strategy = "discount"
        elif demand_ratio < 0.8:
            price_multiplier = 1.0
            pricing_strategy = "normal"
        elif demand_ratio < 1.2:
            price_multiplier = 1.3
            pricing_strategy = "premium"
        else:
            price_multiplier = 1.5
            pricing_strategy = "surge"
        
        # Adjust for confidence
        confidence_multiplier = 0.9 + (confidence * 0.2)
        
        # Final price
        suggested_price = base_price * price_multiplier * confidence_multiplier
        
        return {
            'suggested_price_per_kg': round(suggested_price, 2),
            'base_price': base_price,
            'price_multiplier': round(price_multiplier, 2),
            'confidence_multiplier': round(confidence_multiplier, 2),
            'pricing_strategy': pricing_strategy,
            'demand_ratio': round(demand_ratio, 2),
            'demand_supply_balance': 'oversupply' if demand_ratio < 0.8 else 'balanced' if demand_ratio < 1.2 else 'shortage'
        }
    
    def get_flight_utilization_summary(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """Get utilization summary for flights."""
        flights = self.get_available_flights(origin, destination, from_date, to_date)
        
        summary = []
        for f in flights:
            weight_util = ((f['max_cargo_weight_kg'] - f['available_weight_kg']) / f['max_cargo_weight_kg'] * 100) if f['max_cargo_weight_kg'] > 0 else 0
            volume_util = ((f['max_cargo_volume_m3'] - f['available_volume_m3']) / f['max_cargo_volume_m3'] * 100) if f['max_cargo_volume_m3'] > 0 else 0
            
            summary.append({
                'flight_number': f['flight_number'],
                'flight_date': f['flight_date'],
                'origin': f['origin'],
                'destination': f['destination'],
                'aircraft_type': f['aircraft_type'],
                'weight_utilization_pct': round(weight_util, 1),
                'volume_utilization_pct': round(volume_util, 1),
                'available_weight_kg': round(f['available_weight_kg'], 1),
                'available_volume_m3': round(f['available_volume_m3'], 1),
                'is_near_full': weight_util >= 85 or volume_util >= 85
            })
        
        return summary
    
    def reset_allocations(self):
        """Reset all allocations in the database."""
        self.flight_db.reset_allocations()
