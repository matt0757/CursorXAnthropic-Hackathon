"""
Cargo Marketplace - Generate slots and manage reservations.
"""
from typing import List, Dict, Optional
import uuid
from datetime import datetime

class CargoMarketplace:
    """Marketplace for selling cargo capacity slots."""
    
    def __init__(self):
        """Initialize marketplace with empty reservations."""
        self.reservations: Dict[str, Dict] = {}
        self.base_price_per_kg = 2.0  # Base price per kg
    
    def generate_slots(
        self, 
        predicted_cargo: float, 
        confidence: float = 0.8,
        slot_size_kg: float = 20.0
    ) -> List[Dict]:
        """
        Generate sellable cargo slots from predicted remaining cargo.
        
        Args:
            predicted_cargo: Predicted remaining cargo in kg
            confidence: Confidence score (0-1)
            slot_size_kg: Size of each slot in kg
            
        Returns:
            List of slot dictionaries
        """
        if predicted_cargo <= 0:
            return []
        
        # Calculate number of slots
        num_slots = int(predicted_cargo / slot_size_kg)
        if num_slots == 0:
            # At least create one slot with remaining cargo
            num_slots = 1
            slot_size_kg = predicted_cargo
        
        slots = []
        
        for i in range(num_slots):
            # Calculate risk factor (higher when confidence is low)
            risk_factor = 1.0 + (1.0 - confidence) * 0.5  # 1.0 to 1.5
            
            # Dynamic pricing based on risk
            price_per_kg = self.base_price_per_kg * risk_factor
            total_price = slot_size_kg * price_per_kg
            
            # Risk score (0-1, higher is riskier)
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
        
        # Adjust last slot to match remaining cargo exactly
        if num_slots > 1:
            total_slots_weight = slot_size_kg * (num_slots - 1)
            remaining = predicted_cargo - total_slots_weight
            if remaining > 0:
                slots[-1]['weight'] = round(remaining, 2)
                slots[-1]['price'] = round(remaining * slots[-1]['price_per_kg'], 2)
        
        return slots
    
    def reserve_slot(self, slot_id: str, customer_info: Optional[Dict] = None) -> Dict:
        """
        Reserve a cargo slot.
        
        Args:
            slot_id: ID of the slot to reserve
            customer_info: Optional customer information
            
        Returns:
            Reservation confirmation dictionary
        """
        # Check if slot is already reserved (would need to check from latest generation)
        # For MVP, we'll just create a reservation
        
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

