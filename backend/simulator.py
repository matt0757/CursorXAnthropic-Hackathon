"""
What-If Simulator - Apply scenario changes and recompute predictions.
"""
from typing import Dict, Optional
from .forecaster import CargoForecaster
import numpy as np

class WhatIfSimulator:
    """Simulator for what-if scenarios."""
    
    def __init__(self, forecaster: CargoForecaster):
        """Initialize simulator with a forecaster instance."""
        self.forecaster = forecaster
        
        # Base template flight (average values)
        self.base_template = {
            'passenger_count': 150,
            'year': 2024,
            'month': 6,
            'day_of_week': 2,  # Wednesday
            'day_of_month': 15,
            'is_weekend': 0,
            'group_travel_ratio': 0.2,
            'holiday_flag': 0,
            'delay_probability': 0.1,
            'weather_index': 0.5,
            'origin_encoded': 0,
            'destination_encoded': 0,
            'tail_number_encoded': 0,
            'aircraft_type_encoded': 0,
            'fuel_weight_kg': 5000,
            'fuel_price_per_kg': 0.85,
            'cargo_price_per_kg': 1.5
        }
    
    def simulate(self, changes: Dict, base_template: Optional[Dict] = None) -> Dict:
        """
        Apply scenario changes and return updated predictions.
        
        Args:
            changes: Dictionary with scenario deltas
            base_template: Optional base template (uses default if not provided)
            
        Returns:
            Dictionary with predictions and explanation
        """
        # Start with base template
        scenario = base_template.copy() if base_template else self.base_template.copy()
        
        # Apply changes
        for key, value in changes.items():
            if key in scenario:
                scenario[key] = value
            # Map common aliases
            elif key == 'expected_passengers':
                scenario['passenger_count'] = value
        
        # Make prediction
        prediction = self.forecaster.predict(scenario)
        
        # Get feature importance for explanation
        top_factors = self.forecaster.get_feature_importance(top_n=5)
        
        # Create explanation
        explanation = {
            'top_factors': top_factors,
            'scenario_changes': changes,
            'applied_template': scenario
        }
        
        return {
            'predicted_baggage': prediction['predicted_baggage'],
            'remaining_cargo': prediction['remaining_cargo'],
            'confidence': prediction['confidence'],
            'prediction_details': {
                'baggage_confidence_interval': {
                    'lower': prediction['predicted_baggage_lower'],
                    'upper': prediction['predicted_baggage_upper']
                },
                'cargo_confidence_interval': {
                    'lower': prediction['remaining_cargo_lower'],
                    'upper': prediction['remaining_cargo_upper']
                }
            },
            'explanation': explanation
        }

