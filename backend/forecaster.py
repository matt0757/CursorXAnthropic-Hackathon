"""
Forecaster module - Loads model and makes predictions with confidence intervals.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import glob

class CargoForecaster:
    """Forecaster for cargo capacity predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize forecaster by loading trained ensemble model."""
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "forecaster.pkl"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Support both ensemble and single model formats
        if 'baggage_ensemble' in model_data:
            # New ensemble format
            self.baggage_ensemble = model_data['baggage_ensemble']
            self.remaining_ensemble = model_data['remaining_ensemble']
            self.ensemble_type = model_data.get('ensemble_type', 'weighted')
            self.use_ensemble = True
        else:
            # Legacy single model format
            self.baggage_model = model_data.get('baggage_model')
            self.remaining_model = model_data.get('remaining_model')
            self.use_ensemble = False
        
        self.feature_cols = model_data['feature_cols']
        self.label_encoders = model_data.get('label_encoders', {})
    
    def _load_aircraft_capacities(self):
        """Load aircraft capacity mappings from CSV file."""
        data_dir = Path(__file__).parent.parent / "data"
        capacity_files = glob.glob(str(data_dir / "*aircraft tail*.csv"))
        
        self.tail_to_max_weight = {}
        self.tail_to_max_volume = {}
        
        if capacity_files:
            capacity_df = pd.read_csv(capacity_files[0])
            capacity_df.columns = capacity_df.columns.str.strip()
            
            self.tail_to_max_weight = dict(zip(
                capacity_df['Tail Number'].str.strip(),
                capacity_df['Max Weight (Lower Deck) (kg)']
            ))
            self.tail_to_max_volume = dict(zip(
                capacity_df['Tail Number'].str.strip(),
                capacity_df['Max Volume (Lower Deck) (m³)']
            ))
    
    def _calculate_remaining_cargo(self, baggage_weight_kg: float, existing_cargo_weight_kg: float,
                                   existing_cargo_volume_m3: float, tail_number: str = None) -> float:
        """
        Calculate remaining cargo capacity considering both weight and volume constraints.
        
        Formula: remaining_cargo = min(
            max_weight - baggage - existing_cargo_weight,
            (max_volume - baggage_volume - existing_cargo_volume) * cargo_density
        )
        
        Args:
            baggage_weight_kg: Predicted baggage weight
            existing_cargo_weight_kg: Current cargo weight on flight
            existing_cargo_volume_m3: Current cargo volume on flight
            tail_number: Aircraft tail number for capacity lookup
            
        Returns:
            Remaining cargo capacity in kg (the binding constraint)
        """
        # Get aircraft capacities
        if tail_number and tail_number in self.tail_to_max_weight:
            max_weight = self.tail_to_max_weight[tail_number]
            max_volume = self.tail_to_max_volume.get(tail_number, float('inf'))
        else:
            # Fallback: use default values (will need aircraft_type or estimate)
            max_weight = 5000  # Conservative default
            max_volume = 50  # Conservative default
        
        # Calculate remaining weight capacity
        remaining_weight = max_weight - baggage_weight_kg - existing_cargo_weight_kg
        
        # Calculate baggage volume from weight
        baggage_volume = baggage_weight_kg / self.BAGGAGE_DENSITY
        
        # Calculate remaining volume capacity
        remaining_volume = max_volume - baggage_volume - existing_cargo_volume_m3
        
        # Convert remaining volume to equivalent weight
        remaining_volume_as_weight = remaining_volume * self.CARGO_DENSITY
        
        # Binding constraint is the minimum (whichever is more restrictive)
        remaining_cargo = min(remaining_weight, remaining_volume_as_weight)
        
        return max(0, remaining_cargo)  # Ensure non-negative
    
    def _prepare_features(self, flight_data: Dict) -> np.ndarray:
        """Convert flight data dictionary to feature array."""
        # Create a DataFrame row from the input
        row = pd.DataFrame([flight_data])
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            col_name = f'{col}_encoded'
            if col in row.columns:
                try:
                    # Try to transform, if value not seen, use most common
                    row[col_name] = encoder.transform([str(row[col].iloc[0])])[0]
                except ValueError:
                    # Unknown value - use 0 or most common
                    row[col_name] = 0
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0
        
        # Select features in correct order
        X = row[self.feature_cols].values
        return X
    
    def _predict_ensemble(self, X, ensemble_dict):
        """Make ensemble prediction using weighted average."""
        predictions = []
        weights = ensemble_dict['weights']
        models = ensemble_dict['models']
        
        for model_name, model in models.items():
            pred = model.predict(X)[0]
            weight = weights[model_name]
            predictions.append(pred * weight)
        
        return np.sum(predictions)
    
    def _get_ensemble_std(self, X, ensemble_dict):
        """Estimate prediction uncertainty from ensemble variance."""
        predictions = []
        models = ensemble_dict['models']
        
        for model_name, model in models.items():
            pred = model.predict(X)[0]
            predictions.append(pred)
        
        return np.std(predictions)
    
    def predict(self, flight_data: Dict, existing_cargo_weight_kg: float = 0.0, 
                existing_cargo_volume_m3: float = 0.0) -> Dict:
        """
        Predict baggage weight (function of passenger count) and remaining cargo capacity.
        
        Remaining cargo = min(max_weight, max_volume) - (baggage + cargo)
        The binding constraint is whichever is more restrictive.
        
        Args:
            flight_data: Dictionary with flight features (must include passenger_count)
            existing_cargo_weight_kg: Current cargo weight already on flight (default: 0)
            existing_cargo_volume_m3: Current cargo volume already on flight (default: 0)
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        X = self._prepare_features(flight_data)
        
        # Get tail number for capacity lookup if available
        tail_number = flight_data.get('tail_number')
        if tail_number and isinstance(tail_number, str):
            # If tail_number is encoded, we need to decode it
            # For now, assume it's the actual tail number string
            pass
        else:
            tail_number = None
        
        # Make predictions using ensemble or single model
        if self.use_ensemble:
            # Ensemble prediction for baggage (function of passenger count)
            baggage_pred = self._predict_ensemble(X, self.baggage_ensemble)
            
            # For remaining cargo, we'll calculate it from constraints rather than direct prediction
            # But we can still use the model's prediction as a baseline
            remaining_model_pred = self._predict_ensemble(X, self.remaining_ensemble)
            
            # Estimate uncertainty from ensemble variance
            baggage_std = self._get_ensemble_std(X, self.baggage_ensemble)
            remaining_model_std = self._get_ensemble_std(X, self.remaining_ensemble)
        else:
            # Legacy single model prediction
            baggage_pred = self.baggage_model.predict(X)[0]
            remaining_model_pred = self.remaining_model.predict(X)[0]
            
            # Bootstrap for confidence intervals
            baggage_samples = []
            remaining_samples = []
            
            try:
                n_trees = self.baggage_model.n_estimators_
                step = max(1, n_trees // 50)
                for tree_idx in range(step, n_trees + 1, step):
                    baggage_samples.append(self.baggage_model.predict(X, num_iteration=tree_idx)[0])
            except:
                baggage_samples = [baggage_pred]
            
            try:
                n_trees = self.remaining_model.n_estimators_
                step = max(1, n_trees // 50)
                for tree_idx in range(step, n_trees + 1, step):
                    remaining_samples.append(self.remaining_model.predict(X, num_iteration=tree_idx)[0])
            except:
                remaining_samples = [remaining_model_pred]
            
            baggage_std = np.std(baggage_samples) if baggage_samples else baggage_pred * 0.1
            remaining_model_std = np.std(remaining_samples) if remaining_samples else remaining_model_pred * 0.1
        
        # Calculate remaining cargo based on actual constraints (weight OR volume)
        # This is what Shopee can actually sell
        remaining_cargo = self._calculate_remaining_cargo(
            baggage_weight_kg=baggage_pred,
            existing_cargo_weight_kg=existing_cargo_weight_kg,
            existing_cargo_volume_m3=existing_cargo_volume_m3,
            tail_number=tail_number
        )
        
        # For confidence intervals, propagate uncertainty from baggage prediction
        # Use the constraint calculation with upper/lower baggage bounds
        baggage_lower = baggage_pred - 1.96 * baggage_std
        baggage_upper = baggage_pred + 1.96 * baggage_std
        
        # Calculate remaining cargo for lower and upper baggage bounds
        remaining_lower = self._calculate_remaining_cargo(
            baggage_weight_kg=baggage_upper,  # More baggage = less remaining cargo
            existing_cargo_weight_kg=existing_cargo_weight_kg,
            existing_cargo_volume_m3=existing_cargo_volume_m3,
            tail_number=tail_number
        )
        remaining_upper = self._calculate_remaining_cargo(
            baggage_weight_kg=baggage_lower,  # Less baggage = more remaining cargo
            existing_cargo_weight_kg=existing_cargo_weight_kg,
            existing_cargo_volume_m3=existing_cargo_volume_m3,
            tail_number=tail_number
        )
        
        # Calculate confidence score
        baggage_cv = baggage_std / (baggage_pred + 1e-6)
        remaining_cv = remaining_model_std / (remaining_model_pred + 1e-6)
        confidence = 1.0 / (1.0 + (baggage_cv + remaining_cv) / 2)
        confidence = np.clip(confidence, 0, 1)
        
        return {
            'predicted_baggage': float(baggage_pred),
            'predicted_baggage_lower': float(max(0, baggage_lower)),
            'predicted_baggage_upper': float(baggage_upper),
            'remaining_cargo': float(remaining_cargo),
            'remaining_cargo_lower': float(max(0, remaining_lower)),
            'remaining_cargo_upper': float(remaining_upper),
            'confidence': float(confidence),
            'confidence_std': {
                'baggage': float(baggage_std),
                'remaining': float(remaining_model_std)
            },
            'constraint_info': {
                'binding_constraint': 'weight' if remaining_cargo < remaining_model_pred * 1.1 else 'volume',
                'baggage_is_function_of_passenger_count': True
            }
        }
    
    def get_feature_importance(self, top_n: int = 5) -> list:
        """Get top N most important features from ensemble."""
        if self.use_ensemble:
            # Average feature importance across all models in ensemble
            all_importances = []
            weights = self.baggage_ensemble['weights']
            models = self.baggage_ensemble['models']
            
            for model_name, model in models.items():
                weight = weights[model_name]
                if hasattr(model, 'feature_importances_'):
                    # Weight the importance by model weight
                    weighted_imp = model.feature_importances_ * weight
                    all_importances.append(weighted_imp)
            
            if all_importances:
                importance_baggage = np.sum(all_importances, axis=0)
            else:
                importance_baggage = np.zeros(len(self.feature_cols))
            
            # Same for remaining cargo
            all_importances = []
            weights = self.remaining_ensemble['weights']
            models = self.remaining_ensemble['models']
            
            for model_name, model in models.items():
                weight = weights[model_name]
                if hasattr(model, 'feature_importances_'):
                    weighted_imp = model.feature_importances_ * weight
                    all_importances.append(weighted_imp)
            
            if all_importances:
                importance_remaining = np.sum(all_importances, axis=0)
            else:
                importance_remaining = np.zeros(len(self.feature_cols))
        else:
            # Legacy single model
            importance_baggage = self.baggage_model.feature_importances_
            importance_remaining = self.remaining_model.feature_importances_
        
        # Average importance
        avg_importance = (importance_baggage + importance_remaining) / 2
        
        # Get top features
        feature_importance = list(zip(self.feature_cols, avg_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [feat for feat, _ in feature_importance[:top_n]]

