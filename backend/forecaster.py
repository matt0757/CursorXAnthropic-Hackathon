"""
Forecaster module - Loads model and makes predictions with confidence intervals.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

class CargoForecaster:
    """Forecaster for cargo capacity predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize forecaster by loading trained model."""
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "forecaster.pkl"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.baggage_model = model_data['baggage_model']
        self.remaining_model = model_data['remaining_model']
        self.feature_cols = model_data['feature_cols']
        self.label_encoders = model_data.get('label_encoders', {})
    
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
    
    def predict(self, flight_data: Dict) -> Dict:
        """
        Predict baggage weight and remaining cargo capacity.
        
        Args:
            flight_data: Dictionary with flight features
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        X = self._prepare_features(flight_data)
        
        # Make predictions
        baggage_pred = self.baggage_model.predict(X)[0]
        remaining_pred = self.remaining_model.predict(X)[0]
        
        # Bootstrap for confidence intervals (quick approximation)
        # Use prediction intervals from LightGBM
        # Get standard deviation from individual tree predictions
        baggage_samples = []
        remaining_samples = []
        
        # Get individual tree predictions (if available)
        try:
            n_trees = self.baggage_model.n_estimators_
            # Sample from different tree subsets
            step = max(1, n_trees // 50)  # Sample ~50 trees
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
            remaining_samples = [remaining_pred]
        
        # Calculate confidence intervals
        baggage_std = np.std(baggage_samples) if baggage_samples else baggage_pred * 0.1
        remaining_std = np.std(remaining_samples) if remaining_samples else remaining_pred * 0.1
        
        # 95% confidence interval (approx)
        baggage_lower = baggage_pred - 1.96 * baggage_std
        baggage_upper = baggage_pred + 1.96 * baggage_std
        remaining_lower = remaining_pred - 1.96 * remaining_std
        remaining_upper = remaining_pred + 1.96 * remaining_std
        
        # Calculate confidence score (inverse of coefficient of variation)
        baggage_cv = baggage_std / (baggage_pred + 1e-6)
        remaining_cv = remaining_std / (remaining_pred + 1e-6)
        confidence = 1.0 / (1.0 + (baggage_cv + remaining_cv) / 2)
        confidence = np.clip(confidence, 0, 1)
        
        return {
            'predicted_baggage': float(baggage_pred),
            'predicted_baggage_lower': float(baggage_lower),
            'predicted_baggage_upper': float(baggage_upper),
            'remaining_cargo': float(max(0, remaining_pred)),
            'remaining_cargo_lower': float(max(0, remaining_lower)),
            'remaining_cargo_upper': float(max(0, remaining_upper)),
            'confidence': float(confidence),
            'confidence_std': {
                'baggage': float(baggage_std),
                'remaining': float(remaining_std)
            }
        }
    
    def get_feature_importance(self, top_n: int = 5) -> list:
        """Get top N most important features."""
        importance_baggage = self.baggage_model.feature_importances_
        importance_remaining = self.remaining_model.feature_importances_
        
        # Average importance
        avg_importance = (importance_baggage + importance_remaining) / 2
        
        # Get top features
        feature_importance = list(zip(self.feature_cols, avg_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [feat for feat, _ in feature_importance[:top_n]]

