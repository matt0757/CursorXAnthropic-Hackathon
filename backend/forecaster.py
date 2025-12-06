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
    
    def predict(self, flight_data: Dict) -> Dict:
        """
        Predict baggage weight and remaining cargo capacity using ensemble.
        
        Args:
            flight_data: Dictionary with flight features
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        X = self._prepare_features(flight_data)
        
        # Make predictions using ensemble or single model
        if self.use_ensemble:
            # Ensemble prediction
            baggage_pred = self._predict_ensemble(X, self.baggage_ensemble)
            remaining_pred = self._predict_ensemble(X, self.remaining_ensemble)
            
            # Estimate uncertainty from ensemble variance
            baggage_std = self._get_ensemble_std(X, self.baggage_ensemble)
            remaining_std = self._get_ensemble_std(X, self.remaining_ensemble)
        else:
            # Legacy single model prediction
            baggage_pred = self.baggage_model.predict(X)[0]
            remaining_pred = self.remaining_model.predict(X)[0]
            
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
                remaining_samples = [remaining_pred]
            
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

