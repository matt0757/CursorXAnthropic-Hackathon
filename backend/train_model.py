"""
ML Training Script for Cargo Capacity Forecaster
Automatically detects CSV files and trains a regression model.
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

def find_csv_files():
    """Find all CSV files in the data directory."""
    data_dir = Path(__file__).parent.parent / "data"
    csv_files = list(data_dir.glob("*.csv"))
    # Filter out aircraft tail file (it's metadata)
    csv_files = [f for f in csv_files if "aircraft tail" not in f.name.lower()]
    return csv_files

def load_and_combine_data():
    """Load and combine all flight data CSV files."""
    csv_files = find_csv_files()
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory!")
    
    print(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")
    return df

def clean_and_prepare_data(df):
    """Clean data and prepare features."""
    print("\n=== Data Cleaning ===")
    print(f"Initial shape: {df.shape}")
    
    # Convert flight_date to datetime
    df['flight_date'] = pd.to_datetime(df['flight_date'], errors='coerce')
    
    # Extract temporal features
    df['year'] = df['flight_date'].dt.year
    df['month'] = df['flight_date'].dt.month
    df['day_of_week'] = df['flight_date'].dt.dayofweek
    df['day_of_month'] = df['flight_date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Create derived features
    df['bags_per_passenger'] = df['baggage_weight_kg'] / (df['passenger_count'] + 1)  # +1 to avoid division by zero
    df['cargo_per_passenger'] = df['gross_weight_cargo_kg'] / (df['passenger_count'] + 1)
    
    # Generate mock features for what-if simulator
    # These would ideally come from external data sources
    np.random.seed(42)
    df['group_travel_ratio'] = np.random.beta(2, 5, size=len(df))  # Beta distribution
    df['holiday_flag'] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    df['delay_probability'] = np.random.beta(2, 8, size=len(df))
    df['weather_index'] = np.random.normal(0.5, 0.15, size=len(df)).clip(0, 1)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = ['origin', 'destination', 'tail_number', 'aircraft_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
    
    # Encode categorical variables
    label_encoders = {}
    for col in ['origin', 'destination', 'tail_number', 'aircraft_type']:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    print(f"After cleaning: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df, label_encoders

def prepare_features(df):
    """Prepare feature matrix and targets."""
    # Features for prediction
    feature_cols = [
        'passenger_count',
        'year', 'month', 'day_of_week', 'day_of_month', 'is_weekend',
        'group_travel_ratio', 'holiday_flag', 'delay_probability', 'weather_index',
        'origin_encoded', 'destination_encoded', 'tail_number_encoded', 'aircraft_type_encoded',
        'fuel_weight_kg', 'fuel_price_per_kg', 'cargo_price_per_kg'
    ]
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    
    # Target 1: predict baggage_weight_kg (which is a function of passenger_count)
    # Baggage is primarily determined by passenger count
    y_baggage = df['baggage_weight_kg'].values
    
    # Target 2: predict cargo demand (actual cargo weight)
    y_cargo_demand = df['gross_weight_cargo_kg'].values
    
    # Target 3: predict cargo volume demand
    y_cargo_volume = df['gross_volume_cargo_m3'].values
    
    # Calculate remaining cargo capacity
    # Formula: capacity = min(max_weight, max_volume) - (baggage + cargo)
    # The binding constraint is whichever is more restrictive (weight or volume)
    
    # Load aircraft capacities
    aircraft_capacity_file = Path(__file__).parent.parent / "data" / "*aircraft tail*.csv"
    capacity_files = glob.glob(str(aircraft_capacity_file))
    
    # Typical density for baggage: ~160 kg/m³ (average for airline baggage)
    BAGGAGE_DENSITY = 160  # kg/m³
    # Typical density for cargo: ~200 kg/m³ (can vary, but use average)
    CARGO_DENSITY = 200  # kg/m³
    
    if capacity_files:
        capacity_df = pd.read_csv(capacity_files[0])
        capacity_df.columns = capacity_df.columns.str.strip()
        
        # Create mappings from tail_number to max capacities
        tail_to_max_weight = dict(zip(
            capacity_df['Tail Number'].str.strip(),
            capacity_df['Max Weight (Lower Deck) (kg)']
        ))
        tail_to_max_volume = dict(zip(
            capacity_df['Tail Number'].str.strip(),
            capacity_df['Max Volume (Lower Deck) (m³)']
        ))
        
        # Map capacities to dataframe
        df['aircraft_max_weight_kg'] = df['tail_number'].map(tail_to_max_weight).fillna(
            df.groupby('aircraft_type')['gross_weight_cargo_kg'].transform(lambda x: x.quantile(0.95))
        )
        df['aircraft_max_volume_m3'] = df['tail_number'].map(tail_to_max_volume).fillna(
            df.groupby('aircraft_type')['gross_volume_cargo_m3'].transform(lambda x: x.quantile(0.95))
        )
    else:
        # Fallback: estimate capacity from data
        df['aircraft_max_weight_kg'] = df.groupby('aircraft_type')['gross_weight_cargo_kg'].transform(
            lambda x: x.quantile(0.95)
        )
        df['aircraft_max_volume_m3'] = df.groupby('aircraft_type')['gross_volume_cargo_m3'].transform(
            lambda x: x.quantile(0.95)
        )
    
    # Estimate baggage volume from weight (baggage weight / baggage density)
    df['baggage_volume_m3'] = df['baggage_weight_kg'] / BAGGAGE_DENSITY
    
    # Calculate remaining capacity in both weight and volume
    # Remaining weight capacity
    remaining_weight_capacity = df['aircraft_max_weight_kg'] - df['baggage_weight_kg'] - df['gross_weight_cargo_kg']
    
    # Remaining volume capacity  
    remaining_volume_capacity = df['aircraft_max_volume_m3'] - df['baggage_volume_m3'] - df['gross_volume_cargo_m3']
    
    # Convert remaining volume to equivalent weight using cargo density
    # This gives us the weight of cargo we could fit in the remaining volume
    remaining_volume_as_weight = remaining_volume_capacity * CARGO_DENSITY
    
    # The binding constraint is the minimum of weight capacity and volume capacity (in weight terms)
    # This represents the maximum cargo weight we can sell (Shopee cargo)
    y_remaining = np.minimum(remaining_weight_capacity, remaining_volume_as_weight).values
    y_remaining = np.clip(y_remaining, 0, None)  # Ensure non-negative
    
    return X, y_baggage, y_cargo_demand, y_cargo_volume, y_remaining, feature_cols

def train_base_models(X_train, y_train, X_test, y_test, model_name="baggage"):
    """Train individual base models and return their predictions."""
    models = {}
    predictions = {}
    scores = {}
    
    print(f"\nTraining base models for {model_name}...")
    
    # 1. LightGBM - Excellent for tabular data, handles categoricals
    print("  - LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    models['lightgbm'] = lgb_model
    pred = lgb_model.predict(X_test)
    predictions['lightgbm'] = pred
    scores['lightgbm'] = {
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'r2': r2_score(y_test, pred)
    }
    
    # 2. Random Forest - Good baseline, robust
    print("  - Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    pred = rf_model.predict(X_test)
    predictions['random_forest'] = pred
    scores['random_forest'] = {
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'r2': r2_score(y_test, pred)
    }
    
    # 3. Gradient Boosting - Traditional boosting method
    print("  - Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models['gradient_boosting'] = gb_model
    pred = gb_model.predict(X_test)
    predictions['gradient_boosting'] = pred
    scores['gradient_boosting'] = {
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'r2': r2_score(y_test, pred)
    }
    
    # 4. XGBoost - If available
    if XGBOOST_AVAILABLE:
        print("  - XGBoost...")
        # XGBoost 3.x uses callbacks for early stopping
        try:
            from xgboost import callback
            early_stop = callback.EarlyStopping(rounds=20, save_best=True)
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=3,
                random_state=42,
                eval_metric='rmse'
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[early_stop],
                verbose=False
            )
        except (ImportError, AttributeError, TypeError):
            # Fallback: train without early stopping or use older API
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=7,
                    min_child_weight=3,
                    random_state=42,
                    eval_metric='rmse'
                )
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=20,
                    verbose=False
                )
            except TypeError:
                # Simplest fallback: no early stopping
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=7,
                    min_child_weight=3,
                    random_state=42,
                    eval_metric='rmse'
                )
                xgb_model.fit(X_train, y_train, verbose=False)
        models['xgboost'] = xgb_model
        pred = xgb_model.predict(X_test)
        predictions['xgboost'] = pred
        scores['xgboost'] = {
            'mae': mean_absolute_error(y_test, pred),
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'r2': r2_score(y_test, pred)
        }
    
    # 5. Ridge Regression - Linear baseline with regularization
    print("  - Ridge Regression...")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    models['ridge'] = ridge_model
    pred = ridge_model.predict(X_test)
    predictions['ridge'] = pred
    scores['ridge'] = {
        'mae': mean_absolute_error(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'r2': r2_score(y_test, pred)
    }
    
    # Print individual model scores
    print(f"\n  Individual Model Performance ({model_name}):")
    for model_name_key, score in scores.items():
        print(f"    {model_name_key:20s} - MAE: {score['mae']:8.2f}, RMSE: {score['rmse']:8.2f}, R²: {score['r2']:.4f}")
    
    return models, predictions, scores

def create_weighted_ensemble(models, predictions, scores):
    """Create weighted ensemble based on model performance."""
    # Calculate weights based on inverse RMSE (lower RMSE = higher weight)
    weights = {}
    total_inv_rmse = 0
    
    for model_name, score in scores.items():
        inv_rmse = 1.0 / (score['rmse'] + 1e-6)  # Add small epsilon to avoid division by zero
        weights[model_name] = inv_rmse
        total_inv_rmse += inv_rmse
    
    # Normalize weights
    for model_name in weights:
        weights[model_name] /= total_inv_rmse
    
    print(f"\n  Ensemble weights:")
    for model_name, weight in weights.items():
        print(f"    {model_name:20s}: {weight:.4f}")
    
    return weights

def train_model(X, y_baggage, y_cargo_demand, y_cargo_volume, y_remaining, feature_cols):
    """Train ensemble models for baggage, cargo demand, and remaining cargo prediction."""
    print("\n=== Model Training (Ensemble) ===")
    
    # Split data
    X_train, X_test, y_baggage_train, y_baggage_test = train_test_split(
        X, y_baggage, test_size=0.2, random_state=42
    )
    _, _, y_cargo_demand_train, y_cargo_demand_test = train_test_split(
        X, y_cargo_demand, test_size=0.2, random_state=42
    )
    _, _, y_cargo_volume_train, y_cargo_volume_test = train_test_split(
        X, y_cargo_volume, test_size=0.2, random_state=42
    )
    _, _, y_remaining_train, y_remaining_test = train_test_split(
        X, y_remaining, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train base models for baggage
    models_baggage, preds_baggage, scores_baggage = train_base_models(
        X_train, y_baggage_train, X_test, y_baggage_test, "baggage"
    )
    
    # Train base models for cargo demand
    models_cargo_demand, preds_cargo_demand, scores_cargo_demand = train_base_models(
        X_train, y_cargo_demand_train, X_test, y_cargo_demand_test, "cargo_demand"
    )
    
    # Train base models for cargo volume
    models_cargo_volume, preds_cargo_volume, scores_cargo_volume = train_base_models(
        X_train, y_cargo_volume_train, X_test, y_cargo_volume_test, "cargo_volume"
    )
    
    # Train base models for remaining cargo
    models_remaining, preds_remaining, scores_remaining = train_base_models(
        X_train, y_remaining_train, X_test, y_remaining_test, "remaining_cargo"
    )
    
    # Create weighted ensembles
    weights_baggage = create_weighted_ensemble(models_baggage, preds_baggage, scores_baggage)
    weights_cargo_demand = create_weighted_ensemble(models_cargo_demand, preds_cargo_demand, scores_cargo_demand)
    weights_cargo_volume = create_weighted_ensemble(models_cargo_volume, preds_cargo_volume, scores_cargo_volume)
    weights_remaining = create_weighted_ensemble(models_remaining, preds_remaining, scores_remaining)
    
    # Evaluate ensemble predictions
    print("\n=== Ensemble Evaluation ===")
    
    # Baggage ensemble prediction
    ensemble_pred_baggage = np.zeros(len(y_baggage_test))
    for model_name, pred in preds_baggage.items():
        ensemble_pred_baggage += weights_baggage[model_name] * pred
    
    print(f"\nBaggage Weight Ensemble:")
    print(f"  MAE: {mean_absolute_error(y_baggage_test, ensemble_pred_baggage):.2f} kg")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_baggage_test, ensemble_pred_baggage)):.2f} kg")
    print(f"  R²: {r2_score(y_baggage_test, ensemble_pred_baggage):.4f}")
    
    # Cargo demand ensemble prediction
    ensemble_pred_cargo_demand = np.zeros(len(y_cargo_demand_test))
    for model_name, pred in preds_cargo_demand.items():
        ensemble_pred_cargo_demand += weights_cargo_demand[model_name] * pred
    
    print(f"\nCargo Demand Ensemble:")
    print(f"  MAE: {mean_absolute_error(y_cargo_demand_test, ensemble_pred_cargo_demand):.2f} kg")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_cargo_demand_test, ensemble_pred_cargo_demand)):.2f} kg")
    print(f"  R²: {r2_score(y_cargo_demand_test, ensemble_pred_cargo_demand):.4f}")
    
    # Cargo volume ensemble prediction
    ensemble_pred_cargo_volume = np.zeros(len(y_cargo_volume_test))
    for model_name, pred in preds_cargo_volume.items():
        ensemble_pred_cargo_volume += weights_cargo_volume[model_name] * pred
    
    print(f"\nCargo Volume Ensemble:")
    print(f"  MAE: {mean_absolute_error(y_cargo_volume_test, ensemble_pred_cargo_volume):.2f} m³")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_cargo_volume_test, ensemble_pred_cargo_volume)):.2f} m³")
    print(f"  R²: {r2_score(y_cargo_volume_test, ensemble_pred_cargo_volume):.4f}")
    
    # Remaining cargo ensemble prediction
    ensemble_pred_remaining = np.zeros(len(y_remaining_test))
    for model_name, pred in preds_remaining.items():
        ensemble_pred_remaining += weights_remaining[model_name] * pred
    
    print(f"\nRemaining Cargo Ensemble:")
    print(f"  MAE: {mean_absolute_error(y_remaining_test, ensemble_pred_remaining):.2f} kg")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_remaining_test, ensemble_pred_remaining)):.2f} kg")
    print(f"  R²: {r2_score(y_remaining_test, ensemble_pred_remaining):.4f}")
    
    # Package ensemble models
    ensemble_baggage = {
        'models': models_baggage,
        'weights': weights_baggage
    }
    ensemble_cargo_demand = {
        'models': models_cargo_demand,
        'weights': weights_cargo_demand
    }
    ensemble_cargo_volume = {
        'models': models_cargo_volume,
        'weights': weights_cargo_volume
    }
    ensemble_remaining = {
        'models': models_remaining,
        'weights': weights_remaining
    }
    
    return ensemble_baggage, ensemble_cargo_demand, ensemble_cargo_volume, ensemble_remaining, feature_cols

def save_model(ensemble_baggage, ensemble_cargo_demand, ensemble_cargo_volume, ensemble_remaining, feature_cols, label_encoders):
    """Save trained ensemble models and metadata."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_data = {
        'baggage_ensemble': ensemble_baggage,
        'cargo_demand_ensemble': ensemble_cargo_demand,
        'cargo_volume_ensemble': ensemble_cargo_volume,
        'remaining_ensemble': ensemble_remaining,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'ensemble_type': 'weighted'
    }
    
    model_path = models_dir / "forecaster.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Ensemble models saved to {model_path}")

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Cargo Capacity Forecaster - Model Training")
    print("=" * 60)
    
    # Load data
    df = load_and_combine_data()
    
    # Clean and prepare
    df, label_encoders = clean_and_prepare_data(df)
    
    # Prepare features
    X, y_baggage, y_cargo_demand, y_cargo_volume, y_remaining, feature_cols = prepare_features(df)
    
    # Train ensemble models
    ensemble_baggage, ensemble_cargo_demand, ensemble_cargo_volume, ensemble_remaining, feature_cols = train_model(
        X, y_baggage, y_cargo_demand, y_cargo_volume, y_remaining, feature_cols
    )
    
    # Save ensemble models
    save_model(ensemble_baggage, ensemble_cargo_demand, ensemble_cargo_volume, ensemble_remaining, feature_cols, label_encoders)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

