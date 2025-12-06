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
import lightgbm as lgb

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
    
    # Target: predict baggage_weight_kg and compute remaining cargo
    y_baggage = df['baggage_weight_kg'].values
    
    # Calculate remaining cargo capacity
    # Load aircraft capacities
    aircraft_capacity_file = Path(__file__).parent.parent / "data" / "*aircraft tail*.csv"
    capacity_files = glob.glob(str(aircraft_capacity_file))
    
    if capacity_files:
        capacity_df = pd.read_csv(capacity_files[0])
        capacity_df.columns = capacity_df.columns.str.strip()
        
        # Create mapping from tail_number to max weight
        tail_to_capacity = dict(zip(
            capacity_df['Tail Number'].str.strip(),
            capacity_df['Max Weight (Lower Deck) (kg)']
        ))
        
        # Map capacities
        df['aircraft_capacity'] = df['tail_number'].map(tail_to_capacity).fillna(
            df['gross_weight_cargo_kg'].quantile(0.95)  # Fallback to high percentile
        )
    else:
        # Fallback: estimate capacity from data
        df['aircraft_capacity'] = df.groupby('aircraft_type')['gross_weight_cargo_kg'].transform(
            lambda x: x.quantile(0.95)
        )
    
    # Remaining cargo = capacity - baggage - existing cargo
    y_remaining = (df['aircraft_capacity'] - df['baggage_weight_kg'] - df['gross_weight_cargo_kg']).values
    y_remaining = np.clip(y_remaining, 0, None)  # Ensure non-negative
    
    return X, y_baggage, y_remaining, feature_cols

def train_model(X, y_baggage, y_remaining, feature_cols):
    """Train LightGBM models for baggage and remaining cargo prediction."""
    print("\n=== Model Training ===")
    
    # Split data
    X_train, X_test, y_baggage_train, y_baggage_test = train_test_split(
        X, y_baggage, test_size=0.2, random_state=42
    )
    _, _, y_remaining_train, y_remaining_test = train_test_split(
        X, y_remaining, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train baggage weight model
    print("\nTraining baggage weight model...")
    model_baggage = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbose=-1
    )
    model_baggage.fit(
        X_train, y_baggage_train,
        eval_set=[(X_test, y_baggage_test)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    # Train remaining cargo model
    print("Training remaining cargo model...")
    model_remaining = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbose=-1
    )
    model_remaining.fit(
        X_train, y_remaining_train,
        eval_set=[(X_test, y_remaining_test)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    # Evaluate models
    print("\n=== Model Evaluation ===")
    
    y_baggage_pred = model_baggage.predict(X_test)
    print(f"\nBaggage Weight Model:")
    print(f"  MAE: {mean_absolute_error(y_baggage_test, y_baggage_pred):.2f} kg")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_baggage_test, y_baggage_pred)):.2f} kg")
    print(f"  R²: {r2_score(y_baggage_test, y_baggage_pred):.4f}")
    
    y_remaining_pred = model_remaining.predict(X_test)
    print(f"\nRemaining Cargo Model:")
    print(f"  MAE: {mean_absolute_error(y_remaining_test, y_remaining_pred):.2f} kg")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_remaining_test, y_remaining_pred)):.2f} kg")
    print(f"  R²: {r2_score(y_remaining_test, y_remaining_pred):.4f}")
    
    return model_baggage, model_remaining, feature_cols

def save_model(model_baggage, model_remaining, feature_cols, label_encoders):
    """Save trained models and metadata."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_data = {
        'baggage_model': model_baggage,
        'remaining_model': model_remaining,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders
    }
    
    model_path = models_dir / "forecaster.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Models saved to {model_path}")

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
    X, y_baggage, y_remaining, feature_cols = prepare_features(df)
    
    # Train models
    model_baggage, model_remaining, feature_cols = train_model(X, y_baggage, y_remaining, feature_cols)
    
    # Save models
    save_model(model_baggage, model_remaining, feature_cols, label_encoders)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

