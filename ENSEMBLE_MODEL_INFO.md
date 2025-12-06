# Ensemble Model Architecture

## Model Selection Rationale

Based on the dataset characteristics:
- **Regression problem**: Predicting baggage weight and remaining cargo capacity
- **Mixed data types**: Categorical (origin, destination, aircraft type) and numerical features
- **~3000+ samples**: Combined 2023 and 2024 data
- **Tabular data**: Structured flight records

## Selected Models

### 1. **LightGBM** (Primary - Typically Highest Weight)
- **Why**: Excellent for tabular data, handles categorical features natively
- **Strengths**: Fast training, good performance, handles missing values well
- **Hyperparameters**: 
  - `n_estimators=200`, `learning_rate=0.05`, `max_depth=7`
  - `num_leaves=31`, `min_child_samples=20`
  - Early stopping with 20 rounds patience
  - Trained with validation set monitoring

### 2. **Random Forest** (Robust Baseline)
- **Why**: Robust, handles mixed data types, provides feature importance
- **Strengths**: Less prone to overfitting, good variance reduction
- **Hyperparameters**:
  - `n_estimators=200`, `max_depth=15`
  - `min_samples_split=5`, `min_samples_leaf=2`
  - Bootstrap sampling enabled
  - Parallel training (`n_jobs=-1`)

### 3. **Gradient Boosting** (Traditional Boosting)
- **Why**: Complements tree-based models, captures different patterns
- **Strengths**: Sequential learning, gradient-based optimization
- **Hyperparameters**:
  - `n_estimators=200`, `learning_rate=0.05`, `max_depth=7`
  - `min_samples_split=5`

### 4. **XGBoost** (Optional - If Installed)
- **Why**: Often performs differently than LightGBM, adds diversity
- **Strengths**: Robust to outliers, regularization, sparsity-aware
- **Hyperparameters**:
  - `n_estimators=200`, `learning_rate=0.05`, `max_depth=7`
  - `min_child_weight=3`
  - Early stopping with 20 rounds
- **Note**: Gracefully skipped if not installed (not required)

### 5. **Ridge Regression** (Linear Baseline)
- **Why**: Provides linear baseline, handles multicollinearity
- **Strengths**: Fast, interpretable, L2 regularization
- **Hyperparameters**:
  - `alpha=1.0` (L2 penalty)
  - Provides sanity check against tree-based models

## Ensemble Strategy

### Weighted Average Ensemble
- **Method**: Inverse RMSE weighting
- **Formula**: `weight_i = (1 / RMSE_i) / sum(1 / RMSE_j)`
- **Rationale**: Better performing models get higher weights
- **Advantages**:
  - Automatically balances model contributions
  - Robust to individual model failures
  - Reduces variance compared to single models

### Ensemble Benefits
1. **Reduced Overfitting**: Different models make different errors
2. **Better Generalization**: Combines strengths of multiple algorithms
3. **Uncertainty Estimation**: Variance across models provides confidence intervals
4. **Robustness**: Less sensitive to hyperparameter choices

## Training Process

### 1. **Data Preparation**
- **Load Data**: Automatically finds all CSV files in `data/` folder (excluding aircraft tail reference)
- **Combine Datasets**: Concatenates 2023 and 2024 sample data (~3000+ records)
- **Temporal Features**: Extract year, month, day_of_week, day_of_month, is_weekend from flight_date
- **Derived Features**: 
  - `bags_per_passenger` = baggage_weight / (passenger_count + 1)
  - `cargo_per_passenger` = cargo_weight / (passenger_count + 1)
- **Mock Scenario Features** (for what-if analysis):
  - `group_travel_ratio` (Beta distribution)
  - `holiday_flag` (binary, 15% probability)
  - `delay_probability` (Beta distribution)
  - `weather_index` (Normal distribution, clipped 0-1)
- **Encoding**: LabelEncoder for categorical features (origin, destination, tail_number, aircraft_type)
- **Missing Values**: Median imputation for numeric, 'UNKNOWN' for categorical
- **Train/Test Split**: 80/20

### 2. **Target Calculation**
- **Baggage Weight**: Direct from dataset (function of passenger_count)
- **Cargo Demand**: Historical `gross_weight_cargo_kg`
- **Cargo Volume**: Historical `gross_volume_cargo_m3`
- **Remaining Capacity**: Calculated using dual constraints:
  ```python
  remaining_weight = max_weight - baggage - cargo
  remaining_volume_as_weight = (max_volume - baggage_vol - cargo_vol) * 200
  remaining = min(remaining_weight, remaining_volume_as_weight)
  ```
- Aircraft capacities loaded from `aircraft tail.csv` or estimated from 95th percentile

### 3. **Ensemble Training** (Repeated for Each Target)
For each target (baggage, cargo_demand, cargo_volume, remaining):
1. Train 5 base models independently
2. Make predictions on test set
3. Calculate RMSE for each model
4. Compute inverse RMSE weights: `weight = (1/RMSE) / sum(1/RMSE)`
5. Normalize weights to sum to 1.0
6. Store all models + weights

### 4. **Model Persistence**
```python
model_data = {
    'baggage_ensemble': {'models': {...}, 'weights': {...}},
    'cargo_demand_ensemble': {'models': {...}, 'weights': {...}},
    'cargo_volume_ensemble': {'models': {...}, 'weights': {...}},
    'remaining_ensemble': {'models': {...}, 'weights': {...}},
    'ensemble_type': 'weighted',
    'feature_cols': [...],
    'label_encoders': {...}
}
```
Saved to `models/forecaster.pkl` using pickle

### 5. **Inference (Prediction)**
- Load ensemble from pickle file
- Prepare features (encode categoricals, fill missing)
- For each target:
  - Get prediction from each base model
  - Compute weighted average: `pred = sum(weight_i * pred_i)`
  - Calculate ensemble std dev for confidence intervals
- Calculate constraint-based remaining capacity
- Return predictions + 95% CI + confidence score

## Expected Improvements

- **Accuracy**: 5-15% improvement in RMSE over single best model
- **Robustness**: More stable predictions across different scenarios
- **Confidence Intervals**: Better uncertainty quantification

## Usage

The ensemble is automatically used when loading the trained model. The forecaster class handles:
- Loading ensemble models
- Making weighted predictions
- Calculating confidence intervals
- Feature importance (averaged across models)

## Model Files

- Saved to: `models/forecaster.pkl`
- Contains: All base models + weights + metadata
- Backward compatible: Falls back to single model format if needed

