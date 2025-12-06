# Ensemble Model Architecture

## Model Selection Rationale

Based on the dataset characteristics:
- **Regression problem**: Predicting baggage weight and remaining cargo capacity
- **Mixed data types**: Categorical (origin, destination, aircraft type) and numerical features
- **~3000+ samples**: Combined 2023 and 2024 data
- **Tabular data**: Structured flight records

## Selected Models

### 1. **LightGBM** (Primary)
- **Why**: Excellent for tabular data, handles categorical features natively
- **Strengths**: Fast training, good performance, handles missing values well
- **Hyperparameters**: 
  - `n_estimators=200`, `learning_rate=0.05`, `max_depth=7`
  - Early stopping with validation set

### 2. **Random Forest** (Robust Baseline)
- **Why**: Robust, handles mixed data types, provides feature importance
- **Strengths**: Less prone to overfitting, good for smaller datasets
- **Hyperparameters**:
  - `n_estimators=200`, `max_depth=15`
  - Bootstrap sampling for variance reduction

### 3. **Gradient Boosting** (Traditional Boosting)
- **Why**: Complements tree-based models, different error patterns
- **Strengths**: Sequential learning, captures complex patterns
- **Hyperparameters**:
  - `n_estimators=200`, `learning_rate=0.05`, `max_depth=7`

### 4. **XGBoost** (If Available)
- **Why**: Often performs slightly differently than LightGBM, good complement
- **Strengths**: Robust to outliers, handles regularization well
- **Hyperparameters**:
  - `n_estimators=200`, `learning_rate=0.05`, `max_depth=7`
  - Early stopping enabled

### 5. **Ridge Regression** (Linear Baseline)
- **Why**: Provides linear baseline, handles multicollinearity
- **Strengths**: Fast, interpretable, regularization prevents overfitting
- **Hyperparameters**:
  - `alpha=1.0` (L2 regularization)

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

1. **Data Preparation**: 
   - Load and combine datasets
   - Feature engineering (temporal, encoded categoricals)
   - Train/test split (80/20)

2. **Base Model Training**:
   - Train each model independently
   - Evaluate on validation set
   - Record individual performance metrics

3. **Weight Calculation**:
   - Compute RMSE for each model
   - Calculate inverse RMSE weights
   - Normalize weights

4. **Ensemble Prediction**:
   - Weighted average of all model predictions
   - Estimate uncertainty from ensemble variance

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

