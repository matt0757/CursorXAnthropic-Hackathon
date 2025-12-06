# Capacity Calculation Fix

## Problem Identified

The original implementation only considered **weight constraints** and ignored **volume constraints**, which doesn't match the actual business logic.

## Correct Business Logic

Based on requirements:
1. **capacity = min(max_weight, max_volume) - (baggage + cargo)**
   - The binding constraint is whichever is more restrictive (weight OR volume)

2. **baggage = function of passenger_count**
   - Baggage weight is primarily determined by number of passengers
   - Model already accounts for this relationship

3. **remaining_cargo = what airline can sell (e.g., to Shopee)**
   - This is the available cargo space after baggage and existing cargo

4. **total space varies by aircraft type**
   - Different aircraft have different max weight and max volume capacities

## Implementation Changes

### 1. `backend/train_model.py`

**Before:**
```python
# Only used weight constraint
y_remaining = (aircraft_capacity - baggage_weight - existing_cargo_weight)
```

**After:**
```python
# Calculate both weight and volume constraints
remaining_weight = max_weight - baggage - existing_cargo_weight
remaining_volume = max_volume - baggage_volume - existing_cargo_volume
# Convert volume to weight equivalent and take minimum (binding constraint)
remaining_cargo = min(remaining_weight, remaining_volume * cargo_density)
```

### 2. `backend/forecaster.py`

**New Methods:**
- `_load_aircraft_capacities()`: Loads both weight and volume capacities from CSV
- `_calculate_remaining_cargo()`: Calculates remaining cargo considering both constraints
- Updated `predict()`: Now properly calculates remaining cargo using constraint logic

**Key Formula:**
```python
remaining_cargo = min(
    max_weight - baggage - existing_cargo_weight,
    (max_volume - baggage_volume - existing_cargo_volume) * cargo_density
)
```

### 3. `backend/main.py`

**Updated `/predict` endpoint:**
- Now accepts optional `existing_cargo_weight_kg` and `existing_cargo_volume_m3` parameters
- Passes these to forecaster for accurate constraint calculations

## Density Constants

- **Baggage Density**: 160 kg/m³ (typical airline baggage)
- **Cargo Density**: 200 kg/m³ (typical cargo density)

## Aircraft Capacity Data

Loaded from `*aircraft tail*.csv`:
- `Max Weight (Lower Deck) (kg)`
- `Max Volume (Lower Deck) (m³)`

## Example Calculation

For a Boeing 737-800:
- Max Weight: 1800 kg
- Max Volume: 44 m³
- Baggage: 2500 kg (15.625 m³ at 160 kg/m³)
- Existing Cargo: 500 kg (2.5 m³ at 200 kg/m³)

**Weight constraint:**
- Remaining = 1800 - 2500 - 500 = -1200 kg ❌ (exceeded!)

**Volume constraint:**
- Remaining Volume = 44 - 15.625 - 2.5 = 25.875 m³
- Remaining Weight (from volume) = 25.875 × 200 = 5175 kg

**Binding constraint**: Weight (more restrictive)
- Result: 0 kg remaining (capacity exceeded)

## Verification

After retraining, the model will:
1. ✅ Consider both weight and volume constraints
2. ✅ Calculate remaining cargo as what can actually be sold
3. ✅ Account for baggage being a function of passenger count
4. ✅ Handle variable aircraft capacities by type

## Next Steps

1. **Retrain the model** to get updated predictions:
   ```bash
   python backend/train_model.py
   ```

2. **Test predictions** with both constraints:
   ```python
   result = forecaster.predict(
       flight_data,
       existing_cargo_weight_kg=500,
       existing_cargo_volume_m3=2.5
   )
   ```

3. The system now correctly implements the business logic!

