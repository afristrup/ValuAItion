# Feature Engineering Improvements

This document outlines the feature engineering improvements made to enhance model performance.

## Current Performance Metrics
- **R²**: 0.95048
- **RMSE**: 581,708.29
- **MAE**: 324,808.04
- **MAPE**: 9.64%
- **±20% Precision**: 91.04%
- **±10% Precision**: 63.14%
- **±5% Precision**: 34.97%

## New Features Added

### 1. Polynomial Features (`create_polynomial_features`)
Captures non-linear relationships in key variables:
- **Squared terms**: `AREA_RESIDENTIAL_SQUARED`, `FLOOR_SQUARED`, `PROPERTY_AGE_SQUARED`, `TRADE_YEAR_SQUARED`
- **Square root terms**: `AREA_RESIDENTIAL_SQRT` (useful for area features)

**Rationale**: Real estate prices often have non-linear relationships (e.g., larger properties may have diminishing returns per square meter).

### 2. Binning Features (`create_binning_features`)
Creates categorical versions of continuous variables:
- **Area bins**: `AREA_RESIDENTIAL_BINNED` (0-50, 50-75, 75-100, 100-125, 125-150, 150-200, 200+ sqm)
- **Floor bins**: `FLOOR_BINNED` (basement, 0-2, 2-5, 5-10, 10+ floors)
- **Year bins**: `TRADE_YEAR_BINNED` (captures market periods)

**Rationale**: Binning helps models capture threshold effects (e.g., properties above 100 sqm may have different pricing dynamics).

### 3. Location Features (`create_location_features`)
Geographic features from coordinates (if LAT/LNG available):
- **Distance from center**: `DIST_FROM_CENTER` (distance from Denmark center)
- **Copenhagen area indicator**: `IS_COPENHAGEN_AREA` (binary flag)

**Rationale**: Location is a key factor in real estate pricing. These features capture geographic patterns.

### 4. Aggregated Price Features (`create_aggregated_price_features`)
Location-based price aggregations (calculated from training data only):
- **ZIP level**: `ZIP_PRICE_MEAN`, `ZIP_PRICE_MEDIAN`, `ZIP_PRICE_STD`
- **Municipality level**: `MUNICIPALITY_PRICE_MEAN`, `MUNICIPALITY_PRICE_MEDIAN`, `MUNICIPALITY_PRICE_STD`

**Rationale**: These features provide context about local market conditions without data leakage (calculated only from training data).

### 5. Enhanced Interaction Features (`create_enhanced_interaction_features`)
Additional meaningful interactions:
- **Street price × Area**: `STREET_PRICE_X_AREA` (expected value interaction)
- **Age × Year**: `AGE_X_YEAR` (how age affects value over time)
- **Total area × Floor**: `TOTAL_AREA_X_FLOOR` (larger units on higher floors)
- **Elevator × Age**: `ELEVATOR_X_AGE` (elevators more valuable in older buildings)
- **Residential ratio × Floor**: `RESIDENTIAL_RATIO_X_FLOOR` (efficiency by floor)

**Rationale**: Interactions capture complex relationships between features that simple linear models might miss.

### 6. External Data Integration (`load_external_features`)
Optional integration of external data sources:
- **School distances**: `DIST_TO_CLOSEST_SCHOOL` (if school distance data is available)

**Rationale**: Proximity to amenities (schools, transport, etc.) significantly affects property values.

### 7. Enhanced Price Ratio Features (`create_price_ratio_features`)
- **Expected price**: `EXPECTED_PRICE_STREET` (street average × area)

**Rationale**: Provides a baseline expectation for property value based on local market conditions.

## Usage

The enhanced feature engineering is automatically applied when using `apply_feature_engineering()`. You can control external data inclusion:

```python
df_train, df_val = apply_feature_engineering(
    df_train,
    df_val,
    remove_low_variance=True,
    variance_threshold=0.01,
    include_external_data=True,  # Set to True to include school distances
    external_data_path="datasets/apartment_closest_school.csv"
)
```

## Expected Impact

These improvements should help:
1. **Reduce RMSE/MAE**: Better feature representation should improve prediction accuracy
2. **Improve precision metrics**: More informative features should increase the percentage of predictions within ±5%, ±10%, and ±20% thresholds
3. **Better generalization**: Aggregated features and polynomial terms should help the model capture market patterns

## Next Steps for Further Improvement

1. **Feature importance analysis**: After training, analyze which new features are most important
2. **Feature selection**: Use techniques like recursive feature elimination to remove redundant features
3. **Domain-specific features**: Consider adding features like:
   - Distance to public transport
   - Nearby amenities (parks, shopping centers)
   - Building quality indicators
   - Market trend indicators (price changes over time)
4. **Ensemble methods**: Combine predictions from models trained on different feature sets
5. **Hyperparameter tuning**: Adjust variance threshold and other parameters based on validation performance

## Notes

- All aggregated features are calculated from training data only to prevent data leakage
- Missing values in aggregated features are filled with global statistics
- External data features are optional and gracefully handle missing files
- Low-variance feature removal is still applied to filter out uninformative features
