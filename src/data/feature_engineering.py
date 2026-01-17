"""
Optimized feature engineering functions for TabPFN model training.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple

from .constants import AREA_COLUMNS


def create_advanced_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced date-based features from TRADE_DATE.

    Args:
        df: DataFrame with TRADE_DATE column

    Returns:
        DataFrame with additional date features
    """
    df = df.copy()
    if "TRADE_DATE" not in df.columns:
        logging.warning(
            "TRADE_DATE column not found, skipping date feature engineering"
        )
        return df

    trade_dates = pd.to_datetime(df["TRADE_DATE"])

    # Basic date features
    df["TRADE_YEAR"] = trade_dates.dt.year.values
    df["TRADE_MONTH"] = trade_dates.dt.month.values
    df["TRADE_DOW"] = trade_dates.dt.dayofweek.values
    df["TRADE_DAY"] = trade_dates.dt.day.values

    # Advanced date features
    df["TRADE_QUARTER"] = trade_dates.dt.quarter.values
    df["TRADE_WEEK"] = trade_dates.dt.isocalendar().week.values
    df["TRADE_IS_WEEKEND"] = (trade_dates.dt.dayofweek >= 5).astype(int).values

    # Cyclical encoding for month and day of week (captures cyclical patterns)
    df["TRADE_MONTH_SIN"] = np.sin(2 * np.pi * df["TRADE_MONTH"] / 12)
    df["TRADE_MONTH_COS"] = np.cos(2 * np.pi * df["TRADE_MONTH"] / 12)
    df["TRADE_DOW_SIN"] = np.sin(2 * np.pi * df["TRADE_DOW"] / 7)
    df["TRADE_DOW_COS"] = np.cos(2 * np.pi * df["TRADE_DOW"] / 7)

    # Season encoding (Northern Hemisphere)
    df["TRADE_SEASON"] = trade_dates.dt.month % 12 // 3
    df["TRADE_IS_SPRING"] = (df["TRADE_SEASON"] == 0).astype(int)
    df["TRADE_IS_SUMMER"] = (df["TRADE_SEASON"] == 1).astype(int)
    df["TRADE_IS_FALL"] = (df["TRADE_SEASON"] == 2).astype(int)
    df["TRADE_IS_WINTER"] = (df["TRADE_SEASON"] == 3).astype(int)

    df.drop(["TRADE_DATE"], axis=1, inplace=True)
    return df


def create_age_features(
    df: pd.DataFrame, trade_year_col: str = "TRADE_YEAR"
) -> pd.DataFrame:
    """
    Create age-based features from construction and rebuilding years.

    Args:
        df: DataFrame with CONSTRUCTION_YEAR and REBUILDING_YEAR columns
        trade_year_col: Name of the trade year column

    Returns:
        DataFrame with additional age features
    """
    df = df.copy()

    # Property age at time of trade
    if trade_year_col in df.columns and "CONSTRUCTION_YEAR" in df.columns:
        df["PROPERTY_AGE"] = df[trade_year_col] - df["CONSTRUCTION_YEAR"]
        df["PROPERTY_AGE"] = df["PROPERTY_AGE"].clip(lower=0)  # Handle negative ages

    # Time since last rebuild
    if trade_year_col in df.columns and "REBUILDING_YEAR" in df.columns:
        df["YEARS_SINCE_REBUILD"] = df[trade_year_col] - df["REBUILDING_YEAR"]
        df["YEARS_SINCE_REBUILD"] = df["YEARS_SINCE_REBUILD"].clip(lower=0)

    # Has been rebuilt
    if "REBUILDING_YEAR" in df.columns and "CONSTRUCTION_YEAR" in df.columns:
        df["HAS_BEEN_REBUILT"] = (
            (df["REBUILDING_YEAR"] > df["CONSTRUCTION_YEAR"])
            & (df["REBUILDING_YEAR"].notna())
        ).astype(int)

    # Age categories (binned)
    if "PROPERTY_AGE" in df.columns:
        df["PROPERTY_AGE_CAT"] = pd.cut(
            df["PROPERTY_AGE"],
            bins=[0, 10, 20, 30, 50, 100, np.inf],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype(float)

    return df


def create_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated and ratio features from area columns.

    Args:
        df: DataFrame with area columns

    Returns:
        DataFrame with additional area features
    """
    df = df.copy()

    # Total area (sum of all areas)
    available_area_cols = [col for col in AREA_COLUMNS if col in df.columns]
    if available_area_cols:
        df["TOTAL_AREA"] = df[available_area_cols].sum(axis=1)

        # Individual area ratios
        for col in available_area_cols:
            if "AREA_RESIDENTIAL" in df.columns and col != "AREA_RESIDENTIAL":
                ratio_col = f"{col}_RATIO"
                df[ratio_col] = np.where(
                    df["AREA_RESIDENTIAL"] > 0, df[col] / df["AREA_RESIDENTIAL"], 0
                )

        # Residential area ratio (most important)
        if "AREA_RESIDENTIAL" in df.columns and "TOTAL_AREA" in df.columns:
            df["RESIDENTIAL_AREA_RATIO"] = np.where(
                df["TOTAL_AREA"] > 0, df["AREA_RESIDENTIAL"] / df["TOTAL_AREA"], 0
            )

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful interaction features.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with interaction features
    """
    df = df.copy()

    # Area * Floor interactions
    if "AREA_RESIDENTIAL" in df.columns and "FLOOR" in df.columns:
        df["AREA_RESIDENTIAL_X_FLOOR"] = df["AREA_RESIDENTIAL"] * df["FLOOR"]

    # Age * Area interactions (older properties with more area might be different)
    if "PROPERTY_AGE" in df.columns and "AREA_RESIDENTIAL" in df.columns:
        df["AGE_X_AREA"] = df["PROPERTY_AGE"] * df["AREA_RESIDENTIAL"]

    # Elevator * Floor (higher floors with elevators are more valuable)
    if "HAS_ELEVATOR" in df.columns and "FLOOR" in df.columns:
        df["ELEVATOR_X_FLOOR"] = df["HAS_ELEVATOR"] * df["FLOOR"]

    # Year * Area (market trends over time)
    if "TRADE_YEAR" in df.columns and "AREA_RESIDENTIAL" in df.columns:
        df["YEAR_X_AREA"] = df["TRADE_YEAR"] * df["AREA_RESIDENTIAL"]

    return df


def create_price_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-related ratio features if street price features exist.

    Args:
        df: DataFrame with price-related columns

    Returns:
        DataFrame with price ratio features
    """
    df = df.copy()

    # Price per sqm ratio features
    if "STREET_CODE_MEAN_SQM_PRICE" in df.columns and "AREA_RESIDENTIAL" in df.columns:
        # This would require PRICE, but we can't use target in features
        # Instead, create features that might correlate with price
        pass

    return df


def remove_low_variance_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    threshold: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove features with low variance (likely uninformative).
    Only applies to numeric columns.

    Args:
        X_train: Training features
        X_val: Validation features
        threshold: Variance threshold (default: 0.01)

    Returns:
        Tuple of (X_train, X_val) with low variance features removed
    """
    # Separate numeric and non-numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Only apply variance threshold to numeric columns
    if not numeric_cols:
        logging.warning("No numeric columns found for variance threshold filtering")
        return X_train, X_val
    
    X_train_numeric = X_train[numeric_cols]
    X_val_numeric = X_val[numeric_cols]
    
    selector = VarianceThreshold(threshold=threshold)
    X_train_selected = selector.fit_transform(X_train_numeric)
    X_val_selected = selector.transform(X_val_numeric)

    # Get selected feature names
    selected_numeric_features = X_train_numeric.columns[selector.get_support()].tolist()
    removed_features = set(numeric_cols) - set(selected_numeric_features)

    if removed_features:
        logging.info(
            f"Removed {len(removed_features)} low-variance numeric features: {list(removed_features)[:10]}..."
        )

    # Convert numeric features back to DataFrame
    X_train_numeric_clean = pd.DataFrame(
        X_train_selected, columns=selected_numeric_features, index=X_train.index
    )
    X_val_numeric_clean = pd.DataFrame(
        X_val_selected, columns=selected_numeric_features, index=X_val.index
    )
    
    # Combine numeric and non-numeric columns
    if non_numeric_cols:
        X_train_clean = pd.concat([X_train_numeric_clean, X_train[non_numeric_cols]], axis=1)
        X_val_clean = pd.concat([X_val_numeric_clean, X_val[non_numeric_cols]], axis=1)
    else:
        X_train_clean = X_train_numeric_clean
        X_val_clean = X_val_numeric_clean

    return X_train_clean, X_val_clean


def apply_feature_engineering(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    remove_low_variance: bool = True,
    variance_threshold: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply comprehensive feature engineering to training and validation data.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        remove_low_variance: Whether to remove low-variance features
        variance_threshold: Threshold for variance-based feature selection

    Returns:
        Tuple of (df_train_engineered, df_val_engineered)
    """
    logging.info("Applying advanced feature engineering...")

    # Apply feature engineering to training data
    df_train = df_train.copy()
    df_train = create_advanced_date_features(df_train)
    df_train = create_age_features(df_train)
    df_train = create_area_features(df_train)
    df_train = create_interaction_features(df_train)
    df_train = create_price_ratio_features(df_train)

    # Apply feature engineering to validation data
    df_val = df_val.copy()
    df_val = create_advanced_date_features(df_val)
    df_val = create_age_features(df_val)
    df_val = create_area_features(df_val)
    df_val = create_interaction_features(df_val)
    df_val = create_price_ratio_features(df_val)

    # Ensure both DataFrames have the same columns
    # Add missing columns with 0s
    train_cols = set(df_train.columns)
    val_cols = set(df_val.columns)

    for col in train_cols - val_cols:
        df_val[col] = 0
    for col in val_cols - train_cols:
        df_train[col] = 0

    # Reorder columns to match
    df_val = df_val[df_train.columns]

    # Remove low variance features if requested
    if remove_low_variance and "PRICE" in df_train.columns:
        # Separate features and target
        X_train = df_train.drop("PRICE", axis=1)
        X_val = df_val.drop("PRICE", axis=1)
        y_train = df_train["PRICE"]
        y_val = df_val["PRICE"]

        # Remove low variance features
        X_train_clean, X_val_clean = remove_low_variance_features(
            X_train, X_val, threshold=variance_threshold
        )

        # Recombine
        df_train = pd.concat([X_train_clean, y_train], axis=1)
        df_val = pd.concat([X_val_clean, y_val], axis=1)

    logging.info(
        f"Feature engineering complete. Final feature count: {len(df_train.columns) - 1}"
    )

    return df_train, df_val
