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
        # Expected price based on street average and area
        df["EXPECTED_PRICE_STREET"] = (
            df["STREET_CODE_MEAN_SQM_PRICE"] * df["AREA_RESIDENTIAL"]
        )

    return df


def create_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create polynomial features for key numeric variables to capture non-linear relationships.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with polynomial features
    """
    df = df.copy()

    # Key features to create polynomial terms for
    key_features = []
    if "AREA_RESIDENTIAL" in df.columns:
        key_features.append("AREA_RESIDENTIAL")
    if "FLOOR" in df.columns:
        key_features.append("FLOOR")
    if "PROPERTY_AGE" in df.columns:
        key_features.append("PROPERTY_AGE")
    if "TRADE_YEAR" in df.columns:
        key_features.append("TRADE_YEAR")

    # Create squared terms (degree 2)
    for feat in key_features:
        df[f"{feat}_SQUARED"] = df[feat] ** 2

    # Create square root terms (useful for area features)
    if "AREA_RESIDENTIAL" in df.columns:
        df["AREA_RESIDENTIAL_SQRT"] = np.sqrt(
            df["AREA_RESIDENTIAL"].clip(lower=0)
        )

    return df


def create_binning_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binned/categorical versions of continuous variables.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with binned features
    """
    df = df.copy()

    # Bin area features
    if "AREA_RESIDENTIAL" in df.columns:
        df["AREA_RESIDENTIAL_BINNED"] = pd.cut(
            df["AREA_RESIDENTIAL"],
            bins=[0, 50, 75, 100, 125, 150, 200, np.inf],
            labels=[0, 1, 2, 3, 4, 5, 6],
            include_lowest=True,
        ).astype(float)

    # Bin floor features
    if "FLOOR" in df.columns:
        df["FLOOR_BINNED"] = pd.cut(
            df["FLOOR"],
            bins=[-np.inf, 0, 2, 5, 10, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype(float)

    # Bin trade year (capture market periods)
    if "TRADE_YEAR" in df.columns:
        year_min = df["TRADE_YEAR"].min()
        year_max = df["TRADE_YEAR"].max()
        if year_max - year_min > 5:  # Only bin if there's sufficient range
            n_bins = min(5, int((year_max - year_min) / 2))
            df["TRADE_YEAR_BINNED"] = pd.cut(
                df["TRADE_YEAR"],
                bins=n_bins,
                labels=range(n_bins),
                include_lowest=True,
            ).astype(float)

    return df


def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create location-based features from coordinates if available.

    Args:
        df: DataFrame with potential LAT/LNG columns

    Returns:
        DataFrame with location features
    """
    df = df.copy()

    # Check if coordinates are available
    has_lat = "LAT" in df.columns
    has_lng = "LNG" in df.columns

    if has_lat and has_lng:
        # Denmark approximate center (Copenhagen)
        DENMARK_CENTER_LAT = 56.2639
        DENMARK_CENTER_LNG = 9.5018

        # Distance from center (approximate, in degrees)
        df["DIST_FROM_CENTER"] = np.sqrt(
            (df["LAT"] - DENMARK_CENTER_LAT) ** 2
            + (df["LNG"] - DENMARK_CENTER_LNG) ** 2
        )

        # Geographic regions (rough bins)
        # Copenhagen area (roughly lat 55.6-55.8, lng 12.4-12.7)
        df["IS_COPENHAGEN_AREA"] = (
            (df["LAT"] >= 55.6)
            & (df["LAT"] <= 55.8)
            & (df["LNG"] >= 12.4)
            & (df["LNG"] <= 12.7)
        ).astype(int)

        # Note: We don't drop LAT/LNG here as they might be useful for external data joins
        # They will be dropped in remove_unused_columns if needed

    return df


def create_aggregated_price_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create aggregated price features at ZIP and municipality level.
    Must be called with both train and val to ensure consistency.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame

    Returns:
        Tuple of (df_train, df_val) with aggregated features
    """
    df_train = df_train.copy()
    df_val = df_val.copy()

    # Only create if PRICE column exists and we have location columns
    if "PRICE" not in df_train.columns:
        return df_train, df_val

    # Store original index names
    train_index_name = df_train.index.name
    val_index_name = df_val.index.name

    # Reset index temporarily for merging
    df_train_reset = df_train.reset_index()
    df_val_reset = df_val.reset_index()

    # ZIP level aggregations (from training data only)
    if "ZIP_AREA" in df_train_reset.columns:
        zip_stats = df_train_reset.groupby("ZIP_AREA")["PRICE"].agg(
            ["mean", "median", "std"]
        )
        zip_stats.columns = [
            "ZIP_PRICE_MEAN",
            "ZIP_PRICE_MEDIAN",
            "ZIP_PRICE_STD",
        ]

        # Map to both train and val
        df_train_reset = df_train_reset.merge(
            zip_stats, left_on="ZIP_AREA", right_index=True, how="left"
        )
        df_val_reset = df_val_reset.merge(
            zip_stats, left_on="ZIP_AREA", right_index=True, how="left"
        )

        # Fill missing with global mean/median
        for col in ["ZIP_PRICE_MEAN", "ZIP_PRICE_MEDIAN"]:
            if col in df_train_reset.columns:
                global_val = df_train_reset[col].mean()
                df_train_reset[col] = df_train_reset[col].fillna(global_val)
                df_val_reset[col] = df_val_reset[col].fillna(global_val)

    # Municipality level aggregations
    if "MUNICIPALITY" in df_train_reset.columns:
        mun_stats = df_train_reset.groupby("MUNICIPALITY")["PRICE"].agg(
            ["mean", "median", "std"]
        )
        mun_stats.columns = [
            "MUNICIPALITY_PRICE_MEAN",
            "MUNICIPALITY_PRICE_MEDIAN",
            "MUNICIPALITY_PRICE_STD",
        ]

        # Map to both train and val
        df_train_reset = df_train_reset.merge(
            mun_stats, left_on="MUNICIPALITY", right_index=True, how="left"
        )
        df_val_reset = df_val_reset.merge(
            mun_stats, left_on="MUNICIPALITY", right_index=True, how="left"
        )

        # Fill missing with global mean/median
        for col in ["MUNICIPALITY_PRICE_MEAN", "MUNICIPALITY_PRICE_MEDIAN"]:
            if col in df_train_reset.columns:
                global_val = df_train_reset[col].mean()
                df_train_reset[col] = df_train_reset[col].fillna(global_val)
                df_val_reset[col] = df_val_reset[col].fillna(global_val)

    # Restore index
    if train_index_name and train_index_name in df_train_reset.columns:
        df_train = df_train_reset.set_index(train_index_name)
    else:
        df_train = df_train_reset.set_index(df_train_reset.columns[0])

    if val_index_name and val_index_name in df_val_reset.columns:
        df_val = df_val_reset.set_index(val_index_name)
    else:
        df_val = df_val_reset.set_index(df_val_reset.columns[0])

    return df_train, df_val


def create_enhanced_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional meaningful interaction features.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with enhanced interaction features
    """
    df = df.copy()

    # Street price * Area (expected value interaction)
    if (
        "STREET_CODE_MEAN_SQM_PRICE" in df.columns
        and "AREA_RESIDENTIAL" in df.columns
    ):
        df["STREET_PRICE_X_AREA"] = (
            df["STREET_CODE_MEAN_SQM_PRICE"] * df["AREA_RESIDENTIAL"]
        )

    # Age * Year (captures how age affects value over time)
    if "PROPERTY_AGE" in df.columns and "TRADE_YEAR" in df.columns:
        df["AGE_X_YEAR"] = df["PROPERTY_AGE"] * df["TRADE_YEAR"]

    # Total area * Floor (larger units on higher floors)
    if "TOTAL_AREA" in df.columns and "FLOOR" in df.columns:
        df["TOTAL_AREA_X_FLOOR"] = df["TOTAL_AREA"] * df["FLOOR"]

    # Elevator * Age (elevators more valuable in older buildings)
    if "HAS_ELEVATOR" in df.columns and "PROPERTY_AGE" in df.columns:
        df["ELEVATOR_X_AGE"] = df["HAS_ELEVATOR"] * df["PROPERTY_AGE"]

    # Residential ratio * Floor (efficiency by floor)
    if "RESIDENTIAL_AREA_RATIO" in df.columns and "FLOOR" in df.columns:
        df["RESIDENTIAL_RATIO_X_FLOOR"] = (
            df["RESIDENTIAL_AREA_RATIO"] * df["FLOOR"]
        )

    return df


def load_external_features(
    df: pd.DataFrame, external_data_path: str = None
) -> pd.DataFrame:
    """
    Load and merge external features (e.g., school distances).

    Args:
        df: DataFrame to merge features into
        external_data_path: Path to external data CSV (e.g., closest school distances)

    Returns:
        DataFrame with external features merged
    """
    df = df.copy()

    if external_data_path is None:
        # Try default path
        external_data_path = "datasets/apartment_closest_school.csv"

    try:
        import os

        if os.path.exists(external_data_path):
            external_df = pd.read_csv(external_data_path)
            # Assume TRANSACTION_ID is the key
            if "TRANSACTION_ID" in external_df.columns:
                # Get distance to closest school
                if "distance_m" in external_df.columns:
                    closest_school = (
                        external_df.groupby("TRANSACTION_ID")["distance_m"]
                        .min()
                        .reset_index()
                    )
                    closest_school.columns = ["TRANSACTION_ID", "DIST_TO_CLOSEST_SCHOOL"]
                    
                    # Reset index for merging
                    index_name = df.index.name
                    df_reset = df.reset_index()
                    
                    # Merge
                    df_reset = df_reset.merge(
                        closest_school,
                        left_on=index_name if index_name else df_reset.columns[0],
                        right_on="TRANSACTION_ID",
                        how="left",
                    )
                    
                    # Fill missing with a large value (no school nearby)
                    if "DIST_TO_CLOSEST_SCHOOL" in df_reset.columns:
                        df_reset["DIST_TO_CLOSEST_SCHOOL"] = df_reset[
                            "DIST_TO_CLOSEST_SCHOOL"
                        ].fillna(50000)  # 50km default
                    
                    # Drop TRANSACTION_ID if it was added
                    if "TRANSACTION_ID" in df_reset.columns:
                        df_reset = df_reset.drop("TRANSACTION_ID", axis=1)
                    
                    # Restore index
                    if index_name:
                        df = df_reset.set_index(index_name)
                    else:
                        df = df_reset.set_index(df_reset.columns[0])
                        
            logging.info(f"Loaded external features from {external_data_path}")
        else:
            logging.debug(
                f"External data file not found at {external_data_path}, skipping"
            )
    except Exception as e:
        logging.warning(f"Could not load external features: {e}")

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
    include_external_data: bool = False,
    external_data_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply comprehensive feature engineering to training and validation data.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        remove_low_variance: Whether to remove low-variance features
        variance_threshold: Threshold for variance-based feature selection
        include_external_data: Whether to include external data features (e.g., school distances)
        external_data_path: Path to external data CSV file

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
    df_train = create_polynomial_features(df_train)
    df_train = create_binning_features(df_train)
    df_train = create_location_features(df_train)
    df_train = create_enhanced_interaction_features(df_train)

    # Apply feature engineering to validation data
    df_val = df_val.copy()
    df_val = create_advanced_date_features(df_val)
    df_val = create_age_features(df_val)
    df_val = create_area_features(df_val)
    df_val = create_interaction_features(df_val)
    df_val = create_price_ratio_features(df_val)
    df_val = create_polynomial_features(df_val)
    df_val = create_binning_features(df_val)
    df_val = create_location_features(df_val)
    df_val = create_enhanced_interaction_features(df_val)

    # Aggregated price features (must be done together to ensure consistency)
    df_train, df_val = create_aggregated_price_features(df_train, df_val)

    # External data features (if requested)
    if include_external_data:
        df_train = load_external_features(df_train, external_data_path)
        df_val = load_external_features(df_val, external_data_path)

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
