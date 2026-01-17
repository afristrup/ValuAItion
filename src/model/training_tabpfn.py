import os
import json
import logging
import numpy as np
import pandas as pd
from uuid import uuid4
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabpfn import TabPFNRegressor

from src.data.process import (
    get_raw_data,
    time_based_split,
    transform_values,
    handle_missing_values,
    remove_unused_columns,
)
from src.data.constants import CAT_COLUMNS
from src.data.feature_engineering import apply_feature_engineering
from src.model.utils import get_device


def main():
    device = get_device()

    # TabPFN works best on GPU/MPS, warn if using CPU
    if device == "mps":
        logging.info("TabPFN is running on MPS (Metal Performance Shaders).")
        # Note: Potential float64 to float32 issues may arise with MPS
    elif device == "cuda":
        logging.info("TabPFN is running on CUDA GPU.")
    elif device == "cpu":
        logging.warning(
            "TabPFN is running on CPU. For optimal performance, use a GPU or MPS. "
            "CPU is only feasible for small datasets (≲1000 samples)."
        )

    # setup model run dir
    run_id = str(uuid4())
    model_dir = os.path.join("models", run_id)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # load data config
    with open("data_config.json", "r") as f:
        data_config = json.load(f)

    # load raw training data for time-based splitting
    df_train_raw = get_raw_data("train")

    # perform time-based split
    df_train_split, df_val_split = time_based_split(df_train_raw, test_size=0.1)

    # Limit training data to most recent 50,000 samples (sorted by TRADE_DATE)
    # This helps with TabPFN performance on CPU and memory constraints
    if "TRADE_DATE" in df_train_split.columns:
        df_train_split = df_train_split.sort_values("TRADE_DATE")
        df_train_split = df_train_split.tail(50000)
        logging.info(
            f"Limited training data to most recent 50,000 samples (from {len(df_train_raw)} total)"
        )
    else:
        logging.warning("TRADE_DATE not found, using last 50,000 samples by index")
        df_train_split = df_train_split.tail(50000)

    # Save validation indices for evaluation
    val_indices = df_val_split.index.values
    np.save(os.path.join(model_dir, "val_indices.npy"), val_indices)

    # Process training data
    df_train_split = transform_values(
        df_train_split,
        "train",
        run_id,
        data_config["calculate_street_price_sqm"],
        data_config["reduce_zip"],
        data_config["reduce_municipality"],
    )
    df_train_split = handle_missing_values(
        df_train_split, "train", data_config["reduce_zip"]
    )

    # Process validation data (using training statistics)
    df_val_split = transform_values(
        df_val_split,
        "train",
        run_id,
        data_config["calculate_street_price_sqm"],
        data_config["reduce_zip"],
        data_config["reduce_municipality"],
    )
    df_val_split = handle_missing_values(
        df_val_split, "train", data_config["reduce_zip"]
    )

    # Remove unused columns before feature engineering
    df_train_split = remove_unused_columns(df_train_split)
    df_val_split = remove_unused_columns(df_val_split)

    # Apply optimized feature engineering (includes date expansion, age features,
    # area features, interactions, and low-variance feature removal)
    df_train_split, df_val_split = apply_feature_engineering(
        df_train_split,
        df_val_split,
        remove_low_variance=True,
        variance_threshold=0.01,
    )

    # One-hot encode - need to ensure consistent columns
    # Get all possible categorical values from training data
    # Note: CAT_COLUMNS may have been transformed, so we need to identify categorical columns
    # that still exist and haven't been one-hot encoded yet
    remaining_cat_cols = [col for col in CAT_COLUMNS if col in df_train_split.columns]

    if remaining_cat_cols:
        df_train_encoded = pd.get_dummies(
            df_train_split, columns=remaining_cat_cols, dtype="int8"
        )
        df_val_encoded = pd.get_dummies(
            df_val_split, columns=remaining_cat_cols, dtype="int8"
        )

        # Align columns (add missing columns with 0s)
        train_cols = set(df_train_encoded.columns)
        val_cols = set(df_val_encoded.columns)
        for col in train_cols - val_cols:
            df_val_encoded[col] = 0
        for col in val_cols - train_cols:
            df_train_encoded[col] = 0

        # Reorder columns to match
        df_val_encoded = df_val_encoded[df_train_encoded.columns]

        df_train = df_train_encoded.astype("float32")
        df_val = df_val_encoded.astype("float32")
    else:
        # No categorical columns to encode
        df_train = df_train_split.astype("float32")
        df_val = df_val_split.astype("float32")

    # split into features and labels
    X_train = df_train.drop("PRICE", axis=1)
    y_train = df_train["PRICE"].values
    X_val = df_val.drop("PRICE", axis=1)
    y_val = df_val["PRICE"].values

    # Store original y values for inverse transform
    y_train_original = y_train.copy()
    y_val_original = y_val.copy()

    if data_config["log_y"]:
        y_train = np.log(y_train)
        y_val = np.log(y_val)

    # store training features in list
    with open(os.path.join(model_dir, "train_features.txt"), "w") as f:
        f.writelines("\n".join(X_train.columns.tolist()))

    # Initialize TabPFN regressor
    # TabPFN-2.5 is used by default (trained on synthetic data only for regression)
    # For TabPFN v2, use: TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    # (requires: from tabpfn.constants import ModelVersion)
    logging.info(f"Initializing TabPFNRegressor on device: {device}")
    regressor = TabPFNRegressor(device=device, ignore_pretraining_limits=True)

    # TabPFN works with pandas DataFrames directly
    # Convert to DataFrame if needed (already is, but ensure proper format)
    X_train_df = pd.DataFrame(
        X_train.values, columns=X_train.columns, index=X_train.index
    )
    X_val_df = pd.DataFrame(X_val.values, columns=X_val.columns, index=X_val.index)

    # Fit the model
    logging.info("Fitting TabPFN model...")
    regressor.fit(X_train_df, y_train)

    # Predict on training and validation sets
    logging.info("Making predictions...")
    y_train_pred = regressor.predict(X_train_df)
    y_val_pred = regressor.predict(X_val_df)

    # Inverse log transform if log_y was used
    if data_config["log_y"]:
        y_train_pred = np.exp(y_train_pred)
        y_val_pred = np.exp(y_val_pred)
        y_train_actual = y_train_original
        y_val_actual = y_val_original
    else:
        y_train_actual = y_train_original
        y_val_actual = y_val_original

    # Compute metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    val_mae = mean_absolute_error(y_val_actual, y_val_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)
    val_r2 = r2_score(y_val_actual, y_val_pred)

    model_scores = {
        "train_rmse": float(np.round(train_rmse, 5)),
        "val_rmse": float(np.round(val_rmse, 5)),
        "train_mae": float(np.round(train_mae, 5)),
        "val_mae": float(np.round(val_mae, 5)),
        "train_r2": float(np.round(train_r2, 5)),
        "val_r2": float(np.round(val_r2, 5)),
    }

    logging.info(f"Training RMSE: {train_rmse:.5f}, Validation RMSE: {val_rmse:.5f}")
    logging.info(f"Training R²: {train_r2:.5f}, Validation R²: {val_r2:.5f}")

    # Save model along with other stuff
    logging.info("Saving run with ID %s", run_id)

    # TabPFN models can be saved using the save_fitted_tabpfn_model function
    from tabpfn.model_loading import save_fitted_tabpfn_model

    save_fitted_tabpfn_model(regressor, os.path.join(model_dir, "model.tabpfn_fit"))

    # Save model config (TabPFN doesn't have hyperparameters, but save metadata)
    model_config = {
        "model_type": "TabPFNRegressor",
        "model_version": "2.5",  # Default version
        "device": device,
        "log_y": data_config["log_y"],
        "n_features": len(X_train.columns),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
    }
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    with open(os.path.join(model_dir, "scores.json"), "w") as f:
        json.dump(model_scores, f, indent=4)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    main()
