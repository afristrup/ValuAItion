#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
This example uses Random Forest preprocessing to split data into leaves
and fits separate TabPFN models within each leaf.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from uuid import uuid4
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabpfn import TabPFNRegressor
from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor

from src.data.process import (
    get_raw_data,
    time_based_split,
    transform_values,
    handle_missing_values,
    remove_unused_columns,
)
from src.data.constants import CAT_COLUMNS
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
    logging.info("Loading raw training data...")
    df_train_raw = get_raw_data("train")

    # perform time-based split
    logging.info("Performing time-based split...")
    df_train_split, df_val_split = time_based_split(df_train_raw, test_size=0.1)

    # Using all available training data for RF preprocessing approach
    # RF preprocessing breaks the problem into smaller sub-problems, so it can handle larger datasets
    logging.info(
        f"Using all available training data: {len(df_train_split)} samples "
        f"(from {len(df_train_raw)} total raw samples)"
    )

    # Save validation indices for evaluation
    val_indices = df_val_split.index.values
    np.save(os.path.join(model_dir, "val_indices.npy"), val_indices)

    # Process training data
    logging.info("Processing training data...")
    with tqdm(total=2, desc="Training data processing") as pbar:
        df_train_split = transform_values(
            df_train_split,
            "train",
            run_id,
            data_config["calculate_street_price_sqm"],
            data_config["reduce_zip"],
            data_config["reduce_municipality"],
        )
        pbar.update(1)
        df_train_split = handle_missing_values(
            df_train_split, "train", data_config["reduce_zip"]
        )
        pbar.update(1)

    # Process validation data (using training statistics)
    logging.info("Processing validation data...")
    with tqdm(total=2, desc="Validation data processing") as pbar:
        df_val_split = transform_values(
            df_val_split,
            "train",
            run_id,
            data_config["calculate_street_price_sqm"],
            data_config["reduce_zip"],
            data_config["reduce_municipality"],
        )
        pbar.update(1)
        df_val_split = handle_missing_values(
            df_val_split, "train", data_config["reduce_zip"]
        )
        pbar.update(1)

    # Expand TRADE_DATE for both splits
    logging.info("Expanding TRADE_DATE features...")
    for df in tqdm([df_train_split, df_val_split], desc="Expanding dates"):
        trade_dates = pd.to_datetime(df["TRADE_DATE"])
        df["TRADE_YEAR"] = trade_dates.dt.year.values
        df["TRADE_MONTH"] = trade_dates.dt.month.values
        df["TRADE_DOW"] = trade_dates.dt.dayofweek.values
        df.drop(["TRADE_DATE"], axis=1, inplace=True)

    # Remove unused columns
    df_train_split = remove_unused_columns(df_train_split)
    df_val_split = remove_unused_columns(df_val_split)

    # One-hot encode - need to ensure consistent columns
    # Get all possible categorical values from training data
    df_train_encoded = pd.get_dummies(df_train_split, columns=CAT_COLUMNS, dtype="int8")
    df_val_encoded = pd.get_dummies(df_val_split, columns=CAT_COLUMNS, dtype="int8")

    # Align columns (add missing columns with 0s)
    logging.info("Aligning columns between train and validation sets...")
    train_cols = set(df_train_encoded.columns)
    val_cols = set(df_val_encoded.columns)
    missing_in_val = train_cols - val_cols
    missing_in_train = val_cols - train_cols

    if missing_in_val:
        for col in tqdm(missing_in_val, desc="Adding missing columns to validation"):
            df_val_encoded[col] = 0
    if missing_in_train:
        for col in tqdm(missing_in_train, desc="Adding missing columns to training"):
            df_train_encoded[col] = 0

    # Reorder columns to match
    df_val_encoded = df_val_encoded[df_train_encoded.columns]

    df_train = df_train_encoded.astype("float32")
    df_val = df_val_encoded.astype("float32")

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

    # Initialize base TabPFN regressor
    logging.info(f"Initializing base TabPFNRegressor on device: {device}")
    tabpfn_base = TabPFNRegressor(device=device, ignore_pretraining_limits=True)

    # Initialize RandomForestTabPFNRegressor with RF preprocessing
    # This uses Random Forest to split data into leaves and fits separate TabPFN models
    logging.info("Initializing RandomForestTabPFNRegressor with RF preprocessing...")
    logging.info(
        "This approach splits data into large leaves using Random Forest "
        "and fits separate TabPFN models within each leaf."
    )

    # Default RF parameters - can be tuned if needed
    # Using shallow trees (max_depth=3) for faster training and larger leaves
    regressor = RandomForestTabPFNRegressor(
        tabpfn=tabpfn_base,
        n_estimators=10,  # Number of trees in the Random Forest
        max_depth=3,  # Shallow trees create larger leaves where TabPFN excels
    )

    # TabPFN works with pandas DataFrames directly
    # Convert to DataFrame if needed (already is, but ensure proper format)
    X_train_df = pd.DataFrame(
        X_train.values, columns=X_train.columns, index=X_train.index
    )
    X_val_df = pd.DataFrame(X_val.values, columns=X_val.columns, index=X_val.index)

    # Fit the model
    logging.info("Fitting RandomForestTabPFN model...")
    logging.info("This may take a while as it fits TabPFN models in each RF leaf...")
    with tqdm(total=1, desc="Fitting model") as pbar:
        regressor.fit(X_train_df, y_train)
        pbar.update(1)

    # Predict on training and validation sets
    # logging.info("Making predictions...")
    # with tqdm(total=2, desc="Making predictions") as pbar:
    #     # y_train_pred = regressor.predict(X_train_df)
    #     y_train_pred = np.zeros(len(X_train_df))
    #     pbar.update(1)
    #     y_val_pred = regressor.predict(X_val_df)
    #     pbar.update(1)

    # # Inverse log transform if log_y was used
    # if data_config["log_y"]:
    #     y_train_pred = np.exp(y_train_pred)
    #     y_val_pred = np.exp(y_val_pred)
    #     y_train_actual = y_train_original
    #     y_val_actual = y_val_original
    # else:
    #     y_train_actual = y_train_original
    #     y_val_actual = y_val_original

    # # Compute metrics
    # train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    # val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
    # train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    # val_mae = mean_absolute_error(y_val_actual, y_val_pred)
    # train_r2 = r2_score(y_train_actual, y_train_pred)
    # val_r2 = r2_score(y_val_actual, y_val_pred)

    # model_scores = {
    #     "train_rmse": float(np.round(train_rmse, 5)),
    #     "val_rmse": float(np.round(val_rmse, 5)),
    #     "train_mae": float(np.round(train_mae, 5)),
    #     "val_mae": float(np.round(val_mae, 5)),
    #     "train_r2": float(np.round(train_r2, 5)),
    #     "val_r2": float(np.round(val_r2, 5)),
    # }

    # logging.info(f"Training RMSE: {train_rmse:.5f}, Validation RMSE: {val_rmse:.5f}")
    # logging.info(f"Training R²: {train_r2:.5f}, Validation R²: {val_r2:.5f}")

    # Save model along with other stuff
    logging.info("Saving run with ID %s", run_id)

    # RandomForestTabPFNRegressor may need special handling for saving
    # Try to save using joblib as it's a composite model
    import joblib

    joblib.dump(regressor, os.path.join(model_dir, "model.pkl"))
    logging.info("Saved RandomForestTabPFNRegressor using joblib")

    # Save model config
    model_config = {
        "model_type": "RandomForestTabPFNRegressor",
        "model_version": "2.5",  # Default version
        "device": device,
        "log_y": data_config["log_y"],
        "n_features": len(X_train.columns),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "rf_preprocessing": True,
        "n_estimators": 10,
        "max_depth": 3,
    }
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # with open(os.path.join(model_dir, "scores.json"), "w") as f:
    #     json.dump(model_scores, f, indent=4)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    main()
