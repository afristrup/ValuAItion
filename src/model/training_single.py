import os
import json
import torch
import joblib
import logging
import numpy as np
import pandas as pd
from uuid import uuid4
from pathlib import Path
from xgboost import XGBRegressor

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

    # setup model run dir
    run_id = str(uuid4())
    model_dir = os.path.join("models", run_id)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # load data config
    with open("data_config.json", "r") as f:
        data_config = json.load(f)

    # load model and data configuration
    with open("model_config.json", "r") as f:
        model_config = json.load(f)
    with open("data_config.json", "r") as f:
        data_config = json.load(f)

    # load raw training data for time-based splitting
    df_train_raw = get_raw_data("train")

    # perform time-based split
    df_train_split, df_val_split = time_based_split(df_train_raw, test_size=0.05)

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

    # Expand TRADE_DATE for both splits
    for df in [df_train_split, df_val_split]:
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

    # split into features and labels
    X_train = df_train.drop("PRICE", axis=1)
    y_train = df_train["PRICE"].values
    X_val = df_val.drop("PRICE", axis=1)
    y_val = df_val["PRICE"].values

    if data_config["log_y"]:
        y_train = np.log(y_train)
        y_val = np.log(y_val)

    # store training features in list
    with open(os.path.join(model_dir, "train_features.txt"), "w") as f:
        f.writelines("\n".join(X_train.columns.tolist()))

    # move all arrays to device
    X_train_tensor = torch.from_numpy(X_train.values).to(device)
    X_val_tensor = torch.from_numpy(X_val.values).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    # define model
    model = XGBRegressor(**model_config, device=device)

    # fit on training data
    model = model.fit(X_train_tensor, y_train_tensor)

    # compute scores
    model_scores = {
        "train_score": np.round(model.score(X_train_tensor, y_train_tensor), 5),
        "val_score": np.round(model.score(X_val_tensor, y_val_tensor), 5),
    }

    # save model along with other stuff
    logging.info("Saving run with ID %s", run_id)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(model_dir, "scores.json"), "w") as f:
        json.dump(model_scores, f, indent=4)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    main()
