import os
import json
import torch
import click
import joblib
import logging
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data.process import (
    get_data,
    get_raw_data,
    time_based_split,
    transform_values,
    handle_missing_values,
    remove_unused_columns,
)
from src.data.constants import CAT_COLUMNS
from src.model.utils import get_device


@click.command()
@click.option("--run_id", help="Run ID of trained model.")
@click.option(
    "--split", default="test", help="Data split to evaluate on (test, train, or val)."
)
@click.option(
    "--submit", is_flag=True, help="Whether to submit test predictions to endpoint."
)
def main(run_id: str, split: str, submit: bool):
    model_dir = os.path.join("models", run_id)
    save_path = os.path.join(model_dir, f"df_{split}.csv")
    if os.path.exists(save_path):
        logging.info("Loading previously computed predictions.")
        df_eval = pd.read_csv(save_path)
    else:
        logging.info(f"Computing predictions for {split} set.")
        df_eval = compute_metrics(model_dir, run_id, split)
        df_eval.to_csv(save_path)

    if submit:
        if split != "test":
            logging.warning("Submit flag is only valid for test set. Ignoring.")
        else:
            predictions = df_eval[["TRANSACTION_ID", "PRICE"]].to_dict("records")
            upload_results(predictions, model_dir)


def compute_metrics(model_dir: str, run_id: str, split: str = "test"):
    device = get_device()

    # load data config
    with open("data_config.json", "r") as f:
        data_config = json.load(f)

    # handle validation split specially - it needs to be created from training data
    if split == "val":
        # Check if split indices were saved during training
        split_indices_path = os.path.join(model_dir, "val_indices.npy")
        if os.path.exists(split_indices_path):
            # Load raw training data and recreate the time-based split
            df_train_raw = get_raw_data("train")
            df_train_split, df_val_split = time_based_split(df_train_raw, test_size=0.1)

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

            # Expand TRADE_DATE
            trade_dates = pd.to_datetime(df_val_split["TRADE_DATE"])
            df_val_split["TRADE_YEAR"] = trade_dates.dt.year.values
            df_val_split["TRADE_MONTH"] = trade_dates.dt.month.values
            df_val_split["TRADE_DOW"] = trade_dates.dt.dayofweek.values
            df_val_split.drop(["TRADE_DATE"], axis=1, inplace=True)

            # Remove unused columns and one-hot encode
            df_val_split = remove_unused_columns(df_val_split)
            df_val_encoded = pd.get_dummies(
                df_val_split, columns=CAT_COLUMNS, dtype="int8"
            )

            # Load feature names used for training
            with open(os.path.join(model_dir, "train_features.txt"), "r") as f:
                train_features = [x.strip() for x in f.readlines()]

            # Align columns with training features
            missing_features = [
                x for x in train_features if x not in df_val_encoded.columns
            ]
            for feature in missing_features:
                df_val_encoded.loc[:, feature] = 0

            # Remove extra features and reorder
            df_val_encoded = df_val_encoded[train_features]
            df_eval = df_val_encoded.astype("float32")
        else:
            # Recreate time-based split if indices not found
            logging.warning(
                "Validation split indices not found. Recreating time-based split. "
                "This may not match the exact split used during training."
            )
            df_train_raw = get_raw_data("train")
            _, df_val_split = time_based_split(df_train_raw, test_size=0.1)

            # Process validation data
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

            # Expand TRADE_DATE
            trade_dates = pd.to_datetime(df_val_split["TRADE_DATE"])
            df_val_split["TRADE_YEAR"] = trade_dates.dt.year.values
            df_val_split["TRADE_MONTH"] = trade_dates.dt.month.values
            df_val_split["TRADE_DOW"] = trade_dates.dt.dayofweek.values
            df_val_split.drop(["TRADE_DATE"], axis=1, inplace=True)

            df_val_split = remove_unused_columns(df_val_split)
            df_val_encoded = pd.get_dummies(
                df_val_split, columns=CAT_COLUMNS, dtype="int8"
            )

            # Load feature names used for training
            with open(os.path.join(model_dir, "train_features.txt"), "r") as f:
                train_features = [x.strip() for x in f.readlines()]

            # Align columns with training features
            missing_features = [
                x for x in train_features if x not in df_val_encoded.columns
            ]
            for feature in missing_features:
                df_val_encoded.loc[:, feature] = 0

            # Remove extra features and reorder
            df_val_encoded = df_val_encoded[train_features]
            df_eval = df_val_encoded.astype("float32")

        # Extract y_true from the validation split
        y_true = df_eval["PRICE"].values.copy()
        has_labels = True
        X_eval = df_eval.drop("PRICE", axis=1).copy()
    else:
        # load evaluation data for test or train splits
        df_eval = get_data(split, run_id, **data_config)

        # check if labels are available and save them
        has_labels = "PRICE" in df_eval.columns
        if has_labels:
            y_true = df_eval["PRICE"].values.copy()
            X_eval = df_eval.drop("PRICE", axis=1).copy()
        else:
            X_eval = df_eval.copy()
            y_true = None

    # load feature names used for training
    with open(os.path.join(model_dir, "train_features.txt"), "r") as f:
        train_features = [x.strip() for x in f.readlines()]

    # add missing features
    missing_features = [x for x in train_features if x not in X_eval.columns]
    for feature in missing_features:
        X_eval.loc[:, feature] = 0

    # remove extra features and reorder
    X_eval = X_eval[train_features]

    # scale features if applicable for model
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_eval_scaled = scaler.transform(X_eval)
    else:
        X_eval_scaled = X_eval.values

    # move data to device
    X_eval_tensor = torch.from_numpy(X_eval_scaled).to(device)

    # load model
    model = joblib.load(os.path.join(model_dir, "model.pkl"))

    # predict
    y_hat = model.predict(X_eval_tensor)
    if data_config["log_y"]:
        y_hat = np.exp(y_hat)

    # compute metrics if labels are available
    if has_labels:
        # Filter out NaN values for metric computation
        valid_mask = ~np.isnan(y_true)
        y_true_valid = y_true[valid_mask]
        y_hat_valid = y_hat[valid_mask]

        if len(y_true_valid) == 0:
            logging.warning(
                f"No valid labels found in {split} set. Skipping metrics computation."
            )
        else:
            n_total = len(y_true)
            n_valid = len(y_true_valid)
            n_missing = n_total - n_valid

            # Calculate relative errors
            relative_errors = np.abs((y_true_valid - y_hat_valid) / y_true_valid)

            # Calculate precision metrics (percentage within threshold)
            precision_20 = float(np.mean(relative_errors <= 0.20) * 100)
            precision_10 = float(np.mean(relative_errors <= 0.10) * 100)
            precision_5 = float(np.mean(relative_errors <= 0.05) * 100)

            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_true_valid, y_hat_valid))),
                "mae": float(mean_absolute_error(y_true_valid, y_hat_valid)),
                "r2": float(r2_score(y_true_valid, y_hat_valid)),
                "mean_absolute_percentage_error": float(np.mean(relative_errors) * 100),
                "precision_20pct": precision_20,
                "precision_10pct": precision_10,
                "precision_5pct": precision_5,
                "n_samples": int(n_total),
                "n_valid_samples": int(n_valid),
                "n_missing_labels": int(n_missing),
            }

            # save metrics
            metrics_path = os.path.join(model_dir, f"{split}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(
                f"{split.upper()} set metrics (computed on {n_valid}/{n_total} samples with valid labels):"
            )
            # Log precision metrics first in the format requested
            logging.info(f"  max ±20% precision: {metrics['precision_20pct']:.2f}%")
            logging.info(f"  max ±10% precision: {metrics['precision_10pct']:.2f}%")
            logging.info(f"  max ±5% precision: {metrics['precision_5pct']:.2f}%")
            # Log other metrics
            for metric_name, metric_value in metrics.items():
                if metric_name not in [
                    "precision_20pct",
                    "precision_10pct",
                    "precision_5pct",
                    "n_samples",
                    "n_valid_samples",
                    "n_missing_labels",
                ]:
                    logging.info(f"  {metric_name}: {metric_value:.5f}")
            # Log sample counts
            for metric_name in ["n_samples", "n_valid_samples", "n_missing_labels"]:
                logging.info(f"  {metric_name}: {metrics[metric_name]}")

    # format df_eval with predictions
    df_eval = df_eval.reset_index(drop=False)
    df_eval["PRICE"] = y_hat
    return df_eval


def upload_results(predictions: dict, model_dir: str):
    # load name and email from .env file
    load_dotenv(override=True)
    submit_name = os.getenv("SUBMIT_NAME", None)
    submit_email = os.getenv("SUBMIT_EMAIL", None)
    assert submit_name is not None, "Please set SUBMIT_NAME in .env file."
    assert submit_email is not None, "Please set SUBMIT_EMAIL in .env file."

    logging.info(
        "Submitting results to API with name: %s and email: %s",
        submit_name,
        submit_email,
    )

    r = requests.post(
        url="https://api.resights.dk/hackathon/avm/ejerlejligheder/v1",
        json={
            "name": submit_name,
            "email": submit_email,
            "predictions": predictions,
        },
    )

    with open(os.path.join(model_dir, "upload_result.json"), "w") as f:
        json.dump(r.json(), f, indent=4)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    main()
