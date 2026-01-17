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
from src.data.feature_engineering import (
    create_advanced_date_features,
    create_age_features,
    create_area_features,
    create_interaction_features,
    create_price_ratio_features,
)
from src.model.utils import get_device


@click.command()
@click.option("--run_id", help="Run ID of trained model.")
@click.option(
    "--split", default="test", help="Data split to evaluate on (test, train, or val)."
)
@click.option(
    "--submit", is_flag=True, help="Whether to submit test predictions to endpoint."
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit the number of predictions to compute (e.g., 100).",
)
def main(run_id: str, split: str, submit: bool, limit: int):
    model_dir = os.path.join("models", run_id)
    save_path = os.path.join(model_dir, f"df_{split}.csv")
    if os.path.exists(save_path) and limit is None:
        logging.info("Loading previously computed predictions.")
        df_eval = pd.read_csv(save_path)
    else:
        if limit is not None:
            logging.info(f"Computing predictions for {split} set with limit={limit}.")
        else:
            logging.info(f"Computing predictions for {split} set.")
        df_eval = compute_metrics(model_dir, run_id, split, limit=limit)
        if limit is None:
            df_eval.to_csv(save_path)

    if submit:
        if split != "test":
            logging.warning("Submit flag is only valid for test set. Ignoring.")
        else:
            predictions = df_eval[["TRANSACTION_ID", "PRICE"]].to_dict("records")
            upload_results(predictions, model_dir)


def compute_metrics(
    model_dir: str, run_id: str, split: str = "test", limit: int = None
):
    device = get_device()
    if device == "cuda":
        logging.info("Using CUDA GPU for evaluation")
    elif device == "mps":
        logging.info("Using MPS (Metal) for evaluation")
    else:
        logging.warning("Using CPU for evaluation - GPU acceleration not available")

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

            # Remove unused columns before feature engineering
            df_val_split = remove_unused_columns(df_val_split)

            # Apply the same feature engineering as training
            df_val_split = create_advanced_date_features(df_val_split)
            df_val_split = create_age_features(df_val_split)
            df_val_split = create_area_features(df_val_split)
            df_val_split = create_interaction_features(df_val_split)
            df_val_split = create_price_ratio_features(df_val_split)

            # One-hot encode categorical columns
            remaining_cat_cols = [col for col in CAT_COLUMNS if col in df_val_split.columns]
            if remaining_cat_cols:
                df_val_encoded = pd.get_dummies(
                    df_val_split, columns=remaining_cat_cols, dtype="int8"
                )
            else:
                df_val_encoded = df_val_split.copy()

            # Extract PRICE before filtering to train_features (which doesn't include PRICE)
            y_true = df_val_encoded["PRICE"].values.copy()
            df_val_encoded = df_val_encoded.drop("PRICE", axis=1)

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

            # Remove unused columns before feature engineering
            df_val_split = remove_unused_columns(df_val_split)

            # Apply the same feature engineering as training
            df_val_split = create_advanced_date_features(df_val_split)
            df_val_split = create_age_features(df_val_split)
            df_val_split = create_area_features(df_val_split)
            df_val_split = create_interaction_features(df_val_split)
            df_val_split = create_price_ratio_features(df_val_split)

            # One-hot encode categorical columns
            remaining_cat_cols = [col for col in CAT_COLUMNS if col in df_val_split.columns]
            if remaining_cat_cols:
                df_val_encoded = pd.get_dummies(
                    df_val_split, columns=remaining_cat_cols, dtype="int8"
                )
            else:
                df_val_encoded = df_val_split.copy()

            # Extract PRICE before filtering to train_features (which doesn't include PRICE)
            y_true = df_val_encoded["PRICE"].values.copy()
            df_val_encoded = df_val_encoded.drop("PRICE", axis=1)

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

        # y_true is already extracted above for both validation split branches
        has_labels = True
        X_eval = df_eval.copy()
    else:
        # load evaluation data for test or train splits
        # Process data using the same pipeline as training
        df_eval = get_raw_data(split)
        df_eval = transform_values(
            df_eval,
            split,
            run_id,
            data_config["calculate_street_price_sqm"],
            data_config["reduce_zip"],
            data_config["reduce_municipality"],
        )
        df_eval = handle_missing_values(df_eval, split, data_config["reduce_zip"])
        df_eval = remove_unused_columns(df_eval)

        # Apply the same feature engineering as training
        df_eval = create_advanced_date_features(df_eval)
        df_eval = create_age_features(df_eval)
        df_eval = create_area_features(df_eval)
        df_eval = create_interaction_features(df_eval)
        df_eval = create_price_ratio_features(df_eval)

        # One-hot encode categorical columns
        remaining_cat_cols = [col for col in CAT_COLUMNS if col in df_eval.columns]
        if remaining_cat_cols:
            df_eval = pd.get_dummies(df_eval, columns=remaining_cat_cols, dtype="int8")
        else:
            df_eval = df_eval.copy()

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

    # apply limit if specified
    if limit is not None and limit > 0:
        logging.info(f"Limiting evaluation to {limit} samples.")
        X_eval = X_eval.head(limit)
        df_eval = df_eval.head(limit).copy()
        if has_labels:
            y_true = y_true[:limit]

    # scale features if applicable for model
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_eval_scaled = scaler.transform(X_eval)
    else:
        X_eval_scaled = X_eval.values

    # load model with GPU support
    model_config_path = os.path.join(model_dir, "model_config.json")
    model_type = None
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            model_config_data = json.load(f)
            model_type = model_config_data.get("model_type", None)

    # Try to load TabPFN models with proper device configuration
    tabpfn_model_path = os.path.join(model_dir, "model.tabpfn_fit")
    if model_type and "TabPFN" in model_type and os.path.exists(tabpfn_model_path):
        from tabpfn.model_loading import load_fitted_tabpfn_model

        logging.info(
            f"Loading TabPFN model from .tabpfn_fit file with device: {device}"
        )
        model = load_fitted_tabpfn_model(tabpfn_model_path, device=device)
    else:
        # Load other models (XGBoost, RandomForestTabPFN saved as pkl, etc.)
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        # Check if model was trained on a different device
        if model_type and "TabPFN" in model_type:
            saved_device = None
            if os.path.exists(model_config_path):
                with open(model_config_path, "r") as f:
                    saved_config = json.load(f)
                    saved_device = saved_config.get("device", None)
            if saved_device and saved_device != device:
                logging.warning(
                    f"Model was trained on {saved_device}, but current device is {device}. "
                    f"Model will use {saved_device} if available, otherwise {device}."
                )
            # RandomForestTabPFNRegressor models should already have device set from training
            # The internal TabPFN models will use the device they were trained on
            if hasattr(model, "device"):
                logging.info(f"Model device attribute: {model.device}")

    # Check model type and prepare input accordingly
    # TabPFN models work with DataFrames, sklearn models work with numpy arrays
    if model_type is None:
        model_type = type(model).__name__
    if "TabPFN" in model_type:
        # TabPFN models expect pandas DataFrames
        X_eval_input = pd.DataFrame(
            X_eval_scaled, columns=X_eval.columns, index=X_eval.index
        )
    else:
        # Other models (XGBoost, sklearn, etc.) work with numpy arrays
        X_eval_input = X_eval_scaled

    # predict
    logging.info(f"Making predictions on {device}...")
    y_hat = model.predict(X_eval_input)
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
