"""
BatteryMind ML Training & Explainable AI Pipeline.

Trains core machine learning models for battery prognostics and health monitoring:
1. XGBoost Multi-Target Regressor: Predicts Remaining Useful Life (RUL) and State of Health (SOH %).
2. Isolation Forest: Unsupervised anomaly detector for thermal runaway and voltage collapse events.
3. SHAP TreeExplainer: Computes global and per-cycle local feature importances explaining degradation.

Implements strict group-aware train/test splitting by Battery_ID to prevent data leakage across cycles.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Set matplotlib configuration dir before imports to prevent cache permission issues
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap


# Canonical feature definitions
FEATURE_COLUMNS: List[str] = [
    "Cycle_Count",
    "Ambient_Temp_C",
    "Avg_Voltage_V",
    "Avg_Current_A",
    "Max_Temp_C",
    "Internal_Resistance_Ohm",
    "Discharge_Capacity_Ah",
]

TARGET_COLUMNS: List[str] = [
    "Remaining_Useful_Life_RUL",
    "SOH_Percent",
]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Loads processed battery degradation telemetry CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from: {csv_path} (Shape: {df.shape})")
    return df


def split_data_group_aware(
    df: pd.DataFrame,
    test_cell: str = "B0018",
    features: List[str] = FEATURE_COLUMNS,
    targets: List[str] = TARGET_COLUMNS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Performs group-aware train/test splitting by Battery_ID.
    Guarantees zero data leakage across cycles of the same cell.
    """
    all_cells = df["Battery_ID"].unique().tolist()
    if test_cell not in all_cells:
        raise ValueError(f"Test cell '{test_cell}' not found in dataset cells: {all_cells}")

    train_cells = [c for c in all_cells if c != test_cell]
    train_mask = df["Battery_ID"].isin(train_cells)
    test_mask = df["Battery_ID"] == test_cell

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    X_train = train_df[features].copy()
    y_train = train_df[targets].copy()
    X_test = test_df[features].copy()
    y_test = test_df[targets].copy()

    print(f"\n--- Group-Aware Train/Test Split ---")
    print(f"Train Cells: {train_cells} ({len(train_df)} cycle samples)")
    print(f"Test Cell  : ['{test_cell}'] ({len(test_df)} cycle samples, zero data leakage)")
    print(f"Features   : {features}")
    print(f"Targets    : {targets}")

    return X_train, y_train, X_test, y_test, train_cells


def train_rul_and_soh_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    seed: int = 42,
) -> Tuple[xgb.XGBRegressor, Dict[str, dict]]:
    """
    Trains multi-target XGBoost Regressor to predict RUL and SOH.
    Evaluates test RMSE, MAE, and R² scores on unseen cell.
    """
    print(f"\n--- Training XGBoost Multi-Target Regressor ---")
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # Evaluate predictions on unseen cell
    preds = model.predict(X_test)
    rul_pred = preds[:, 0]
    soh_pred = preds[:, 1]

    rul_true = y_test["Remaining_Useful_Life_RUL"].values
    soh_true = y_test["SOH_Percent"].values

    metrics = {
        "Remaining_Useful_Life_RUL": {
            "rmse": float(np.sqrt(mean_squared_error(rul_true, rul_pred))),
            "mae": float(mean_absolute_error(rul_true, rul_pred)),
            "r2": float(r2_score(rul_true, rul_pred)),
        },
        "SOH_Percent": {
            "rmse": float(np.sqrt(mean_squared_error(soh_true, soh_pred))),
            "mae": float(mean_absolute_error(soh_true, soh_pred)),
            "r2": float(r2_score(soh_true, soh_pred)),
        },
    }

    print("\n--- Test Metrics on Unseen Cell ---")
    print(
        f"RUL Prediction -> RMSE: {metrics['Remaining_Useful_Life_RUL']['rmse']:.2f} cycles | "
        f"MAE: {metrics['Remaining_Useful_Life_RUL']['mae']:.2f} cycles | "
        f"R²: {metrics['Remaining_Useful_Life_RUL']['r2']:.4f}"
    )
    print(
        f"SOH Prediction -> RMSE: {metrics['SOH_Percent']['rmse']:.2f}% | "
        f"MAE: {metrics['SOH_Percent']['mae']:.2f}% | "
        f"R²: {metrics['SOH_Percent']['r2']:.4f}"
    )

    return model, metrics


def train_anomaly_detector(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    contamination: float = 0.03,
    seed: int = 42,
) -> Tuple[IsolationForest, Dict[str, any]]:
    """
    Trains an Isolation Forest anomaly detector to identify thermal excursions and voltage drops.
    """
    print(f"\n--- Training Isolation Forest Anomaly Detector ---")
    iso = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=100,
    )
    iso.fit(X_train)

    # Predict anomalies on unseen cell (-1 = anomaly, 1 = normal)
    preds = iso.predict(X_test)
    anomaly_indices = np.where(preds == -1)[0]
    anomaly_count = int(len(anomaly_indices))

    anomaly_info = {
        "contamination": contamination,
        "test_total_cycles": len(X_test),
        "test_anomalies_detected": anomaly_count,
        "anomalous_cycles": (X_test.iloc[anomaly_indices]["Cycle_Count"].tolist()
                             if "Cycle_Count" in X_test.columns else anomaly_indices.tolist()),
    }

    print(f"Anomaly Detection -> Found {anomaly_count} anomalous cycles in unseen test cell "
          f"({anomaly_info['anomalous_cycles']}) out of {len(X_test)} cycles")

    return iso, anomaly_info


def compute_shap_explanations(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    features: List[str] = FEATURE_COLUMNS,
) -> Tuple[shap.TreeExplainer, np.ndarray, Dict[str, any]]:
    """
    Integrates SHAP TreeExplainer to compute global and per-prediction feature importances.
    Explains the primary physical drivers of battery RUL degradation.
    """
    print(f"\n--- Computing SHAP TreeExplainer Feature Importances ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features, 2)

    # SHAP values for Target 0: Remaining_Useful_Life_RUL
    shap_rul = shap_values[:, :, 0]
    mean_abs_rul = np.mean(np.abs(shap_rul), axis=0)

    # Rank features globally by RUL importance
    importance_ranking = []
    for rank, (feat, val) in enumerate(
        sorted(zip(features, mean_abs_rul), key=lambda x: x[1], reverse=True),
        start=1,
    ):
        importance_ranking.append({
            "rank": rank,
            "feature": feat,
            "mean_abs_shap": round(float(val), 4),
        })

    # Sample cycle local explanations (fresh cycle, mid-life cycle, and near-EOL cycle)
    sample_indices = [
        0,  # Cycle 1 (Fresh cell)
        min(len(X_test) // 2, len(X_test) - 1),  # Mid-life cycle
        len(X_test) - 1,  # Near-EOL cycle
    ]

    local_explanations = {}
    for idx in sample_indices:
        cycle_num = int(X_test.iloc[idx]["Cycle_Count"])
        contributions = {
            feat: round(float(shap_rul[idx, f_idx]), 4)
            for f_idx, feat in enumerate(features)
        }
        # Sort contributions by absolute impact
        sorted_contrib = dict(
            sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        )
        local_explanations[f"cycle_{cycle_num}"] = {
            "cycle_number": cycle_num,
            "base_value_rul": round(float(explainer.expected_value[0]), 2),
            "feature_contributions": sorted_contrib,
        }

    shap_summary = {
        "target_explained": "Remaining_Useful_Life_RUL",
        "global_importance_ranking": importance_ranking,
        "sample_cycle_explanations": local_explanations,
        "physics_interpretation": {
            "Discharge_Capacity_Ah": "Direct measure of retained active lithium; drops strongly reduce RUL.",
            "Internal_Resistance_Ohm": "Growth of SEI passivation layer and electrolyte decay accelerates RUL drop.",
            "Cycle_Count": "Linear base accumulation of wear and thermal aging.",
            "Max_Temp_C": "Elevated operating temperatures indicate higher Joule heating from increased impedance.",
            "Avg_Voltage_V": "Lower average voltage plateau reflects higher internal polarization losses.",
        },
    }

    print("\n--- SHAP Global Feature Importance (Why RUL Drops) ---")
    for item in importance_ranking:
        print(f"  Rank {item['rank']}: {item['feature']:<25} | Mean |SHAP| = {item['mean_abs_shap']:.4f}")

    return explainer, shap_values, shap_summary


def save_artifacts(
    artifacts_dir: Path,
    rul_model: xgb.XGBRegressor,
    anomaly_model: IsolationForest,
    scaler: StandardScaler,
    metadata: Dict[str, any],
    shap_summary: Dict[str, any],
    shap_values: np.ndarray,
) -> None:
    """Saves trained models, scaler, metadata, and SHAP explanations to artifacts directory."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rul_path = artifacts_dir / "rul_model.joblib"
    anomaly_path = artifacts_dir / "anomaly_model.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    metadata_path = artifacts_dir / "metadata.json"
    shap_path = artifacts_dir / "shap_summary.json"
    shap_vals_path = artifacts_dir / "shap_values.joblib"

    joblib.dump(rul_model, rul_path)
    joblib.dump(anomaly_model, anomaly_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(shap_values, shap_vals_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(shap_path, "w", encoding="utf-8") as f:
        json.dump(shap_summary, f, indent=2)

    print(f"\n--- Artifacts Successfully Saved to {artifacts_dir} ---")
    print(f"  • RUL & SOH XGBoost Model : {rul_path}")
    print(f"  • Isolation Forest Model  : {anomaly_path}")
    print(f"  • Feature Scaler          : {scaler_path}")
    print(f"  • Model Metadata JSON     : {metadata_path}")
    print(f"  • SHAP Summary JSON       : {shap_path}")
    print(f"  • Raw SHAP Values         : {shap_vals_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BatteryMind ML Training & Explainable AI (XGBoost, Isolation Forest, SHAP)."
    )
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data = project_root / "data" / "processed" / "battery_cycles.csv"
    default_artifacts = project_root / "src" / "models" / "artifacts"

    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help=f"Path to processed telemetry CSV (default: {default_data})",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=default_artifacts,
        help=f"Directory to store model artifacts (default: {default_artifacts})",
    )
    parser.add_argument(
        "--test-cell",
        type=str,
        default="B0018",
        help="Battery ID to isolate as unseen test cell (default: B0018)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # 1. Load telemetry dataset
    df = load_dataset(args.data)

    # 2. Group-aware train/test split by Battery_ID
    X_train, y_train, X_test, y_test, train_cells = split_data_group_aware(
        df=df,
        test_cell=args.test_cell,
        features=FEATURE_COLUMNS,
        targets=TARGET_COLUMNS,
    )

    # 3. Fit feature scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 4. Train multi-target XGBoost Regressor
    rul_model, test_metrics = train_rul_and_soh_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=args.seed,
    )

    # 5. Train Isolation Forest Anomaly Detector
    anomaly_model, anomaly_info = train_anomaly_detector(
        X_train=X_train,
        X_test=X_test,
        contamination=0.03,
        seed=args.seed,
    )

    # 6. Compute SHAP explainability
    explainer, shap_values, shap_summary = compute_shap_explanations(
        model=rul_model,
        X_test=X_test,
        features=FEATURE_COLUMNS,
    )

    # 7. Compile metadata
    metadata = {
        "model_architecture": "XGBoost Multi-Target Regressor",
        "anomaly_detector": "Isolation Forest",
        "explainer": "SHAP TreeExplainer",
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "train_cells": train_cells,
        "test_cell": args.test_cell,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "metrics": test_metrics,
        "anomaly_detection": anomaly_info,
        "random_seed": args.seed,
    }

    # 8. Save all artifacts
    save_artifacts(
        artifacts_dir=args.artifacts_dir,
        rul_model=rul_model,
        anomaly_model=anomaly_model,
        scaler=scaler,
        metadata=metadata,
        shap_summary=shap_summary,
        shap_values=shap_values,
    )


if __name__ == "__main__":
    main()
