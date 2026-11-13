"""
BatteryMind FastAPI Application.

Production REST API for real-time battery health monitoring, Remaining Useful Life (RUL)
prognostics, State of Health (SOH) tracking, anomaly detection, and explainable AI (SHAP).
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set matplotlib configuration dir to avoid cache permission issues in containerized/sandboxed runs
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
import shap


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "battery_cycles.csv"

FEATURE_COLUMNS = [
    "Cycle_Count",
    "Ambient_Temp_C",
    "Avg_Voltage_V",
    "Avg_Current_A",
    "Max_Temp_C",
    "Internal_Resistance_Ohm",
    "Discharge_Capacity_Ah",
]

TARGET_COLUMNS = [
    "Remaining_Useful_Life_RUL",
    "SOH_Percent",
]


# -----------------------------------------------------------------------------
# Global State & Artifact Store
# -----------------------------------------------------------------------------
class ModelStore:
    def __init__(self):
        self.rul_model = None
        self.anomaly_model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.shap_summary: Dict[str, Any] = {}
        self.explainer: Optional[shap.TreeExplainer] = None
        self.battery_df: Optional[pd.DataFrame] = None
        self.precomputed_batteries: Dict[str, Any] = {}

    def load_all(self):
        """Loads all persisted machine learning models, scalers, metadata, and dataset."""
        print(f"Loading BatteryMind artifacts from: {ARTIFACTS_DIR}")
        
        rul_path = ARTIFACTS_DIR / "rul_model.joblib"
        anomaly_path = ARTIFACTS_DIR / "anomaly_model.joblib"
        scaler_path = ARTIFACTS_DIR / "scaler.joblib"
        metadata_path = ARTIFACTS_DIR / "metadata.json"
        shap_path = ARTIFACTS_DIR / "shap_summary.json"

        if not rul_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {rul_path}. Run train_models.py first.")

        self.rul_model = joblib.load(rul_path)
        self.anomaly_model = joblib.load(anomaly_path)
        self.scaler = joblib.load(scaler_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(shap_path, "r", encoding="utf-8") as f:
            self.shap_summary = json.load(f)

        # Initialize fast TreeExplainer for real-time per-prediction attributions
        self.explainer = shap.TreeExplainer(self.rul_model)

        # Load cycle telemetry dataset
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            self.battery_df = df
            self._precompute_battery_curves(df)
            print(f"Loaded telemetry dataset ({len(df)} rows across {df['Battery_ID'].nunique()} cells)")
        else:
            print(f"Warning: Telemetry dataset not found at {DATA_PATH}")

    def _precompute_battery_curves(self, df: pd.DataFrame):
        """Pre-computes predicted vs actual curves and anomaly classifications for fast retrieval."""
        all_features = df[FEATURE_COLUMNS]
        preds = self.rul_model.predict(all_features)
        iso_preds = self.anomaly_model.predict(all_features)
        iso_scores = self.anomaly_model.decision_function(all_features)

        df_enriched = df.copy()
        df_enriched["Predicted_RUL"] = np.round(np.clip(preds[:, 0], 0, None), 2)
        df_enriched["Predicted_SOH"] = np.round(preds[:, 1], 2)
        df_enriched["Is_Anomaly"] = (iso_preds == -1)
        df_enriched["Anomaly_Score"] = np.round(iso_scores, 4)

        for battery_id, group in df_enriched.groupby("Battery_ID"):
            cycles_data = []
            anomalous_cycles = []
            
            for _, row in group.iterrows():
                cyc = int(row["Cycle_Count"])
                is_anom = bool(row["Is_Anomaly"])
                if is_anom:
                    anomalous_cycles.append(cyc)

                cycles_data.append({
                    "cycle_count": cyc,
                    "ambient_temp_c": float(row["Ambient_Temp_C"]),
                    "avg_voltage_v": float(row["Avg_Voltage_V"]),
                    "avg_current_a": float(row["Avg_Current_A"]),
                    "max_temp_c": float(row["Max_Temp_C"]),
                    "internal_resistance_ohm": float(row["Internal_Resistance_Ohm"]),
                    "discharge_capacity_ah": float(row["Discharge_Capacity_Ah"]),
                    "actual_rul": float(row["Remaining_Useful_Life_RUL"]),
                    "predicted_rul": float(row["Predicted_RUL"]),
                    "actual_soh": float(row["SOH_Percent"]),
                    "predicted_soh": float(row["Predicted_SOH"]),
                    "is_anomaly": is_anom,
                    "anomaly_score": float(row["Anomaly_Score"]),
                })

            self.precomputed_batteries[battery_id] = {
                "battery_id": battery_id,
                "total_cycles": len(group),
                "anomaly_count": len(anomalous_cycles),
                "anomalous_cycles": anomalous_cycles,
                "summary": {
                    "initial_capacity_ah": float(group["Discharge_Capacity_Ah"].iloc[0]),
                    "final_capacity_ah": float(group["Discharge_Capacity_Ah"].iloc[-1]),
                    "initial_soh_percent": float(group["SOH_Percent"].iloc[0]),
                    "final_soh_percent": float(group["SOH_Percent"].iloc[-1]),
                    "peak_temp_c": float(group["Max_Temp_C"].max()),
                    "max_resistance_ohm": float(group["Internal_Resistance_Ohm"].max()),
                    "nominal_rated_capacity_ah": 2.0,
                },
                "cycles": cycles_data,
            }


store = ModelStore()


# -----------------------------------------------------------------------------
# Lifespan Management
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models and dataset
    store.load_all()
    yield
    # Shutdown: Clean up if needed
    pass


# -----------------------------------------------------------------------------
# FastAPI Application & Middleware
# -----------------------------------------------------------------------------
app = FastAPI(
    title="BatteryMind API",
    description="Explainable Li-ion Battery Health, RUL Prognostics & Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class TelemetryInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Cycle_Count: int = Field(
        ...,
        ge=1,
        description="Cycle count index (1, 2, ...)",
        alias="cycle_count",
    )
    Ambient_Temp_C: float = Field(
        ...,
        description="Ambient testing chamber temperature in °C",
        alias="ambient_temp_c",
    )
    Avg_Voltage_V: float = Field(
        ...,
        ge=2.0,
        le=5.0,
        description="Mean discharge plateau terminal voltage in Volts",
        alias="avg_voltage_v",
    )
    Avg_Current_A: float = Field(
        ...,
        ge=0.0,
        description="Discharge current in Amperes",
        alias="avg_current_a",
    )
    Max_Temp_C: float = Field(
        ...,
        description="Maximum recorded cell surface temperature during discharge in °C",
        alias="max_temp_c",
    )
    Internal_Resistance_Ohm: float = Field(
        ...,
        gt=0.0,
        description="Internal resistance / impedance in Ohms",
        alias="internal_resistance_ohm",
    )
    Discharge_Capacity_Ah: float = Field(
        ...,
        gt=0.0,
        description="Measured discharge capacity in Ampere-hours",
        alias="discharge_capacity_ah",
    )


class PredictionResponse(BaseModel):
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    predicted_soh_percent: float = Field(..., description="Predicted State of Health percentage (%)")
    is_anomaly: bool = Field(..., description="True if telemetry pattern indicates abnormal behavior")
    anomaly_score: float = Field(..., description="Isolation Forest anomaly decision score (lower = more anomalous)")
    base_rul: float = Field(..., description="Expected baseline RUL from training distribution")
    shap_contributions: Dict[str, float] = Field(
        ...,
        description="Additive SHAP contribution per feature explaining the RUL prediction",
    )
    prediction_summary: str = Field(..., description="Natural language physical explanation of the prediction")


class HealthResponse(BaseModel):
    status: str
    version: str
    model_architecture: str
    anomaly_detector: str
    explainer: str
    train_cells: List[str]
    test_cell: str
    metrics: Dict[str, Any]
    anomaly_detection: Dict[str, Any]


class BatteryListResponse(BaseModel):
    batteries: List[str]


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "service": "BatteryMind Prognostics API",
        "docs_url": "/docs",
        "health_url": "/health",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def get_health():
    """Returns API health status, loaded model specifications, and validation metrics."""
    if store.rul_model is None or not store.metadata:
        # Fallback to load on the fly if lifespan hasn't triggered
        store.load_all()

    meta = store.metadata
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_architecture=meta.get("model_architecture", "XGBoost Multi-Target Regressor"),
        anomaly_detector=meta.get("anomaly_detector", "Isolation Forest"),
        explainer=meta.get("explainer", "SHAP TreeExplainer"),
        train_cells=meta.get("train_cells", []),
        test_cell=meta.get("test_cell", ""),
        metrics=meta.get("metrics", {}),
        anomaly_detection=meta.get("anomaly_detection", {}),
    )


@app.get("/batteries", response_model=BatteryListResponse, tags=["Batteries"])
def list_batteries():
    """Lists all available battery cell identifiers in the dataset."""
    if not store.precomputed_batteries:
        store.load_all()
    return BatteryListResponse(batteries=list(store.precomputed_batteries.keys()))


@app.get("/batteries/{battery_id}", tags=["Batteries"])
def get_battery_telemetry(battery_id: str):
    """
    Returns complete historical cycle telemetry, anomaly flags, and
    predicted vs. actual RUL/SOH curves for a cell (e.g. B0005, B0006, B0007, B0018).
    """
    if not store.precomputed_batteries:
        store.load_all()

    battery_key = battery_id.upper()
    if battery_key not in store.precomputed_batteries:
        available = list(store.precomputed_batteries.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Battery cell '{battery_id}' not found. Available cells: {available}",
        )

    return store.precomputed_batteries[battery_key]


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_telemetry(payload: TelemetryInput):
    """
    Accepts live battery cycle telemetry features, predicts RUL & SOH %,
    detects anomalies, and computes real-time SHAP feature attributions.
    """
    if store.rul_model is None or store.explainer is None:
        store.load_all()

    # Create input DataFrame matching canonical feature order
    feature_dict = {
        "Cycle_Count": payload.Cycle_Count,
        "Ambient_Temp_C": payload.Ambient_Temp_C,
        "Avg_Voltage_V": payload.Avg_Voltage_V,
        "Avg_Current_A": payload.Avg_Current_A,
        "Max_Temp_C": payload.Max_Temp_C,
        "Internal_Resistance_Ohm": payload.Internal_Resistance_Ohm,
        "Discharge_Capacity_Ah": payload.Discharge_Capacity_Ah,
    }
    input_df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

    # 1. Predict RUL and SOH
    raw_preds = store.rul_model.predict(input_df)
    pred_rul = float(np.round(np.clip(raw_preds[0, 0], 0.0, None), 2))
    pred_soh = float(np.round(raw_preds[0, 1], 2))

    # 2. Anomaly Detection
    iso_pred = store.anomaly_model.predict(input_df)[0]
    iso_score = float(np.round(store.anomaly_model.decision_function(input_df)[0], 4))
    is_anomaly = bool(iso_pred == -1)

    # 3. Real-Time SHAP Attributions for RUL (Target 0)
    shap_vals = store.explainer.shap_values(input_df)  # shape: (1, 7, 2)
    shap_rul = shap_vals[0, :, 0]
    base_rul = float(np.round(store.explainer.expected_value[0], 2))

    shap_dict = {
        feat: float(np.round(shap_rul[idx], 4))
        for idx, feat in enumerate(FEATURE_COLUMNS)
    }

    # Top drivers explanation
    top_negative_feat = min(shap_dict.items(), key=lambda x: x[1])
    top_positive_feat = max(shap_dict.items(), key=lambda x: x[1])

    summary_parts = [
        f"Cell predicted at cycle {payload.Cycle_Count} with RUL: {pred_rul:.1f} cycles and SOH: {pred_soh:.1f}%."
    ]
    if is_anomaly:
        summary_parts.append(f"⚠️ ANOMALY DETECTED (Score: {iso_score:.4f}): telemetry departs significantly from baseline normal cycling.")
    else:
        summary_parts.append("Nominal operating conditions detected.")

    summary_parts.append(
        f"Top positive driver extending RUL: {top_positive_feat[0]} (+{top_positive_feat[1]:.2f} cycles). "
        f"Top factor reducing RUL: {top_negative_feat[0]} ({top_negative_feat[1]:.2f} cycles)."
    )

    return PredictionResponse(
        predicted_rul=pred_rul,
        predicted_soh_percent=pred_soh,
        is_anomaly=is_anomaly,
        anomaly_score=iso_score,
        base_rul=base_rul,
        shap_contributions=shap_dict,
        prediction_summary=" ".join(summary_parts),
    )


@app.get("/explain/global", tags=["Explainability"])
def get_global_explanations():
    """
    Returns global SHAP feature importance rankings and lifecycle stage explanations
    describing the primary physical causes of Li-ion battery degradation.
    """
    if not store.shap_summary:
        store.load_all()

    return store.shap_summary
