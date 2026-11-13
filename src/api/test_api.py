"""
Test suite for BatteryMind FastAPI Backend.
Verifies all endpoints using FastAPI TestClient:
- GET /health
- GET /batteries
- GET /batteries/{battery_id} (valid and invalid)
- POST /predict (nominal, anomalous, and validation error)
- GET /explain/global
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app, store


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "metrics" in data
    assert "Remaining_Useful_Life_RUL" in data["metrics"]
    assert "SOH_Percent" in data["metrics"]
    assert data["test_cell"] == "B0018"
    assert "B0005" in data["train_cells"]


def test_list_batteries():
    response = client.get("/batteries")
    assert response.status_code == 200
    data = response.json()
    assert "batteries" in data
    assert "B0005" in data["batteries"]
    assert "B0018" in data["batteries"]


def test_get_battery_telemetry_valid():
    response = client.get("/batteries/B0005")
    assert response.status_code == 200
    data = response.json()
    assert data["battery_id"] == "B0005"
    assert data["total_cycles"] == 168
    assert len(data["cycles"]) == 168
    first_cycle = data["cycles"][0]
    assert "cycle_count" in first_cycle
    assert "actual_rul" in first_cycle
    assert "predicted_rul" in first_cycle
    assert "actual_soh" in first_cycle
    assert "predicted_soh" in first_cycle
    assert "is_anomaly" in first_cycle
    assert "anomaly_score" in first_cycle


def test_get_battery_telemetry_unseen_cell():
    response = client.get("/batteries/B0018")
    assert response.status_code == 200
    data = response.json()
    assert data["battery_id"] == "B0018"
    assert data["total_cycles"] == 132
    assert "anomalous_cycles" in data
    assert data["anomaly_count"] >= 1


def test_get_battery_telemetry_not_found():
    response = client.get("/batteries/UNKNOWN_CELL_999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_predict_nominal_telemetry():
    payload = {
        "Cycle_Count": 40,
        "Ambient_Temp_C": 24.1,
        "Avg_Voltage_V": 3.78,
        "Avg_Current_A": 2.00,
        "Max_Temp_C": 33.5,
        "Internal_Resistance_Ohm": 0.088,
        "Discharge_Capacity_Ah": 1.76,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rul" in data
    assert "predicted_soh_percent" in data
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "shap_contributions" in data
    assert data["predicted_rul"] > 0
    assert 60 <= data["predicted_soh_percent"] <= 100
    assert len(data["shap_contributions"]) == 7


def test_predict_anomalous_telemetry():
    # Extreme overheating thermal excursion
    payload = {
        "Cycle_Count": 110,
        "Ambient_Temp_C": 24.0,
        "Avg_Voltage_V": 3.25,
        "Avg_Current_A": 2.01,
        "Max_Temp_C": 52.0,  # runaway heating anomaly
        "Internal_Resistance_Ohm": 0.22,
        "Discharge_Capacity_Ah": 1.30,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is True
    assert data["anomaly_score"] < 0
    assert "ANOMALY DETECTED" in data["prediction_summary"]


def test_predict_validation_error():
    # Missing required field
    payload = {
        "Cycle_Count": -5,  # ge=1 violation
        "Ambient_Temp_C": 24.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_global_explanations():
    response = client.get("/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert "global_importance_ranking" in data
    assert len(data["global_importance_ranking"]) == 7
    assert "physics_interpretation" in data
    assert "sample_cycle_explanations" in data


if __name__ == "__main__":
    print("Running test suite...")
    test_root_endpoint()
    test_health_endpoint()
    test_list_batteries()
    test_get_battery_telemetry_valid()
    test_get_battery_telemetry_unseen_cell()
    test_get_battery_telemetry_not_found()
    test_predict_nominal_telemetry()
    test_predict_anomalous_telemetry()
    test_predict_validation_error()
    test_global_explanations()
    print("All 10 tests passed successfully!")
