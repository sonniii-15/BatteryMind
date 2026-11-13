"""
NASA Li-ion Battery Telemetry Generator & Pipeline.

Generates realistic synthetic NASA-style Li-ion battery degradation telemetry
across multiple cells modeled after the NASA Ames Prognostics Center of Excellence
(PCoE) battery aging datasets (e.g., cells B0005, B0006, B0007, B0018).

Features simulated across charge/discharge cycles:
- Battery_ID: Unique identifier for each battery cell
- Cycle_Count: Progressive discharge cycle index (1, 2, ...)
- Ambient_Temp_C: Controlled testing chamber temperature (°C)
- Avg_Voltage_V: Mean discharge plateau voltage (V) reflecting internal resistance growth
- Avg_Current_A: Controlled discharge current (A) with transducer noise
- Max_Temp_C: Maximum cell surface temperature (°C) during discharge (Joule heating)
- Internal_Resistance_Ohm: Battery impedance / internal resistance (Ohm)
- Discharge_Capacity_Ah: Measured discharge capacity (Ah) with nonlinear aging & rest regeneration
- SOH_Percent: State of Health (% of nominal 2.0 Ah rated capacity)
- Remaining_Useful_Life_RUL: Cycles remaining until reaching End-of-Life (EOL) threshold
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Default cell configurations modeled after the NASA Ames Li-ion 18650 dataset
DEFAULT_CELL_CONFIGS: Dict[str, dict] = {
    "B0005": {
        "num_cycles": 168,
        "init_cap": 1.856,
        "init_r": 0.075,
        "fade_rate": 0.0023,
        "knee_factor": 0.18,
        "knee_scale": 0.70,
        "rest_cycles": [35, 72, 110, 145],
        "ambient_base": 24.0,
    },
    "B0006": {
        "num_cycles": 168,
        "init_cap": 2.035,
        "init_r": 0.082,
        "fade_rate": 0.0031,
        "knee_factor": 0.25,
        "knee_scale": 0.65,
        "rest_cycles": [30, 68, 102, 138],
        "ambient_base": 24.0,
    },
    "B0007": {
        "num_cycles": 168,
        "init_cap": 1.891,
        "init_r": 0.072,
        "fade_rate": 0.0019,
        "knee_factor": 0.15,
        "knee_scale": 0.75,
        "rest_cycles": [38, 76, 115, 150],
        "ambient_base": 24.0,
    },
    "B0018": {
        "num_cycles": 132,
        "init_cap": 1.855,
        "init_r": 0.078,
        "fade_rate": 0.0030,
        "knee_factor": 0.22,
        "knee_scale": 0.68,
        "rest_cycles": [28, 60, 90],
        "ambient_base": 24.0,
    },
}


def simulate_cell_telemetry(
    cell_id: str,
    config: dict,
    rng: np.random.Generator,
    rated_capacity: float = 2.0,
    eol_threshold: float = 1.40,
    anomaly_rate: float = 0.025,
) -> pd.DataFrame:
    """
    Simulates degradation telemetry for an individual NASA Li-ion battery cell.

    Parameters:
    -----------
    cell_id : str
        Battery identifier (e.g., 'B0005').
    config : dict
        Physical aging and cycling parameters for the cell.
    rng : np.random.Generator
        Numpy random generator instance for reproducibility.
    rated_capacity : float
        Nominal rated capacity in Ah (default: 2.0 Ah for 18650 cells).
    eol_threshold : float
        End-of-Life capacity threshold in Ah (default: 1.40 Ah, 70% of rated).
    anomaly_rate : float
        Fraction of cycles exhibiting occasional thermal or voltage anomalies.

    Returns:
    --------
    pd.DataFrame
        DataFrame with 10 telemetry columns for the given cell.
    """
    n = config["num_cycles"]
    cycles = np.arange(1, n + 1)

    # 1. Discharge Capacity (Ah) with nonlinear aging & rest-period recovery
    linear_fade = config["fade_rate"] * cycles
    knee_scale = config["knee_scale"]
    knee_norm = (np.exp(cycles / (n * knee_scale)) - 1.0) / (np.exp(1.0 / knee_scale) - 1.0)
    exp_fade = config["knee_factor"] * knee_norm

    # Capacity regeneration jumps after experimental resting intervals
    regeneration = np.zeros(n)
    for rc in config.get("rest_cycles", []):
        if rc <= n:
            decay_window = np.maximum(0, 1.0 - (cycles - rc) / 14.0)
            regeneration += (cycles >= rc) * 0.026 * decay_window

    cap_noise = rng.normal(0, 0.0035, size=n)
    capacity = config["init_cap"] - linear_fade - exp_fade + regeneration + cap_noise
    capacity = np.clip(capacity, 1.15, 2.15)

    # 2. State of Health (SOH %) relative to rated nominal capacity (2.0 Ah)
    soh = np.round((capacity / rated_capacity) * 100.0, 2)

    # 3. Remaining Useful Life (RUL in cycles)
    eol_candidates = np.where(capacity <= eol_threshold)[0]
    eol_cycle = int(cycles[eol_candidates[0]]) if len(eol_candidates) > 0 else n
    rul = np.maximum(0, eol_cycle - cycles)

    # 4. Internal Resistance (Ohm)
    # R increases as SEI film grows, electrolyte decomposes, and active lithium is lost
    r_base = (
        config["init_r"]
        + 0.00045 * cycles
        + 0.04 * (cycles / n) ** 2
        + 0.045 * (1.0 - capacity / config["init_cap"])
    )
    # Slight impedance recovery following rest periods
    r_rest = np.zeros(n)
    for rc in config.get("rest_cycles", []):
        if rc <= n:
            decay_r = np.maximum(0, 1.0 - (cycles - rc) / 10.0)
            r_rest -= (cycles >= rc) * 0.004 * decay_r
    r_noise = rng.normal(0, 0.0012, size=n)
    resistance = np.round(np.clip(r_base + r_rest + r_noise, 0.05, 0.35), 4)

    # 5. Ambient Temperature (°C)
    ambient_noise = rng.normal(0, 0.18, size=n)
    ambient_temp = np.round(config.get("ambient_base", 24.0) + ambient_noise, 2)

    # 6. Discharge Current (A) - CC regulated discharge (nominally 2.0 A)
    current_noise = rng.normal(0, 0.0035, size=n)
    current = np.round(2.000 + current_noise, 4)

    # 7. Average Discharge Voltage (V)
    # Mean plateau drops due to IR drop (Ohm's law) and polarization loss
    volt_drop_from_ir = 0.55 * (resistance - config["init_r"])
    base_voltage = 3.82 - 0.0012 * cycles - volt_drop_from_ir - 0.035 * (cycles / n) ** 2
    volt_noise = rng.normal(0, 0.005, size=n)
    voltage = base_voltage + volt_noise

    # 8. Maximum Temperature (°C) - Surface thermocouple during discharge
    # Driven by ambient temperature + Joule heating (I^2 * R) + entropic polarization
    joule_temp_rise = 18.0 * (resistance - config["init_r"]) + 7.5 * (cycles / n)
    base_max_temp = ambient_temp + 8.5 + joule_temp_rise
    temp_noise = rng.normal(0, 0.30, size=n)
    max_temp = base_max_temp + temp_noise

    # 9. Occasional Thermal and Voltage Anomalies
    num_anomalies = max(2, int(anomaly_rate * n))
    thermal_anomaly_cycles = rng.choice(cycles, size=num_anomalies, replace=False)
    voltage_anomaly_cycles = rng.choice(cycles, size=num_anomalies, replace=False)

    for cyc in thermal_anomaly_cycles:
        idx = cyc - 1
        # Thermal spike: +4.5°C to +8.5°C (micro-short or cooling restriction)
        max_temp[idx] += rng.uniform(4.5, 8.5)

    for cyc in voltage_anomaly_cycles:
        idx = cyc - 1
        # Voltage sag: -0.18V to -0.32V (momentary contact resistance or transducer glitch)
        voltage[idx] -= rng.uniform(0.18, 0.32)

    # Final rounding and physical clipping
    voltage = np.round(np.clip(voltage, 3.10, 3.90), 4)
    max_temp = np.round(max_temp, 2)
    capacity = np.round(capacity, 4)

    return pd.DataFrame({
        "Battery_ID": [cell_id] * n,
        "Cycle_Count": cycles,
        "Ambient_Temp_C": ambient_temp,
        "Avg_Voltage_V": voltage,
        "Avg_Current_A": current,
        "Max_Temp_C": max_temp,
        "Internal_Resistance_Ohm": resistance,
        "Discharge_Capacity_Ah": capacity,
        "SOH_Percent": soh,
        "Remaining_Useful_Life_RUL": rul,
    })


def generate_nasa_dataset(
    cells: Optional[List[str]] = None,
    seed: int = 42,
    rated_capacity: float = 2.0,
    eol_threshold: float = 1.40,
    anomaly_rate: float = 0.025,
) -> pd.DataFrame:
    """
    Generates a combined multi-cell NASA battery degradation dataset.

    Parameters:
    -----------
    cells : Optional[List[str]]
        List of cell IDs to include (defaults to all configured cells).
    seed : int
        Master random seed for reproducibility.
    rated_capacity : float
        Nominal rated capacity (Ah).
    eol_threshold : float
        End-of-Life capacity threshold (Ah).
    anomaly_rate : float
        Rate of occasional thermal/voltage anomalies per cell.

    Returns:
    --------
    pd.DataFrame
        Consolidated DataFrame with telemetry for all specified cells.
    """
    if cells is None:
        cells = list(DEFAULT_CELL_CONFIGS.keys())

    rng = np.random.default_rng(seed)
    cell_dfs = []

    for cell_id in cells:
        if cell_id not in DEFAULT_CELL_CONFIGS:
            raise ValueError(
                f"Unknown cell '{cell_id}'. Available: {list(DEFAULT_CELL_CONFIGS.keys())}"
            )
        cfg = DEFAULT_CELL_CONFIGS[cell_id]
        cell_df = simulate_cell_telemetry(
            cell_id=cell_id,
            config=cfg,
            rng=rng,
            rated_capacity=rated_capacity,
            eol_threshold=eol_threshold,
            anomaly_rate=anomaly_rate,
        )
        cell_dfs.append(cell_df)

    return pd.concat(cell_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic NASA Li-ion battery degradation telemetry across multiple cells."
    )
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "processed" / "battery_cycles.csv"

    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path to output CSV (default: {default_output})",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELL_CONFIGS.keys()),
        help=f"Battery cell IDs to simulate (default: {list(DEFAULT_CELL_CONFIGS.keys())})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--rated-capacity",
        type=float,
        default=2.0,
        help="Nominal rated capacity in Ah (default: 2.0)",
    )
    parser.add_argument(
        "--eol-threshold",
        type=float,
        default=1.40,
        help="End-of-Life capacity threshold in Ah (default: 1.40)",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.025,
        help="Frequency of thermal and voltage anomalies (default: 0.025)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating NASA Li-ion battery telemetry across cells: {args.cells} (seed={args.seed})...")
    df = generate_nasa_dataset(
        cells=args.cells,
        seed=args.seed,
        rated_capacity=args.rated_capacity,
        eol_threshold=args.eol_threshold,
        anomaly_rate=args.anomaly_rate,
    )

    df.to_csv(args.output, index=False)
    print(f"Successfully saved clean dataset to: {args.output}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(f"\nDataset Shape: {df.shape}")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nSummary Statistics per Cell:")
    summary = df.groupby("Battery_ID").agg(
        total_cycles=("Cycle_Count", "max"),
        min_capacity=("Discharge_Capacity_Ah", "min"),
        max_capacity=("Discharge_Capacity_Ah", "max"),
        max_temp=("Max_Temp_C", "max"),
        min_voltage=("Avg_Voltage_V", "min"),
        min_soh=("SOH_Percent", "min"),
        max_soh=("SOH_Percent", "max"),
    )
    print(summary)


if __name__ == "__main__":
    main()
