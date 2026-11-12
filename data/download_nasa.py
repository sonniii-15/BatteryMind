"""
NASA Li-ion Battery Telemetry Generator & Pipeline.

Generates realistic sample NASA Li-ion battery cycle telemetry based on
experimental aging datasets (e.g., NASA Ames Battery Aging Dataset B0005).

Features simulated:
- Cycle Count: Progressive discharge cycle index
- Discharge Capacity: Capacity degradation (Ah) with aging and rest-period regeneration
- Voltage: Mean discharge terminal voltage (V) reflecting internal resistance growth
- Current: Controlled discharge current (A) with sensor noise
- Temp: Operating cell temperature (°C) influenced by internal resistance heating
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_nasa_battery_telemetry(num_cycles: int = 168, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic NASA Li-ion battery cycle telemetry modeling cell B0005.

    Parameters:
    -----------
    num_cycles : int
        Number of charge-discharge cycles to simulate (default: 168 cycles).
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: [Cycle Count, Voltage, Current, Temp, Discharge Capacity]
    """
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, num_cycles + 1)

    # 1. Discharge Capacity (Ah)
    # Starts around nominal fresh cell capacity 1.856 Ah, decays towards ~1.30 Ah EOL
    initial_capacity = 1.856
    linear_fade = 0.0028 * cycles
    exp_fade = 0.05 * (np.exp(cycles / 140.0) - 1.0) / (np.exp(168 / 140.0) - 1.0)

    # Capacity regeneration jumps (phenomenon seen in NASA dataset after rest intervals)
    regeneration = np.zeros(num_cycles)
    rest_cycles = [35, 72, 110, 145]
    for rc in rest_cycles:
        if rc <= num_cycles:
            # Rebound boost that slowly fades
            decay_window = np.maximum(0, 1.0 - (cycles - rc) / 18.0)
            regeneration += (cycles >= rc) * 0.025 * decay_window

    noise = rng.normal(0, 0.004, size=num_cycles)
    discharge_capacity = initial_capacity - linear_fade - exp_fade + regeneration + noise
    discharge_capacity = np.clip(discharge_capacity, 1.25, 1.90)

    # 2. Voltage (V) - Mean discharge plateau voltage
    # Fresh cell has lower internal resistance (~3.85V mean), rises in impedance with aging (~3.52V)
    base_voltage = 3.82 - 0.0018 * cycles - 0.04 * (cycles / num_cycles) ** 2
    voltage_noise = rng.normal(0, 0.006, size=num_cycles)
    voltage = np.round(base_voltage + voltage_noise, 4)

    # 3. Current (A) - Regulated constant discharge current
    # Nominal discharge is 2.0A with standard transducer sensor noise
    current_noise = rng.normal(0, 0.004, size=num_cycles)
    current = np.round(2.000 + current_noise, 4)

    # 4. Temperature (°C) - Cell surface temperature during discharge
    # Base ambient is 24°C; internal resistance increases Joule heating (I^2 * R)
    base_temp = 24.2 + 6.5 * (1.0 - (discharge_capacity / initial_capacity)) + 3.0 * (cycles / num_cycles)
    temp_noise = rng.normal(0, 0.35, size=num_cycles)
    temp = np.round(base_temp + temp_noise, 2)

    df = pd.DataFrame({
        "Cycle Count": cycles,
        "Voltage": voltage,
        "Current": current,
        "Temp": temp,
        "Discharge Capacity": np.round(discharge_capacity, 4),
    })

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample NASA Li-ion battery cycle telemetry."
    )
    # Default output relative to project root
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "processed" / "battery_cycles.csv"

    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path to output CSV (default: {default_output})",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=168,
        help="Number of cycles to simulate (default: 168, matching NASA B0005)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating NASA Li-ion battery telemetry for {args.cycles} cycles...")
    df = generate_nasa_battery_telemetry(num_cycles=args.cycles, seed=args.seed)

    df.to_csv(args.output, index=False)
    print(f"Successfully saved clean dataset to: {args.output}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    print("\nSample Preview:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
