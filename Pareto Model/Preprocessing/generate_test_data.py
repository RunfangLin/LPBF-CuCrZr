"""Generate the full-space LPBF CuCrZr candidate dataset for EHVI.

By default, the generated candidate file is written directly to:

    Pareto Model/Data/generate_test.csv

This is the location read by Pareto_Model.py.
"""

from itertools import product
from pathlib import Path

import pandas as pd


PREPROCESSING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PREPROCESSING_DIR.parent
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "Data" / "generate_test.csv"

POWER_VALUES = range(350, 601, 10)  # W
SPEED_VALUES = range(200, 801, 10)  # mm/s
AGEING_TEMPERATURES_C = range(400, 621, 5)
AGEING_TIMES_MIN = (0, 60, 90, 120)

VED_MIN = 120  # J/mm^3
VED_MAX = 350  # J/mm^3

COLUMNS = ("power", "speed", "lt", "hd", "VED", "T", "t")

# Preserve the historical dataset's integer-Kelvin convention.
# This reproduces the original candidate space exactly.
LEGACY_KELVIN_OFFSET = 273


def generate_test_data(
    limit_ved_range=1,
    output_file=None,
    *,
    lt_range=(0.03,),
    hd_range=(0.09,),
):
    """Generate and save the discrete EHVI candidate space.

    Parameters
    ----------
    limit_ved_range : bool or int
        If True, retain candidates with raw VED between 120 and
        350 J/mm^3. If False, do not apply the VED filter.

    output_file : str, pathlib.Path, or None
        Optional output path. When omitted, the CSV is written to
        Pareto Model/Data/generate_test.csv.

    lt_range : iterable of positive numbers
        Layer thickness values in mm. The default is 0.03 mm.

    hd_range : iterable of positive numbers
        Hatch-spacing values in mm. The default is 0.09 mm.

    Notes
    -----
    Power ranges from 350 to 600 W in 10 W increments.

    Scan speed ranges from 200 to 800 mm/s in 10 mm/s increments.

    Each retained build condition has:

    - one as-built state: t = 0 min and T = 273 K;
    - aged states from 400 to 620 deg C in 5 deg C increments;
    - ageing times of 60, 90, and 120 min.

    VED is calculated before rounding as:

        power / (speed * layer thickness * hatch spacing)

    With the default settings and VED filtering enabled, the script
    produces 106,624 candidates.

    No predicted-density or manufacturability filter is applied.
    """

    if limit_ved_range not in (0, 1, False, True):
        raise ValueError(
            "limit_ved_range must be 0, 1, False, or True."
        )

    lt_values = tuple(lt_range)
    hd_values = tuple(hd_range)

    for name, values in (
        ("lt_range", lt_values),
        ("hd_range", hd_values),
    ):
        if not values:
            raise ValueError(f"{name} must contain at least one value.")

        if any(value <= 0 for value in values):
            raise ValueError(f"All values in {name} must be positive.")

    aged_temperatures = tuple(
        temperature_c + LEGACY_KELVIN_OFFSET
        for temperature_c in AGEING_TEMPERATURES_C
    )

    thermal_states = []

    for ageing_time in AGEING_TIMES_MIN:
        if ageing_time == 0:
            temperatures = (LEGACY_KELVIN_OFFSET,)
        else:
            temperatures = aged_temperatures

        thermal_states.extend(
            (temperature, ageing_time)
            for temperature in temperatures
        )

    records = []

    for power, speed, layer_thickness, hatch_spacing in product(
        POWER_VALUES,
        SPEED_VALUES,
        lt_values,
        hd_values,
    ):
        ved = power / (
            speed * layer_thickness * hatch_spacing
        )

        if limit_ved_range and not VED_MIN <= ved <= VED_MAX:
            continue

        build_condition = (
            power,
            speed,
            layer_thickness,
            hatch_spacing,
            round(ved, 2),
        )

        records.extend(
            (*build_condition, temperature, ageing_time)
            for temperature, ageing_time in thermal_states
        )

    candidates = pd.DataFrame.from_records(
        records,
        columns=COLUMNS,
    )

    output_path = (
        DEFAULT_OUTPUT_FILE
        if output_file is None
        else Path(output_file).expanduser().resolve()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)

    print("Candidate generation complete.")
    print(f"Total candidates: {len(candidates):,}")
    print(f"Saved to: {output_path}")

    print("\nData preview:")
    print(candidates.head(10))

    print(f"\nData shape: {candidates.shape}")

    print("\nParameter ranges:")

    if candidates.empty:
        print(
            "No candidates satisfy the selected parameter "
            "ranges and constraints."
        )
    else:
        print(
            f"power: {candidates['power'].min()} - "
            f"{candidates['power'].max()} W; grid step: 10 W"
        )
        print(
            f"speed: {candidates['speed'].min()} - "
            f"{candidates['speed'].max()} mm/s; grid step: 10 mm/s"
        )
        print(
            f"lt: {sorted(candidates['lt'].unique().tolist())} mm"
        )
        print(
            f"hd: {sorted(candidates['hd'].unique().tolist())} mm"
        )
        print(
            f"VED: {candidates['VED'].min():.2f} - "
            f"{candidates['VED'].max():.2f} J/mm^3"
        )
        print(
            f"T (all rows): {candidates['T'].min()} - "
            f"{candidates['T'].max()} K"
        )

        aged_temperatures_present = candidates.loc[
            candidates["t"] > 0,
            "T",
        ]

        if not aged_temperatures_present.empty:
            temperature_min = aged_temperatures_present.min()
            temperature_max = aged_temperatures_present.max()

            print(
                f"T (aged rows): {temperature_min} - "
                f"{temperature_max} K "
                f"({temperature_min - LEGACY_KELVIN_OFFSET} - "
                f"{temperature_max - LEGACY_KELVIN_OFFSET} deg C); "
                "grid step: 5 K"
            )

        time_values = sorted(
            candidates["t"].unique().tolist()
        )
        time_hours = sorted(
            (candidates["t"].unique() / 60).tolist()
        )

        print(f"t: {time_values} min")
        print(f"t: {time_hours} h")

    print(
        "As-built rows: t = 0 min with T = 273 K "
        "as the historical placeholder."
    )
    print(
        "Aged rows: t > 0 min with T = 673-893 K."
    )

    if limit_ved_range:
        print(
            f"VED filter: enabled "
            f"({VED_MIN} <= raw VED <= {VED_MAX} J/mm^3)."
        )
    else:
        print("VED filter: disabled.")

    return candidates


if __name__ == "__main__":
    generate_test_data()