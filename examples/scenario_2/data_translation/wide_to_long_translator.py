import pandas as pd
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).parent

input_file = BASE_DIR / "lexcen_data.csv"

# Go up one directory from data_translation
output_file = BASE_DIR.parent / "forecast_timeseries.csv"

# Read CSV
df = pd.read_csv(input_file, sep="\t")

# Keep only E1 suffix records
df = df[df["NmiSuffix"] == "E1"]

print(f"Rows after E1 filter: {len(df):,}")
# Identify the interval columns
interval_cols = [c for c in df.columns if c.startswith("Data_")]

# Convert from wide to long
long_df = df.melt(
    id_vars=["Nmi", "IntervalDay"],
    value_vars=interval_cols,
    var_name="interval",
    value_name="energy"
)

# Remove blanks/nulls
long_df = long_df.dropna(subset=["energy"])

# Convert Data_07_30 -> 07:30
long_df["time"] = (
    long_df["interval"]
    .str.replace("Data_", "", regex=False)
    .str.replace("_", ":", regex=False)
)

# Create timestamp
long_df["timestamp"] = (
    pd.to_datetime(long_df["IntervalDay"]).dt.strftime("%Y-%m-%d")
    + " "
    + long_df["time"]
)

# NMI becomes load_id
long_df["load_id"] = long_df["Nmi"]

# Convert kW to W
long_df["real_power_w"] = long_df["energy"] * 1000

# Estimate reactive power
# Example uses roughly 0.4 VAR per W
long_df["reactive_power_var"] = long_df["real_power_w"] * 0.4

# Final format
output_df = long_df[
    ["timestamp", "load_id", "real_power_w", "reactive_power_var"]
].copy()

# Round values
output_df["real_power_w"] = output_df["real_power_w"].round(0).astype(int)
output_df["reactive_power_var"] = output_df["reactive_power_var"].round(0).astype(int)

# Sort
output_df = output_df.sort_values(
    ["timestamp", "load_id"]
)

# Export
output_df.to_csv(output_file, index=False)

print(f"Created {output_file}")
print(f"Rows: {len(output_df):,}")