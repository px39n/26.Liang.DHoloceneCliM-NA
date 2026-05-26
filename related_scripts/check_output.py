"""Quick sanity check of recon_station_tas.csv."""
import pandas as pd
import numpy as np

csv = r"D:\Dataset\DPastCliM-NA\interim\trace21k\station_cal\recon_station_tas.csv"

print("Loading first 100k rows ...")
df = pd.read_csv(csv, nrows=100000)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")
print("First 10 rows:")
print(df.head(10).to_string())
print("\nOverall stats:")
print(df.describe().to_string())

print("\n=== value vs value_real ===")
diff = df["value_real"] - df["value"]
print(f"  Mean diff: {diff.mean():.4f}")
print(f"  Std diff:  {diff.std():.4f}")
print(f"  Max |diff|: {diff.abs().max():.4f}")

print("\n=== PI width ===")
width = df["pi_hi"] - df["pi_lo"]
print(f"  Mean width: {width.mean():.4f}")
print(f"  Min width:  {width.min():.4f}")
print(f"  Max width:  {width.max():.4f}")

print("\n=== Time range (full file) ===")
full_time = pd.read_csv(csv, usecols=["time"])
print(f"  Time range: {full_time['time'].min()} to {full_time['time'].max()}")
print(f"  Unique times: {full_time['time'].nunique()}")
print(f"  Total rows: {len(full_time)}")

print("\n=== NaN check ===")
for col in df.columns:
    n_nan = df[col].isna().sum()
    print(f"  {col}: {n_nan} NaN ({100*n_nan/len(df):.1f}%)")
