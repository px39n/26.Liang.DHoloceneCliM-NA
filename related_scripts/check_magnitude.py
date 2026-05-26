"""Check magnitude reasonableness of recon_station_tas.csv."""
import pandas as pd
import numpy as np

csv = r"D:\Dataset\DPastCliM-NA\interim\trace21k\station_cal\recon_station_tas.csv"
print("Loading full file ...")
df = pd.read_csv(csv)
print(f"Shape: {df.shape}\n")

# Time period analysis
periods = {
    "LGM (-20050 to -18000)": (-20050, -18000),
    "Deglaciation (-18000 to -10000)": (-18000, -10000),
    "Early Holocene (-10000 to -5000)": (-10000, -5000),
    "Mid Holocene (-5000 to 0)": (-5000, 0),
    "Late Holocene (0 to 1000)": (0, 1000),
    "Medieval (800 to 1300)": (800, 1300),
    "LIA (1400 to 1850)": (1400, 1850),
    "Modern (1850 to 1990)": (1850, 1990),
}

print("=== Temperature by period (annual mean, all stations) ===")
for name, (t0, t1) in periods.items():
    mask = (df["time"] >= t0) & (df["time"] <= t1)
    sub = df.loc[mask, "value"]
    if len(sub) > 0:
        print(f"  {name}: mean={sub.mean():.2f}°C  std={sub.std():.2f}  "
              f"[{sub.min():.1f}, {sub.max():.1f}]  N={mask.sum()}")

# Latitude band analysis for modern period
print("\n=== Modern period by latitude band ===")
mod = df[(df["time"] >= 1850) & (df["time"] <= 1990)]
for lat_lo, lat_hi, label in [(7, 25, "Tropical"), (25, 40, "Subtropical"),
                                (40, 55, "Temperate"), (55, 75, "Subarctic")]:
    mask = (mod["lat"] >= lat_lo) & (mod["lat"] < lat_hi)
    sub = mod.loc[mask, "value"]
    if len(sub) > 0:
        print(f"  {label} ({lat_lo}-{lat_hi}°N): mean={sub.mean():.2f}°C  "
              f"[{sub.min():.1f}, {sub.max():.1f}]")

# LGM vs Modern temperature anomaly
print("\n=== LGM-Modern anomaly ===")
lgm = df[(df["time"] >= -20050) & (df["time"] <= -18000)]["value"].mean()
modern = df[(df["time"] >= 1900) & (df["time"] <= 1990)]["value"].mean()
print(f"  LGM mean: {lgm:.2f}°C")
print(f"  Modern mean: {modern:.2f}°C")
print(f"  LGM - Modern: {lgm - modern:.2f}°C")
print(f"  Expected: ~-5 to -8°C (PMIP consensus for NH continental)")

# PI width by latitude
print("\n=== PI width by latitude ===")
for lat_lo, lat_hi, label in [(7, 25, "Tropical"), (25, 40, "Subtropical"),
                                (40, 55, "Temperate"), (55, 75, "Subarctic")]:
    mask = (df["lat"] >= lat_lo) & (df["lat"] < lat_hi)
    sub = df.loc[mask]
    width = sub["pi_hi"] - sub["pi_lo"]
    print(f"  {label}: mean width={width.mean():.2f}°C")
