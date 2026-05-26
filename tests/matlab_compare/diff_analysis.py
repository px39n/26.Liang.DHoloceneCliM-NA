import pandas as pd
import numpy as np

py = pd.read_parquet(r"D:\Dataset\DPastCliM-NA\GHCN\interim\ghcn_tas_obs.parquet")
ml = pd.read_parquet(r"D:\Dataset\DPastCliM-NA\GHCN\interim\matlab_ghcn_tas_obs.parquet").rename(columns={"Value": "value"})
ml["ID"] = ml["ID"].astype(str)

m = py.merge(
    ml[["ID", "year", "month", "value"]],
    on=["ID", "year", "month"],
    how="outer",
    suffixes=("_py", "_ml"),
    indicator=True,
)
both = m[m["_merge"] == "both"]
diff = both[~np.isclose(both["value_py"], both["value_ml"], atol=0.005)]
print(f"value-diff: {len(diff):,} of {len(both):,} matched rows")

per = m.groupby("ID")["_merge"].agg(
    both=lambda s: (s == "both").sum(),
    total=len,
).reset_index()
diff_per = diff.groupby("ID").size().rename("n_diff")
per = per.join(diff_per, on="ID").fillna(0)
per["frac_diff"] = per["n_diff"] / per["total"]

print(f"\nstations with 0 diff: {(per.n_diff == 0).sum():,} / {len(per):,}")
print(f"stations with >50% diff: {(per.frac_diff > 0.5).sum():,}")
print(f"stations with any diff: {(per.n_diff > 0).sum():,}")
print("\nstations with worst diff fraction:")
print(per.sort_values("frac_diff", ascending=False).head(5).to_string(index=False))

# Are differences concentrated in *early years* (matches the year-shift hypothesis)?
print("\nyear-diff: are diffs older years or newer?")
print(diff.groupby(diff["year"] // 10 * 10).size().sort_index().tail(15).rename("n_diff").to_string())
