"""T4-5: Mean Skill Score (our Table 4-5 = Guaita Table 4-5).

MSS = weighted average of per-metric skill scores.
Skill score SS_A = (ESM_error - PCR_error) / (ESM_error - perfect_error)
where perfect_error = 0 for MB/MAE, 1 for KGE/r.

Reads T1_timestep_metrics.csv and T2_station_metrics.csv.

Output: T45_mss.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commons"))

import numpy as np
import pandas as pd
from _common import RESULTS_MAIN

RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_MAIN / "T45_mss.csv"


def skill_score(pcr_val, esm_val, metric):
    """Compute skill score for a single metric.

    For error metrics (MB, MAE): SS = 1 - |PCR| / |ESM|
    For efficiency metrics (KGE, r): SS = (PCR - ESM) / (perfect - ESM)
    """
    if metric in ("MB",):
        # Use absolute values for bias
        pcr_abs = abs(pcr_val)
        esm_abs = abs(esm_val)
        if esm_abs == 0:
            return 0.0
        return 1.0 - pcr_abs / esm_abs
    elif metric in ("MAE",):
        if esm_val == 0:
            return 0.0
        return 1.0 - pcr_val / esm_val
    elif metric in ("KGE", "r"):
        perfect = 1.0
        if esm_val == perfect:
            return 0.0
        return (pcr_val - esm_val) / (perfect - esm_val)
    return np.nan


def compute_mss(df, label, gcm_label="TraCE"):
    """Compute MSS from a metrics DataFrame."""
    rows = []
    for var in df["var"].unique():
        dv = df[df["var"] == var]
        metrics = dv["metric"].unique()

        weights = {}
        for m in metrics:
            vals = dv[dv["metric"] == m]["ESM"].abs().values
            weights[m] = np.nanmean(vals) if len(vals) > 0 else 1.0

        w_total = sum(weights.values())
        for m in weights:
            weights[m] /= w_total

        ss_list = []
        w_list = []
        for _, row in dv.iterrows():
            pcr_v = row["PCR"]
            esm_v = row["ESM"]
            m = row["metric"]
            if np.isnan(pcr_v) or np.isnan(esm_v):
                continue
            ss = skill_score(pcr_v, esm_v, m)
            if np.isfinite(ss):
                ss_list.append(ss)
                w_list.append(weights.get(m, 1.0))

        mss = np.average(ss_list, weights=w_list) * 100 if ss_list else np.nan

        rows.append({
            "level": label, "gcm": gcm_label, "var": var,
            "MSS_pct": round(mss, 1), "n_components": len(ss_list),
        })
        print(f"  {label} {gcm_label} {var}: MSS = {mss:.1f}%")

    return rows


def main():
    t1_path = RESULTS_MAIN / "T1_timestep_metrics.csv"
    t2_path = RESULTS_MAIN / "T2_station_metrics.csv"

    rows = []

    for path, label in [(t1_path, "timestep"), (t2_path, "station")]:
        if not path.exists():
            print(f"WARNING: {path} not found")
            continue
        df = pd.read_csv(path)
        if "gcm" in df.columns:
            for gcm_label in df["gcm"].unique():
                sub = df[df["gcm"] == gcm_label]
                rows.extend(compute_mss(sub, label, gcm_label))
        else:
            rows.extend(compute_mss(df, label, "TraCE"))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
