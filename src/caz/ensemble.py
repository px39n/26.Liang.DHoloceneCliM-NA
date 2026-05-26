"""Combine per-GCM downscaled outputs into ensemble products (mean, spread, total PI)."""
from __future__ import annotations
import numpy as np
import xarray as xr


def ensemble_combine(per_gcm: list[xr.Dataset]) -> xr.Dataset:
    """Stack per-GCM outputs along a 'gcm' dim and produce mean, std, combined PI.

    Each input must have variables: tas, pr, tas_pi_lo, tas_pi_hi, pr_pi_lo, pr_pi_hi.
    """
    stack = xr.concat(per_gcm, dim="gcm")

    out = xr.Dataset()
    for v in ("tas", "pr"):
        out[f"{v}_mean"] = stack[v].mean("gcm")
        out[f"{v}_gcm_spread"] = stack[v].std("gcm")
        # combine: PI half-width^2_total = mean(stat_var) + GCM_var
        stat_half = (stack[f"{v}_pi_hi"] - stack[f"{v}_pi_lo"]) / 2.0
        stat_var = (stat_half ** 2).mean("gcm")
        total_half = np.sqrt(stat_var + out[f"{v}_gcm_spread"] ** 2)
        out[f"{v}_total_lo"] = out[f"{v}_mean"] - 1.96 * total_half
        out[f"{v}_total_hi"] = out[f"{v}_mean"] + 1.96 * total_half
    return out
