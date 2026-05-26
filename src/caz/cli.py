"""Command-line entry: caz <command> ..."""
from __future__ import annotations
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def preprocess(gcm: str, ghcn: str, out: str = "data/interim/pp.nc") -> None:
    """§1 preprocessing: regrid + anomaly + station subset + split."""
    typer.echo(f"[stub] preprocess gcm={gcm} ghcn={ghcn} -> {out}")


@app.command()
def fit(pp: str, out: str = "data/interim/model.nc") -> None:
    """§2 fit PCA + per-station GLM + mode selection."""
    typer.echo(f"[stub] fit {pp} -> {out}")


@app.command()
def predict(model: str, gcm_past: str, out: str = "data/processed/Ohat.nc") -> None:
    """§3 + §5: project Holocene to station + grid + variance rescale."""
    typer.echo(f"[stub] predict {model} {gcm_past} -> {out}")


@app.command()
def ensemble(per_gcm_glob: str, out: str = "data/output/ensemble.zarr") -> None:
    """Combine multiple per-GCM outputs into ensemble product."""
    typer.echo(f"[stub] ensemble {per_gcm_glob} -> {out}")


if __name__ == "__main__":
    app()
