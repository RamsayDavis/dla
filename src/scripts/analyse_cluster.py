from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim import utils  # type: ignore[import]


def analyze_cluster(npz_path: str | Path, show_plot: bool = True) -> None:
    npz_path = str(npz_path)
    print(f"Loading {npz_path}...")

    try:
        # Prefer the higher-level loader so this works for all models
        result = utils.load_cluster(npz_path)
        meta = result.ensure_meta()

        x: np.ndarray | None = None
        y: np.ndarray | None = None

        # New-style Koh / lattice outputs: integer coordinates in meta
        if "x_coords" in meta and "y_coords" in meta:
            x = np.asarray(meta["x_coords"], dtype=float)
            y = np.asarray(meta["y_coords"], dtype=float)
        # Continuous model or older runs: positions as (N, 2)
        elif result.positions is not None:
            pos = np.asarray(result.positions, dtype=float)
            if pos.ndim != 2 or pos.shape[1] < 2:
                raise ValueError(
                    f"positions array has unexpected shape {pos.shape}, "
                    "cannot extract x/y coordinates"
                )
            x = pos[:, 0]
            y = pos[:, 1]
        else:
            raise ValueError(
                "Could not find particle coordinates in file. "
                "Expected meta['x_coords']/['y_coords'] or a positions array."
            )

        # Filter out any NaNs that may have crept in
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        n_particles = len(x)
        if n_particles < 2:
            raise ValueError(f"Too few particles ({n_particles}) for analysis.")

        print(f"Particles found: {n_particles:,}")

    except Exception as e:  # pragma: no cover - CLI safety
        print(f"Error loading or parsing file: {e}")
        return

    # --- 1. Vectorized cumulative radius of gyration ---
    cum_x = np.cumsum(x)
    cum_y = np.cumsum(y)
    cum_x2 = np.cumsum(x ** 2)
    cum_y2 = np.cumsum(y ** 2)

    ns = np.arange(1, n_particles + 1, dtype=float)

    cm_x = cum_x / ns
    cm_y = cum_y / ns

    rg_sq = (cum_x2 + cum_y2) / ns - (cm_x ** 2 + cm_y ** 2)
    rg = np.sqrt(np.maximum(0.0, rg_sq))

    # --- 2. Fit scaling law Rg ~ N^(1 / Df) on a log-log plot ---
    # Ignore very early growth where the cluster is dominated by seed noise.
    start_idx = min(1000, n_particles // 10)

    # Guard against zeros which would break log()
    fit_rg = rg[start_idx:]
    fit_ns = ns[start_idx:]
    valid = fit_rg > 0
    fit_rg = fit_rg[valid]
    fit_ns = fit_ns[valid]

    if fit_rg.size < 2:
        print("Not enough valid data points for linear regression in log-log space.")
        return

    log_n = np.log(fit_ns)
    log_rg = np.log(fit_rg)

    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_rg)

    fractal_dim = 1.0 / slope
    growth_exponent = slope  # beta in Rg ~ N^beta

    print("-" * 40)
    print("CLUSTER SCALING ANALYSIS")
    print("-" * 40)
    print("Expected DLA Df : ~1.71")
    print(f"Measured Df     :  {fractal_dim:.4f}")
    print(f"Growth Exp (beta):  {growth_exponent:.44f}")
    print(f"R^2 (linearity) :  {r_value ** 2:.6f}")
    print("-" * 40)

    if r_value ** 2 < 0.99:
        print(
            "Warning: log-log fit is not strongly linear "
            "(R^2 < 0.99). Cluster may be too small or bounded."
        )

    # --- 3. Plotting ---
    if show_plot:
        plt.figure(figsize=(10, 6))

        # Raw data
        plt.loglog(ns, rg, label="Simulation data", color="blue", alpha=0.6, linewidth=2)

        # Best-fit line: Rg = exp(intercept) * N^slope
        fit_y = np.exp(intercept) * fit_ns ** slope
        plt.loglog(
            fit_ns,
            fit_y,
            "r--",
            label=f"Fit: Df = {fractal_dim:.2f}",
            linewidth=2,
        )

        plt.xlabel(r"Number of particles $N$")
        plt.ylabel(r"Radius of gyration $R_g$")
        plt.title(
            f"DLA scaling analysis\nN={n_particles:,} | Df={fractal_dim:.3f}",
        )
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute scaling statistics (Df, beta) for a saved DLA cluster (.npz)."
    )
    parser.add_argument("file", help="Path to the .npz file (e.g. results/*.npz)")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable log-log Rg vs N plot (print stats only).",
    )
    args = parser.parse_args()

    analyze_cluster(args.file, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()


