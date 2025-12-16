"""
Universal DLA Analysis Script.
Handles:
1. Lattice models (meta['x_coords'], meta['y_coords'])
2. Continuous models (positions array of complex128)
3. Standard XY models (positions array of shape (N, 2))
"""
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


def get_coordinates(result) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustly extract x and y coordinate arrays from a ClusterResult.
    """
    meta = result.ensure_meta()

    # 1. Check for split coordinates (Koh / Lattice optimizations)
    if "x_coords" in meta and "y_coords" in meta:
        x = np.asarray(meta["x_coords"], dtype=np.float64)
        y = np.asarray(meta["y_coords"], dtype=np.float64)
        return x, y

    # 2. Check for positions array
    if result.positions is not None:
        pos = np.asarray(result.positions)
        
        # Case A: Complex Numbers (Bell / Continuous DLA)
        if np.iscomplexobj(pos):
            x = pos.real
            y = pos.imag
            return x, y
            
        # Case B: Standard (N, 2) array
        if pos.ndim == 2 and pos.shape[1] >= 2:
            x = pos[:, 0]
            y = pos[:, 1]
            return x, y
            
        # Case C: Flattened 1D array (Edge case)
        if pos.ndim == 1:
            raise ValueError("Positions array is 1D but not complex. Cannot determine X/Y.")

    raise ValueError(
        "Could not find valid coordinates. Checked: meta['x_coords'], positions(complex), positions(N,2)."
    )


def analyze_cluster(npz_path: str | Path, show_plot: bool = True) -> None:
    npz_path = str(npz_path)
    print(f"Loading {npz_path}...")

    try:
        result = utils.load_cluster(npz_path)
        
        # --- ROBUST LOADING ---
        x, y = get_coordinates(result)

        # Filter out NaNs/Infs
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        n_particles = len(x)
        if n_particles < 100:
            raise ValueError(f"Too few particles ({n_particles}) for reliable scaling analysis.")

        print(f"Particles found: {n_particles:,}")

    except Exception as e:
        print(f"Error loading or parsing file: {e}")
        return

    # --- 1. Vectorized Radius of Gyration Calculation ---
    # We assume particles are stored in order of aggregation (0 to N).
    # This allows us to calculate the 'evolution' of Rg over time using cumsum.
    
    ns = np.arange(1, n_particles + 1, dtype=np.float64)
    
    # Cumulative sums for Center of Mass (CM)
    cum_x = np.cumsum(x)
    cum_y = np.cumsum(y)
    
    # Cumulative sums for squared distances
    cum_x2 = np.cumsum(x**2)
    cum_y2 = np.cumsum(y**2)

    # CM at step n
    cm_x = cum_x / ns
    cm_y = cum_y / ns

    # Radius of Gyration squared at step n
    # Rg^2 = (1/N) * sum(r_i^2) - r_cm^2
    rg_sq = (cum_x2 + cum_y2) / ns - (cm_x**2 + cm_y**2)
    
    # Numerical stability: clamp negative epsilon to 0
    rg = np.sqrt(np.maximum(0.0, rg_sq))

    # --- 2. Scaling Analysis (Log-Log Fit) ---
    
    # Cut off the "Transient" phase.
    # For small N, the seed dominates. For Continuous DLA, the first few particles
    # are just sticking to a seed circle, which is 1D, not fractal.
    # We ignore the first ~1% or 1000 particles, whichever is larger.
    start_idx = max(100, int(n_particles * 0.01))
    
    fit_rg = rg[start_idx:]
    fit_ns = ns[start_idx:]
    
    # Remove any zero radii (usually only index 0)
    valid = fit_rg > 0
    fit_rg = fit_rg[valid]
    fit_ns = fit_ns[valid]

    # Perform Linear Regression on Log-Log data
    log_n = np.log(fit_ns)
    log_rg = np.log(fit_rg)

    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_rg)

    # Interpretation: Rg ~ N^(1/Df)  =>  log(Rg) = (1/Df)*log(N) + C
    # Therefore: slope = 1/Df  =>  Df = 1/slope
    fractal_dim = 1.0 / slope
    
    print("-" * 40)
    print("CLUSTER SCALING ANALYSIS")
    print("-" * 40)
    print(f"Model Type      : Continuous / Off-Lattice")
    print(f"Particle Count  : {n_particles}")
    print(f"Fitting Range   : N = {int(fit_ns[0])} to {int(fit_ns[-1])}")
    print("-" * 40)
    print(f"Slope (beta)    : {slope:.5f}")
    print(f"Fractal Dim (Df): {fractal_dim:.5f}")
    print(f"R^2 (Linearity) : {r_value**2:.6f}")
    print(f"Std Error       : {std_err:.6f}")
    print("-" * 40)
    
    # Expected Df for Off-Lattice DLA is often cited ~1.71
    # Check bounds
    if not (1.6 < fractal_dim < 1.8):
        print("NOTE: Df is outside typical range (1.6-1.8). Check for finite size effects.")

    # --- 3. Plotting ---
    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Data
        ax.loglog(ns, rg, color='black', alpha=0.3, label='Simulation Data', linewidth=1)
        
        # Plot Fit
        # y = exp(intercept) * x^slope
        fit_curve = np.exp(intercept) * fit_ns**slope
        ax.loglog(fit_ns, fit_curve, color='red', linestyle='--', linewidth=2, 
                  label=f'Fit: $D_f = {fractal_dim:.3f}$')

        # Aesthetics
        ax.set_xlabel(r'Number of Particles ($N$)')
        ax.set_ylabel(r'Radius of Gyration ($R_g$)')
        ax.set_title(f'Scaling Analysis: $R_g \\sim N^{{1/{fractal_dim:.2f}}}$')
        ax.legend()
        ax.grid(True, which="both", linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute scaling statistics (Df, beta) for DLA clusters."
    )
    parser.add_argument("file", help="Path to the .npz file")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot display.",
    )
    args = parser.parse_args()

    analyze_cluster(args.file, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()