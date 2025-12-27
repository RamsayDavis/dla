"""
Dual Fractal Analysis Script for Standardized DLA Clusters.

Implements two methods to calculate Fractal Dimension:
1. Scaling Relation (Growth History) - R_g evolution
2. Sandbox Method (Static Geometry) - mass-radius relation
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


def validate_positions(positions: np.ndarray) -> np.ndarray:
    """
    Validate and return positions array.
    
    Args:
        positions: Input positions array
        
    Returns:
        Validated positions array of shape (N, 2)
        
    Raises:
        ValueError: If positions is None or invalid
    """
    if positions is None:
        raise ValueError("result.positions is None. Cannot perform analysis.")
    
    pos = np.asarray(positions, dtype=np.float64)
    
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(
            f"Expected positions to be shape (N, 2), got {pos.shape}. "
            "Legacy formats are not supported."
        )
    
    # Filter out NaNs/Infs
    mask = np.isfinite(pos).all(axis=1)
    pos = pos[mask]
    
    if len(pos) < 100:
        raise ValueError(
            f"Too few valid particles ({len(pos)}) for reliable fractal analysis. "
            "Need at least 100 particles."
        )
    
    return pos


def calculate_scaling_dimension(positions: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Method A: Calculate fractal dimension using scaling relation (growth history).
    
    Calculates R_g(n) evolution and performs log-log fit:
    R_g ~ N^(1/Df) => log(R_g) = (1/Df) * log(N) + C
    
    Args:
        positions: Array of shape (N, 2) with particle positions
        
    Returns:
        Tuple of (Df_scaling, r_squared, intercept, log_n, log_rg, fit_log_n, fit_log_rg)
    """
    n_particles = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Calculate R_g(n) for all n using cumulative sums
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
    
    # Fitting range: 1% to 100% of N
    start_idx = max(1, int(n_particles * 0.01))
    
    fit_rg = rg[start_idx:]
    fit_ns = ns[start_idx:]
    
    # Remove any zero radii
    valid = fit_rg > 0
    fit_rg = fit_rg[valid]
    fit_ns = fit_ns[valid]
    
    if len(fit_rg) < 10:
        raise ValueError("Too few valid points for scaling analysis after filtering.")
    
    # Perform Linear Regression on Log-Log data
    log_n = np.log(ns)
    log_rg = np.log(rg)
    
    fit_log_n = np.log(fit_ns)
    fit_log_rg = np.log(fit_rg)
    
    slope, intercept, r_value, p_value, std_err = linregress(fit_log_n, fit_log_rg)
    
    # Interpretation: Rg ~ N^(1/Df)  =>  log(Rg) = (1/Df)*log(N) + C
    # Therefore: slope = 1/Df  =>  Df = 1/slope
    Df_scaling = 1.0 / slope
    r_squared = r_value**2
    
    return Df_scaling, r_squared, intercept, log_n, log_rg, fit_log_n, fit_log_rg


def calculate_sandbox_dimension(positions: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Method B: Calculate fractal dimension using sandbox method (static geometry).
    
    Counts mass M(<R) inside radius R and performs log-log fit:
    M(<R) ~ R^Df => log(M) = Df * log(R) + C
    
    Args:
        positions: Array of shape (N, 2) with particle positions
        
    Returns:
        Tuple of (Df_sandbox, r_squared, intercept, log_r, log_m)
    """
    # Calculate distance from seed (0, 0) for each particle
    distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    max_distance = np.max(distances)
    
    # Define log-spaced radii from R_min=2.0 to max_distance
    R_min = 2.0
    if max_distance <= R_min:
        raise ValueError(
            f"Maximum particle distance ({max_distance:.2f}) is too small. "
            f"Need at least {R_min} for sandbox analysis."
        )
    
    # Use log-spaced radii (typically 50-100 points)
    n_radii = min(100, max(50, len(positions) // 10))
    radii = np.logspace(np.log10(R_min), np.log10(max_distance), n_radii)
    
    # Count mass M(<R) for each radius
    masses = np.array([np.sum(distances <= R) for R in radii])
    
    # Filter out zero masses and ensure we have enough points
    valid = masses > 0
    radii = radii[valid]
    masses = masses[valid]
    
    if len(radii) < 10:
        raise ValueError("Too few valid points for sandbox analysis after filtering.")
    
    # Perform log-log linear regression
    log_r = np.log(radii)
    log_m = np.log(masses)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_r, log_m)
    
    # Interpretation: M(<R) ~ R^Df => log(M) = Df * log(R) + C
    # Therefore: slope = Df
    Df_sandbox = slope
    r_squared = r_value**2
    
    return Df_sandbox, r_squared, intercept, log_r, log_m


def analyze_cluster(
    npz_path: str | Path,
    output_path: str | Path | None = None,
    show_plot: bool = False
) -> None:
    """
    Perform dual fractal analysis on a standardized DLA cluster.
    
    Args:
        npz_path: Path to input .npz file
        output_path: Optional path to save output image
        show_plot: Whether to display plot interactively
    """
    npz_path = Path(npz_path)
    print(f"Loading {npz_path}...")
    
    try:
        result = utils.load_cluster(npz_path)
        positions = validate_positions(result.positions)
        n_particles = len(positions)
        print(f"Particles found: {n_particles:,}")
        
    except Exception as e:
        print(f"Error loading or validating file: {e}")
        raise
    
    # Method A: Scaling Relation
    print("\n" + "=" * 60)
    print("METHOD A: Scaling Relation (Growth History)")
    print("=" * 60)
    try:
        Df_scaling, r2_scaling, intercept_scaling, log_n, log_rg, fit_log_n, fit_log_rg = calculate_scaling_dimension(positions)
        print(f"Fractal Dimension (Df_scaling): {Df_scaling:.5f}")
        print(f"R² (Linearity): {r2_scaling:.6f}")
        print(f"Fitting Range: N = {int(np.exp(fit_log_n[0]))} to {int(np.exp(fit_log_n[-1]))}")
    except Exception as e:
        print(f"Error in scaling analysis: {e}")
        raise
    
    # Method B: Sandbox Method
    print("\n" + "=" * 60)
    print("METHOD B: Sandbox Method (Static Geometry)")
    print("=" * 60)
    try:
        Df_sandbox, r2_sandbox, intercept_sandbox, log_r, log_m = calculate_sandbox_dimension(positions)
        print(f"Fractal Dimension (Df_sandbox): {Df_sandbox:.5f}")
        print(f"R² (Linearity): {r2_sandbox:.6f}")
    except Exception as e:
        print(f"Error in sandbox analysis: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Scaling Method:  Df = {Df_scaling:.5f} (R² = {r2_scaling:.6f})")
    print(f"Sandbox Method:  Df = {Df_sandbox:.5f} (R² = {r2_sandbox:.6f})")
    print(f"Difference:      |ΔDf| = {abs(Df_scaling - Df_sandbox):.5f}")
    print("=" * 60)
    
    # Create dual subplot visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Scaling Law
    ax1.scatter(log_n, log_rg, color='black', alpha=0.3, s=1, label='Simulation Data')
    
    # Overlay best-fit line using the regression intercept
    slope_scaling = 1.0 / Df_scaling
    fit_curve_log_rg = slope_scaling * fit_log_n + intercept_scaling
    ax1.plot(fit_log_n, fit_curve_log_rg, color='red', linestyle='--', linewidth=2,
             label=f'Fit: $D_f^{{scaling}} = {Df_scaling:.3f}$')
    
    ax1.set_xlabel(r'$\log(N)$')
    ax1.set_ylabel(r'$\log(R_g)$')
    ax1.set_title(f'Scaling Relation: $R_g \\sim N^{{1/{Df_scaling:.2f}}}$\n'
                  f'$D_f^{{scaling}} = {Df_scaling:.3f}$ (R² = {r2_scaling:.4f})')
    ax1.legend()
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # Right subplot: Sandbox Method
    ax2.scatter(log_r, log_m, color='blue', alpha=0.3, s=1, label='Simulation Data')
    
    # Overlay best-fit line using the regression intercept
    slope_sandbox = Df_sandbox
    fit_curve_log_m = slope_sandbox * log_r + intercept_sandbox
    ax2.plot(log_r, fit_curve_log_m, color='red', linestyle='--', linewidth=2,
             label=f'Fit: $D_f^{{sandbox}} = {Df_sandbox:.3f}$')
    
    ax2.set_xlabel(r'$\log(R)$')
    ax2.set_ylabel(r'$\log(M(<R))$')
    ax2.set_title(f'Sandbox Method: $M(<R) \\sim R^{{D_f}}$\n'
                  f'$D_f^{{sandbox}} = {Df_sandbox:.3f}$ (R² = {r2_sandbox:.4f})')
    ax2.legend()
    ax2.grid(True, which="both", linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = npz_path.with_suffix('.png').with_name(
            npz_path.stem + '_analysis.png'
        )
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform dual fractal analysis on standardized DLA clusters."
    )
    parser.add_argument("file", help="Path to the .npz file")
    parser.add_argument(
        "--out",
        type=str,
        help="Output path for the analysis figure (default: <input>_analysis.png)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively"
    )
    args = parser.parse_args()
    
    analyze_cluster(args.file, output_path=args.out, show_plot=args.show)


if __name__ == "__main__":
    main()
