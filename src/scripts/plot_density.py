"""
Coarse-Grained Density Plotter for DLA Clusters.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim import utils

@njit(cache=True)
def compute_density_grid(x_coords, y_coords, grid, scale, pad_x, pad_y):
    """
    Numba-accelerated 2D Histogram.
    Iterates through particles and increments the corresponding pixel counter.
    """
    H, W = grid.shape
    num_particles = len(x_coords)
    
    # Note: For strict parallel safety, one would use atomics.
    # However, for visualization histograms of sparse DLA, standard reduction 
    # is performant and visual artifacts from race conditions are negligible.
    for i in range(num_particles):
        px = int((x_coords[i] - pad_x) * scale)
        py = int((y_coords[i] - pad_y) * scale)
        
        if 0 <= px < W and 0 <= py < H:
            grid[py, px] += 1


def render_density(
    result, 
    output_path, 
    res_power=11, 
    mode="log", 
    cmap="Greys"
):
    # 1. Setup Resolution
    res = 1 << res_power  # e.g., 2^11 = 2048
    print(f"Target Resolution: {res}x{res} (2^{res_power})")
    
    # 2. Get Coordinates
    if hasattr(result, 'meta') and "x_coords" in result.meta:
        x = np.array(result.meta["x_coords"])
        y = np.array(result.meta["y_coords"])
    elif result.positions is not None:
        x = result.positions[:, 0]
        y = result.positions[:, 1]
    else:
        raise ValueError("No coordinates found.")
        
    # 3. Calculate Scale & Grain
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    width_data = x_max - x_min
    height_data = y_max - y_min
    max_dim = max(width_data, height_data)
    
    # Add 2% padding so it doesn't touch the edge
    max_dim *= 1.02
    
    scale = res / max_dim
    grain_size = 1.0 / scale
    
    print(f"Cluster Width:  {width_data:.1f} units")
    print(f"Grain Size:     1 Pixel = {grain_size:.2f} x {grain_size:.2f} unit block")
    
    # 4. Center Data
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    pad_x = center_x - (max_dim / 2)
    pad_y = center_y - (max_dim / 2)
    
    # 5. Binning
    grid = np.zeros((res, res), dtype=np.int32)
    print(f"Binning {len(x):,} particles...")
    compute_density_grid(x, y, grid, scale, pad_x, pad_y)
    
    max_count = np.max(grid)
    print(f"Max Density: {max_count} particles per pixel")

    # 6. Plotting Setup
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert to float for plotting
    grid_float = grid.astype(np.float32)
    
    # MASKING: Make empty pixels transparent (or white)
    # We use a masked array so 0 values don't get plotted
    grid_masked = np.ma.masked_where(grid == 0, grid_float)
    
    # 7. Apply Modes
    if mode == "binary":
        # Binary: Any pixel > 0 is treated as 1.0
        # We set vmin=0, vmax=1, so all occupied pixels get the "max" color
        norm = mcolors.Normalize(vmin=0, vmax=0.1) 
        # (vmax=0.1 ensures even a count of 1 is fully saturated color)
        title_mode = "Binary (Occupancy)"
        
    elif mode == "log":
        # Logarithmic: Good for seeing dynamic range
        norm = mcolors.LogNorm(vmin=1, vmax=max_count)
        title_mode = "Logarithmic Density"
        
    elif mode == "linear":
        # Linear: Shows true mass concentration
        norm = mcolors.Normalize(vmin=0, vmax=max_count)
        title_mode = "Linear Density"
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Plot Image
    im = ax.imshow(
        grid_masked, 
        cmap=cmap, 
        norm=norm, 
        interpolation='nearest', # Keep pixels sharp (no blurring)
        origin='lower'
    )
    
    # Add Colorbar only if not binary (binary bar is boring)
    if mode != "binary":
        plt.colorbar(im, label="Particles per Pixel")
    
    ax.axis('off')
    ax.set_title(f"Coarse Grained DLA ({title_mode})\n$2^{{{res_power}}}\\times2^{{{res_power}}}$ Grid | Grain $\\approx$ {grain_size:.1f}")
    
    if output_path:
        # Save with high DPI
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Coarse-Grained Density Plotter")
    parser.add_argument("file", help="Input .npz file")
    parser.add_argument("--res-power", type=int, default=11, help="Resolution power of 2 (default 11 -> 2048px)")
    parser.add_argument("--mode", choices=["binary", "log", "linear"], default="log", help="Plotting mode: binary, log, or linear")
    parser.add_argument("--cmap", default="Greys", help="Matplotlib colormap (default: Greys)")
    parser.add_argument("--out", default=None, help="Output filename")

    args = parser.parse_args()
    
    # Load
    result = utils.load_cluster(args.file)
    
    # Auto-name output if not provided
    if args.out is None:
        input_path = Path(args.file)
        out_name = input_path.stem + f"_density_{args.mode}.png"
        out_path = input_path.parent / out_name
    else:
        out_path = args.out
        
    render_density(result, out_path, args.res_power, args.mode, args.cmap)

if __name__ == "__main__":
    main()