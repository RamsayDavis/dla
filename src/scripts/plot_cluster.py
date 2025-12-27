# src/scripts/plot_cluster.py
import argparse
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim import utils  # type: ignore[import]


@njit(parallel=True, cache=True)
def render_grid_numba(x_coords, y_coords, ages, grid, scale, pad_x, pad_y, radius=0.5):
    """
    Numba-accelerated rasterizer that paints circles onto a grid.
    
    Args:
        x_coords: Array of x coordinates (float)
        y_coords: Array of y coordinates (float)
        ages: Array of ages (0.0-1.0) for coloring
        grid: 2D array to write to (will be modified in-place)
        scale: Scale factor to convert data units to pixels
        pad_x: X offset to center data in grid
        pad_y: Y offset to center data in grid
        radius: Particle radius in data units (default 0.5)
    """
    H, W = grid.shape
    num_particles = len(x_coords)
    r_px = radius * scale
    
    # Optimization: if radius is less than 0.5 pixels, just set single pixel
    if r_px < 0.75:
        for i in prange(num_particles):
            px = int((x_coords[i] - pad_x) * scale)
            py = int((y_coords[i] - pad_y) * scale)
            
            if 0 <= px < W and 0 <= py < H:
                # Only overwrite if new age is greater (or if current is NaN)
                current = grid[py, px]
                if np.isnan(current) or ages[i] > current:
                    grid[py, px] = ages[i]
    else:
        # Full circle rendering: iterate bounding box
        r_px_sq = r_px * r_px
        r_int = int(np.ceil(r_px)) + 1  # Integer radius for bounding box
        
        for i in prange(num_particles):
            px_center = (x_coords[i] - pad_x) * scale
            py_center = (y_coords[i] - pad_y) * scale
            
            px_min = int(np.floor(px_center - r_int))
            px_max = int(np.ceil(px_center + r_int)) + 1
            py_min = int(np.floor(py_center - r_int))
            py_max = int(np.ceil(py_center + r_int)) + 1
            
            # Clamp to grid bounds
            px_min = max(0, px_min)
            px_max = min(W, px_max)
            py_min = max(0, py_min)
            py_max = min(H, py_max)
            
            # Fill pixels within circle
            for py in range(py_min, py_max):
                for px in range(px_min, px_max):
                    dx = px - px_center
                    dy = py - py_center
                    dist_sq = dx * dx + dy * dy
                    
                    if dist_sq <= r_px_sq:
                        # Only overwrite if new age is greater (or if current is NaN)
                        current = grid[py, px]
                        if np.isnan(current) or ages[i] > current:
                            grid[py, px] = ages[i]


def format_title(meta, num_particles=None):
    """
    Format a title string with important statistics from metadata.
    
    Args:
        meta: Dictionary containing model metadata
        num_particles: Optional particle count to use (overrides metadata)
        
    Returns:
        Formatted title string
    """
    if not meta:
        return None
    
    model = meta.get("model", "?")
    # Use provided count, or try metadata, or use "?"
    if num_particles is not None:
        num = num_particles
    else:
        num = meta.get("num", "?")
    seed = meta.get("seed")
    seed_str = str(seed) if seed is not None else "?"
    
    parts = [f"Model={model}", f"N={num}", f"seed={seed_str}"]
    
    # Add model-specific parameters
    if model == "continuous":
        pr = meta.get("particle_radius")
        if pr is not None:
            parts.append(f"PR={pr:.2f}")
    elif model == "koh":
        lmax = meta.get("lmax")
        if lmax is not None:
            parts.append(f"L={lmax}")
    elif model == "lattice":
        radius = meta.get("radius")
        if radius is not None:
            parts.append(f"R={radius}")
    elif model == "offlattice":
        rb = meta.get("Rb")
        rd = meta.get("Rd")
        if rb is not None:
            parts.append(f"Rb={rb:.1f}")
        if rd is not None:
            parts.append(f"Rd={rd:.1f}")
    
    return " | ".join(parts)


def get_coordinates_from_result(result):
    """
    Extract x and y coordinates from a ClusterResult, handling all formats.
    
    Returns:
        tuple: (x_coords, y_coords) as numpy arrays
    """
    meta = result.meta or {}
    
    # Check for split coordinates (preferred format)
    if "x_coords" in meta and "y_coords" in meta:
        x = np.asarray(meta["x_coords"], dtype=np.float64)
        y = np.asarray(meta["y_coords"], dtype=np.float64)
        return x, y
    
    # Check for positions array
    if result.positions is not None:
        pos = np.asarray(result.positions)
        
        # Case A: Complex Numbers (Bell / Continuous DLA)
        if np.iscomplexobj(pos):
            x = pos.real.astype(np.float64)
            y = pos.imag.astype(np.float64)
            return x, y
            
        # Case B: Standard (N, 2) array
        if pos.ndim == 2 and pos.shape[1] >= 2:
            x = pos[:, 0].astype(np.float64)
            y = pos[:, 1].astype(np.float64)
            return x, y
    
    raise ValueError(
        "Could not find valid coordinates. Checked: meta['x_coords'], positions(complex), positions(N,2)."
    )


def render(positions, title=None, output=None, cmap="magma", dpi=300, res=2048, auto_res=False):
    """
    Unified high-performance rasterizer for DLA clusters.
    
    Args:
        positions: Tuple of (x_coords, y_coords) arrays, or ClusterResult
        title: Optional title string
        output: Output file path (None to skip saving)
        cmap: Matplotlib colormap name
        dpi: DPI for output (affects file size, not internal resolution)
        res: Output image width in pixels (ignored if auto_res=True)
        auto_res: If True, calculates resolution automatically (2 pixels per particle diameter)
    """
    # Handle ClusterResult input
    if hasattr(positions, 'meta') or hasattr(positions, 'positions'):
        result = positions
        x, y = get_coordinates_from_result(result)
    else:
        # Assume it's a tuple of (x, y) arrays
        x, y = positions
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    num_particles = len(x)
    if num_particles == 0:
        print("No particles to render")
        return
    
    # Filter out any NaN or Inf values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    num_particles = len(x)
    
    if num_particles == 0:
        print("No valid particles to render")
        return
    
    # Calculate bounding box
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    W_data = x_max - x_min
    H_data = y_max - y_min
    
    # Use larger dimension for square aspect ratio
    max_dim = max(W_data, H_data)
    
    # Handle edge case: if all particles are at the same point
    if max_dim == 0.0:
        max_dim = 1.0  # Use a default minimum size
    
    # Resolution logic
    if auto_res:
        # Ensure ~2 pixels per particle diameter (radius=0.5, so diameter=1.0)
        # Scale based on data width
        res = int(max_dim * 2.0)
        # Clamp to reasonable bounds
        res = max(512, min(res, 16384))
        print(f"Auto-resolution: {res}x{res} pixels (ensures 2px per particle diameter)")
    else:
        # Use provided resolution
        res = int(res)
        res = max(256, min(res, 16384))  # Clamp to reasonable bounds
    
    # Calculate scale: fit to width with 5% padding
    padding_factor = 1.05
    scale = res / (max_dim * padding_factor)
    
    # Normalization: center coordinates relative to the grid
    # We want the data centered in the grid
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    # Calculate offsets to center data
    pad_x = center_x - (max_dim * padding_factor) / 2.0
    pad_y = center_y - (max_dim * padding_factor) / 2.0
    
    # Allocate grid (H x W, where H=W=res for square)
    H = res
    W = res
    grid = np.full((H, W), np.nan, dtype=np.float32)
    
    # Create age gradient (0 to 1) for coloring
    ages = np.linspace(0.0, 1.0, num_particles, dtype=np.float32)
    
    # Run Numba kernel
    print(f"Rasterizing {num_particles:,} particles onto {W}x{H} grid...")
    render_grid_numba(x, y, ages, grid, scale, pad_x, pad_y, radius=0.5)
    
    # Count non-NaN pixels for info
    num_occupied = np.sum(~np.isnan(grid))
    print(f"Rendered {num_occupied:,} pixels ({100.0 * num_occupied / (H * W):.2f}% fill)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Set background color to white
    bg_color = "white"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Handle special "black" colormap
    if cmap.lower() == "black":
        black_cmap = mcolors.ListedColormap(['black'])
        im = ax.imshow(grid, interpolation='nearest', origin='lower', cmap=black_cmap, vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(grid, interpolation='nearest', origin='lower', cmap=cmap, vmin=0.0, vmax=1.0)
    
    ax.set_aspect("equal")
    ax.axis("off")
    
    if title:
        ax.set_title(title, pad=10)
    
    if output:
        os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
        plt.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
        print(f"Saved figure to {output} ({W}x{H} @ {dpi} DPI)")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a saved DLA cluster .npz using high-performance rasterizer"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="results/cluster.npz",
        help="Path to .npz cluster file"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (PNG, auto-generated if not provided)"
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap (default: magma)"
    )
    parser.add_argument(
        "--res",
        type=int,
        default=2048,
        help="Output image width in pixels (default: 2048, ignored if --full-res is set)"
    )
    parser.add_argument(
        "--full-res",
        action="store_true",
        help="Auto-calculate resolution to ensure 2 pixels per particle diameter (ignores --res)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output file (default: 300, affects file size, not internal resolution)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively (not recommended for large clusters)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        return

    # Auto-generate output filename if not provided
    if args.out is None:
        input_path = Path(args.file)
        base_name = input_path.stem
        # Remove existing colormap suffix if present
        known_colormaps = [
            "magma", "plasma", "inferno", "viridis", "cividis",
            "turbo", "hot", "cool", "jet", "black"
        ]
        for cmap in known_colormaps:
            if base_name.endswith(f"_{cmap}"):
                base_name = base_name[:-len(f"_{cmap}")]
                break
        # Generate output path with colormap suffix
        output_path = input_path.parent / f"{base_name}_{args.cmap}.png"
        args.out = str(output_path)

    # Load cluster
    result = utils.load_cluster(args.file)
    meta = result.meta or {}
    
    # Calculate actual particle count from coordinates
    try:
        x, y = get_coordinates_from_result(result)
        num_particles = len(x)
    except (ValueError, AttributeError):
        # Fallback: try to get count from metadata or positions
        if result.positions is not None:
            num_particles = len(result.positions)
        else:
            num_particles = None
    
    title = format_title(meta, num_particles=num_particles)
    
    # Render using unified rasterizer
    render(
        positions=result,
        title=title,
        output=args.out,
        cmap=args.cmap,
        dpi=args.dpi,
        res=args.res,
        auto_res=args.full_res
    )


if __name__ == "__main__":
    main()
