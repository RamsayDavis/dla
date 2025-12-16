# src/scripts/plot_cluster.py
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim import utils  # type: ignore[import]


def format_title(meta):
    """
    Format a title string with important statistics from metadata.
    
    Args:
        meta: Dictionary containing model metadata
        
    Returns:
        Formatted title string
    """
    if not meta:
        return None
    
    model = meta.get("model", "?")
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


def render_lattice(x_coords, y_coords, grid_size, title=None, cmap="magma", save_path=None, show=False, dpi=300):
    """
    Render lattice-based models using imshow with strict pixel-perfect rendering.
    
    Args:
        x_coords: Integer x coordinates
        y_coords: Integer y coordinates
        grid_size: Initial grid size (may be recomputed based on data)
        title: Optional title
        cmap: Colormap name
        save_path: Path to save figure
        show: Whether to show interactively
        dpi: Base DPI (will be adjusted to ensure 1 pixel = 1 data point)
    """
    # --- Trailing zeros / artifact fix for lattice-style coordinates ---
    # Many lattice exporters (especially Koh) leave trailing (0, 0) entries
    # after the cluster has finished growing. These can:
    #   - artificially inflate the inferred grid size
    #   - distort the apparent center/extent of the cluster
    #
    # We treat (0, 0) as an artifact and drop those points before computing
    # the final grid geometry.
    if len(x_coords) > 0:
        x_coords_arr = np.asarray(x_coords, dtype=np.int32)
        y_coords_arr = np.asarray(y_coords, dtype=np.int32)
        
        # Keep only points that are not exactly at the origin
        valid_mask = (x_coords_arr != 0) | (y_coords_arr != 0)
        x_coords_arr = x_coords_arr[valid_mask]
        y_coords_arr = y_coords_arr[valid_mask]
        
        # If everything was filtered out, fall back to the original inputs
        if len(x_coords_arr) == 0:
            x_coords_arr = np.asarray(x_coords, dtype=np.int32)
            y_coords_arr = np.asarray(y_coords, dtype=np.int32)
        
        # Recalculate grid size based on the actual data range (ignoring the
        # passed-in grid_size which may have been influenced by zeros).
        if len(x_coords_arr) > 0:
            x_min, x_max = x_coords_arr.min(), x_coords_arr.max()
            y_min, y_max = y_coords_arr.min(), y_coords_arr.max()
            x_range = int(x_max - x_min)
            y_range = int(y_max - y_min)
            max_range = max(x_range, y_range)
            
            # Recompute a strict grid size with ~10% padding
            new_grid_size = int(max_range * 1.1) + 1
            # Round up to next power of 2 for efficiency / FFT-friendliness
            grid_size = 1 << (new_grid_size - 1).bit_length()
            # Enforce a reasonable minimum for high-res output
            grid_size = max(grid_size, 512)
            
            # Center the cluster in the new grid by computing offsets
            # NOTE: x_range/y_range are extents in data units; we center the
            # bounding box of the occupied points inside the [0, grid_size) box.
            x_offset = x_min - (grid_size - x_range) // 2
            y_offset = y_min - (grid_size - y_range) // 2
            
            # Shift coordinates into local grid frame
            x_coords_local = x_coords_arr - x_offset
            y_coords_local = y_coords_arr - y_offset
        else:
            # Degenerate case: no valid coordinates
            x_coords_local = np.asarray(x_coords, dtype=np.int32)
            y_coords_local = np.asarray(y_coords, dtype=np.int32)
    else:
        x_coords_local = np.asarray(x_coords, dtype=np.int32)
        y_coords_local = np.asarray(y_coords, dtype=np.int32)
    
    # Allocate the grid based on the (possibly recomputed) grid_size
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    num_particles = len(x_coords_local)
    
    if num_particles > 0:
        ages = np.linspace(0, 1, num_particles, dtype=np.float32)
        
        # Convert to integer coordinates and clamp to valid range
        x_coords_int = np.asarray(x_coords_local, dtype=np.int32)
        y_coords_int = np.asarray(y_coords_local, dtype=np.int32)
        
        # Clamp to valid range
        x_coords_int = np.clip(x_coords_int, 0, grid_size - 1)
        y_coords_int = np.clip(y_coords_int, 0, grid_size - 1)
        
        grid[x_coords_int, y_coords_int] = ages
    
    # Calculate DPI to guarantee 1 pixel = 1 data point
    base_size_inches = 6
    required_dpi = grid_size / base_size_inches
    final_dpi = max(dpi, required_dpi)
    
    fig, ax = plt.subplots(figsize=(base_size_inches, base_size_inches))
    
    # Set background color to white
    bg_color = "white"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Handle special "black" colormap for plain black rendering
    if cmap.lower() == "black":
        # Create a colormap that maps all values to black
        black_cmap = mcolors.ListedColormap(['black'])
        im = ax.imshow(grid.T, origin="lower", cmap=black_cmap, interpolation="none", vmin=0.0, vmax=1.0)
    else:
        # Use interpolation='none' to preserve sharp fractal pixels
        im = ax.imshow(grid.T, origin="lower", cmap=cmap, interpolation="none", vmin=0.0, vmax=1.0)
    
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, pad=10)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=final_dpi, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
        print(f"Saved figure to {save_path} (Grid: {grid_size}x{grid_size}, DPI: {final_dpi:.0f})")
    
    if show:
        plt.show()
    plt.close(fig)


def render_continuous(x, y, particle_radius, title=None, cmap="magma", save_path=None, show=False, dpi=300):
    """
    Render continuous models using CircleCollection for exact particle rendering.
    
    Args:
        x: Array of x coordinates (float)
        y: Array of y coordinates (float)
        particle_radius: Radius of particles in data units
        title: Optional title
        cmap: Colormap name
        save_path: Path to save figure
        show: Whether to show interactively
        dpi: DPI for saving
    """
    num_particles = len(x)
    if num_particles == 0:
        print("No particles to render")
        return
    
    # Create age gradient (0 to 1) for coloring
    ages = np.linspace(0, 1, num_particles)
    
    # Calculate bounds with padding
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = max(x_range, y_range) * 0.1
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Set background color to white
    bg_color = "white"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Calculate circle sizes in points^2
    # CircleCollection sizes are in points^2, where 1 point = 1/72 inch
    # We need to convert particle_radius from data units to points
    
    # Estimate based on figure size and data range
    # The figure is 6 inches, and we need to account for bbox_inches="tight" padding
    # A reasonable estimate: assume ~5.5 inches of usable space after padding
    usable_size_inches = 5.5
    
    # Calculate the data range (use the larger dimension for scaling)
    data_range = max(x_max - x_min, y_max - y_min)
    
    # Convert radius from data units to inches
    # Account for the fact that the figure will be scaled to fit the data
    radius_inches = (particle_radius / data_range) * usable_size_inches
    
    # Convert inches to points (1 inch = 72 points)
    radius_points = radius_inches * 72.0
    
    # CircleCollection sizes are in points^2
    # Size = (diameter)^2 = (2 * radius)^2
    # Ensure minimum size for visibility
    sizes = max((2 * radius_points) ** 2, 1.0)
    
    # Create CircleCollection
    offsets = np.column_stack([x, y])
    
    # Handle special "black" colormap for plain black rendering
    if cmap.lower() == "black":
        # Use black color for all particles (no gradient)
        collection = mcollections.CircleCollection(
            sizes=[sizes] * num_particles,  # Same size for all particles
            offsets=offsets,
            transOffset=ax.transData,
            facecolors='black',  # All particles black
            rasterized=True,  # Critical for large N to avoid memory issues
        )
    else:
        collection = mcollections.CircleCollection(
            sizes=[sizes] * num_particles,  # Same size for all particles
            offsets=offsets,
            transOffset=ax.transData,
            cmap=cmap,
            array=ages,  # Color by age
            rasterized=True,  # Critical for large N to avoid memory issues
        )
        # Set color limits
        collection.set_clim(0.0, 1.0)
    
    ax.add_collection(collection)
    
    if title:
        ax.set_title(title, pad=10)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
        print(f"Saved figure to {save_path} (N={num_particles}, DPI: {dpi})")
    
    if show:
        plt.show()
    plt.close(fig)


def plot_occupied(occupied, title=None, cmap="binary", origin="lower", save_path=None, show=False, dpi=300):
    """Legacy function for backward compatibility."""
    fig, ax = plt.subplots(figsize=(6,6))
    # Set background color to white
    bg_color = "white"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Handle special "black" colormap for plain black rendering
    if cmap.lower() == "black":
        black_cmap = mcolors.ListedColormap(['black'])
        im = ax.imshow(occupied.T if origin == "lower" else occupied,
                       origin=origin, cmap=black_cmap, interpolation="nearest")
    else:
        im = ax.imshow(occupied.T if origin == "lower" else occupied,
                       origin=origin, cmap=cmap, interpolation="nearest")
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, pad=10)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_positions(positions, title=None, cmap="viridis", save_path=None, show=False, dpi=300, s=3):
    """Legacy function for backward compatibility."""
    fig, ax = plt.subplots(figsize=(6,6))
    # Set background color to white
    bg_color = "white"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    xs = positions[:, 0]
    ys = positions[:, 1]
    # Handle special "black" colormap for plain black rendering
    if cmap.lower() == "black":
        ax.scatter(xs, ys, s=s, c='black', marker=".")
    else:
        ax.scatter(xs, ys, s=s, c=np.hypot(xs, ys), cmap=cmap, marker=".")
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, pad=10)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_cluster_from_result(result, save_path=None, show=False, cmap="magma", dpi=300):
    """
    Plot a cluster from a ClusterResult object.
    """
    occupied = result.occupied
    positions = result.positions
    meta = result.meta or {}
    title = format_title(meta)
    
    # Check model type and route to appropriate renderer
    model_type = meta.get("model", "").lower() if meta else ""
    has_coords = (meta and "x_coords" in meta and "y_coords" in meta)
    
    if model_type == "continuous" and has_coords:
        # Continuous model: use CircleCollection
        x_coords = np.asarray(meta["x_coords"])
        y_coords = np.asarray(meta["y_coords"])
        particle_radius = meta.get("particle_radius", 0.5)  # Default from continuous_dla.py
        render_continuous(x_coords, y_coords, particle_radius, title=title, cmap=cmap, 
                         save_path=save_path, show=show, dpi=dpi)
    elif model_type in ("koh", "lattice","koh_optimized") and has_coords:
        # Lattice model: use imshow with interpolation='none'
        x_coords = np.asarray(meta["x_coords"])
        y_coords = np.asarray(meta["y_coords"])
        # Determine grid size
        if occupied is not None:
            grid_size = occupied.shape[0]
        else:
            # Estimate from coordinate bounds
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_range = x_coords.max() - x_coords.min()
                y_range = y_coords.max() - y_coords.min()
                max_range = max(x_range, y_range)
                # Add padding and round up to next power of 2
                grid_size = int(max_range * 1.2) + 1
                grid_size = 1 << (grid_size - 1).bit_length()
                # Ensure minimum size
                grid_size = max(grid_size, 256)
            else:
                grid_size = 256  # default
        render_lattice(x_coords, y_coords, grid_size, title=title, cmap=cmap, 
                      save_path=save_path, show=show, dpi=dpi)
    elif positions is not None:
        # Fallback to legacy scatter plot
        plot_positions(positions, title=title, cmap=cmap, save_path=save_path, show=show, s=3)
    elif occupied is not None:
        # Fallback to legacy occupied plot
        plot_occupied(occupied, title=title, cmap=cmap, save_path=save_path, show=show, dpi=dpi)
    else:
        print("No recognized data in result to plot.")


def main():
    parser = argparse.ArgumentParser(description="Plot a saved DLA cluster .npz")
    parser.add_argument("--file", default="results/cluster.npz", help="Path to .npz cluster")
    parser.add_argument("--out", default=None, help="Output image path (PNG, auto-generated if not provided)")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap (for age heatmap)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--pointsize", type=float, default=3.0, help="point size for scatter (positions)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        return

    # Auto-generate output filename if not provided
    if args.out is None:
        input_path = Path(args.file)
        # Extract base name (without .npz extension)
        base_name = input_path.stem
        # Remove existing colormap suffix if present (e.g., remove "_magma" from "file_magma")
        # This handles cases where the file was already plotted with a different colormap
        known_colormaps = ["magma", "plasma", "inferno", "viridis", "cividis", "turbo", "hot", "cool", "jet"]
        for cmap in known_colormaps:
            if base_name.endswith(f"_{cmap}"):
                base_name = base_name[:-len(f"_{cmap}")]
                break
        # Generate output path with colormap suffix
        output_path = input_path.parent / f"{base_name}_{args.cmap}.png"
        args.out = str(output_path)

    result = utils.load_cluster(args.file)
    occupied = result.occupied
    positions = result.positions
    meta = result.meta or {}
    title = format_title(meta)
    
    # Check model type and route to appropriate renderer
    model_type = meta.get("model", "").lower() if meta else ""
    
    # Check for coordinate data
    has_coords = (meta and "x_coords" in meta and "y_coords" in meta)
    
    if model_type == "continuous" and has_coords:
        # Continuous model: use CircleCollection
        x_coords = np.asarray(meta["x_coords"])
        y_coords = np.asarray(meta["y_coords"])
        particle_radius = meta.get("particle_radius", 0.5)  # Default from continuous_dla.py
        render_continuous(x_coords, y_coords, particle_radius, title=title, cmap=args.cmap, 
                         save_path=args.out, show=args.show, dpi=300)
    elif model_type in ("koh", "lattice","koh_optimized") and has_coords:
        # Lattice model: use imshow with interpolation='none'
        x_coords = np.asarray(meta["x_coords"])
        y_coords = np.asarray(meta["y_coords"])
        # Determine grid size
        if occupied is not None:
            grid_size = occupied.shape[0]
        else:
            # Estimate from coordinate bounds
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_range = x_coords.max() - x_coords.min()
                y_range = y_coords.max() - y_coords.min()
                max_range = max(x_range, y_range)
                # Add padding and round up to next power of 2
                grid_size = int(max_range * 1.2) + 1
                grid_size = 1 << (grid_size - 1).bit_length()
                # Ensure minimum size
                grid_size = max(grid_size, 256)
            else:
                grid_size = 256  # default
        render_lattice(x_coords, y_coords, grid_size, title=title, cmap=args.cmap, 
                      save_path=args.out, show=args.show, dpi=300)
    elif positions is not None:
        # Fallback to legacy scatter plot
        plot_positions(positions, title=title, cmap=args.cmap, save_path=args.out, show=args.show, s=args.pointsize)
    elif occupied is not None:
        # Fallback to legacy occupied plot
        plot_occupied(occupied, title=title, cmap=args.cmap, save_path=args.out, show=args.show)
    else:
        print("No recognized data in file to plot.")


if __name__ == "__main__":
    main()
