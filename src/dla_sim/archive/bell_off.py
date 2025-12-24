import time
from dataclasses import dataclass

import numpy as np
from numba import njit
from numba.typed import List

from . import utils


PI = 3.1415926535
TWO_PI = 6.2831853072
CX_1 = complex(1.0, 0.0)
CX_0 = complex(0.0, 0.0)
CX_J = complex(0.0, 1.0)


@dataclass
class BellOffParams:
    """Configuration for Bell off-lattice Fast Grid DLA."""

    num_particles: int = 10_000
    noise_reduction_factor: float = 1.0
    max_min_mesh: float = 0.0
    scale_of_points_grid: float = 0.0
    seed: int | None = None


@njit
def rand_circ():
    """
    Returns a complex number uniformly on the unit circle.
    """
    theta = np.random.uniform(0.0, TWO_PI)
    return complex(np.sin(theta), np.cos(theta))


@njit
def harmonic_to_circle(abs_pos):
    """
    Maps the harmonic measure from infinity to the circle.
    """
    z = rand_circ()
    z = (z - CX_1) / (z + CX_1)
    z = z * ((abs_pos - 1.0) / (abs_pos + 1.0))
    z = -(z + CX_1) / (z - CX_1)
    return z


@njit
def get_indices(curr_point, max_radius, mesh):
    """
    Converts a complex coordinate into grid indices.
    """
    index1 = int(np.floor((max_radius - curr_point.imag) / mesh))
    index2 = int(np.floor((max_radius + curr_point.real) / mesh))
    return index1, index2


# --- STATIC CHAINING GRID FUNCTIONS ---


@njit
def add_to_points_grid_static(curr_point, max_radius,
                              particle_idx, points_grid_size, points_grid_mesh,
                              grid_head, particle_next):
    """
    Adds a particle to the spatial hash using static array chaining.
    """
    idx1, idx2 = get_indices(curr_point, max_radius, points_grid_mesh)

    # Clamp indices to grid bounds
    if idx1 < 0:
        idx1 = 0
    elif idx1 >= points_grid_size:
        idx1 = points_grid_size - 1

    if idx2 < 0:
        idx2 = 0
    elif idx2 >= points_grid_size:
        idx2 = points_grid_size - 1

    flat_idx = idx1 * points_grid_size + idx2

    # 1. Point this particle to the previous head of the list
    particle_next[particle_idx] = grid_head[flat_idx]

    # 2. Make this particle the new head
    grid_head[flat_idx] = particle_idx


@njit
def check_for_closer_static(i, j, curr_point, points,
                            grid_head, particle_next, points_grid_size,
                            nearest, dist_to_nearest2, max_safe_dist2):
    """
    Iterates through the linked list in the grid cell (i, j).
    """
    flat_idx = i * points_grid_size + j

    # Start at the head of the list
    p_idx = grid_head[flat_idx]

    # Traverse until end (-1)
    while p_idx != -1:
        diff = curr_point - points[p_idx]
        dist2 = (diff.real * diff.real + diff.imag * diff.imag)

        if dist2 < max_safe_dist2:
            if dist2 < dist_to_nearest2:
                nearest = points[p_idx]
                max_safe_dist2 = dist_to_nearest2
                dist_to_nearest2 = dist2
            else:
                max_safe_dist2 = dist2

        p_idx = particle_next[p_idx]

    return nearest, dist_to_nearest2, max_safe_dist2


@njit
def find_nearest_static(curr_point, max_radius,
                        points, grid_head, particle_next,
                        points_grid_size, points_grid_mesh):
    """
    Checks the 3x3 neighborhood using the static grid arrays.
    """
    nearest = CX_0
    max_safe_dist2 = points_grid_mesh * points_grid_mesh
    dist_to_nearest2 = max_safe_dist2

    idx1, idx2 = get_indices(curr_point, max_radius, points_grid_mesh)

    i_min = max(idx1 - 1, 0)
    i_max = min(idx1 + 2, points_grid_size)
    j_min = max(idx2 - 1, 0)
    j_max = min(idx2 + 2, points_grid_size)

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            nearest, dist_to_nearest2, max_safe_dist2 = check_for_closer_static(
                i, j, curr_point, points,
                grid_head, particle_next, points_grid_size,
                nearest, dist_to_nearest2, max_safe_dist2
            )

    return nearest, dist_to_nearest2, max_safe_dist2


@njit
def is_marked_at_layer(curr_point, max_radius,
                       layer, layer_sizes, layer_meshes, layers):
    mesh = layer_meshes[layer]
    size = layer_sizes[layer]
    idx1, idx2 = get_indices(curr_point, max_radius, mesh)

    if idx1 < 0 or idx1 >= size or idx2 < 0 or idx2 >= size:
        return False

    flat_idx = idx1 * size + idx2
    return layers[layer][flat_idx] != 0


@njit
def find_best_layer(curr_point, max_radius, layer_count,
                    layer_sizes, layer_meshes, layers):
    """
    Binary search through hierarchy grids.
    """
    lower_bound = 0
    upper_bound = layer_count
    while lower_bound != upper_bound:
        midpoint = (lower_bound + upper_bound) // 2
        if is_marked_at_layer(curr_point, max_radius,
                              midpoint, layer_sizes, layer_meshes, layers):
            lower_bound = midpoint + 1
        else:
            upper_bound = midpoint
    return lower_bound


@njit
def set_marks(curr_point, max_radius,
              layer, layer_sizes, layer_meshes, layers):
    mesh = layer_meshes[layer]
    size = layer_sizes[layer]
    idx1, idx2 = get_indices(curr_point, max_radius, mesh)

    i_min = max(idx1 - 1, 0)
    i_max = min(idx1 + 2, size)
    j_min = max(idx2 - 1, 0)
    j_max = min(idx2 + 2, size)

    grid = layers[layer]

    for i in range(i_min, i_max):
        base = i * size
        for j in range(j_min, j_max):
            grid[base + j] = 1


@njit
def mark_particle(curr_point, max_radius,
                  layer_count, layer_sizes, layer_meshes, layers):
    for layer in range(layer_count - 1, -1, -1):
        set_marks(curr_point, max_radius, layer,
                  layer_sizes, layer_meshes, layers)


@njit
def finishing_step(curr_point, nearest, dist_to_nearest2, max_safe_dist2,
                   noise_reduction_factor):
    """
    Exact adhering step using conformal mapping (Bell thesis, Sec. 4.3).
    """
    backup = curr_point

    d1 = np.sqrt(dist_to_nearest2) / 2.0
    d2 = np.sqrt(max_safe_dist2) / 2.0

    if d2 <= 1.0 or d1 <= 0.0:
        return curr_point, False

    theta = np.arccos((1.0 + (d2 - 1.0) * (d2 - 1.0) - d1 * d1) /
                      (2.0 * (d2 - 1.0)))
    r2 = ((d2 - 1.0) * (d2 - 1.0) + d1 * d1 - 1.0) / (2.0 * d1)
    r1 = np.sqrt(1.0 - (d1 - r2) * (d1 - r2))

    alpha = complex(r1, -r2)
    beta = complex(-r1, -r2)
    D = (d2 - 1.0 - beta) / (d2 - 1.0 - alpha)

    y1 = (alpha * D / beta) ** (PI / theta)
    y2 = y1.real + y1.imag * np.random.standard_cauchy()

    y3 = complex(y2, 0.0) ** (theta / PI)
    y4 = (-beta * y3 + D * alpha) / (-y3 + D)

    curr_point = curr_point + (CX_J * y4 * (nearest - curr_point) / d1)
    particle_free = (y2 >= 0.0)

    if (curr_point == backup) or (np.isnan(curr_point.real) and max_safe_dist2 < 4.01):
        curr_point = backup
        particle_free = False

    if noise_reduction_factor != 1.0 and (not particle_free):
        curr_point = (noise_reduction_factor * curr_point +
                      nearest * (1.0 - noise_reduction_factor))

    return curr_point, particle_free


@njit
def reset_particle(curr_point, start_dist):
    """
    Exact resetting using harmonic measure (Bell thesis, Sec. 4.2).
    """
    ratio_out = np.abs(curr_point) / start_dist
    z = harmonic_to_circle(ratio_out)
    curr_point = curr_point * (z / ratio_out)
    return curr_point


@njit
def step(curr_point, length):
    curr_point = curr_point + length * rand_circ()
    return curr_point


@njit
def update_start_dist(curr_point, start_dist, max_radius):
    start_dist = max(start_dist, np.abs(curr_point) + 2.0)
    if start_dist > max_radius:
        start_dist = max_radius
    return start_dist


@njit
def aggregate_loop(
    num_to_add,
    max_radius, noise_reduction_factor,
    layer_count, layer_sizes, layer_meshes, layers,
    points_grid_size, points_grid_mesh,
    grid_head, particle_next, points
):
    """
    Main aggregation loop using static chaining.
    """
    points_added = 2
    start_dist = 4.0
    curr_point = CX_0

    for _ in range(num_to_add):
        curr_point = start_dist * rand_circ()
        particle_free = True

        while particle_free:
            if np.abs(curr_point) > start_dist + 1e-5:
                curr_point = reset_particle(curr_point, start_dist)
            else:
                best = find_best_layer(curr_point, max_radius,
                                       layer_count, layer_sizes,
                                       layer_meshes, layers)
                if best == layer_count:
                    nearest, dist2, max_safe2 = find_nearest_static(
                        curr_point, max_radius,
                        points, grid_head, particle_next,
                        points_grid_size, points_grid_mesh
                    )
                    if dist2 < points_grid_mesh * points_grid_mesh:
                        curr_point, particle_free = finishing_step(
                            curr_point, nearest, dist2, max_safe2,
                            noise_reduction_factor
                        )
                    else:
                        curr_point = step(curr_point, points_grid_mesh - 2.0)
                else:
                    curr_point = step(curr_point, layer_meshes[best] - 2.0)

        points[points_added] = curr_point

        # Optimized static grid add
        add_to_points_grid_static(
            curr_point, max_radius,
            points_added, points_grid_size, points_grid_mesh,
            grid_head, particle_next
        )

        mark_particle(curr_point, max_radius,
                      layer_count, layer_sizes, layer_meshes, layers)
        points_added += 1
        start_dist = update_start_dist(curr_point, start_dist, max_radius)

    return points_added, start_dist


def fast_dla(number_of_particles,
             seed=0,
             noise_reduction_factor=1.0,
             max_min_mesh=0.0,
             scale_of_points_grid=0.0):
    """
    High-level API: off-lattice Bell Fast Grid DLA using static-chained
    spatial grid and hierarchical occupancy grids.
    """
    if max_min_mesh == 0.0:
        max_min_mesh = 6.0 + noise_reduction_factor * 18.0

    if scale_of_points_grid == 0.0:
        scale_of_points_grid = 1.0 + noise_reduction_factor

    if scale_of_points_grid < 1.0:
        raise ValueError("scaleOfPointsGrid must be greater than 1")

    if noise_reduction_factor == 1.0:
        max_radius = 22.0 + 2.2 * (number_of_particles ** (1.0 / 1.7))
    else:
        max_radius = 20.0 + 4.0 * (
            (number_of_particles * noise_reduction_factor) ** (1.0 / 1.7)
        )

    # Hierarchy setup
    layer_count = 1 + int(np.ceil(np.log2(max_radius / max_min_mesh)))
    layer_sizes = np.empty(layer_count, dtype=np.int32)
    layer_meshes = np.empty(layer_count, dtype=np.float64)

    layer_sizes[0] = 2
    for i in range(layer_count):
        if i > 0:
            layer_sizes[i] = layer_sizes[i - 1] * 2
        layer_meshes[i] = 2.0 * max_radius / layer_sizes[i]

    layers = List()
    for i in range(layer_count):
        size = layer_sizes[i] * layer_sizes[i]
        arr = np.zeros(size, dtype=np.uint8)
        layers.append(arr)

    # Static Points Grid Setup
    points_grid_size = int(
        np.floor(2.0 * max_radius / (scale_of_points_grid * layer_meshes[layer_count - 1]))
    )
    if points_grid_size <= 0:
        points_grid_size = 1

    points_grid_mesh = 2.0 * max_radius / points_grid_size

    # ALLOCATE STATIC ARRAYS (optimization versus per-cell typed lists)
    total_cells = points_grid_size * points_grid_size
    grid_head = np.full(total_cells, -1, dtype=np.int64)
    particle_next = np.full(number_of_particles + 1, -1, dtype=np.int64)

    points = np.empty(number_of_particles + 1, dtype=np.complex128)

    np.random.seed(seed)

    # Initialize first 2 particles
    curr_point = CX_0
    points[0] = curr_point
    add_to_points_grid_static(
        curr_point, max_radius,
        0, points_grid_size, points_grid_mesh,
        grid_head, particle_next
    )
    mark_particle(curr_point, max_radius,
                  layer_count, layer_sizes, layer_meshes, layers)

    curr_point = 2.0 * rand_circ()
    points[1] = curr_point
    add_to_points_grid_static(
        curr_point, max_radius,
        1, points_grid_size, points_grid_mesh,
        grid_head, particle_next
    )
    mark_particle(curr_point, max_radius,
                  layer_count, layer_sizes, layer_meshes, layers)

    # Run aggregation loop
    num_to_add = number_of_particles - 1
    if num_to_add < 0:
        num_to_add = 0

    points_added, _ = aggregate_loop(
        num_to_add,
        max_radius, noise_reduction_factor,
        layer_count, layer_sizes, layer_meshes, layers,
        points_grid_size, points_grid_mesh,
        grid_head, particle_next, points
    )

    return points[:points_added]


def run_model(params: BellOffParams | dict | None = None) -> utils.ClusterResult:
    """
    Run Bell's off-lattice Fast Grid DLA and return a ClusterResult.
    """
    if params is None:
        params = BellOffParams()
    elif isinstance(params, dict):
        params = BellOffParams(**params)

    # Timing instrumentation
    t_start = time.perf_counter()

    seed = 0 if params.seed is None else int(params.seed)
    pts = fast_dla(
        int(params.num_particles),
        seed=seed,
        noise_reduction_factor=float(params.noise_reduction_factor),
        max_min_mesh=float(params.max_min_mesh),
        scale_of_points_grid=float(params.scale_of_points_grid),
    )

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    particles_per_sec = params.num_particles / elapsed if elapsed > 0 else 0.0

    print(f"Simulation completed: {params.num_particles} particles in {elapsed:.2f}s ({particles_per_sec:.0f} particles/s)")

    # Convert complex positions to Nx2 array
    n = pts.shape[0]
    positions = np.empty((n, 2), dtype=np.float64)
    positions[:, 0] = pts.real
    positions[:, 1] = pts.imag

    meta = {
        "model": "bell_off",
        "num": int(params.num_particles),
        "noise_reduction_factor": float(params.noise_reduction_factor),
        "max_min_mesh": float(params.max_min_mesh),
        "scale_of_points_grid": float(params.scale_of_points_grid),
        "seed": params.seed,
        "time_elapsed": elapsed,
    }

    return utils.ClusterResult(occupied=None, positions=positions, meta=meta)


__all__ = ["fast_dla", "BellOffParams", "run_model"]


