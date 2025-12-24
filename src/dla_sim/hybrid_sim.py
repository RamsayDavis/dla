from dataclasses import dataclass

import numpy as np
from numba import njit
from numba.typed import List


PI = 3.1415926535
TWO_PI = 6.2831853072
CX_1 = complex(1.0, 0.0)
CX_0 = complex(0.0, 0.0)
CX_J = complex(0.0, 1.0)
GRID_SPACING = 2.0  # Grid spacing for hybrid model
EPSILON = 0.1  # Tolerance for exact position matching


@dataclass
class HybridParams:
    """Configuration for Hybrid off-lattice diffusion with on-lattice aggregation DLA."""

    num_particles: int = 10_000
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
    Optimized Möbius form: (R*z + 1) / (R + z)
    """
    z = rand_circ()
    numerator = abs_pos * z + CX_1
    denominator = abs_pos + z
    return numerator / denominator


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
        Implements a static linked list. The 'grid_head' array points to the 
        first particle in a cell. We link the new particle to the old head, 
        then make the new particle the new head (Push-Front operation).    """
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
    Traverses the linked list for a single grid cell to find neighbors.
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
    Calculates the walker's current cell, then calls 'check_for_closer_static'
    for that cell and its 8 neighbors. Guarantees finding the nearest neighbor
    if the mesh size is appropriately calibrated.
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
def is_occupied_exact(target_pos, max_radius, points, grid_head, particle_next,
                      points_grid_size, points_grid_mesh):
    """
    Checks if a position is exactly occupied (within epsilon tolerance).
    
    Uses the spatial hash (linked list) to search for particles at the target
    position. Returns True if a particle exists within EPSILON distance.
    """
    # Get grid cell indices for target position
    idx1, idx2 = get_indices(target_pos, max_radius, points_grid_mesh)
    
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
    
    # Traverse linked list for this cell and its 8 neighbors
    i_min = max(idx1 - 1, 0)
    i_max = min(idx1 + 2, points_grid_size)
    j_min = max(idx2 - 1, 0)
    j_max = min(idx2 + 2, points_grid_size)
    
    epsilon2 = EPSILON * EPSILON
    
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            cell_flat_idx = i * points_grid_size + j
            p_idx = grid_head[cell_flat_idx]
            
            # Traverse linked list
            while p_idx != -1:
                diff = target_pos - points[p_idx]
                dist2 = (diff.real * diff.real + diff.imag * diff.imag)
                
                if dist2 < epsilon2:
                    return True
                
                p_idx = particle_next[p_idx]
    
    return False


@njit
def is_marked_at_layer(curr_point, max_radius,
                       layer, layer_sizes, layer_meshes, layers):
    """
    Checks if a location is marked as 'occupied' in a specific hierarchy layer.
    Used by the binary search to see if a specific step size is safe.
    """                   
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
    Determines the maximum safe step size using Binary Search.
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
    """
    Marks the walker's current cell and its 8 neighbors as occupied in the grid.
    
    Logic:
        Marking the 3x3 neighborhood creates a safety buffer. If a cell is 
        'Empty' in the hierarchy, it guarantees the cluster is at least one 
        full grid cell away, making the jump safe.
    """
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
    """
    Updates the entire grid hierarchy when a particle sticks.
    Iterates from finest to coarsest layer, calling set_marks on each.
    """
    for layer in range(layer_count - 1, -1, -1):
        set_marks(curr_point, max_radius, layer,
                  layer_sizes, layer_meshes, layers)


@njit
def finishing_step(curr_point, nearest, dist_to_nearest2, max_safe_dist2,
                   max_radius, points, grid_head, particle_next,
                   points_grid_size, points_grid_mesh):
    """
    Hybrid adhering step: Conformal mapping (continuous) + Grid snapping (discrete).
    
    Step 1: Preserve Bell's conformal mapping physics to calculate candidate position.
    Step 2: Snap to nearest cardinal direction on grid (spacing 2.0).
    Step 3: Check occupancy and try primary/secondary candidates.
    
    Returns:
        (curr_point, particle_free) where particle_free=False means stuck.
    """
    backup = curr_point

    # Step 1: Conformal mapping (preserve original physics)
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

    curr_point_continuous = curr_point + (CX_J * y4 * (nearest - curr_point) / d1)
    particle_free_continuous = (y2 >= 0.0)

    if (curr_point_continuous == backup) or (np.isnan(curr_point_continuous.real) and max_safe_dist2 < 4.01):
        curr_point_continuous = backup
        particle_free_continuous = False

    # If conformal mapping says particle is free, return early
    if particle_free_continuous:
        return curr_point_continuous, True

    # Step 2: Calculate vector from nearest neighbor
    diff = curr_point_continuous - nearest
    
    # Step 3: Determine primary and secondary candidates (cardinal directions)
    abs_real = abs(diff.real)
    abs_imag = abs(diff.imag)
    
    # Primary candidate: closest cardinal direction
    # Secondary candidate: second closest cardinal direction
    if abs_real > abs_imag:
        # Primary is horizontal
        if diff.real > 0:
            primary_candidate = nearest + complex(GRID_SPACING, 0.0)  # Right
        else:
            primary_candidate = nearest + complex(-GRID_SPACING, 0.0)  # Left
        
        # Secondary is vertical
        if diff.imag > 0:
            secondary_candidate = nearest + complex(0.0, GRID_SPACING)  # Up
        else:
            secondary_candidate = nearest + complex(0.0, -GRID_SPACING)  # Down
    else:
        # Primary is vertical
        if diff.imag > 0:
            primary_candidate = nearest + complex(0.0, GRID_SPACING)  # Up
        else:
            primary_candidate = nearest + complex(0.0, -GRID_SPACING)  # Down
        
        # Secondary is horizontal
        if diff.real > 0:
            secondary_candidate = nearest + complex(GRID_SPACING, 0.0)  # Right
        else:
            secondary_candidate = nearest + complex(-GRID_SPACING, 0.0)  # Left
    
    # Step 4: Check occupancy
    # Try primary candidate first
    if not is_occupied_exact(primary_candidate, max_radius, points, grid_head,
                             particle_next, points_grid_size, points_grid_mesh):
        # Primary is empty - stick here
        return primary_candidate, False
    
    # Primary is occupied, try secondary
    if not is_occupied_exact(secondary_candidate, max_radius, points, grid_head,
                             particle_next, points_grid_size, points_grid_mesh):
        # Secondary is empty - stick here
        return secondary_candidate, False
    
    # Both occupied - bounce back
    return backup, True


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
    """Expands the launch circle as the cluster grows, keeping it just outside."""
    start_dist = max(start_dist, np.abs(curr_point) + 2.0)
    if start_dist > max_radius:
        start_dist = max_radius
    return start_dist


@njit
def aggregate_loop(
    num_to_add,
    max_radius, 
    layer_count, layer_sizes, layer_meshes, layers,
    points_grid_size, points_grid_mesh,
    grid_head, particle_next, points
):
    """
    1. Spawn particle on bounding circle.
    2. While particle is free:
       a. If too far away -> Reset (Teleport back).
       b. Find best grid layer -> Determine max safe step size.
       c. If in finest layer (Danger Zone):
          - Check neighbors.
          - If very close -> Run Finishing Step (Hybrid: conformal + grid snap).
          - Else -> Small random step.
       d. Else (Safe Zone) -> Big random step.
    3. When stuck:
       - Add to points array.
       - Add to Spatial Hash (O(1)).
       - Update Hierarchy Marks.
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
                            max_radius, points, grid_head, particle_next,
                            points_grid_size, points_grid_mesh
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
             max_min_mesh=0.0,
             scale_of_points_grid=0.0):
    """
    High-level API: Hybrid DLA using off-lattice diffusion (radius=1.0) 
    with on-lattice aggregation (grid spacing=2.0).
    """
    if max_min_mesh == 0.0:
        max_min_mesh = 24.0

    if scale_of_points_grid == 0.0:
        scale_of_points_grid = 2.0

    if scale_of_points_grid < 1.0:
        raise ValueError("scaleOfPointsGrid must be greater than 1")

    max_radius = 22.0 + 2.2 * (number_of_particles ** (1.0 / 1.7))

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

    # ALLOCATE STATIC ARRAYS (optimisation versus per-cell typed lists)
    total_cells = points_grid_size * points_grid_size
    grid_head = np.full(total_cells, -1, dtype=np.int64)
    particle_next = np.full(number_of_particles + 1, -1, dtype=np.int64)

    points = np.empty(number_of_particles + 1, dtype=np.complex128)

    np.random.seed(seed)

    # Initialize first 2 particles on grid (spacing 2.0)
    curr_point = CX_0
    points[0] = curr_point
    add_to_points_grid_static(
        curr_point, max_radius,
        0, points_grid_size, points_grid_mesh,
        grid_head, particle_next
    )
    mark_particle(curr_point, max_radius,
                  layer_count, layer_sizes, layer_meshes, layers)

    # Second particle at grid position (2.0, 0.0)
    curr_point = complex(GRID_SPACING, 0.0)
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
        max_radius,
        layer_count, layer_sizes, layer_meshes, layers,
        points_grid_size, points_grid_mesh,
        grid_head, particle_next, points
    )

    return points[:points_added]


class HybridSimulator:
    """
    Hybrid DLA Simulator: Off-lattice diffusion with on-lattice aggregation.
    
    This class provides a standardized interface for running hybrid DLA simulation:
    - Pure off-lattice diffusion (radius=1.0) using Bell's conformal mapping
    - Strict on-lattice aggregation (grid spacing=2.0)
    """
    
    def __init__(self, params: HybridParams | None = None):
        """
        Initialize the simulator with parameters.
        
        Args:
            params: Configuration parameters. If None, uses default values.
        """
        self.params = params or HybridParams()
        self.pts = None
    
    def run(self) -> None:
        """
        Run the simulation using the stored parameters.
        Stores the resulting particles internally as complex numbers.
        """
        seed = 0 if self.params.seed is None else int(self.params.seed)
        self.pts = fast_dla(
            int(self.params.num_particles),
            seed=seed,
            max_min_mesh=float(self.params.max_min_mesh),
            scale_of_points_grid=float(self.params.scale_of_points_grid),
        )
    
    def get_centered_coords(self) -> np.ndarray:
        """
        Returns the particles as a standard (N, 2) numpy array of floats.
        
        Since the grid spacing is 2.0, we divide by 2.0 to normalize to
        standard lattice format (integers 0, 1, 2...).
        
        Returns:
            Array of shape (N, 2) with particle coordinates normalized to grid spacing 1.0.
        """
        if self.pts is None:
            raise RuntimeError("Simulation has not been run. Call run() first.")
        
        # Divide by 2.0 to normalize from grid spacing 2.0 to spacing 1.0
        normalized_pts = self.pts / 2.0
        
        return np.column_stack((normalized_pts.real, normalized_pts.imag))


__all__ = ["fast_dla", "HybridParams", "HybridSimulator"]


if __name__ == "__main__":
    # Standalone execution for testing
    sim = HybridSimulator()
    sim.run()
    coords = sim.get_centered_coords()
    print(f"Generated {coords.shape[0]} particles")

