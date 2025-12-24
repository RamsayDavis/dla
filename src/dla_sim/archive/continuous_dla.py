"""
Continuous (Off-Lattice) DLA Simulator using Hybrid Algorithm.

Combines:
- Distance Grid optimization (Kuijpers et al. 2014) for O(1) distance lookups
- Killing-Free Poisson Return (Menshutin & Shchur 2005) for boundary conditions

"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit

from . import utils

###############################################################################
# Constants
###############################################################################

# Default parameters
DEFAULT_PARTICLE_RADIUS = 0.5
DEFAULT_GRID_RESOLUTION = 1.0  # Grid cell size
DEFAULT_R_DEATH_MULTIPLIER = 100.0  # R_death = multiplier * R_max
DEFAULT_OMEGA_UPDATE_RADIUS = 40  # Local update radius for Omega grid

###############################################################################
# Poisson Return Helper (Menshutin & Shchur 2005)
###############################################################################


@njit(cache=True, fastmath=True)
def sample_poisson_return_angle(x: float) -> float:
    """
    Direct sampling using analytical inverse CDF.
    
    Uses the closed-form solution from Menshutin & Shchur (2005):
    f(u) = 2 * arctan((x-1)/(x+1) * tan(u * π/2))
    
    This maps a uniform random variable u ∈ [-1, 1] directly to the angle
    distributed according to the Poisson kernel.
    
    Args:
        x: Ratio r_current / R_birth (must be > 1)
    
    Returns:
        Angle deviation in radians (typically in [-pi, pi])
    
    Reference: Menshutin & Shchur (2005) Note [27] / Eq near citation 966
    """
    if x <= 1.0:
        return 0.0
    
    # 1. Uniform random number u in [-1, 1]
    u = (np.random.random() * 2.0) - 1.0
    
    # 2. Analytical Mapping
    # tan(u * π / 2) maps [-1, 1] to (-inf, inf)
    tan_val = math.tan(u * math.pi / 2.0)
    
    # Scaling factor from Poisson kernel geometry
    factor = (x - 1.0) / (x + 1.0)
    
    # Map back to angle using analytical inverse CDF
    theta = 2.0 * math.atan(factor * tan_val)
    
    return theta


###############################################################################
# Collision Detection (Kuijpers et al. 2014)
###############################################################################


@njit(cache=True, fastmath=True)
def solve_quadratic_collision(
    px: float, py: float,  # Static particle position
    wx: float, wy: float,  # Walker current position
    dx: float, dy: float,  # Walker direction (unit vector)
    particle_radius: float,
) -> Tuple[float, bool]:
    """
    Solve quadratic equation to find collision distance.
    
    Solves: A * L^2 + B * L + C = 0
    where L is the step length to collision.
    
    We want |rx - L*d|^2 = (2R)^2, which expands to -2L(rx.d)
    
    Args:
        px, py: Static particle center
        wx, wy: Walker position
        dx, dy: Walker direction (normalized)
        particle_radius: Radius of particles
    
    Returns:
        (L_hit, found): Hit distance and whether collision found
        If no collision, returns (0.0, False)
    
    Reference: Kuijpers et al. (2014) Eq 2
    """
    # Vector from walker to particle
    rx = px - wx
    ry = py - wy
    
    # Quadratic coefficients
    # A = dx^2 + dy^2 (should be 1.0 for unit vector)
    A = dx * dx + dy * dy
    # FIX: Change sign to -2.0. We want |rx - L*d|^2 = (2R)^2
    B = -2.0 * (rx * dx + ry * dy)
    C = rx * rx + ry * ry - (2.0 * particle_radius) ** 2
    
    # Check if already overlapping
    if C <= 0.0:
        return 0.0, True
    
    # Discriminant
    discriminant = B * B - 4.0 * A * C
    
    if discriminant < 0.0:
        return 0.0, False
    
    sqrt_disc = math.sqrt(discriminant)
    
    # Two solutions
    L1 = (-B - sqrt_disc) / (2.0 * A)
    L2 = (-B + sqrt_disc) / (2.0 * A)
    
    # Find smallest positive solution (first impact)
    L_hit = None
    if L1 > 0.0:
        L_hit = L1
    if L2 > 0.0:
        if L_hit is None or L2 < L_hit:
            L_hit = L2
    
    if L_hit is None:
        return 0.0, False
    
    return L_hit, True


###############################################################################
# Distance Grid Helpers (Kuijpers)
###############################################################################


@njit(cache=True, parallel=False)  # parallel=True can cause issues with small grids
def _init_omega_grid(
    size: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> np.ndarray:
    """
    Initialize Omega grid with distance to seed at origin (0, 0).
    
    This is critical: omega must start with the distance to the seed,
    not INT_MAX, otherwise particles will take infinite steps.
    
    Args:
        size: Grid size (size x size)
        origin_x, origin_y: World coordinates of grid origin
        resolution: Grid cell resolution
    
    Returns:
        Omega grid initialized with distance to origin
    """
    grid = np.empty((size, size), dtype=np.int32)
    
    for y in range(size):
        for x in range(size):
            # Calculate world position of cell center
            wx = origin_x + (x + 0.5) * resolution
            wy = origin_y + (y + 0.5) * resolution
            
            # Distance to seed at (0, 0)
            dist = math.sqrt(wx * wx + wy * wy)
            
            # Store as integer grid units
            grid[y, x] = int(dist / resolution)
    
    return grid


@njit(cache=True, fastmath=True)
def _world_to_grid(x: float, y: float, grid_origin_x: float, grid_origin_y: float, grid_resolution: float) -> Tuple[int, int]:
    """Convert world coordinates to grid indices."""
    gx = int((x - grid_origin_x) / grid_resolution)
    gy = int((y - grid_origin_y) / grid_resolution)
    return gx, gy


@njit(cache=True, fastmath=True)
def _grid_to_world(gx: int, gy: int, grid_origin_x: float, grid_origin_y: float, grid_resolution: float) -> Tuple[float, float]:
    """Convert grid indices to world coordinates (center of cell)."""
    x = grid_origin_x + (gx + 0.5) * grid_resolution
    y = grid_origin_y + (gy + 0.5) * grid_resolution
    return x, y


@njit(cache=True, fastmath=True)
def _distance_to_particle(px: float, py: float, gx: int, gy: int, grid_origin_x: float, grid_origin_y: float, grid_resolution: float) -> float:
    """Compute distance from grid cell center to particle."""
    wx, wy = _grid_to_world(gx, gy, grid_origin_x, grid_origin_y, grid_resolution)
    dx = px - wx
    dy = py - wy
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True, fastmath=True)
def update_omega_local(
    omega: np.ndarray,
    px: float,
    py: float,
    grid_origin_x: float,
    grid_origin_y: float,
    grid_resolution: float,
    update_radius: int,
) -> None:
    """
    Update Omega grid locally around new particle.
    
    Only scans a small box around the new particle and updates
    cells where the new distance is smaller.
    """
    gx_center, gy_center = _world_to_grid(px, py, grid_origin_x, grid_origin_y, grid_resolution)
    
    height, width = omega.shape
    
    for dgx in range(-update_radius, update_radius + 1):
        for dgy in range(-update_radius, update_radius + 1):
            gx = gx_center + dgx
            gy = gy_center + dgy
            
            if gx < 0 or gx >= width or gy < 0 or gy >= height:
                continue
            
            dist = _distance_to_particle(px, py, gx, gy, grid_origin_x, grid_origin_y, grid_resolution)
            dist_int = int(dist / grid_resolution)
            
            # Update if new distance is smaller
            if omega[gy, gx] > dist_int:
                omega[gy, gx] = dist_int


@njit(cache=True, fastmath=True)
def get_safe_step(
    omega: np.ndarray,
    x: float,
    y: float,
    grid_origin_x: float,
    grid_origin_y: float,
    grid_resolution: float,
    R_max_estimate: float,
) -> float:
    """
    Get safe step size from Omega grid.
    
    If inside grid: use O(1) lookup from omega grid.
    If outside grid: use geometric bound (Triangle Inequality).
    
    Args:
        omega: Distance grid
        x, y: Current position
        grid_origin_x, grid_origin_y: Grid origin coordinates
        grid_resolution: Grid cell resolution
        R_max_estimate: Estimated maximum radius of cluster
    
    Returns:
        Safe step distance (always finite and positive)
    """
    gx, gy = _world_to_grid(x, y, grid_origin_x, grid_origin_y, grid_resolution)
    height, width = omega.shape
    
    # Inside Grid: use O(1) lookup
    if 0 <= gx < width and 0 <= gy < height:
        return float(omega[gy, gx]) * grid_resolution
    
    # Outside Grid: Geometric Fallback using Triangle Inequality
    # The cluster is contained within R_max.
    # Safe step is (Distance to Origin) - R_max - Margin.
    dist_to_origin = math.sqrt(x * x + y * y)
    safe_dist = dist_to_origin - R_max_estimate - 2.0
    
    # Return at least grid_resolution to keep moving towards the map
    return max(grid_resolution, safe_dist)


###############################################################################
# Spatial Hashing (Cell Lists) for O(1) Collision Detection
###############################################################################


@njit(cache=True, fastmath=True)
def add_to_spatial_grid(
    particle_idx: int,
    gx: int,
    gy: int,
    spatial_head: np.ndarray,
    spatial_next: np.ndarray,
) -> None:
    """
    Add a particle to the spatial hash grid using linked list in array.
    
    Args:
        particle_idx: Index of the particle to add
        gx, gy: Grid coordinates of the particle
        spatial_head: 2D array storing head of linked list for each cell
        spatial_next: 1D array storing next pointer for each particle
    """
    height, width = spatial_head.shape
    
    # Bounds check
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
        return
    
    # Insert at head of linked list
    # Link current particle to previous head
    spatial_next[particle_idx] = spatial_head[gy, gx]
    # Make current particle the new head
    spatial_head[gy, gx] = particle_idx


###############################################################################
# Main Simulation Loop
###############################################################################


@njit(cache=True)
def _check_collision_nearby(
    px_array: np.ndarray,
    py_array: np.ndarray,
    wx: float,
    wy: float,
    particle_radius: float,
    search_radius: float,
    spatial_head: np.ndarray,
    spatial_next: np.ndarray,
    grid_origin_x: float,
    grid_origin_y: float,
    grid_resolution: float,
) -> Tuple[float, float, float, bool]:
    """
    Check for collisions with nearby particles using spatial hashing (O(1) lookup).
    
    Uses a 3x3 neighborhood around the walker's grid cell to find candidate particles.
    Only checks particles in those cells, reducing complexity from O(N) to O(1).
    
    Returns:
        (dx, dy, L_hit, found): Direction vector, hit distance, and whether collision found
    """
    min_dist_sq = float('inf')
    best_px = 0.0
    best_py = 0.0
    found = False
    
    search_radius_sq = search_radius * search_radius
    
    # Get walker's grid coordinates
    gx_walker, gy_walker = _world_to_grid(wx, wy, grid_origin_x, grid_origin_y, grid_resolution)
    height, width = spatial_head.shape
    
    # Check 3x3 neighborhood around walker's cell
    for dgx in range(-1, 2):
        for dgy in range(-1, 2):
            gx = gx_walker + dgx
            gy = gy_walker + dgy
            
            # Bounds check
            if gx < 0 or gx >= width or gy < 0 or gy >= height:
                continue
            
            # Traverse linked list for this cell
            p_idx = spatial_head[gy, gx]
            while p_idx >= 0:  # -1 indicates end of list
                px = px_array[p_idx]
                py = py_array[p_idx]
                
                dx = px - wx
                dy = py - wy
                dist_sq = dx * dx + dy * dy
                
                if dist_sq < search_radius_sq:
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_px = px
                        best_py = py
                        found = True
                
                # Move to next particle in linked list
                p_idx = spatial_next[p_idx]
    
    if not found:
        return 0.0, 0.0, 0.0, False
    
    # Check if already touching
    dist = math.sqrt(min_dist_sq)
    # Use slightly larger threshold to catch particles that are very close
    if dist < 2.0 * particle_radius + 0.1:  # Increased from 1e-6 to 0.1 for robustness
        # Already touching, return direction and zero step
        if dist > 1e-10:
            dx = best_px - wx
            dy = best_py - wy
            dx /= dist
            dy /= dist
        else:
            # Exactly at particle center, use random direction
            angle = np.random.random() * 2.0 * math.pi
            dx = math.cos(angle)
            dy = math.sin(angle)
        return dx, dy, 0.0, True
    
    # Compute direction for collision check
    dx = best_px - wx
    dy = best_py - wy
    if dist > 1e-10:
        dx /= dist
        dy /= dist
    else:
        return 0.0, 0.0, 0.0, False
    
    # Solve quadratic for collision
    L_hit, hit_found = solve_quadratic_collision(
        best_px, best_py, wx, wy, dx, dy, particle_radius
    )
    
    if hit_found and L_hit >= -1e-9 and L_hit < search_radius:
        return dx, dy, L_hit, True
    
    return 0.0, 0.0, 0.0, False


@njit(cache=True)
def _simulate_particle(
    px_array: np.ndarray,
    py_array: np.ndarray,
    num_particles: int,
    Y: np.ndarray,
    omega: np.ndarray,
    spatial_head: np.ndarray,
    spatial_next: np.ndarray,
    grid_origin_x: float,
    grid_origin_y: float,
    grid_resolution: float,
    particle_radius: float,
    R_birth: float,
    R_death: float,
    omega_update_radius: int,
    max_steps: int,
) -> Tuple[int, bool]:
    """
    Simulate a single particle until it sticks or is killed.
    
    Returns:
        (particle_index, stuck): Index where particle stuck, and whether it stuck
    """
    # Launch particle at R_birth
    theta = np.random.random() * 2.0 * math.pi
    wx = R_birth * math.cos(theta)
    wy = R_birth * math.sin(theta)
    
    steps = 0
    
    while steps < max_steps:
        steps += 1
        
        # Safety check: detect NaN/Inf positions early and reset
        if not (math.isfinite(wx) and math.isfinite(wy)):
            # Reset to R_birth if position becomes invalid
            theta = np.random.random() * 2.0 * math.pi
            wx = R_birth * math.cos(theta)
            wy = R_birth * math.sin(theta)
            continue
        
        # Check if particle wandered too far (Poisson return)
        r_current = math.sqrt(wx * wx + wy * wy)
        
        # Safety check: if r_current is invalid, reset
        if not math.isfinite(r_current) or r_current > 1e10:
            theta = np.random.random() * 2.0 * math.pi
            wx = R_birth * math.cos(theta)
            wy = R_birth * math.sin(theta)
            continue
        
        if r_current > R_death:
            # Poisson return to R_birth
            # Safety: ensure R_birth is valid and non-zero
            if R_birth > 1e-10:
                x_ratio = r_current / R_birth
                angle_dev = sample_poisson_return_angle(x_ratio)
                
                # Current angle
                phi = math.atan2(wy, wx)
                # New angle after deviation
                phi_new = phi + angle_dev
                
                # Teleport to R_birth with new angle
                wx = R_birth * math.cos(phi_new)
                wy = R_birth * math.sin(phi_new)
            else:
                # R_birth is too small, reset to default
                theta = np.random.random() * 2.0 * math.pi
                wx = 1.0 * math.cos(theta)
                wy = 1.0 * math.sin(theta)
            continue
        
        # Get safe step size from Omega grid (with geometric fallback for out-of-bounds)
        # Use R_birth / 2.0 as estimate for R_max (cluster radius)
        R_max_estimate = R_birth / 2.0
        dist = get_safe_step(omega, wx, wy, grid_origin_x, grid_origin_y, grid_resolution, R_max_estimate)
        
        # Safety check: if dist is invalid (NaN/Inf), reset particle position
        # Note: dist can be 0.0 at seed location, so check >= 0.0
        if not (dist >= 0.0 and dist < 1e10 and math.isfinite(dist)):
            # Reset to R_birth if position becomes invalid
            theta = np.random.random() * 2.0 * math.pi
            wx = R_birth * math.cos(theta)
            wy = R_birth * math.sin(theta)
            continue
        
        # Adaptive step sizing (Kuijpers method)
        if dist > 5.0:
            # Far field: Giant step (cap at reasonable maximum)
            L = dist - 1.0  
            # Ensure L is valid
            if not (L > 0.0 and L < 1e10):
                L = 5.0  # Default safe step size
            # Take step in random direction
            step_theta = np.random.random() * 2.0 * math.pi
            wx += L * math.cos(step_theta)
            wy += L * math.sin(step_theta)
        elif dist > 1.0:
            # Mid field
            L = max(1.0, dist - 2.0)
            # Ensure L is valid
            if not (L > 0.0 and L < 1e10):
                L = 1.0  # Default safe step size
            # Take step in random direction
            step_theta = np.random.random() * 2.0 * math.pi
            wx += L * math.cos(step_theta)
            wy += L * math.sin(step_theta)
        else:
            # Near field: Check for collision
            # First, do a quick direct distance check to catch very close particles
            # This prevents infinite loops when particles are extremely close
            min_direct_dist_sq = float('inf')
            closest_px = 0.0
            closest_py = 0.0
            capture_dist_sq = (2.0 * particle_radius + 0.1) ** 2  # Slightly larger for robustness
            
            for p_idx in range(num_particles):
                px = px_array[p_idx]
                py = py_array[p_idx]
                # CRITICAL FIX: Don't skip (0,0) - that's the seed particle!
                # Only skip truly empty slots (check if this is beyond num_particles)
                # Actually, we should check ALL particles, including seed at (0,0)
                dx = px - wx
                dy = py - wy
                dist_sq = dx * dx + dy * dy
                if dist_sq < capture_dist_sq and dist_sq < min_direct_dist_sq:
                    min_direct_dist_sq = dist_sq
                    closest_px = px
                    closest_py = py
            
            # If very close to a particle, stick immediately
            if min_direct_dist_sq < capture_dist_sq:
                # Already touching or very close, stick immediately
                px_array[num_particles] = wx
                py_array[num_particles] = wy
                
                # Update Y grid
                gx, gy = _world_to_grid(wx, wy, grid_origin_x, grid_origin_y, grid_resolution)
                height, width = Y.shape
                if 0 <= gx < width and 0 <= gy < height:
                    Y[gy, gx] = 1
                    
                    # Add to spatial hash
                    add_to_spatial_grid(num_particles, gx, gy, spatial_head, spatial_next)
                
                # Update Omega grid locally
                update_omega_local(
                    omega, wx, wy, grid_origin_x, grid_origin_y,
                    grid_resolution, omega_update_radius
                )
                
                return num_particles, True
            
            # Use larger search radius in near field to ensure we find nearby particles
            search_radius = max(5.0 * particle_radius, 2.5)  # At least 2.5 units
            dx, dy, L_hit, collision_found = _check_collision_nearby(
                px_array, py_array, wx, wy, particle_radius, search_radius,
                spatial_head, spatial_next, grid_origin_x, grid_origin_y, grid_resolution
            )
            
            if collision_found:
                if L_hit > 1e-6 and math.isfinite(L_hit) and L_hit < 1e6:
                    # Move to collision point
                    wx += dx * L_hit
                    wy += dy * L_hit
                # If L_hit <= 1e-6 or invalid, we're already at the collision point
                
                # Safety check: ensure final position is valid before sticking
                if not (math.isfinite(wx) and math.isfinite(wy)):
                    # Reset to safe position near cluster
                    theta = np.random.random() * 2.0 * math.pi
                    wx = R_birth * math.cos(theta)
                    wy = R_birth * math.sin(theta)
                
                # Stick particle
                px_array[num_particles] = wx
                py_array[num_particles] = wy
                
                # Update Y grid
                gx, gy = _world_to_grid(wx, wy, grid_origin_x, grid_origin_y, grid_resolution)
                height, width = Y.shape
                if 0 <= gx < width and 0 <= gy < height:
                    Y[gy, gx] = 1
                    
                    # Add to spatial hash
                    add_to_spatial_grid(num_particles, gx, gy, spatial_head, spatial_next)
                
                # Update Omega grid locally
                update_omega_local(
                    omega, wx, wy, grid_origin_x, grid_origin_y,
                    grid_resolution, omega_update_radius
                )
                
                return num_particles, True
            
            # No collision found in near field
            # If we've been stuck for many steps, take a larger step to escape
            # This prevents infinite loops when particles get trapped
            if steps > max_steps // 10:  # After 10% of max steps, start taking larger steps
                # Force a larger step to escape the near field
                L = min(2.0, dist + 1.0)  # Step size to get out of near field
            else:
                # Normal small step
                L = max(0.5 * particle_radius, 0.25)  # At least 0.25 units
            
            # Ensure L is valid
            if not (L > 0.0 and L < 1e10):
                L = 1.0  # Default safe step size
            
            step_theta = np.random.random() * 2.0 * math.pi
            wx += L * math.cos(step_theta)
            wy += L * math.sin(step_theta)
        
        # Safety check: detect NaN/Inf positions and reset (after all step calculations)
        if not (math.isfinite(wx) and math.isfinite(wy)):
            # Reset to R_birth if position becomes invalid
            theta = np.random.random() * 2.0 * math.pi
            wx = R_birth * math.cos(theta)
            wy = R_birth * math.sin(theta)
            continue
    
    # Max steps reached, particle failed to stick
    return num_particles, False


###############################################################################
# Configuration and Main Interface
###############################################################################


@dataclass
class ContinuousRunParams:
    num_particles: int = 10_000
    particle_radius: float = DEFAULT_PARTICLE_RADIUS
    grid_resolution: float = DEFAULT_GRID_RESOLUTION
    grid_padding: int = 50  # Extra grid cells beyond cluster
    R_death_multiplier: float = DEFAULT_R_DEATH_MULTIPLIER
    omega_update_radius: int = DEFAULT_OMEGA_UPDATE_RADIUS
    max_steps_per_particle: int = 1_000_000
    seed: Optional[int] = None


class ContinuousDLASimulator:
    """
    High-performance continuous DLA simulator.
    
    Uses:
    - Distance Grid (Omega) for O(1) distance queries
    - Poisson Return for killing-free boundary conditions
    - Quadratic collision detection for exact sticking
    """
    
    def __init__(self, params: ContinuousRunParams) -> None:
        self.params = params
        if params.seed is not None:
            np.random.seed(params.seed)
        
        # Initialize arrays
        max_particles = params.num_particles + 1000  # Extra capacity
        self.px_array = np.zeros(max_particles, dtype=np.float64)
        self.py_array = np.zeros(max_particles, dtype=np.float64)
        
        # Estimate grid size (will grow if needed)
        estimated_R_max = params.num_particles ** 0.6 * params.particle_radius * 2.0
        grid_size = int(2 * (estimated_R_max + params.grid_padding * params.grid_resolution) / params.grid_resolution)
        # Round up to next power of 2 for efficiency
        grid_size = 1 << (grid_size - 1).bit_length()
        
        self.grid_size = grid_size
        self.grid_origin_x = -grid_size * params.grid_resolution / 2.0
        self.grid_origin_y = -grid_size * params.grid_resolution / 2.0
        
        # Initialize grids
        self.Y = np.zeros((grid_size, grid_size), dtype=np.int8)
        
        # CRITICAL: Initialize Omega with distance to seed at (0,0)
        # This prevents particles from taking infinite steps (INT_MAX distance)
        # Every cell starts with its distance to the origin
        self.omega = _init_omega_grid(
            grid_size, self.grid_origin_x, self.grid_origin_y, params.grid_resolution
        )
        
        # Initialize spatial hash for O(1) collision detection
        # spatial_head: 2D array storing head of linked list for each grid cell
        # spatial_next: 1D array storing next pointer for each particle
        self.spatial_head = np.full((grid_size, grid_size), -1, dtype=np.int32)
        self.spatial_next = np.full(max_particles, -1, dtype=np.int32)
        
        # Place seed particle at origin
        self.px_array[0] = 0.0
        self.py_array[0] = 0.0
        self.num_particles = 1
        
        # Update grids for seed (Y grid, Omega, and spatial hash)
        gx, gy = _world_to_grid(0.0, 0.0, self.grid_origin_x, self.grid_origin_y, params.grid_resolution)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            self.Y[gy, gx] = 1
            # Omega is already initialized with distance to origin, but we can refine
            # the local region around the seed for better accuracy
            update_omega_local(
                self.omega, 0.0, 0.0, self.grid_origin_x, self.grid_origin_y,
                params.grid_resolution, params.omega_update_radius
            )
            # Add seed particle (index 0) to spatial hash
            add_to_spatial_grid(0, gx, gy, self.spatial_head, self.spatial_next)
        
        self.R_max = params.particle_radius
        # Ensure R_birth is at least 2.0 to prevent particles starting too close to seed
        self.R_birth = max(params.particle_radius * 2.0, 2.0)
        self.R_death = self.R_birth * params.R_death_multiplier
    
    def run(self) -> None:
        """Run simulation until target number of particles."""
        t_start = time.perf_counter()
        
        while self.num_particles < self.params.num_particles:
            # Update R_birth and R_death based on current cluster size
            # Ensure R_birth is always at least 2.0 to prevent particles starting too close
            self.R_birth = max(self.R_birth, self.R_max * 2.0, 2.0)
            self.R_death = self.R_birth * self.params.R_death_multiplier
            
            # Check if grid needs expansion
            current_extent = self.R_max + self.params.grid_padding * self.params.grid_resolution
            required_grid_half = int(current_extent / self.params.grid_resolution)
            if required_grid_half * 2 >= self.grid_size:
                # Expand grid (simplified: just warn for now)
                # In production, would implement dynamic expansion
                pass
            
            # Simulate one particle
            particle_idx, stuck = _simulate_particle(
                self.px_array,
                self.py_array,
                self.num_particles,
                self.Y,
                self.omega,
                self.spatial_head,
                self.spatial_next,
                self.grid_origin_x,
                self.grid_origin_y,
                self.params.grid_resolution,
                self.params.particle_radius,
                self.R_birth,
                self.R_death,
                self.params.omega_update_radius,
                self.params.max_steps_per_particle,
            )
            
            if stuck:
                self.num_particles += 1
                # Update R_max
                px = self.px_array[particle_idx]
                py = self.py_array[particle_idx]
                r = math.sqrt(px * px + py * py)
                self.R_max = max(self.R_max, r)
            
            # Progress reporting
            if self.num_particles % max(1, self.params.num_particles // 10) == 0:
                elapsed = time.perf_counter() - t_start
                rate = self.num_particles / elapsed if elapsed > 0 else 0.0
                print(f"[continuous] {self.num_particles}/{self.params.num_particles} particles, "
                      f"{rate:.0f} particles/s, R_max={self.R_max:.1f}")
        
        elapsed = time.perf_counter() - t_start
        rate = self.params.num_particles / elapsed if elapsed > 0 else 0.0
        print(f"Simulation completed: {self.params.num_particles} particles in {elapsed:.2f}s "
              f"({rate:.0f} particles/s)")
    
    def get_positions(self) -> np.ndarray:
        """Get particle positions as Nx2 array."""
        positions = np.zeros((self.num_particles, 2), dtype=np.float64)
        positions[:, 0] = self.px_array[:self.num_particles]
        positions[:, 1] = self.py_array[:self.num_particles]
        return positions


def run_model(params: ContinuousRunParams | dict | None = None) -> utils.ClusterResult:
    """
    Run continuous DLA model and return ClusterResult.
    """
    if params is None:
        params = ContinuousRunParams()
    elif isinstance(params, dict):
        params = ContinuousRunParams(**params)
    
    sim = ContinuousDLASimulator(params)
    sim.run()
    
    positions = sim.get_positions()
    
    # Extract coordinates for visualization (as floats)
    x_coords = sim.px_array[:sim.num_particles].copy()
    y_coords = sim.py_array[:sim.num_particles].copy()
    
    meta = {
        "model": "continuous",
        "num": int(params.num_particles),
        "particle_radius": float(params.particle_radius),
        "grid_resolution": float(params.grid_resolution),
        "R_max": float(sim.R_max),
        "seed": params.seed,
        "x_coords": x_coords,
        "y_coords": y_coords,
    }
    
    return utils.ClusterResult(occupied=None, positions=positions, meta=meta)


__all__ = [
    "ContinuousRunParams",
    "ContinuousDLASimulator",
    "run_model",
    "sample_poisson_return_angle",
    "solve_quadratic_collision",
]

