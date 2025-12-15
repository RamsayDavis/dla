"""
Verification tests for spatial hashing optimization.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dla_sim.continuous_dla import (
    _check_collision_nearby,
    _world_to_grid,
    add_to_spatial_grid,
    ContinuousDLASimulator,
    ContinuousRunParams,
)


def _check_collision_brute_force(
    px_array: np.ndarray,
    py_array: np.ndarray,
    num_particles: int,
    wx: float,
    wy: float,
    particle_radius: float,
    search_radius: float,
) -> tuple:
    """
    Brute force O(N) collision check for verification.
    Returns same format as _check_collision_nearby.
    """
    min_dist_sq = float('inf')
    best_px = 0.0
    best_py = 0.0
    found = False
    
    search_radius_sq = search_radius * search_radius
    
    for p_idx in range(num_particles):
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
    
    if not found:
        return 0.0, 0.0, 0.0, False
    
    import math
    dist = math.sqrt(min_dist_sq)
    if dist < 2.0 * particle_radius + 0.1:
        if dist > 1e-10:
            dx = best_px - wx
            dy = best_py - wy
            dx /= dist
            dy /= dist
        else:
            angle = np.random.random() * 2.0 * math.pi
            dx = math.cos(angle)
            dy = math.sin(angle)
        return dx, dy, 0.0, True
    
    dx = best_px - wx
    dy = best_py - wy
    if dist > 1e-10:
        dx /= dist
        dy /= dist
    else:
        return 0.0, 0.0, 0.0, False
    
    from src.dla_sim.continuous_dla import solve_quadratic_collision
    L_hit, hit_found = solve_quadratic_collision(
        best_px, best_py, wx, wy, dx, dy, particle_radius
    )
    
    if hit_found and L_hit >= -1e-9 and L_hit < search_radius:
        return dx, dy, L_hit, True
    
    return 0.0, 0.0, 0.0, False


def test_spatial_hash_verification():
    """Verify spatial hash returns same results as brute force."""
    # Create a scenario with 100 scattered particles
    num_particles = 100
    particle_radius = 0.5
    grid_resolution = 1.0
    grid_size = 64
    grid_origin_x = -32.0
    grid_origin_y = -32.0
    
    # Initialize arrays
    max_particles = 200
    px_array = np.zeros(max_particles, dtype=np.float64)
    py_array = np.zeros(max_particles, dtype=np.float64)
    
    # Place particles in a scattered pattern
    np.random.seed(42)
    for i in range(num_particles):
        # Place particles in a ring around origin
        angle = np.random.random() * 2.0 * np.pi
        radius = 5.0 + np.random.random() * 10.0
        px_array[i] = radius * np.cos(angle)
        py_array[i] = radius * np.sin(angle)
    
    # Initialize spatial hash
    spatial_head = np.full((grid_size, grid_size), -1, dtype=np.int32)
    spatial_next = np.full(max_particles, -1, dtype=np.int32)
    
    # Add all particles to spatial hash
    for i in range(num_particles):
        gx, gy = _world_to_grid(
            px_array[i], py_array[i], grid_origin_x, grid_origin_y, grid_resolution
        )
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            add_to_spatial_grid(i, gx, gy, spatial_head, spatial_next)
    
    # Verify spatial hash is populated
    # Check that at least some cells have particles
    has_particles = False
    for gy in range(grid_size):
        for gx in range(grid_size):
            if spatial_head[gy, gx] >= 0:
                has_particles = True
                break
        if has_particles:
            break
    
    assert has_particles, "Spatial hash should contain particles"
    
    # Test walker positions near specific particles
    test_cases = [
        (px_array[0] + 0.8, py_array[0], "near particle 0"),
        (px_array[10] + 0.7, py_array[10], "near particle 10"),
        (px_array[50] - 0.6, py_array[50], "near particle 50"),
        (20.0, 20.0, "far from all particles"),
    ]
    
    search_radius = 3.0 * particle_radius
    
    for wx, wy, desc in test_cases:
        # Test with spatial hash (optimized)
        result_spatial = _check_collision_nearby(
            px_array,
            py_array,
            wx,
            wy,
            particle_radius,
            search_radius,
            spatial_head,
            spatial_next,
            grid_origin_x,
            grid_origin_y,
            grid_resolution,
        )
        
        # Test with brute force (reference)
        result_brute = _check_collision_brute_force(
            px_array,
            py_array,
            num_particles,
            wx,
            wy,
            particle_radius,
            search_radius,
        )
        
        # Compare results
        dx_s, dy_s, L_hit_s, found_s = result_spatial
        dx_b, dy_b, L_hit_b, found_b = result_brute
        
        assert found_s == found_b, (
            f"{desc}: Found mismatch - spatial={found_s}, brute={found_b}"
        )
        
        if found_s and found_b:
            # Both found collision, check values match
            assert abs(dx_s - dx_b) < 1e-6, (
                f"{desc}: dx mismatch - spatial={dx_s}, brute={dx_b}"
            )
            assert abs(dy_s - dy_b) < 1e-6, (
                f"{desc}: dy mismatch - spatial={dy_s}, brute={dy_b}"
            )
            assert abs(L_hit_s - L_hit_b) < 1e-6, (
                f"{desc}: L_hit mismatch - spatial={L_hit_s}, brute={L_hit_b}"
            )
        
        print(f"✓ {desc}: Results match (found={found_s})")


def test_spatial_hash_linked_list():
    """Verify spatial hash correctly links particles in the same cell."""
    grid_size = 32
    max_particles = 50
    grid_origin_x = -16.0
    grid_origin_y = -16.0
    grid_resolution = 1.0
    
    # Initialize spatial hash
    spatial_head = np.full((grid_size, grid_size), -1, dtype=np.int32)
    spatial_next = np.full(max_particles, -1, dtype=np.int32)
    
    # Place multiple particles in the same grid cell
    # Cell (16, 16) corresponds to world position (0.5, 0.5)
    target_gx, target_gy = 16, 16
    
    particle_indices = [0, 1, 2, 3]
    for idx in particle_indices:
        add_to_spatial_grid(idx, target_gx, target_gy, spatial_head, spatial_next)
    
    # Verify linked list structure
    assert spatial_head[target_gy, target_gx] == 3, "Head should point to last added particle"
    
    # Traverse linked list
    p_idx = spatial_head[target_gy, target_gx]
    found_indices = []
    while p_idx >= 0:
        found_indices.append(p_idx)
        p_idx = spatial_next[p_idx]
    
    # Should find all particles in reverse order (LIFO)
    assert set(found_indices) == set(particle_indices), (
        f"Should find all particles, got {found_indices}, expected {particle_indices}"
    )
    
    print("✓ Spatial hash linked list structure is correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

