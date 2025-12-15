"""
Unit tests for continuous DLA simulator.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim.continuous_dla import (
    sample_poisson_return_angle,
    solve_quadratic_collision,
    get_safe_step,
    _init_omega_grid,
    ContinuousDLASimulator,
    ContinuousRunParams,
)


def test_sample_poisson_return_angle():
    """Test Poisson return angle sampling."""
    # Test with x = 2.0
    x = 2.0
    
    # Sample multiple angles
    angles = [sample_poisson_return_angle(x) for _ in range(100)]
    angles = np.array(angles)
    
    # Angles should be in reasonable range (typically [-pi, pi])
    assert np.all(angles >= -np.pi)
    assert np.all(angles <= np.pi)
    
    # Distribution should be peaked near 0 (small deviations)
    abs_angles = np.abs(angles)
    assert np.mean(abs_angles) < np.pi / 2  # Most angles should be small
    
    # Test with x = 1.0 (should return 0)
    angle_at_one = sample_poisson_return_angle(1.0)
    assert abs(angle_at_one) < 1e-6
    
    # Test with x < 1.0 (should return 0)
    angle_below_one = sample_poisson_return_angle(0.5)
    assert abs(angle_below_one) < 1e-6


def test_solve_quadratic_collision():
    """Test quadratic collision detection."""
    particle_radius = 0.5
    
    # Test case: particle at (0, 0), walker at (5, 0) moving towards origin
    px, py = 0.0, 0.0
    wx, wy = 5.0, 0.0
    dx, dy = -1.0, 0.0  # Moving left towards origin
    
    L_hit, found = solve_quadratic_collision(px, py, wx, wy, dx, dy, particle_radius)
    
    assert found, "Collision should be found"
    assert L_hit > 0.0, "Hit distance should be positive"
    
    # Expected: collision at distance ~ (5.0 - 2*radius) = 4.0
    expected_L = 5.0 - 2.0 * particle_radius
    assert abs(L_hit - expected_L) < 0.1, f"Expected L ~ {expected_L}, got {L_hit}"
    
    # Verify collision point is correct
    hit_x = wx + dx * L_hit
    hit_y = wy + dy * L_hit
    dist_to_particle = np.sqrt((hit_x - px)**2 + (hit_y - py)**2)
    expected_dist = 2.0 * particle_radius
    assert abs(dist_to_particle - expected_dist) < 0.01, \
        f"Collision point should be at distance {expected_dist} from particle"
    
    # Test case: walker moving away (no collision)
    dx_away, dy_away = 1.0, 0.0  # Moving right away from origin
    L_hit_away, found_away = solve_quadratic_collision(
        px, py, wx, wy, dx_away, dy_away, particle_radius
    )
    # Should either not find collision or find negative solution
    if found_away:
        assert L_hit_away <= 0.0, "Collision when moving away should have negative L"
    
    # Test case: walker already touching
    wx_touching, wy_touching = 1.0, 0.0  # At distance 1.0 from origin
    L_hit_touch, found_touch = solve_quadratic_collision(
        px, py, wx_touching, wy_touching, -1.0, 0.0, particle_radius
    )
    # Should find collision very close
    if found_touch:
        assert L_hit_touch >= 0.0
        assert L_hit_touch < 0.1  # Very small step


def test_solve_quadratic_collision_off_axis():
    """Test collision detection with off-axis approach."""
    particle_radius = 0.5
    
    # Particle at (0, 0), walker at (3, 4) = distance 5, moving towards origin
    px, py = 0.0, 0.0
    wx, wy = 3.0, 4.0
    dist_initial = np.sqrt(wx**2 + wy**2)
    
    # Normalize direction towards origin
    dx = -wx / dist_initial
    dy = -wy / dist_initial
    
    L_hit, found = solve_quadratic_collision(px, py, wx, wy, dx, dy, particle_radius)
    
    assert found, "Collision should be found"
    assert L_hit > 0.0
    
    # Verify final position
    hit_x = wx + dx * L_hit
    hit_y = wy + dy * L_hit
    dist_final = np.sqrt(hit_x**2 + hit_y**2)
    expected_dist = 2.0 * particle_radius
    assert abs(dist_final - expected_dist) < 0.01


def test_get_distance_from_omega_out_of_bounds():
    """Test that get_safe_step returns finite value when out of bounds."""
    import math
    
    # Create a small omega grid
    grid_size = 10
    origin_x = -5.0
    origin_y = -5.0
    resolution = 1.0
    R_max = 5.0  # Estimated cluster radius
    
    omega = _init_omega_grid(grid_size, origin_x, origin_y, resolution)
    
    # Test point inside grid (should return finite value)
    dist_inside = get_safe_step(
        omega, 0.0, 0.0, origin_x, origin_y, resolution, R_max
    )
    assert math.isfinite(dist_inside), "Distance inside grid should be finite"
    assert dist_inside >= 0.0, "Distance should be non-negative (can be 0.0 at seed)"
    
    # Test point outside grid (should return large finite value using geometric fallback)
    dist_outside = get_safe_step(
        omega, 100.0, 100.0, origin_x, origin_y, resolution, R_max
    )
    assert math.isfinite(dist_outside), "Distance outside grid should be finite (not infinity)"
    # Dist to origin ~141. R_max 5. Result should be huge (> 100)
    assert dist_outside > 100.0, f"Out-of-bounds should return large value, got {dist_outside}"
    
    # Test negative coordinates
    dist_negative = get_safe_step(
        omega, -100.0, -100.0, origin_x, origin_y, resolution, R_max
    )
    assert math.isfinite(dist_negative), "Distance for negative coords should be finite"
    assert dist_negative > 100.0, f"Out-of-bounds should return large value, got {dist_negative}"


def test_get_distance_from_omega_no_infinity():
    """Test that get_safe_step never returns infinity for any valid input."""
    import math
    
    grid_size = 20
    origin_x = -10.0
    origin_y = -10.0
    resolution = 1.0
    R_max = 10.0  # Estimated cluster radius
    
    omega = _init_omega_grid(grid_size, origin_x, origin_y, resolution)
    
    # Test various positions, including edge cases
    test_positions = [
        (0.0, 0.0),  # Center (seed location - can be 0.0)
        (100.0, 100.0),  # Far outside
        (-100.0, -100.0),  # Far negative
        (100.0, 0.0),  # Far right
        (0.0, 100.0),  # Far up
        (-100.0, 0.0),  # Far left
        (0.0, -100.0),  # Far down
    ]
    
    for x, y in test_positions:
        dist = get_safe_step(omega, x, y, origin_x, origin_y, resolution, R_max)
        assert math.isfinite(dist), f"Distance at ({x}, {y}) should be finite, got {dist}"
        # Distance can be 0.0 if at the seed location
        assert dist >= 0.0, f"Distance should be non-negative, got {dist}"


def test_simulator_no_nan_inf_particles():
    """Test that simulator doesn't create NaN/Inf ghost particles."""
    import math
    
    params = ContinuousRunParams(
        num_particles=20,
        particle_radius=0.5,
        grid_resolution=1.0,
        grid_padding=10,
        max_steps_per_particle=10_000,
        seed=42,
    )
    
    sim = ContinuousDLASimulator(params)
    
    # Run simulation for a small number of particles
    # We'll manually check after each particle is added
    initial_count = sim.num_particles
    
    # Run until we have a few more particles
    target_count = min(initial_count + 5, params.num_particles)
    while sim.num_particles < target_count:
        # Store current count
        old_count = sim.num_particles
        
        # Run one iteration (this will add one particle if successful)
        # We need to manually call the simulation logic
        # Instead, let's just run a very short simulation and check results
        break
    
    # Actually, let's just run a small full simulation and check
    sim.run()
    
    # Check all particles are valid
    for i in range(sim.num_particles):
        px = sim.px_array[i]
        py = sim.py_array[i]
        assert math.isfinite(px), f"Particle {i} x coordinate should be finite, got {px}"
        assert math.isfinite(py), f"Particle {i} y coordinate should be finite, got {py}"


def test_small_simulation_run():
    """Test a small simulation run to ensure no NaN/Inf issues."""
    import math
    
    params = ContinuousRunParams(
        num_particles=50,
        particle_radius=0.5,
        grid_resolution=1.0,
        grid_padding=20,
        max_steps_per_particle=50_000,
        seed=123,
    )
    
    sim = ContinuousDLASimulator(params)
    sim.run()
    
    # Verify all particles have valid coordinates
    positions = sim.get_positions()
    assert len(positions) == params.num_particles, "Should have correct number of particles"
    
    for i, (px, py) in enumerate(positions):
        assert math.isfinite(px), f"Particle {i} x coordinate should be finite, got {px}"
        assert math.isfinite(py), f"Particle {i} y coordinate should be finite, got {py}"
        assert abs(px) < 1e6, f"Particle {i} x coordinate should be reasonable, got {px}"
        assert abs(py) < 1e6, f"Particle {i} y coordinate should be reasonable, got {py}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

