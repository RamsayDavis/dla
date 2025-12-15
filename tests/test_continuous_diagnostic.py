"""
Diagnostic tests to identify infinite loop issues in continuous DLA.
"""

import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import math

from src.dla_sim.continuous_dla import (
    ContinuousDLASimulator,
    ContinuousRunParams,
    get_safe_step,
    _init_omega_grid,
    _simulate_particle,
)


def test_single_particle_simulation():
    """Test simulating just ONE particle to see if it gets stuck."""
    print("\n=== Testing Single Particle Simulation ===")
    
    params = ContinuousRunParams(
        num_particles=1,  # Just one particle
        particle_radius=0.5,
        grid_resolution=1.0,
        grid_padding=10,
        max_steps_per_particle=1000,  # Small limit for testing
        seed=42,
    )
    
    sim = ContinuousDLASimulator(params)
    print(f"Initial R_birth: {sim.R_birth}")
    print(f"Initial R_death: {sim.R_death}")
    print(f"Initial R_max: {sim.R_max}")
    
    # Try to simulate one particle manually
    print("\nAttempting to simulate one particle...")
    start_time = time.time()
    
    particle_idx, stuck = _simulate_particle(
        sim.px_array,
        sim.py_array,
        sim.num_particles,
        sim.Y,
        sim.omega,
        sim.grid_origin_x,
        sim.grid_origin_y,
        sim.params.grid_resolution,
        sim.params.particle_radius,
        sim.R_birth,
        sim.R_death,
        sim.params.omega_update_radius,
        1000,  # max_steps
    )
    
    elapsed = time.time() - start_time
    print(f"Particle simulation completed in {elapsed:.2f}s")
    print(f"Stuck: {stuck}, Index: {particle_idx}")
    
    if stuck:
        px = sim.px_array[particle_idx]
        py = sim.py_array[particle_idx]
        print(f"Particle stuck at: ({px:.3f}, {py:.3f})")
        dist = math.sqrt(px*px + py*py)
        print(f"Distance from origin: {dist:.3f}")
    else:
        print("Particle did NOT stick (reached max steps)")
    
    return elapsed < 5.0  # Should complete quickly


def test_omega_grid_values():
    """Test omega grid values to see if they're correct."""
    print("\n=== Testing Omega Grid Values ===")
    
    params = ContinuousRunParams(
        num_particles=1,
        particle_radius=0.5,
        grid_resolution=1.0,
        grid_padding=10,
        seed=42,
    )
    
    sim = ContinuousDLASimulator(params)
    
    # Check omega values at various positions
    test_positions = [
        (0.0, 0.0, "origin (seed)"),
        (1.0, 0.0, "1 unit right"),
        (2.0, 0.0, "2 units right (R_birth)"),
        (5.0, 0.0, "5 units right"),
    ]
    
    R_max_estimate = sim.R_birth / 2.0
    print(f"R_max_estimate: {R_max_estimate}")
    
    for x, y, desc in test_positions:
        dist = get_safe_step(
            sim.omega, x, y, sim.grid_origin_x, sim.grid_origin_y,
            sim.params.grid_resolution, R_max_estimate
        )
        actual_dist = math.sqrt(x*x + y*y)
        print(f"{desc:25s}: omega={dist:.3f}, actual={actual_dist:.3f}, diff={abs(dist-actual_dist):.3f}")


def test_particle_trajectory():
    """Track a particle's trajectory to see where it gets stuck."""
    print("\n=== Testing Particle Trajectory ===")
    
    params = ContinuousRunParams(
        num_particles=1,
        particle_radius=0.5,
        grid_resolution=1.0,
        grid_padding=10,
        max_steps_per_particle=100,  # Very small for debugging
        seed=42,
    )
    
    sim = ContinuousDLASimulator(params)
    
    # Launch particle
    theta = np.random.random() * 2.0 * math.pi
    wx = sim.R_birth * math.cos(theta)
    wy = sim.R_birth * math.sin(theta)
    
    print(f"Starting position: ({wx:.3f}, {wy:.3f}), R_birth={sim.R_birth:.3f}")
    
    # Simulate a few steps manually
    steps = 0
    positions = [(wx, wy)]
    
    while steps < 20:  # Just 20 steps for debugging
        steps += 1
        r_current = math.sqrt(wx * wx + wy * wy)
        
        R_max_estimate = sim.R_birth / 2.0
        dist = get_safe_step(
            sim.omega, wx, wy, sim.grid_origin_x, sim.grid_origin_y,
            sim.params.grid_resolution, R_max_estimate
        )
        
        print(f"Step {steps:2d}: pos=({wx:7.3f}, {wy:7.3f}), r={r_current:.3f}, dist={dist:.3f}", end="")
        
        if dist > 5.0:
            L = dist - 1.0
            print(f" [FAR] L={L:.3f}")
        elif dist > 1.0:
            L = max(1.0, dist - 2.0)
            print(f" [MID] L={L:.3f}")
        else:
            L = 0.25
            print(f" [NEAR] L={L:.3f}")
        
        step_theta = np.random.random() * 2.0 * math.pi
        wx += L * math.cos(step_theta)
        wy += L * math.sin(step_theta)
        positions.append((wx, wy))
        
        # Check if stuck in loop
        if len(positions) > 5:
            recent = positions[-5:]
            if all(abs(p[0] - recent[0][0]) < 0.1 and abs(p[1] - recent[0][1]) < 0.1 for p in recent):
                print("  *** POTENTIAL LOOP DETECTED ***")
                break


def test_collision_detection_near_seed():
    """Test if collision detection works when particle is near seed."""
    print("\n=== Testing Collision Detection Near Seed ===")
    
    params = ContinuousRunParams(
        num_particles=1,
        particle_radius=0.5,
        grid_resolution=1.0,
        seed=42,
    )
    
    sim = ContinuousDLASimulator(params)
    
    # Seed is at (0, 0)
    # Test positions near seed
    test_positions = [
        (0.5, 0.0, "0.5 units right"),
        (1.0, 0.0, "1.0 units right"),
        (1.1, 0.0, "1.1 units right (should capture)"),
        (0.0, 1.0, "1.0 units up"),
    ]
    
    capture_dist = 2.0 * params.particle_radius + 0.1  # 1.1
    
    for wx, wy, desc in test_positions:
        dist_to_seed = math.sqrt(wx*wx + wy*wy)
        should_capture = dist_to_seed < capture_dist
        
        # Check direct distance
        dx = 0.0 - wx  # seed at (0,0)
        dy = 0.0 - wy
        dist_sq = dx*dx + dy*dy
        capture_dist_sq = capture_dist * capture_dist
        
        print(f"{desc:25s}: dist={dist_to_seed:.3f}, should_capture={should_capture}, "
              f"dist_sq={dist_sq:.3f}, threshold={capture_dist_sq:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Diagnostic Tests for Continuous DLA Infinite Loop Issue")
    print("=" * 60)
    
    try:
        test_omega_grid_values()
        test_collision_detection_near_seed()
        test_particle_trajectory()
        result = test_single_particle_simulation()
        
        if result:
            print("\n✓ Single particle test completed quickly")
        else:
            print("\n✗ Single particle test took too long (potential infinite loop)")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

