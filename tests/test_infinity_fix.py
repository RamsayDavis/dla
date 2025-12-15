#!/usr/bin/env python3
"""
Standalone test script for infinity bug fixes.
Run this to verify the fixes work correctly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
import numpy as np

from src.dla_sim.continuous_dla import (
    get_distance_from_omega,
    _init_omega_grid,
    ContinuousDLASimulator,
    ContinuousRunParams,
)


def test_get_distance_from_omega_finite():
    """Test that get_distance_from_omega never returns infinity."""
    print("Testing get_distance_from_omega...")
    
    grid_size = 10
    origin_x = -5.0
    origin_y = -5.0
    resolution = 1.0
    
    omega = _init_omega_grid(grid_size, origin_x, origin_y, resolution)
    
    # Test various positions
    test_cases = [
        (0.0, 0.0, "center"),
        (100.0, 100.0, "far outside"),
        (-100.0, -100.0, "far negative"),
        (100.0, 0.0, "far right"),
        (0.0, 100.0, "far up"),
    ]
    
    all_passed = True
    for x, y, desc in test_cases:
        dist = get_distance_from_omega(omega, x, y, origin_x, origin_y, resolution)
        is_finite = math.isfinite(dist)
        is_inf = math.isinf(dist)
        
        if not is_finite or is_inf:
            print(f"  FAIL: {desc} at ({x}, {y}) returned {dist} (finite={is_finite}, inf={is_inf})")
            all_passed = False
        elif x == 100.0 or y == 100.0 or x == -100.0 or y == -100.0:
            # Out of bounds should return 20.0
            if dist != 20.0:
                print(f"  WARN: {desc} at ({x}, {y}) returned {dist}, expected 20.0")
        else:
            print(f"  PASS: {desc} at ({x}, {y}) returned {dist} (finite)")
    
    if all_passed:
        print("✓ All get_distance_from_omega tests passed!")
    return all_passed


def test_small_simulation():
    """Test a small simulation to ensure no NaN/Inf particles."""
    print("\nTesting small simulation (100 particles)...")
    
    params = ContinuousRunParams(
        num_particles=100,
        particle_radius=0.5,
        grid_resolution=1.0,
        grid_padding=20,
        max_steps_per_particle=50_000,
        seed=42,
    )
    
    try:
        sim = ContinuousDLASimulator(params)
        sim.run()
        
        # Check all particles
        positions = sim.get_positions()
        print(f"  Generated {len(positions)} particles")
        
        invalid_count = 0
        for i, (px, py) in enumerate(positions):
            if not (math.isfinite(px) and math.isfinite(py)):
                print(f"  FAIL: Particle {i} has invalid coordinates: ({px}, {py})")
                invalid_count += 1
            elif abs(px) > 1e6 or abs(py) > 1e6:
                print(f"  WARN: Particle {i} has very large coordinates: ({px}, {py})")
        
        if invalid_count == 0:
            print("✓ All particles have valid coordinates!")
            print(f"  Position range: x=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
                  f"y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
            return True
        else:
            print(f"✗ Found {invalid_count} particles with invalid coordinates")
            return False
            
    except Exception as e:
        print(f"✗ Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Infinity Bug Fixes")
    print("=" * 60)
    
    test1 = test_get_distance_from_omega_finite()
    test2 = test_small_simulation()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

