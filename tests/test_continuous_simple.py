"""
Simplified test to identify the exact issue.
"""

import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dla_sim.continuous_dla import ContinuousDLASimulator, ContinuousRunParams


def test_minimal_simulation():
    """Minimal test - just try to add ONE particle."""
    print("Testing minimal simulation (1 particle)...")
    
    params = ContinuousRunParams(
        num_particles=1,  # Just ONE particle
        particle_radius=0.5,
        grid_resolution=1.0,
        grid_padding=10,
        max_steps_per_particle=100,  # Very small limit
        seed=42,
    )
    
    sim = ContinuousDLASimulator(params)
    print(f"R_birth={sim.R_birth}, R_death={sim.R_death}, R_max={sim.R_max}")
    
    start = time.time()
    try:
        sim.run()
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.2f}s")
        print(f"Particles: {sim.num_particles}/{params.num_particles}")
        return elapsed < 2.0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_minimal_simulation()

