# tests/test_core.py
from dla_sim import lattice, utils

def test_small_run():
    utils.set_seed(0)
    occ = lattice.run_simple_dla(num_particles=10, radius=20)
    assert occ.sum() >= 10
