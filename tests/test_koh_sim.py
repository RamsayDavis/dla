import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim.koh_lattice2 import KohLatticeConfig, KohLatticeSimulator


def test_koh_simulator_small_cluster():
    data_dir = Path(__file__).resolve().parents[1] / "src" / "dla_sim" / "data"
    assert data_dir.exists(), "koh lattice data directory missing"

    config = KohLatticeConfig(lmax=7, seed=123)
    sim = KohLatticeSimulator(config=config, data_dir=data_dir)
    sim.run(max_mass=64)

    snap = sim.snapshot()
    assert snap["mass"] >= 64

    grid = sim.cluster_grid()
    expected_size = 1 << (config.lmax - 1)
    assert grid.shape == (expected_size, expected_size)
    center = expected_size // 2
    assert grid[center, center], "seed center must remain occupied"
    assert np.count_nonzero(grid) >= snap["mass"]

    # radius of gyration should remain finite and positive after aggregation
    assert snap["r_gyration"] > 0.0

