from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from . import utils


@dataclass
class OffLatticeParams:
    num_particles: int = 1000
    Rb: float = 20.0
    Rd: float = 60.0
    step_size: float = 1.0
    particle_radius: float = 1.0
    max_attempts_per_particle: int = 100_000
    verbose: bool = True


def _spawn_on_circle(Rb):
    theta = np.random.uniform(0, 2.0 * math.pi)
    return np.array([Rb * math.cos(theta), Rb * math.sin(theta)], dtype=np.float64)


def _random_step(step_size):
    theta = np.random.uniform(0, 2.0 * math.pi)
    return np.array(
        [step_size * math.cos(theta), step_size * math.sin(theta)], dtype=np.float64
    )


def _dist_sq(a, b):
    d = a - b
    return d[0] * d[0] + d[1] * d[1]


def run_model(
    params: OffLatticeParams | dict | None = None,
) -> utils.ClusterResult:
    """
    Naive off-lattice DLA simulator returning a ClusterResult.
    """
    if params is None:
        params = OffLatticeParams()
    elif isinstance(params, dict):
        params = OffLatticeParams(**params)
    start_time = time.time()

    cluster = [np.array([0.0, 0.0], dtype=np.float64)]
    particles_added = 0
    steps_taken = 0
    Rd_sq = params.Rd * params.Rd
    capture_dist_sq = (2.0 * params.particle_radius) ** 2

    while particles_added < params.num_particles:
        w = _spawn_on_circle(params.Rb)
        attempts = 0
        while True:
            step = _random_step(params.step_size)
            w = w + step
            steps_taken += 1
            attempts += 1

            if w[0] * w[0] + w[1] * w[1] > Rd_sq:
                w = _spawn_on_circle(params.Rb)
                attempts = 0
                continue

            attached = False
            for p in cluster:
                if _dist_sq(w, p) <= capture_dist_sq:
                    cluster.append(w.copy())
                    particles_added += 1
                    attached = True
                    break
            if attached:
                break
            if attempts > params.max_attempts_per_particle:
                w = _spawn_on_circle(params.Rb)
                attempts = 0

        if params.verbose and (particles_added % max(1, params.num_particles // 10) == 0):
            elapsed = time.time() - start_time
            print(
                f"[off] Added {particles_added}/{params.num_particles}, steps={steps_taken}, elapsed={elapsed:.1f}s"
            )

    positions = np.vstack(cluster)
    meta = {
        "model": "offlattice",
        "num": int(params.num_particles),
        "Rb": float(params.Rb),
        "Rd": float(params.Rd),
        "step_size": float(params.step_size),
        "particle_radius": float(params.particle_radius),
        "seed": None,
        "time_elapsed": time.time() - start_time,
        "steps_taken": int(steps_taken),
    }

    return utils.ClusterResult(occupied=None, positions=positions, meta=meta)


def simulate_offlattice(
    num_particles=1000,
    Rb=20.0,
    Rd=60.0,
    step_size=1.0,
    particle_radius=1.0,
    max_attempts_per_particle=100000,
    save_path=None,
    verbose=True,
):
    """
    Backwards-compatible shim returning positions + meta and optionally saving.
    """
    result = run_model(
        {
            "num_particles": num_particles,
            "Rb": Rb,
            "Rd": Rd,
            "step_size": step_size,
            "particle_radius": particle_radius,
            "max_attempts_per_particle": max_attempts_per_particle,
            "verbose": verbose,
        }
    )
    if save_path:
        utils.save_cluster_result(save_path, result)
    return result.positions, result.meta
