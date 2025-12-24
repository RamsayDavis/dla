from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import utils


@dataclass
class LatticeParams:
    num_particles: int = 1000
    radius: int = 150
    max_attempts: int = 10_000
    margin: int = 5


def make_grid(radius):
    """Create an empty square grid with a seed at the center."""
    size = 2 * radius + 1
    occupied = np.zeros((size, size), dtype=bool)
    center = size // 2
    occupied[center, center] = True
    return occupied, center


def spawn_on_circle(center, spawn_radius):
    """Return random coordinates on a circle around center."""
    theta = np.random.uniform(0, 2 * np.pi)
    x = int(round(center + spawn_radius * np.cos(theta)))
    y = int(round(center + spawn_radius * np.sin(theta)))
    return x, y


def is_neighbor_occupied(x, y, occupied):
    """Check 8-neighborhood for an occupied cell."""
    n = occupied.shape[0]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and occupied[nx, ny]:
                return True
    return False


def random_step(x, y, grid_size):
    """Move randomly one step (4-neighbor walk)."""
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dx, dy = moves[np.random.randint(len(moves))]
    x = min(max(0, x + dx), grid_size - 1)
    y = min(max(0, y + dy), grid_size - 1)
    return x, y


def simulate(params: LatticeParams | None = None) -> utils.ClusterResult:
    """
    Run the simple on-lattice DLA simulation and return a ClusterResult.
    """
    params = params or LatticeParams()
    occupied, center = make_grid(params.radius)
    grid_size = occupied.shape[0]
    stuck = 1

    while stuck < params.num_particles:
        coords = np.argwhere(occupied)
        maxd = np.max((coords - center) ** 2)
        spawn_radius = min(
            int(np.sqrt(maxd)) + params.margin, grid_size // 2 - 2
        )

        x, y = spawn_on_circle(center, spawn_radius)
        attempts = 0

        while attempts < params.max_attempts:
            if is_neighbor_occupied(x, y, occupied):
                occupied[x, y] = True
                stuck += 1
                break
            x, y = random_step(x, y, grid_size)
            attempts += 1
        else:
            # respawn walker if it failed to stick
            continue

    meta = {
        "model": "lattice",
        "num": params.num_particles,
        "radius": params.radius,
    }
    return utils.ClusterResult(occupied=occupied, positions=None, meta=meta)


def run_model(config: dict | None = None) -> utils.ClusterResult:
    params = config or {}
    return simulate(
        LatticeParams(
            num_particles=params.get("num_particles", 1000),
            radius=params.get("radius", 150),
            max_attempts=params.get("max_attempts", 10_000),
            margin=params.get("margin", 5),
        )
    )


def run_simple_dla(
    num_particles: int = 1000,
    radius: int = 150,
    max_attempts: int = 10_000,
    margin: int = 5,
):
    """
    Backwards-compatible wrapper returning the raw occupancy grid.
    """
    result = simulate(
        LatticeParams(
            num_particles=num_particles,
            radius=radius,
            max_attempts=max_attempts,
            margin=margin,
        )
    )
    return result.occupied
