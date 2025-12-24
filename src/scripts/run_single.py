#!/usr/bin/env python3
"""
Single DLA Simulation Runner

A clean, standardized CLI for running individual DLA simulations.
Supports three core models: lattice, offlattice, and hybrid.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dla_sim import (
    LatticeSimulator,
    LatticeConfig,
    BellOffSimulator,
    BellOffParams,
    HybridSimulator,
    HybridParams,
    utils,
)


def run_lattice_simulation(N: int, seed: int) -> tuple:
    """Run on-lattice DLA simulation."""
    config = LatticeConfig(seed=seed)
    simulator = LatticeSimulator(config)
    simulator.run(max_mass=N)
    coords = simulator.get_centered_coords()
    return coords, {"model": "lattice", "num_particles": N, "seed": seed}


def run_offlattice_simulation(N: int, seed: int) -> tuple:
    """Run off-lattice DLA simulation."""
    params = BellOffParams(num_particles=N, seed=seed)
    simulator = BellOffSimulator(params)
    simulator.run()
    coords = simulator.get_centered_coords()
    return coords, {"model": "offlattice", "num_particles": N, "seed": seed}


def run_hybrid_simulation(N: int, seed: int) -> tuple:
    """Run hybrid DLA simulation."""
    params = HybridParams(num_particles=N, seed=seed)
    simulator = HybridSimulator(params)
    simulator.run()
    coords = simulator.get_centered_coords()
    return coords, {"model": "hybrid", "num_particles": N, "seed": seed}


def main():
    parser = argparse.ArgumentParser(
        description="Run a single DLA simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["lattice", "offlattice", "hybrid"],
        required=True,
        help="Simulation model to use",
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Number of particles to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz file path (auto-generated if not provided)",
    )

    args = parser.parse_args()

    # Set seed globally
    utils.set_seed(args.seed)

    # Run simulation
    print(f"Running {args.model} simulation: N={args.N}, seed={args.seed}")
    start_time = time.time()

    if args.model == "lattice":
        coords, meta = run_lattice_simulation(args.N, args.seed)
    elif args.model == "offlattice":
        coords, meta = run_offlattice_simulation(args.N, args.seed)
    elif args.model == "hybrid":
        coords, meta = run_hybrid_simulation(args.N, args.seed)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    elapsed_time = time.time() - start_time

    # Generate output path if not provided
    if args.out is None:
        timestamp = utils.now_str()
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        args.out = str(
            output_dir / f"{args.model}_N{args.N}_S{args.seed}_{timestamp}.npz"
        )

    # Save results
    utils.save_cluster(
        path=args.out,
        positions=coords,
        meta=meta,
    )

    # Print summary
    print(f"\n✅ Simulation completed successfully!")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    print(f"   Particles generated: {coords.shape[0]}")
    print(f"   Output saved to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

