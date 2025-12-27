#!/usr/bin/env python3
"""
Batch DLA Simulation Runner

Generates multiple DLA clusters in parallel for dataset creation.
Supports three core models: lattice, offlattice, and hybrid.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any

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


def run_single_simulation(
    model: str, N: int, seed: int, output_path: str
) -> Dict[str, Any]:
    """
    Run a single simulation and save it.
    
    This function is designed to be called in parallel by ProcessPoolExecutor.
    It must be at module level (not nested) for pickling.
    """
    # Set seed for this process
    utils.set_seed(seed)
    
    # Run simulation based on model
    if model == "lattice":
        config = LatticeConfig(seed=seed)
        simulator = LatticeSimulator(config)
        simulator.run(max_mass=N)
        coords = simulator.get_centered_coords()
    elif model == "offlattice":
        params = BellOffParams(num_particles=N, seed=seed)
        simulator = BellOffSimulator(params)
        simulator.run()
        coords = simulator.get_centered_coords()
    elif model == "hybrid":
        params = HybridParams(num_particles=N, seed=seed)
        simulator = HybridSimulator(params)
        simulator.run()
        coords = simulator.get_centered_coords()
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Prepare metadata
    meta = {
        "model": model,
        "num_particles": N,
        "seed": seed,
        "actual_particles": int(coords.shape[0]),
    }
    
    # Save cluster
    utils.save_cluster(
        path=output_path,
        positions=coords,
        meta=meta,
    )
    
    return {
        "output_path": output_path,
        "seed": seed,
        "particles": int(coords.shape[0]),
        "success": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a batch of DLA simulations",
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
        help="Number of particles per simulation",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of simulations to generate",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="batch",
        help="Batch name for output folder (default: 'batch')",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed (each simulation gets base_seed + index) (default: 42)",
    )

    args = parser.parse_args()

    # Calculate seed range
    first_seed = args.base_seed
    last_seed = args.base_seed + args.count - 1
    
    # Create output directory with model name, cluster size, and seed range
    timestamp = utils.now_str()
    batch_dir = Path("results") / "batches" / f"{args.model}_N{args.N}_S{first_seed}-{last_seed}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create manifest
    manifest = {
        "model": args.model,
        "num_particles": args.N,
        "count": args.count,
        "base_seed": args.base_seed,
        "jobs": args.jobs,
        "timestamp": timestamp,
        "batch_name": args.name,
    }

    # Save manifest
    manifest_path = batch_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Batch generation started:")
    print(f"  Model: {args.model}")
    print(f"  Particles per simulation: {args.N}")
    print(f"  Total simulations: {args.count}")
    print(f"  Parallel jobs: {args.jobs}")
    print(f"  Output directory: {batch_dir}")
    print(f"  Base seed: {args.base_seed}")
    print()

    # Prepare tasks
    tasks = []
    for i in range(args.count):
        seed = args.base_seed + i
        output_path = str(batch_dir / f"{seed}.npz")
        tasks.append((args.model, args.N, seed, output_path))

    # Run simulations in parallel
    start_time = time.time()
    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_simulation, *task): task
            for task in tasks
        }

        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"  [{completed}/{args.count}] Completed: seed={result['seed']}, "
                    f"particles={result['particles']}"
                )
            except Exception as e:
                failed.append({"task": task, "error": str(e)})
                print(f"  [{completed}/{args.count}] FAILED: {task[2]} - {e}")

    elapsed_time = time.time() - start_time

    # Update manifest with results
    manifest["results"] = {
        "total": args.count,
        "successful": len(results),
        "failed": len(failed),
        "elapsed_seconds": elapsed_time,
    }
    manifest["simulations"] = results
    if failed:
        manifest["failures"] = failed

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("Batch generation completed!")
    print(f"  Successful: {len(results)}/{args.count}")
    print(f"  Failed: {len(failed)}/{args.count}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    if len(results) > 0:
        print(f"  Average time per simulation: {elapsed_time/len(results):.2f} seconds")
    print(f"  Output directory: {batch_dir}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 60)

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

