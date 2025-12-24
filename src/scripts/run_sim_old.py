# src/scripts/run_sim.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dla_sim import (
    koh_lattice2,
    koh_lattice_optimized,
    lattice,
    offlattice,
    utils,
    continuous_dla,
    bell_off,
)  # type: ignore[import]
from src.scripts import plot_cluster  # type: ignore[import]


MODEL_REGISTRY = {
    "lattice": {
        "runner": lattice.run_model,
        "defaults": {"num_particles": 500, "radius": 100, "max_attempts": 10_000, "margin": 5},
        "cli": lambda args: {
            "num_particles": args.num,
            "radius": args.radius,
        },
    },
    "offlattice": {
        "runner": offlattice.run_model,
        "defaults": {
            "num_particles": 500,
            "Rb": 30.0,
            "Rd": 90.0,
            "step_size": 1.0,
            "particle_radius": 1.0,
            "max_attempts_per_particle": 100_000,
        },
        "cli": lambda args: {
            "num_particles": args.num,
            "Rb": args.Rb,
            "Rd": args.Rd,
            "step_size": args.step,
            "particle_radius": args.particle_radius,
            "max_attempts_per_particle": args.off_attempts,
            "verbose": not args.quiet,
        },
    },
    "koh": {
        "runner": koh_lattice2.run_model,
        "defaults": {"num_particles": 5000, "lmax": 10},
        "cli": lambda args: {
            "num_particles": args.num,
            "lmax": args.koh_lmax,
            "data_dir": args.koh_data,
            "seed": args.seed,
        },
    },
    "koh_optimized": {
        "runner": koh_lattice_optimized.run_model,
        "defaults": {"num_particles": 5000, "lmax": 10},
        "cli": lambda args: {
            "num_particles": args.num,
            "lmax": args.koh_lmax,
            "data_dir": args.koh_data,
            "seed": args.seed,
        },
    },
    "continuous": {
        "runner": continuous_dla.run_model,
        "defaults": {
            "num_particles": 10_000,
            "particle_radius": 0.5,
            "grid_resolution": 1.0,
            "grid_padding": 50,
            "R_death_multiplier": 100.0,
            "omega_update_radius": 40,
            "max_steps_per_particle": 1_000_000,
        },
        "cli": lambda args: {
            "num_particles": args.num,
            "particle_radius": args.particle_radius,
            "grid_resolution": args.continuous_grid_res,
            "grid_padding": args.continuous_grid_padding,
            "seed": args.seed,
        },
    },
    "bell_off": {
        "runner": bell_off.run_model,
        "defaults": {
            "num_particles": 10_000,
            "noise_reduction_factor": 1.0,
            "max_min_mesh": 0.0,
            "scale_of_points_grid": 0.0,
        },
        "cli": lambda args: {
            "num_particles": args.num,
            "noise_reduction_factor": 1.0,
            "seed": args.seed,
        },
    },
}


def build_model_config(model_name: str, args, file_cfg: Dict[str, Any]) -> Dict[str, Any]:
    entry = MODEL_REGISTRY[model_name]
    model_cfg: Dict[str, Any] = dict(entry.get("defaults", {}))
    file_models = file_cfg.get("models", {})
    if isinstance(file_models, dict) and model_name in file_models:
        model_cfg.update(file_models[model_name])
    model_cfg.update({k: v for k, v in file_cfg.items() if k in model_cfg})
    cli_overrides = entry["cli"](args)
    for key, value in cli_overrides.items():
        if value is not None:
            model_cfg[key] = value
    return model_cfg


def main():
    parser = argparse.ArgumentParser(description="Unified DLA runner")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="lattice",
        help="Which model to simulate",
    )
    parser.add_argument("--config", help="Optional JSON/TOML parameter file")
    parser.add_argument("--num", type=int, default=500, help="Target number of particles/mass")
    parser.add_argument("--radius", type=int, default=100, help="Grid radius for on-lattice model")
    parser.add_argument("--out", default=None, help="Output .npz path (auto-generated if not provided)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip automatic plot generation")
    parser.add_argument("--cmap", default="magma", help="Colormap for visualization (e.g., magma, plasma, inferno)")

    # off-lattice overrides
    parser.add_argument("--Rb", type=float, default=30.0, help="Birth radius (off-lattice)")
    parser.add_argument("--Rd", type=float, default=90.0, help="Death radius (off-lattice)")
    parser.add_argument("--step", type=float, default=1.0, help="Step size (off-lattice)")
    parser.add_argument("--particle-radius", type=float, default=1.0, help="Particle radius (off-lattice)")
    parser.add_argument("--off-attempts", type=int, default=100_000, help="Max attempts per particle (off-lattice)")
    parser.add_argument("--quiet", action="store_true", help="Silence off-lattice progress output")

    # koh-specific
    parser.add_argument("--koh-lmax", type=int, default=10, help="Hierarchy depth for Koh lattice")
    parser.add_argument(
        "--koh-data",
        default=str(Path("src/dla_sim/data")),
        help="Directory containing Koh lattice lookup tables",
    )
    
    # continuous-specific
    parser.add_argument("--continuous-grid-res", type=float, default=1.0, help="Grid resolution for continuous model")
    parser.add_argument("--continuous-grid-padding", type=int, default=50, help="Grid padding for continuous model")
    args = parser.parse_args()

    file_cfg: Dict[str, Any] = {}
    if args.config:
        file_cfg = utils.load_params(args.config)

    model_name = file_cfg.get("model", args.model)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")

    utils.set_seed(args.seed)
    model_cfg = build_model_config(model_name, args, file_cfg)
    result = MODEL_REGISTRY[model_name]["runner"](model_cfg)
    meta = result.ensure_meta()
    meta.setdefault("model", model_name)
    meta["seed"] = args.seed
    
    # Generate smart filename if not provided
    if args.out is None:
        timestamp = utils.now_str()
        num = model_cfg.get("num_particles", args.num)
        seed = args.seed
        
        # Extract model-specific parameters for filename
        if model_name == "koh" or model_name == "koh_optimized":
            lmax = model_cfg.get("lmax", args.koh_lmax)
            base_name = f"{model_name}_N{num}_L{lmax}_S{seed}_{timestamp}"
        elif model_name == "lattice":
            radius = model_cfg.get("radius", args.radius)
            base_name = f"{model_name}_N{num}_R{radius}_S{seed}_{timestamp}"
        elif model_name == "offlattice":
            Rb = int(model_cfg.get("Rb", args.Rb))
            Rd = int(model_cfg.get("Rd", args.Rd))
            base_name = f"{model_name}_N{num}_Rb{Rb}_Rd{Rd}_S{seed}_{timestamp}"
        elif model_name == "continuous":
            pr = model_cfg.get("particle_radius", args.particle_radius)
            base_name = f"{model_name}_N{num}_PR{pr:.2f}_S{seed}_{timestamp}"
        else:
            base_name = f"{model_name}_N{num}_S{seed}_{timestamp}"
        
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        args.out = str(results_dir / f"{base_name}.npz")
    
    # Save .npz file
    utils.save_cluster_result(args.out, result)
    print(f"✅ {model_name} cluster saved to {args.out}")
    
    # Auto-generate plot with same base filename + colormap
    if not args.no_plot:
        # Extract base name and add colormap before .png extension
        npz_path = Path(args.out)
        base_name = npz_path.stem  # filename without extension
        plot_path = npz_path.parent / f"{base_name}_{args.cmap}.png"
        plot_cluster.plot_cluster_from_result(result, save_path=str(plot_path), cmap=args.cmap)
        print(f"✅ Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
