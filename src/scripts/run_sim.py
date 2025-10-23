# src/scripts/run_sim.py
import argparse
from dla_sim import lattice, utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500, help="number of particles")
    parser.add_argument("--radius", type=int, default=100, help="grid radius")
    parser.add_argument("--out", default="results/cluster.npz", help="output file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    utils.set_seed(args.seed)
    occupied = lattice.run_simple_dla(num_particles=args.num, radius=args.radius)
    meta = {"num": args.num, "radius": args.radius, "seed": args.seed}
    utils.save_cluster(args.out, occupied, meta)
    print(f"✅ Cluster saved to {args.out}")

if __name__ == "__main__":
    main()
