# DLA (Diffusion-Limited Aggregation) Simulator

A high-performance Python library for simulating Diffusion-Limited Aggregation (DLA) clusters using three different algorithms: on-lattice, off-lattice, and hybrid models. This repository provides optimized implementations with Numba JIT compilation for fast simulations, along with analysis and visualization tools.

## Overview

Diffusion-Limited Aggregation is a process where particles undergo random walks and stick together to form fractal clusters. This repository implements three distinct simulation approaches:

1. **Lattice DLA** (`ongrid_sim.py`): High-performance on-lattice simulator with hierarchical acceleration
2. **Off-Lattice DLA** (`offgrid_sim.py`): Continuous space simulator using conformal mapping
3. **Hybrid DLA** (`hybrid_sim.py`): Combines off-lattice diffusion with on-lattice aggregation

### Key Features

- **Optimized Performance**: Numba JIT compilation for critical loops
- **Multiple Algorithms**: Three different simulation models to choose from
- **Batch Processing**: Parallel execution for generating large datasets
- **Analysis Tools**: Fractal dimension calculation and scaling analysis
- **Visualization**: High-quality plotting with customizable rendering

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd dla
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package in development mode:
```bash
pip install -e .
```

### Required Data Files

The lattice simulator requires Green's function lookup tables:
- `src/dla_sim/data/SquareLatticeGreenFunction64.raw` (for Walk-to-Line)
- `src/dla_sim/data/DirichletGFs.raw` (for Walk-to-Square)

These should already be included in the repository.

## Usage

### Running a Single Simulation

Use the `run_single.py` script to generate one DLA cluster:

```bash
python src/scripts/run_single.py --model <model> --N <particles> [options]
```

**Models:**
- `lattice`: On-lattice DLA simulation
- `offlattice`: Off-lattice DLA simulation  
- `hybrid`: Hybrid DLA simulation

**Examples:**

```bash
# Generate a lattice DLA cluster with 100,000 particles
python src/scripts/run_single.py --model lattice --N 100000 --seed 42

# Generate an off-lattice cluster with 50,000 particles
python src/scripts/run_single.py --model offlattice --N 50000 --seed 123

# Generate a hybrid cluster and save to specific location
python src/scripts/run_single.py --model hybrid --N 10000 --seed 42 --out results/my_cluster.npz
```

**Options:**
- `--model`: Simulation model (`lattice`, `offlattice`, or `hybrid`)
- `--N`: Number of particles to simulate
- `--seed`: Random seed for reproducibility (default: 42)
- `--out`: Output file path (auto-generated if not provided)

Output files are saved as compressed `.npz` files in the `results/` directory by default.

### Running Batch Simulations

Generate multiple clusters in parallel using `run_batch.py`:

```bash
python src/scripts/run_batch.py --model <model> --N <particles> --count <num> [options]
```

**Example:**

```bash
# Generate 100 lattice clusters with 10,000 particles each, using 4 parallel processes
python src/scripts/run_batch.py --model lattice --N 10000 --count 100 --jobs 4 --name my_dataset
```

**Options:**
- `--model`: Simulation model
- `--N`: Number of particles per simulation
- `--count`: Number of simulations to generate
- `--jobs`: Number of parallel processes (default: 1)
- `--name`: Batch name for output folder (default: 'batch')
- `--base-seed`: Base seed (each simulation gets `base_seed + index`)

Batch outputs are saved in `results/batches/<name>_<timestamp>/` with:
- Individual cluster files (`001.npz`, `002.npz`, ...)
- A `manifest.json` file with metadata

### Visualizing Clusters

Plot a saved cluster using `plot_cluster.py`:

```bash
python src/scripts/plot_cluster.py <path-to-npz-file> [options]
```

**Example:**

```bash
# Plot a cluster with default settings
python src/scripts/plot_cluster.py results/lattice_N100000_S42_20251224-175355.npz

# Save plot without displaying
python src/scripts/plot_cluster.py results/my_cluster.npz --save results/plot.png --no-show
```

**Options:**
- `--save`: Path to save the plot image
- `--no-show`: Don't display the plot interactively
- `--dpi`: Resolution for saved images (default: 300)
- `--cmap`: Colormap name (default: 'magma')

### Analyzing Clusters

Compute fractal dimension and scaling statistics with `analyse_cluster.py`:

```bash
python src/scripts/analyse_cluster.py <path-to-npz-file> [options]
```

**Example:**

```bash
# Analyze a cluster and display results
python src/scripts/analyse_cluster.py results/lattice_N100000_S42_20251224-175355.npz

# Analyze without plotting
python src/scripts/analyse_cluster.py results/my_cluster.npz --no-plot
```

This script computes:
- **Fractal Dimension (Df)**: From the scaling relation Rg ~ N^(1/Df)
- **Scaling Exponent (β)**: Slope of log(Rg) vs log(N)
- **R² Value**: Linearity of the scaling relationship

Expected fractal dimension for off-lattice DLA is typically around 1.71.

### Programmatic Usage

You can also use the simulators directly in Python:

```python
from src.dla_sim import (
    LatticeSimulator, LatticeConfig,
    BellOffSimulator, BellOffParams,
    HybridSimulator, HybridParams,
    utils
)

# Set random seed
utils.set_seed(42)

# Example: Lattice simulation
config = LatticeConfig(seed=42)
simulator = LatticeSimulator(config)
simulator.run(max_mass=10000)
coords = simulator.get_centered_coords()

# Example: Off-lattice simulation
params = BellOffParams(num_particles=10000, seed=42)
simulator = BellOffSimulator(params)
simulator.run()
coords = simulator.get_centered_coords()

# Example: Hybrid simulation
params = HybridParams(num_particles=10000, seed=42)
simulator = HybridSimulator(params)
simulator.run()
coords = simulator.get_centered_coords()

# Save results
utils.save_cluster(
    path="my_cluster.npz",
    positions=coords,
    meta={"model": "lattice", "num_particles": 10000, "seed": 42}
)

# Load results
result = utils.load_cluster("my_cluster.npz")
coords = result.positions
metadata = result.meta
```

## Project Structure

```
dla/
├── src/
│   ├── dla_sim/          # Core simulation library
│   │   ├── ongrid_sim.py      # Lattice DLA simulator
│   │   ├── offgrid_sim.py     # Off-lattice DLA simulator
│   │   ├── hybrid_sim.py      # Hybrid DLA simulator
│   │   ├── utils.py           # Utility functions (save/load, etc.)
│   │   ├── data/              # Green's function lookup tables
│   │   └── archive/           # Legacy implementations
│   └── scripts/           # Runnable scripts
│       ├── run_single.py      # Single simulation runner
│       ├── run_batch.py        # Batch simulation runner
│       ├── plot_cluster.py    # Visualization tool
│       └── analyse_cluster.py # Analysis tool
├── notebooks/             # Exploratory Jupyter notebooks
├── tests/                 # Unit tests
├── results/               # Simulation outputs (gitignored)
├── data/                  # Additional data files (gitignored)
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Package configuration
└── README.md             # This file
```

## Algorithm Details

### Lattice DLA (`LatticeSimulator`)

- **Algorithm**: Optimized square lattice DLA with hierarchical acceleration
- **Key Features**:
  - **Walk-to-Square (WTS)**: Multi-resolution grid for efficient empty space skipping
  - **Walk-to-Line (WTL)**: Green's function lookup for instant transport to cluster boundary
  - **Fuzzy Annulus Launching**: Kaiser-Bessel windowing to eliminate lattice artifacts
- **Performance**: Optimized with Numba JIT compilation
- **Reference**: Ported from Yen Lee Loh's `koh_lattice.cc`

### Off-Lattice DLA (`BellOffSimulator`)

- **Algorithm**: Continuous space DLA using conformal mapping
- **Key Features**:
  - Harmonic measure mapping from infinity to unit circle
  - Continuous random walk in complex plane
  - Grid-based neighbor detection for aggregation
- **Performance**: Numba-accelerated critical loops

### Hybrid DLA (`HybridSimulator`)

- **Algorithm**: Combines off-lattice diffusion with on-lattice aggregation
- **Key Features**:
  - Continuous random walk (off-lattice)
  - Grid-based aggregation (on-lattice)
  - Configurable grid spacing
- **Use Case**: Balances continuous dynamics with efficient aggregation

## Output Format

Simulations save results as compressed NumPy `.npz` files containing:

- `positions`: Array of particle coordinates (complex numbers or (N,2) array)
- `x_coords`, `y_coords`: Separate coordinate arrays (for lattice models)
- `meta`: Dictionary with metadata:
  - `model`: Simulation model name
  - `num_particles`: Target number of particles
  - `seed`: Random seed used
  - Additional model-specific parameters

## Testing

Run the test suite:

```bash
pytest tests/
```

## Dependencies

- `numpy`: Numerical computations
- `numba`: JIT compilation for performance
- `matplotlib`: Visualization
- `scipy`: Scientific computing (for analysis)
- `tqdm`: Progress bars
- `pytest`: Testing framework

## License

See `LICENSE` file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- Original lattice algorithm: Yen Lee Loh's `koh_lattice.cc`
- DLA theory: Witten & Sander (1981), "Diffusion-limited aggregation, a kinetic critical phenomenon"
