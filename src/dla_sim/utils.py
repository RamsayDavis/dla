# src/dla_sim/utils.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


@dataclass
class ClusterResult:
    """Common container for DLA simulation outputs."""

    occupied: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

    def ensure_meta(self) -> Dict[str, Any]:
        if self.meta is None:
            self.meta = {}
        return self.meta


def set_seed(seed: int = 0) -> None:
    """Set random seed for reproducibility (global numpy RNG)."""
    np.random.seed(seed)


def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_cluster(path: str | os.PathLike[str], occupied=None, positions=None, meta=None):
    """
    Backwards compatible helper to save clusters to .npz.
    """
    result = ClusterResult(
        occupied=None if occupied is None else np.asarray(occupied),
        positions=None if positions is None else np.asarray(positions),
        meta=meta or {},
    )
    save_cluster_result(path, result)


def save_cluster_result(
    path: str | os.PathLike[str], result: ClusterResult, *, overwrite: bool = True
) -> None:
    """Serialize a ClusterResult to disk."""
    out_dir = Path(path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {}
    if result.occupied is not None:
        out["occupied"] = result.occupied.astype("uint8")
    if result.positions is not None:
        out["positions"] = np.asarray(result.positions, dtype=np.float64)
    
    # Save metadata, extracting numpy arrays to top level for easier access
    meta = result.meta or {}
    meta_clean = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            # Save numpy arrays at top level for easier loading
            out[key] = value
        else:
            meta_clean[key] = value
    out["meta"] = meta_clean
    
    if not overwrite and Path(path).exists():  # pragma: no cover - defensive
        raise FileExistsError(f"{path} already exists")
    np.savez_compressed(path, **out)


def load_cluster(path: str | os.PathLike[str]) -> ClusterResult:
    """
    Load cluster .npz into a ClusterResult.
    """
    data = np.load(path, allow_pickle=True)
    occupied = data["occupied"].astype(bool) if "occupied" in data else None
    positions = data["positions"].astype(float) if "positions" in data else None
    meta = None
    if "meta" in data:
        meta_raw = data["meta"]
        if hasattr(meta_raw, "item"):
            try:
                meta = meta_raw.item()
            except ValueError:
                meta = meta_raw
        else:
            meta = meta_raw
    
    # Load x_coords and y_coords if they exist at top level
    if meta is None:
        meta = {}
    if "x_coords" in data and "x_coords" not in meta:
        meta["x_coords"] = data["x_coords"]
    if "y_coords" in data and "y_coords" not in meta:
        meta["y_coords"] = data["y_coords"]
    
    return ClusterResult(occupied=occupied, positions=positions, meta=meta)


def load_params(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """
    Load simulation parameters from JSON or TOML for future workflow integration.
    """
    path = str(path)
    with open(path, "rb") as fh:
        data = fh.read()
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ""}:
        return json.loads(data.decode("utf-8"))
    if suffix in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("tomllib is unavailable; cannot parse TOML files")
        return tomllib.loads(data.decode("utf-8"))
    raise ValueError(f"Unsupported parameter file format: {suffix}")
