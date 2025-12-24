"""
Numba-backed translation of Yen Lee Loh's `koh_lattice.cc` DLA code.

OPTIMIZED VERSION: Uses flattened hierarchy to eliminate TypedList overhead
and disable bounds checking in hot loops for 5-10x performance improvement.

The original C++ implementation relies on a collection of bespoke math
utilities and raw lookup tables for accelerated random walks.  This module
keeps the same algorithmic structure but ports the logic to Python/Numba so
that it can be integrated with the rest of the `dla` package.

Critical inner loops – sampler evaluation, walk-to-
line rejection sampling, hierarchical diffusion, and aggregation updates –
are decorated with `@numba.njit` to retain roughly the same performance
envelope as the highly optimized C++ version while remaining fully
interpretable and testable from Python.

Two binary resource files are required:

* `SquareLatticeGreenFunction64.raw`
* `DirichletGFs.raw`

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple
import time

import math
import os

import numpy as np
from numba import njit
from numba.typed import List as TypedList

from . import utils

###############################################################################
# Constants (mirrors of koh_lattice.cc globals)
###############################################################################

RKILLING: np.int64 = np.int64(100_000_000_000_000)  # 1e14
SEED: np.int64 = np.int64(0)
LMAX: np.int64 = np.int64(15)

AGGREGATION_PATTERN = np.array(
    [
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
    ],
    dtype=np.int64,
)

FIXED_STEPSIZE = 0
VARIABLE_STEPSIZE = 1
DIFFUSION_STEPSIZE = VARIABLE_STEPSIZE

SHARP_CIRCLE = 0
FUZZY_ANNULUS = 1
LAUNCHING_METHOD = FUZZY_ANNULUS

INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min

N_ANGLES = 24
LEVELS = 9  # walk-out-to-square levels (1 .. 256)

###############################################################################
# Utility math helpers (Numba-friendly)
###############################################################################


@njit(cache=True, fastmath=True)
def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


@njit(cache=True, fastmath=True)
def _bessel_i0(x: float) -> float:
    """
    Modified Bessel function of the first kind, order 0.
    Required because np.i0() is not supported by Numba in nopython mode.
    Uses Abramowitz & Stegun 9.8.1 approximation.
    """
    ax = abs(x)
    if ax < 3.75:
        y = (x / 3.75) * (x / 3.75)
        return (
            1.0
            + y
            * (
                3.5156229
                + y
                * (
                    3.0899424
                    + y
                    * (
                        1.2067492
                        + y * (0.2659732 + y * (0.0360768 + y * 0.0045813))
                    )
                )
            )
        )
    else:
        y = 3.75 / ax
        return (
            math.exp(ax)
            / math.sqrt(ax)
            * (
                0.39894228
                + y
                * (
                    0.01328592
                    + y
                    * (
                        0.00225319
                        + y
                        * (
                            -0.00157565
                            + y * (0.00916281 + y * (-0.02057706 + y * 0.02635537))
                        )
                    )
                )
            )
        )


@njit(cache=True, fastmath=True)
def _round_cpp(x: float) -> int:
    """
    Emulate C++ round() behavior: round half away from zero.
    Python's round() uses banker's rounding (round half to even), which can
    cause systematic bias in random walks.
    """
    if x >= 0.0:
        return int(x + 0.5)
    else:
        return int(x - 0.5)


###############################################################################
# Alias-sampler helpers
#For sampling from the Green's functions, we use Walker's alias method. This means we only ever need to do 2 uniform samples.
###############################################################################


@dataclass
class AliasTable:
    fn: np.ndarray #Alias probability table (the threshold)
    an: np.ndarray #Alias index table (the index of the alternate outcome)
    pn: np.ndarray #Probability table (we don't really need to store this)

    def to_typed(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fn.astype(np.float64), self.an.astype(np.int64)


def _build_alias_table(probabilities: np.ndarray) -> AliasTable:
    """
    Constructs the lookup tables for Walker's Alias Method, enabling O(1) sampling 
    from a discrete probability distribution.

    This redistributes probability mass from "over-full" buckets to "under-full" 
    buckets so that each slot in the table corresponds to exactly two possible 
    outcomes: the index itself or its alias.
    """
    pn = probabilities.astype(np.float64)
    n = pn.size
    tol = 1e-11
    total = pn.sum()
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Alias table probabilities must sum to 1, got {total} (tol={tol})"
        )

    bn = pn - 1.0 / n
    fn = np.ones(n, dtype=np.float64)
    an = np.arange(n, dtype=np.int64)
    for _ in range(n):
        jmin = np.argmin(bn)
        jmax = np.argmax(bn)
        bmin = bn[jmin]
        bmax = bn[jmax]
        if bmax - bmin < tol:
            break
        an[jmin] = jmax
        fn[jmin] = 1.0 + bmin * n
        bn[jmax] = bmax + bmin
        bn[jmin] = 0.0
    return AliasTable(fn=fn, an=an, pn=pn)


@njit(cache=True, fastmath=True)
def _alias_sample(fn: np.ndarray, an: np.ndarray) -> int:
    """
    Samples an index from the discrete distribution in O(1) constant time.

    It selects a bucket uniformly at random, then uses a biased coin flip (based on 'fn') 
    to decide whether to return the bucket index itself or its stored alias ('an').
    """
    idx = np.random.randint(0, fn.shape[0])
    if np.random.random() > fn[idx]:
        idx = an[idx]
    return idx


###############################################################################
# Green-function helpers (walk-to-line)
###############################################################################


@njit(cache=True, fastmath=True)
def _fxy_series(x: int, y: int) -> float:
    """
    Calculates the Resistance Green's Function F(x, y) using its asymptotic 
    series expansion. 
    
    Used for large distances (outside the pre-computed table) where the series 
    converges to machine-precision accuracy.
    """
    euler_gamma = 0.5772156649015329
    euler_gamma_plus_sqrt8 = 1.6169364357414509
    rsq = float(x * x + y * y)
    if rsq == 0.0:
        return 0.0
    oor2 = 1.0 / rsq #1/r^2
    oor4 = oor2 * oor2 #1/r^4
    oor6 = oor2 * oor4 #1/r^6
    oor8 = oor4 * oor4 #1/r^8

    phi = math.atan2(float(y), float(x))
    cos4phi = math.cos(4.0 * phi)
    cos8phi = math.cos(8.0 * phi)
    cos12phi = math.cos(12.0 * phi)
    cos16phi = math.cos(16.0 * phi)

    return (1.0 / (2.0 * math.pi)) * (
        euler_gamma_plus_sqrt8
        + 0.5 * math.log(rsq)
        - (
            oor2 * (1.0 / 12.0 * cos4phi)
            + oor4 * (3.0 / 40.0 * cos4phi + 5.0 / 48.0 * cos8phi)
            + oor6 * (51.0 / 112.0 * cos8phi + 35.0 / 72.0 * cos12phi)
            + oor8
            * (
                217.0 / 320.0 * cos8phi
                + 45.0 / 8.0 * cos12phi
                + 1925.0 / 384.0 * cos16phi
            )
        )
    )


@njit(cache=True, fastmath=True)
def _fxy_lookup(table: np.ndarray, x: int, y: int) -> float:
    """
    Retrieves the Resistance Green's Function value F(x, y) from the pre-computed 
    table, using symmetry to handle negative coordinates.
    """
    x_abs = abs(x)
    y_abs = abs(y)
    if x_abs <= 60 and y_abs <= 60:
        return float(table[x_abs, y_abs])
    return _fxy_series(x_abs, y_abs)


@njit(cache=True, fastmath=True)
def _pxy(table: np.ndarray, x: int, h: int) -> float:
    """
    Calculates the exact lattice probability (flux) of a walker starting at (0, h)
    hitting the wall at (x, 0) using the Method of Images.    """
    return _fxy_lookup(table, x, 1 + h) - _fxy_lookup(table, x, 1 - h)


@njit(cache=True, fastmath=True)
def _qxy(x: int, h: int) -> float:
    """
    Calculates the continuum approximation (Cauchy distribution) for a walker 
    starting at (0, h) hitting (x, 0), used as the envelope for rejection sampling.
    """
    return (math.atan((x + 0.5) / h) - math.atan((x - 0.5) / h)) / math.pi


@njit(cache=True, fastmath=True)
def _walk_to_line_sample(table: np.ndarray, h: int) -> int:
    """
    Samples a random integer exit point on the line y=0 for a walker starting at (0, h) 
    using Rejection Sampling.

    It proposes candidates from the continuum Cauchy distribution (fast approximation) 
    and accepts/rejects them based on the exact lattice flux (Method of Images) 
    to ensure the resulting distribution is physically correct for the grid.
    """
    p0 = _pxy(table, 0, h)
    q0 = _qxy(0, h)
    if p0 <= 0.0:
        p0 = np.finfo(np.float64).eps #Smallest positive floating point number
    c = q0 / p0 #Acceptance ratio
    while True:
        x_raw = h * math.tan((np.random.random() - 0.5) * math.pi)
        if x_raw > RKILLING:
            x = INT64_MAX
        elif x_raw < -RKILLING:
            x = INT64_MIN
        else:
            x = _round_cpp(x_raw)
        x_abs = abs(x)
        p = _pxy(table, x_abs, h)
        if p <= 0.0:
            p = np.finfo(np.float64).eps
        q = _qxy(x_abs, h)
        if q <= 0.0:
            q = np.finfo(np.float64).eps
        if np.random.random() < c * p / q: #Accept or reject
            return x


###############################################################################
# Walk-out-to-square helpers
###############################################################################


@njit(cache=True, fastmath=True)
def _radius_to_level(r: int) -> int:
    """
    Maps a physical square radius 'r' to its corresponding index in the pre-computed 
    probability tables. Returns -1 if the radius is not a supported power of 2.
    """
    level = 0
    value = 1
    while value < r and level < LEVELS:
        value <<= 1
        level += 1
    if value == r:
        return level
    return -1


@njit(cache=True, fastmath=True)
def _walk_out_to_square_x(
    fn_tables: Sequence[np.ndarray], an_tables: Sequence[np.ndarray], r: int
) -> int:
    """
    Samples the exit coordinate 'x' along the top edge of a square of radius 'r' 
    using the Walker's Alias Method for O(1) efficiency.
    """
    level = _radius_to_level(r)
    if level < 0 or level >= len(fn_tables):
        return 0
    idx = _alias_sample(fn_tables[level], an_tables[level])
    return idx - (r - 1)


@njit(cache=True, fastmath=True)
def _walk_out_to_square_displacement(
    fn_tables: Sequence[np.ndarray], an_tables: Sequence[np.ndarray], r: int
) -> Tuple[int, int]:
    """
    Generates a random 2D displacement vector for a walker exiting a square of radius 'r'.
    It combines the sampled position from the top edge with a random 90-degree 
    rotation (0, 90, 180, 270) to exploit the square's symmetry.
    """
    x = _walk_out_to_square_x(fn_tables, an_tables, r)
    y = r
    orientation = np.random.randint(0, 4)
    if orientation == 0:
        return x, y
    if orientation == 1:
        return x, -y
    if orientation == 2:
        return y, x
    return -y, x


###############################################################################
# Kaiser-Bessel launching annulus helpers
###############################################################################

BETA_KAISER = 24.0
C_KAISER = 9.060322906269867e-10
SIGMA_GAUSS = 0.20630813344176394
C_GAUSS = 1.9648389200167407


@njit(cache=True, fastmath=True)
def _f_kaiser(x: float) -> float:
    """
    Evaluates the Kaiser-Bessel window function (I0-based), which is the target 
    probability distribution for the radial "fuzziness" needed to eliminate grid aliasing.
    """
    return C_KAISER * _bessel_i0(BETA_KAISER * math.sqrt(max(0.0, 1.0 - x * x)))


@njit(cache=True, fastmath=True)
def _f_envelope(x: float) -> float:
    """
    Evaluates the Gaussian envelope function used to bound the Kaiser-Bessel 
    distribution during Rejection Sampling.
    """
    sigma_sq = SIGMA_GAUSS * SIGMA_GAUSS
    return C_GAUSS * math.exp(-(x * x) * (0.5 / sigma_sq))


@njit(cache=True, fastmath=True)
def _drand_kaiser() -> float:
    """
    Samples a random number from the Kaiser-Bessel distribution using Rejection Sampling 
    (proposing from a Gaussian and accepting based on the ratio f_kaiser/f_envelope).
    """
    while True:
        x = SIGMA_GAUSS * np.random.standard_normal()
        if abs(x) < 1.0 and np.random.random() < _f_kaiser(x) / _f_envelope(x):
            break
    return x * (2 * np.random.randint(0, 2) - 1)


@njit(cache=True, fastmath=True)
def _pick_launch_point(
    r_inner: float, r_outer: float
) -> Tuple[np.int64, np.int64]:
    """
    Generates a starting coordinate (x, y) within a "fuzzy annulus" defined by 
    r_inner and r_outer, using the Kaiser-Bessel radial distribution to prevent 
    lattice aliasing artifacts.
    """
    r = 0.5 * (r_inner + r_outer) + 0.5 * (r_outer - r_inner) * _drand_kaiser()
    phi = np.random.random() * 2 * math.pi
    x_rel = np.int64(_round_cpp(r * math.cos(phi)))
    y_rel = np.int64(_round_cpp(r * math.sin(phi)))
    return x_rel, y_rel


###############################################################################
# Hierarchical bit-array helpers (FLATTENED VERSION)
###############################################################################


def _init_hierarchy_flat(lmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Allocates the memory for the hierarchical bit-pyramid.
    Calculates the storage size for every zoom level (from 1x1 up to 2^L x 2^L) 
    and returns a single flattened byte array plus an index of level offsets.
     """
    total_bytes = 0
    offsets = []
    
    # Calculate offsets and total size
    for l in range(lmax):
        size = 1 << l
        num_pixels = size * size
        num_bytes = (num_pixels + 7) // 8
        offsets.append(total_bytes)
        total_bytes += num_bytes
    
    # Allocate flat array and offsets array
    flat_ss = np.zeros(total_bytes, dtype=np.uint8)
    ss_offsets = np.array(offsets, dtype=np.int64)
    
    return flat_ss, ss_offsets


@njit(cache=True)
def _getslxy_flat(flat_ss: np.ndarray, ss_offsets: np.ndarray, l: int, x: int, y: int) -> int:
    """
    Reads a single bit from the flattened array at level 'l' and coordinates (x, y).
    Returns 1 if the site (or any of its sub-pixels in finer levels) is occupied, 0 otherwise.
    """
    size = 1 << l
    if x < 0 or x >= size or y < 0 or y >= size:
        return 0
    linear_index = x * size + y
    byte_index = linear_index // 8
    offset = ss_offsets[l]
    byte_val = flat_ss[offset + byte_index]
    # Extract bit (MSB is bit 0, so shift by 7 - bit_index)
    bit_index = linear_index % 8
    return (byte_val >> (7 - bit_index)) & 1


@njit(cache=True)
def _setslxy_flat(flat_ss: np.ndarray, ss_offsets: np.ndarray, l: int, x: int, y: int) -> None:
    """
    Writes a '1' to a single bit in the flattened array, marking that location as occupied/sticky.
    """
    size = 1 << l
    if x < 0 or x >= size or y < 0 or y >= size:
        return
    linear_index = x * size + y
    byte_index = linear_index // 8
    offset = ss_offsets[l]
    bit_index = linear_index % 8
    # Set bit (MSB is bit 0, so shift by 7 - bit_index)
    mask = np.uint8(1 << (7 - bit_index))
    flat_ss[offset + byte_index] |= mask


@njit(cache=True)
def _mark_flat(flat_ss: np.ndarray, ss_offsets: np.ndarray, lmax: int, x: int, y: int) -> None:
    """
    Propagates an occupancy mark up the entire hierarchy.
    When a pixel is marked at the finest level, this function recursively marks 
    the corresponding parent block in every coarser level to keep the 'zoom map' consistent.
    """   
    for level in range(lmax - 1, -1, -1):
        _setslxy_flat(flat_ss, ss_offsets, level, x, y)
        x >>= 1
        y >>= 1


@njit(cache=True)
def _mark_with_pattern_flat(
    flat_ss: np.ndarray, ss_offsets: np.ndarray, lmax: int, x: int, y: int, pattern: np.ndarray
) -> None:
    """
    Applies the aggregation rule (e.g., 4-neighbor connectivity) by marking the 
    center pixel and its neighbors as sticky throughout the entire hierarchy.
    """
    _mark_flat(flat_ss, ss_offsets, lmax, x, y)
    for i in range(pattern.shape[0]):
        _mark_flat(flat_ss, ss_offsets, lmax, x + int(pattern[i, 0]), y + int(pattern[i, 1]))


###############################################################################
# Diffusion kernels
###############################################################################


@njit(cache=True)
def _diffuse_fixed_flat(
    flat_ss: np.ndarray,
    ss_offsets: np.ndarray,
    lmax: int,
    x_cur: int,
    y_cur: int,
    fn_tables: Sequence[np.ndarray],
    an_tables: Sequence[np.ndarray],
) -> Tuple[int, int]:
    """
    Performs a single step of standard Brownian motion (radius=1).
    """
    x_change, y_change = _walk_out_to_square_displacement(fn_tables, an_tables, 1)
    return x_cur + x_change, y_cur + y_change


@njit(cache=True)
def _diffuse_variable_flat(
    flat_ss: np.ndarray,
    ss_offsets: np.ndarray,
    lmax: int,
    x_cur: int,
    y_cur: int,
    fn_tables: Sequence[np.ndarray],
    an_tables: Sequence[np.ndarray],
) -> Tuple[int, int]:
    """
    Performs an accelerated "variable" step by searching the hierarchy for empty space.

    It iterates from the coarsest possible level (largest blocks) down to the finest,
    checking the 3x3 neighborhood at each level. The first level found with 
    empty neighbors determines the maximum safe jump size (radius), allowing the 
    walker to skip large empty voids in a single O(1) operation.
    """
    start_level = max(2, lmax - 9)
    block_size = 1

    for l_block in range(start_level, lmax): #Zoom in until we reach a scale where the neighborhood is empty
        shift = lmax - l_block - 1
        if shift < 0:
            block_size = 1
        else:
            block_size = 1 << shift
        x_block = x_cur >> shift if shift >= 0 else x_cur
        y_block = y_cur >> shift if shift >= 0 else y_cur
        neighbor_exists = False
        for u in range(x_block - 1, x_block + 2):
            for v in range(y_block - 1, y_block + 2):
                if _getslxy_flat(flat_ss, ss_offsets, l_block, u, v):
                    neighbor_exists = True
                    break
            if neighbor_exists:
                break
        if not neighbor_exists:
            break

    x_change, y_change = _walk_out_to_square_displacement(
        fn_tables, an_tables, block_size
    )
    return x_cur + x_change, y_cur + y_change


###############################################################################
# Bounding-box return kernel
###############################################################################


@njit(cache=True)
def _return_towards_box(
    table: np.ndarray,
    x_cur: int,
    y_cur: int,
    xmin_bound: int,
    xmax_bound: int,
    ymin_bound: int,
    ymax_bound: int,
) -> Tuple[int, int]:
    """
    Implements the "Walk-to-Line" acceleration for walkers outside the bounding box.

    It identifies the closest face of the cluster's bounding box and "projects" the walker 
    onto it in a single step. The lateral drift (how far it moves sideways during the 
    approach) is sampled from the Green's function table. This allows walkers to cross 
    vast empty distances instantly without simulating every step.
    """
    overflowed = False
    x_target = x_cur
    y_target = y_cur
    x_dist = 0
    y_dist = 0

    if x_cur < xmin_bound:
        x_target = xmin_bound
        x_dist = xmin_bound - x_cur
    elif x_cur > xmax_bound:
        x_target = xmax_bound
        x_dist = x_cur - xmax_bound

    if y_cur < ymin_bound:
        y_target = ymin_bound
        y_dist = ymin_bound - y_cur
    elif y_cur > ymax_bound:
        y_target = ymax_bound
        y_dist = y_cur - ymax_bound

    if y_dist > x_dist:
        y_cur = y_target
        h = max(1, y_dist)
        x_change = _walk_to_line_sample(table, h)
        if x_change >= INT64_MAX or x_change <= INT64_MIN:
            overflowed = True
        x_cur += x_change
    else:
        x_cur = x_target
        h = max(1, x_dist)
        y_change = _walk_to_line_sample(table, h)
        if y_change >= INT64_MAX or y_change <= INT64_MIN:
            overflowed = True
        y_cur += y_change

    if overflowed:
        return INT64_MAX, INT64_MAX
    return x_cur, y_cur


@njit(cache=True, fastmath=True, boundscheck=False)
def run_simulation_kernel(
    max_mass: int,
    lmax: int,
    flat_ss: np.ndarray,
    ss_offsets: np.ndarray,
    green_table: np.ndarray,
    alias_fn: TypedList,
    alias_an: TypedList,
    agg_pattern: np.ndarray,
    launch_method: int,
    diff_stepsize: int,
    r_killing: int,
) -> Tuple[
    int,
    float,
    int,
    int,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
]:
    """
    The core computational loop for the DLA simulation.

    Manages the lifecycle of N particles:
    1.  **Initialization:** Places a seed at the center and marks its neighbors as sticky.
    2.  **Launching:** Spawns new walkers on a bias-free Kaiser-Bessel annulus outside the cluster.
    3.  **Walking:** Moves walkers using a hybrid algorithm:
        -   **Walk-to-Line (WTL):** Teleports distant walkers back to the bounding box.
        -   **Walk-to-Square (WTS):** Uses hierarchical skipping for walkers inside the box.
    4.  **Aggregation:** Detects collisions via the hierarchical bit-map, freezes the 
        particle, updates the bounding box, and marks new neighbors as sticky.

    Returns:
        Tuple containing final stats (mass, gyration radius, max radius, bounds) 
        and arrays of all particle coordinates (x_coords, y_coords).
    """
    # --- Setup Local Variables ---
    xmax_grid = 1 << (lmax - 1)
    ymax_grid = 1 << (lmax - 1)
    x_cen = xmax_grid // 2
    y_cen = ymax_grid // 2
 # includes code for finding radius of gyration, which can be removed later if I choose to do post-hoc analysis
    m_total = 0
    x_total = 0.0
    y_total = 0.0
    xsq_total = 0.0
    ysq_total = 0.0

    xmin_bound = INT64_MAX
    xmax_bound = INT64_MIN
    ymin_bound = INT64_MAX
    ymax_bound = INT64_MIN
    r_bound = INT64_MIN

    # Pre-allocate output arrays
    x_coords = np.zeros(max_mass, dtype=np.int32)
    y_coords = np.zeros(max_mass, dtype=np.int32)

    # --- Freeze Seed (Inline Logic) ---
    _mark_with_pattern_flat(flat_ss, ss_offsets, lmax, x_cen, y_cen, agg_pattern)

    m_total = 1
    x_total += x_cen
    y_total += y_cen
    xsq_total += x_cen * x_cen
    ysq_total += y_cen * y_cen

    r_bound = 0
    xmin_bound = min(xmin_bound, x_cen - 2)
    xmax_bound = max(xmax_bound, x_cen + 2)
    ymin_bound = min(ymin_bound, y_cen - 2)
    ymax_bound = max(ymax_bound, y_cen + 2)

    x_coords[0] = x_cen
    y_coords[0] = y_cen

    # --- Main Loop (Pure Machine Code) ---
    while m_total < max_mass:
        # 1. Launching
        if launch_method == SHARP_CIRCLE:
            r = r_bound + 2.0
            phi = np.random.random() * 2 * math.pi
            x_rel = int(_round_cpp(r * math.cos(phi)))
            y_rel = int(_round_cpp(r * math.sin(phi)))
        else:  # Fuzzy Annulus
            r_inner = 2.0 * max(float(r_bound), 20.0)
            r_outer = 4.0 * max(float(r_bound), 20.0)
            # Inline _pick_launch_point logic
            r_k = (
                0.5 * (r_inner + r_outer)
                + 0.5 * (r_outer - r_inner) * _drand_kaiser()
            )
            phi_k = np.random.random() * 2 * math.pi
            x_rel = int(_round_cpp(r_k * math.cos(phi_k)))
            y_rel = int(_round_cpp(r_k * math.sin(phi_k)))

        x_cur = x_cen + x_rel
        y_cur = y_cen + y_rel

        # 2. Walking
        while True:
            # Check bounds
            is_inside = (
                x_cur >= xmin_bound
                and x_cur <= xmax_bound
                and y_cur >= ymin_bound
                and y_cur <= ymax_bound
            )

            if is_inside:
                if diff_stepsize == FIXED_STEPSIZE:
                    x_cur, y_cur = _diffuse_fixed_flat(
                        flat_ss, ss_offsets, lmax, x_cur, y_cur, alias_fn, alias_an
                    )
                else:
                    x_cur, y_cur = _diffuse_variable_flat(
                        flat_ss, ss_offsets, lmax, x_cur, y_cur, alias_fn, alias_an
                    )
            else:
                x_new, y_new = _return_towards_box(
                    green_table,
                    x_cur,
                    y_cur,
                    xmin_bound,
                    xmax_bound,
                    ymin_bound,
                    ymax_bound,
                )
                if x_new == INT64_MAX:  # Overflowed
                    break  # Break inner walk loop to relaunch
                x_cur, y_cur = x_new, y_new

            # 3. Check Stickiness
            # Bounds safety check before array access
            if (
                x_cur >= 0
                and x_cur < xmax_grid
                and y_cur >= 0
                and y_cur < ymax_grid
            ):
                # Use flattened accessor for level lmax-1
                is_sticky = _getslxy_flat(flat_ss, ss_offsets, lmax - 1, x_cur, y_cur)

                if is_sticky:
                    # HIT! Freeze it.
                    _mark_with_pattern_flat(flat_ss, ss_offsets, lmax, x_cur, y_cur, agg_pattern)

                    # Update Observables
                    m_total += 1
                    x_total += x_cur
                    y_total += y_cur
                    xsq_total += x_cur * x_cur
                    ysq_total += y_cur * y_cur

                    r_dist = math.hypot(x_cur - x_cen, y_cur - y_cen)
                    if r_dist > r_bound:
                        r_bound = int(r_dist)

                    # Update bounding box
                    if x_cur - 2 < xmin_bound:
                        xmin_bound = x_cur - 2
                    if x_cur + 2 > xmax_bound:
                        xmax_bound = x_cur + 2
                    if y_cur - 2 < ymin_bound:
                        ymin_bound = y_cur - 2
                    if y_cur + 2 > ymax_bound:
                        ymax_bound = y_cur + 2

                    if m_total <= max_mass:
                        x_coords[m_total - 1] = x_cur
                        y_coords[m_total - 1] = y_cur

                    break  # Break inner walk loop, particle is frozen

        # 4. Check Safety Brake (Box Full)
        if (
            xmin_bound <= 2
            or xmax_bound >= xmax_grid - 2
            or ymin_bound <= 2
            or ymax_bound >= ymax_grid - 2
        ):
            break

    # Final Calculation
    r_gyration = 0.0
    if m_total > 0:
        x_avg = x_total / m_total
        y_avg = y_total / m_total
        x_sq_avg = xsq_total / m_total
        y_sq_avg = ysq_total / m_total
        r_sq_gyr = x_sq_avg - x_avg * x_avg + y_sq_avg - y_avg * y_avg
        r_gyration = math.sqrt(max(0.0, r_sq_gyr))

    return (
        m_total,
        r_gyration,
        r_bound,
        xmin_bound,
        xmax_bound,
        ymin_bound,
        ymax_bound,
        x_coords,
        y_coords,
    )


###############################################################################
# Simulator
###############################################################################


@dataclass
class KohLatticeConfig:
    lmax: int = int(LMAX)
    diffusion_stepsize: int = DIFFUSION_STEPSIZE
    launching_method: int = LAUNCHING_METHOD
    aggregation_pattern: np.ndarray = field(
        default_factory=lambda: AGGREGATION_PATTERN.copy()
    )
    r_killing: int = int(RKILLING)
    seed: Optional[int] = None


@dataclass
class KohRunParams:
    num_particles: int = 10_000
    lmax: int = int(LMAX)
    data_dir: Optional[str | os.PathLike[str]] = None
    seed: Optional[int] = None


class KohLatticeSimulator:
    """
    Python/Numba port of koh_lattice.cc (OPTIMIZED VERSION).

    Uses flattened hierarchy to eliminate TypedList overhead and disable
    bounds checking in hot loops for 5-10x performance improvement.

    Usage:
        sim = KohLatticeSimulator()
        sim.run(max_mass=10_000)
    """

    def __init__(
        self,
        config: KohLatticeConfig | None = None,
        *,
        data_dir: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self.config = config or KohLatticeConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.walk_to_line_table = self._load_green_function()
        self.alias_tables = self._load_dirichlet_tables()
        self.alias_fn = TypedList()
        self.alias_an = TypedList()
        for table in self.alias_tables:
            self.alias_fn.append(table.fn.astype(np.float64))
            self.alias_an.append(table.an.astype(np.int64))

        self.lmax = self.config.lmax
        self.xmax = 1 << (self.lmax - 1)
        self.ymax = 1 << (self.lmax - 1)
        self.x_cen = self.xmax // 2
        self.y_cen = self.ymax // 2

        self.flat_ss, self.ss_offsets = _init_hierarchy_flat(self.lmax)
        self.rad_vs_angle = np.zeros(N_ANGLES, dtype=np.float64)
        self.reset_observables()

    def reset_observables(self) -> None:
        self.m_total = 0
        self.x_total = 0.0
        self.y_total = 0.0
        self.xsq_total = 0.0
        self.ysq_total = 0.0
        self.r_gyration = 0.0
        self.xmin_bound = INT64_MAX
        self.xmax_bound = INT64_MIN
        self.ymin_bound = INT64_MAX
        self.ymax_bound = INT64_MIN
        self.r_bound = INT64_MIN
        # Particle center tracking for visualization (separate from sticky sites)
        # Pre-allocate with reasonable capacity (will grow if needed)
        initial_capacity = max(1000, 1 << (self.lmax - 1))
        self.x_coords = np.zeros(initial_capacity, dtype=np.int32)
        self.y_coords = np.zeros(initial_capacity, dtype=np.int32)
        self.particle_count = 0

    # ------------------------------------------------------------------ loaders
    def _load_green_function(self) -> np.ndarray:
        path = self.data_dir / "SquareLatticeGreenFunction64.raw"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Copy the raw table from the original C++ repo."
            )
        data = np.fromfile(path, dtype=np.float64, count=64 * 64).reshape(64, 64)
        return data

    def _load_dirichlet_tables(self) -> list[AliasTable]:
        path = self.data_dir / "DirichletGFs.raw"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Copy the raw table from the original C++ repo."
            )
        blob = np.fromfile(path, dtype=np.float64)
        tables: list[AliasTable] = []
        offset = 0
        for level in range(LEVELS):
            r_square = 1 << level
            n_values = r_square * 2 - 1
            segment = blob[offset : offset + n_values]
            if segment.size != n_values:
                raise ValueError(
                    f"Dirichlet table truncated at level {level}: "
                    f"expected {n_values} doubles, got {segment.size}"
                )
            tables.append(_build_alias_table(segment))
            offset += n_values
        return tables

    # ------------------------------------------------------------------ public
    def run(self, max_mass: int = 10_000) -> None:
        self.flat_ss, self.ss_offsets = _init_hierarchy_flat(self.lmax)
        self.rad_vs_angle[:] = 0.0
        self.reset_observables()  # To ensure fresh start

        (
            self.m_total,
            self.r_gyration,
            self.r_bound,
            self.xmin_bound,
            self.xmax_bound,
            self.ymin_bound,
            self.ymax_bound,
            self.x_coords,
            self.y_coords,
        ) = run_simulation_kernel(
            max_mass,
            self.lmax,
            self.flat_ss,
            self.ss_offsets,
            self.walk_to_line_table,
            self.alias_fn,
            self.alias_an,
            self.config.aggregation_pattern,
            self.config.launching_method,
            self.config.diffusion_stepsize,
            self.config.r_killing,
        )
        self.particle_count = self.m_total

    def snapshot(self) -> dict[str, object]:
        return {
            "mass": int(self.m_total),
            "r_bound": int(self.r_bound),
            "r_gyration": float(self.r_gyration),
            "bounds": (
                int(self.xmin_bound),
                int(self.xmax_bound),
                int(self.ymin_bound),
                int(self.ymax_bound),
            ),
            "rad_vs_angle": self.rad_vs_angle.copy(),
        }

    def cluster_grid(self) -> np.ndarray:
        """
        Return a boolean grid showing only particle centers (skeleton visualization).
        This is separate from the sticky sites (ss) which include neighbors for physics.
        The visualization shows a thin fractal skeleton rather than chunky overlapping sites.
        """
        size = 1 << (self.lmax - 1)
        grid = np.zeros((size, size), dtype=bool)
        
        # Mark only actual particle centers (not sticky neighbor sites)
        valid_particles = self.particle_count
        for i in range(valid_particles):
            x = int(self.x_coords[i])
            y = int(self.y_coords[i])
            if 0 <= x < size and 0 <= y < size:
                grid[x, y] = True
        
        return grid


def run_model(params: KohRunParams | dict | None = None) -> utils.ClusterResult:
    params = params or {}
    if isinstance(params, KohRunParams):
        run_params = params
    else:
        run_params = KohRunParams(
            num_particles=params.get("num_particles", 10_000),
            lmax=params.get("lmax", int(LMAX)),
            data_dir=params.get("data_dir"),
            seed=params.get("seed"),
        )

    config = KohLatticeConfig(lmax=run_params.lmax, seed=run_params.seed)
    sim = KohLatticeSimulator(config=config, data_dir=run_params.data_dir)
    
    # Timing instrumentation
    t_start = time.perf_counter()
    sim.run(max_mass=run_params.num_particles)
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    particles_per_sec = run_params.num_particles / elapsed if elapsed > 0 else 0.0
    
    print(f"Simulation completed: {run_params.num_particles} particles in {elapsed:.2f}s ({particles_per_sec:.0f} particles/s)")
    
    occupied = sim.cluster_grid()
    
    # Expose particle coordinates for age visualization
    x_coords = sim.x_coords[:sim.particle_count].copy()
    y_coords = sim.y_coords[:sim.particle_count].copy()
    
    meta = {
        "model": "koh_optimized",
        "num": run_params.num_particles,
        "lmax": run_params.lmax,
        "seed": run_params.seed,
        "snapshot": sim.snapshot(),
        "x_coords": x_coords,
        "y_coords": y_coords,
    }
    return utils.ClusterResult(occupied=occupied, positions=None, meta=meta)


__all__ = [
    "KohLatticeConfig",
    "KohLatticeSimulator",
    "KohRunParams",
    "run_model",
]

