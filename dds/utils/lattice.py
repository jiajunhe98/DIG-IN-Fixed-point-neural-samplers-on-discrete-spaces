"""Site colourings for blocked (independent-set) updates on periodic lattices.

A blocked Gibbs / Metropolis sweep updates all sites of one colour at once,
which is only a valid kernel when no two sites of that colour interact.  On
a torus with even sides the checkerboard `(i + j) % 2` does that with two
colours.  With an odd side the wrap-around bond `(L-1) -- 0` joins two sites
of equal parity, the torus is no longer bipartite, and the checkerboard is
*silently wrong*: it updates interacting sites together and samples the
wrong distribution.

Every lattice this package benchmarks on is square, and 25 is odd, so this
is not hypothetical.  The construction below is exact for any side lengths:

* each axis gets a proper colouring of its cycle with values in {0, 1, 2}:
  the parity for even lengths, and for odd lengths the parity with the last
  site recoloured 2 (the cycle 0,1,0,1,...,0,1,2 has no equal neighbours,
  including across the wrap);
* the lattice colour is the sum over axes, mod 3.  Torus neighbours differ
  in exactly one axis by a proper-colouring step, which changes the sum by
  a nonzero amount mod 3, so it is a proper colouring of the torus.

Three is the chromatic number of a torus with an odd side, so a sweep costs
3 colour passes instead of 2 -- against 9 for a product colouring.  All-even
lattices keep the two-colour checkerboard.
"""

from __future__ import annotations

import numpy as np


def _cycle_colouring(length: int) -> np.ndarray:
    c = np.arange(length) % 2
    if length % 2 and length > 1:
        c[-1] = 2
    return c


def independent_groups(lattice_shape: tuple[int, ...]) -> np.ndarray:
    """Proper colouring of the periodic lattice, shape `lattice_shape`,
    values in {0, 1} (all sides even) or {0, 1, 2} (some side odd)."""
    shape = tuple(int(s) for s in lattice_shape)
    grids = np.meshgrid(*[_cycle_colouring(s) for s in shape], indexing="ij")
    if all(s % 2 == 0 for s in shape):
        return (sum(grids) % 2).astype(np.int64)
    return (sum(grids) % 3).astype(np.int64)


def check_independent(lattice_shape: tuple[int, ...], colour: np.ndarray) -> bool:
    """True iff no two torus neighbours share a colour."""
    for axis in range(colour.ndim):
        if lattice_shape[axis] > 1 and np.any(colour == np.roll(colour, 1, axis=axis)):
            return False
    return True
