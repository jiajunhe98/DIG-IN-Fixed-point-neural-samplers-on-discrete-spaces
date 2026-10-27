"""NeRF placement, dihedrals and angle wrapping, batched over leading dimensions."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def _normalize(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def nerf(pj: Tensor, pk: Tensor, pl: Tensor, r: Tensor, theta: Tensor, phi: Tensor) -> Tensor:
    """Place atom i bonded to j at distance r, angle i-j-k = theta, dihedral i-j-k-l = phi
    (Parsons et al. 2005)."""
    bc = _normalize(pj - pk)
    n = _normalize(torch.cross(pk - pl, bc, dim=-1))
    m = torch.cross(n, bc, dim=-1)
    d0 = -r * torch.cos(theta)
    d1 = r * torch.sin(theta) * torch.cos(phi)
    d2 = r * torch.sin(theta) * torch.sin(phi)
    return pj + d0.unsqueeze(-1) * bc + d1.unsqueeze(-1) * m + d2.unsqueeze(-1) * n


def dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """IUPAC dihedral p0-p1-p2-p3 in (-pi, pi]."""
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    y = (b1.norm(dim=-1, keepdim=True) * b0 * n2).sum(-1)
    x = (n1 * n2).sum(-1)
    return torch.atan2(y, x)


def wrap(a: Tensor) -> Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
