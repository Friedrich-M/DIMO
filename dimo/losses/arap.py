"""As-rigid-as-possible (ARAP) regularisation of key-point trajectories.

Given the positions of the same set of nodes at several time steps, edges are built between nodes
that stay within a ball radius of each other at *every* time step, and the deformation energy
``sum_ij w_ij || (p_i^t - p_j^t) - R_i (p_i^0 - p_j^0) ||^2`` is accumulated over time steps, with
per-node rotations ``R_i`` estimated in closed form (Procrustes).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import ball_query

Edges = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (ii, jj, nn): edge i -> j is the n-th neighbour of i


def build_connectivity(points: torch.Tensor, K: int = 10, radius: float = 0.1) -> Edges:
    """Edges between nodes that are within ``radius`` of each other at all time steps.

    ``points`` is ``(T, N, 3)``. Returns ``(ii, jj, nn)`` index tensors of shape ``(E,)``.
    """
    T, N = points.shape[:2]
    device = points.device
    _, nn_idx, _ = ball_query(points, points, K=K + 1, radius=radius)  # (T, N, K+1), -1 = empty slot
    nn_idx = nn_idx[:, :, 1:]

    # (N, N+1) indicator: neighbour j (shifted by one so that -1 maps to column 0) present in all frames
    present = F.one_hot(nn_idx + 1, num_classes=N + 1).to(torch.bool).any(dim=2).all(dim=0).float()
    present[:, 0] = 0.0
    num_valid = present.sum(dim=1).long()  # (N,)
    _, top_idx = torch.topk(present, k=K, dim=1, largest=True)  # (N, K)
    top_idx = (top_idx - 1).abs()

    ii = torch.arange(N, device=device)[:, None].expand(N, K)
    nn = torch.arange(K, device=device)[None].expand(N, K)
    valid = torch.arange(K, device=device)[None].expand(N, K) < num_valid[:, None]
    return ii[valid], top_idx[valid], nn[valid]


def edge_matrix(verts: torch.Tensor, edges: Edges, K: int) -> torch.Tensor:
    """``E[i, n] = p_i - p_j`` for every edge, as ``(N, K, 3)`` (zeros for missing neighbours)."""
    ii, jj, nn = edges
    E = torch.zeros((verts.shape[0], K, 3), device=verts.device, dtype=verts.dtype)
    E[ii, nn] = verts[ii] - verts[jj]
    return E


def estimate_rotations(source_edges: torch.Tensor, target_edges: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-node rotations aligning ``source_edges`` to ``target_edges`` (both ``(N, K, 3)``) via SVD."""
    D = torch.diag_embed(weight)  # (N, K, K)
    S = torch.bmm(source_edges.permute(0, 2, 1), torch.bmm(D, target_edges))  # (N, 3, 3)
    # Undeformed nodes get S = 0 so that R = I (avoids numerical noise).
    unchanged = torch.unique(torch.where((source_edges == target_edges).all(dim=1))[0])
    S[unchanged] = 0

    U, sig, W = torch.svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))
    # Fix reflections: flip the column of U belonging to the smallest singular value.
    flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()
    if len(flip) > 0:
        U_mod = U.clone()
        cols = torch.argmin(sig[flip], dim=1)
        U_mod[flip, :, cols] *= -1
        R[flip] = torch.bmm(W[flip], U_mod[flip].permute(0, 2, 1))
    return R


def arap_error(nodes: torch.Tensor, edges: Edges, K: int = 10, sample_num: int = 512) -> torch.Tensor:
    """ARAP energy of node trajectories ``nodes (T, N, 3)`` relative to the first time step."""
    T, N, _ = nodes.shape
    ii, _, nn = edges
    weight = torch.zeros(N, K, device=nodes.device)
    weight[ii, nn] = 1

    source = edge_matrix(nodes[0], edges, K)
    if N > sample_num:
        sample_idx = torch.from_numpy(np.random.choice(N, sample_num)).long().to(nodes.device)
    else:
        sample_idx = torch.arange(N, device=nodes.device)
    source = source[sample_idx]
    weight = weight[sample_idx]

    error = 0.0
    for t in range(1, T):
        target = edge_matrix(nodes[t], edges, K)[sample_idx]
        with torch.no_grad():
            R = estimate_rotations(source, target, weight)
        rigid = torch.bmm(R, source.permute(0, 2, 1)).permute(0, 2, 1)
        stretch = (target - rigid).norm(dim=2) ** 2
        error = error + (weight * stretch).sum()
    return error


def arap_loss(nodes: torch.Tensor, K: int = 10, radius: float = 0.1,
              edges: Optional[Edges] = None) -> Tuple[torch.Tensor, Edges]:
    """ARAP loss of node trajectories ``(T, N, 3)``; returns the loss and the edges used."""
    if edges is None:
        edges = build_connectivity(nodes, K=K, radius=radius)
    return arap_error(nodes, edges, K=K), edges
