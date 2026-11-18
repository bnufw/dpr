"""Stable feature subspace from the complete centered classifier row space."""
import torch


@torch.no_grad()
def centered_head_basis(weights):
    """Return a CPU float64 basis and singular values for C-by-D head weights.

    Center across all seen classes, then keep singular values above 1e-7
    times the largest, capped at min(C - 1, D).
    """
    centered = weights.detach().double().cpu()
    centered = centered - centered.mean(0)
    _, values, right = torch.linalg.svd(centered, full_matrices=False)
    maximum = min(weights.shape[0] - 1, weights.shape[1])
    rank = min(int((values > values[0] * 1e-7).sum()), maximum)
    return right[:rank].T.contiguous(), values[:rank]


def spectral_drift_losses(features, old_features, basis):
    """Squared stable and complement drift: sum over directions, mean over samples."""
    delta = features - old_features.detach()
    projected = delta @ basis
    stable = projected.square().sum(dim=1).mean()
    residual = delta - projected @ basis.T
    complement = residual.square().sum(dim=1).mean()
    return stable, complement
