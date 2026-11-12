"""Targets and finite offline sample pools used by the training example."""

import math

import torch
from torch import Tensor


class GaussianMixture:
    """Two equal isotropic Gaussian modes, separated along the first coordinate."""

    def __init__(self, state_dim=2, separation=2.0, device="cpu"):
        if state_dim < 1 or separation <= 0:
            raise ValueError("Require positive dimension and separation")
        self.means = torch.zeros(2, state_dim, device=device)
        self.means[:, 0] = self.means.new_tensor([-separation, separation])

    def sample(self, count: int) -> Tensor:
        labels = torch.randint(0, 2, (count,), device=self.means.device)
        return self.means[labels] + torch.randn(
            count, self.means.shape[-1], device=self.means.device
        )

    def log_prob(self, x: Tensor) -> Tensor:
        components = -0.5 * (x.unsqueeze(-2) - self.means).square().sum(-1)
        normalizer = math.log(2) + 0.5 * self.means.shape[-1] * math.log(2 * math.pi)
        return torch.logsumexp(components, -1) - normalizer


class SamplePool:
    def __init__(self, samples: Tensor):
        if samples.ndim != 2 or len(samples) < 2 or samples.shape[1] < 1:
            raise ValueError(
                "Sample data must have shape (N,D), with N >= 2 and D >= 1"
            )
        if not samples.is_floating_point() or not torch.isfinite(samples).all():
            raise ValueError("Samples must be finite floating-point values")
        self.samples = samples.detach()

    def __call__(self, count: int) -> Tensor:
        indices = torch.randint(len(self.samples), (count,), device=self.samples.device)
        return self.samples[indices]
