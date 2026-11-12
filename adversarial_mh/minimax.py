"""Neural critics and the finite-horizon game in Section 3.5.

Endpoint acceptance is marginalized over a binary tree, not relaxed. The tree
has 2**T leaves, so only short horizons are intended. All critic classes here
are restricted neural classes, not certificates of balance or spectral gap.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import MHSampler, _mlp


class CollapsedObservableError(RuntimeError):
    """The observable cannot be normalized to empirical variance one."""


class BoundedCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = _mlp(input_dim, 1, hidden_dims)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1).tanh()


class MixingCritic(nn.Module):
    def __init__(
        self, state_dim: int, hidden_dims: Sequence[int], variance_floor: float = 1e-8
    ):
        super().__init__()
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        self.network = _mlp(state_dim, 1, hidden_dims)
        self.variance_floor = variance_floor

    def raw(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)

    def normalization(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        if reference.ndim != 2 or len(reference) < 2:
            raise ValueError("Normalization reference requires at least two states")
        values = self.raw(reference)
        mean, variance = values.mean(), values.var(unbiased=False)
        if not torch.isfinite(variance) or variance.detach() < self.variance_floor:
            raise CollapsedObservableError("Mixing critic variance below floor")
        # No epsilon: whenever defined, empirical variance is exactly one.
        # Mean and variance remain differentiable during critic optimization.
        return mean, variance.sqrt()

    def normalized(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        return (self.raw(x) - mean) / scale

    def reset(self) -> None:
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


class Critics(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Sequence[int] = (64, 64)):
        super().__init__()
        self.endpoint = BoundedCritic(state_dim, hidden_dims)
        self.balance = BoundedCritic(2 * state_dim, hidden_dims)
        self.mixing = MixingCritic(state_dim, hidden_dims)

    def antisymmetric_balance(self, x: Tensor, y: Tensor) -> Tensor:
        return self.balance(torch.cat([x, y], -1)) - self.balance(torch.cat([y, x], -1))


def endpoint_expectation(
    sampler: MHSampler, critic: nn.Module, initial: Tensor, horizon: int = 1
) -> Tensor:
    """Unbiased reparameterized estimate of E[f(X_T)] per initial state.

    Each tree node samples one proposal and branches into rejection/acceptance,
    carrying the exact Bernoulli probabilities. Gradients include path weights
    and proposal states. This avoids score estimators and soft-state bias.
    """
    if not isinstance(horizon, int) or not 1 <= horizon <= 8:
        raise ValueError("horizon must be an integer in [1,8] (exponential tree)")
    if initial.ndim != 2 or len(initial) == 0:
        raise ValueError("initial must be a nonempty batch of states")
    states = initial.unsqueeze(1)
    weights = initial.new_ones((len(initial), 1))
    for _ in range(horizon):
        candidates = sampler.proposal.sample(states)
        acceptance = sampler.log_acceptance(states, candidates).exp()
        states = torch.cat([states, candidates], dim=1)
        weights = torch.cat([weights * (1 - acceptance), weights * acceptance], dim=1)
    return (weights * critic(states)).sum(dim=1)


@dataclass
class GameTerms:
    endpoint: Tensor
    balance: Tensor
    dirichlet: Tensor
    total: Tensor
    acceptance: Tensor
    squared_jump: Tensor
    observable_scale: Tensor

    def metrics(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in vars(self).items()}


class MinimaxModel(nn.Module):
    def __init__(self, sampler: MHSampler, critics: Critics):
        super().__init__()
        self.sampler = sampler
        self.critics = critics

    def game(
        self,
        target: Tensor,
        initial: Tensor,
        reference: Tensor,
        horizon: int = 1,
        lambda_balance: float = 10.0,
        lambda_mix: float = 1.0,
    ) -> GameTerms:
        """Return L_endpoint + lambda_balance*L_balance - lambda_mix*E_g.

        Sampler minimizes; all three critics maximize this same scalar. x~pi
        drives balance/mixing; independent x~nu_0 drives the endpoint term.
        g has empirical mean zero and variance one on the reference batch.
        """
        if lambda_balance < 0 or lambda_mix < 0:
            raise ValueError("Objective weights must be nonnegative")
        if (
            target.ndim != 2
            or initial.ndim != 2
            or target.shape[-1] != initial.shape[-1]
        ):
            raise ValueError("Target and initial must be compatible state batches")
        y = self.sampler.proposal.sample(target)
        acceptance = self.sampler.log_acceptance(target, y).exp()
        endpoint = (
            self.critics.endpoint(target).mean()
            - endpoint_expectation(
                self.sampler, self.critics.endpoint, initial, horizon
            ).mean()
        )
        balance = (acceptance * self.critics.antisymmetric_balance(target, y)).mean()
        mean, scale = self.critics.mixing.normalization(reference)
        gx = self.critics.mixing.normalized(target, mean, scale)
        gy = self.critics.mixing.normalized(y, mean, scale)
        energy = 0.5 * (acceptance * (gy - gx).square()).mean()
        total = endpoint + lambda_balance * balance - lambda_mix * energy
        return GameTerms(
            endpoint,
            balance,
            energy,
            total,
            acceptance.mean(),
            (acceptance * (y - target).square().sum(-1)).mean(),
            scale,
        )
