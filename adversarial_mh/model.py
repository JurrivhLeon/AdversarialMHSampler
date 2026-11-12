"""Shared proposals and chains with analytic, classifier, or direct acceptance.

States have shape (..., state_dim); log densities/ratios have shape (...,).
Sampling runs without autograd. Training uses proposal.sample(), classifier_loss(),
or one_step_expectation(), rather than gradients through sampled decisions.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LogDensity = Callable[[Tensor], Tensor]


def _mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        if width <= 0:
            raise ValueError("Hidden dimensions must be positive")
        layers.extend([nn.Linear(previous, width), nn.SiLU()])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class Proposal(nn.Module, ABC):
    @abstractmethod
    def sample(self, x: Tensor) -> Tensor:
        """Generate one candidate per state, preserving the shape of x."""

    def log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        """Return log q(y | x), if evaluable."""
        raise NotImplementedError(
            "Exact MH requires proposal.log_prob(y, x). "
            "Use an evaluable proposal or LearnedMHRatio for an implicit proposal."
        )


class ConditionalGaussianProposal(Proposal):
    """q(y|x) = N(x + drift(x), diag(scale(x)**2)).

    A zero-initialized output layer starts as a symmetric Gaussian random walk.
    Both drift and scale are trainable; reverse density is evaluated at y.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        initial_scale: float = 1.0,
        min_scale: float = 1e-4,
    ):
        super().__init__()
        if state_dim <= 0 or not 0 < min_scale < initial_scale:
            raise ValueError("Require state_dim > 0 and 0 < min_scale < initial_scale")
        self.state_dim = state_dim
        self.min_scale = min_scale
        self.network = _mlp(state_dim, 2 * state_dim, hidden_dims)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        scale = initial_scale - min_scale
        # Stable inverse softplus, including for large initial scales.
        self.register_buffer(
            "scale_offset", torch.tensor(scale + math.log(-math.expm1(-scale)))
        )

    def distribution(self, x: Tensor) -> torch.distributions.Normal:
        if x.shape[-1] != self.state_dim:
            raise ValueError("State dimension does not match proposal")
        drift, raw_scale = self.network(x).chunk(2, dim=-1)
        scale = F.softplus(raw_scale + self.scale_offset) + self.min_scale
        return torch.distributions.Normal(x + drift, scale)

    def sample(self, x: Tensor) -> Tensor:
        return self.distribution(x).rsample()

    def log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        return self.distribution(x).log_prob(y).sum(dim=-1)


class ImplicitProposal(Proposal):
    """Wrap a reparameterizable generator G(x,z); no density is assumed."""

    def __init__(self, generator: nn.Module, noise_dim: int):
        super().__init__()
        if noise_dim <= 0:
            raise ValueError("noise_dim must be positive")
        self.generator = generator
        self.noise_dim = noise_dim

    def sample(self, x: Tensor) -> Tensor:
        z = torch.randn(*x.shape[:-1], self.noise_dim, device=x.device, dtype=x.dtype)
        y = self.generator(x, z)
        if y.shape != x.shape:
            raise ValueError("Implicit generator must preserve the state shape")
        return y


class ExactMHRatio(nn.Module):
    """Analytic reverse/forward flow log ratio; target normalizer cancels."""

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        proposal: Proposal,
        target_log_prob: LogDensity | None = None,
    ) -> Tensor:
        if target_log_prob is None:
            raise ValueError("Exact MH requires target_log_prob")
        target_x, target_y = target_log_prob(x), target_log_prob(y)
        forward_q, reverse_q = proposal.log_prob(y, x), proposal.log_prob(x, y)
        expected_shape = x.shape[:-1]
        if any(
            value.shape != expected_shape
            for value in (target_x, target_y, forward_q, reverse_q)
        ):
            raise ValueError("Log densities must return one scalar per state")
        return target_y - target_x + reverse_q - forward_q


class LearnedMHRatio(nn.Module):
    """Antisymmetric pair classifier estimating log[B(x,y)/F(x,y)].

    D(x,y) = sigmoid(s(x,y)) classifies a forward pair as label one.
    s(x,y) = u(x,y) - u(y,x), so log MH ratio = -s(x,y).
    Reciprocity is architectural; correctness of the ratio must be learned.
    """

    def __init__(self, state_dim: int, hidden_dims: Sequence[int] = (64, 64)):
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.network = _mlp(2 * state_dim, 1, hidden_dims)

    def forward_logit(self, x: Tensor, y: Tensor) -> Tensor:
        xy = self.network(torch.cat([x, y], dim=-1))
        yx = self.network(torch.cat([y, x], dim=-1))
        return (xy - yx).squeeze(-1)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        proposal: Proposal | None = None,
        target_log_prob: LogDensity | None = None,
    ) -> Tensor:
        # Density inputs are accepted for a common interface, but never used.
        return -self.forward_logit(x, y)


class DirectAcceptance(nn.Module):
    """A freely learned sigmoid acceptance, optimized by the minimax game.

    Unlike LearnedMHRatio this need not estimate any density ratio. It returns
    log acceptance, and does not impose detailed balance architecturally.
    """

    def __init__(self, state_dim: int, hidden_dims: Sequence[int] = (64, 64)):
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.network = _mlp(2 * state_dim, 1, hidden_dims)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.logsigmoid(self.network(torch.cat([x, y], dim=-1)).squeeze(-1))


def classifier_loss(ratio: LearnedMHRatio, x: Tensor, y: Tensor) -> Tensor:
    """Balanced forward/reversed BCE; states are detached from proposal training.

    x must be target data and y a proposal from x. Since the swapped logit is
    exactly -s, average BCE of (x,y,label=1) and (y,x,label=0) is softplus(-s).
    Use fresh proposals after proposal updates, and hold out target states.
    """
    return F.softplus(-ratio.forward_logit(x.detach(), y.detach())).mean()


def one_step_expectation(
    x: Tensor,
    y: Tensor,
    log_ratio: Tensor,
    observable: Callable[[Tensor], Tensor],
) -> Tensor:
    """Per-state E[f(X_next)|x,y], integrating out the Bernoulli decision.

    Supports gradients through reparameterized proposals and acceptance with a
    fixed ratio network, not through refitting that network. Returns one scalar
    per state. This identity is for the one-step kernel, not a soft state update.
    """
    fx, fy = observable(x), observable(y)
    if fx.shape != x.shape[:-1] or fy.shape != fx.shape or log_ratio.shape != fx.shape:
        raise ValueError("Observable and log ratio must return one scalar per state")
    acceptance = log_ratio.clamp(max=0).exp()
    return (1 - acceptance) * fx + acceptance * fy


@dataclass
class Transition:
    state: Tensor
    proposal: Tensor
    log_ratio: Tensor | None  # None for direct acceptance
    acceptance_probability: Tensor
    accepted: Tensor


@dataclass
class ChainResult:
    samples: Tensor  # (draws, chains, state_dim), including rejection repeats
    acceptance_rate: Tensor  # (chains,), over all post-burn-in transitions


class MHSampler(nn.Module):
    def __init__(
        self,
        proposal: Proposal,
        ratio: ExactMHRatio | LearnedMHRatio | DirectAcceptance,
        target_log_prob: LogDensity | None = None,
    ):
        super().__init__()
        self.proposal = proposal
        self.ratio = ratio
        self.target_log_prob = target_log_prob

    def log_ratio(self, x: Tensor, y: Tensor) -> Tensor:
        if isinstance(self.ratio, DirectAcceptance):
            raise TypeError("DirectAcceptance defines no MH ratio; use log_acceptance")
        result = self.ratio(x, y, self.proposal, self.target_log_prob)
        if result.shape != x.shape[:-1]:
            raise ValueError("Log densities/ratios must return one scalar per state")
        return result

    def log_acceptance(self, x: Tensor, y: Tensor) -> Tensor:
        """Differentiable log acceptance for exact, classifier, and direct modes."""
        if isinstance(self.ratio, DirectAcceptance):
            result = self.ratio(x, y)
            if result.shape != x.shape[:-1]:
                raise ValueError("Acceptance must return one scalar per state")
            return result
        return self.log_ratio(x, y).clamp(max=0)

    @torch.no_grad()
    def step(self, x: Tensor) -> Transition:
        if x.ndim < 1 or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("States must be finite floating-point tensors")
        y = self.proposal.sample(x)
        if y.shape != x.shape or not torch.isfinite(y).all():
            raise ValueError("Proposals must be finite and preserve the state shape")
        if isinstance(self.ratio, DirectAcceptance):
            log_ratio = None
            log_acceptance = self.log_acceptance(x, y)
        else:
            log_ratio = self.log_ratio(x, y)
            log_acceptance = log_ratio.clamp(max=0)
        if torch.isnan(log_acceptance).any():
            raise ValueError(
                "Undefined MH ratio; check target support and proposal log densities"
            )
        # Infinite ratios are valid: +inf accepts and -inf rejects.
        accepted = torch.rand_like(log_acceptance).log() < log_acceptance
        state = torch.where(accepted.unsqueeze(-1), y, x)
        return Transition(state, y, log_ratio, log_acceptance.exp(), accepted)

    @torch.no_grad()
    def sample(
        self, initial: Tensor, num_samples: int, burn_in: int = 0, thin: int = 1
    ) -> ChainResult:
        if num_samples <= 0 or burn_in < 0 or thin <= 0:
            raise ValueError("Require num_samples > 0, burn_in >= 0, and thin > 0")
        if initial.ndim not in (1, 2) or initial.numel() == 0:
            raise ValueError(
                "initial must have shape (state_dim,) or (chains,state_dim)"
            )
        state = initial.detach().clone()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        for _ in range(burn_in):
            state = self.step(state).state
        accepted = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        samples = []
        for _ in range(num_samples):
            for _ in range(thin):
                transition = self.step(state)
                state = transition.state
                accepted += transition.accepted.to(state.dtype)
            samples.append(state.clone())
        return ChainResult(torch.stack(samples), accepted / (num_samples * thin))
