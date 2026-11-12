"""Alternating descent/ascent on the Section 3.5 game."""

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn

from .minimax import CollapsedObservableError, GameTerms, MinimaxModel
from .model import LearnedMHRatio, classifier_loss


@contextmanager
def frozen(module: nn.Module):
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


@dataclass
class TrainingConfig:
    batch_size: int = 256
    reference_size: int = 512
    horizon: int = 1
    critic_steps: int = 3
    classifier_steps: int = 0
    sampler_lr: float = 1e-4
    critic_lr: float = 1e-3
    acceptance_lr: float = 1e-4
    lambda_balance: float = 10.0
    lambda_mix: float = 1.0
    grad_clip: float = 10.0

    def __post_init__(self):
        if self.batch_size < 2 or self.reference_size < 2:
            raise ValueError("Batch and reference sizes must be at least two")
        if (
            not 1 <= self.horizon <= 8
            or self.critic_steps < 1
            or self.classifier_steps < 0
        ):
            raise ValueError(
                "Require horizon in [1,8], critic_steps >= 1, classifier_steps >= 0"
            )
        positives = [
            self.sampler_lr,
            self.critic_lr,
            self.acceptance_lr,
            self.grad_clip,
        ]
        weights = [self.lambda_balance, self.lambda_mix]
        if any(not math.isfinite(x) or x <= 0 for x in positives):
            raise ValueError(
                "Learning rates and gradient clip must be finite and positive"
            )
        if any(not math.isfinite(x) or x < 0 for x in weights):
            raise ValueError("Objective weights must be finite and nonnegative")


class MinimaxTrainer:
    def __init__(
        self,
        model: MinimaxModel,
        target_samples: Callable[[int], Tensor],
        initial_samples: Callable[[int], Tensor],
        config: TrainingConfig | None = None,
    ):
        self.model = model
        self.target_samples = target_samples
        self.initial_samples = initial_samples
        self.config = config or TrainingConfig()
        self.iteration = 0
        self.observable_restarts = 0
        self.sampler_parameters = [
            p for p in model.sampler.proposal.parameters() if p.requires_grad
        ]
        self.acceptance_parameters = [
            p for p in model.sampler.ratio.parameters() if p.requires_grad
        ]
        groups = []
        if self.sampler_parameters:
            groups.append(
                {"params": self.sampler_parameters, "lr": self.config.sampler_lr}
            )
        if self.acceptance_parameters:
            groups.append(
                {"params": self.acceptance_parameters, "lr": self.config.acceptance_lr}
            )
        if not groups:
            raise ValueError(
                "The sampler must have trainable proposal or acceptance parameters"
            )
        if self.config.classifier_steps and not isinstance(
            model.sampler.ratio, LearnedMHRatio
        ):
            raise ValueError("Auxiliary classifier steps require LearnedMHRatio")
        self.sampler_optimizer = torch.optim.Adam(groups)
        self.critic_optimizer = torch.optim.Adam(
            model.critics.parameters(), lr=self.config.critic_lr
        )

    def _game(self) -> GameTerms:
        config = self.config
        with torch.no_grad():
            target = self.target_samples(config.batch_size).detach()
            initial = self.initial_samples(config.batch_size).detach()
            reference = self.target_samples(config.reference_size).detach()
        for _ in range(3):
            try:
                return self.model.game(
                    target,
                    initial,
                    reference,
                    config.horizon,
                    config.lambda_balance,
                    config.lambda_mix,
                )
            except CollapsedObservableError:
                self.model.critics.mixing.reset()
                for parameter in self.model.critics.mixing.parameters():
                    self.critic_optimizer.state.pop(parameter, None)
                self.observable_restarts += 1
        raise CollapsedObservableError(
            "Observable remains collapsed after three restarts"
        )

    def _update(self, loss: Tensor, optimizer, parameters) -> float:
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite minimax loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(
            parameters, self.config.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        return float(norm.detach())

    def critic_step(self) -> dict[str, float]:
        # Maximizing the total means f/h ascend and g minimizes Dirichlet energy.
        self.sampler_optimizer.zero_grad(set_to_none=True)
        with frozen(self.model.sampler):
            self.critic_optimizer.zero_grad(set_to_none=True)
            terms = self._game()
            norm = self._update(
                -terms.total,
                self.critic_optimizer,
                list(self.model.critics.parameters()),
            )
        self.critic_optimizer.zero_grad(set_to_none=True)
        return {**terms.metrics(), "grad_norm": norm}

    def sampler_step(self) -> dict[str, float]:
        self.critic_optimizer.zero_grad(set_to_none=True)
        target = self.model.sampler.target_log_prob
        # A neural target density is fixed, but its input derivatives are retained.
        target_context = (
            frozen(target) if isinstance(target, nn.Module) else nullcontext()
        )
        with frozen(self.model.critics), target_context:
            self.sampler_optimizer.zero_grad(set_to_none=True)
            terms = self._game()
            norm = self._update(
                terms.total,
                self.sampler_optimizer,
                self.sampler_parameters + self.acceptance_parameters,
            )
        self.sampler_optimizer.zero_grad(set_to_none=True)
        return {**terms.metrics(), "grad_norm": norm}

    def classifier_step(self) -> float:
        """Optional BCE auxiliary step, distinct from the Section 3.5 game."""
        ratio = self.model.sampler.ratio
        if not isinstance(ratio, LearnedMHRatio):
            raise ValueError("Classifier training requires LearnedMHRatio")
        with torch.no_grad():
            x = self.target_samples(self.config.batch_size)
            y = self.model.sampler.proposal.sample(x)
        self.sampler_optimizer.zero_grad(set_to_none=True)
        loss = classifier_loss(ratio, x, y)
        self._update(loss, self.sampler_optimizer, self.acceptance_parameters)
        self.sampler_optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    def step(self) -> dict[str, float]:
        target = self.model.sampler.target_log_prob
        target_modes = (
            [(module, module.training) for module in target.modules()]
            if isinstance(target, nn.Module)
            else []
        )
        self.model.train()
        # A fixed neural density must retain its evaluation/dropout/BN modes.
        for module, training in target_modes:
            module.training = training
        metrics = {}
        for _ in range(self.config.classifier_steps):
            metrics["classifier_bce"] = self.classifier_step()
        for _ in range(self.config.critic_steps):
            metrics.update({"critic/" + k: v for k, v in self.critic_step().items()})
        metrics.update({"sampler/" + k: v for k, v in self.sampler_step().items()})
        self.iteration += 1
        metrics["iteration"] = self.iteration
        metrics["observable_restarts"] = self.observable_restarts
        return metrics

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "sampler_optimizer": self.sampler_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": asdict(self.config),
            "iteration": self.iteration,
            "observable_restarts": self.observable_restarts,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["config"] != asdict(self.config):
            raise ValueError("Checkpoint training config does not match trainer")
        self.model.load_state_dict(state["model"])
        self.sampler_optimizer.load_state_dict(state["sampler_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.iteration = state["iteration"]
        self.observable_restarts = state["observable_restarts"]
