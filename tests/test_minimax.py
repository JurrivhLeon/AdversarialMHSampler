import math

import pytest
import torch
from torch import nn

from adversarial_mh import (
    Critics,
    DirectAcceptance,
    MHSampler,
    MinimaxModel,
    MinimaxTrainer,
    Proposal,
    TrainingConfig,
    endpoint_expectation,
)
from adversarial_mh.minimax import CollapsedObservableError
from adversarial_mh.targets import GaussianMixture
from adversarial_mh.train import build_model, main
from adversarial_mh.training import frozen


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(71)


class ShiftProposal(Proposal):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(1.0))

    def sample(self, x):
        return x + self.shift


def direct_model():
    proposal = ShiftProposal()
    acceptance = DirectAcceptance(1, hidden_dims=())
    with torch.no_grad():
        acceptance.network[-1].weight.zero_()
        acceptance.network[-1].bias.zero_()
    sampler = MHSampler(proposal, acceptance)
    critics = Critics(1, hidden_dims=())
    with torch.no_grad():
        critics.endpoint.network[-1].weight.fill_(1)
        critics.endpoint.network[-1].bias.zero_()
        critics.balance.network[-1].weight.copy_(torch.tensor([[1.0, -1.0]]))
        critics.balance.network[-1].bias.zero_()
        critics.mixing.network[-1].weight.fill_(1)
        critics.mixing.network[-1].bias.zero_()
    return MinimaxModel(sampler, critics)


def test_game_matches_section_35_by_hand():
    model = direct_model()
    target = torch.tensor([[0.0], [1.0]])
    initial = torch.tensor([[-2.0], [2.0]])
    reference = torch.tensor([[-1.0], [0.0], [1.0]])
    terms = model.game(target, initial, reference, lambda_balance=2, lambda_mix=3)
    expected_endpoint = 0.5 * math.tanh(1) - 0.25 * (
        math.tanh(-2) + math.tanh(-1) + math.tanh(2) + math.tanh(3)
    )
    assert terms.endpoint.item() == pytest.approx(expected_endpoint, abs=1e-6)
    assert terms.balance.item() == pytest.approx(-math.tanh(1), abs=1e-6)
    assert terms.dirichlet.item() == pytest.approx(0.375, abs=1e-6)
    assert terms.total.item() == pytest.approx(
        expected_endpoint - 2 * math.tanh(1) - 3 * 0.375, abs=1e-6
    )
    assert terms.acceptance.item() == pytest.approx(0.5)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_endpoint_tree_matches_binomial_chain_and_its_gradients(horizon):
    model = direct_model()
    initial = torch.tensor([[-1.0], [2.0]])
    values = endpoint_expectation(
        model.sampler, lambda x: x.square().sum(-1), initial, horizon
    )
    expected = (initial[:, 0] + 0.5 * horizon).square() + 0.25 * horizon
    torch.testing.assert_close(values, expected)
    mean = endpoint_expectation(
        model.sampler, lambda x: x.sum(-1), initial, horizon
    ).mean()
    mean.backward()
    assert model.sampler.proposal.shift.grad.item() == pytest.approx(0.5 * horizon)
    assert model.sampler.ratio.network[-1].bias.grad.item() == pytest.approx(
        0.25 * horizon
    )


def test_normalization_is_constrained_and_rejects_constant_observables():
    critic = Critics(2).mixing
    reference = torch.randn(128, 2)
    mean, scale = critic.normalization(reference)
    normalized = critic.normalized(reference, mean, scale)
    assert normalized.mean().item() == pytest.approx(0, abs=1e-6)
    assert normalized.var(unbiased=False).item() == pytest.approx(1, abs=1e-6)
    normalized.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters()
    )
    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()
    with pytest.raises(CollapsedObservableError):
        critic.normalization(reference)


def test_alternating_steps_follow_opposite_gradients_and_freeze_other_player():
    model = direct_model()
    data = torch.tensor([[-2.0], [-0.5], [0.3], [1.1]])
    initial = torch.tensor([[-3.0], [-1.0], [0.5], [2.0]])
    trainer = MinimaxTrainer(
        model,
        lambda n: data,
        lambda n: initial,
        TrainingConfig(
            batch_size=4,
            reference_size=4,
            critic_steps=1,
            sampler_lr=1e-5,
            critic_lr=1e-5,
        ),
    )
    sampler_params = list(model.sampler.parameters())
    critic_params = list(model.critics.parameters())
    with frozen(model.sampler):
        total = model.game(data, initial, data).total
        gradients = torch.autograd.grad(total, critic_params)
    before_critic = [p.detach().clone() for p in critic_params]
    before_sampler = [p.detach().clone() for p in sampler_params]
    trainer.critic_step()
    ascent = sum(
        ((p.detach() - old) * g).sum()
        for p, old, g in zip(critic_params, before_critic, gradients)
    )
    assert ascent > 0
    assert all(torch.equal(p, old) for p, old in zip(sampler_params, before_sampler))
    assert all(p.grad is None for p in sampler_params)
    with frozen(model.critics):
        gradients = torch.autograd.grad(
            model.game(data, initial, data).total, sampler_params
        )
    before_critic = [p.detach().clone() for p in critic_params]
    trainer.sampler_step()
    descent = sum(
        ((p.detach() - old) * g).sum()
        for p, old, g in zip(sampler_params, before_sampler, gradients)
    )
    assert descent < 0
    assert all(torch.equal(p, old) for p, old in zip(critic_params, before_critic))
    assert all(p.grad is None for p in critic_params)


@pytest.mark.parametrize("acceptance", ["exact", "learned", "direct"])
def test_training_modes_update_parameters_and_restore_checkpoint(acceptance):
    target = GaussianMixture(2)
    model = build_model(
        2,
        acceptance,
        (16, 16),
        target_log_prob=target.log_prob if acceptance == "exact" else None,
    )
    config = TrainingConfig(
        batch_size=32,
        reference_size=64,
        critic_steps=1,
        classifier_steps=int(acceptance == "learned"),
    )
    trainer = MinimaxTrainer(
        model, target.sample, lambda n: 3 * torch.randn(n, 2), config
    )
    before = [p.detach().clone() for p in model.sampler.proposal.parameters()]
    acceptance_before = [p.detach().clone() for p in model.sampler.ratio.parameters()]
    metrics = trainer.step()
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(p, old)
        for p, old in zip(model.sampler.proposal.parameters(), before)
    )
    if acceptance_before:
        assert any(
            not torch.equal(p, old)
            for p, old in zip(model.sampler.ratio.parameters(), acceptance_before)
        )
    duplicate = build_model(
        2,
        acceptance,
        (16, 16),
        target_log_prob=target.log_prob if acceptance == "exact" else None,
    )
    restored = MinimaxTrainer(
        duplicate, target.sample, lambda n: torch.randn(n, 2), config
    )
    restored.load_state_dict(trainer.state_dict())
    assert restored.iteration == 1
    for p, q in zip(model.parameters(), duplicate.parameters()):
        torch.testing.assert_close(p, q)
    assert math.isfinite(restored.step()["sampler/total"])


def test_collapsed_observable_is_restarted_and_direct_sampling_works():
    model = build_model(1, "direct", (16, 16))
    trainer = MinimaxTrainer(
        model,
        lambda n: torch.randn(n, 1),
        lambda n: torch.randn(n, 1),
        TrainingConfig(batch_size=32, reference_size=64, critic_steps=1),
    )
    with torch.no_grad():
        for p in model.critics.mixing.parameters():
            p.zero_()
    trainer.critic_step()
    assert trainer.observable_restarts == 1
    transition = model.sampler.step(torch.zeros(32, 1))
    assert transition.log_ratio is None
    assert torch.all(
        (transition.acceptance_probability > 0)
        & (transition.acceptance_probability < 1)
    )
    with pytest.raises(TypeError, match="no MH ratio"):
        model.sampler.log_ratio(torch.zeros(1, 1), torch.ones(1, 1))


def test_offline_training_cli_writes_loadable_artifacts(tmp_path):
    data = tmp_path / "data.pt"
    torch.save(torch.randn(100, 2), data)
    output = tmp_path / "run"
    main(
        [
            "--acceptance",
            "direct",
            "--data",
            str(data),
            "--iterations",
            "2",
            "--critic-steps",
            "1",
            "--batch-size",
            "16",
            "--reference-size",
            "32",
            "--horizon",
            "2",
            "--hidden-width",
            "8",
            "--draws",
            "5",
            "--burn-in",
            "2",
            "--chains",
            "3",
            "--output",
            str(output),
        ]
    )
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert checkpoint["iteration"] == 2
    samples = torch.load(output / "samples.pt", weights_only=True)
    assert samples["samples"].shape == (5, 3, 2)
    assert (output / "evaluation.json").is_file()
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2


def test_fixed_neural_target_keeps_parameters_and_evaluation_mode():
    class NeuralTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.2))
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            assert not self.training and not self.dropout.training
            return -0.5 * self.dropout(x).square().sum(-1) + self.offset

    target = NeuralTarget().eval()
    model = build_model(1, "exact", (16, 16), target_log_prob=target)
    trainer = MinimaxTrainer(
        model,
        lambda n: torch.randn(n, 1),
        lambda n: 3 * torch.randn(n, 1),
        TrainingConfig(batch_size=32, reference_size=64, critic_steps=1),
    )
    trainer.step()
    assert not target.training
    assert target.offset.requires_grad
    assert target.offset.grad is None
    torch.testing.assert_close(target.offset, torch.tensor(0.2))
