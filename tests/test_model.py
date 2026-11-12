import pytest
import torch
from torch import nn
from torch.nn import functional as F

from adversarial_mh import (
    ConditionalGaussianProposal,
    ExactMHRatio,
    ImplicitProposal,
    LearnedMHRatio,
    MHSampler,
    classifier_loss,
    one_step_expectation,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(23)
    torch.set_num_threads(1)


def normal_log_prob(x):
    return -0.5 * x.square().sum(-1)


def test_exact_ratio_includes_asymmetric_reverse_density_and_balances_flow():
    proposal = ConditionalGaussianProposal(2, hidden_dims=()).double()
    with torch.no_grad():
        proposal.network[-1].weight.copy_(
            torch.tensor(
                [[0.3, -0.2], [0.1, 0.4], [0.2, 0.1], [-0.1, 0.3]],
                dtype=torch.double,
            )
        )
        proposal.network[-1].bias.copy_(torch.tensor([0.4, -0.2, 0.1, -0.1]))
    x, y = torch.randn(40, 2, dtype=torch.double), torch.randn(
        40, 2, dtype=torch.double
    )
    ratio = ExactMHRatio()
    forward = ratio(x, y, proposal, normal_log_prob)
    reverse = ratio(y, x, proposal, normal_log_prob)
    torch.testing.assert_close(forward, -reverse)
    target_only = normal_log_prob(y) - normal_log_prob(x)
    assert not torch.allclose(forward, target_only)
    forward_flow = normal_log_prob(x) + proposal.log_prob(y, x) + forward.clamp(max=0)
    reverse_flow = normal_log_prob(y) + proposal.log_prob(x, y) + reverse.clamp(max=0)
    torch.testing.assert_close(forward_flow, reverse_flow)
    shifted = ratio(x, y, proposal, lambda z: normal_log_prob(z) + 123.0)
    torch.testing.assert_close(forward, shifted)


def test_learned_ratio_reciprocity_and_classifier_label_convention():
    ratio = LearnedMHRatio(2)
    x, y = torch.randn(10, 2), torch.randn(10, 2)
    logit = ratio.forward_logit(x, y)
    torch.testing.assert_close(ratio(x, y), -logit)
    torch.testing.assert_close(ratio(x, y), -ratio(y, x))
    torch.testing.assert_close(ratio(x, x), torch.zeros(10))
    full_bce = F.binary_cross_entropy_with_logits(
        torch.cat([logit, ratio.forward_logit(y, x)]),
        torch.cat([torch.ones(10), torch.zeros(10)]),
    )
    torch.testing.assert_close(classifier_loss(ratio, x, y), full_bce)


def test_classifier_learns_acceptance_without_target_density_or_proposal_gradients():
    proposal = ConditionalGaussianProposal(1, initial_scale=1.5)
    ratio = LearnedMHRatio(1, hidden_dims=(32, 32))
    optimizer = torch.optim.Adam(ratio.parameters(), lr=0.003)
    x_test = torch.randn(4096, 1)
    y_test = proposal.sample(x_test).detach()
    exact = ExactMHRatio()(x_test, y_test, proposal, normal_log_prob).detach()
    with torch.no_grad():
        before = (
            (ratio(x_test, y_test).clamp(max=0).exp() - exact.clamp(max=0).exp())
            .abs()
            .mean()
        )
    for _ in range(300):
        x = torch.randn(256, 1)
        y = proposal.sample(x)
        optimizer.zero_grad()
        loss = classifier_loss(ratio, x, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = (
            (ratio(x_test, y_test).clamp(max=0).exp() - exact.clamp(max=0).exp())
            .abs()
            .mean()
        )
    assert after < before / 2
    assert after < 0.08
    assert all(parameter.grad is None for parameter in proposal.parameters())


def test_exact_chain_recovers_normal_moments_and_keeps_initial_state_unchanged():
    proposal = ConditionalGaussianProposal(1, initial_scale=1.0)
    sampler = MHSampler(proposal, ExactMHRatio(), normal_log_prob)
    initial = torch.full((32, 1), 5.0)
    result = sampler.sample(initial, num_samples=1200, burn_in=150, thin=2)
    assert result.samples.shape == (1200, 32, 1)
    assert abs(result.samples.mean().item()) < 0.07
    assert abs(result.samples.square().mean().item() - 1) < 0.09
    assert torch.all((result.acceptance_rate > 0.5) & (result.acceptance_rate < 0.9))
    assert not result.samples.requires_grad
    torch.testing.assert_close(initial, torch.full((32, 1), 5.0))
    assert sampler.sample(torch.zeros(1), 2).samples.shape == (2, 1, 1)


class ShiftGenerator(nn.Module):
    def forward(self, x, z):
        return x + 1.0


def test_rejections_are_retained_and_learned_mode_never_evaluates_densities():
    proposal = ImplicitProposal(ShiftGenerator(), noise_dim=1)
    ratio = LearnedMHRatio(1, hidden_dims=())
    with torch.no_grad():
        ratio.network[-1].weight.copy_(torch.tensor([[-500.0, 500.0]]))
        ratio.network[-1].bias.zero_()

    def forbidden_density(x):
        raise AssertionError("Learned mode must not evaluate the target")

    sampler = MHSampler(proposal, ratio, forbidden_density)
    result = sampler.sample(torch.zeros(4, 1), 8, thin=2)
    torch.testing.assert_close(result.samples, torch.zeros(8, 4, 1))
    torch.testing.assert_close(result.acceptance_rate, torch.zeros(4))
    transition = sampler.step(torch.zeros(4, 1))
    torch.testing.assert_close(transition.proposal, torch.ones(4, 1))
    torch.testing.assert_close(transition.acceptance_probability, torch.zeros(4))


def test_exact_mode_reports_missing_density_capabilities():
    proposal = ConditionalGaussianProposal(1)
    with pytest.raises(ValueError, match="target_log_prob"):
        MHSampler(proposal, ExactMHRatio()).step(torch.zeros(4, 1))
    implicit = ImplicitProposal(ShiftGenerator(), noise_dim=1)
    with pytest.raises(NotImplementedError, match="proposal.log_prob"):
        MHSampler(implicit, ExactMHRatio(), normal_log_prob).step(torch.zeros(4, 1))


def test_one_step_expectation_supports_proposal_and_acceptance_gradients():
    x = torch.tensor([[0.2], [-0.4]], dtype=torch.double)
    y = torch.tensor([[0.5], [0.7]], dtype=torch.double, requires_grad=True)
    log_ratio = torch.tensor([-0.7, -0.4], dtype=torch.double, requires_grad=True)
    observable = lambda z: z.square().sum(-1)
    assert torch.autograd.gradcheck(
        lambda candidate, log_r: one_step_expectation(x, candidate, log_r, observable),
        (y, log_ratio),
    )
    # Independently integrate Bernoulli outcomes by Monte Carlo.
    probabilities = log_ratio.detach().exp()
    accepted = torch.rand(100000, 2) < probabilities
    monte_carlo = torch.where(accepted, observable(y.detach()), observable(x)).mean(0)
    expected = one_step_expectation(x, y, log_ratio, observable)
    torch.testing.assert_close(expected.detach(), monte_carlo, atol=0.004, rtol=0)


def test_shape_errors_nan_ratios_and_invalid_sampling_arguments():
    proposal = ConditionalGaussianProposal(1)
    sampler = MHSampler(
        proposal, ExactMHRatio(), lambda x: normal_log_prob(x).unsqueeze(-1)
    )
    with pytest.raises(ValueError, match="one scalar"):
        sampler.step(torch.zeros(4, 1))
    sampler.target_log_prob = lambda x: normal_log_prob(x).sum()
    with pytest.raises(ValueError, match="one scalar"):
        sampler.step(torch.zeros(4, 1))
    sampler.target_log_prob = lambda x: torch.full_like(x[..., 0], -torch.inf)
    with pytest.raises(ValueError, match="Undefined MH ratio"):
        sampler.step(torch.zeros(4, 1))
    for samples, burn, thin in [(0, 0, 1), (1, -1, 1), (1, 0, 0)]:
        with pytest.raises(ValueError):
            sampler.sample(torch.zeros(1), samples, burn, thin)


def test_infinite_ratios_handle_target_support_boundaries():
    proposal = ConditionalGaussianProposal(1)
    target = lambda x: torch.where(x[..., 0] > 0, 0.0, -torch.inf)
    ratio = ExactMHRatio()
    x, y = torch.tensor([[-1.0], [1.0]]), torch.tensor([[1.0], [-1.0]])
    log_ratio = ratio(x, y, proposal, target)
    assert log_ratio[0] == torch.inf
    assert log_ratio[1] == -torch.inf
    torch.testing.assert_close(log_ratio.clamp(max=0).exp(), torch.tensor([1.0, 0.0]))
