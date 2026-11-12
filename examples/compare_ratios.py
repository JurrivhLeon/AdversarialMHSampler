"""Train a pair classifier, then compare exact/learned MH on the same proposal.

Run from the repository root: python -m examples.compare_ratios
Target density is used only for benchmarking and exact acceptance.
"""

import argparse
import json
import time

import torch

from adversarial_mh import (
    ConditionalGaussianProposal,
    ExactMHRatio,
    LearnedMHRatio,
    MHSampler,
    classifier_loss,
)


def target_log_prob(x):
    means = x.new_tensor([[-2.0, 0.0], [2.0, 0.0]])
    component = -0.5 * (x.unsqueeze(-2) - means).square().sum(-1)
    return torch.logsumexp(component, dim=-1)  # common normalizer cancels


def target_samples(count, device):
    x = torch.randn(count, 2, device=device)
    x[:, 0] += 4 * torch.randint(0, 2, (count,), device=device) - 2
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.train_steps < 0 or args.batch_size <= 0 or args.chains <= 0:
        parser.error("Require nonnegative train steps and positive batch size/chains")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    proposal = ConditionalGaussianProposal(2, initial_scale=1.5).to(device)
    proposal.requires_grad_(False)  # isolate ratio estimation in this experiment
    learned = LearnedMHRatio(2).to(device)
    optimizer = torch.optim.Adam(learned.parameters(), lr=1e-3)
    started = time.perf_counter()
    for _ in range(args.train_steps):
        x = target_samples(args.batch_size, device)
        y = proposal.sample(x)
        loss = classifier_loss(learned, x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_seconds = time.perf_counter() - started
    learned.eval()
    x = target_samples(8192, device)
    y = proposal.sample(x)
    exact = ExactMHRatio()
    with torch.no_grad():
        true_ratio = exact(x, y, proposal, target_log_prob)
        estimated_ratio = learned(x, y)
        diagnostics = {
            "training_seconds": training_seconds,
            "held_out_log_ratio_mae": (true_ratio - estimated_ratio)
            .abs()
            .mean()
            .item(),
            "held_out_acceptance_mae": (
                true_ratio.clamp(max=0).exp() - estimated_ratio.clamp(max=0).exp()
            )
            .abs()
            .mean()
            .item(),
            "held_out_classifier_bce": classifier_loss(learned, x, y).item(),
        }
    initial = torch.zeros(args.chains, 2, device=device)
    for name, ratio in [("exact", exact), ("learned", learned)]:
        sampler = MHSampler(
            proposal, ratio, target_log_prob if name == "exact" else None
        )
        # Same initial states and noise stream give a controlled comparison.
        torch.manual_seed(args.seed + 1)
        started = time.perf_counter()
        result = sampler.sample(initial, args.draws, args.burn_in)
        elapsed = time.perf_counter() - started
        samples = result.samples.reshape(-1, 2)
        diagnostics[name] = {
            "acceptance_rate": result.acceptance_rate.mean().item(),
            "right_half_probability": (samples[:, 0] > 0).float().mean().item(),
            "mean": samples.mean(0).tolist(),
            "second_moment": samples.square().mean(0).tolist(),
            "sampling_seconds": elapsed,
        }
    diagnostics["target_reference"] = {
        "right_half_probability": 0.5,
        "mean": [0.0, 0.0],
        "second_moment": [5.0, 1.0],
    }
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
