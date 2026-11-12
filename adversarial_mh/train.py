"""Train the Section 3.5 minimax sampler on a mixture or an offline tensor.

Examples:
  python -m adversarial_mh.train --acceptance exact --iterations 1000
  python -m adversarial_mh.train --acceptance direct --iterations 1000
  python -m adversarial_mh.train --acceptance learned --classifier-warmup 200
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import torch

from .minimax import Critics, MinimaxModel
from .model import (
    ConditionalGaussianProposal,
    DirectAcceptance,
    ExactMHRatio,
    LearnedMHRatio,
    MHSampler,
)
from .targets import GaussianMixture, SamplePool
from .trainer import MinimaxTrainer, TrainingConfig


def build_model(
    state_dim,
    acceptance="exact",
    hidden_dims=(64, 64),
    initial_scale=1.0,
    target_log_prob=None,
):
    proposal = ConditionalGaussianProposal(state_dim, hidden_dims, initial_scale)
    if acceptance == "exact":
        if target_log_prob is None:
            raise ValueError("Exact acceptance requires target_log_prob")
        ratio = ExactMHRatio()
    elif acceptance == "learned":
        ratio = LearnedMHRatio(state_dim, hidden_dims)
    elif acceptance == "direct":
        ratio = DirectAcceptance(state_dim, hidden_dims)
    else:
        raise ValueError("Unknown acceptance mode")
    return MinimaxModel(
        MHSampler(proposal, ratio, target_log_prob), Critics(state_dim, hidden_dims)
    )


def save_checkpoint(path, trainer, arguments):
    state = trainer.state_dict()
    state["arguments"] = arguments
    state["torch_rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def held_out_game(model, pool, initial_sampler, config):
    terms = model.game(
        pool(config.batch_size),
        initial_sampler(config.batch_size),
        pool(config.reference_size),
        config.horizon,
        config.lambda_balance,
        config.lambda_mix,
    )
    return {"validation/" + name: value for name, value in terms.metrics().items()}


@torch.no_grad()
def evaluate_chains(model, initial, draws, burn_in, validation_data, output):
    model.eval()
    result = model.sampler.sample(initial, draws, burn_in)
    torch.save(
        {
            "samples": result.samples.cpu(),
            "acceptance_rate": result.acceptance_rate.cpu(),
        },
        output / "samples.pt",
    )
    samples = result.samples.flatten(0, 1)
    diagnostics = {
        "acceptance_rate": float(result.acceptance_rate.mean()),
        "mean": samples.mean(0).tolist(),
        "second_moment": samples.square().mean(0).tolist(),
        "validation_mean": validation_data.mean(0).tolist(),
        "validation_second_moment": validation_data.square().mean(0).tolist(),
        "right_half_probability": float((samples[:, 0] > 0).float().mean()),
        "validation_right_half_probability": float(
            (validation_data[:, 0] > 0).float().mean()
        ),
    }
    if draws > 1:
        samples = result.samples
        diagnostics["squared_jump_per_step"] = float(
            (samples[1:] - samples[:-1]).square().sum(-1).mean()
        )
        diagnostics["half_space_switch_rate"] = float(
            ((samples[1:, :, 0] > 0) != (samples[:-1, :, 0] > 0)).float().mean()
        )
    (output / "evaluation.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n"
    )
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acceptance", choices=["exact", "learned", "direct"], default="exact"
    )
    parser.add_argument(
        "--data", help="Offline .pt tensor (N,D), or dict containing 'samples'"
    )
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=16384)
    parser.add_argument("--validation-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reference-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--classifier-steps", type=int, default=0)
    parser.add_argument("--classifier-warmup", type=int, default=0)
    parser.add_argument("--sampler-lr", type=float, default=1e-4)
    parser.add_argument("--acceptance-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-balance", type=float, default=10.0)
    parser.add_argument("--lambda-mix", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--initial-proposal-scale", type=float, default=1.0)
    parser.add_argument("--init-scale", type=float, default=3.0)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output", help="New output directory (defaults to timestamped runs directory)"
    )
    args = parser.parse_args(argv)
    positive = [
        args.iterations,
        args.dim,
        args.train_size,
        args.validation_size,
        args.hidden_width,
        args.draws,
        args.chains,
        args.log_every,
        args.save_every,
        args.threads,
        args.init_scale,
    ]
    if (
        any(value <= 0 for value in positive)
        or args.burn_in < 0
        or args.classifier_warmup < 0
    ):
        parser.error(
            "Counts/scales must be positive; burn-in and warmup must be nonnegative"
        )
    if (
        args.classifier_steps or args.classifier_warmup
    ) and args.acceptance != "learned":
        parser.error("Classifier training is available only with --acceptance learned")
    if args.data and args.acceptance == "exact":
        parser.error(
            "Offline samples provide no density; choose learned/direct, or use the Python API with a target density"
        )
    try:
        config = TrainingConfig(
            batch_size=args.batch_size,
            reference_size=args.reference_size,
            horizon=args.horizon,
            critic_steps=args.critic_steps,
            classifier_steps=args.classifier_steps,
            sampler_lr=args.sampler_lr,
            critic_lr=args.critic_lr,
            acceptance_lr=args.acceptance_lr,
            lambda_balance=args.lambda_balance,
            lambda_mix=args.lambda_mix,
            grad_clip=args.grad_clip,
        )
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    target = GaussianMixture(args.dim, device=device)
    if args.data:
        loaded = torch.load(args.data, map_location="cpu", weights_only=True)
        data = loaded["samples"] if isinstance(loaded, dict) else loaded
        if not isinstance(data, torch.Tensor) or data.ndim != 2 or len(data) < 8:
            parser.error("Offline data must be a tensor of shape (N,D), with N >= 8")
        data = data.to(device=device, dtype=torch.float32)
        SamplePool(data)  # validate all values before splitting
        indices = torch.randperm(len(data), device=device)
        split = max(2, len(data) // 5)
        validation_data, training_data = data[indices[:split]], data[indices[split:]]
        args.dim = data.shape[1]
    else:
        training_data, validation_data = target.sample(args.train_size), target.sample(
            args.validation_size
        )
    train_pool, validation_pool = SamplePool(training_data), SamplePool(validation_data)
    initial_sampler = lambda count: args.init_scale * torch.randn(
        count, args.dim, device=device
    )
    model = build_model(
        args.dim,
        args.acceptance,
        (args.hidden_width, args.hidden_width),
        args.initial_proposal_scale,
        target.log_prob if args.acceptance == "exact" else None,
    ).to(device)
    trainer = MinimaxTrainer(model, train_pool, initial_sampler, config)
    output = (
        Path(args.output)
        if args.output
        else Path("runs")
        / (args.acceptance + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps({**vars(args), "training": asdict(config)}, indent=2) + "\n"
    )
    started = time.perf_counter()
    for _ in range(args.classifier_warmup):
        trainer.classifier_step()
    with (output / "metrics.jsonl").open("w") as log:
        for iteration in range(1, args.iterations + 1):
            metrics = trainer.step()
            if (
                iteration == 1
                or iteration % args.log_every == 0
                or iteration == args.iterations
            ):
                metrics.update(
                    held_out_game(model, validation_pool, initial_sampler, config)
                )
                metrics["elapsed_seconds"] = time.perf_counter() - started
                line = json.dumps(metrics, allow_nan=False)
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)
            if iteration % args.save_every == 0:
                save_checkpoint(output / "checkpoint.pt", trainer, vars(args))
    save_checkpoint(output / "checkpoint.pt", trainer, vars(args))
    evaluation = evaluate_chains(
        model,
        initial_sampler(args.chains),
        args.draws,
        args.burn_in,
        validation_data,
        output,
    )
    print(
        json.dumps({"output": str(output), "evaluation": evaluation}, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
