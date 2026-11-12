# Adversarial MH sampler

PyTorch models and alternating training for the minimax objective in Section 3.5
of [`tex/gan_mcmc.tex`](tex/gan_mcmc.tex). The same proposal and sampling kernel
support analytic MH, classifier-based MH, and directly learned acceptance.

## Architecture

| Component | Implementation | Role |
| --- | --- | --- |
| Conditional proposal | `ConditionalGaussianProposal` | Residual mean and positive diagonal standard deviations; starts as a Gaussian random walk. |
| Optional implicit proposal | `ImplicitProposal` | Wraps a reparameterizable generator `G(x,z)` without a density. |
| Analytic acceptance | `ExactMHRatio` | Target log-density difference plus reverse/forward proposal log-density difference. |
| Classifier acceptance | `LearnedMHRatio` | Antisymmetric pair logit `s(x,y)=u(x,y)-u(y,x)`, with MH log ratio `-s(x,y)`. |
| Direct acceptance | `DirectAcceptance` | Pair MLP with sigmoid probability, optimized jointly with the proposal. |
| Endpoint critic f | `Critics.endpoint` | Bounded scalar state network, maximizes endpoint distribution mismatch. |
| Balance critic h | `Critics.balance` | Bounded scalar pair network; its forward-minus-reverse value tests accepted-flow asymmetry. |
| Mixing critic g | `Critics.mixing` | Scalar state observable, normalized to reference-batch mean zero and variance one. |
| Game and optimization | `MinimaxModel`, `MinimaxTrainer` | Alternating critic ascent and proposal/acceptance descent. |
| Chain generation | `MHSampler` | Parallel chains, burn-in, thinning, and retained rejection repeats. |

Networks default to two hidden layers of width 64 and SiLU activations. Endpoint
and balance outputs use tanh. The mixing observable has no output bound; its
empirical normalization enforces the game's variance constraint.

Exact MH requires evaluable target and proposal densities, but no target
normalizing constant. Exact MH with a fully implicit proposal needs a separately
derived construction; the implementation raises if proposal density is unavailable.
In learned and direct modes the sampler and game do not call a target density.
The default CLI synthetic target supplies both target samples and a density;
with offline data only learned/direct modes are available through the CLI.

## Objective and updates

The implemented scalar is exactly the empirical neural-class game

```text
L = mean_pi f - mean_nu0_K^T f
    + lambda_balance * mean_pi_q [a(x,y) * (h(x,y) - h(y,x))]
    - lambda_mix * 0.5 * mean_pi_q [a(x,y) * (g(y) - g(x))^2].
```

Each iteration performs `critic_steps` ascent updates on L with the sampler
frozen, then one descent update on L with the critics frozen. Freezing parameters
preserves derivatives through their inputs. Both proposal and learned acceptance
parameters participate in sampler descent; exact acceptance has no parameters.
The signs make f/h maximize their discrepancies and g minimize Dirichlet energy.
Different fresh training batches are used for each update. A separate target
reference batch defines g's mean/variance consistently for both states in a pair.

The finite-T endpoint term explicitly expands rejection/acceptance branches with
their exact probabilities. This gives differentiable, unbiased Monte Carlo
estimates of endpoint expectations for reparameterizable proposals without
changing the chain into soft averaged states. Default T=1; `--horizon` supports
1 through 8, with O(batch_size * 2**T) leaves. The reference normalization is an
empirical approximation to the population constraint. It uses no denominator
epsilon that could admit constant observables; collapsed observables are restarted
and their optimizer state cleared. Restarts and raw observable scale are logged.

Learned classifier acceptance can optionally receive BCE warmup and auxiliary
BCE updates using forward/reversed proposal pairs. Both default to zero: the main
updates then optimize only the Section 3.5 objective. Enabling BCE changes the
training algorithm by adding a separate supervised step; it is not silently
included in L. Direct acceptance has no density-ratio classifier loss.

The game uses target training samples even in exact mode. An energy function alone
does not supply these training samples; bootstrapping them is outside this example.
For pathwise exact-mode training, log densities must also support input gradients.

## Run training

Requires Python 3.10+ and PyTorch 2.0+. Run from this directory or install with
`python -m pip install -e .`. Tests require pytest.

```bash
# Analytic MH: train the proposal while preserving the target analytically.
python train.py --acceptance exact --iterations 1000 --output runs/exact

# Section 3.5 with a directly learned acceptance function.
python train.py --acceptance direct --iterations 1000 --output runs/direct

# Classifier acceptance: pure game, jointly updating proposal and classifier.
python train.py --acceptance learned --iterations 1000 --output runs/learned

# Optional classifier supervision; fresh proposal pairs are generated each step.
python train.py --acceptance learned --classifier-warmup 200 --classifier-steps 1 \
  --iterations 1000 --output runs/learned_aux

# Offline data: a finite floating-point tensor (N,D), N >= 8.
python train.py --data training_samples.pt --acceptance direct --output runs/offline

# Short two-step endpoint objective.
python train.py --acceptance exact --horizon 2 --iterations 100 --output runs/two_step
```

`python -m adversarial_mh.train` is equivalent. The default benchmark is an
equal-weight Gaussian mixture centered at (-2,0) and (2,0), with identity component
covariances. Synthetic training and validation pools are generated separately;
offline tensors are randomly split 80/20, with disjoint rows. Duplicate input rows
may still occur across the split. Training samples are drawn from the training pool
with replacement, and validation never participates in optimization.

Use `--lambda-balance`, `--lambda-mix`, `--critic-steps`, `--sampler-lr`,
`--acceptance-lr`, `--critic-lr`, `--batch-size`, `--reference-size`, and `--horizon`
to configure the game. Other options include `--hidden-width`, `--device`, `--seed`,
`--init-scale`, and `--initial-proposal-scale`. Run `python train.py --help` for all
options. Existing nonempty output directories are not overwritten.

Each run saves:

- `config.json`: architecture, training settings, and seed.
- `metrics.jsonl`: training and held-out endpoint/balance witnesses, Dirichlet
  surrogate, acceptance, accepted squared jump, gradient norms, and restarts.
- `checkpoint.pt`: model, both optimizers, configuration, iteration, and Torch RNG
  states; saved periodically and at the end of training.
- `samples.pt`: final frozen-chain samples with shape (draws,chains,D), plus acceptance rates.
- `evaluation.json`: sample moments, validation moments, half-space occupancy,
  switching frequency, and accepted-state jump distances.

Witness values are signed evaluations of the current critics, not optimized
nonnegative discrepancy estimates. A neural Dirichlet surrogate can overestimate
the true gap. Validation metrics and moment checks are diagnostics, not convergence
certificates. Learned/direct acceptance can introduce stationary bias. Exact MH
has the usual invariant-target guarantee under MH assumptions, but does not
ensure rapid mixing. All networks are frozen during final chain generation.

## Python API

```python
import torch
from adversarial_mh import (
    ConditionalGaussianProposal, DirectAcceptance, MHSampler,
    Critics, MinimaxModel, MinimaxTrainer, TrainingConfig,
)

sampler = MHSampler(ConditionalGaussianProposal(2), DirectAcceptance(2))
model = MinimaxModel(sampler, Critics(2))
# Replace this normal target with a sampler from your training-data pool.
target_samples = lambda n: torch.randn(n, 2)
initial_samples = lambda n: 3 * torch.randn(n, 2)
trainer = MinimaxTrainer(model, target_samples, initial_samples, TrainingConfig())
for _ in range(1000):
    metrics = trainer.step()
result = sampler.sample(torch.zeros(16, 2), 1000, burn_in=200)
```

Use `ExactMHRatio()` with `MHSampler(..., target_log_prob=log_pi)` for exact MH,
or `LearnedMHRatio(2)` for classifier acceptance. The common differentiable API is
`sampler.log_acceptance(x,y)`; `sampler.log_ratio` is undefined for direct acceptance.
Standalone classifier training remains available through `classifier_loss`.
`one_step_expectation` integrates out a single decision for scalar observables.

To restore a training checkpoint, reconstruct the model and trainer using the saved
architecture, target density (for exact mode), and `TrainingConfig`, then call
`trainer.load_state_dict(torch.load(path, weights_only=True))`. Restore
`torch_rng_state` and, if applicable, `cuda_rng_states` explicitly for random-stream
continuation. The CLI starts new runs; checkpoint continuation is through this API.

## Verification

```bash
python -m pytest -q
python -m examples.compare_ratios
```

Tests cover analytic detailed balance for asymmetric proposals, learned-ratio
training, target moments, direct acceptance, the game against a hand-computed
example, exact branch expectations and gradients, observable normalization and
collapse recovery, ascent/descent signs, freezing the other player, checkpoint
restoration, and offline CLI artifacts. The older ratio-comparison example keeps
its proposal fixed to isolate classifier accuracy.

### Initial CPU results

With seed 123, the default 1,000 training iterations, and 32 chains of 1,000
post-burn-in draws, all three modes completed without nonfinite losses or
observable restarts. The pure minimax learned/direct runs had substantial bias:

| Mode | E[x1^2] | E[x2^2] | P(x1 > 0) |
| --- | ---: | ---: | ---: |
| Target | 5.000 | 1.000 | 0.500 |
| Exact MH | 4.994 | 0.992 | 0.478 |
| Learned MH, pure game | 4.868 | 2.094 | 0.742 |
| Direct acceptance | 3.615 | 1.271 | 0.851 |

These are single-seed diagnostics, not estimates with uncertainty intervals. They
verify runnable training and expose the present optimization/correctness limits;
the default learned/direct settings should not be regarded as calibrated samplers.
