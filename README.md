# LeJEPA-SAE

Sparse feature discovery from individual LLM residual activations. The new `rdm_sae` pilot
combines reconstruction and RDMReg; the existing single-token dimension-mask JEPA (`proposed`)
and SAE baselines remain available with their existing checkpoints. This repository contains
no multi-token Transformer/CLS model.

## Reconstruction + RDM SAE pilot

`rdm_sae` uses one full residual token, with no masks, local views, invariance, rate loss,
L1 penalty, TopK, AuxK, or threshold calibration:

```text
z = ReLU(W_enc (h - b_pre) + b_enc)
h_hat = W_dec z + b_pre
L = reconstruction_weight * mean((h_hat - h)^2)
  + lambda_rdm * (L_random(z, target) + axis_weight * L_axis(z, target))
```

The untied decoder columns are initialized and maintained at unit norm, with encoder weights
initialized to the decoder transpose. Inputs are not normalized; reconstruction MSE and RDM
reductions use float32. RDM sees a **single view at full weight**, using 8192 random projections
and 512 axes. Train, evaluation and probing all use the same pointwise encoder.

The dedicated preset keeps width 16384, target active fraction 0.05, batch 512 / accumulation 1,
seed 42, 10,000 optimizer steps (5.12M token presentations), and the shared AdamW/schedule.
It starts from **fresh SAE initialization**, not a JEPA checkpoint. By default its forward is
exact ReLU with leaky backward slope 0.1; select ordinary ReLU with `FEATURE_ACTIVATION=relu`.
The original configs and comparison pipeline are unchanged; this launcher runs one model only.

```bash
bash scripts/run_rdm_sae.sh runs/rdm-sae/pilot-seed42-steps10000

# Explicit weights and target amplitude; use a fresh output for every experiment.
RDM_WEIGHT=1 RECONSTRUCTION_WEIGHT=1 RDM_TARGET_SCALE=1 \
  EXPECTED_L0_FRACTION=0.05 LEAKY_BACKWARD_SLOPE=0.1 \
  bash scripts/run_rdm_sae.sh runs/rdm-sae/custom-pilot
```

`lambda_rdm=1.0` is an **untuned starting point**, not a recommendation that the loss terms are
balanced. The old JEPA value 125 is not assumed appropriate for raw reconstruction MSE.
`loss.rdm_target_scale` multiplies the **whole rectified target**, including its effective mean
shift: `target = scale * ReLU(GN_p(mu, sigma_GN))`. It changes target amplitude but preserves
its expected active fraction (not a guarantee of the model's measured L0). Defaults to 1 for
old checkpoints as well. Target shape/shift can also be changed through the existing loss config.
Positive `reconstruction_weight` and non-negative `lambda_rdm` are required for `rdm_sae`;
`RDM_WEIGHT=0` is a reconstruction-only control that skips RDM sampling/computation entirely.

Train/validation logs include MSE, FVU, raw random/axis RDM, weighted reconstruction/RDM terms,
and log-time active fraction, L0, feature std and dead fractions. With
`loss.rdm_gradient_diagnostics=true` (preset default), log steps also measure each **weighted**
term's gradient RMS at encoder preactivations, before gradient clipping/accumulation scaling.
This includes the activation surrogate derivative; it is not an encoder-parameter gradient norm.
Gradient diagnostics are omitted during validation and non-log batches. Compare their ratio and
the validation MSE/FVU/L0 before choosing a new RDM weight; loss magnitudes alone are insufficient.
L0 and FVU in training logs are per-batch diagnostics; standalone evaluation pools the test set.

Other environment overrides: `FEATURE_DIM`, `SEED`, `MAX_STEPS`, `BATCH_SIZE`, `GRAD_ACCUM`,
`EVAL_BATCHES`, `CHECKPOINT_EVERY`, `RDM_PROJECTIONS`, `AXIS_PROJECTIONS`, `AXIS_WEIGHT`,
`RDM_GRADIENT_DIAGNOSTICS`, `CONFIG`, `OUTPUT_DIR` (positional output takes precedence).
The launcher refuses existing outputs. On OOM, use a fresh output with `BATCH_SIZE=256 GRAD_ACCUM=2`,
then `128/4` if needed; SWD then uses a smaller microbatch and is not numerically equivalent.
For dataset/device/optimizer overrides or explicit resume, use `lejepa-train --config
configs/pythia-6.9b-layer16-rdm-sae.yaml --set section.key=value` directly.

```bash
RUN=runs/rdm-sae/pilot-seed42-steps10000
lejepa-evaluate --config "$RUN/config.resolved.yaml" \
  --checkpoint "$RUN/checkpoint-00010000.pt" --output-dir "$RUN/evaluation"
bash scripts/run_probe_pilot.sh smoke "$RUN"
bash scripts/run_probe_pilot.sh probe "$RUN"
```

Evaluation generates `index.html` and training curves including active fraction, reconstruction/FVU,
random/axis RDM, weighted losses and gradient balance. Probing needs no threshold calibration for
`rdm_sae`; the existing hook-parity preflight and `normal`, k=1/16 task protocol are unchanged.
The probe-pilot launcher expects a 10k-step checkpoint. Compare measured L0 (not only target rho),
FVU and the same complete task set against BatchTopK and the retained RDM-only/JEPA runs.

### W1 versus W2-squared ablation

Set `RDM_WASSERSTEIN_POWER=1` in the RDM-SAE launcher, or use
`lejepa-train --config configs/pythia-6.9b-layer16-rdm-sae.yaml --set loss.rdm_wasserstein_power=1`.
The default is `2`, including when loading older configs that omit this field.
Only integer `1` or `2` is accepted. This transport power is independent of the target shape
`loss.lp_norm_parameter` (`1.0` for Laplace, `2.0` for Gaussian).

For sorted projected samples with difference `d = x_sorted - y_sorted`, both the random and
coordinate-axis terms use `mean(abs(d))` for power 1, or the historical `mean(d.square())`
for power 2. The latter is **W2 squared**, without a square root. Projection sampling, target
sampling/scaling, float32 reductions, view weights, decoder constraints and activation behavior
are unchanged. The existing `sliced_wasserstein_2_*` Python functions retain W2-squared behavior;
the generic `sliced_wasserstein_with_projections` and `sliced_wasserstein_on_axes` functions accept
the keyword-only `wasserstein_power=1|2`.

Run each condition from fresh initialization with the same settings and data:

```bash
# Control: W2 squared
RDM_TARGET_SCALE=1.5 AXIS_WEIGHT=4 RDM_WEIGHT=1 RDM_WASSERSTEIN_POWER=2 \
  EXPECTED_L0_FRACTION=0.05 SEED=42 MAX_STEPS=10000 \
  bash scripts/run_rdm_sae.sh runs/rdm-sae/w2-scale1.5-axis4-seed42-steps10000

# Ablation: W1
RDM_TARGET_SCALE=1.5 AXIS_WEIGHT=4 RDM_WEIGHT=1 RDM_WASSERSTEIN_POWER=1 \
  EXPECTED_L0_FRACTION=0.05 SEED=42 MAX_STEPS=10000 \
  bash scripts/run_rdm_sae.sh runs/rdm-sae/w1-scale1.5-axis4-seed42-steps10000
```

The preset keeps target shape Laplace, forward ReLU / leaky backward slope 0.1, batch 512,
and accumulation 1. The automatically generated output name includes `wp1` or `wp2` and the
axis weight; explicit positional output directories still take precedence.

RDM-SAE train log steps and validation now include `active_fraction_gt_0`,
`active_fraction_gt_1e-4`, `active_fraction_gt_1e-3`, `active_fraction_gt_1e-2`,
`active_fraction_gt_5e-2`, and `active_fraction_gt_1e-1`. These measure strict `z > threshold`
in float32 over all token-feature entries; they never threshold the actual encoder output.
`active_fraction_gt_0` matches the existing strict `active_fraction`. Standalone
`lejepa-evaluate` also exports these diagnostics in `metrics.json` and the report, pooling
counts over evaluated tokens (including a short final batch), independently of `--support-epsilon`.
It can therefore diagnose leakage in existing W2 checkpoints without retraining.

With `RDM_GRADIENT_DIAGNOSTICS=true`, training log steps additionally record
`rdm_random_preactivation_grad_rms` for `lambda_rdm * L_random` and
`rdm_axis_preactivation_grad_rms` for `lambda_rdm * axis_weight * L_axis`.
They include the activation surrogate derivative and are measured before gradient clipping
or accumulation scaling. These RMS values do not add up to the combined RDM gradient RMS.
They are omitted during validation and non-log steps, and are zero when `RDM_WEIGHT=0`.
Training-history CSV and curves retain the transport power, threshold diagnostics and component
gradients. Positive-activation quantiles are omitted to keep logging overhead bounded.

Compare FVU, strict L0, thresholded activity and feature usage together. The existing
`random_distribution` and `axis_distribution` logs use the configured metric, so their absolute
values are not directly comparable between W1 and W2 squared. Equal loss coefficients also do
not imply equal regularization strength. For the Laplace target above, the theoretical
`P(target > t) = 0.05 * exp(-t / (1.5 / sqrt(2)))`; at `t=0.1` the reference is about 0.04550,
not 0.05. Batch-level `dead_feature_fraction` is not a measure of permanent feature death;
use pooled evaluation feature rates as well. No L1/L0 penalty, TopK, shrinkage or calibration
is introduced by this ablation.

## Proposed model

For one frozen residual activation `h_t ∈ R^4096`, the global view is complete and each of four
local views retains an independently sampled, exact half of its coordinates. The shared encoder is:

```text
h_t → subtract learned pre-bias → exact coordinate mask → configurable mask scaling
    → Linear(4096, 16384) → ReLU → z
```

Missing coordinates are therefore filled with the learned pre-bias and become zero after
centering. The global and four local views are encoded in one batched Linear call. There is no
Transformer, CLS token, decoder, target encoder, or stop-gradient.
The default remains inverted scaling (`1/q`); the `sqrt` and `none` ablations below only
change the local input multiplier.

The objective is:

```text
L = 25 · mean_v MSE(z_global, z_local_v)
  + 125 · (L_random-RDMReg + 1.0 · L_axis-RDMReg)
```

RDMReg follows Rectified LpJEPA: every view is matched to an independent rectified generalized
Gaussian target. The default is `ReLU(Laplace(-2.78299, 1/√2))`, whose expected nonzero fraction is
`0.009765625`; at width 16384 this is 160 active features per sample in expectation. The mean shift
is derived from `loss.expected_l0_fraction`, including for nonstandard `p`, rather than being tied
to this default. Set it to `null` to use `loss.mean_shift_value` directly.

The distribution loss uses 8192 shared random unit projections. In addition, every optimizer step
samples 512 coordinate axes without replacement and directly matches the selected feature
marginals. Axis indices are shared by all five views, and the same view-specific target is used by
both losses. Coordinate values are gathered directly; no one-hot matrix is materialized. Projection
matmul is mixed precision; sorting and loss reduction are float32.

## Environment

Target: Python 3.10+, CUDA 12.1, PyTorch 2.5.1, one RTX 4090.

```bash
git clone https://github.com/fumin0ri/LeJEPA-SAE.git
cd LeJEPA-SAE
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[dev]'
# Only on the probing environment (adds TransformerLens, SAELens, sklearn, etc.)
pip install -e '.[probes]'
```

## 1. Extract Pythia residual activations

The default extraction streams the Hugging Face-hosted `monology/pile-uncopyrighted-parquet`
mirror, sends each document to Pythia in chunks of up to 1024 tokens, and stops at 100M tokenized
source tokens:

```bash
bash scripts/extract_the_pile.sh
```

This mirror removes Books3, BookCorpus2, OpenSubtitles, YTSubtitles, and OWT2 from The Pile; it is
used by default because it streams through the same Hugging Face access path as the model and does
not depend on The Eye. To use the exact standard Pile when it is mounted locally, override the
dataset files and token budget without editing the script:

```bash
DATA_FILES='/another/path/*.jsonl.zst' MAX_SOURCE_TOKENS=100000000 \
  bash scripts/extract_the_pile.sh
```

The extractor hooks zero-based Pythia block output 16 and writes bf16 activations plus token IDs to
`data/the-pile/pythia-6.9b/layer-16-ctx1024-100m`. The manifest records both the requested limit and
the source-token count actually consumed. These are source-token presentations, not semantic corpus
deduplication. At 100M tokens, width-4096 bf16 activations require about 763 GiB before filesystem
overhead. The manifest uses `minimum_window_size: 1`; every training and evaluation sample is one
token.

For locally stored raw Pile shards:

```bash
lejepa-extract \
  --dataset json \
  --data-files '/datasets/the-pile/train/*.jsonl.zst' \
  --source-split train \
  --text-column text \
  --model EleutherAI/pythia-6.9b \
  --layer 16 \
  --context-length 1024 \
  --window-size 1 \
  --dtype bfloat16 \
  --max-source-tokens 100000000 \
  --output-dir data/the-pile/pythia-6.9b/layer-16-ctx1024-100m
```

Extraction refuses to overwrite an existing output directory. Choose a new directory or move the
old extraction before rerunning with different settings.

## 2. Train the proposed model

```bash
bash scripts/run_proposed.sh
```

The default uses batch 512, no gradient accumulation, and `train.max_steps: one_epoch`. After
loading the activation manifest, this resolves to the number of complete optimizer batches in the
actual train split. With 100M total tokens and the default 98%/1%/1% document split, this is about
191,406 steps; the exact value and consumed sample count are written to `training_plan.json`. The
fresh output is
`runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d16384-l0-0.009765625-axis512`.
The output is deliberately separate from
pre-axis checkpoints and starts from a fresh initialization. Do not resume a collapsed checkpoint:
axis loss discourages feature death while gradients still cross ReLU, but cannot guarantee revival
once every sample for a feature is in ReLU's negative region.

### Mask scaling ablations at q=0.5

`model.mask_scaling` controls scaling after masking the centered activation `x = h_t - b`:

| Setting | Local input | Multiplier at q=0.5 |
|---|---|---:|
| `inverted` (default) | `m * x / q` | 2 |
| `sqrt` | `m * x / sqrt(q)` | 1.41421356 |
| `none` | `m * x` | 1 |

All three retain exactly 2048 of 4096 coordinates, independently for every sample and local view.
Global input remains `h_t - b`, including in the batched all-one-mask path. Missing coordinates
remain centered zeros, and encoder bias is not scaled. The actual keep fraction after rounding
determines the divisor when a different `q` is used.

Run the new experiments from fresh initialization:

```bash
MASK_SCALING=sqrt bash scripts/run_proposed.sh
MASK_SCALING=none bash scripts/run_proposed.sh
```

These use q=0.5 by default and append `-q0.5-mask-sqrt` or `-q0.5-mask-none` to the output
directory. The original `inverted`/q=0.5 output path is unchanged. To combine with leaky backward,
use the same environment variable with `scripts/run_leaky_backward.sh`. To change the keep rate,
set `DIMENSION_KEEP_FRACTION`; direct CLI overrides are also supported:

```bash
lejepa-train --config configs/pythia-6.9b-layer16.yaml \
  --set model.dimension_keep_fraction=0.5 \
  --set model.mask_scaling=sqrt \
  --set train.output_dir=runs/proposed-q0.5-mask-sqrt \
  --set train.resume_from=null
```

Use the run's `config.resolved.yaml` for evaluation: the choice is saved in both that file and
checkpoint config. Existing configs without `mask_scaling` continue to use `inverted`. Baseline
SAEs are unmasked and unaffected. Start each ablation fresh rather than resuming another scaling
mode's checkpoint.

`sqrt` preserves the centered input's squared norm in expectation over the masks. It does **not**
guarantee equal encoder preactivation variance when coordinates are correlated. Unlike `inverted`,
it also changes the expected local linear projection: it is `sqrt(q)` times the global linear
term (`q` times for `none`, before adding encoder bias). These are distinct training objectives.

### Target-rate-only ablation (no support or margin loss)

The original proposed objective is unchanged by default (`loss.rate_weight: 0`). The opt-in
ablation adds a target-anchored rate penalty, sharing `loss.expected_l0_fraction = rho` with RDMReg:

```text
ell(r) = (r - rho)^2 / (2 * rho * (1 - rho))
L_rate = 0.5 * ell(r_global) + 0.5 * mean_v ell(r_local_v)
L_total = 25 * L_invariance + 125 * L_RDM + rate_weight * L_rate
```

The 25/125 base weights remain configurable. Each `r` averages over that view's batch and feature
axes. Local penalties are computed separately before averaging, not after pooling their rates.
This controls the view-level mean, not an exact per-token TopK. The rate penalty is normalized
squared error, the second-order approximation to Bernoulli KL at the target. It remains finite
at rates 0 and 1 without clamping the rate or blocking its surrogate gradient.

Rates use hard-forward/sigmoid-backward gates directly on preactivations (not ReLU outputs):

```text
scale = stop_gradient(std(global_preactivations, unbiased=False)).clamp_min(rate_scale_floor)
soft = sigmoid(a.float() / (rate_temperature * scale))
gate = (a > 0).float() + (soft - stop_gradient(soft))
```

Gate calculations and reductions use float32. The default temperature multiplier is 0.1 and
scale floor is 1e-6; the same detached global scale is used for all local views. Both branches
receive rate gradients directly through preactivation, independently of the ReLU backward
choice. Sigmoid saturation can still suppress gradients far from zero: this is not a guarantee
of dead-feature revival. Weight zero skips gate/scale computations and preserves the old
loss, gradients, RNG usage, and state-dict format.

To compare the current `sqrt(q), q=0.5, slope=0.1, rho=0.05` experiment against `+ rate`:

```bash
RATE_WEIGHT=1.0 bash scripts/run_rate_comparison.sh
```

This runs **only two proposed-model pilots**, sequentially, with shared seed 42, initialization,
data-order seed, and training settings. Both use ReLU-forward/leaky-backward with slope 0.1.
Each pilot runs 10,000 optimizer steps by default, not a full 100M-token epoch. It uses separate
`base/` and `rate/` directories below `runs/rate-ablation/`; it checks all selected destinations
before starting and refuses existing ones. Neither run resumes an old checkpoint. `base` or
`rate` as the first argument runs only that condition. SAE baselines and the 15-run comparison
pipeline are not launched or modified.

`RATE_WEIGHT=1.0` is an **uncalibrated pilot starting value**, not a demonstrated optimum. The
comparison launcher enables gradient diagnostics at train log steps: `rate_to_base_grad_ratio`
is RMS(weighted rate gradient) / RMS(weighted base gradient), measured over the complete
preactivation tensor. A ratio around 0.05–0.1 is an initial heuristic, not a constraint or an
automatic weight adjustment. The diagnostic adds backward work on log steps only; disable it
with `RATE_GRAD_DIAGNOSTICS=false`. Validation never computes these gradients.

For example, override the pilot settings and give the pair a new output root:

```bash
RATE_WEIGHT=0.3 RATE_TEMPERATURE=0.1 TRAIN_SEED=43 MAX_STEPS=10000 \
RATE_COMPARISON_ROOT=runs/rate-ablation/seed43-weight03 \
bash scripts/run_rate_comparison.sh both
```

Other shared overrides include `EXPECTED_L0_FRACTION`, `DIMENSION_KEEP_FRACTION`, `MASK_SCALING`,
`LEAKY_BACKWARD_SLOPE`, `FEATURE_DIM`, `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`,
`RATE_SCALE_FLOOR`, and `CONFIG`. Direct training supports `--set loss.rate_weight=...`,
`--set loss.rate_temperature=...`, `--set loss.rate_scale_floor=...`, and
`--set loss.rate_gradient_diagnostics=true`. The existing `run_proposed.sh` and
`run_leaky_backward.sh` also accept these rate environment variables; enabled-rate runs append
a weight/temperature/floor suffix to their default output path. Use the dedicated comparison
launcher for fresh paired outputs. Rate loss is rejected for SAE baselines or when
`expected_l0_fraction` is null.

Enabled-rate logs include `rate_loss`, `global_rate_loss`, `local_rate_loss`, `rate_contribution`,
`base_loss`, `rate_global_active_fraction`, `rate_local_active_fraction`, and `rate_scale`.
These are collected every loss batch (train logs average the interval), whereas the existing
`global_active_fraction`/`local_active_fraction` diagnostics sample the final microbatch of a log
step. Compare matched aggregation scopes. Evaluation reports made with each run's resolved config
include the rate-loss and optional gradient-ratio training curves and CSV fields.
Judge the ablation by target-rate error, `support_disagreement = off_to_on + on_to_off`,
global/local MSE, RDMReg, feature std, and dead-feature diagnostics, not sparsity gap alone.
No Jaccard loss, margin loss, or automatic hyperparameter sweep is included.

### Global-only RDMReg control (no mask, no invariance)

Set `model.num_local_views=0`, `loss.invariance_weight=0`, and `loss.rate_weight=0`
to train the same proposed encoder using only the complete global residual. This performs
one unmasked encoder forward and one independent RDM target per batch. No dimension masks,
local branches, invariance MSE, rate loss, or decoder are computed. The objective is:

```text
L = lambda_rdm * (L_global_random + axis_weight * L_global_axis)
```

The global term gets **100%** of the RDM weight, not the 50% contribution used in the
global/local experiment. Random and axis projection counts, target distribution/L0,
encoder activation (including leaky backward), optimizer/schedule and batch settings
are inherited from the chosen run's saved config. Mask fraction/scaling fields stay in
that config for provenance but are unused. Nonzero invariance/rate weights are rejected
when `num_local_views=0`.

Use the base run as the config source; its checkpoint is neither read nor resumed:

```bash
BASE_DIR="runs/rate-ablation/d16384-rho0.05-q0.5-sqrt-slope0.1-rate100-tau0.1-floor0.000001-seed42-steps10000/base"
OUT_DIR="runs/global-rdm-only/d16384-rho0.05-slope0.1-seed42-steps10000"
bash scripts/run_global_rdm_only.sh "$BASE_DIR" "$OUT_DIR"

# After training, evaluate all normal tasks at k=1,16 using the existing shared cache.
bash scripts/run_probe_pilot.sh probe "$OUT_DIR"
```

This launcher fixes 10,000 steps and seed 42, starts from a fresh initialization, refuses
any existing output path, and launches only this single training run. Batch 512 means
5.12M token presentations, as in the other pilots; the number of encoded views differs.
For other step/seed settings use `lejepa-train --config ... --set ...` directly.

Logs retain global active fraction, feature std, batch-dead fraction, L0/L1 metrics,
random/axis RDMReg and throughput. `global_rdm_contribution == distribution` here.
Local metrics, gate transitions and invariance are **absent**, not reported as zero;
standalone evaluation likewise does not sample a diagnostic mask. The checkpoint and
global probe adapter remain compatible with the proposed model. Data-order seed is
preserved, but deleting mask/local-target draws changes the stochastic RDM RNG stream,
so this is not a matched-random-target experiment.

### Global/local gate-transition diagnostics

Training and validation log `off_to_on = P(a_G <= 0, a_L > 0)` and
`on_to_off = P(a_G > 0, a_L <= 0)`. The denominator is **all** paired feature coordinates
across tokens and local views, not only globally off/on features. Training uses every configured
local view (four by default), paired with the same token's global features. Both forward activations
are exact ReLU, so `z > 0` gives the same gate as `a > 0`, including the zero boundary.

Two additional fields check the identity:

```text
local_global_active_fraction_gap = local_active_fraction - global_active_fraction
transition_rate_gap = off_to_on - on_to_off
```

These should agree up to floating-point rounding. JSONL/CSV values are fractions (`0.041` means
4.1%); the report plots transitions in percent and the gaps in percentage points. The training
curves include both train/validation transitions and the overlapping gap curves. Older logs
without these fields remain readable, but transition rates cannot be recovered from L0 alone.

Like other feature diagnostics, training measures these only on the final microbatch of each
log step; validation measures them on every evaluated batch. This adds no extra encoder forward,
does not change the objective, and works with existing checkpoints. Standalone evaluation also
reports the rates for its sampled global/local pair per token, using strict zero independently
of `--support-epsilon`.

### ReLU forward + leaky backward ablation

For the high-sparsity regime, a dedicated experiment keeps exact ReLU feature values in the
forward pass but uses a leaky surrogate derivative in the non-positive region:

```text
forward:  z = max(0, a)
backward: dz/da = 1 if a > 0, otherwise alpha
```

Run it from a fresh initialization with:

```bash
bash scripts/run_leaky_backward.sh
```

The default is `alpha=0.01`. Override it without editing the preset, for example:

```bash
LEAKY_BACKWARD_SLOPE=0.05 bash scripts/run_leaky_backward.sh
```

This changes only the training gradient: activations, active fraction, L0 diagnostics, RDMReg
inputs, and evaluation remain based on exact ReLU outputs. The run is written to a distinct
`...-relu_forward_leaky_backward-s<alpha>` directory. Normal ReLU remains the main preset default,
and `model.feature_activation` plus `model.leaky_backward_slope` are stored in the resolved config
for reproducibility.

If batch 512 is out of memory, preserve the effective batch in this order:

```bash
BATCH_SIZE=256 GRADIENT_ACCUMULATION_STEPS=2 bash scripts/run_proposed.sh
BATCH_SIZE=128 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/run_proposed.sh
```

Set `MAX_STEPS` to an integer to override one-epoch mode; `EVAL_BATCHES` is also an environment
override. Training logs core losses every
batch—including separate `random_distribution` and `axis_distribution` values—collapse
diagnostics at log steps, interval throughput, and CUDA peak memory. Axis ablations can use, for
example, `AXIS_PROJECTIONS=256 AXIS_WEIGHT=2.0 bash scripts/run_proposed.sh`; the default output
directory includes the selected width, expected L0 fraction, and axis count. Width and target L0
can be changed without editing the preset, for example:

```bash
FEATURE_DIM=32768 EXPECTED_L0_FRACTION=0.005 bash scripts/run_proposed.sh
```

RDMReg gives the complete global view and the group of four local views equal weight:

```text
L_RDM = 0.5 * L_global + 0.5 * mean(L_local_1, ..., L_local_4)
```

This grouping is applied independently to both random-projection and axis-projection losses. Logs
include the raw global/local losses and their weighted contributions, whose sum is the reported
`distribution` loss.

## 3. Strong SAE baselines and probing comparison

The former `standard_sae` and `dimension_denoising_sae` baselines have been removed. The supported
model types are now:

| Type | Sparse mechanism | Training objective |
|---|---|---|
| `proposed` | ReLU or ReLU-forward/leaky-backward | invariance + random/axis RDMReg |
| `batch_topk_sae` | [global BatchTopK](https://github.com/bartbussmann/BatchTopK), target `k=160` | full-token reconstruction + AuxK |
| `jump_relu_sae` | [learned feature-wise JumpReLU threshold](https://storage.googleapis.com/jumprelu-saes-paper/JumpReLU_SAEs.pdf) | reconstruction + warmed-up λL0 |
| `matryoshka_sae` | [Matryoshka BatchTopK](https://proceedings.mlr.press/v267/bussmann25a.html) with five nested prefixes | equally weighted prefix reconstructions + AuxK |

All SAE baselines use a learned decoder/pre-bias, an untied decoder whose columns remain unit norm,
and an encoder initialized to the decoder transpose. They do not normalize the residual input. The
outer training envelope is shared with the proposed model: width 16384, batch 512, no accumulation,
one pass over the activation dataset, AdamW at `1e-4`, and the same warmup/cosine schedule. The
default target is derived from `loss.expected_l0_fraction`, so
`round(16384 × 0.009765625) = 160`; set `baseline.k` to override it explicitly.

BatchTopK and Matryoshka use batch-level TopK only during training. Validation activations calibrate
a fixed scalar threshold, which is stored in `threshold_calibration.json` and the final checkpoint;
evaluation and probing are therefore pointwise and independent of batch composition. Matryoshka
uses groups `[512,1024,2048,4096,8704]`, hence cumulative prefix widths
`[512,1536,3584,7680,16384]`.

The complete comparison is resumable:

```bash
bash scripts/run_comparison.sh train
bash scripts/run_comparison.sh probe
# or both phases
bash scripts/run_comparison.sh all
```

Training first runs a seed-42, 20,000-step JumpReLU λ calibration over
`[1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2]`. If no candidate is within 10% of L0=160,
the grid extends by one decade in the required direction, at most three times. The selected λ and
all pilot L0/FVU results are stored in `jumprelu-pilot/jumprelu_calibration.json`. It is reused for
all three full seeds.

The full matrix is five series—proposed ReLU, proposed leaky-backward ablation, BatchTopK,
JumpReLU, and Matryoshka—at seeds 42/43/44. This is 15 full 100M-token runs plus the pilots. Runs
with a complete `latest.json`/`training_plan.json` pair are skipped. Batch 512 is intentionally
required by this primary pipeline because BatchTopK is defined over the actual microbatch.

The probe phase uses official [`sae-probes==0.4.*`](https://github.com/sae-probes/sae-probes), Pythia's
`blocks.16.hook_resid_post`, the `normal` setting, L1 logistic probes, mean-activation
normalization, probe seed 42, and `k=[1,2,4,8,16,32,64,128,256,512]`. Before the first probe it
checks the Hugging Face extraction hook against TransformerLens on identical token IDs. A raw
residual dense logistic probe is run once as a separate reference ceiling. All five Matryoshka
prefixes are also probed as supplementary results.

Probe aggregation refuses to produce a formal report unless every run has the identical complete
task×k set. It writes `comparison/index.html`, `summary.md`, JSON/CSV summaries, all-k curves,
paired per-task deltas, a L0/FVU/dead-feature table, and Matryoshka prefix results. Metrics are first
macro-averaged across tasks within each seed and then reported as mean ± standard deviation over
the three seeds.

Useful overrides are:

```bash
COMPARISON_ROOT=/scratch/lejepa-comparison \
PROBE_CACHE=/scratch/sae-probes-cache \
PILOT_STEPS=20000 \
  bash scripts/run_comparison.sh all
```

### Start with a 10k-step probing pilot on a single RTX 4090

Do not use `run_comparison.sh all` for this pilot: it launches the full training matrix.
Keep the trained base/rate=100/rate=200 checkpoints and train BatchTopK separately. For the
rate=200 run whose validation global active fraction is `0.037643183`, an actual-L0-matched
BatchTopK uses `baseline.k=617` (`16384 * 0.037643183`), not the nominal 5% target's 819.
Check the final `threshold_calibration.json` and report the measured L0 as well.

Install the optional probe dependencies in your environment (or a clone of the training
environment if you want to leave its packages untouched):

```bash
git pull --ff-only
python -m pip install -e '.[probes]'
python -m pip check
```

The probe extra pins `sae-probes=0.4.0`, `transformer-lens=2.15.4`, `sae-lens=6.5.3`, and
`scikit-learn=1.6.1` for the PyTorch 2.5.1 runtime and a reproducible probe API. Training does
not require this extra. Keep the installed CUDA build of PyTorch 2.5.1 on the remote GPU host.

First run the real official evaluator for just **one task at k=1,16** on both checkpoints:

```bash
# Replace the first path with your actual rate=200 training directory.
R200="runs/your-rate200-run"
BT_DIR="runs/pilot-comparison/seed42/batch-topk-k617-10k"

bash scripts/run_probe_pilot.sh smoke "$R200" "$BT_DIR"
```

This does not train any models. It requires `config.resolved.yaml` and
`checkpoint-00010000.pt` in each directory. `smoke` selects the first official task in sorted
order and keeps that task's standard train/test split; it is not a benchmark score on all
tasks. Only after both smoke tests succeed, run all official tasks at the same two k values:

```bash
bash scripts/run_probe_pilot.sh probe "$R200" "$BT_DIR"
# Optional additional checkpoints, evaluated with the identical task set and probe seed:
bash scripts/run_probe_pilot.sh probe "$BASE_DIR" "$RATE100_DIR"
```

The entry point preserves `normal`, L1, mean-activation normalization and probe seed 42.
The proposed model encodes the complete unmasked global input; BatchTopK/Matryoshka use
their checkpoint's calibrated pointwise threshold, never a probe-batch-dependent TopK.
The adapter matches training's encoder autocast precision and returns float32 probe features.

Both HF and TransformerLens in hook parity use explicit precision. TransformerLens uses
`from_pretrained_no_processing`: weight centering must not change the residual coordinates.
HF is released before TL is loaded, and the LLM is released before the SAE is moved onto the
GPU. CUDA defaults to bfloat16 (CPU to float32), and activation generation defaults to batch 1
with context length 1024. To try a larger activation batch or relocate the shared cache:

```bash
ACTIVATION_BATCH_SIZE=2 PROBE_CACHE=/scratch/sae-probes \
  bash scripts/run_probe_pilot.sh smoke "$R200" "$BT_DIR"
```

Equivalent single-checkpoint CLI (use `--datasets TASK_NAME` to choose a specific task):

```bash
lejepa-probe \
  --config "$R200/config.resolved.yaml" \
  --checkpoint "$R200/checkpoint-00010000.pt" \
  --results-path "$R200/probe-smoke-k1-k16" \
  --model-cache-path data/sae-probes/pythia-6.9b-layer16 \
  --llm-precision auto \
  --activation-batch-size 1 \
  --max-seq-len 1024 \
  --smoke-test
```

Each output contains `hook_parity.json`, `probe_manifest.json`, the official per-task raw JSON,
and (only after all requested task/k pairs succeed) `probe_summary.json` with macro F1, AUROC,
and accuracy per k (`f1`, `auroc`, `accuracy`). F1 is the task-macro average of the official
per-task class-weighted `test_f1`, not a replacement metric. Smoke outputs go to
`probe-smoke-k1-k16`; pilot outputs go to
`probes-normal-k1-k16`. Rerun the same command to resume completed tasks. Different checkpoints,
task sets, k sets, or precision/config changes require a new results directory; the official
evaluator otherwise skips an existing task file even when k changes.

Activation caches are namespaced by no-processing mode, precision, context length and package
versions, and accompanied by a provenance manifest. Legacy unversioned caches are not silently
reused. Cache generation is completed with an explicitly configured small-batch LLM before the
official evaluator can invoke its default float32 loader. All runs should use the same settings
and task list in `probe_manifest.json`; compare summaries only when those task lists agree.
LLM outputs are stored as float32 without changing their computed bfloat16 values, so the
optional raw-residual logistic reference can also pass them to sklearn/NumPy.

The local test suite covers CPU fixtures, optional real-library probe integration, precision
handling and loader contracts. It does not establish 6.9B/4090 hook parity or peak memory:
the remote smoke run is the required hardware preflight. A parity failure stops evaluation;
do not use `--skip-parity` to hide a mismatch. One seed and 5.12M token presentations are a
directional pilot, not the final 100M-token/multi-seed comparison.

If BF16 elementwise parity fails, the preflight now records the actual activation scale,
relative RMSE and maximum per-token relative L2 error in `hook_parity.json`, including on
failure. HF GPT-NeoX sums `(MLP + attention) + residual`, whereas TransformerLens sums
`(residual + attention) + MLP`; these need not agree in BF16. For example, adding two 8s
to 2048 in the two orders can differ by 16. Absolute maximum error alone cannot distinguish
this from a wrong hook.

The preflight therefore checks a failing BF16 comparison again using FP32 arithmetic on
the same BF16-rounded checkpoint weights. CPU weights remain in their original precision;
only one embedding/block at a time is promoted onto the execution device, TF32 is disabled,
and the forward stops after block 16. This avoids placing the entire 6.9B model in FP32 on
the 4090. It needs host RAM for one BF16 model plus loading overhead, and the first failing
BF16 check takes extra checkpoint loading/transfer time. The original elementwise tolerances
are unchanged for the FP32 check. A genuine FP32 mismatch still stops evaluation.

Even a correct FP32 hook mapping does not establish that BF16 errors are harmless to every
sparse encoder. A maximum per-token relative L2 discrepancy above 5% is rejected, not rescued
by the reference. This is a safety bound, not a quality guarantee. A successful fallback is
explicitly labeled `verification: float32_reference_on_bf16_weights`, with
`elementwise_allclose: false` and both precision diagnostics retained. Probe caches still use
the originally requested LLM precision; this fallback does not switch the experiment to FP32.
After updating, retry the same smoke command. If it still fails, inspect/share
`RUN_DIR/probe-smoke-k1-k16/hook_parity.json` instead of bypassing the preflight.

## 4. Evaluate and visualize

```bash
lejepa-evaluate \
  --config runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d16384-l0-0.009765625-axis512/config.resolved.yaml \
  --checkpoint runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d16384-l0-0.009765625-axis512/checkpoint-00010000.pt \
  --max-tokens 10000 \
  --top-k 20 \
  --output-dir runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d16384-l0-0.009765625-axis512/evaluation
```

Open `evaluation/index.html` first. Evaluation writes:

- `index.html`: metric cards, training curves, collapse status, feature histograms, and searchable top activations
- `summary.md`: compact core-metric table, training curves, and the 20 highest-variance features
- `training_curves.svg`: train/validation Active fraction, Global-local MSE, random/axis RDMReg, balanced global/local RDMReg contributions, feature standard deviation, gate-transition rates, and sparsity-gap decomposition over steps
- `training_history.csv`: the scalar history used to draw the training curves
- `feature_diagnostics.svg`: active-rate, standard-deviation, and maximum-activation distributions
- `feature_metrics.csv`: per-feature active rate, mean, standard deviation, and maximum
- `metrics.json`: machine-readable aggregate metrics
- `top_tokens.jsonl`: all requested top decoded token examples for every feature

The proposed model reports active fraction, dead-feature fraction, feature variance,
global-local MSE, feature-support Jaccard, and global/local gate-transition rates. Optionally pass `--concept-labels labels.json`, where
the JSON maps `document_id` to a concept label, to add merging/splitting proxies.

Evaluation automatically reads `metrics.jsonl` from the `train.output_dir` stored in the resolved
config. If the history was moved, pass `--training-metrics /path/to/metrics.jsonl`. Missing inferred
history does not prevent evaluation; the report explains how to attach it.

## 5. Local gradient intervention

The proposed model has no decoder vector, so interventions are explicitly sample-dependent:

```bash
lejepa-intervene \
  --config runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d16384-l0-0.009765625-axis512/config.resolved.yaml \
  --checkpoint runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-d16384-l0-0.009765625-axis512/checkpoint-00010000.pt \
  --token-index 42 \
  --feature-index 123 \
  --alpha 5 \
  --output runs/interventions/feature-123-token-42.pt
```

## Tests

```bash
pytest
ruff check .
```

Tests cover exact coordinate masking, pre-bias filling, inverted-mask scaling, vectorized
five-view encoding and RDMReg, paper target sampling, training/checkpoints, data isolation, and all
evaluation report artifacts.
