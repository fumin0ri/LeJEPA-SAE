# LeJEPA-SAE

Reconstruction-free sparse feature discovery from individual LLM residual activations. The
proposed model is the single-token dimension-mask JEPA; this repository contains no multi-token
Transformer/CLS model.

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
- `training_curves.svg`: train/validation Active fraction, Global-local MSE, random/axis RDMReg, balanced global/local RDMReg contributions, and feature standard deviation over steps
- `training_history.csv`: the scalar history used to draw the training curves
- `feature_diagnostics.svg`: active-rate, standard-deviation, and maximum-activation distributions
- `feature_metrics.csv`: per-feature active rate, mean, standard deviation, and maximum
- `metrics.json`: machine-readable aggregate metrics
- `top_tokens.jsonl`: all requested top decoded token examples for every feature

The proposed model reports active fraction, dead-feature fraction, feature variance,
global-local MSE, and feature-support Jaccard. Optionally pass `--concept-labels labels.json`, where
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
