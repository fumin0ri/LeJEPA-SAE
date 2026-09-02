# LeJEPA-SAE

Reconstruction-free sparse feature discovery from individual LLM residual activations. The
proposed model is the single-token dimension-mask JEPA; this repository contains no multi-token
Transformer/CLS model.

## Proposed model

For one frozen residual activation `h_t ∈ R^4096`, the global view is complete and each of four
local views retains an independently sampled, exact half of its coordinates. The shared encoder is:

```text
h_t → subtract learned pre-bias → exact coordinate mask → inverted-mask scaling
    → Linear(4096, 16384) → ReLU → z
```

Missing coordinates are therefore filled with the learned pre-bias and become zero after
centering. The global and four local views are encoded in one batched Linear call. There is no
Transformer, CLS token, decoder, target encoder, or stop-gradient.

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

The optional single-token reconstruction baselines remain available:

```bash
bash scripts/run_comparison.sh
```

| Type | Input | Objective |
|---|---|---|
| `proposed` | complete global + four half-coordinate local views | invariance + RDMReg |
| `standard_sae` | complete token | reconstruction + L1 |
| `dimension_denoising_sae` | four half-coordinate views | full-token reconstruction + L1 |

## 3. Evaluate and visualize

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
- `training_curves.svg`: train/validation Active fraction, Global-local MSE, random/axis RDMReg, and feature standard deviation over steps
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

## 4. Local gradient intervention

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
