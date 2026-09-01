# LeJEPA-SAE

Reconstruction-free sparse feature discovery from individual LLM residual activations. The
proposed model is the single-token dimension-mask JEPA; this repository contains no multi-token
Transformer/CLS model.

## Proposed model

For one frozen residual activation `h_t ∈ R^4096`, the global view is complete and each of four
local views retains an independently sampled, exact half of its coordinates. The shared encoder is:

```text
h_t → subtract learned pre-bias → exact coordinate mask → inverted-mask scaling
    → Linear(4096, 8192) → ReLU → z
```

Missing coordinates are therefore filled with the learned pre-bias and become zero after
centering. The global and four local views are encoded in one batched Linear call. There is no
Transformer, CLS token, decoder, target encoder, or stop-gradient.

The objective is:

```text
L = 25 · mean_v MSE(z_global, z_local_v)
  + 125 · (L_random-RDMReg + 1.0 · L_axis-RDMReg)
```

RDMReg follows Rectified LpJEPA: every view is matched to an independent
`ReLU(Laplace(0, 1/√2))` target with 8192 shared random unit projections. In addition, every
optimizer step samples 512 coordinate axes without replacement and directly matches the selected
feature marginals. Axis indices are shared by all five views, and the same view-specific target is
used by both losses. Coordinate values are gathered directly; no one-hot matrix is materialized.
Projection matmul is mixed precision; sorting and loss reduction are float32.

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

The default extraction reads the local raw Pile shards, sends each document to Pythia in chunks of
up to 1024 tokens, and stops at 100M tokenized source tokens:

```bash
bash scripts/extract_the_pile.sh
```

The default input glob is `/datasets/the-pile/train/*.jsonl.zst`. Override it and the token budget
without editing the script:

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

The pilot preset uses batch 512, no gradient accumulation, 10,000 steps, and a fresh output at
`runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-axis512`. The output is deliberately separate from
pre-axis checkpoints and starts from a fresh initialization. Do not resume a collapsed checkpoint:
axis loss discourages feature death while gradients still cross ReLU, but cannot guarantee revival
once every sample for a feature is in ReLU's negative region.

If batch 512 is out of memory, preserve the effective batch in this order:

```bash
BATCH_SIZE=256 GRADIENT_ACCUMULATION_STEPS=2 bash scripts/run_proposed.sh
BATCH_SIZE=128 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/run_proposed.sh
```

`MAX_STEPS` and `EVAL_BATCHES` are also environment overrides. Training logs core losses every
batch—including separate `random_distribution` and `axis_distribution` values—collapse
diagnostics at log steps, interval throughput, and CUDA peak memory. Axis ablations can use, for
example, `AXIS_PROJECTIONS=256 AXIS_WEIGHT=2.0 bash scripts/run_proposed.sh`; the default output
directory includes the selected axis count.

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
  --config runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-axis512/config.resolved.yaml \
  --checkpoint runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-axis512/checkpoint-00010000.pt \
  --max-tokens 10000 \
  --top-k 20 \
  --output-dir runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-axis512/evaluation
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
  --config runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-axis512/config.resolved.yaml \
  --checkpoint runs/the-pile/pythia-6.9b-layer16-ctx1024-100m/proposed-axis512/checkpoint-00010000.pt \
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
