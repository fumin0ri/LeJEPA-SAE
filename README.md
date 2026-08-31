# LeJEPA-SAE

Reconstruction-free sparse feature discovery from frozen LLM residual streams. The proposed
model transfers LeVJEPA-style token dropping and CLS invariance to 10-token LLM activation
windows, then replaces Gaussian distribution matching with a rectified-Gaussian RDMReg target.

The implementation is intentionally split into two GPU phases so Pythia-6.9B and the trainable
model never need to occupy a 24 GB RTX 4090 at the same time.

## Method

For a frozen block output `H ∈ R^(10×4096)`, the global view contains all ten residual vectors.
Each of four local views independently retains `k=3` tokens. Dropped tokens are actually removed
before the encoder; they are not zero-masked. Each retained token keeps its original position in
the 0–9 window.

All views share:

```text
residuals → LayerNorm → 4096→256 projection → original position embedding
          → prepend CLS → 3-layer/4-head Transformer
          → CLS → 256→8192 sparse head → ReLU → z
```

In causal mode, residual tokens see only prior residual tokens and cannot see CLS, while CLS sees
the complete retained span. Both global and local branches receive gradients. There is no target
encoder, stop-gradient, decoder, token ID input, or reconstruction loss in the proposed model.

The objective is:

```text
L = mean_v MSE(z_global, z_local_v) + λ · mean_views RDMReg(z_view)
```

RDMReg samples an equally shaped target from `ReLU(N(μ, σ²))`, choosing `μ/σ` so its expected
active fraction is exactly the configured value (10% by default). It compares empirical sorted
projections along 32 random unit directions, i.e. a sliced 2-Wasserstein loss.

## Environment

Target: Linux over SSH, Python 3.10+, CUDA 12.1, PyTorch 2.5.1, one RTX 4090.

```bash
git clone https://github.com/fumin0ri/LeJEPA-SAE.git
cd LeJEPA-SAE
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[dev]'
```

Check the runtime before a long extraction:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

## 1. Extract the exact residual hook point

The extractor loads `AutoModel` (not the LM head), freezes it, and hooks the zero-based Pythia
block output. The default corpus is the standard (non-deduplicated) The Pile, matching
`EleutherAI/pythia-6.9b` rather than the `-deduped` model family.

The practical single-workstation preset uses EleutherAI's official 5M-row random sample of the
Pythia-tokenized standard Pile. It consumes the existing `Tokens` IDs directly, avoiding a
tokenization mismatch:

```bash
bash scripts/extract_the_pile.sh
```

Each sampled source sequence contains 64 Pythia tokens. The preset extracts 10,000 sequences
(640k tokens and about 4.9 GiB of bf16 activations). Set `MAX_SEQUENCES=100000` to scale it to
6.4M tokens and about 49 GiB. The sampled dataset calls the standard, non-deduplicated variant `duped`:
`EleutherAI/pile-duped-pythia-random-sampled`.
The extraction script pins its current dataset commit instead of following a mutable `main`.

Each sampled 64-token source sequence is hash-assigned before segmentation, so its
adjacent windows cannot cross splits. The random-sample table does not expose original Pile
document IDs, however, so strict original-document isolation cannot be proven in this mode. For
that requirement, point the extractor at locally obtained raw Pile JSONL shards, where one row is
one original document:

```bash
lejepa-extract \
  --dataset json \
  --data-files '/datasets/the-pile/train/*.jsonl.zst' \
  --source-split train \
  --text-column text \
  --model EleutherAI/pythia-6.9b \
  --layer 16 \
  --context-length 512 \
  --window-size 10 \
  --dtype bfloat16 \
  --output-dir data/the-pile/pythia-6.9b/layer-16
```

Raw-document mode assigns each row to train/validation/test by a stable content hash before it is
segmented. Consequently adjacent windows and all context segments from one document remain in one
split. Review the licenses and access terms of every Pile component before obtaining or using the
raw corpus.

Use the same `block_output:16` manifest for every method. The activation shards contain residuals
and token IDs; token IDs are retained only for decoding evaluation examples and never enter an
interpretation model. Long documents are evaluated in independent 512-token causal segments.

For a smoke extraction, add `--max-documents 100`. Storage is approximately 8 GB per 1 million
tokens at width 4096 in bf16. Change `--shard-tokens` to trade file size for loader cache pressure.

## 2. Train the proposed model

```bash
lejepa-train --config configs/pythia-6.9b-layer16.yaml
```

The default microbatch is 64 windows with eight accumulation steps (effective batch 512). On an
otherwise occupied 4090, lower the microbatch without changing the effective batch:

```bash
lejepa-train --config configs/pythia-6.9b-layer16.yaml \
  --set train.batch_size=32 \
  --set train.gradient_accumulation_steps=16
```

Training writes the fully resolved config, append-only JSONL metrics, periodic restartable
checkpoints, and a `latest.json` pointer. A healthy run should show falling invariance loss,
nonzero mean feature standard deviation, and active fraction moving toward 0.10.

## 3. Retention sweep and required baselines

```bash
bash scripts/run_retention_sweep.sh
bash scripts/run_baselines.sh
```

The retention sweep runs `k ∈ {1,2,3,5,8,10}`. Baseline behavior is selected with `model.type`:

| Type | Unit | Objective | Dropping | Distribution |
|---|---:|---|---:|---|
| `standard_sae` | 1 token | reconstruction + L1 | no | none |
| `window_autoencoder` | 10 tokens | reconstruction + L1 | no | none |
| `sparse_jepa_full_view` | 10 tokens | invariance + RDMReg | 10/10 | rectified Gaussian |
| `jepa_sigreg` | 10 tokens | invariance + distribution | yes | Gaussian |
| `proposed` | 10 tokens | invariance + RDMReg | yes | rectified Gaussian |

The window autoencoder uses the same small Transformer encoder and a capacity-controlled
factorized Transformer decoder, avoiding a confounding 335M-parameter dense 8192→(10×4096)
decoder.

Bidirectional and sparsity ablations require only scalar overrides:

```bash
lejepa-train --config configs/pythia-6.9b-layer16.yaml \
  --set model.attention=bidirectional \
  --set loss.target_active_fraction=0.05 \
  --set train.output_dir=runs/bidirectional-active05
```

## 4. Evaluate semantics and drop robustness

```bash
lejepa-evaluate \
  --config runs/the-pile/pythia-6.9b-layer16/proposed-k3/config.resolved.yaml \
  --checkpoint runs/the-pile/pythia-6.9b-layer16/proposed-k3/checkpoint-00100000.pt \
  --max-windows 10000 \
  --top-k 20 \
  --output-dir runs/the-pile/pythia-6.9b-layer16/proposed-k3/evaluation
```

Outputs include feature sparsity/deadness, global-local MSE, exact-support Jaccard, and a JSONL
file of top activating decoded spans for monosemanticity review. Optionally pass
`--concept-labels labels.json`, where the JSON maps `document_id` to a concept label. This adds
dominant-concept consistency, concepts-per-feature (merging proxy), and features-per-concept
(splitting proxy).

## 5. Local gradient intervention

The proposed model has no decoder vector. The supplied intervention is therefore explicitly
sample-dependent and should not be presented as SAE-style global controllability:

```bash
lejepa-intervene \
  --config runs/the-pile/pythia-6.9b-layer16/proposed-k3/config.resolved.yaml \
  --checkpoint runs/the-pile/pythia-6.9b-layer16/proposed-k3/checkpoint-00100000.pt \
  --window-index 42 \
  --feature-index 123 \
  --alpha 5 \
  --token-position 7 \
  --output runs/interventions/feature-123-window-42.pt
```

The output stores the original residuals, normalized local gradient direction, modified
residuals, token IDs, and metadata. Inject `modified_residuals` at the same block-output hook when
measuring downstream causal effects.

## Tests

```bash
pytest
ruff check .
```

Tests cover true token removal and original-position retention, the asymmetric CLS attention
mask, target sparsity calibration, document split isolation, shard window boundaries, every model
type, and a backward training step.
