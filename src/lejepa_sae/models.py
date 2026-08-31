from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import ExperimentConfig, ModelConfig


@dataclass
class ModelOutput:
    features: torch.Tensor
    reconstruction: torch.Tensor | None = None


class SharedWindowEncoder(nn.Module):
    def __init__(self, config: ModelConfig, max_window_size: int) -> None:
        super().__init__()
        self.attention = config.attention
        self.max_window_size = max_window_size
        self.input_norm = nn.LayerNorm(config.d_llm)
        self.input_projection = nn.Linear(config.d_llm, config.d_encoder)
        self.position_embedding = nn.Embedding(max_window_size, config.d_encoder)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_encoder))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_encoder,
            nhead=config.num_heads,
            dim_feedforward=config.d_encoder * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_encoder),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    @staticmethod
    def attention_mask(
        length: int, topology: str, device: torch.device | None = None
    ) -> torch.Tensor:
        """Mask with a globally-reading CLS and no token-to-CLS edges."""
        mask = torch.zeros(length + 1, length + 1, dtype=torch.bool, device=device)
        mask[1:, 0] = True
        if topology == "causal":
            token_future = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
            )
            mask[1:, 1:] = token_future
        elif topology != "bidirectional":
            raise ValueError(f"Unknown attention topology: {topology}")
        return mask

    def forward(self, residuals: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if residuals.ndim != 3:
            raise ValueError("residuals must have shape [batch, tokens, d_llm]")
        batch, length, width = residuals.shape
        if width != self.input_projection.in_features:
            expected = self.input_projection.in_features
            raise ValueError(f"Expected residual width {expected}, got {width}")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        if positions.shape != (batch, length):
            raise ValueError("positions must have shape [tokens] or [batch, tokens]")
        if positions.numel() and (positions.min() < 0 or positions.max() >= self.max_window_size):
            raise ValueError("positions are outside the configured window")

        tokens = self.input_projection(self.input_norm(residuals))
        tokens = tokens + self.position_embedding(positions)
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat((cls, tokens), dim=1)
        mask = self.attention_mask(length, self.attention, residuals.device)
        encoded = self.transformer(sequence, mask=mask)
        return encoded[:, 0]


class SparseJEPA(nn.Module):
    def __init__(self, config: ModelConfig, window_size: int) -> None:
        super().__init__()
        self.encoder = SharedWindowEncoder(config, window_size)
        self.sparse_head = nn.Linear(config.d_encoder, config.feature_dim)
        self.dense_output = config.type == "jepa_sigreg"
        nn.init.zeros_(self.sparse_head.bias)
        if config.sparse_bias:
            nn.init.constant_(self.sparse_head.bias, config.sparse_bias)

    def forward(self, residuals: torch.Tensor, positions: torch.Tensor) -> ModelOutput:
        logits = self.sparse_head(self.encoder(residuals, positions))
        features = logits if self.dense_output else logits.relu()
        return ModelOutput(features=features)


class StandardSAE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(config.d_llm))
        self.encoder = nn.Linear(config.d_llm, config.feature_dim)
        self.decoder = nn.Linear(config.feature_dim, config.d_llm, bias=False)
        nn.init.zeros_(self.encoder.bias)

    def forward(
        self, residuals: torch.Tensor, positions: torch.Tensor | None = None
    ) -> ModelOutput:
        del positions
        centered = residuals - self.pre_bias
        features = self.encoder(centered).relu()
        reconstruction = self.decoder(features) + self.pre_bias
        return ModelOutput(features=features, reconstruction=reconstruction)


class WindowAutoencoder(nn.Module):
    """Capacity-controlled 10-token reconstruction baseline with sparse CLS features."""

    def __init__(self, config: ModelConfig, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.encoder = SharedWindowEncoder(config, window_size)
        self.sparse_head = nn.Linear(config.d_encoder, config.feature_dim)
        self.latent_projection = nn.Linear(config.feature_dim, config.d_encoder, bias=False)
        self.decoder_positions = nn.Parameter(torch.empty(1, window_size, config.d_encoder))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_encoder,
            nhead=config.num_heads,
            dim_feedforward=config.d_encoder * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_encoder),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(config.d_encoder, config.d_llm)
        nn.init.normal_(self.decoder_positions, std=0.02)
        nn.init.zeros_(self.sparse_head.bias)

    def forward(self, residuals: torch.Tensor, positions: torch.Tensor) -> ModelOutput:
        if residuals.shape[1] != self.window_size:
            raise ValueError("WindowAutoencoder requires the complete fixed-size window")
        cls = self.encoder(residuals, positions)
        features = self.sparse_head(cls).relu()
        seed = self.latent_projection(features).unsqueeze(1)
        decoded = self.decoder_transformer(seed + self.decoder_positions)
        reconstruction = self.output_projection(decoded)
        return ModelOutput(features=features, reconstruction=reconstruction)


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model.type == "standard_sae":
        return StandardSAE(config.model)
    if config.model.type == "window_autoencoder":
        return WindowAutoencoder(config.model, config.data.window_size)
    return SparseJEPA(config.model, config.data.window_size)
