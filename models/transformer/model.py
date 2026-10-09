from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


class PatchEmbedding(nn.Module):
    """Patch embedding for Image inputs in Vision Transformer."""

    def __init__(self, image_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x_patches = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, self.patch_dim)
        return self.proj(x_patches)


class TextEmbedding(nn.Module):
    """Token + Positional embedding for Text inputs in Transformer."""

    def __init__(self, vocab_size: int = 1000, max_seq_len: int = 64, embed_dim: int = 64) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(b, -1)
        return self.tok_embed(x) + self.pos_embed(pos)


class AudioEmbedding(nn.Module):
    """Convolutional feature embedding for Audio inputs (Spectrogram 2D or Waveform 1D)."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 64, is_spectrogram: bool = True) -> None:
        super().__init__()
        self.is_spectrogram = is_spectrogram
        if is_spectrogram:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, embed_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        if self.is_spectrogram:
            # (B, D, H, W) -> (B, H*W, D)
            b, d, h, w = feat.shape
            return feat.permute(0, 2, 3, 1).reshape(b, h * w, d)
        else:
            # (B, D, T) -> (B, T, D)
            return feat.permute(0, 2, 1)


class TransformerModel(nn.Module):
    """Unified Transformer architecture supporting Vision (image), Text, Audio, and Multimodal inputs.

    Unifies feature embeddings across single and multi-modal inputs, passes sequence tokens
    through PyTorch TransformerEncoder, pools features, and projects to output classes.
    """

    def __init__(
        self,
        modality: str = "image",
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 128,
        dropout: float = 0.1,
        # Modality specific settings
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        vocab_size: int = 1000,
        max_seq_len: int = 64,
        audio_channels: int = 1,
        is_spectrogram: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.modality = modality.lower()
        self.embed_dim = embed_dim

        # Support common parameter aliases
        image_size = kwargs.get("img_size", image_size)
        depth = kwargs.get("num_layers", depth)
        if "freq_bins" in kwargs:
            in_channels = kwargs["freq_bins"]
        if "audio_bins" in kwargs:
            in_channels = kwargs["audio_bins"]

        # Modular Embedders
        self.patch_embedder = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.text_embedder = TextEmbedding(vocab_size, max_seq_len, embed_dim)
        self.audio_embedder = AudioEmbedding(audio_channels, embed_dim, is_spectrogram)

        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def _embed_batch(self, x: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if self.modality == "image":
            img = x["image"] if isinstance(x, dict) else x
            return self.patch_embedder(img)

        elif self.modality == "text":
            txt = x["text"] if isinstance(x, dict) else x
            return self.text_embedder(txt)

        elif self.modality == "audio":
            aud = x["audio"] if isinstance(x, dict) else x
            return self.audio_embedder(aud)

        elif self.modality == "multimodal":
            if not isinstance(x, dict):
                raise ValueError("Multimodal inputs must be passed as a dictionary containing 'image', 'text', or 'audio'.")

            tokens_list = []
            if "image" in x:
                tokens_list.append(self.patch_embedder(x["image"]))
            if "text" in x:
                tokens_list.append(self.text_embedder(x["text"]))
            if "audio" in x:
                tokens_list.append(self.audio_embedder(x["audio"]))

            if not tokens_list:
                raise ValueError("Multimodal batch must contain at least one of 'image', 'text', or 'audio'.")

            return torch.cat(tokens_list, dim=1)

        else:
            raise ValueError(f"Unknown modality '{self.modality}'. Must be 'image', 'text', 'audio', or 'multimodal'.")

    def forward(self, x: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        tokens = self._embed_batch(x)
        out_tokens = self.transformer(tokens)
        pooled = out_tokens.mean(dim=1)  # Mean pooling across sequence dimension
        return self.head(pooled)
