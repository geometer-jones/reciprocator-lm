import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.complex64, torch.complex128):
        return dtype
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def complex_from_real(real_tensor: Tensor) -> Tensor:
    real_dtype = real_tensor.dtype
    if real_dtype not in (torch.float32, torch.float64):
        real_tensor = real_tensor.float()
    imag = torch.zeros_like(real_tensor)
    return torch.complex(real_tensor, imag).to(dtype=complex_dtype_for(real_tensor.dtype))


def complex_dropout(hidden: Tensor, p: float, training: bool) -> Tensor:
    if p <= 0.0 or not training:
        return hidden
    mask = F.dropout(torch.ones_like(hidden.real), p=p, training=training)
    return hidden * mask


def complex_modulation_factor(
    *,
    log_scale: Tensor,
    phase: Tensor,
    reference: Tensor,
) -> Tensor:
    scale = torch.exp(log_scale.to(dtype=reference.real.dtype, device=reference.device))
    phase = phase.to(dtype=reference.real.dtype, device=reference.device)
    return torch.polar(scale, phase).to(dtype=reference.dtype)


def complex_rope_frequencies(
    width: int,
    *,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if width <= 0:
        raise ValueError("width must be positive.")
    if base <= 1.0:
        raise ValueError("base must be greater than 1.")
    real_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    indices = torch.arange(width, device=device, dtype=real_dtype)
    return torch.exp(-math.log(base) * indices / width)


def complex_rope_factor(
    *,
    positions: Tensor,
    inverse_frequencies: Tensor,
    reference: Tensor,
) -> Tensor:
    if reference.ndim < 2:
        raise ValueError("reference must have at least sequence and feature dimensions.")
    if positions.ndim != 1:
        raise ValueError("positions must be a 1D tensor.")
    if positions.numel() != reference.size(-2):
        raise ValueError("positions length must match the reference sequence dimension.")

    inverse_frequencies = inverse_frequencies.to(dtype=reference.real.dtype, device=reference.device)
    if inverse_frequencies.ndim != 1:
        raise ValueError("inverse_frequencies must be 1D.")
    if inverse_frequencies.numel() != reference.size(-1):
        raise ValueError("inverse_frequencies length must match the reference feature dimension.")

    angles = positions.to(dtype=inverse_frequencies.dtype, device=reference.device).unsqueeze(-1)
    angles = angles * inverse_frequencies.unsqueeze(0)
    broadcast_shape = (1,) * (reference.ndim - 2) + angles.shape
    phase = angles.view(*broadcast_shape)
    return complex_modulation_factor(
        log_scale=torch.zeros_like(phase),
        phase=phase,
        reference=reference,
    )


def apply_complex_rope(
    hidden: Tensor,
    positions: Tensor,
    *,
    inverse_frequencies: Tensor,
) -> Tensor:
    if not torch.is_complex(hidden):
        hidden = complex_from_real(hidden)
    return hidden * complex_rope_factor(
        positions=positions,
        inverse_frequencies=inverse_frequencies,
        reference=hidden,
    )


def complex_readout_features(hidden: Tensor, mode: str) -> Tensor:
    if not torch.is_complex(hidden):
        hidden = complex_from_real(hidden)
    if mode == "magnitude":
        return hidden.abs()
    if mode != "phase_aware":
        raise ValueError(f"Unsupported complex readout mode '{mode}'.")

    # Replace discontinuous gauge fixing with continuous bilinear invariants.
    # The pooled reference rotates with the same global U(1) phase as `hidden`,
    # so hidden * pooled.conj() is phase-invariant while preserving relative-phase
    # information for the real-valued lm_head.
    pooled = hidden.mean(dim=-1, keepdim=True)
    invariant_cross = hidden * pooled.conj()
    return torch.cat([invariant_cross.real, invariant_cross.imag, hidden.abs()], dim=-1)


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)

    def forward(self, hidden: Tensor) -> Tensor:
        if not torch.is_complex(hidden):
            hidden = complex_from_real(hidden)

        weight_real = self.weight_real.to(dtype=hidden.real.dtype, device=hidden.device)
        weight_imag = self.weight_imag.to(dtype=hidden.real.dtype, device=hidden.device)
        real = F.linear(hidden.real, weight_real) - F.linear(hidden.imag, weight_imag)
        imag = F.linear(hidden.real, weight_imag) + F.linear(hidden.imag, weight_real)
        return torch.complex(real, imag)


class ComplexLayerNorm(nn.Module):
    def __init__(self, model_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(model_dim))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        if not torch.is_complex(hidden):
            hidden = complex_from_real(hidden)
        centered = hidden - hidden.mean(dim=-1, keepdim=True)
        scale = centered.abs().square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        weight = self.weight.to(dtype=centered.real.dtype, device=centered.device)
        return centered / scale * weight


class ModReLU(nn.Module):
    def __init__(self, model_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        magnitude = hidden.abs()
        bias = self.bias.to(dtype=magnitude.dtype, device=magnitude.device)
        scale = F.relu(magnitude + bias) / magnitude.clamp_min(self.eps)
        return hidden * scale


class ComplexFeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = ComplexLinear(input_dim, hidden_dim)
        self.activation = ModReLU(hidden_dim)
        self.output_proj = ComplexLinear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.input_proj(hidden)
        hidden = self.activation(hidden)
        hidden = complex_dropout(hidden, self.dropout, self.training)
        hidden = self.output_proj(hidden)
        hidden = complex_dropout(hidden, self.dropout, self.training)
        return hidden


class ComplexMultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout

        self.q_proj = ComplexLinear(model_dim, model_dim)
        self.k_proj = ComplexLinear(model_dim, model_dim)
        self.v_proj = ComplexLinear(model_dim, model_dim)
        self.out_proj = ComplexLinear(model_dim, model_dim)

    def forward(self, hidden: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = hidden.shape
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [batch, seq] matching hidden.")

        query = self.q_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Hermitian inner product — complex pairwise similarities
        hermitian = torch.matmul(query, key.conj().transpose(-2, -1))

        # Magnitude for softmax (selection), phase for rotational modulation
        magnitudes = hermitian.abs() / math.sqrt(self.head_dim)
        phase_offsets = hermitian / hermitian.abs().clamp_min(1e-8)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool),
            diagonal=1,
        )
        magnitudes = magnitudes.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))

        if attention_mask is not None:
            invalid_keys = ~attention_mask.bool()
            magnitudes = magnitudes.masked_fill(invalid_keys[:, None, None, :], float("-inf"))

        # Softmax on magnitudes — real-valued selection weights
        weights = F.softmax(magnitudes, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        # Selection (magnitude) × rotation (phase) → complex attention weights
        rotated_weights = weights * phase_offsets
        attn_output = torch.matmul(rotated_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        return self.out_proj(attn_output)


class ComplexTransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ffw_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = ComplexLayerNorm(model_dim)
        self.attn = ComplexMultiheadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = dropout
        self.norm2 = ComplexLayerNorm(model_dim)
        ffw_dim = model_dim * ffw_multiplier
        self.ffw = ComplexFeedForward(
            input_dim=model_dim,
            hidden_dim=ffw_dim,
            output_dim=model_dim,
            dropout=dropout,
        )

    def forward(self, hidden: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        attn_output = self.attn(self.norm1(hidden), attention_mask=attention_mask)
        hidden = hidden + complex_dropout(attn_output, self.dropout, self.training)
        hidden = hidden + self.ffw(self.norm2(hidden))
        return hidden


__all__ = [
    "ComplexFeedForward",
    "ComplexLayerNorm",
    "ComplexLinear",
    "ComplexMultiheadAttention",
    "ComplexTransformerBlock",
    "ModReLU",
    "apply_complex_rope",
    "complex_dropout",
    "complex_from_real",
    "complex_modulation_factor",
    "complex_rope_factor",
    "complex_rope_frequencies",
    "complex_readout_features",
]
