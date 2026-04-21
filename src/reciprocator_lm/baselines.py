import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .complex import (
    ComplexLayerNorm,
    ComplexTransformerBlock,
    apply_complex_rope,
    complex_dropout,
    complex_modulation_factor,
    complex_rope_frequencies,
    complex_readout_features,
)


@dataclass(frozen=True)
class BaselineTransformerConfig:
    vocab_size: int
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffw_multiplier: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0
    complex_dtype: torch.dtype = torch.complex64
    readout_mode: str = "magnitude"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative.")
        if self.complex_dtype not in (torch.complex64, torch.complex128):
            raise ValueError("complex_dtype must be torch.complex64 or torch.complex128.")
        if self.readout_mode not in {"magnitude", "phase_aware"}:
            raise ValueError("readout_mode must be 'magnitude' or 'phase_aware'.")


@dataclass(frozen=True)
class SmallMambaConfig:
    vocab_size: int
    model_dim: int = 256
    num_layers: int = 4
    state_size: int = 16
    expand: int = 2
    conv_kernel: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        if self.state_size <= 0:
            raise ValueError("state_size must be positive.")
        if self.expand <= 0:
            raise ValueError("expand must be positive.")
        if self.conv_kernel <= 0:
            raise ValueError("conv_kernel must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative.")


def _causal_lm_loss(logits: Tensor, labels: Optional[Tensor]) -> Optional[Tensor]:
    if labels is None:
        return None
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, seq] matching logits.")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class RealMultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, hidden: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = hidden.shape
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [batch, seq] matching hidden.")

        query = self.q_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))

        if attention_mask is not None:
            invalid_keys = ~attention_mask.bool()
            scores = scores.masked_fill(invalid_keys[:, None, None, :], float("-inf"))

        attn_weights = scores.softmax(dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        return self.out_proj(attn_output)


class RealTransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ffw_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = RealMultiheadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        ffw_dim = model_dim * ffw_multiplier
        self.ffw = nn.Sequential(
            nn.Linear(model_dim, ffw_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffw_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.dropout = dropout

    def forward(self, hidden: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        attn_output = self.attn(self.norm1(hidden), attention_mask=attention_mask)
        hidden = hidden + F.dropout(attn_output, p=self.dropout, training=self.training)
        hidden = hidden + self.ffw(self.norm2(hidden))
        return hidden


class PlainTransformerLM(nn.Module):
    def __init__(self, config: BaselineTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dim)
        self.dropout = config.dropout
        self.blocks = nn.ModuleList(
            RealTransformerBlock(
                model_dim=config.model_dim,
                num_heads=config.num_heads,
                ffw_multiplier=config.ffw_multiplier,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq].")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len {self.config.max_seq_len}.")
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [batch, seq] matching input_ids.")

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        result: Dict[str, Any] = {"hidden": hidden, "logits": logits}
        loss = _causal_lm_loss(logits, labels)
        if loss is not None:
            result["loss"] = loss
        return result


class ComplexTransformerLM(nn.Module):
    def __init__(self, config: BaselineTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.dtype = config.complex_dtype
        self.token_log_scale = nn.Embedding(config.vocab_size, config.model_dim)
        self.register_buffer(
            "rope_inverse_frequencies",
            complex_rope_frequencies(config.model_dim, dtype=config.complex_dtype),
            persistent=False,
        )
        self.dropout = config.dropout
        self.blocks = nn.ModuleList(
            ComplexTransformerBlock(
                model_dim=config.model_dim,
                num_heads=config.num_heads,
                ffw_multiplier=config.ffw_multiplier,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = ComplexLayerNorm(config.model_dim)
        # Preserve the existing wide-head shape for phase-aware checkpoints while
        # routing it through continuous U(1)-invariant readout features.
        readout_dim = config.model_dim if config.readout_mode == "magnitude" else (3 * config.model_dim)
        self.lm_head = nn.Linear(readout_dim, config.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_log_scale.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _load_from_state_dict(self, state_dict: dict[str, Tensor], prefix: str, *args: object, **kwargs: object) -> None:
        # Backward compatibility for checkpoints saved before rotary position encoding.
        for legacy_key in (
            "token_phase.weight",
            "position_log_scale.weight",
            "position_phase.weight",
        ):
            state_dict.pop(prefix + legacy_key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq].")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len {self.config.max_seq_len}.")
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [batch, seq] matching input_ids.")

        positions = torch.arange(seq_len, device=input_ids.device)
        reference = torch.ones(batch_size, seq_len, self.config.model_dim, dtype=self.dtype, device=input_ids.device)
        hidden = complex_modulation_factor(
            log_scale=0.1 * torch.tanh(self.token_log_scale(input_ids)),
            phase=torch.zeros(
                batch_size,
                seq_len,
                self.config.model_dim,
                dtype=reference.real.dtype,
                device=input_ids.device,
            ),
            reference=reference,
        )
        hidden = apply_complex_rope(
            hidden,
            positions,
            inverse_frequencies=self.rope_inverse_frequencies,
        )
        hidden = complex_dropout(hidden, self.dropout, self.training)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(complex_readout_features(hidden, self.config.readout_mode))

        result: Dict[str, Any] = {"hidden": hidden, "logits": logits}
        loss = _causal_lm_loss(logits, labels)
        if loss is not None:
            result["loss"] = loss
        return result


class _SelectiveSSMMixer(nn.Module):
    """Minimal causal Mamba-style selective state-space mixer."""

    def __init__(
        self,
        model_dim: int,
        state_size: int,
        expand: int,
        conv_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.state_size = state_size
        self.inner_dim = model_dim * expand
        self.dropout = dropout
        self.in_proj = nn.Linear(model_dim, self.inner_dim * 2, bias=False)
        self.conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel,
            groups=self.inner_dim,
            padding=conv_kernel - 1,
        )
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.b_proj = nn.Linear(self.inner_dim, self.inner_dim * state_size, bias=False)
        self.c_proj = nn.Linear(self.inner_dim, self.inner_dim * state_size, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, model_dim, bias=False)
        initial_a = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.inner_dim, 1)
        self.a_log = nn.Parameter(initial_a.log())
        self.d = nn.Parameter(torch.ones(self.inner_dim))

    def forward(self, hidden: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden.shape
        projected = self.in_proj(hidden)
        u, gate = projected.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        u = F.silu(u)
        gate = torch.sigmoid(gate)

        delta = F.softplus(self.dt_proj(u))
        b = self.b_proj(u).view(batch_size, seq_len, self.inner_dim, self.state_size)
        c = self.c_proj(u).view(batch_size, seq_len, self.inner_dim, self.state_size)
        a = -torch.exp(self.a_log).unsqueeze(0)
        state = u.new_zeros(batch_size, self.inner_dim, self.state_size)
        outputs = []

        for step in range(seq_len):
            delta_t = delta[:, step, :].unsqueeze(-1)
            u_t = u[:, step, :].unsqueeze(-1)
            state = torch.exp(delta_t * a) * state + delta_t * b[:, step, :, :] * u_t
            y_t = (c[:, step, :, :] * state).sum(dim=-1) + self.d * u[:, step, :]
            outputs.append(y_t * gate[:, step, :])

        mixed = torch.stack(outputs, dim=1)
        return F.dropout(self.out_proj(mixed), p=self.dropout, training=self.training)


class SmallMambaBlock(nn.Module):
    def __init__(self, config: SmallMambaConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.model_dim)
        self.mixer = _SelectiveSSMMixer(
            model_dim=config.model_dim,
            state_size=config.state_size,
            expand=config.expand,
            conv_kernel=config.conv_kernel,
            dropout=config.dropout,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.mixer(self.norm(hidden))


class SmallMambaLM(nn.Module):
    def __init__(self, config: SmallMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList(SmallMambaBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq].")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds configured max_seq_len {self.config.max_seq_len}.")
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [batch, seq] matching input_ids.")

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        result: Dict[str, Any] = {"hidden": hidden, "logits": logits}
        loss = _causal_lm_loss(logits, labels)
        if loss is not None:
            result["loss"] = loss
        return result


__all__ = [
    "BaselineTransformerConfig",
    "ComplexTransformerLM",
    "PlainTransformerLM",
    "SmallMambaConfig",
    "SmallMambaLM",
]
