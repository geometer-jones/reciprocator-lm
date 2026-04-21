import torch

from reciprocator_lm import (
    BaselineTransformerConfig,
    ComplexTransformerLM,
    PlainTransformerLM,
    SmallMambaConfig,
    SmallMambaLM,
    complex_readout_features,
)


def test_plain_transformer_lm_returns_logits_and_loss() -> None:
    model = PlainTransformerLM(
        BaselineTransformerConfig(
            vocab_size=32,
            model_dim=24,
            num_heads=4,
            num_layers=2,
            max_seq_len=12,
            dropout=0.0,
        )
    )
    input_ids = torch.randint(0, 32, (2, 6))
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs["hidden"].shape == (2, 6, 24)
    assert outputs["logits"].shape == (2, 6, 32)
    assert outputs["loss"].ndim == 0


def test_complex_transformer_lm_returns_complex_hidden_and_loss() -> None:
    model = ComplexTransformerLM(
        BaselineTransformerConfig(
            vocab_size=32,
            model_dim=24,
            num_heads=4,
            num_layers=2,
            max_seq_len=12,
            dropout=0.0,
        )
    )
    input_ids = torch.randint(0, 32, (2, 6))
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert torch.is_complex(outputs["hidden"])
    assert outputs["hidden"].shape == (2, 6, 24)
    assert outputs["logits"].shape == (2, 6, 32)
    assert outputs["loss"].ndim == 0


def test_complex_transformer_phase_aware_head_preserves_global_phase_in_logits() -> None:
    torch.manual_seed(0)
    model = ComplexTransformerLM(
        BaselineTransformerConfig(
            vocab_size=32,
            model_dim=24,
            num_heads=4,
            num_layers=1,
            max_seq_len=12,
            dropout=0.0,
            readout_mode="phase_aware",
        )
    ).eval()
    input_ids = torch.randint(0, 32, (2, 6))
    outputs = model(input_ids=input_ids)
    rotated_hidden = outputs["hidden"] * torch.exp(1j * torch.tensor(0.37, dtype=torch.float32))
    rotated_logits = model.lm_head(complex_readout_features(rotated_hidden, "phase_aware"))

    torch.testing.assert_close(rotated_logits, outputs["logits"], atol=1e-5, rtol=1e-5)


def test_phase_aware_complex_readout_features_are_global_phase_invariant_and_shape_stable() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, dtype=torch.complex64)
    rotated_hidden = hidden * torch.exp(1j * torch.tensor(-0.41, dtype=torch.float32))

    features = complex_readout_features(hidden, "phase_aware")
    rotated_features = complex_readout_features(rotated_hidden, "phase_aware")

    assert features.shape == (2, 5, 21)
    torch.testing.assert_close(rotated_features, features, atol=1e-5, rtol=1e-5)


def test_small_mamba_lm_returns_logits_and_loss() -> None:
    model = SmallMambaLM(
        SmallMambaConfig(
            vocab_size=32,
            model_dim=24,
            num_layers=2,
            state_size=8,
            expand=2,
            conv_kernel=3,
            max_seq_len=12,
            dropout=0.0,
        )
    )
    input_ids = torch.randint(0, 32, (2, 6))
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs["hidden"].shape == (2, 6, 24)
    assert outputs["logits"].shape == (2, 6, 32)
    assert outputs["loss"].ndim == 0


def test_small_mamba_is_causal() -> None:
    torch.manual_seed(0)
    model = SmallMambaLM(
        SmallMambaConfig(
            vocab_size=32,
            model_dim=24,
            num_layers=2,
            state_size=8,
            expand=2,
            conv_kernel=3,
            max_seq_len=12,
            dropout=0.0,
        )
    ).eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    altered = torch.tensor([[1, 2, 3, 9, 9, 9]])

    prefix_logits = model(input_ids=prefix)["logits"]
    altered_logits = model(input_ids=altered)["logits"]

    torch.testing.assert_close(prefix_logits[:, :3], altered_logits[:, :3], atol=1e-5, rtol=1e-5)
