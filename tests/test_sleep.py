from pathlib import Path

import torch

from reciprocator_lm import (
    ModelConfig,
    OnlineReplayBuffer,
    ReciprocatorOnlyLM,
    collect_online_episode,
    distillation_kl_loss,
    load_online_replay_buffer,
    load_reciprocator_checkpoint,
    sample_replay_batch,
    save_online_replay_buffer,
    save_reciprocator_checkpoint,
    train_sentencepiece_tokenizer,
)


def _make_model(vocab_size: int) -> ReciprocatorOnlyLM:
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=8,
        dim=16,
        n_layers=1,
        n_heads=4,
        state_rank=3,
        state_mode_sizes=(2, 2, 2),
        max_mode_sizes=(3, 3, 3),
        num_cube_engines=1,
        normalization="per_mode",
        dropout=0.0,
    )
    return ReciprocatorOnlyLM(config)


def test_reciprocator_checkpoint_round_trips_tokenizer(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tokenizer = train_sentencepiece_tokenizer(
        text=("Socrates asks careful questions.\n" * 32).strip(),
        vocab_size=24,
        model_prefix=tmp_path / "sleep_spm",
    )
    model = _make_model(tokenizer.vocab_size)
    checkpoint_path = tmp_path / "model.pt"

    save_reciprocator_checkpoint(
        checkpoint_path,
        model=model,
        config=model.config,
        tokenizer=tokenizer,
        metadata={"tag": "roundtrip"},
    )
    loaded = load_reciprocator_checkpoint(checkpoint_path)

    assert loaded.metadata["tag"] == "roundtrip"
    assert loaded.config.vocab_size == model.config.vocab_size
    loaded_tokenizer = loaded.build_tokenizer()
    assert loaded_tokenizer is not None
    token_ids = loaded_tokenizer.encode("Socrates asks careful questions.")
    assert loaded_tokenizer.decode(token_ids) == "Socrates asks careful questions."
    assert torch.equal(
        loaded.model_state_dict["lm_head.weight"],
        model.state_dict()["lm_head.weight"].cpu(),
    )


def test_collect_online_episode_and_replay_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _make_model(vocab_size=32).eval()
    token_ids = list(range(1, 18))

    episode = collect_online_episode(model, token_ids, seq_len=4, episode_id="toy")
    buffer = OnlineReplayBuffer(episodes=(episode,), tokenizer_vocab_size=32)
    replay_path = tmp_path / "replay.pt"

    save_online_replay_buffer(replay_path, buffer)
    loaded = load_online_replay_buffer(replay_path)
    batch = sample_replay_batch(
        loaded,
        batch_size=2,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )

    assert len(loaded.episodes) == 1
    assert loaded.episodes[0].metadata["seq_len"] == 4
    assert batch.input_ids.shape == (2, 4)
    assert batch.target_ids.shape == (2, 4)
    assert batch.teacher_logits.shape == (2, 4, 32)


def test_distillation_loss_is_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 4, 8)

    loss = distillation_kl_loss(logits, logits, temperature=1.5)

    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)
