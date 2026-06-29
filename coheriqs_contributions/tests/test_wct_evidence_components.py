from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid

from coheriqs_contributions.moabb_pipelines.common import (
    augment_paired_cwt_batch,
    print_torch_custom_model_summary,
    print_torch_parameter_hashes,
    torch_parameter_hashes,
)
from coheriqs_contributions.moabb_pipelines.wct_phase_gnn_classifier import (
    WCTPhaseGNNClassifier,
    _compute_wct_window_features,
)
from coheriqs_contributions.moabb_pipelines.wct_evidence_gnn_classifier import (
    WCTEvidenceGNNClassifier,
    WCTEvidenceGNNCore,
)


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def _random_wct_batch(
    *,
    batch_size: int = 2,
    n_channels: int = 3,
    n_time: int = 64,
    nfreqs: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    raw_x = torch.randn(batch_size, n_channels, n_time, generator=generator)
    w_real = torch.randn(batch_size, n_channels, n_time, nfreqs, generator=generator)
    w_imag = torch.randn(batch_size, n_channels, n_time, nfreqs, generator=generator)
    freqs = torch.linspace(8.0, 35.0, nfreqs).expand(batch_size, nfreqs)
    return raw_x, w_real, w_imag, freqs


def test_legacy_profile_keeps_current_trainable_dense_shapes() -> None:
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        hidden_dim=7,
        message_dim=5,
        use_mag=True,
        use_ang=False,
        use_raw=True,
        readout_mode="flatten",
        component_profile="legacy",
    )

    linear_shapes = [(m.in_features, m.out_features) for m in _linear_layers(core)]

    assert linear_shapes == [(13, 5), (5, 7), (21, 2)]


def test_message_component_accepts_edge_frequency_payload_shape() -> None:
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        hidden_dim=7,
        message_dim=5,
        use_mag=True,
        use_ang=False,
        use_raw=True,
        message_layer_norm=True,
    )

    out = core.message_mlp(torch.randn(2, 6, 4, 13))

    assert out.shape == (2, 6, 4, 7)
    assert any(isinstance(layer, nn.LayerNorm) for layer in core.message_mlp.modules())


def test_component_controls_are_sklearn_parameter_grid_friendly() -> None:
    estimator = WCTEvidenceGNNClassifier()

    assert estimator.get_params()["component_profile"] == "legacy"
    estimator.set_params(message_layer_norm=True)
    assert estimator.message_layer_norm is True

    combos = list(ParameterGrid({"message_layer_norm": [False, True]}))
    assert combos == [
        {"message_layer_norm": False},
        {"message_layer_norm": True},
    ]


def test_window_compute_controls_are_sklearn_parameter_grid_friendly() -> None:
    estimator = WCTEvidenceGNNClassifier()

    assert estimator.get_params()["window_compute_mode"] == "auto"
    assert estimator.get_params()["max_windows_per_chunk"] is None

    estimator.set_params(
        window_compute_mode="chunked",
        max_windows_per_chunk=2,
    )
    assert estimator.window_compute_mode == "chunked"
    assert estimator.max_windows_per_chunk == 2

    combos = list(
        ParameterGrid(
            {
                "window_compute_mode": ["sequential", "chunked"],
                "max_windows_per_chunk": [1, 2],
            }
        )
    )
    assert len(combos) == 4


@pytest.mark.parametrize("use_mag", [True, False])
def test_window_compute_modes_match_for_exact_windowed_config(use_mag: bool) -> None:
    batch_inputs = _random_wct_batch(n_time=96)
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        coherence_threshold=0.0,
        phase_threshold_deg=-180.0,
        window_size=32,
        use_mag=use_mag,
        use_ang=True,
        use_raw=True,
        use_freq=True,
        use_time=True,
        model_init_seed=23,
        window_compute_mode="sequential",
    )
    core.eval()

    outputs = {}
    with torch.no_grad():
        for mode in ["sequential", "chunked", "single_pass_windowed", "auto"]:
            core.window_compute_mode = mode
            core.max_windows_per_chunk = 2 if mode == "chunked" else None
            outputs[mode] = core(*batch_inputs)

    sequential_logits, sequential_density = outputs["sequential"]
    for mode in ["chunked", "single_pass_windowed", "auto"]:
        logits, density = outputs[mode]
        assert torch.allclose(logits, sequential_logits, rtol=1e-5, atol=1e-5)
        assert density == pytest.approx(sequential_density, abs=1e-7)


def test_wct_window_features_can_skip_magnitude_without_changing_phase_or_coh() -> None:
    generator = torch.Generator().manual_seed(31)
    src_r = torch.randn(2, 6, 16, 4, generator=generator)
    src_i = torch.randn(2, 6, 16, 4, generator=generator)
    dst_r = torch.randn(2, 6, 16, 4, generator=generator)
    dst_i = torch.randn(2, 6, 16, 4, generator=generator)
    freqs = torch.linspace(8.0, 35.0, 4).expand(2, 4)

    mean_mag, mean_phase, coh = _compute_wct_window_features(
        src_r,
        src_i,
        dst_r,
        dst_i,
        freqs,
    )
    skipped_mag, skipped_phase, skipped_coh = _compute_wct_window_features(
        src_r,
        src_i,
        dst_r,
        dst_i,
        freqs,
        compute_mag=False,
    )

    assert mean_mag is not None
    assert skipped_mag is None
    assert torch.allclose(skipped_phase, mean_phase)
    assert torch.allclose(skipped_coh, coh)


def test_single_pass_continuous_outputs_valid_shape() -> None:
    batch_inputs = _random_wct_batch()
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        coherence_threshold=0.0,
        phase_threshold_deg=-180.0,
        window_size=32,
        model_init_seed=23,
        window_compute_mode="single_pass_continuous",
    )
    core.eval()

    with torch.no_grad():
        logits, edge_density = core(*batch_inputs)

    assert logits.shape == (2, 2)
    assert 0.0 <= edge_density <= 1.0


def test_window_compute_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="window_compute_mode"):
        WCTEvidenceGNNCore(
            n_channels=3,
            nfreqs=4,
            n_classes=2,
            window_compute_mode="invalid",
        )

    with pytest.raises(ValueError, match="max_windows_per_chunk"):
        WCTEvidenceGNNCore(
            n_channels=3,
            nfreqs=4,
            n_classes=2,
            max_windows_per_chunk=0,
        )


def test_custom_model_summary_default_noop(capsys) -> None:
    print_torch_custom_model_summary(nn.Linear(2, 1), header="Plain")

    assert capsys.readouterr().out == ""


def test_wct_evidence_custom_model_summary_prints_mode_and_memory(capsys) -> None:
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        window_size=32,
        window_compute_mode="auto",
    )
    core.configure_summary_context(
        batch_size=8,
        n_time=64,
        dtype=torch.float32,
        n_samples=12,
    )

    print_torch_custom_model_summary(core, header="Evidence")
    output = capsys.readouterr().out

    assert "[Evidence] WCTEvidence config" in output
    assert "window_compute_mode=auto" in output
    assert "effective_window_compute_mode=single_pass_windowed" in output
    assert "B=8 C=3 E=6 T=64 W=32 N=2 F=4" in output
    assert "message_payload" in output
    assert "approx_memory=" in output


def test_run_wct_gnn_evidence_config_smoke_matches_grid() -> None:
    from coheriqs_contributions import run_wct_gnn

    estimator = run_wct_gnn._make_wct_evidence_gnn()
    params = next(iter(ParameterGrid(run_wct_gnn.PIPELINE_PARAM_GRIDS["WCT-Evidence-GNN"])))
    estimator.set_params(**params)

    batch_inputs = _random_wct_batch(
        batch_size=2,
        n_channels=3,
        n_time=200,
        nfreqs=16,
    )
    core = estimator._build_model_from_features(batch_inputs, n_classes=2)
    core.eval()

    assert estimator.use_mag is False
    assert estimator.window_compute_mode == params["window_compute_mode"]
    assert estimator.max_windows_per_chunk == params["max_windows_per_chunk"]
    assert core._resolve_window_compute_mode() == params["window_compute_mode"]

    with torch.no_grad():
        logits, edge_density = core(*batch_inputs)

    assert logits.shape == (2, 2)
    assert 0.0 <= edge_density <= 1.0


def test_noise_augmentation_controls_are_sklearn_parameter_grid_friendly() -> None:
    estimator = WCTEvidenceGNNClassifier()

    estimator.set_params(
        noise_augmentation_enabled=True,
        noise_apply_prob=0.5,
        noise_strength=0.25,
        noise_bank_size=7,
        noise_bank_seed=11,
    )

    params = estimator.get_params()
    assert params["noise_augmentation_enabled"] is True
    assert params["noise_apply_prob"] == 0.5
    assert params["noise_strength"] == 0.25
    assert params["noise_bank_size"] == 7
    assert params["noise_bank_seed"] == 11

    sibling = WCTPhaseGNNClassifier(noise_strength=0.1)
    assert sibling.get_params()["noise_strength"] == 0.1


def test_paired_cwt_noise_augmentation_preserves_pairing_and_shapes() -> None:
    raw_x = torch.zeros(3, 2, 4)
    w_real = torch.zeros(3, 2, 4, 2)
    w_imag = torch.zeros(3, 2, 4, 2)
    raw_bank = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    real_bank = raw_bank[:, :, None].expand(2, 4, 2) * 10.0
    imag_bank = raw_bank[:, :, None].expand(2, 4, 2) * 100.0

    augmented = augment_paired_cwt_batch(
        (raw_x, w_real, w_imag),
        noise_bank=(raw_bank, real_bank, imag_bank),
        channel_std=torch.tensor([2.0, 3.0]),
        apply_prob=1.0,
        strength=0.5,
    )

    raw_aug, real_aug, imag_aug = augmented
    assert raw_aug.shape == raw_x.shape
    assert real_aug.shape == w_real.shape
    assert imag_aug.shape == w_imag.shape

    scale = torch.tensor([1.0, 1.5]).view(1, 2, 1)
    raw_unit = raw_aug / scale
    real_unit = real_aug / scale.unsqueeze(-1)
    imag_unit = imag_aug / scale.unsqueeze(-1)
    assert torch.allclose(real_unit, raw_unit.unsqueeze(-1).expand_as(real_unit) * 10.0)
    assert torch.allclose(imag_unit, raw_unit.unsqueeze(-1).expand_as(imag_unit) * 100.0)


def test_paired_cwt_noise_augmentation_noop_modes() -> None:
    raw_x = torch.randn(2, 3, 4)
    w_real = torch.randn(2, 3, 4, 2)
    w_imag = torch.randn(2, 3, 4, 2)
    bank = (
        torch.randn(5, 4),
        torch.randn(5, 4, 2),
        torch.randn(5, 4, 2),
    )

    for apply_prob, strength in [(0.0, 1.0), (1.0, 0.0)]:
        augmented = augment_paired_cwt_batch(
            (raw_x, w_real, w_imag),
            noise_bank=bank,
            channel_std=torch.ones(3),
            apply_prob=apply_prob,
            strength=strength,
        )
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(augmented, (raw_x, w_real, w_imag), strict=True)
        )


def test_paired_cwt_noise_augmentation_rejects_device_mismatch() -> None:
    raw_x = torch.zeros(2, 3, 4)
    w_real = torch.zeros(2, 3, 4, 2)
    w_imag = torch.zeros(2, 3, 4, 2)
    bank = (
        torch.empty(5, 4, device="meta"),
        torch.empty(5, 4, 2, device="meta"),
        torch.empty(5, 4, 2, device="meta"),
    )

    with pytest.raises(ValueError, match="batch device"):
        augment_paired_cwt_batch(
            (raw_x, w_real, w_imag),
            noise_bank=bank,
            channel_std=torch.ones(3),
            apply_prob=1.0,
            strength=1.0,
        )


def test_classifier_prepares_noise_state_on_device_and_uses_it_for_batches() -> None:
    class RecordingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_raw = None

        def forward(self, raw_x, w_real, w_imag, freqs=None):
            self.seen_raw = raw_x.detach().clone()
            return torch.zeros(raw_x.shape[0], 2, device=raw_x.device)

    estimator = WCTEvidenceGNNClassifier(
        noise_augmentation_enabled=True,
        noise_apply_prob=1.0,
        noise_strength=1.0,
    )
    estimator.device_ = torch.device("cpu")
    estimator.noise_bank_ = (
        torch.ones(2, 4),
        torch.ones(2, 4, 3),
        torch.ones(2, 4, 3),
    )
    estimator.noise_channel_std_ = torch.ones(2)
    estimator._prepare_training_state_on_device()

    assert estimator.noise_bank_device_ is not None
    assert estimator.noise_channel_std_device_ is not None
    assert all(tensor.device == estimator.device_ for tensor in estimator.noise_bank_device_)
    assert estimator.noise_channel_std_device_.device == estimator.device_

    estimator.noise_bank_ = (
        torch.empty(2, 4, device="meta"),
        torch.empty(2, 4, 3, device="meta"),
        torch.empty(2, 4, 3, device="meta"),
    )
    estimator.noise_channel_std_ = torch.empty(2, device="meta")
    estimator.model_ = RecordingModel()
    estimator.model_.train()
    batch_inputs = (
        torch.zeros(2, 2, 4),
        torch.zeros(2, 2, 4, 3),
        torch.zeros(2, 2, 4, 3),
    )

    estimator._model_forward(batch_inputs)

    assert torch.equal(estimator.model_.seen_raw, torch.ones_like(batch_inputs[0]))


def test_noise_channel_std_uses_training_split_after_resampling(monkeypatch) -> None:
    estimator = WCTEvidenceGNNClassifier(
        noise_augmentation_enabled=True,
        noise_apply_prob=1.0,
        noise_strength=0.5,
        noise_bank_size=3,
    )
    raw_x = torch.tensor(
        [
            [[1.0, 3.0], [10.0, 14.0]],
            [[5.0, 7.0], [20.0, 24.0]],
            [[100.0, 100.0], [200.0, 200.0]],
        ]
    )
    features = (
        raw_x,
        torch.zeros(3, 2, 2, 4),
        torch.zeros(3, 2, 2, 4),
    )
    captured = {}

    def fake_noise_bank(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros(3, 2),
            torch.zeros(3, 2, 4),
            torch.zeros(3, 2, 4),
        )

    monkeypatch.setattr(
        "coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier."
        "compute_paired_cwt_noise_bank",
        fake_noise_bank,
    )
    estimator.transform_ = object()

    estimator._fit_noise_augmentation_state(
        features,
        X=torch.zeros(3, 2, 8).numpy(),
        train_idx=torch.tensor([0, 1]),
    )

    expected = torch.std(raw_x[:2], dim=(0, 2), unbiased=False)
    assert torch.equal(estimator.noise_channel_std_, expected)
    assert captured["bank_size"] == 3
    assert captured["segment_length"] == 8


def test_component_seed_grid_propagates_independently_to_core() -> None:
    estimator = WCTEvidenceGNNClassifier(seed=23)
    params = next(
        iter(
            ParameterGrid(
                {
                    "message_init_seed": [29],
                    "readout_init_seed": [31],
                }
            )
        )
    )
    estimator.set_params(**params)
    assert estimator.get_params()["message_init_seed"] == 29
    assert estimator.get_params()["readout_init_seed"] == 31

    reference = estimator._build_model(n_channels=3, n_classes=2)
    repeated = estimator._build_model(n_channels=3, n_classes=2)
    different_message = estimator.set_params(message_init_seed=37)._build_model(3, 2)
    different_readout = estimator.set_params(
        message_init_seed=29,
        readout_init_seed=41,
    )._build_model(3, 2)

    _assert_parameters_equal(reference, repeated)
    _assert_parameters_differ(reference.message_mlp, different_message.message_mlp)
    _assert_parameters_equal(reference.classifier, different_message.classifier)
    _assert_parameters_equal(reference.message_mlp, different_readout.message_mlp)
    _assert_parameters_differ(reference.classifier, different_readout.classifier)


def test_model_init_seed_reproduces_unspecified_components() -> None:
    first = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        model_init_seed=23,
    )
    second = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        model_init_seed=23,
    )

    for first_param, second_param in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        assert torch.equal(first_param, second_param)


def test_message_seed_override_does_not_change_readout_seed_stream() -> None:
    first = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        model_init_seed=23,
        message_init_seed=29,
    )
    second = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        model_init_seed=23,
        message_init_seed=31,
    )

    assert any(
        not torch.equal(first_param, second_param)
        for first_param, second_param in zip(
            first.message_mlp.parameters(),
            second.message_mlp.parameters(),
            strict=True,
        )
    )
    for first_param, second_param in zip(
        first.classifier.parameters(),
        second.classifier.parameters(),
        strict=True,
    ):
        assert torch.equal(first_param, second_param)


def test_parameter_hashes_track_reproducible_initialization(capsys) -> None:
    first = WCTEvidenceGNNCore(3, 4, 2, model_init_seed=23)
    second = WCTEvidenceGNNCore(3, 4, 2, model_init_seed=23)

    first_hashes, first_model_hash = torch_parameter_hashes(first)
    second_hashes, second_model_hash = torch_parameter_hashes(second)
    assert first_hashes == second_hashes
    assert first_model_hash == second_model_hash

    with torch.no_grad():
        next(second.parameters()).view(-1)[0].add_(1e-4)
    changed_hashes, changed_model_hash = torch_parameter_hashes(second)
    assert changed_hashes != first_hashes
    assert changed_model_hash != first_model_hash

    print_torch_parameter_hashes(first, header="HashTest")
    output = capsys.readouterr().out
    assert all(name in output for name in first_hashes)
    assert f"model_hash={first_model_hash}" in output


def _assert_parameters_equal(first: nn.Module, second: nn.Module) -> None:
    for first_param, second_param in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        assert torch.equal(first_param, second_param)


def _assert_parameters_differ(first: nn.Module, second: nn.Module) -> None:
    assert any(
        not torch.equal(first_param, second_param)
        for first_param, second_param in zip(
            first.parameters(),
            second.parameters(),
            strict=True,
        )
    )
