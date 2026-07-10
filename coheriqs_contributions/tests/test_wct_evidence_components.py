from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, TensorDataset

from coheriqs_contributions.moabb_pipelines.common import (
    TorchEEGClassifier,
    _collect_selector_training_specs,
    _selector_extra_loss,
    _selector_gate_modules,
    _selector_params,
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
from coheriqs_contributions.nn_components import CategoricalGateConfig, SelectPath


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def _select_path_branches(path: SelectPath) -> list[nn.Module]:
    return [candidate.branch for candidate in path.choice.candidates]


class _ToySelectorModel(nn.Module):
    def __init__(
        self,
        *,
        n_classes: int,
        selector_mode: str,
        gate_mode: str,
        eval_mode: str = "same",
        entropy_weight: float = 0.0,
        frozen_index: int | None = None,
    ) -> None:
        super().__init__()
        alpha_optim = "shared" if selector_mode == "shared_train" else "separate"
        alpha_update_split = "val" if selector_mode == "separate_val" else "train"
        gate = CategoricalGateConfig(
            num_choices=2,
            mode=gate_mode,
            eval_mode=eval_mode,
            entropy_weight=entropy_weight,
            frozen_index=frozen_index,
            alpha_optim=alpha_optim,
            alpha_update_split=alpha_update_split,
        )
        self.selector = SelectPath(
            [
                nn.Linear(2, n_classes),
                nn.Sequential(nn.Linear(2, 3), nn.GELU(), nn.Linear(3, n_classes)),
            ],
            gate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selector(x)


class _ToySelectorEstimator(TorchEEGClassifier):
    def __init__(
        self,
        *,
        selector_mode: str = "separate_train",
        gate_mode: str = "soft",
        eval_mode: str = "same",
        entropy_weight: float = 0.0,
        frozen_index: int | None = None,
        epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        validation_split: float | None = 0.5,
        seed: int | None = 7,
        last_batch_min_ratio: float = 0.0,
        selector_alpha_val_update_rate: float = 1.0,
        verbose: int = 0,
    ) -> None:
        self.selector_mode = selector_mode
        self.gate_mode = gate_mode
        self.eval_mode = eval_mode
        self.entropy_weight = entropy_weight
        self.frozen_index = frozen_index
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split = validation_split
        self.seed = seed
        self.last_batch_min_ratio = last_batch_min_ratio
        self.selector_alpha_val_update_rate = selector_alpha_val_update_rate
        self.verbose = verbose
        self._init_torch_classifier(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split=validation_split,
            seed=seed,
            last_batch_min_ratio=last_batch_min_ratio,
            selector_alpha_val_update_rate=selector_alpha_val_update_rate,
            use_class_weights=False,
            verbose=verbose,
            device="cpu",
        )

    def _prepare_features(self, X: np.ndarray, *, fit: bool, train_idx=None):
        del fit, train_idx
        return X.reshape(X.shape[0], -1).astype(np.float32)

    def _build_model_from_features(self, features, n_classes: int, **kwargs) -> nn.Module:
        del features, kwargs
        return _ToySelectorModel(
            n_classes=n_classes,
            selector_mode=self.selector_mode,
            gate_mode=self.gate_mode,
            eval_mode=self.eval_mode,
            entropy_weight=self.entropy_weight,
            frozen_index=self.frozen_index,
        )


def _toy_eeg_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [[-1.0, -0.5]],
            [[1.0, 0.5]],
            [[-0.8, -0.6]],
            [[0.8, 0.6]],
            [[-0.6, -1.0]],
            [[0.6, 1.0]],
            [[-1.2, -0.4]],
            [[1.2, 0.4]],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return X, y


def _selector_num_forwards(history: list[list[dict[str, object]]]) -> int:
    return int(history[0][0]["num_forwards"])


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
    assert estimator.get_params()["last_batch_min_ratio"] == 0.0
    assert estimator.get_params()["selector_alpha_val_update_rate"] == 1.0
    estimator.set_params(message_layer_norm=True)
    assert estimator.message_layer_norm is True

    combos = list(ParameterGrid({"message_layer_norm": [False, True]}))
    assert combos == [
        {"message_layer_norm": False},
        {"message_layer_norm": True},
    ]


def test_select_message_mlp_controls_are_grid_friendly_and_build_select_path() -> None:
    estimator = WCTEvidenceGNNClassifier(seed=23, hidden_dim=7, message_dim=5)
    params = next(
        iter(
            ParameterGrid(
                {
                    "select_message_mlp": [
                        [
                            {"init_seed": 101},
                            {
                                "init_seed": 202,
                                "message_dim": 6,
                                "message_layer_norm": True,
                            },
                        ]
                    ],
                    "select_message_mlp_gate": [
                        {"mode": "soft", "eval_mode": "same"},
                    ],
                    "message_mlp_selector_mode": ["separate_val"],
                }
            )
        )
    )

    estimator.set_params(**params)

    assert estimator.get_params()["select_message_mlp"] == params["select_message_mlp"]
    assert estimator.get_params()["select_message_mlp_gate"] == {
        "mode": "soft",
        "eval_mode": "same",
    }
    assert estimator.get_params()["message_mlp_selector_mode"] == "separate_val"

    core = estimator._build_model(n_channels=3, n_classes=2)
    assert isinstance(core.message_mlp, SelectPath)
    assert core.message_mlp.gate.alpha_optim == "separate"
    assert core.message_mlp.gate.alpha_update_split == "val"

    output = core.message_mlp(torch.randn(2, 6, 4, core.payload_dim))
    assert output.shape == (2, 6, 4, 7)

    branches = _select_path_branches(core.message_mlp)
    assert not any(isinstance(layer, nn.LayerNorm) for layer in branches[0].modules())
    assert any(isinstance(layer, nn.LayerNorm) for layer in branches[1].modules())


@pytest.mark.parametrize(
    ("candidate", "match"),
    [
        ({"in_features": 3}, "shape-derived"),
        ({"out_features": 3}, "shape-derived"),
        ({"message_dim": 5, "hidden_features": 5}, "both 'message_dim'"),
        ({"seed_key": "branch-a"}, "Unsupported"),
    ],
)
def test_select_message_mlp_rejects_invalid_candidate_keys(
    candidate: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        WCTEvidenceGNNCore(
            n_channels=3,
            nfreqs=4,
            n_classes=2,
            select_message_mlp=[candidate],
        )


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
        for mode, max_windows_per_chunk in [
            ("sequential", None),
            ("chunked_cap_1", 1),
            ("chunked_cap_2", 2),
            ("chunked_cap_4", 4),
            ("single_pass_windowed", None),
            ("auto", None),
        ]:
            core.window_compute_mode = "chunked" if mode.startswith("chunked") else mode
            core.max_windows_per_chunk = max_windows_per_chunk
            outputs[mode] = core(*batch_inputs)

    sequential_logits, sequential_density = outputs["sequential"]
    for mode in [
        "chunked_cap_1",
        "chunked_cap_2",
        "chunked_cap_4",
        "single_pass_windowed",
        "auto",
    ]:
        logits, density = outputs[mode]
        assert torch.allclose(logits, sequential_logits, rtol=1e-5, atol=1e-5)
        assert density == pytest.approx(sequential_density, abs=1e-7)


@pytest.mark.parametrize("use_mag", [True, False])
def test_chunked_caps_match_for_exact_windowed_config(use_mag: bool) -> None:
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
        window_compute_mode="chunked",
    )
    core.eval()

    outputs = {}
    with torch.no_grad():
        for max_windows_per_chunk in [1, 2, 4]:
            core.window_compute_mode = "chunked"
            core.max_windows_per_chunk = max_windows_per_chunk
            outputs[max_windows_per_chunk] = core(*batch_inputs)

    reference_logits, reference_density = outputs[1]
    for max_windows_per_chunk in [2, 4]:
        logits, density = outputs[max_windows_per_chunk]
        assert torch.allclose(logits, reference_logits, rtol=1e-5, atol=1e-5)
        assert density == pytest.approx(reference_density, abs=1e-7)


def test_single_pass_windowed_fallback_matches_chunked_for_shorter_kernel() -> None:
    batch_inputs = _random_wct_batch(n_time=96)
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        coherence_threshold=0.0,
        phase_threshold_deg=-180.0,
        window_size=32,
        smooth_kernel_size=(9, 3),
        use_mag=False,
        use_ang=True,
        use_raw=True,
        use_freq=True,
        use_time=True,
        model_init_seed=23,
        window_compute_mode="chunked",
        max_windows_per_chunk=2,
    )
    core.eval()

    with torch.no_grad():
        chunked_logits, chunked_density = core(*batch_inputs)
        core.window_compute_mode = "single_pass_windowed"
        windowed_logits, windowed_density = core(*batch_inputs)

    assert torch.allclose(windowed_logits, chunked_logits, rtol=1e-5, atol=1e-5)
    assert windowed_density == pytest.approx(chunked_density, abs=1e-7)


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


def test_wct_evidence_summary_includes_selectable_message_mlp_diagnostics(
    capsys,
) -> None:
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        select_message_mlp=[
            {"init_seed": 101},
            {"init_seed": 202, "message_dim": 6},
        ],
        select_message_mlp_gate={"mode": "soft", "eval_mode": "same"},
        message_mlp_selector_mode="separate_val",
    )

    print_torch_custom_model_summary(core, header="Evidence")
    output = capsys.readouterr().out

    assert "selectable message_mlp" in output
    assert "candidates=2" in output
    assert "selector_optimizer_mode=separate_val" in output
    assert "initial_probs=" in output
    assert "entropy=" in output
    assert "candidate_params=" in output


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
    assert estimator.last_batch_min_ratio == params["last_batch_min_ratio"]
    assert (
        estimator.selector_alpha_val_update_rate
        == params["selector_alpha_val_update_rate"]
    )
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


def test_select_message_mlp_duplicate_explicit_seed_reuses_initialization() -> None:
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        message_dim=5,
        model_init_seed=23,
        message_init_seed=29,
        select_message_mlp=[
            {"init_seed": 101},
            {"init_seed": 101},
        ],
    )

    branches = _select_path_branches(core.message_mlp)

    _assert_parameters_equal(branches[0], branches[1])


def test_parameter_hashes_are_value_based_for_duplicate_selectable_candidates() -> None:
    core = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        message_dim=5,
        model_init_seed=23,
        message_init_seed=29,
        select_message_mlp=[
            {"init_seed": 101},
            {"init_seed": 101},
        ],
    )

    hashes, _, _ = torch_parameter_hashes(core)

    for suffix in ["1.weight", "1.bias", "4.weight", "4.bias"]:
        first_name = f"message_mlp.choice.candidates.0.branch.{suffix}"
        second_name = f"message_mlp.choice.candidates.1.branch.{suffix}"
        assert (
            hashes[first_name]
            == hashes[second_name]
        )


def test_select_message_mlp_missing_seeds_consume_message_seed_stream() -> None:
    without_explicit = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        message_dim=5,
        model_init_seed=23,
        message_init_seed=29,
        select_message_mlp=[
            {},
            {},
        ],
    )
    with_explicit_between = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        message_dim=5,
        model_init_seed=23,
        message_init_seed=29,
        select_message_mlp=[
            {},
            {"init_seed": 101},
            {},
        ],
    )

    without_branches = _select_path_branches(without_explicit.message_mlp)
    with_branches = _select_path_branches(with_explicit_between.message_mlp)

    _assert_parameters_equal(with_branches[0], without_branches[0])
    _assert_parameters_equal(with_branches[2], without_branches[1])
    _assert_parameters_differ(without_branches[0], without_branches[1])


@pytest.mark.parametrize(
    "selector_mode",
    ["shared_train", "separate_train", "separate_val"],
)
def test_selector_optimizer_grouping_respects_message_mlp_selector_mode(
    selector_mode: str,
) -> None:
    estimator = WCTEvidenceGNNClassifier(
        seed=23,
        weight_decay=0.125,
        message_mlp_selector_mode=selector_mode,
        select_message_mlp=[
            {"init_seed": 101},
            {"init_seed": 202},
        ],
    )
    estimator.model_ = estimator._build_model(n_channels=3, n_classes=2)
    selector_param_ids = {
        id(param) for param in estimator.model_.message_mlp.selection_parameters()
    }

    optimizer, alpha_optimizer, _ = estimator._build_training_optimizers()
    train_group_by_selector = [
        group
        for group in optimizer.param_groups
        if any(id(param) in selector_param_ids for param in group["params"])
    ]
    alpha_group_by_selector = []
    if alpha_optimizer is not None:
        alpha_group_by_selector = [
            group
            for group in alpha_optimizer.param_groups
            if any(id(param) in selector_param_ids for param in group["params"])
        ]

    if selector_mode == "shared_train":
        assert alpha_optimizer is None
        assert len(train_group_by_selector) == 1
        assert train_group_by_selector[0]["weight_decay"] == pytest.approx(0.125)
    elif selector_mode == "separate_train":
        assert alpha_optimizer is None
        assert len(train_group_by_selector) == 1
        assert train_group_by_selector[0]["weight_decay"] == 0.0
    else:
        assert train_group_by_selector == []
        assert len(alpha_group_by_selector) == 1
        assert alpha_group_by_selector[0]["weight_decay"] == 0.0


def test_training_without_selectors_keeps_selector_histories_empty() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_train",
        gate_mode="soft",
        epochs=1,
        batch_size=2,
        validation_split=0.5,
    )
    estimator._build_model_from_features = lambda features, n_classes, **kwargs: nn.Linear(
        features.shape[-1],
        n_classes,
    )

    estimator.fit(X, y)

    assert estimator.selector_train_history_ == []
    assert estimator.selector_val_history_ == []
    assert estimator.selector_alpha_val_history_ == []


def test_default_last_batch_ratio_keeps_small_remainder_batches() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_train",
        epochs=1,
        batch_size=3,
        validation_split=0.5,
        last_batch_min_ratio=0.0,
    )

    estimator.fit(X, y)

    assert _selector_num_forwards(estimator.selector_train_history_) == 2


def test_last_batch_ratio_one_skips_undersized_train_and_alpha_batches() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        epochs=1,
        batch_size=3,
        validation_split=0.5,
        last_batch_min_ratio=1.0,
    )

    estimator.fit(X, y)

    assert _selector_num_forwards(estimator.selector_train_history_) == 1
    assert _selector_num_forwards(estimator.selector_alpha_val_history_) == 1


def test_last_batch_ratio_errors_when_no_train_batch_is_eligible() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_train",
        epochs=1,
        batch_size=16,
        validation_split=0.5,
        last_batch_min_ratio=1.0,
    )

    with pytest.raises(ValueError, match="no eligible training batches"):
        estimator.fit(X, y)


def test_last_batch_ratio_errors_when_no_val_alpha_batch_is_eligible() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        epochs=1,
        batch_size=4,
        validation_split=0.25,
        last_batch_min_ratio=1.0,
        selector_alpha_val_update_rate=1.0,
    )

    with pytest.raises(ValueError, match="eligible validation batch"):
        estimator.fit(X, y)


def test_val_alpha_allows_validation_smaller_than_batch_when_ratio_allows() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        epochs=1,
        batch_size=16,
        validation_split=0.5,
        last_batch_min_ratio=0.0,
        selector_alpha_val_update_rate=1.0,
    )

    estimator.fit(X, y)

    assert _selector_num_forwards(estimator.selector_train_history_) == 1
    assert _selector_num_forwards(estimator.selector_alpha_val_history_) == 1


def test_selector_alpha_val_update_rate_half_updates_every_other_step() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        epochs=1,
        batch_size=1,
        validation_split=0.5,
        selector_alpha_val_update_rate=0.5,
    )

    estimator.fit(X, y)

    assert _selector_num_forwards(estimator.selector_train_history_) == 4
    assert _selector_num_forwards(estimator.selector_alpha_val_history_) == 2


def test_selector_alpha_val_update_rate_above_one_runs_multiple_alpha_steps() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        epochs=1,
        batch_size=1,
        validation_split=0.5,
        selector_alpha_val_update_rate=2.0,
    )

    estimator.fit(X, y)

    assert _selector_num_forwards(estimator.selector_train_history_) == 4
    assert _selector_num_forwards(estimator.selector_alpha_val_history_) == 8


def test_selector_alpha_val_update_rate_zero_disables_val_alpha_updates() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        epochs=1,
        batch_size=2,
        validation_split=None,
        selector_alpha_val_update_rate=0.0,
    )

    estimator.fit(X, y)

    assert estimator.selector_alpha_val_history_ == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"last_batch_min_ratio": -0.1}, "last_batch_min_ratio"),
        ({"last_batch_min_ratio": 1.1}, "last_batch_min_ratio"),
        ({"selector_alpha_val_update_rate": -0.1}, "selector_alpha_val_update_rate"),
    ],
)
def test_batch_control_params_are_validated(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _ToySelectorEstimator(**kwargs)


def test_single_shared_selectable_message_mlp_matches_original_mlp_forward() -> None:
    original = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        hidden_dim=7,
        message_dim=5,
        model_init_seed=23,
        message_init_seed=29,
    )
    selectable = WCTEvidenceGNNCore(
        n_channels=3,
        nfreqs=4,
        n_classes=2,
        hidden_dim=7,
        message_dim=5,
        model_init_seed=23,
        message_init_seed=29,
        select_message_mlp=[{"init_seed": 29}],
        select_message_mlp_gate={"mode": "soft", "eval_mode": "same"},
        message_mlp_selector_mode="shared_train",
    )

    payload = torch.randn(2, 6, 4, original.payload_dim)
    original_out = original.message_mlp(payload)
    selectable_out = selectable.message_mlp(payload)

    assert torch.equal(original_out, selectable_out)


def test_selector_histories_are_populated_after_synthetic_train_loop() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        entropy_weight=0.01,
        epochs=1,
        batch_size=2,
        validation_split=0.5,
    )

    estimator.fit(X, y)

    assert len(estimator.selector_train_history_) == 1
    assert len(estimator.selector_val_history_) == 1
    assert len(estimator.selector_alpha_val_history_) == 1

    train_summary = estimator.selector_train_history_[0][0]
    val_summary = estimator.selector_val_history_[0][0]
    alpha_summary = estimator.selector_alpha_val_history_[0][0]
    assert train_summary["name"] == "selector"
    assert val_summary["name"] == "selector"
    assert alpha_summary["name"] == "selector"
    assert len(train_summary["probs_mean"]) == 2
    assert len(train_summary["logits_mean"]) == 2
    assert train_summary["entropy_mean"] >= 0.0
    assert alpha_summary["alpha_update_split"] == "val"


@pytest.mark.parametrize(
    ("gate_mode", "frozen_index"),
    [
        ("gumbel_hard", None),
        ("argmax", None),
        ("frozen", 1),
    ],
)
def test_selector_history_records_selected_counts_for_hard_modes(
    gate_mode: str,
    frozen_index: int | None,
) -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_train",
        gate_mode=gate_mode,
        frozen_index=frozen_index,
        epochs=1,
        batch_size=2,
        validation_split=0.5,
    )

    estimator.fit(X, y)

    summary = estimator.selector_train_history_[0][0]
    assert "selected_counts" in summary
    assert sum(summary["selected_counts"]) > 0
    assert "selected_fractions" in summary
    assert sum(summary["selected_fractions"]) == pytest.approx(1.0)
    if frozen_index is not None:
        assert summary["selected_counts"][frozen_index] > 0
        assert summary["selected_fractions"][frozen_index] == pytest.approx(1.0)


def test_train_selector_extra_loss_excludes_separate_val_regularization() -> None:
    model = _ToySelectorModel(
        n_classes=2,
        selector_mode="separate_val",
        gate_mode="soft",
        entropy_weight=10.0,
    )
    logits = model(torch.randn(4, 2))
    base_loss = logits.sum() * 0.0
    selector_specs = _collect_selector_training_specs(model)

    train_extra_loss = _selector_extra_loss(
        selector_specs,
        base_loss,
        alpha_update_split="train",
    )
    val_extra_loss = _selector_extra_loss(
        selector_specs,
        base_loss,
        alpha_optim="separate",
        alpha_update_split="val",
    )

    assert train_extra_loss.item() == pytest.approx(0.0)
    assert val_extra_loss.item() > 0.0


def test_validation_alpha_update_does_not_touch_non_alpha_grads_or_state() -> None:
    X, y = _toy_eeg_data()
    estimator = _ToySelectorEstimator(
        selector_mode="separate_val",
        gate_mode="soft",
        epochs=1,
        batch_size=4,
        validation_split=0.5,
    )
    features = estimator._prepare_features(X, fit=True)
    estimator.device_ = torch.device("cpu")
    estimator.classes_ = np.unique(y)
    estimator.model_ = estimator._build_model_from_features(
        features,
        n_classes=2,
        device=estimator.device_,
    ).to(estimator.device_)

    _, alpha_optimizer, selector_specs = estimator._build_training_optimizers()
    assert alpha_optimizer is not None
    val_alpha_specs = [
        spec
        for spec in selector_specs
        if spec.alpha_optim == "separate" and spec.alpha_update_split == "val"
    ]
    alpha_params = _selector_params(
        val_alpha_specs,
        alpha_optim="separate",
        alpha_update_split="val",
    )
    alpha_param_ids = {id(param) for param in alpha_params}
    gate_modules = _selector_gate_modules(
        val_alpha_specs,
        alpha_optim="separate",
        alpha_update_split="val",
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(y).long(),
        ),
        batch_size=4,
        shuffle=False,
    )

    estimator._update_selector_alpha_from_validation(
        val_loader=loader,
        val_iter=iter(loader),
        alpha_optimizer=alpha_optimizer,
        criterion=nn.CrossEntropyLoss(),
        selector_specs=val_alpha_specs,
        alpha_params=alpha_params,
        gate_modules=gate_modules,
    )

    non_alpha_params = [
        param
        for param in estimator.model_.parameters()
        if id(param) not in alpha_param_ids
    ]
    assert all(param.grad is None for param in non_alpha_params)
    assert {
        id(param)
        for param in alpha_optimizer.state
    }.issubset(alpha_param_ids)


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

    first_hashes, first_weight_model_hash, first_named_model_hash = torch_parameter_hashes(
        first
    )
    (
        second_hashes,
        second_weight_model_hash,
        second_named_model_hash,
    ) = torch_parameter_hashes(second)
    assert first_hashes == second_hashes
    assert first_weight_model_hash == second_weight_model_hash
    assert first_named_model_hash == second_named_model_hash

    with torch.no_grad():
        next(second.parameters()).view(-1)[0].add_(1e-4)
    (
        changed_hashes,
        changed_weight_model_hash,
        changed_named_model_hash,
    ) = torch_parameter_hashes(second)
    assert changed_hashes != first_hashes
    assert changed_weight_model_hash != first_weight_model_hash
    assert changed_named_model_hash != first_named_model_hash

    print_torch_parameter_hashes(first, header="HashTest")
    output = capsys.readouterr().out
    assert all(name in output for name in first_hashes)
    assert "weight_hash=" in output
    assert f"weight_model_hash={first_weight_model_hash}" in output
    assert f"named_model_hash={first_named_model_hash}" in output


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
