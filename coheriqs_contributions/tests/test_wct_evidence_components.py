from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid

from coheriqs_contributions.moabb_pipelines.common import (
    print_torch_parameter_hashes,
    torch_parameter_hashes,
)
from coheriqs_contributions.moabb_pipelines.wct_evidence_gnn_classifier import (
    WCTEvidenceGNNClassifier,
    WCTEvidenceGNNCore,
)


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


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

    assert linear_shapes == [(3, 5), (5, 7), (21, 2)]


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

    out = core.message_mlp(torch.randn(2, 6, 4, 3))

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
