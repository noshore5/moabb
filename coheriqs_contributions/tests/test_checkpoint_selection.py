from copy import deepcopy
from functools import partial
from pickle import dumps, loads

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.base import clone
from torch.utils.data import DataLoader, TensorDataset

from coheriqs_contributions.checkpoint_selection import (
    CandidateGenerator,
    CheckpointController,
    CheckpointScorer,
    TrainingState,
    load_checkpoint_bundle,
    recent_history,
    require_metric,
    save_checkpoint_bundle,
    select_metric,
)
import coheriqs_contributions.moabb_pipelines.common as common_module
from coheriqs_contributions.moabb_pipelines.common import (
    LoaderEvaluation,
    TorchEEGClassifier,
)
from coheriqs_contributions.moabb_pipelines.coherence_cnn_classifier import (
    CoherenceCNNClassifier,
)
from coheriqs_contributions.moabb_pipelines.CWT_CNN import CWTCNNClassifier
from coheriqs_contributions.moabb_pipelines.EEGNet import EEGNetClassifier
from coheriqs_contributions.moabb_pipelines.wct_evidence_gnn_classifier import (
    WCTEvidenceGNNClassifier,
)
from coheriqs_contributions.moabb_pipelines.wct_phase_gnn_classifier import (
    WCTPhaseGNNClassifier,
    WCTPhaseGNNV2Classifier,
)
from coheriqs_contributions.moabb_pipelines.xwt_phase_gnn_classifier import (
    XWTPhaseGNNClassifier,
    XWTPhaseGNNV2Classifier,
)


def _state(epoch: int) -> TrainingState:
    return TrainingState(
        epoch=epoch,
        optimizer_step_count=epoch * 3,
        model_state={"weight": torch.tensor([float(epoch)])},
        optimizer_state={"state": {}, "param_groups": []},
        alpha_optimizer_state=None,
        learning_rates={"main": [0.001 * epoch]},
        selector_state={"epoch": epoch},
    )


@pytest.mark.parametrize(
    "classifier",
    [
        EEGNetClassifier,
        CWTCNNClassifier,
        CoherenceCNNClassifier,
        XWTPhaseGNNClassifier,
        XWTPhaseGNNV2Classifier,
        WCTPhaseGNNClassifier,
        WCTPhaseGNNV2Classifier,
        WCTEvidenceGNNClassifier,
    ],
)
def test_all_neural_estimators_expose_checkpoint_configuration(classifier):
    params = classifier().get_params()
    assert params["clean_train_mode"] == "disabled"
    assert params["checkpoint_scorer"] == "val_loss"
    assert params["candidate_generators"] is None
    assert params["clean_train_interval"] == 1
    assert params["save_selected_checkpoint"] is False


def _nonlinear_recent_score(metrics, history, *, window, recent_weight):
    prior = recent_history(history, window - 1)
    prior_losses = select_metric(prior, "val_data_loss")
    if len(prior_losses) != window - 1 or any(
        value is None for value in prior_losses
    ):
        return None
    losses = (*prior_losses, require_metric(metrics, "val_data_loss"))
    weights = (*range(1, window), recent_weight)
    return -sum(
        weight * np.log1p(loss)
        for weight, loss in zip(weights, losses)
    ) / sum(weights)


def _clean_loss_with_validation_auc(metrics, history, *, auc_weight):
    return (
        -require_metric(metrics, "clean_train_data_loss")
        + auc_weight * require_metric(metrics, "val_roc_auc")
    )


def test_custom_scorer_supports_history_nonlinearity_and_warmup():
    scorer = CheckpointScorer(
        function=partial(
            _nonlinear_recent_score,
            window=3,
            recent_weight=5.0,
        ),
        name="recent-log-loss-3-5",
    )
    history = [{"val_data_loss": 4.0}, {"val_data_loss": 2.0}]
    assert scorer.score({"val_data_loss": 1.0}, history[:1]) is None
    assert select_metric(
        [{"val_data_loss": 4.0}, {}, {"val_data_loss": np.nan}],
        "val_data_loss",
    ) == (4.0, None, None)
    score = scorer.score(
        {"val_data_loss": 1.0},
        history,
    )
    expected = -(
        np.log1p(4.0) + 2 * np.log1p(2.0) + 5 * np.log1p(1.0)
    ) / 8
    assert score == pytest.approx(expected)


def test_custom_scorers_must_be_explicitly_named_and_wrapped():
    with pytest.raises(TypeError, match="built-in string or CheckpointScorer"):
        CheckpointController(
            mode="disabled",
            checkpoint_scorer=_nonlinear_recent_score,
        )
    with pytest.raises(TypeError, match="built-in string or CheckpointScorer"):
        CheckpointController(
            mode="disabled",
            checkpoint_scorer={"metric": "val_data_loss"},
        )
    with pytest.raises(ValueError, match="Unsupported candidate generator keys"):
        CheckpointController(
            mode="deferred_candidates",
            candidate_generators=[{"metric": "val_data_loss"}],
        )


def test_none_score_skips_epoch_without_capturing_state():
    scorer = CheckpointScorer(
        function=partial(
            _nonlinear_recent_score,
            window=2,
            recent_weight=2.0,
        ),
        name="two-value-log-loss",
    )
    controller = CheckpointController(mode="disabled", checkpoint_scorer=scorer)

    first = controller.observe(
        epoch=1,
        metrics={"epoch": 1, "val_data_loss": 0.8},
        state_factory=lambda: pytest.fail("warm-up must not capture state"),
    )
    second = controller.observe(
        epoch=2,
        metrics={"epoch": 2, "val_data_loss": 0.6},
        state_factory=lambda: _state(2),
    )

    assert first.opportunity is False
    assert second == type(second)(opportunity=True, progress=True)
    assert controller.finalize().epoch == 2


def test_disabled_selects_online_without_candidate_pool():
    controller = CheckpointController(mode="disabled", checkpoint_scorer="val_loss")
    for epoch, loss in enumerate([0.7, 0.4, 0.5], start=1):
        controller.observe(
            epoch=epoch,
            metrics={"epoch": epoch, "val_data_loss": loss},
            state_factory=lambda epoch=epoch: _state(epoch),
        )
    selected = controller.finalize()
    assert selected.epoch == 2
    assert controller.candidates() == []


def test_interval_counts_only_eligible_opportunities_and_deduplicates_union():
    controller = CheckpointController(
        mode="interval",
        checkpoint_scorer="clean_train_loss",
        clean_train_interval=2,
        candidate_generators=(
            CandidateGenerator(
                CheckpointScorer(
                    lambda metrics, history: -require_metric(
                        metrics, "val_data_loss"
                    ),
                    name="loss",
                ),
                top_k=2,
            ),
            CandidateGenerator(
                CheckpointScorer(
                    lambda metrics, history: require_metric(
                        metrics, "val_roc_auc"
                    ),
                    name="auc",
                ),
                top_k=2,
            ),
        ),
    )
    observations = []
    for epoch, (loss, auc) in enumerate(
        [(0.9, 0.5), (0.8, 0.6), (0.7, 0.55), (0.75, 0.7)], start=1
    ):
        observations.append(
            controller.observe(
                epoch=epoch,
                metrics={
                    "epoch": epoch,
                    "val_data_loss": loss,
                    "val_roc_auc": auc,
                    **(
                        {"clean_train_data_loss": loss + 0.1}
                        if epoch % 2 == 0
                        else {}
                    ),
                },
                state_factory=lambda epoch=epoch: _state(epoch),
            )
        )
    assert [observation.opportunity for observation in observations] == [
        False,
        True,
        False,
        True,
    ]
    candidates = controller.candidates()
    assert len(candidates) <= 4
    assert len({candidate.epoch for candidate in candidates}) == len(candidates)
    assert controller.finalize().epoch == 4


def test_deferred_enriches_only_bounded_union():
    controller = CheckpointController(
        mode="deferred_candidates",
        checkpoint_scorer="clean_train_loss",
        candidate_generators=[
            {"name": "loss", "scorer": "val_data_loss", "top_k": 2},
            {"name": "auc", "scorer": "val_roc_auc", "top_k": 2},
        ],
    )
    for epoch in range(1, 8):
        controller.observe(
            epoch=epoch,
            metrics={
                "epoch": epoch,
                "val_data_loss": 1.0 / epoch,
                "val_roc_auc": 0.5 + 0.01 * epoch,
            },
            state_factory=lambda epoch=epoch: _state(epoch),
        )
    evaluated = []

    def enrich(candidate):
        evaluated.append(candidate.epoch)
        return {"clean_train_data_loss": abs(candidate.epoch - 6)}

    selected = controller.finalize(enrich_candidate=enrich)
    assert len(evaluated) <= 4
    assert evaluated == sorted(set(evaluated))
    assert selected.epoch == 6


class _ModeProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(0.8)
        self.linear = nn.Linear(2, 2)
        self.forward_contexts = []

    def forward(self, x):
        self.forward_contexts.append((self.training, torch.is_grad_enabled()))
        return self.linear(self.dropout(self.batch_norm(x)))


class _TinyClassifier(TorchEEGClassifier):
    model_label = "Tiny"

    def __init__(
        self,
        *,
        epochs=3,
        validation_split=0.5,
        early_stopping_patience=None,
        clean_train_mode="disabled",
        checkpoint_scorer="val_loss",
        candidate_generators=None,
        clean_train_interval=1,
        save_selected_checkpoint=False,
        seed=7,
    ):
        self._init_torch_classifier(
            epochs=epochs,
            batch_size=2,
            learning_rate=0.01,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            clean_train_mode=clean_train_mode,
            checkpoint_scorer=checkpoint_scorer,
            candidate_generators=candidate_generators,
            clean_train_interval=clean_train_interval,
            save_selected_checkpoint=save_selected_checkpoint,
            device="cpu",
            seed=seed,
        )

    def _prepare_features(self, X, *, fit, train_idx=None):
        return X[:, 0, :2]

    def _build_model_from_features(self, features, n_classes, **kwargs):
        return _ModeProbe()


class _ConstantEvalTinyClassifier(_TinyClassifier):
    def _evaluate_loader(self, loader, criterion, n_classes, **kwargs):
        return LoaderEvaluation(
            data_loss=0.5,
            selector_regularization=0.0,
            accuracy=0.5,
            roc_auc=0.5,
            selector_summary=[],
        )


def _tiny_data():
    X = np.array(
        [
            [[-2.0, -1.0]],
            [[-1.0, -2.0]],
            [[1.0, 2.0]],
            [[2.0, 1.0]],
            [[-1.5, -0.5]],
            [[1.5, 0.5]],
            [[-0.5, -1.5]],
            [[0.5, 1.5]],
        ],
        dtype=np.float32,
    )
    return X, np.array([0, 0, 1, 1, 0, 1, 0, 1])


def test_clean_evaluation_is_deterministic_and_restores_model_state_and_mode():
    estimator = _TinyClassifier()
    estimator.device_ = torch.device("cpu")
    estimator.model_ = _ModeProbe()
    estimator.model_.train()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0, 2.0], [2.0, 1.0], [0.5, 1.5]]),
            torch.tensor([0, 1, 0]),
        ),
        batch_size=2,
        shuffle=False,
    )
    before = deepcopy(estimator.model_.state_dict())
    first = estimator._evaluate_loader(loader, nn.CrossEntropyLoss(), 2)
    second = estimator._evaluate_loader(loader, nn.CrossEntropyLoss(), 2)
    assert estimator.model_.training is True
    assert set(estimator.model_.forward_contexts) == {(False, False)}
    assert first == second
    for name, value in before.items():
        assert torch.equal(value, estimator.model_.state_dict()[name])


def test_interval_fit_uses_named_custom_partial_with_clean_metrics(monkeypatch):
    events = []

    def capture_event(logger, category, message, **kwargs):
        events.append((category, message, kwargs))

    monkeypatch.setattr(
        common_module,
        "is_experiment_logging_configured",
        lambda: True,
    )
    monkeypatch.setattr(common_module, "log_event", capture_event)
    scorer = CheckpointScorer(
        function=partial(
            _clean_loss_with_validation_auc,
            auc_weight=0.25,
        ),
        name="clean-loss-plus-quarter-val-auc",
    )
    estimator = _TinyClassifier(
        epochs=4,
        clean_train_mode="interval",
        clean_train_interval=2,
        checkpoint_scorer=scorer,
    )
    X, y = _tiny_data()

    estimator.fit(X, y)

    assert estimator.best_epoch_ in {2, 4}
    assert len(estimator.clean_train_loss_history_) == 2
    assert estimator.selected_checkpoint_metrics_["clean_train_data_loss"] is not None
    assert estimator.fit_summary_["checkpoint_policy"]["checkpoint_scorer"] == {
        "name": "clean-loss-plus-quarter-val-auc",
        "min_delta": 0.0,
    }
    assert estimator.fit_summary_["selected_score"] is not None
    fit_message = next(
        message
        for category, message, _ in events
        if category == common_module.EventCategory.FIT_SUMMARY
    )
    assert "scorer=clean-loss-plus-quarter-val-auc" in fit_message
    assert "score=" in fit_message
    assert "val_data_loss=" in fit_message
    assert "val_roc_auc=" in fit_message
    assert "clean_train_data_loss=" in fit_message
    assert "clean_train_roc_auc=" in fit_message


def test_named_partial_scorer_survives_sklearn_clone_and_serialization():
    scorer = CheckpointScorer(
        function=partial(
            _clean_loss_with_validation_auc,
            auc_weight=0.25,
        ),
        name="clean-loss-plus-quarter-val-auc",
    )

    cloned = clone(_TinyClassifier(checkpoint_scorer=scorer))
    round_tripped = loads(dumps(cloned.checkpoint_scorer))

    assert cloned.checkpoint_scorer.name == scorer.name
    assert round_tripped.score(
        {"clean_train_data_loss": 0.4, "val_roc_auc": 0.8},
    ) == pytest.approx(-0.2)


def test_fit_discards_unrequested_optimizer_state_and_emits_exact_summary():
    estimator = _TinyClassifier(epochs=2)
    X, y = _tiny_data()
    estimator.fit(
        X,
        y,
        fit_context={
            "run_id": "run-1",
            "fit_id": "fit-1",
            "fit_role": "final_refit",
            "pipeline": "tiny",
            "evaluation_type": "CrossSession",
            "dataset": "fake",
            "subject": 1,
            "outer_session": "1test",
            "outer_fold": 0,
        },
    )
    assert estimator.best_epoch_ in {1, 2}
    assert estimator.optimizer_ is None
    assert estimator.alpha_optimizer_ is None
    assert estimator.selected_training_state_ is None
    assert estimator.fit_summary_["fit_id"] == "fit-1"
    assert estimator.fit_summary_["is_final_refit"] is True
    assert estimator.fit_summary_["selected_epoch"] == estimator.best_epoch_
    assert estimator.fit_summary_["stop_reason"] == "max_epochs"


def test_interval_early_stopping_counts_selection_opportunities():
    estimator = _ConstantEvalTinyClassifier(
        epochs=8,
        clean_train_mode="interval",
        clean_train_interval=2,
        early_stopping_patience=1,
    )
    X, y = _tiny_data()

    estimator.fit(X, y)

    assert estimator.epochs_completed_ == 4
    assert estimator.best_epoch_ == 2
    assert estimator.stop_reason_ == "early_stopping"


def test_selected_bundle_is_final_refit_only_and_opt_in(tmp_path):
    X, y = _tiny_data()
    inner = _ConstantEvalTinyClassifier(
        epochs=1,
        save_selected_checkpoint=True,
    )
    inner.fit(
        X,
        y,
        fit_context={
            "fit_id": "inner",
            "fit_role": "inner_candidate",
            "artifact_root": tmp_path,
        },
    )
    assert inner.selected_checkpoint_bundle_ is None
    assert not (tmp_path / "inner").exists()

    final = _ConstantEvalTinyClassifier(
        epochs=1,
        save_selected_checkpoint=True,
    )
    final.fit(
        X,
        y,
        fit_context={
            "fit_id": "final",
            "fit_role": "final_refit",
            "artifact_root": tmp_path,
        },
    )
    assert (tmp_path / "final" / "selected.pt").is_file()
    assert (tmp_path / "final" / "selected.json").is_file()
    payload, _ = load_checkpoint_bundle(tmp_path / "final")
    assert payload["training_state"]["optimizer_state"] is not None
    assert final.optimizer_ is None
    assert final.selected_training_state_ is None


def test_checkpoint_bundle_completion_and_loading(tmp_path):
    bundle = save_checkpoint_bundle(
        artifact_root=tmp_path,
        fit_id="fit-a",
        state=_state(3),
        metadata={"run_id": "run-a", "pipeline": "tiny"},
    )
    payload, metadata = load_checkpoint_bundle(bundle)
    assert payload["training_state"]["epoch"] == 3
    assert metadata["run_id"] == "run-a"

    (bundle / "selected.json").unlink()
    with pytest.raises(ValueError, match="completion marker"):
        load_checkpoint_bundle(bundle)
