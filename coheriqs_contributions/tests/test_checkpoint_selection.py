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
    CandidateSource,
    CheckpointController,
    CheckpointReporting,
    CheckpointScorer,
    PredictionMetric,
    ScoreRanking,
    TrainingState,
    history_after_epoch,
    history_before_epoch,
    history_except_epoch,
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
def test_all_neural_estimators_expose_named_checkpoint_configuration(classifier):
    estimator = classifier()
    params = estimator.get_params()
    assert params["clean_train_mode"] == "disabled"
    assert params["final_checkpoint_score"] == "val_loss"
    assert params["checkpoint_scores"] == ()
    assert params["clean_train_scores"] == ()
    assert params["candidate_sources"] == ()
    assert params["prediction_metrics"] == ()
    assert params["checkpoint_reporting"] is None
    assert params["clean_train_interval"] == 1
    assert params["save_selected_checkpoint"] is False
    assert clone(estimator).get_params()["final_checkpoint_score"] == "val_loss"


def _recent_score(metrics, history, *, window):
    prior = recent_history(history, window - 1)
    values = select_metric(prior, "val_data_loss")
    if len(values) != window - 1 or any(value is None for value in values):
        return None
    return -sum((*values, require_metric(metrics, "val_data_loss")))


def _clean_loss_plus_auc(metrics, history, *, auc_weight):
    return (
        -require_metric(metrics, "clean_train_data_loss")
        + auc_weight * require_metric(metrics, "val_roc_auc")
    )


def _negative_val_loss(metrics, history):
    return -require_metric(metrics, "val_data_loss")


def _val_auc(metrics, history):
    return require_metric(metrics, "val_roc_auc")


def _brier(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities)
    one_hot = np.eye(probabilities.shape[1])[y_true]
    return np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))


def test_scorer_history_helpers_warmup_and_future_views():
    scorer = CheckpointScorer(
        partial(_recent_score, window=3),
        name="recent-loss",
    )
    records = [
        {"epoch": 1, "val_data_loss": 4.0},
        {"epoch": 2, "val_data_loss": 2.0},
        {"epoch": 3, "val_data_loss": 1.0},
    ]
    assert scorer.score(records[1], records[:1]) is None
    assert scorer.score(records[2], records[:2]) == pytest.approx(-7.0)
    assert [record["epoch"] for record in history_before_epoch(records, 2)] == [1]
    assert [record["epoch"] for record in history_after_epoch(records, 2)] == [3]
    assert [record["epoch"] for record in history_except_epoch(records, 2)] == [1, 3]
    assert select_metric(
        [{"val_data_loss": 4.0}, {}, {"val_data_loss": np.nan}],
        "val_data_loss",
    ) == (4.0, None, None)


def test_registry_and_mode_validation_is_explicit():
    with pytest.raises(ValueError, match="reserved or duplicated"):
        CheckpointController(
            checkpoint_scores=(
                CheckpointScorer(_negative_val_loss, name="val_loss"),
            )
        )
    with pytest.raises(ValueError, match="Unknown checkpoint score"):
        CheckpointController(final_checkpoint_score="missing")
    with pytest.raises(ValueError, match="requires an explicit ScoreRanking"):
        CheckpointController(
            mode="deferred_candidates",
            final_checkpoint_score="clean_train_loss",
            clean_train_scores=("clean_train_loss",),
            candidate_sources=(CandidateSource("val_loss", 1),),
        )
    reporting = CheckpointReporting(
        score_rankings=(ScoreRanking("val_loss", top_k=1),)
    )
    with pytest.raises(ValueError, match="exceeds ScoreRanking"):
        CheckpointController(
            mode="deferred_candidates",
            final_checkpoint_score="clean_train_loss",
            clean_train_scores=("clean_train_loss",),
            candidate_sources=(CandidateSource("val_loss", 2),),
            reporting=reporting,
        )
    with pytest.raises(ValueError, match="not valid in disabled"):
        CheckpointController(
            candidate_sources=(CandidateSource("val_loss", 1),),
            reporting=reporting,
        )
    with pytest.raises(ValueError, match="cannot be clean-gated"):
        CheckpointController(
            final_checkpoint_score="val_loss",
            clean_train_scores=("val_loss",),
        )
    with pytest.raises(ValueError, match="must be in clean_train_scores"):
        CheckpointController(mode="interval")


def test_disabled_keeps_clean_diagnostics_dormant_and_ranks_other_scores():
    reporting = CheckpointReporting(
        selection_runner_ups=2,
        score_rankings=(
            ScoreRanking("val_roc_auc", top_k=2, show_each_epoch=True),
            ScoreRanking("custom-val-loss", top_k=2),
            ScoreRanking("clean_train_roc_auc", top_k=2),
        ),
    )
    controller = CheckpointController(
        checkpoint_scores=(
            CheckpointScorer(_negative_val_loss, "custom-val-loss"),
        ),
        clean_train_scores=("clean_train_roc_auc",),
        reporting=reporting,
    )
    state_calls = []
    for epoch, (loss, auc) in enumerate(
        [(0.5, 0.50), (0.6, 0.80), (0.7, 0.70)],
        start=1,
    ):
        observation = controller.observe(
            epoch=epoch,
            metrics={
                "epoch": epoch,
                "val_data_loss": loss,
                "val_roc_auc": auc,
            },
            state_factory=lambda epoch=epoch: (
                state_calls.append(epoch) or _state(epoch)
            ),
        )
        assert "val_roc_auc" in observation.evaluated_scores
        assert "custom-val-loss" in observation.evaluated_scores
        assert "clean_train_roc_auc" not in observation.evaluated_scores
    assert controller.finalize().epoch == 1
    assert state_calls == [1]
    final_ranking = controller.final_ranking_summary()
    assert [item["epoch"] for item in final_ranking] == [1, 2, 3]
    assert "val_roc_auc" in final_ranking[0]["scores"]
    assert "clean_train_roc_auc" in controller.policy_summary()["inactive_scores"]


def test_default_epoch_event_includes_authoritative_selection_ranking():
    controller = CheckpointController()
    observation = controller.observe(
        epoch=1,
        metrics={"val_data_loss": 0.5},
        state_factory=lambda: _state(1),
    )

    event_data = controller.epoch_event_data(
        metrics={"epoch": 1, "val_data_loss": 0.5},
        observation=observation,
    )

    assert event_data["checkpoint_rankings"] == {}
    assert event_data["checkpoint_selection_ranking"] == {
        "scorer": "val_loss",
        "score": -0.5,
        "rank": 1,
        "top_k": 1,
        "accepted": True,
        "retained": True,
        "selected": True,
        "became_selected": True,
    }


def test_final_score_show_at_end_remains_user_controlled():
    controller = CheckpointController(
        reporting=CheckpointReporting(
            score_rankings=(
                ScoreRanking("val_loss", 1, show_at_end=True),
            )
        )
    )
    controller.observe(
        epoch=1,
        metrics={"val_data_loss": 0.5},
        state_factory=lambda: _state(1),
    )
    controller.finalize()

    diagnostics = controller.score_rankings_summary(only_show_at_end=True)
    assert diagnostics["val_loss"][0]["selected"] is True


def test_exact_ties_prefer_earlier_epoch_and_none_never_captures_state():
    scorer = CheckpointScorer(partial(_recent_score, window=2), name="warm")
    controller = CheckpointController(
        final_checkpoint_score="warm",
        checkpoint_scores=(scorer,),
    )
    calls = []
    first = controller.observe(
        epoch=1,
        metrics={"epoch": 1, "val_data_loss": 0.5},
        state_factory=lambda: pytest.fail("warm-up must not retain state"),
    )
    second = controller.observe(
        epoch=2,
        metrics={"epoch": 2, "val_data_loss": 0.5},
        state_factory=lambda: calls.append(2) or _state(2),
    )
    third = controller.observe(
        epoch=3,
        metrics={"epoch": 3, "val_data_loss": 0.5},
        state_factory=lambda: pytest.fail("equal later epoch must not retain state"),
    )
    assert first.opportunity is False
    assert second.progress is True
    assert third.progress is False
    assert calls == [2]
    assert controller.finalize().epoch == 2


def test_interval_validation_scores_run_each_epoch_and_clean_scores_on_cadence():
    calls = {"val": [], "clean": []}

    def val_score(metrics, history):
        calls["val"].append(int(metrics["epoch"]))
        return -require_metric(metrics, "val_data_loss")

    def clean_score(metrics, history):
        calls["clean"].append(int(metrics["epoch"]))
        return -require_metric(metrics, "clean_train_data_loss")

    controller = CheckpointController(
        mode="interval",
        final_checkpoint_score="clean-score",
        checkpoint_scores=(
            CheckpointScorer(val_score, "val-score"),
            CheckpointScorer(clean_score, "clean-score"),
        ),
        clean_train_scores=("clean-score",),
        candidate_sources=(CandidateSource("clean-score", 1),),
        reporting=CheckpointReporting(
            score_rankings=(
                ScoreRanking("val-score", 4),
                ScoreRanking("clean-score", 2),
            )
        ),
        clean_train_interval=2,
    )
    observations = []
    for epoch in range(1, 5):
        clean = epoch % 2 == 0
        observations.append(
            controller.observe(
                epoch=epoch,
                metrics={
                    "epoch": epoch,
                    "val_data_loss": 1.0 / epoch,
                    **(
                        {"clean_train_data_loss": abs(epoch - 4)}
                        if clean
                        else {}
                    ),
                },
                clean_evaluated=clean,
                state_factory=lambda epoch=epoch: _state(epoch),
            )
        )
    assert calls == {"val": [1, 2, 3, 4], "clean": [2, 4]}
    assert [item.opportunity for item in observations] == [False, True, False, True]
    assert controller.finalize().epoch == 4
    assert [entry["epoch"] for entry in controller.final_ranking_summary()] == [4]


def test_diagnostic_rank_changes_never_count_as_semantic_progress():
    controller = CheckpointController(
        reporting=CheckpointReporting(
            score_rankings=(ScoreRanking("val_roc_auc", 3),)
        )
    )
    observations = []
    for epoch, auc in enumerate([0.5, 0.6, 0.7], start=1):
        observations.append(
            controller.observe(
                epoch=epoch,
                metrics={
                    "epoch": epoch,
                    "val_data_loss": 0.5,
                    "val_roc_auc": auc,
                },
                state_factory=lambda epoch=epoch: _state(epoch),
            )
        )
    assert [item.progress for item in observations] == [True, False, False]
    assert [
        item.rankings["val_roc_auc"]["accepted"] for item in observations
    ] == [True, True, True]


def test_deferred_enriches_union_before_full_history_clean_scoring():
    scoring_calls = []
    enriched = []

    def final_score(metrics, history):
        assert len(enriched) == 3
        epoch = int(metrics["epoch"])
        scoring_calls.append(
            {
                "epoch": epoch,
                "history_epochs": [int(record["epoch"]) for record in history],
                "clean_epochs": [
                    int(record["epoch"])
                    for record in history
                    if "clean_train_data_loss" in record
                ],
            }
        )
        return -require_metric(metrics, "clean_train_data_loss")

    controller = CheckpointController(
        mode="deferred_candidates",
        final_checkpoint_score="final-clean",
        checkpoint_scores=(CheckpointScorer(final_score, "final-clean"),),
        clean_train_scores=("final-clean",),
        candidate_sources=(CandidateSource("val_loss", 3),),
        reporting=CheckpointReporting(
            selection_runner_ups=2,
            score_rankings=(
                ScoreRanking("val_loss", 3),
            ),
        ),
    )
    for epoch in range(1, 6):
        controller.observe(
            epoch=epoch,
            metrics={"epoch": epoch, "val_data_loss": 1.0 / epoch},
            state_factory=lambda epoch=epoch: _state(epoch),
        )

    def enrich(candidate):
        enriched.append(candidate.epoch)
        return {"clean_train_data_loss": abs(candidate.epoch - 4)}

    selected = controller.finalize(enrich_candidate=enrich)
    assert enriched == [3, 4, 5]
    assert selected.epoch == 4
    assert {call["epoch"] for call in scoring_calls} == {3, 4, 5}
    for call in scoring_calls:
        assert call["epoch"] not in call["history_epochs"]
        assert set(call["history_epochs"]) == set(range(1, 6)) - {call["epoch"]}
        assert set(call["clean_epochs"]) == {3, 4, 5} - {call["epoch"]}
    assert [entry["epoch"] for entry in controller.final_ranking_summary()] == [4, 3, 5]
    assert controller.final_ranking_summary()[0]["candidate_sources"] == ["val_loss"]
    assert controller.candidate_nominations_summary()["val_loss"][1]["selected"]
    assert controller.final_ranking_scope == "enriched_candidate_union"


def test_end_diagnostic_provenance_excludes_displaced_candidate_entries():
    controller = CheckpointController(
        mode="deferred_candidates",
        final_checkpoint_score="clean_train_loss",
        clean_train_scores=("clean_train_loss",),
        candidate_sources=(CandidateSource("val_loss", 1),),
        reporting=CheckpointReporting(
            score_rankings=(
                ScoreRanking("val_loss", 3, show_at_end=True),
            )
        ),
    )
    for epoch, loss in enumerate((0.5, 0.4, 0.3), start=1):
        controller.observe(
            epoch=epoch,
            metrics={"val_data_loss": loss},
            state_factory=lambda epoch=epoch: _state(epoch),
        )

    controller.finalize(
        enrich_candidate=lambda candidate: {
            "clean_train_data_loss": float(candidate.epoch)
        }
    )
    entries = controller.score_rankings_summary(only_show_at_end=True)["val_loss"]

    assert [entry["epoch"] for entry in entries] == [3, 2, 1]
    assert entries[0]["candidate_sources"] == ["val_loss"]
    assert entries[0]["candidate_source_ranks"] == {"val_loss": 1}
    assert entries[1]["candidate_sources"] == []
    assert entries[1]["candidate_source_ranks"] == {}
    assert entries[2]["candidate_sources"] == []
    assert entries[2]["candidate_source_ranks"] == {}


def test_candidate_sources_share_epoch_state_and_release_prefixes():
    controller = CheckpointController(
        mode="deferred_candidates",
        final_checkpoint_score="clean_train_loss",
        clean_train_scores=("clean_train_loss",),
        candidate_sources=(
            CandidateSource("val_loss", 1),
            CandidateSource("val_roc_auc", 1),
        ),
        reporting=CheckpointReporting(
            score_rankings=(
                ScoreRanking("val_loss", 2),
                ScoreRanking("val_roc_auc", 2),
            )
        ),
    )
    state_calls = []
    observation = controller.observe(
        epoch=1,
        metrics={
            "epoch": 1,
            "val_data_loss": 0.5,
            "val_roc_auc": 0.8,
        },
        state_factory=lambda: state_calls.append(1) or _state(1),
    )
    assert state_calls == [1]
    assert set(observation.candidate_sources) == {"val_loss", "val_roc_auc"}
    assert len(controller.candidates()) == 1
    controller.finalize(
        enrich_candidate=lambda candidate: {"clean_train_data_loss": 0.4}
    )
    controller.release_unselected_candidate_states()
    assert controller.candidates() == []


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
        final_checkpoint_score="val_loss",
        checkpoint_scores=(),
        clean_train_scores=(),
        candidate_sources=(),
        prediction_metrics=(),
        checkpoint_reporting=None,
        clean_train_interval=1,
        save_selected_checkpoint=False,
        seed=7,
        verbose=0,
    ):
        self._init_torch_classifier(
            epochs=epochs,
            batch_size=2,
            learning_rate=0.01,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            clean_train_mode=clean_train_mode,
            final_checkpoint_score=final_checkpoint_score,
            checkpoint_scores=checkpoint_scores,
            clean_train_scores=clean_train_scores,
            candidate_sources=candidate_sources,
            prediction_metrics=prediction_metrics,
            checkpoint_reporting=checkpoint_reporting,
            clean_train_interval=clean_train_interval,
            save_selected_checkpoint=save_selected_checkpoint,
            device="cpu",
            seed=seed,
            verbose=verbose,
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


def test_prediction_metric_uses_existing_evaluation_predictions_once():
    calls = []

    def metric(y_true, probabilities):
        calls.append((np.asarray(y_true).copy(), np.asarray(probabilities).copy()))
        return _brier(y_true, probabilities)

    estimator = _TinyClassifier(
        prediction_metrics=(PredictionMetric(metric, "brier_score"),)
    )
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
    result = estimator._evaluate_loader(loader, nn.CrossEntropyLoss(), 2)
    assert len(calls) == 1
    assert result.prediction_metrics["brier_score"] is not None
    assert "val_brier_score" in result.checkpoint_metrics("val")
    assert "clean_train_brier_score" in result.checkpoint_metrics("clean_train")
    assert estimator.model_.training is True
    assert set(estimator.model_.forward_contexts) == {(False, False)}
    for name, value in before.items():
        assert torch.equal(value, estimator.model_.state_dict()[name])


@pytest.mark.parametrize("bad_value", ["bad", np.inf])
def test_prediction_metric_errors_name_the_metric(bad_value):
    metric = PredictionMetric(lambda y, p: bad_value, "broken_metric")
    with pytest.raises((TypeError, ValueError), match="broken_metric"):
        metric.evaluate(np.array([0]), np.array([[0.8, 0.2]]))


def test_prediction_metric_none_and_report_raw_name_validation():
    metric = PredictionMetric(lambda y, p: None, "optional")
    assert metric.evaluate(np.array([0]), np.array([[0.8, 0.2]])) is None
    with pytest.raises(ValueError, match="reserved"):
        _TinyClassifier(prediction_metrics=(PredictionMetric(_brier, "roc_auc"),))
    with pytest.raises(ValueError, match="Unknown show_epoch_metrics"):
        _TinyClassifier(
            checkpoint_reporting=CheckpointReporting(
                show_epoch_metrics=("val_missing",)
            )
        )


def test_interval_fit_logging_summary_and_named_partial(monkeypatch):
    events = []
    prediction_calls = []

    def capture_event(logger, category, message, **kwargs):
        events.append((category, message, kwargs))

    monkeypatch.setattr(common_module, "is_experiment_logging_configured", lambda: True)
    monkeypatch.setattr(common_module, "log_event", capture_event)
    scorer = CheckpointScorer(
        partial(_clean_loss_plus_auc, auc_weight=0.25),
        name="clean-loss-plus-auc",
    )

    def counted_brier(y_true, probabilities):
        prediction_calls.append(len(y_true))
        return _brier(y_true, probabilities)

    estimator = _TinyClassifier(
        epochs=4,
        clean_train_mode="interval",
        clean_train_interval=2,
        final_checkpoint_score=scorer.name,
        checkpoint_scores=(scorer,),
        clean_train_scores=(scorer.name,),
        prediction_metrics=(PredictionMetric(counted_brier, "brier_score"),),
        checkpoint_reporting=CheckpointReporting(
            selection_runner_ups=1,
            score_rankings=(
                ScoreRanking(
                    "val_roc_auc",
                    3,
                    show_each_epoch=True,
                    show_at_end=True,
                ),
                ScoreRanking(scorer.name, 2, show_each_epoch=True),
            ),
            show_epoch_metrics=("val_brier_score",),
        ),
    )
    X, y = _tiny_data()
    estimator.fit(X, y)

    assert estimator.best_epoch_ in {2, 4}
    assert len(estimator.clean_train_loss_history_) == 2
    assert len(prediction_calls) == 6
    assert estimator.fit_summary_["schema_version"] == 2
    assert estimator.fit_summary_["selection_ranking_scope"] == (
        "interval_eligible_epochs"
    )
    assert estimator.fit_summary_["selected_scores"][scorer.name] is not None
    epoch_event = next(
        kwargs
        for category, _, kwargs in events
        if category == common_module.EventCategory.EPOCH
        and kwargs["epoch"] == 2
    )
    assert "checkpoint_metrics" in epoch_event["data"]
    assert "checkpoint_scores" in epoch_event["data"]
    assert "checkpoint_rankings" in epoch_event["data"]
    assert "checkpoint_selection_ranking" in epoch_event["data"]
    assert epoch_event["data"]["checkpoint_selection_ranking"]["scorer"] == (
        scorer.name
    )
    assert any(
        "checkpoint_ranks:" in message
        for category, message, _ in events
        if category == common_module.EventCategory.EPOCH
    )
    status_messages = [
        message
        for category, message, _ in events
        if category == common_module.EventCategory.STATUS
    ]
    assert any("final runner-up rank=#2" in message for message in status_messages)
    assert any(
        "diagnostic ranking=val_roc_auc" in message
        for message in status_messages
    )
    assert not any(
        "diagnostic ranking=clean-loss-plus-auc" in message
        for message in status_messages
    )


def test_named_configuration_survives_sklearn_clone_and_pickle():
    scorer = CheckpointScorer(
        partial(_clean_loss_plus_auc, auc_weight=0.25),
        name="clean-loss-plus-auc",
    )
    metric = PredictionMetric(_brier, "brier_score")
    reporting = CheckpointReporting(
        score_rankings=(ScoreRanking("val_loss", 2),)
    )
    estimator = _TinyClassifier(
        checkpoint_scores=(scorer,),
        prediction_metrics=(metric,),
        checkpoint_reporting=reporting,
    )
    cloned = clone(estimator)
    round_tripped = loads(dumps(cloned))
    assert cloned.checkpoint_scores[0].name == scorer.name
    assert round_tripped.prediction_metrics[0].name == metric.name
    assert round_tripped.checkpoint_reporting == reporting


def test_result_digest_ignores_reporting_but_not_semantic_policy():
    pytest.importorskip("mne")
    from moabb.analysis.results import get_digest

    first = _TinyClassifier(
        checkpoint_reporting=CheckpointReporting(selection_runner_ups=1)
    )
    changed_reporting = _TinyClassifier(
        checkpoint_reporting=CheckpointReporting(selection_runner_ups=5)
    )
    changed_semantics = _TinyClassifier(final_checkpoint_score="val_roc_auc")
    assert get_digest(first) == get_digest(changed_reporting)
    assert get_digest(first) != get_digest(changed_semantics)


def test_cwt_estimators_merge_operational_hash_exclusions():
    assert WCTEvidenceGNNClassifier._moabb_result_ignored_params == {
        "checkpoint_reporting": None,
        "wct_cache_root": None,
    }


def test_interval_early_stopping_counts_only_final_rank_opportunities():
    estimator = _ConstantEvalTinyClassifier(
        epochs=8,
        clean_train_mode="interval",
        clean_train_scores=("val_loss",),
        clean_train_interval=2,
        early_stopping_patience=1,
    )
    X, y = _tiny_data()
    estimator.fit(X, y)
    assert estimator.epochs_completed_ == 4
    assert estimator.best_epoch_ == 2
    assert estimator.stop_reason_ == "early_stopping"


def test_deferred_early_stopping_counts_candidate_prefix_progress():
    estimator = _ConstantEvalTinyClassifier(
        epochs=8,
        clean_train_mode="deferred_candidates",
        final_checkpoint_score="clean_train_loss",
        clean_train_scores=("clean_train_loss",),
        candidate_sources=(CandidateSource("val_loss", 2),),
        checkpoint_reporting=CheckpointReporting(
            score_rankings=(ScoreRanking("val_loss", 2),)
        ),
        early_stopping_patience=1,
    )
    X, y = _tiny_data()
    estimator.fit(X, y)

    assert estimator.epochs_completed_ == 3
    assert estimator.best_epoch_ == 1
    assert estimator.stop_reason_ == "early_stopping"
    assert [
        entry["epoch"]
        for entry in estimator.fit_summary_["candidate_nominations"]["val_loss"]
    ] == [1, 2]


def test_fit_discards_unrequested_optimizer_state_and_emits_schema_v2():
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
        },
    )
    assert estimator.optimizer_ is None
    assert estimator.alpha_optimizer_ is None
    assert estimator.selected_training_state_ is None
    assert estimator.fit_summary_["schema_version"] == 2
    assert estimator.fit_summary_["selected_epoch"] == estimator.best_epoch_
    assert estimator.fit_summary_["selection_ranking"][0]["selected"] is True


def test_selected_bundle_is_final_refit_only_and_selected_only(tmp_path):
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
    assert sorted(path.name for path in (tmp_path / "final").iterdir()) == [
        "selected.json",
        "selected.pt",
    ]
    payload, metadata = load_checkpoint_bundle(tmp_path / "final")
    assert payload["training_state"]["optimizer_state"] is not None
    assert "selected_scores" in metadata


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
