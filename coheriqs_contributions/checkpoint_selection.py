"""Checkpoint scoring, bounded candidate retention, and continuation bundles."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence
from uuid import uuid4

import torch


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_METADATA_VERSION = 1
MetricValue = float | int | None
MetricRecord = Mapping[str, MetricValue]
MetricHistory = Sequence[MetricRecord]
ScoreFunction = Callable[[MetricRecord, MetricHistory], float | None]
PredictionFunction = Callable[[object, object], float | None]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _optional_metric_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def require_metric(metrics: MetricRecord, name: str) -> float:
    """Select one current metric and require a finite numeric value."""

    value = _optional_metric_value(metrics.get(name))
    if value is None:
        raise ValueError(
            f"Required checkpoint metric '{name}' is missing or non-finite."
        )
    return value


def select_metric(
    records: MetricHistory,
    name: str,
) -> tuple[float | None, ...]:
    """Select one aligned metric history, preserving unavailable values."""

    return tuple(_optional_metric_value(record.get(name)) for record in records)


def recent_history(
    history: MetricHistory,
    window: int,
) -> tuple[MetricRecord, ...]:
    """Return up to ``window`` prior records in chronological order."""

    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("window must be a non-negative integer.")
    if window == 0:
        return ()
    return tuple(history[-window:])


def history_before_epoch(
    history: MetricHistory,
    epoch: int,
) -> tuple[MetricRecord, ...]:
    """Return records strictly before ``epoch`` in chronological order."""

    return tuple(
        record
        for record in history
        if int(require_metric(record, "epoch")) < int(epoch)
    )


def history_except_epoch(
    history: MetricHistory,
    epoch: int,
) -> tuple[MetricRecord, ...]:
    """Return the complete trajectory except for the current checkpoint."""

    return tuple(
        record
        for record in history
        if int(require_metric(record, "epoch")) != int(epoch)
    )


def history_after_epoch(
    history: MetricHistory,
    epoch: int,
) -> tuple[MetricRecord, ...]:
    """Return records strictly after ``epoch`` in chronological order."""

    return tuple(
        record
        for record in history
        if int(require_metric(record, "epoch")) > int(epoch)
    )


@dataclass(frozen=True)
class CheckpointScorer:
    """A named custom checkpoint utility; larger finite scores are better."""

    function: ScoreFunction
    name: str

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")

    def score(
        self,
        metrics: MetricRecord,
        history: MetricHistory = (),
    ) -> float | None:
        score = self.function(metrics, history)
        if score is None:
            return None
        try:
            numeric = float(score)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Checkpoint scorer '{self.name}' must return a number or None."
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"Checkpoint scorer '{self.name}' returned a non-finite score."
            )
        return numeric


@dataclass(frozen=True)
class PredictionMetric:
    """A named scalar calculated from encoded targets and class probabilities."""

    function: PredictionFunction
    name: str

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")

    def evaluate(self, y_true: object, probabilities: object) -> float | None:
        value = self.function(y_true, probabilities)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Prediction metric '{self.name}' must return a number or None."
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"Prediction metric '{self.name}' returned a non-finite value."
            )
        return numeric


@dataclass(frozen=True)
class ScoreRanking:
    """A lightweight top-K ranking for one registered checkpoint score."""

    scorer: str
    top_k: int = 1
    show_each_epoch: bool = False
    show_at_end: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.scorer, "scorer")
        _validate_top_k(self.top_k)
        if not isinstance(self.show_each_epoch, bool):
            raise TypeError("show_each_epoch must be a boolean.")
        if not isinstance(self.show_at_end, bool):
            raise TypeError("show_at_end must be a boolean.")


@dataclass(frozen=True)
class CandidateSource:
    """A state-retaining prefix projected from a configured score ranking."""

    scorer: str
    top_k: int = 1

    def __post_init__(self) -> None:
        _validate_name(self.scorer, "scorer")
        _validate_top_k(self.top_k)


@dataclass(frozen=True)
class CheckpointReporting:
    """Operational checkpoint diagnostics; excluded from result identity."""

    selection_runner_ups: int = 0
    score_rankings: tuple[ScoreRanking, ...] = ()
    show_epoch_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.selection_runner_ups, bool)
            or not isinstance(self.selection_runner_ups, int)
            or self.selection_runner_ups < 0
        ):
            raise ValueError("selection_runner_ups must be a non-negative integer.")
        rankings = tuple(self.score_rankings)
        metrics = tuple(self.show_epoch_metrics)
        object.__setattr__(self, "score_rankings", rankings)
        object.__setattr__(self, "show_epoch_metrics", metrics)
        if not all(isinstance(ranking, ScoreRanking) for ranking in rankings):
            raise TypeError("score_rankings must contain ScoreRanking objects.")
        names = [ranking.scorer for ranking in rankings]
        if len(names) != len(set(names)):
            raise ValueError("Only one ScoreRanking may be configured per scorer.")
        for name in metrics:
            _validate_name(name, "show_epoch_metrics entry")
        if len(metrics) != len(set(metrics)):
            raise ValueError("show_epoch_metrics names must be unique.")


def _validate_name(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")


def _validate_top_k(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("top_k must be a positive integer.")


@dataclass
class TrainingState:
    """Selected model state plus optional optimizer state for a saved bundle."""

    epoch: int
    optimizer_step_count: int
    model_state: dict
    optimizer_state: dict | None
    alpha_optimizer_state: dict | None
    learning_rates: dict[str, list[float]]
    selector_state: dict[str, object] = field(default_factory=dict)


@dataclass
class CheckpointCandidate:
    epoch: int
    metrics: dict[str, MetricValue]
    state: TrainingState
    score_values: dict[str, float | None] = field(default_factory=dict)
    candidate_sources: set[str] = field(default_factory=set)
    final_score: float | None = None


@dataclass
class RankingEntry:
    """Lightweight observation retained for reporting; contains no model state."""

    epoch: int
    score: float
    metrics: dict[str, MetricValue]
    score_values: dict[str, float | None] = field(default_factory=dict)
    candidate_sources: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class SelectionObservation:
    opportunity: bool
    progress: bool
    evaluated_scores: Mapping[str, float | None] = field(default_factory=dict)
    rankings: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    candidate_sources: tuple[str, ...] = ()


def _negative_val_data_loss(metrics: MetricRecord, history: MetricHistory) -> float:
    return -require_metric(metrics, "val_data_loss")


def _val_roc_auc(metrics: MetricRecord, history: MetricHistory) -> float:
    return require_metric(metrics, "val_roc_auc")


def _negative_clean_train_data_loss(
    metrics: MetricRecord,
    history: MetricHistory,
) -> float:
    return -require_metric(metrics, "clean_train_data_loss")


def _clean_train_roc_auc(metrics: MetricRecord, history: MetricHistory) -> float:
    return require_metric(metrics, "clean_train_roc_auc")


_SCORER_ALIASES = {
    "val_loss": CheckpointScorer(_negative_val_data_loss, name="val_loss"),
    "val_data_loss": CheckpointScorer(
        _negative_val_data_loss, name="val_data_loss"
    ),
    "val_roc_auc": CheckpointScorer(_val_roc_auc, name="val_roc_auc"),
    "clean_train_loss": CheckpointScorer(
        _negative_clean_train_data_loss, name="clean_train_loss"
    ),
    "clean_train_data_loss": CheckpointScorer(
        _negative_clean_train_data_loss, name="clean_train_data_loss"
    ),
    "clean_train_roc_auc": CheckpointScorer(
        _clean_train_roc_auc, name="clean_train_roc_auc"
    ),
}


def validate_selection_config(
    *,
    mode: str,
    interval: int,
) -> None:
    if mode not in {"disabled", "interval", "deferred_candidates"}:
        raise ValueError(
            "clean_train_mode must be 'disabled', 'interval', or "
            "'deferred_candidates'."
        )
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("clean_train_interval must be a positive integer.")


class CheckpointController:
    """One score stream feeding reporting, candidate retention, and selection."""

    def __init__(
        self,
        *,
        mode: str = "disabled",
        final_checkpoint_score: str = "val_loss",
        checkpoint_scores: Sequence[CheckpointScorer] = (),
        clean_train_scores: Sequence[str] = (),
        candidate_sources: Sequence[CandidateSource] = (),
        reporting: CheckpointReporting | None = None,
        clean_train_interval: int = 1,
    ) -> None:
        validate_selection_config(mode=mode, interval=clean_train_interval)
        _validate_name(final_checkpoint_score, "final_checkpoint_score")
        self.mode = mode
        self.final_score_name = final_checkpoint_score
        self.reporting = reporting or CheckpointReporting()
        if not isinstance(self.reporting, CheckpointReporting):
            raise TypeError("checkpoint_reporting must be CheckpointReporting or None.")
        self.scorers = dict(_SCORER_ALIASES)
        configured_scores = tuple(checkpoint_scores)
        for scorer in configured_scores:
            if not isinstance(scorer, CheckpointScorer):
                raise TypeError(
                    "checkpoint_scores must contain CheckpointScorer objects."
                )
            if scorer.name in self.scorers:
                raise ValueError(
                    f"Checkpoint score name '{scorer.name}' is reserved or duplicated."
                )
            self.scorers[scorer.name] = scorer
        clean_names = tuple(clean_train_scores)
        for name in clean_names:
            _validate_name(name, "clean_train_scores entry")
        self.clean_score_names = frozenset(clean_names)
        unknown_clean = self.clean_score_names.difference(self.scorers)
        if unknown_clean:
            raise ValueError(
                f"Unknown clean_train_scores: {sorted(unknown_clean)}."
            )
        self.candidate_sources = tuple(candidate_sources)
        if not all(
            isinstance(source, CandidateSource) for source in self.candidate_sources
        ):
            raise TypeError("candidate_sources must contain CandidateSource objects.")
        source_names = [source.scorer for source in self.candidate_sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("Only one CandidateSource may be configured per scorer.")
        ranking_configs = {
            ranking.scorer: ranking for ranking in self.reporting.score_rankings
        }
        referenced = {
            self.final_score_name,
            *source_names,
            *ranking_configs,
        }
        unknown = referenced.difference(self.scorers)
        if unknown:
            raise ValueError(f"Unknown checkpoint score names: {sorted(unknown)}.")
        for source in self.candidate_sources:
            ranking = ranking_configs.get(source.scorer)
            if ranking is None:
                raise ValueError(
                    f"CandidateSource '{source.scorer}' requires an explicit "
                    "ScoreRanking for the same scorer."
                )
            if source.top_k > ranking.top_k:
                raise ValueError(
                    f"CandidateSource '{source.scorer}' top_k={source.top_k} "
                    f"exceeds ScoreRanking top_k={ranking.top_k}."
                )
        if mode == "disabled":
            if self.candidate_sources:
                raise ValueError("candidate_sources are not valid in disabled mode.")
            if self.final_score_name in self.clean_score_names:
                raise ValueError(
                    "disabled mode final_checkpoint_score cannot be clean-gated."
                )
        elif mode == "interval":
            required_clean = {self.final_score_name, *source_names}
            missing = required_clean.difference(self.clean_score_names)
            if missing:
                raise ValueError(
                    "interval mode final and candidate-source scores must be in "
                    f"clean_train_scores; missing {sorted(missing)}."
                )
        else:
            if self.final_score_name not in self.clean_score_names:
                raise ValueError(
                    "deferred_candidates final_checkpoint_score must be clean-gated."
                )
            if not self.candidate_sources:
                raise ValueError(
                    "deferred_candidates requires at least one CandidateSource."
                )
            gated_sources = set(source_names).intersection(self.clean_score_names)
            if gated_sources:
                raise ValueError(
                    "deferred_candidates CandidateSource scores must not be "
                    f"clean-gated: {sorted(gated_sources)}."
                )
        self.clean_train_interval = clean_train_interval
        self.active_score_names = frozenset(
            {
                self.final_score_name,
                *(scorer.name for scorer in configured_scores),
                *source_names,
                *ranking_configs,
            }
        )
        self.history: list[dict[str, MetricValue]] = []
        self._score_rankings: dict[str, list[RankingEntry]] = {
            name: [] for name in ranking_configs
        }
        self._candidate_rankings: dict[
            str, list[tuple[float, CheckpointCandidate]]
        ] = {
            source.scorer: [] for source in self.candidate_sources
        }
        self._final_ranking: list[RankingEntry] = []
        self._final_capacity = 1 + self.reporting.selection_runner_ups
        self._selected: CheckpointCandidate | None = None
        self._ranking_configs = ranking_configs
        self._last_scores_by_epoch: dict[int, dict[str, float | None]] = {}

    @property
    def requires_interval_clean_evaluation(self) -> bool:
        return self.mode == "interval"

    def is_eligible_epoch(self, epoch: int) -> bool:
        if self.mode != "interval":
            return True
        return epoch % self.clean_train_interval == 0

    def observe(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, MetricValue],
        state_factory: Callable[[], TrainingState],
        clean_evaluated: bool = False,
    ) -> SelectionObservation:
        record = dict(metrics)
        record["epoch"] = int(epoch)
        prior_history = tuple(self.history)
        self.history.append(record)
        scores = self._evaluate_scores(
            record,
            prior_history,
            allow_clean=clean_evaluated,
        )
        self._last_scores_by_epoch[int(epoch)] = dict(scores)
        ranking_status = self._update_score_rankings(epoch, record, scores)

        state_cache: list[TrainingState] = []

        def shared_state() -> TrainingState:
            if not state_cache:
                state_cache.append(state_factory())
            return state_cache[0]

        final_score = scores.get(self.final_score_name)
        final_progress = False
        if self.mode in {"disabled", "interval"} and final_score is not None:
            self._insert_lightweight_ranking(
                self._final_ranking,
                entry=RankingEntry(
                    epoch=int(epoch),
                    score=final_score,
                    metrics=dict(record),
                    score_values=dict(scores),
                ),
                top_k=self._final_capacity,
            )
            winner = self._final_ranking[0]
            if self._selected is None or winner.epoch != self._selected.epoch:
                if winner.epoch != int(epoch):
                    raise RuntimeError(
                        "An existing runner-up cannot become a new online winner."
                    )
                self._selected = CheckpointCandidate(
                    epoch=int(epoch),
                    metrics=dict(record),
                    state=shared_state(),
                    score_values=dict(scores),
                    final_score=final_score,
                )
                final_progress = True

        retained_sources = []
        source_progress = False
        ready_sources = 0
        for source in self.candidate_sources:
            score = scores.get(source.scorer)
            if score is None:
                continue
            ready_sources += 1
            ranking = self._candidate_rankings[source.scorer]
            accepted = self._would_enter_state_ranking(
                ranking, score, epoch, source.top_k
            )
            if not accepted:
                continue
            candidate = CheckpointCandidate(
                epoch=epoch,
                metrics=record,
                state=shared_state(),
                score_values=dict(scores),
                candidate_sources={source.scorer},
            )
            self._insert_state_ranking(
                ranking,
                score=score,
                candidate=candidate,
                top_k=source.top_k,
            )
            retained_sources.append(source.scorer)
            source_progress = True
        if retained_sources:
            self._mark_candidate_sources(int(epoch), retained_sources)

        if self.mode == "deferred_candidates":
            opportunity = ready_sources > 0
            progress = source_progress
        else:
            opportunity = final_score is not None
            progress = final_progress
        return SelectionObservation(
            opportunity=opportunity,
            progress=progress,
            evaluated_scores=scores,
            rankings=ranking_status,
            candidate_sources=tuple(retained_sources),
        )

    def candidates(self) -> list[CheckpointCandidate]:
        by_epoch: dict[int, CheckpointCandidate] = {}
        for ranking in self._candidate_rankings.values():
            for _, candidate in ranking:
                existing = by_epoch.get(candidate.epoch)
                if existing is None:
                    by_epoch[candidate.epoch] = candidate
                else:
                    existing.score_values.update(candidate.score_values)
                    existing.candidate_sources.update(candidate.candidate_sources)
        return [by_epoch[epoch] for epoch in sorted(by_epoch)]

    def finalize(
        self,
        *,
        enrich_candidate: Callable[[CheckpointCandidate], Mapping[str, MetricValue]]
        | None = None,
    ) -> CheckpointCandidate:
        if self.mode in {"disabled", "interval"}:
            if self._selected is None:
                raise RuntimeError("No checkpoint was eligible for selection.")
            return self._selected

        candidates = self.candidates()
        if not candidates:
            raise RuntimeError("No checkpoint candidates were retained.")
        if self.mode == "deferred_candidates":
            if enrich_candidate is None:
                raise ValueError(
                    "deferred_candidates requires a clean metric evaluator."
                )
            for candidate in candidates:
                candidate.metrics.update(dict(enrich_candidate(candidate)))
                for history_record in self.history:
                    if int(require_metric(history_record, "epoch")) == candidate.epoch:
                        history_record.update(candidate.metrics)
                        break
        self._final_ranking.clear()
        self._selected = None
        for candidate in candidates:
            history = history_except_epoch(self.history, candidate.epoch)
            clean_scores = self._evaluate_scores(
                candidate.metrics,
                history,
                allow_clean=True,
                only_clean=True,
            )
            candidate.score_values.update(clean_scores)
            self._last_scores_by_epoch[candidate.epoch].update(clean_scores)
            self._update_score_rankings(
                candidate.epoch,
                candidate.metrics,
                clean_scores,
                only_clean=True,
                all_score_values=candidate.score_values,
            )
            score = clean_scores.get(self.final_score_name)
            if score is None:
                continue
            candidate.final_score = score
            self._insert_lightweight_ranking(
                self._final_ranking,
                entry=RankingEntry(
                    epoch=candidate.epoch,
                    score=score,
                    metrics=dict(candidate.metrics),
                    score_values=dict(candidate.score_values),
                    candidate_sources=set(candidate.candidate_sources),
                ),
                top_k=self._final_capacity,
            )
        if not self._final_ranking:
            raise RuntimeError(
                "No candidate was eligible for final checkpoint scoring."
            )
        selected_epoch = self._final_ranking[0].epoch
        self._selected = next(
            candidate for candidate in candidates if candidate.epoch == selected_epoch
        )
        return self._selected

    def _evaluate_scores(
        self,
        metrics: MetricRecord,
        history: MetricHistory,
        *,
        allow_clean: bool,
        only_clean: bool = False,
    ) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for name in sorted(self.active_score_names):
            is_clean = name in self.clean_score_names
            if only_clean and not is_clean:
                continue
            if is_clean and not allow_clean:
                continue
            values[name] = self.scorers[name].score(metrics, history)
        return values

    @staticmethod
    def _would_enter_state_ranking(
        ranking: Sequence[tuple[float, CheckpointCandidate]],
        score: float,
        epoch: int,
        top_k: int,
    ) -> bool:
        if len(ranking) < top_k:
            return True
        worst_score, worst = ranking[-1]
        return score > worst_score or (
            score == worst_score and int(epoch) < worst.epoch
        )

    @staticmethod
    def _insert_state_ranking(
        ranking: list[tuple[float, CheckpointCandidate]],
        *,
        score: float,
        candidate: CheckpointCandidate,
        top_k: int,
    ) -> None:
        ranking[:] = [
            item for item in ranking if item[1].epoch != candidate.epoch
        ]
        ranking.append((score, candidate))
        ranking.sort(key=lambda item: (-item[0], item[1].epoch))
        del ranking[top_k:]

    @staticmethod
    def _insert_lightweight_ranking(
        ranking: list[RankingEntry],
        *,
        entry: RankingEntry,
        top_k: int,
    ) -> None:
        ranking[:] = [item for item in ranking if item.epoch != entry.epoch]
        ranking.append(entry)
        ranking.sort(key=lambda item: (-item.score, item.epoch))
        del ranking[top_k:]

    def _update_score_rankings(
        self,
        epoch: int,
        metrics: Mapping[str, MetricValue],
        scores: Mapping[str, float | None],
        *,
        only_clean: bool = False,
        all_score_values: Mapping[str, float | None] | None = None,
    ) -> dict[str, Mapping[str, object]]:
        status: dict[str, Mapping[str, object]] = {}
        for name, config in self._ranking_configs.items():
            if only_clean and name not in self.clean_score_names:
                continue
            if name not in scores:
                continue
            score = scores[name]
            if score is None:
                status[name] = {
                    "score": None,
                    "rank": None,
                    "top_k": config.top_k,
                    "accepted": False,
                    "retained": False,
                }
                continue
            ranking = self._score_rankings[name]
            self._insert_lightweight_ranking(
                ranking,
                entry=RankingEntry(
                    epoch=int(epoch),
                    score=score,
                    metrics=dict(metrics),
                    score_values=dict(
                        scores if all_score_values is None else all_score_values
                    ),
                ),
                top_k=config.top_k,
            )
            retained_epochs = [item.epoch for item in ranking]
            rank = (
                retained_epochs.index(int(epoch)) + 1
                if int(epoch) in retained_epochs
                else None
            )
            status[name] = {
                "score": score,
                "rank": rank,
                "top_k": config.top_k,
                "accepted": rank is not None,
                "retained": rank is not None,
            }
        return status

    def final_ranking_summary(self) -> list[dict[str, object]]:
        return [
            self._ranking_entry_summary(entry, rank=rank)
            for rank, entry in enumerate(self._final_ranking, start=1)
        ]

    def score_rankings_summary(
        self,
        *,
        only_show_at_end: bool = False,
    ) -> dict[str, list[dict[str, object]]]:
        summaries = {}
        for name, ranking in self._score_rankings.items():
            config = self._ranking_configs[name]
            if only_show_at_end and not config.show_at_end:
                continue
            summaries[name] = [
                self._ranking_entry_summary(entry, rank=rank)
                for rank, entry in enumerate(ranking, start=1)
            ]
        return summaries

    def candidate_nominations_summary(self) -> dict[str, list[dict[str, object]]]:
        return {
            name: [
                {
                    "rank": rank,
                    "epoch": candidate.epoch,
                    "score": score,
                    "selected": (
                        self._selected is not None
                        and candidate.epoch == self._selected.epoch
                    ),
                }
                for rank, (score, candidate) in enumerate(ranking, start=1)
            ]
            for name, ranking in self._candidate_rankings.items()
        }

    @property
    def final_ranking_scope(self) -> str:
        return {
            "disabled": "all_validation_epochs",
            "interval": "interval_eligible_epochs",
            "deferred_candidates": "enriched_candidate_union",
        }[self.mode]

    def release_unselected_candidate_states(self) -> None:
        """Release candidate-prefix snapshots after summaries are materialized."""

        for ranking in self._candidate_rankings.values():
            ranking.clear()

    def format_epoch_rankings(self, observation: SelectionObservation) -> str:
        parts = []
        for ranking in self.reporting.score_rankings:
            if not ranking.show_each_epoch:
                continue
            status = observation.rankings.get(ranking.scorer)
            if status is None:
                continue
            score = status["score"]
            if score is None:
                continue
            score_text = f"{float(score):.6f}"
            rank = status["rank"]
            if rank is None:
                parts.append(f"{ranking.scorer}={score_text}")
            else:
                parts.append(
                    f"{ranking.scorer}={score_text} current#{rank}/{ranking.top_k}"
                )
        return "" if not parts else " checkpoint_ranks: " + ", ".join(parts)

    def format_epoch_metrics(self, metrics: Mapping[str, MetricValue]) -> str:
        parts = []
        for name in self.reporting.show_epoch_metrics:
            if name not in metrics:
                continue
            value = metrics[name]
            text = "n/a" if value is None else f"{float(value):.6f}"
            parts.append(f"{name}={text}")
        return "" if not parts else " checkpoint_metrics: " + " ".join(parts)

    def epoch_event_data(
        self,
        *,
        metrics: Mapping[str, MetricValue],
        observation: SelectionObservation,
    ) -> dict[str, object]:
        return {
            "checkpoint_metrics": dict(metrics),
            "checkpoint_scores": dict(observation.evaluated_scores),
            "checkpoint_rankings": {
                name: dict(status)
                for name, status in observation.rankings.items()
            },
            "candidate_sources_retained": list(observation.candidate_sources),
        }

    def _candidate_sources_for_epoch(self, epoch: int) -> set[str]:
        sources: set[str] = set()
        for name, ranking in self._candidate_rankings.items():
            if any(candidate.epoch == epoch for _, candidate in ranking):
                sources.add(name)
        return sources

    def _ranking_entry_summary(
        self,
        entry: RankingEntry,
        *,
        rank: int,
    ) -> dict[str, object]:
        sources = set(entry.candidate_sources)
        sources.update(self._candidate_sources_for_epoch(entry.epoch))
        return {
            "rank": rank,
            "epoch": entry.epoch,
            "score": entry.score,
            "metrics": dict(entry.metrics),
            "scores": dict(entry.score_values),
            "selected": (
                self._selected is not None and entry.epoch == self._selected.epoch
            ),
            "candidate_sources": sorted(sources),
            "candidate_source_ranks": self._candidate_source_positions(entry.epoch),
        }

    def _candidate_source_positions(self, epoch: int) -> dict[str, int]:
        positions = {}
        for name, ranking in self._candidate_rankings.items():
            for rank, (_, candidate) in enumerate(ranking, start=1):
                if candidate.epoch == epoch:
                    positions[name] = rank
                    break
        return positions

    def _mark_candidate_sources(
        self,
        epoch: int,
        source_names: Sequence[str],
    ) -> None:
        for ranking in (*self._score_rankings.values(), self._final_ranking):
            for entry in ranking:
                if entry.epoch == epoch:
                    entry.candidate_sources.update(source_names)

    def policy_summary(self) -> dict[str, object]:
        dormant = (
            sorted(self.active_score_names.intersection(self.clean_score_names))
            if self.mode == "disabled"
            else []
        )
        return {
            "mode": self.mode,
            "final_checkpoint_score": self.final_score_name,
            "registered_scores": sorted(self.scorers),
            "active_scores": sorted(self.active_score_names),
            "checkpoint_scores": [
                scorer_to_dict(self.scorers[name])
                for name in sorted(self.scorers)
            ],
            "clean_train_scores": sorted(self.clean_score_names),
            "candidate_sources": [asdict(source) for source in self.candidate_sources],
            "checkpoint_reporting": {
                "selection_runner_ups": self.reporting.selection_runner_ups,
                "score_rankings": [
                    asdict(ranking) for ranking in self.reporting.score_rankings
                ],
                "show_epoch_metrics": list(self.reporting.show_epoch_metrics),
            },
            "inactive_scores": {
                name: "clean_train_mode=disabled; clean evaluation is unavailable"
                for name in dormant
            },
            "clean_train_interval": self.clean_train_interval,
        }


def scorer_to_dict(scorer: CheckpointScorer) -> dict[str, object]:
    return {"name": scorer.name}


def save_checkpoint_bundle(
    *,
    artifact_root: str | Path,
    fit_id: str,
    state: TrainingState,
    metadata: Mapping[str, object],
) -> Path:
    """Publish a validated tensor checkpoint and JSON completion marker."""

    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / fit_id
    if destination.exists():
        raise FileExistsError(f"Checkpoint bundle already exists: {destination}")
    temporary = root / f".{fit_id}.tmp-{uuid4().hex}"
    temporary.mkdir()
    checkpoint_path = temporary / "selected.pt"
    sidecar_path = temporary / "selected.json"
    try:
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "fit_id": fit_id,
            "training_state": asdict(state),
        }
        torch.save(payload, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            loaded.get("format_version") != CHECKPOINT_FORMAT_VERSION
            or loaded.get("fit_id") != fit_id
        ):
            raise ValueError("Newly written checkpoint failed identity validation.")
        sidecar = {
            "schema_version": CHECKPOINT_METADATA_VERSION,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "fit_id": fit_id,
            "checkpoint_file": checkpoint_path.name,
            **dict(metadata),
        }
        sidecar_path.write_text(
            json.dumps(sidecar, default=_json_default, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint_bundle(
    bundle_path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load a complete selected-checkpoint bundle and reject partial pairs."""

    bundle = Path(bundle_path).expanduser().resolve()
    sidecar_path = bundle / "selected.json"
    if not sidecar_path.is_file():
        raise ValueError(f"Checkpoint completion marker is missing: {sidecar_path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checkpoint_path = bundle / str(metadata.get("checkpoint_file", ""))
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint tensor file is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != metadata.get("format_version"):
        raise ValueError("Checkpoint and metadata format versions do not match.")
    if payload.get("fit_id") != metadata.get("fit_id"):
        raise ValueError("Checkpoint and metadata fit IDs do not match.")
    return payload, metadata
