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


VALID_METRICS = frozenset(
    {
        "val_data_loss",
        "val_selector_regularization",
        "val_accuracy",
        "val_roc_auc",
        "clean_train_data_loss",
        "clean_train_selector_regularization",
        "clean_train_accuracy",
        "clean_train_roc_auc",
    }
)
CLEAN_TRAIN_METRICS = frozenset(
    metric for metric in VALID_METRICS if metric.startswith("clean_train_")
)
CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_METADATA_VERSION = 1


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


@dataclass(frozen=True)
class MetricTerm:
    """One explicitly transformed term in a checkpoint score."""

    metric: str
    weight: float = 1.0
    scale: float = 1.0
    offset: float = 0.0
    smoothing_window: int = 1

    def __post_init__(self) -> None:
        if self.metric not in VALID_METRICS:
            raise ValueError(
                f"Unknown checkpoint metric '{self.metric}'. "
                f"Expected one of {sorted(VALID_METRICS)}."
            )
        for name, value in (
            ("weight", self.weight),
            ("scale", self.scale),
            ("offset", self.offset),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if (
            isinstance(self.smoothing_window, bool)
            or not isinstance(self.smoothing_window, int)
            or self.smoothing_window < 1
        ):
            raise ValueError("smoothing_window must be a positive integer.")

    def transformed_value(
        self,
        metrics: Mapping[str, float | None],
        history: Sequence[Mapping[str, float | None]],
    ) -> float:
        if float(self.weight) == 0.0:
            return 0.0
        current = metrics.get(self.metric)
        if current is None or not math.isfinite(float(current)):
            raise ValueError(
                f"Required checkpoint metric '{self.metric}' is unavailable "
                "for the current weights."
            )
        values: list[float] = [float(current)]
        for record in reversed(history):
            if len(values) == self.smoothing_window:
                break
            value = record.get(self.metric)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        if len(values) != self.smoothing_window:
            raise ValueError(
                f"Metric '{self.metric}' needs {self.smoothing_window} fresh "
                "value(s) for causal scoring."
            )
        value = sum(values) / len(values)
        return float(self.weight) * (float(self.scale) * value + float(self.offset))


@dataclass(frozen=True)
class CheckpointScorer:
    """A maximized or minimized explicit weighted metric formula."""

    terms: tuple[MetricTerm, ...]
    direction: str
    name: str = "weighted"
    min_delta: float = 1e-12

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("A checkpoint scorer needs at least one metric term.")
        if self.direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'.")
        if not math.isfinite(float(self.min_delta)) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and >= 0.")

    @property
    def required_metrics(self) -> frozenset[str]:
        return frozenset(term.metric for term in self.terms if term.weight != 0.0)

    def score(
        self,
        metrics: Mapping[str, float | None],
        history: Sequence[Mapping[str, float | None]] = (),
    ) -> float:
        return sum(term.transformed_value(metrics, history) for term in self.terms)

    def is_better(self, score: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        if self.direction == "min":
            return score < incumbent - self.min_delta
        return score > incumbent + self.min_delta


@dataclass(frozen=True)
class CandidateGenerator:
    """A bounded ranking used to nominate final checkpoint candidates."""

    scorer: CheckpointScorer
    top_k: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer.")

    @property
    def label(self) -> str:
        return self.name or self.scorer.name


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
    metrics: dict[str, float | None]
    state: TrainingState
    generator_scores: dict[str, float] = field(default_factory=dict)
    final_score: float | None = None


@dataclass(frozen=True)
class SelectionObservation:
    opportunity: bool
    progress: bool


def _single_metric_scorer(
    metric: str,
    direction: str,
    *,
    smoothing_window: int = 1,
    name: str | None = None,
) -> CheckpointScorer:
    return CheckpointScorer(
        terms=(MetricTerm(metric, smoothing_window=smoothing_window),),
        direction=direction,
        name=name or metric,
    )


_SCORER_ALIASES = {
    "val_loss": _single_metric_scorer("val_data_loss", "min", name="val_loss"),
    "val_data_loss": _single_metric_scorer("val_data_loss", "min"),
    "val_roc_auc": _single_metric_scorer("val_roc_auc", "max"),
    "clean_train_loss": _single_metric_scorer(
        "clean_train_data_loss", "min", name="clean_train_loss"
    ),
    "clean_train_data_loss": _single_metric_scorer("clean_train_data_loss", "min"),
    "clean_train_roc_auc": _single_metric_scorer("clean_train_roc_auc", "max"),
}


def resolve_scorer(
    config: str | CheckpointScorer | Mapping[str, object],
) -> CheckpointScorer:
    """Validate a scorer config and return its canonical immutable form."""

    if isinstance(config, CheckpointScorer):
        return config
    if isinstance(config, str):
        try:
            return _SCORER_ALIASES[config]
        except KeyError as exc:
            raise ValueError(
                f"Unknown checkpoint scorer '{config}'. "
                f"Expected one of {sorted(_SCORER_ALIASES)} or a scorer mapping."
            ) from exc
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint scorer must be a string, mapping, or CheckpointScorer.")

    if "terms" in config:
        raw_terms = config["terms"]
        if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, (str, bytes)):
            raise TypeError("scorer 'terms' must be a sequence.")
        terms = tuple(
            term if isinstance(term, MetricTerm) else MetricTerm(**dict(term))
            for term in raw_terms
        )
    else:
        metric = str(config.get("metric", ""))
        terms = (
            MetricTerm(
                metric=metric,
                weight=float(config.get("weight", 1.0)),
                scale=float(config.get("scale", 1.0)),
                offset=float(config.get("offset", 0.0)),
                smoothing_window=int(config.get("smoothing_window", 1)),
            ),
        )
    return CheckpointScorer(
        terms=terms,
        direction=str(config.get("direction", "min")),
        name=str(config.get("name", "weighted")),
        min_delta=float(config.get("min_delta", 1e-12)),
    )


def resolve_generators(
    configs: Sequence[CandidateGenerator | Mapping[str, object]] | None,
) -> tuple[CandidateGenerator, ...]:
    if configs is None:
        return (CandidateGenerator(_SCORER_ALIASES["val_loss"]),)
    generators = []
    for config in configs:
        if isinstance(config, CandidateGenerator):
            generators.append(config)
            continue
        if not isinstance(config, Mapping):
            raise TypeError("candidate generators must be mappings or CandidateGenerator.")
        scorer_config = config.get("scorer", config.get("metric", "val_loss"))
        if "metric" in config and "scorer" not in config:
            scorer_config = {
                key: value
                for key, value in config.items()
                if key not in {"top_k", "name"}
            }
        generators.append(
            CandidateGenerator(
                scorer=resolve_scorer(scorer_config),
                top_k=int(config.get("top_k", 1)),
                name=None if config.get("name") is None else str(config["name"]),
            )
        )
    if not generators:
        raise ValueError("candidate_generators cannot be empty in candidate modes.")
    labels = [generator.label for generator in generators]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate generator names must be unique.")
    return tuple(generators)


def validate_selection_config(
    *,
    mode: str,
    scorer: CheckpointScorer,
    generators: Sequence[CandidateGenerator],
    interval: int,
) -> None:
    if mode not in {"disabled", "interval", "deferred_candidates"}:
        raise ValueError(
            "clean_train_mode must be 'disabled', 'interval', or "
            "'deferred_candidates'."
        )
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("clean_train_interval must be a positive integer.")
    if mode == "disabled" and scorer.required_metrics & CLEAN_TRAIN_METRICS:
        raise ValueError("disabled mode cannot score clean-training metrics.")
    if mode == "deferred_candidates":
        for generator in generators:
            if generator.scorer.required_metrics & CLEAN_TRAIN_METRICS:
                raise ValueError(
                    "deferred candidate generators must be validation-only; "
                    f"'{generator.label}' requires clean-training metrics."
                )


class CheckpointController:
    """Mode-aware online selection with a bounded, deduplicated candidate union."""

    def __init__(
        self,
        *,
        mode: str = "disabled",
        checkpoint_scorer: str | CheckpointScorer | Mapping[str, object] = "val_loss",
        candidate_generators: Sequence[
            CandidateGenerator | Mapping[str, object]
        ]
        | None = None,
        clean_train_interval: int = 1,
    ) -> None:
        self.mode = mode
        self.scorer = resolve_scorer(checkpoint_scorer)
        self.generators = (
            () if mode == "disabled" else resolve_generators(candidate_generators)
        )
        self.clean_train_interval = clean_train_interval
        validate_selection_config(
            mode=mode,
            scorer=self.scorer,
            generators=self.generators,
            interval=clean_train_interval,
        )
        self.history: list[dict[str, float | None]] = []
        self._rankings: dict[str, list[tuple[float, CheckpointCandidate]]] = {
            generator.label: [] for generator in self.generators
        }
        self._best: CheckpointCandidate | None = None
        self._best_score: float | None = None

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
        metrics: Mapping[str, float | None],
        state_factory: Callable[[], TrainingState],
    ) -> SelectionObservation:
        record = dict(metrics)
        prior_history = tuple(self.history)
        self.history.append(record)
        if self.mode == "interval" and not self.is_eligible_epoch(epoch):
            return SelectionObservation(opportunity=False, progress=False)

        if self.mode == "disabled":
            try:
                score = self.scorer.score(record, prior_history)
            except ValueError as exc:
                if "fresh value(s) for causal scoring" not in str(exc):
                    raise
                return SelectionObservation(opportunity=False, progress=False)
            progress = self.scorer.is_better(score, self._best_score)
            if progress:
                self._best_score = score
                self._best = CheckpointCandidate(
                    epoch=epoch,
                    metrics=record,
                    state=state_factory(),
                    final_score=score,
                )
            return SelectionObservation(opportunity=True, progress=progress)

        state_cache: list[TrainingState] = []

        def shared_state() -> TrainingState:
            if not state_cache:
                state_cache.append(state_factory())
            return state_cache[0]

        progress = False
        ready_generators = 0
        for generator in self.generators:
            try:
                score = generator.scorer.score(record, prior_history)
            except ValueError as exc:
                if "fresh value(s) for causal scoring" not in str(exc):
                    raise
                continue
            ready_generators += 1
            ranking = self._rankings[generator.label]
            if len(ranking) < generator.top_k:
                accepted = True
            else:
                worst_score = ranking[-1][0]
                accepted = generator.scorer.is_better(score, worst_score)
            if not accepted:
                continue
            candidate = CheckpointCandidate(
                epoch=epoch,
                metrics=record,
                state=shared_state(),
                generator_scores={generator.label: score},
            )
            ranking.append((score, candidate))
            ranking.sort(
                key=lambda item: item[0],
                reverse=generator.scorer.direction == "max",
            )
            del ranking[generator.top_k :]
            progress = True
        return SelectionObservation(
            opportunity=ready_generators > 0,
            progress=progress,
        )

    def candidates(self) -> list[CheckpointCandidate]:
        by_epoch: dict[int, CheckpointCandidate] = {}
        for ranking in self._rankings.values():
            for _, candidate in ranking:
                existing = by_epoch.get(candidate.epoch)
                if existing is None:
                    by_epoch[candidate.epoch] = candidate
                else:
                    existing.generator_scores.update(candidate.generator_scores)
        return [by_epoch[epoch] for epoch in sorted(by_epoch)]

    def finalize(
        self,
        *,
        enrich_candidate: Callable[[CheckpointCandidate], Mapping[str, float | None]]
        | None = None,
    ) -> CheckpointCandidate:
        if self.mode == "disabled":
            if self._best is None:
                raise RuntimeError("No checkpoint was eligible for selection.")
            return self._best

        candidates = self.candidates()
        if not candidates:
            raise RuntimeError("No checkpoint candidates were retained.")
        if self.mode == "deferred_candidates":
            if enrich_candidate is None:
                raise ValueError("deferred_candidates requires a clean metric evaluator.")
            for candidate in candidates:
                candidate.metrics.update(dict(enrich_candidate(candidate)))
                for history_record in self.history:
                    if int(history_record.get("epoch", 0) or 0) == candidate.epoch:
                        history_record.update(candidate.metrics)
                        break

        best: CheckpointCandidate | None = None
        best_score: float | None = None
        for candidate in candidates:
            history = tuple(
                metrics
                for metrics in self.history
                if int(metrics.get("epoch", 0) or 0) < candidate.epoch
            )
            score = self.scorer.score(candidate.metrics, history)
            candidate.final_score = score
            if self.scorer.is_better(score, best_score):
                best = candidate
                best_score = score
        if best is None:
            raise RuntimeError("Final checkpoint scoring produced no selection.")
        return best

    def policy_summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "checkpoint_scorer": scorer_to_dict(self.scorer),
            "candidate_generators": [
                {
                    "name": generator.label,
                    "top_k": generator.top_k,
                    "scorer": scorer_to_dict(generator.scorer),
                }
                for generator in self.generators
            ],
            "clean_train_interval": self.clean_train_interval,
        }


def scorer_to_dict(scorer: CheckpointScorer) -> dict[str, object]:
    return {
        "name": scorer.name,
        "direction": scorer.direction,
        "min_delta": scorer.min_delta,
        "terms": [asdict(term) for term in scorer.terms],
    }


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
